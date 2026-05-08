#!/usr/bin/env python3
"""Import a trained adapter into the registry from a trainLLM result.json.

Reads result.json (the artifact contract from trainLLM), copies adapter weights
into the registry, writes a manifest, and optionally promotes to production.

Usage:
    python3 tools/import_adapter.py <registry_root> <route_key> <version> --result /path/to/result.json
    python3 tools/import_adapter.py <registry_root> <route_key> v2 --result ~/trainLLM/lora/my-adapter/final/result.json --promote
    python3 tools/import_adapter.py <registry_root> videoamp/api-merged v1 --result ~/trainLLM/merged/default/result.json --promote
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from registry_lib import derive_vendor_base_dir, load_json, route_json_path


def copy_adapter_files(source_dir: Path, dest: Path, fmt: str) -> int:
    """Copy model files from source to registry destination. Returns file count."""
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    if fmt == "peft-lora":
        extensions = {".safetensors", ".json", ".jinja"}
    else:
        extensions = {".safetensors", ".json", ".model", ".txt"}

    for f in source_dir.iterdir():
        if f.is_file() and f.suffix in extensions and f.name != "result.json":
            shutil.copy2(f, dest / f.name)
            count += 1
    return count


def write_manifest(dest: Path, result: dict, route_key: str, role: str, version: str) -> Path:
    """Write manifest.json into the registry version directory."""
    base_model_id = result["base_model"]
    vendor = base_model_id.split("/", 1)[0]

    adapter_block: dict = {
        "format": result["format"],
        "path": str(dest),
        "source": "local",
    }
    if result["format"] == "merged-full" and "merge_config" in result:
        adapter_block["merge_config"] = result["merge_config"]

    manifest = {
        "schema_version": 1,
        "route_key": route_key,
        "role": role,
        "version": version,
        "status": "production",
        "base_model": {
            "id": base_model_id,
            "family": base_model_id,
            "vendor": vendor,
        },
        "adapter": adapter_block,
        "created_at": datetime.now().isoformat() + "Z",
    }

    if result.get("eval_avg_score") is not None:
        manifest["evaluation"] = {"avg_score": result["eval_avg_score"]}

    if result.get("lora_rank") is not None:
        manifest["training"] = {
            "lora_rank": result["lora_rank"],
            "lora_alpha": result.get("lora_alpha"),
            "max_seq_length": result.get("max_seq_length"),
            "training_steps": result.get("training_steps"),
            "final_loss": result.get("final_loss"),
        }

    manifest_path = dest / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def scaffold_route(registry_root: Path, route_key: str, role: str, version: str,
                   base_model: str, fmt: str) -> Path:
    """Create route.json if it doesn't exist; return the path."""
    rpath = route_json_path(registry_root, route_key)
    if rpath.exists():
        return rpath

    rpath.parent.mkdir(parents=True, exist_ok=True)

    serving_pool_base = base_model.replace("/", "_").replace("-", "_").lower()
    if fmt == "merged-full":
        serving_pool = f"{serving_pool_base}_merged"
        serving_mode = "merged"
    else:
        serving_pool = serving_pool_base
        serving_mode = "lora"

    route_data = {
        "route_key": route_key,
        "kind": "leaf",
        "roles": {
            role: {
                "base_model": base_model,
                "base_family": base_model,
                "serving_pool": serving_pool,
                "serving_mode": serving_mode,
                "clarification_threshold": 0.8,
                "production": {"version": version},
                "candidates": [],
            }
        },
        "children": [],
        "status": "active",
    }
    rpath.write_text(json.dumps(route_data, indent=2) + "\n")
    print(f"  Scaffolded route: {route_key}")
    return rpath


def promote_version(registry_root: Path, route_key: str, role: str, version: str) -> None:
    """Set a version as production in route.json."""
    rpath = route_json_path(registry_root, route_key)
    route_data = load_json(rpath)

    role_cfg = route_data["roles"].get(role)
    if role_cfg is None:
        route_data["roles"][role] = {
            "base_model": "",
            "production": {"version": version},
            "candidates": [],
        }
    else:
        old = role_cfg["production"]["version"]
        if old == version:
            print(f"  Version {version} is already production")
            return
        role_cfg["production"]["version"] = version
        role_cfg["candidates"] = [
            c for c in role_cfg.get("candidates", []) if c["version"] != version
        ]

    rpath.write_text(json.dumps(route_data, indent=2) + "\n")
    print(f"  Promoted {route_key}/{role} to {version}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("registry_root", help="Path to the registry root")
    parser.add_argument("route_key", help="Logical route key (e.g. acme/api/products)")
    parser.add_argument("version", help="Version label (e.g. v2)")
    parser.add_argument("--result", required=True, help="Path to trainLLM result.json")
    parser.add_argument("--role", choices=["router", "responder"], default="responder")
    parser.add_argument("--promote", action="store_true",
                        help="Set this version as production (default: register as candidate)")
    args = parser.parse_args()

    result_path = Path(args.result).expanduser()
    if not result_path.exists():
        print(f"result.json not found: {result_path}", file=sys.stderr)
        return 1

    result = json.loads(result_path.read_text())
    registry_root = Path(args.registry_root).expanduser()
    adapter_source = Path(result["adapter_path"])

    if not adapter_source.exists():
        print(f"Adapter path does not exist: {adapter_source}", file=sys.stderr)
        return 1

    fmt = result["format"]
    base_model = result["base_model"]
    vendor, base_model_dir = derive_vendor_base_dir(base_model)

    dest = (registry_root / "adapters" / vendor / base_model_dir
            / Path(args.route_key) / args.role / args.version)

    print(f"Importing {fmt} adapter into registry:")
    print(f"  Source:  {adapter_source}")
    print(f"  Dest:    {dest}")
    print(f"  Route:   {args.route_key}/{args.role} @ {args.version}")

    n_files = copy_adapter_files(adapter_source, dest, fmt)
    print(f"  Copied {n_files} files")

    manifest = write_manifest(dest, result, args.route_key, args.role, args.version)
    print(f"  Manifest: {manifest}")

    scaffold_route(registry_root, args.route_key, args.role, args.version,
                   base_model, fmt)

    if args.promote:
        promote_version(registry_root, args.route_key, args.role, args.version)
    else:
        print(f"  Registered as candidate. Use promote.py to make it production.")

    if result.get("eval_avg_score") is not None:
        print(f"  Eval score: {result['eval_avg_score']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
