#!/usr/bin/env python3
"""Generate a concrete VideoAmp registry from current trainLLM artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def route_dir(root: Path, route_key: str) -> Path:
    return root / "routes" / Path(route_key)


def adapter_version_dir(root: Path, vendor: str, base_model_dir: str, route_key: str, role: str) -> Path:
    return root / "adapters" / vendor / base_model_dir / Path(route_key) / role / "v1"


def copy_adapter_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        target = dst / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        else:
            shutil.copy2(child, target)


def make_role_config(base_model_id: str, base_family: str, threshold: float) -> dict:
    return {
        "base_model": base_model_id,
        "base_family": base_family,
        "serving_pool": "qwen25_coder_1p5b",
        "clarification_threshold": threshold,
        "production": {"version": "v1"},
        "candidates": []
    }


def make_manifest(
    route_key: str,
    role: str,
    version_dir: Path,
    base_model_id: str,
    base_family: str,
    vendor: str,
    eval_data: dict | None,
) -> dict:
    evaluation = {}
    if eval_data is not None:
        evaluation = {
            "type": "holdout",
            "primary_metric": "average_similarity",
            "primary_score": eval_data.get("summary", {}).get("avg_score"),
            "report_files": ["eval.json", "eval.md"]
        }
    return {
        "schema_version": 1,
        "route_key": route_key,
        "role": role,
        "version": "v1",
        "status": "production",
        "base_model": {
            "id": base_model_id,
            "family": base_family,
            "vendor": vendor
        },
        "adapter": {
            "format": "peft-lora",
            "path": str(version_dir),
            "source": "local"
        },
        "evaluation": evaluation,
        "created_at": "2026-05-01T00:00:00Z",
        "notes": "Promoted from trainLLM artifacts into the model registry."
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("deployment_spec", help="Path to deployments/videoamp-production.json")
    args = parser.parse_args()

    spec_path = Path(args.deployment_spec).expanduser().resolve()
    spec = load_json(spec_path)

    registry_root = Path(spec["registry_root"]).expanduser()
    trainllm_root = Path(spec["trainllm_root"]).expanduser()
    vendor = spec["vendor"]
    base_model_dir = spec["base_model_dir"]
    base_model_id = spec["base_model_id"]
    base_family = spec["base_family"]

    registry_root.mkdir(parents=True, exist_ok=True)
    (registry_root / "routes").mkdir(exist_ok=True)
    (registry_root / "adapters").mkdir(exist_ok=True)

    registry_yaml = (
        f"registry_root: {registry_root}\n"
        "default_adapter_source: local\n"
        "serving_pools:\n"
        "  qwen25_coder_1p5b:\n"
        f"    base_model: {base_model_id}\n"
        "    notes: Primary pool for current VideoAmp router and endpoint adapters.\n"
    )
    (registry_root / "registry.yaml").write_text(registry_yaml)

    for route_key, route_spec in spec["routes"].items():
        roles: dict = {}
        if route_key == "videoamp/api" and route_spec.get("router_adapter"):
            roles["router"] = make_role_config(base_model_id, base_family, 0.75)
        if route_spec.get("responder_adapter"):
            roles["responder"] = make_role_config(base_model_id, base_family, 0.8)

        write_json(
            route_dir(registry_root, route_key) / "route.json",
            {
                "route_key": route_key,
                "kind": route_spec["kind"],
                "roles": roles,
                "children": route_spec.get("children", []),
                "status": "active"
            },
        )

        if route_key == "videoamp/api" and route_spec.get("router_adapter"):
            src = trainllm_root / "lora" / route_spec["router_adapter"] / "final"
            dst = adapter_version_dir(registry_root, vendor, base_model_dir, route_key, "router")
            copy_adapter_tree(src, dst)
            write_json(
                dst / "manifest.json",
                make_manifest(route_key, "router", dst, base_model_id, base_family, vendor, None),
            )

        if route_spec.get("responder_adapter"):
            src = trainllm_root / "lora" / route_spec["responder_adapter"] / "final"
            dst = adapter_version_dir(registry_root, vendor, base_model_dir, route_key, "responder")
            copy_adapter_tree(src, dst)

            eval_json_path = trainllm_root / "evals" / route_spec["eval_json"]
            eval_md_path = eval_json_path.with_suffix(".md")
            eval_data = load_json(eval_json_path)
            shutil.copy2(eval_json_path, dst / "eval.json")
            shutil.copy2(eval_md_path, dst / "eval.md")
            write_json(
                dst / "manifest.json",
                make_manifest(route_key, "responder", dst, base_model_id, base_family, vendor, eval_data),
            )

    print(f"Generated registry at {registry_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
