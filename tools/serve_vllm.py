#!/usr/bin/env python3
"""Launch vLLM for the base model implied by a registry route."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path

from registry_lib import list_production_targets, resolve_target


def read_lora_rank(adapter_path: str) -> int:
    config_path = Path(adapter_path) / "adapter_config.json"
    data = json.loads(config_path.read_text())
    return int(data["r"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry_root", help="Path to the registry root")
    parser.add_argument("route_key", help="Route key whose role defines the serving pool/base model")
    parser.add_argument("--role", choices=["router", "responder"], default="responder")
    parser.add_argument("--vllm", default="vllm", help="Path to the vLLM CLI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default="8000")
    parser.add_argument("--dtype", default=None, help="Optional dtype, e.g. bfloat16")
    parser.add_argument("--served-model-name", default=None, help="Optional explicit base served model name")
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-loras", type=int, default=None)
    parser.add_argument("--max-lora-rank", type=int, default=None)
    parser.add_argument("--print-only", action="store_true", help="Print command instead of executing it")
    args, extra_args = parser.parse_known_args()

    registry_root = Path(args.registry_root).expanduser()
    anchor = resolve_target(registry_root, args.route_key, requested_role=args.role, selector="production")
    targets = list_production_targets(registry_root, role=args.role, base_model=anchor["base_model"])

    # Also load router adapter if available (for LoRA-based routing)
    router_targets = list_production_targets(registry_root, role="router", base_model=anchor["base_model"])
    for rt in router_targets:
        if rt["route_key"] not in {t["route_key"] for t in targets}:
            targets.append(rt)

    if not targets:
        raise SystemExit(f"No production targets found for role={args.role} base_model={anchor['base_model']}")

    max_lora_rank = args.max_lora_rank if args.max_lora_rank is not None else max(
        read_lora_rank(target["adapter_path"]) for target in targets
    )

    cmd = [
        args.vllm,
        "serve",
        anchor["base_model"],
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--enable-lora",
        "--max-lora-rank",
        str(max_lora_rank),
    ]
    if args.dtype:
        cmd.extend(["--dtype", args.dtype])
    if args.max_model_len is not None:
        cmd.extend(["--max-model-len", str(args.max_model_len)])
    served_model_name = args.served_model_name or anchor["base_model"]
    cmd.extend(["--served-model-name", served_model_name])
    max_loras = args.max_loras if args.max_loras is not None else len(targets)
    cmd.extend(["--max-loras", str(max_loras)])

    cmd.append("--lora-modules")
    cmd.extend(f"{target['route_key']}={target['adapter_path']}" for target in targets)

    if extra_args:
        extra = extra_args
        if extra and extra[0] == "--":
            extra = extra[1:]
        cmd.extend(extra)

    print("Resolved base model:")
    print(f"  base_model: {anchor['base_model']}")
    print(f"  serving_pool: {anchor['serving_pool']}")
    print("Registered LoRA modules:")
    for target in targets:
        print(f"  {target['route_key']} -> {target['adapter_path']}")
    print(f"  max_lora_rank: {max_lora_rank}")
    print("vLLM command:")
    print("  " + " ".join(shlex.quote(part) for part in cmd))

    if args.print_only:
        return 0

    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
