#!/usr/bin/env python3
"""Launch LoRAX for the base model implied by a registry route."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

from registry_lib import resolve_target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry_root", help="Path to the registry root")
    parser.add_argument("route_key", help="Route key whose role defines the serving pool/base model")
    parser.add_argument("--role", choices=["router", "responder"], help="Explicit role to resolve")
    parser.add_argument("--selector", default="production", help="production, version:vN, or candidate:candidateN")
    parser.add_argument("--lorax-launcher", default="lorax-launcher", help="Path to lorax-launcher")
    parser.add_argument("--hostname", default="0.0.0.0")
    parser.add_argument("--port", default="8080")
    parser.add_argument("--dtype", default=None, help="Optional dtype, e.g. bfloat16")
    parser.add_argument("--print-only", action="store_true", help="Print command instead of executing it")
    args, extra_args = parser.parse_known_args()

    resolved = resolve_target(Path(args.registry_root).expanduser(), args.route_key, args.role, args.selector)

    cmd = [
        args.lorax_launcher,
        "--model-id",
        resolved["base_model"],
        "--adapter-source",
        "local",
        "--hostname",
        args.hostname,
        "--port",
        str(args.port),
    ]
    if args.dtype:
        cmd.extend(["--dtype", args.dtype])
    if extra_args:
        extra = extra_args
        if extra and extra[0] == "--":
            extra = extra[1:]
        cmd.extend(extra)

    print("Resolved route:")
    print(f"  route_key: {resolved['route_key']}")
    print(f"  role: {resolved['role']}")
    print(f"  base_model: {resolved['base_model']}")
    print(f"  serving_pool: {resolved['serving_pool']}")
    print("LoRAX command:")
    print("  " + " ".join(shlex.quote(part) for part in cmd))

    if args.print_only:
        return 0

    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
