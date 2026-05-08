#!/usr/bin/env python3
"""Scaffold the minimum registry directories for a route and role."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry_root", help="Path to the registry root on the server")
    parser.add_argument("route_key", help="Logical route key, e.g. acme/api/products")
    parser.add_argument("role", choices=["router", "responder"], help="Model role")
    parser.add_argument("vendor", help="Base model vendor directory, e.g. Qwen")
    parser.add_argument("base_model_dir", help="Base model directory name, e.g. Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("version", help="Immutable version name, e.g. v1")
    args = parser.parse_args()

    root = Path(args.registry_root).expanduser()
    route_dir = root / "routes" / Path(args.route_key)
    adapter_dir = root / "adapters" / args.vendor / args.base_model_dir / Path(args.route_key) / args.role / args.version

    for path in (root, root / "routes", root / "adapters", route_dir, adapter_dir):
        path.mkdir(parents=True, exist_ok=True)

    print(f"Created route directory: {route_dir}")
    print(f"Created adapter version directory: {adapter_dir}")
    print("Populate route.json and manifest.json manually or with an agent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
