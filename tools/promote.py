#!/usr/bin/env python3
"""Promote a candidate or specific version to production for a route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from registry_lib import load_json, route_json_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry_root", help="Path to the registry root")
    parser.add_argument("route_key", help="Logical route key")
    parser.add_argument("version", help="Version to promote, e.g. v2")
    parser.add_argument("--role", choices=["router", "responder"], default="responder")
    args = parser.parse_args()

    path = route_json_path(Path(args.registry_root).expanduser(), args.route_key)
    route_data = load_json(path)
    role_cfg = route_data["roles"].get(args.role)
    if role_cfg is None:
        print(f"Role '{args.role}' not defined on route {args.route_key}")
        return 1

    old_version = role_cfg["production"]["version"]
    if old_version == args.version:
        print(f"Version {args.version} is already production for {args.route_key}/{args.role}")
        return 0

    role_cfg["production"]["version"] = args.version
    role_cfg["candidates"] = [c for c in role_cfg.get("candidates", []) if c["version"] != args.version]

    path.write_text(json.dumps(route_data, indent=2) + "\n")
    print(f"Promoted {args.route_key}/{args.role}: {old_version} -> {args.version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
