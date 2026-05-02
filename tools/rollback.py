#!/usr/bin/env python3
"""Roll back production to a previous version by swapping with a candidate or explicit version."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from registry_lib import load_json, route_json_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry_root", help="Path to the registry root")
    parser.add_argument("route_key", help="Logical route key")
    parser.add_argument("version", help="Version to roll back to, e.g. v1")
    parser.add_argument("--role", choices=["router", "responder"], default="responder")
    args = parser.parse_args()

    path = route_json_path(Path(args.registry_root).expanduser(), args.route_key)
    route_data = load_json(path)
    role_cfg = route_data["roles"].get(args.role)
    if role_cfg is None:
        print(f"Role '{args.role}' not defined on route {args.route_key}")
        return 1

    current = role_cfg["production"]["version"]
    if current == args.version:
        print(f"Version {args.version} is already production for {args.route_key}/{args.role}")
        return 0

    role_cfg["production"]["version"] = args.version
    role_cfg["candidates"] = [c for c in role_cfg.get("candidates", []) if c["version"] != args.version]

    demoted_exists = any(c["version"] == current for c in role_cfg["candidates"])
    if not demoted_exists:
        role_cfg["candidates"].append({"label": "candidate1", "version": current, "weight": 0})

    path.write_text(json.dumps(route_data, indent=2) + "\n")
    print(f"Rolled back {args.route_key}/{args.role}: {current} -> {args.version}")
    print(f"Previous production {current} demoted to candidate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
