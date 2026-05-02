#!/usr/bin/env python3
"""Register a candidate version on a route for testing or gradual rollout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from registry_lib import load_json, route_json_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry_root", help="Path to the registry root")
    parser.add_argument("route_key", help="Logical route key")
    parser.add_argument("version", help="Version to register, e.g. v2")
    parser.add_argument("--role", choices=["router", "responder"], default="responder")
    parser.add_argument("--label", default=None, help="Candidate label (auto-assigned if omitted)")
    parser.add_argument("--weight", type=int, default=0, help="Traffic weight 0-100 (default 0)")
    args = parser.parse_args()

    path = route_json_path(Path(args.registry_root).expanduser(), args.route_key)
    route_data = load_json(path)
    role_cfg = route_data["roles"].get(args.role)
    if role_cfg is None:
        print(f"Role '{args.role}' not defined on route {args.route_key}")
        return 1

    candidates = role_cfg.get("candidates", [])
    existing_versions = {c["version"] for c in candidates}
    if args.version in existing_versions:
        print(f"Version {args.version} is already a candidate on {args.route_key}/{args.role}")
        return 1

    if args.version == role_cfg["production"]["version"]:
        print(f"Version {args.version} is already production on {args.route_key}/{args.role}")
        return 1

    if args.label:
        label = args.label
    else:
        existing_nums = []
        for c in candidates:
            if c["label"].startswith("candidate"):
                try:
                    existing_nums.append(int(c["label"][len("candidate"):]))
                except ValueError:
                    pass
        next_num = max(existing_nums, default=0) + 1
        label = f"candidate{next_num}"

    candidates.append({"label": label, "version": args.version, "weight": args.weight})
    role_cfg["candidates"] = candidates

    path.write_text(json.dumps(route_data, indent=2) + "\n")
    print(f"Registered candidate {label} ({args.version}, weight={args.weight}) on {args.route_key}/{args.role}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
