#!/usr/bin/env python3
"""Resolve a logical route to a concrete model artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from registry_lib import resolve_target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry_root", help="Path to the registry root")
    parser.add_argument("route_key", help="Logical route key, e.g. acme/api/products")
    parser.add_argument("--role", choices=["router", "responder"], help="Explicit role to resolve")
    parser.add_argument(
        "--selector",
        default="production",
        help="production, version:vN, or candidate:candidateN",
    )
    args = parser.parse_args()

    resolved = resolve_target(Path(args.registry_root).expanduser(), args.route_key, args.role, args.selector)
    print(json.dumps(resolved, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
