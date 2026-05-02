#!/usr/bin/env python3
"""Validate basic registry invariants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate_route_file(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        data = json.loads(path.read_text())
    except Exception as exc:  # pragma: no cover - defensive
        return [f"{path}: failed to parse JSON: {exc}"]

    route_key = data.get("route_key")
    if not route_key:
        errors.append(f"{path}: missing route_key")
    if "roles" not in data:
        errors.append(f"{path}: missing roles")
    if "children" not in data:
        errors.append(f"{path}: missing children")
    return errors


def validate_manifest_file(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        data = json.loads(path.read_text())
    except Exception as exc:  # pragma: no cover - defensive
        return [f"{path}: failed to parse JSON: {exc}"]

    for field in ("route_key", "role", "version", "base_model", "adapter"):
        if field not in data:
            errors.append(f"{path}: missing {field}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry_root", help="Path to the registry root on the server")
    args = parser.parse_args()

    root = Path(args.registry_root).expanduser()
    errors: list[str] = []

    if not root.exists():
        print(f"Registry root does not exist: {root}")
        return 1

    for route_path in root.glob("routes/**/route.json"):
        errors.extend(validate_route_file(route_path))

    for manifest_path in root.glob("adapters/**/manifest.json"):
        errors.extend(validate_manifest_file(manifest_path))

    if errors:
        print("Validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Validation passed for {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
