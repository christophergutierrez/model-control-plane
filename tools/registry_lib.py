#!/usr/bin/env python3
"""Helpers for resolving routes and versions from a machine-local model registry."""

from __future__ import annotations

import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def route_json_path(registry_root: Path, route_key: str) -> Path:
    return registry_root / "routes" / Path(route_key) / "route.json"


def manifest_path(
    registry_root: Path,
    vendor: str,
    base_model_dir: str,
    route_key: str,
    role: str,
    version: str,
) -> Path:
    return (
        registry_root
        / "adapters"
        / vendor
        / base_model_dir
        / Path(route_key)
        / role
        / version
        / "manifest.json"
    )


def infer_role(route_data: dict, requested_role: str | None) -> str:
    roles = route_data.get("roles", {})
    if requested_role:
        if requested_role not in roles:
            raise ValueError(f"Route does not define role '{requested_role}'")
        return requested_role
    if len(roles) != 1:
        raise ValueError("Route has multiple roles; specify --role explicitly")
    return next(iter(roles.keys()))


def derive_vendor_base_dir(base_model_id: str) -> tuple[str, str]:
    vendor = base_model_id.split("/", 1)[0]
    base_model_dir = base_model_id.split("/", 1)[1] if "/" in base_model_id else base_model_id
    return vendor, base_model_dir


def resolve_target(
    registry_root: Path,
    route_key: str,
    requested_role: str | None = None,
    selector: str = "production",
) -> dict:
    route_data = load_json(route_json_path(registry_root, route_key))
    role = infer_role(route_data, requested_role)
    role_cfg = route_data["roles"][role]

    if selector == "production":
        version = role_cfg["production"]["version"]
        selection = {"type": "production"}
    elif selector.startswith("version:"):
        version = selector.split(":", 1)[1]
        selection = {"type": "version", "version": version}
    elif selector.startswith("candidate:"):
        label = selector.split(":", 1)[1]
        candidates = {c["label"]: c for c in role_cfg.get("candidates", [])}
        if label not in candidates:
            raise ValueError(f"Candidate '{label}' not found for route {route_key}")
        version = candidates[label]["version"]
        selection = {"type": "candidate", "label": label}
    else:
        raise ValueError(f"Unsupported selector '{selector}'")

    base_model = role_cfg["base_model"]
    vendor, base_model_dir = derive_vendor_base_dir(base_model)
    manifest = load_json(manifest_path(registry_root, vendor, base_model_dir, route_key, role, version))

    return {
        "route_key": route_key,
        "role": role,
        "kind": route_data["kind"],
        "selection": selection,
        "version": version,
        "base_model": base_model,
        "base_family": role_cfg["base_family"],
        "serving_pool": role_cfg["serving_pool"],
        "clarification_threshold": role_cfg["clarification_threshold"],
        "manifest_path": str(
            manifest_path(registry_root, vendor, base_model_dir, route_key, role, version)
        ),
        "adapter_path": manifest["adapter"]["path"],
        "adapter_source": manifest["adapter"]["source"],
        "manifest": manifest,
    }
