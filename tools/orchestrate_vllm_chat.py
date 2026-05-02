#!/usr/bin/env python3
"""Minimal orchestrator stub for registry-backed vLLM chat requests."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from registry_lib import resolve_target


def build_messages(system: str | None, prompt: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def post_chat(base_url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry_root", help="Path to the registry root")
    parser.add_argument("route_key", help="Logical route key, e.g. videoamp/api/programs")
    parser.add_argument("prompt", help="User prompt")
    parser.add_argument("--selector", default="production", help="production, version:vN, or candidate:candidateN")
    parser.add_argument("--role", choices=["router", "responder"], default="responder")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--system", default=None, help="Optional system prompt")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--print-payload", action="store_true", help="Print the resolved dispatch payload before sending")
    args = parser.parse_args()

    resolved = resolve_target(
        Path(args.registry_root).expanduser(),
        args.route_key,
        requested_role=args.role,
        selector=args.selector,
    )

    payload = {
        "model": resolved["route_key"],
        "messages": build_messages(args.system, args.prompt),
        "max_tokens": args.max_tokens,
    }
    if args.temperature is not None:
        payload["temperature"] = args.temperature

    dispatch = {
        "route_key": resolved["route_key"],
        "role": resolved["role"],
        "version": resolved["version"],
        "base_model": resolved["base_model"],
        "serving_pool": resolved["serving_pool"],
        "adapter_path": resolved["adapter_path"],
        "request": payload,
    }

    if args.print_payload:
        print(json.dumps(dispatch, indent=2))

    response = post_chat(args.base_url, payload)
    print(json.dumps(response, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
