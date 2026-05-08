#!/usr/bin/env python3
"""Send a chat request to vLLM using a route key resolved from the registry."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from registry_lib import resolve_target


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
    parser.add_argument("route_key", help="Logical route key, e.g. acme/api/products")
    parser.add_argument("prompt", help="User prompt")
    parser.add_argument("--role", choices=["router", "responder"], default="responder")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=None)
    args = parser.parse_args()

    resolve_target(Path(args.registry_root).expanduser(), args.route_key, requested_role=args.role, selector="production")

    payload = {
        "model": args.route_key,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
    }
    if args.temperature is not None:
        payload["temperature"] = args.temperature

    print(json.dumps(post_chat(args.base_url, payload), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
