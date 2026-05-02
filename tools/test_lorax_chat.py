#!/usr/bin/env python3
"""Send a chat request to LoRAX using an adapter resolved from the registry."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from registry_lib import resolve_target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry_root", help="Path to the registry root")
    parser.add_argument("route_key", help="Logical route key, e.g. videoamp/api/programs")
    parser.add_argument("prompt", help="User prompt")
    parser.add_argument("--role", choices=["router", "responder"], help="Explicit role to resolve")
    parser.add_argument("--selector", default="production", help="production, version:vN, or candidate:candidateN")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--system", default=None, help="Optional system prompt")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    resolved = resolve_target(Path(args.registry_root).expanduser(), args.route_key, args.role, args.selector)

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})

    payload = {
        "model": resolved["adapter_path"],
        "adapter_source": "local",
        "messages": messages,
        "max_tokens": args.max_tokens,
    }

    req = urllib.request.Request(
        args.base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        body = json.loads(resp.read().decode("utf-8"))

    print(json.dumps(body, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
