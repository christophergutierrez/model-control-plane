#!/usr/bin/env python3
"""Orchestrator: route a user query through classification, registry resolution, and vLLM dispatch.

Flow:
  1. Load available leaf routes from the registry.
  2. Run the router to classify the query -> route_key + confidence.
  3. If confidence < threshold, print a clarification request and exit.
  4. Resolve the route through the registry to get the production target.
  5. Dispatch to vLLM using the route key as the OpenAI model name.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

from registry_lib import list_route_keys, load_json, resolve_target, route_json_path
from router import EmbeddingRouter, RouteCandidate, keyword_route


def get_leaf_routes(registry_root: Path) -> list[str]:
    leaves: list[str] = []
    for route_key in list_route_keys(registry_root):
        route_data = load_json(route_json_path(registry_root, route_key))
        if route_data.get("kind") == "leaf":
            leaves.append(route_key)
    return leaves


def get_threshold(registry_root: Path, route_key: str, role: str) -> float:
    route_data = load_json(route_json_path(registry_root, route_key))
    role_cfg = route_data.get("roles", {}).get(role, {})
    return role_cfg.get("clarification_threshold", 0.8)


def post_chat(base_url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def format_clarification(candidates: list[RouteCandidate], threshold: float) -> str:
    lines = [
        f"I'm not confident enough to route this request (threshold: {threshold}).",
        "Top candidates:",
    ]
    for c in candidates[:5]:
        lines.append(f"  {c.route_key}  (confidence: {c.confidence})")
    lines.append("")
    lines.append("Could you clarify which endpoint you're asking about?")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("registry_root", help="Path to the registry root")
    parser.add_argument("prompt", help="User query")
    parser.add_argument("--role", choices=["router", "responder"], default="responder")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--system", default=None, help="Optional system prompt")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=None, help="Override clarification threshold")
    parser.add_argument("--dry-run", action="store_true", help="Route and resolve but don't call vLLM")
    parser.add_argument("--router", choices=["embedding", "keyword"], default="embedding",
                        help="Router backend (default: embedding)")
    parser.add_argument("--descriptions", default=None,
                        help="Path to route_descriptions.json (default: auto-detect next to this script)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    registry_root = Path(args.registry_root).expanduser()

    leaves = get_leaf_routes(registry_root)
    if not leaves:
        print("No leaf routes found in registry.", file=sys.stderr)
        return 1

    if args.router == "embedding":
        desc_path = args.descriptions or Path(__file__).parent / "route_descriptions.json"
        if args.verbose:
            print("Loading embedding router...", file=sys.stderr)
        emb_router = EmbeddingRouter(desc_path)
        candidates = emb_router.route(args.prompt)
    else:
        candidates = keyword_route(args.prompt, leaves)
    top = candidates[0] if candidates else None

    if args.verbose:
        print("Router results:")
        for c in candidates[:5]:
            print(f"  {c.route_key}  confidence={c.confidence}")
        print()

    if top is None:
        print("Router returned no candidates.", file=sys.stderr)
        return 1

    threshold = args.threshold if args.threshold is not None else get_threshold(registry_root, top.route_key, args.role)

    if top.confidence < threshold:
        print(format_clarification(candidates, threshold))
        return 2

    resolved = resolve_target(registry_root, top.route_key, requested_role=args.role, selector="production")

    if args.verbose or args.dry_run:
        print(f"Routed to: {resolved['route_key']}  (confidence: {top.confidence})")
        print(f"  version:  {resolved['version']}")
        print(f"  base:     {resolved['base_model']}")
        print(f"  adapter:  {resolved['adapter_path']}")
        print(f"  pool:     {resolved['serving_pool']}")
        print()

    messages: list[dict[str, str]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})

    payload = {
        "model": resolved["route_key"],
        "messages": messages,
        "max_tokens": args.max_tokens,
    }
    if args.temperature is not None:
        payload["temperature"] = args.temperature

    if args.dry_run:
        print("Payload:")
        print(json.dumps(payload, indent=2))
        return 0

    response = post_chat(args.base_url, payload)
    print(json.dumps(response, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
