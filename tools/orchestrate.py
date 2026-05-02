#!/usr/bin/env python3
"""Orchestrator: route a user query through classification, registry resolution, and vLLM dispatch.

Flow:
  1. Load available leaf routes from the registry.
  2. Run the router to classify the query -> route_key + confidence.
  3. If confidence < threshold, print a clarification request and exit.
  4. Resolve the route through the registry to get the production target.
  5. Dispatch to vLLM using the route key as the OpenAI model name.

Modes:
  - Single query:  orchestrate.py /home/chris/models "List all programs"
  - Interactive:   orchestrate.py /home/chris/models --interactive
  - HTTP server:   orchestrate.py /home/chris/models --serve --port 8080
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


def handle_query(
    prompt: str,
    registry_root: Path,
    emb_router: EmbeddingRouter | None,
    leaves: list[str],
    args: argparse.Namespace,
) -> dict:
    """Route and dispatch a single query. Returns a result dict."""
    if emb_router is not None:
        candidates = emb_router.route(prompt)
    else:
        candidates = keyword_route(prompt, leaves)
    top = candidates[0] if candidates else None

    if top is None:
        return {"status": "error", "message": "Router returned no candidates."}

    threshold = args.threshold if args.threshold is not None else get_threshold(registry_root, top.route_key, args.role)

    if top.confidence < threshold:
        return {
            "status": "clarification",
            "threshold": threshold,
            "candidates": [{"route_key": c.route_key, "confidence": c.confidence} for c in candidates[:5]],
        }

    resolved = resolve_target(registry_root, top.route_key, requested_role=args.role, selector="production")

    routing = {
        "route_key": resolved["route_key"],
        "confidence": top.confidence,
        "version": resolved["version"],
        "base_model": resolved["base_model"],
        "adapter_path": resolved["adapter_path"],
        "serving_pool": resolved["serving_pool"],
    }

    if args.dry_run:
        return {"status": "dry_run", "routing": routing}

    messages: list[dict[str, str]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": resolved["route_key"],
        "messages": messages,
        "max_tokens": args.max_tokens,
    }
    if args.temperature is not None:
        payload["temperature"] = args.temperature

    response = post_chat(args.base_url, payload)
    return {"status": "ok", "routing": routing, "response": response}


def run_interactive(registry_root, emb_router, leaves, args):
    print("Orchestrator ready. Type a query, or 'quit' to exit.\n")
    while True:
        try:
            prompt = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt or prompt.lower() in ("quit", "exit", "q"):
            break

        result = handle_query(prompt, registry_root, emb_router, leaves, args)

        if result["status"] == "clarification":
            print(format_clarification(
                [RouteCandidate(c["route_key"], c["confidence"]) for c in result["candidates"]],
                result["threshold"],
            ))
        elif result["status"] == "ok":
            routing = result["routing"]
            content = result["response"]["choices"][0]["message"]["content"]
            print(f"[{routing['route_key']}] {content}")
        elif result["status"] == "dry_run":
            print(json.dumps(result["routing"], indent=2))
        else:
            print(result.get("message", "Unknown error"), file=sys.stderr)
        print()


def run_server(registry_root, emb_router, leaves, args):
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            prompt = body.get("prompt", "")
            if not prompt:
                self._respond(400, {"error": "missing 'prompt' field"})
                return
            result = handle_query(prompt, registry_root, emb_router, leaves, args)
            self._respond(200, result)

        def _respond(self, code, data):
            payload = json.dumps(data).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt, *a):
            sys.stderr.write(f"{self.client_address[0]} {fmt % a}\n")

    HTTPServer.allow_reuse_address = True
    server = HTTPServer((args.serve_host, args.serve_port), Handler)
    print(f"Orchestrator serving on http://{args.serve_host}:{args.serve_port}")
    print(f"  POST /  with {{\"prompt\": \"...\"}}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("registry_root", help="Path to the registry root")
    parser.add_argument("prompt", nargs="?", default=None, help="User query (omit for --interactive or --serve)")
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
    parser.add_argument("--interactive", action="store_true", help="Interactive REPL mode (keeps router warm)")
    parser.add_argument("--serve", action="store_true", help="Run as HTTP server (keeps router warm)")
    parser.add_argument("--serve-host", default="127.0.0.1")
    parser.add_argument("--serve-port", type=int, default=8080)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    registry_root = Path(args.registry_root).expanduser()

    leaves = get_leaf_routes(registry_root)
    if not leaves:
        print("No leaf routes found in registry.", file=sys.stderr)
        return 1

    emb_router = None
    if args.router == "embedding":
        desc_path = args.descriptions or Path(__file__).parent / "route_descriptions.json"
        print("Loading embedding router...", file=sys.stderr)
        emb_router = EmbeddingRouter(desc_path)
        print("Router ready.", file=sys.stderr)

    if args.serve:
        run_server(registry_root, emb_router, leaves, args)
        return 0

    if args.interactive:
        run_interactive(registry_root, emb_router, leaves, args)
        return 0

    if not args.prompt:
        parser.error("prompt is required unless using --interactive or --serve")

    result = handle_query(args.prompt, registry_root, emb_router, leaves, args)

    if result["status"] == "clarification":
        print(format_clarification(
            [RouteCandidate(c["route_key"], c["confidence"]) for c in result["candidates"]],
            result["threshold"],
        ))
        return 2

    if result["status"] == "ok":
        if args.verbose:
            routing = result["routing"]
            print(f"Routed to: {routing['route_key']}  (confidence: {routing['confidence']})")
            print(f"  version:  {routing['version']}")
            print(f"  base:     {routing['base_model']}")
            print(f"  adapter:  {routing['adapter_path']}")
            print(f"  pool:     {routing['serving_pool']}")
            print()
        print(json.dumps(result["response"], indent=2))
        return 0

    if result["status"] == "dry_run":
        print(json.dumps(result["routing"], indent=2))
        return 0

    print(result.get("message", "Unknown error"), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
