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
from router import EmbeddingRouter, LoRARouter, RouteCandidate, keyword_route

SYSTEM_PROMPT = (
    "You are a VideoAmp API assistant. "
    "Given a natural language request, respond with the correct API call "
    "as a JSON object inside a code block. "
    "The JSON must have an \"endpoint\" field (e.g. \"GET /v1/audiences\") "
    "and a \"params\" field containing the query or path parameters."
)


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

    system = args.system if args.system else SYSTEM_PROMPT
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]

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


def get_vllm_models(base_url: str) -> set[str]:
    try:
        req = urllib.request.Request(base_url.rstrip("/") + "/v1/models")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return {m["id"] for m in data.get("data", []) if m.get("parent")}
    except Exception:
        return set()


def build_routes_status(registry_root: Path, emb_router, leaves: list[str], base_url: str) -> list[dict]:
    vllm_models = get_vllm_models(base_url)
    routable = set(emb_router.route_keys) if emb_router else set()

    routes = []
    for route_key in leaves:
        try:
            resolved = resolve_target(registry_root, route_key, requested_role="responder", selector="production")
        except Exception:
            resolved = None

        serving = route_key in vllm_models
        has_description = route_key in routable

        if serving:
            status = "warm"
        elif resolved:
            status = "cold"
        else:
            status = "error"

        entry = {
            "route_key": route_key,
            "status": status,
            "serving": serving,
            "routable": has_description,
        }
        if resolved:
            entry["version"] = resolved["version"]
            entry["base_model"] = resolved["base_model"]
            entry["adapter_path"] = resolved["adapter_path"]

        routes.append(entry)

    return routes


DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Model Control Plane</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
         background: #0d1117; color: #c9d1d9; padding: 24px; }
  h1 { font-size: 20px; font-weight: 600; margin-bottom: 16px; color: #e6edf3; }
  .health { display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
  .health-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                 padding: 12px 20px; min-width: 160px; }
  .health-card .label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px;
                        color: #8b949e; margin-bottom: 4px; }
  .health-card .value { font-size: 24px; font-weight: 600; }
  .health-card .value.ok { color: #3fb950; }
  .health-card .value.err { color: #f85149; }
  table { width: 100%; border-collapse: collapse; background: #161b22;
          border: 1px solid #30363d; border-radius: 8px; overflow: hidden; }
  th { text-align: left; padding: 10px 14px; font-size: 11px; text-transform: uppercase;
       letter-spacing: 0.5px; color: #8b949e; background: #0d1117;
       border-bottom: 1px solid #30363d; }
  td { padding: 10px 14px; border-bottom: 1px solid #21262d; font-size: 13px; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: #1c2128; }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
         margin-right: 8px; vertical-align: middle; }
  .dot.warm { background: #3fb950; }
  .dot.cold { background: #8b949e; }
  .dot.error { background: #f85149; }
  .tag { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px;
         font-weight: 500; }
  .tag.yes { background: #1a3a2a; color: #3fb950; }
  .tag.no { background: #3a1a1a; color: #f85149; }
  .route-key { color: #58a6ff; font-weight: 500; }
  .mono { font-family: monospace; font-size: 12px; color: #8b949e; }
  .updated { font-size: 11px; color: #484f58; margin-top: 16px; }
  .query-box { margin-bottom: 24px; display: flex; gap: 8px; }
  .query-box input { flex: 1; padding: 8px 12px; background: #0d1117; border: 1px solid #30363d;
                     border-radius: 6px; color: #c9d1d9; font-size: 14px; font-family: inherit; }
  .query-box input:focus { outline: none; border-color: #58a6ff; }
  .query-box button { padding: 8px 16px; background: #238636; border: 1px solid #2ea043;
                      border-radius: 6px; color: #fff; font-size: 14px; cursor: pointer;
                      font-weight: 500; }
  .query-box button:hover { background: #2ea043; }
  .query-box button:disabled { opacity: 0.5; cursor: not-allowed; }
  .result-box { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                padding: 16px; margin-bottom: 24px; display: none; }
  .result-box .result-route { font-size: 12px; color: #8b949e; margin-bottom: 6px; }
  .result-box .result-content { font-size: 14px; color: #e6edf3; white-space: pre-wrap; }
  .result-box .result-error { color: #f85149; }
</style>
</head>
<body>

<h1>Model Control Plane</h1>

<div class="query-box">
  <input type="text" id="queryInput" placeholder="Ask a question (e.g. list all programs)" autofocus>
  <button id="queryBtn" onclick="sendQuery()">Send</button>
</div>
<div class="result-box" id="resultBox">
  <div class="result-route" id="resultRoute"></div>
  <div class="result-content" id="resultContent"></div>
</div>

<div class="health" id="health"></div>
<table>
  <thead>
    <tr>
      <th>Status</th>
      <th>Route</th>
      <th>Version</th>
      <th>Serving</th>
      <th>Routable</th>
      <th>Base Model</th>
    </tr>
  </thead>
  <tbody id="routeBody"></tbody>
</table>
<div class="updated" id="updated"></div>

<script>
async function refresh() {
  try {
    const [hRes, rRes] = await Promise.all([fetch('/health'), fetch('/routes')]);
    const health = await hRes.json();
    const routes = await rRes.json();

    document.getElementById('health').innerHTML = `
      <div class="health-card">
        <div class="label">Orchestrator</div>
        <div class="value ${health.orchestrator === 'ok' ? 'ok' : 'err'}">${health.orchestrator}</div>
      </div>
      <div class="health-card">
        <div class="label">vLLM</div>
        <div class="value ${health.vllm === 'ok' ? 'ok' : 'err'}">${health.vllm}</div>
      </div>
      <div class="health-card">
        <div class="label">Registered</div>
        <div class="value ok">${health.routes_registered}</div>
      </div>
      <div class="health-card">
        <div class="label">Serving</div>
        <div class="value ${health.routes_serving === health.routes_registered ? 'ok' : 'err'}">${health.routes_serving}</div>
      </div>
    `;

    const tbody = document.getElementById('routeBody');
    tbody.innerHTML = routes.routes.map(r => `
      <tr>
        <td><span class="dot ${r.status}"></span>${r.status}</td>
        <td class="route-key">${r.route_key}</td>
        <td class="mono">${r.version || '-'}</td>
        <td><span class="tag ${r.serving ? 'yes' : 'no'}">${r.serving ? 'yes' : 'no'}</span></td>
        <td><span class="tag ${r.routable ? 'yes' : 'no'}">${r.routable ? 'yes' : 'no'}</span></td>
        <td class="mono">${r.base_model || '-'}</td>
      </tr>
    `).join('');

    document.getElementById('updated').textContent = 'Updated ' + new Date().toLocaleTimeString();
  } catch (e) {
    document.getElementById('health').innerHTML =
      '<div class="health-card"><div class="label">Error</div><div class="value err">fetch failed</div></div>';
  }
}

async function sendQuery() {
  const input = document.getElementById('queryInput');
  const btn = document.getElementById('queryBtn');
  const box = document.getElementById('resultBox');
  const routeEl = document.getElementById('resultRoute');
  const contentEl = document.getElementById('resultContent');
  const prompt = input.value.trim();
  if (!prompt) return;

  btn.disabled = true;
  box.style.display = 'block';
  routeEl.textContent = '';
  contentEl.textContent = 'Routing...';
  contentEl.className = 'result-content';

  try {
    const res = await fetch('/', { method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({prompt}) });
    const data = await res.json();

    if (data.status === 'ok') {
      routeEl.textContent = `${data.routing.route_key}  v${data.routing.version}  conf=${data.routing.confidence}`;
      contentEl.textContent = data.response.choices[0].message.content;
    } else if (data.status === 'clarification') {
      routeEl.textContent = 'Clarification needed';
      contentEl.textContent = data.candidates.map(c => `${c.route_key}  (${c.confidence})`).join('\\n');
      contentEl.className = 'result-content result-error';
    } else if (data.status === 'dry_run') {
      routeEl.textContent = 'Dry run';
      contentEl.textContent = JSON.stringify(data.routing, null, 2);
    } else {
      contentEl.textContent = data.message || 'Unknown error';
      contentEl.className = 'result-content result-error';
    }
  } catch (e) {
    contentEl.textContent = 'Request failed: ' + e.message;
    contentEl.className = 'result-content result-error';
  }
  btn.disabled = false;
}

document.getElementById('queryInput').addEventListener('keydown', e => {
  if (e.key === 'Enter') sendQuery();
});

refresh();
setInterval(refresh, 10000);
</script>
</body>
</html>
"""


def run_server(registry_root, emb_router, leaves, args):
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/routes":
                routes = build_routes_status(registry_root, emb_router, leaves, args.base_url)
                self._respond_json(200, {"routes": routes})
            elif self.path == "/health":
                vllm_models = get_vllm_models(args.base_url)
                self._respond_json(200, {
                    "orchestrator": "ok",
                    "vllm": "ok" if vllm_models else "unreachable",
                    "routes_registered": len(leaves),
                    "routes_serving": len(vllm_models),
                })
            elif self.path in ("/", "/dashboard"):
                payload = DASHBOARD_HTML.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            else:
                self._respond_json(404, {"error": "not found", "endpoints": ["GET /", "GET /routes", "GET /health", "POST /"]})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            prompt = body.get("prompt", "")
            if not prompt:
                self._respond_json(400, {"error": "missing 'prompt' field"})
                return
            result = handle_query(prompt, registry_root, emb_router, leaves, args)
            self._respond_json(200, result)

        def _respond_json(self, code, data):
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
    print(f"  GET  /        — dashboard")
    print(f"  GET  /routes  — route status JSON")
    print(f"  GET  /health  — health check JSON")
    print(f"  POST /        — query dispatch {{\"prompt\": \"...\"}}")
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
    parser.add_argument("--router", choices=["embedding", "keyword", "lora"], default="embedding",
                        help="Router backend (default: embedding)")
    parser.add_argument("--descriptions", default=None,
                        help="Path to route_descriptions.json (default: auto-detect next to this script)")
    parser.add_argument("--interactive", action="store_true", help="Interactive REPL mode (keeps router warm)")
    parser.add_argument("--serve", action="store_true", help="Run as HTTP server (keeps router warm)")
    parser.add_argument("--serve-host", default="0.0.0.0")
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
    elif args.router == "lora":
        print("Using LoRA router (vLLM adapter)...", file=sys.stderr)
        emb_router = LoRARouter(base_url=args.base_url)
        emb_router.set_route_keys(leaves)

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
