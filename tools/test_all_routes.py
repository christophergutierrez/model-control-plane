#!/usr/bin/env python3
"""Test all production responder routes through vLLM."""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

from registry_lib import list_production_targets

PROMPTS = {
    "videoamp/api/audience": "Describe the audience endpoint",
    "videoamp/api/audience-exports": "How do I export audiences?",
    "videoamp/api/audience-statuses": "What audience statuses are there?",
    "videoamp/api/audiences": "List available audiences",
    "videoamp/api/consents": "Show consent records",
    "videoamp/api/currency-of-record": "What is the currency of record?",
    "videoamp/api/episode": "Get episode details",
    "videoamp/api/episodes": "List all episodes",
    "videoamp/api/me": "Show my account info",
    "videoamp/api/measurement": "Describe this measurement",
    "videoamp/api/measurements": "List all measurements",
    "videoamp/api/media-groups": "What media groups exist?",
    "videoamp/api/metric-types": "List metric types",
    "videoamp/api/network": "Describe this network",
    "videoamp/api/networks": "List all networks",
    "videoamp/api/program": "Describe this program",
    "videoamp/api/programs": "List 5 programs",
}


def post_chat(base_url: str, model: str, prompt: str, max_tokens: int = 64) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    registry_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/chris/models")
    base_url = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000"

    targets = list_production_targets(registry_root, role="responder")
    print(f"Testing {len(targets)} production responder routes\n")

    passed = 0
    failed = 0
    results = []

    for target in targets:
        route_key = target["route_key"]
        prompt = PROMPTS.get(route_key, f"Describe {route_key}")
        sys.stdout.write(f"  {route_key:42s} ")
        sys.stdout.flush()

        start = time.time()
        try:
            response = post_chat(base_url, route_key, prompt)
            elapsed = time.time() - start
            content = response["choices"][0]["message"]["content"]
            tokens = response["usage"]["completion_tokens"]
            print(f"OK  {elapsed:.2f}s  {tokens} tok  {content[:60]}...")
            passed += 1
            results.append({"route_key": route_key, "status": "ok", "elapsed_s": round(elapsed, 2), "tokens": tokens})
        except Exception as exc:
            elapsed = time.time() - start
            print(f"FAIL  {elapsed:.2f}s  {exc}")
            failed += 1
            results.append({"route_key": route_key, "status": "fail", "elapsed_s": round(elapsed, 2), "error": str(exc)})

    print(f"\n{passed} passed, {failed} failed out of {len(targets)}")

    report_path = Path("/tmp/route_test_results.json")
    report_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Results written to {report_path}")
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
