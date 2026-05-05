#!/usr/bin/env python3
"""Batch retrain all VideoAmp API adapters and the router.

Steps:
  1. Prepare per-endpoint training data (raw → ShareGPT format)
  2. Train each responder adapter via trainLLM/train.py
  3. Prepare and train the router adapter
  4. Promote all new adapters to the registry as v2

Usage:
    python3 tools/retrain_all.py                    # full run
    python3 tools/retrain_all.py --prepare-only     # just prepare data, don't train
    python3 tools/retrain_all.py --endpoint programs # train one endpoint only
    python3 tools/retrain_all.py --router-only      # train router only
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

TRAINLLM_DIR = Path("~/git_home/trainLLM").expanduser()
RAW_DATA_DIR = TRAINLLM_DIR / "data" / "videoamp"
PREPARED_DIR = TRAINLLM_DIR / "data" / "videoamp_prepared"
RUNS_DIR = TRAINLLM_DIR / "data" / "videoamp_runs"
ROUTER_DATA_DIR = TRAINLLM_DIR / "data" / "videoamp_router"
REGISTRY_ROOT = Path("~/models").expanduser()

UNSLOTH_PYTHON = Path("~/.unsloth/studio/unsloth_studio/bin/python3").expanduser()
TRAIN_SCRIPT = TRAINLLM_DIR / "train.py"
HF_HOME = Path("~/git_home/trainLLM/models/hf").expanduser()

BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

RESPONDER_SYSTEM_PROMPT = (
    "You are a VideoAmp API assistant. "
    "Given a natural language request, respond with the correct API call "
    "as a JSON object inside a code block. "
    'The JSON must have an "endpoint" field (e.g. "GET /v1/audiences") '
    'and a "params" field containing the query or path parameters.'
)

ROUTER_SYSTEM_PROMPT = (
    "You are a route classifier for the VideoAmp API. "
    "Given a user query, respond with only the route key that best matches the request. "
    "Output nothing else — just the route key."
)


def format_api_response(api_call: dict) -> str:
    return "```json\n" + json.dumps(api_call, indent=2) + "\n```"


def raw_to_sharegpt(record: dict) -> dict:
    return {
        "conversations": [
            {"from": "system", "value": RESPONDER_SYSTEM_PROMPT},
            {"from": "human", "value": record["question"]},
            {"from": "gpt", "value": format_api_response(record["api_call"])},
        ]
    }


def router_to_sharegpt(record: dict) -> dict:
    return {
        "conversations": [
            {"from": "system", "value": ROUTER_SYSTEM_PROMPT},
            {"from": "human", "value": record["question"]},
            {"from": "gpt", "value": record["route_key"]},
        ]
    }


def prepare_endpoint(endpoint: str) -> tuple[int, int]:
    """Convert raw data to ShareGPT format for one endpoint. Returns (train_count, holdout_count)."""
    raw_dir = RAW_DATA_DIR / endpoint
    out_dir = PREPARED_DIR / endpoint
    out_dir.mkdir(parents=True, exist_ok=True)

    train_count = 0
    if (raw_dir / "training.jsonl").exists():
        with open(raw_dir / "training.jsonl") as f, open(out_dir / "training.jsonl", "w") as out:
            for line in f:
                record = json.loads(line.strip())
                out.write(json.dumps(raw_to_sharegpt(record)) + "\n")
                train_count += 1

    holdout_count = 0
    if (raw_dir / "holdout.jsonl").exists():
        with open(raw_dir / "holdout.jsonl") as f, open(out_dir / "holdout.jsonl", "w") as out:
            for line in f:
                record = json.loads(line.strip())
                out.write(json.dumps(raw_to_sharegpt(record)) + "\n")
                holdout_count += 1

    return train_count, holdout_count


def prepare_router() -> tuple[int, int]:
    """Convert router data to ShareGPT format."""
    out_dir = PREPARED_DIR / "router"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_count = 0
    with open(ROUTER_DATA_DIR / "router_train.jsonl") as f, \
         open(out_dir / "training.jsonl", "w") as out:
        for line in f:
            record = json.loads(line.strip())
            out.write(json.dumps(router_to_sharegpt(record)) + "\n")
            train_count += 1

    holdout_count = 0
    with open(ROUTER_DATA_DIR / "router_test.jsonl") as f, \
         open(out_dir / "holdout.jsonl", "w") as out:
        for line in f:
            record = json.loads(line.strip())
            out.write(json.dumps(router_to_sharegpt(record)) + "\n")
            holdout_count += 1

    return train_count, holdout_count


def compute_max_steps(train_count: int, batch_size: int = 8, grad_accum: int = 2, epochs: float = 4.0) -> int:
    effective_batch = batch_size * grad_accum
    steps_per_epoch = max(1, train_count // effective_batch)
    return int(steps_per_epoch * epochs)


def write_run_config(endpoint: str, adapter_name: str, train_count: int) -> Path:
    """Write a per-endpoint config.yaml for trainLLM."""
    run_dir = RUNS_DIR / endpoint
    run_dir.mkdir(parents=True, exist_ok=True)

    max_steps = compute_max_steps(train_count)
    save_steps = max(10, max_steps // 4)

    config = {
        "model": BASE_MODEL,
        "adapter_name": adapter_name,
        "chat_template": "qwen-2.5",
        "runtime": "vllm",
        "paths": {
            "base_dir": str(TRAINLLM_DIR),
            "hf_home": str(HF_HOME),
            "unsloth_python": str(UNSLOTH_PYTHON),
        },
        "data": {
            "train": str(PREPARED_DIR / endpoint / "training.jsonl"),
            "holdout": str(PREPARED_DIR / endpoint / "holdout.jsonl"),
        },
        "training": {
            "max_seq_length": 512,
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0,
            "batch_size": 8,
            "gradient_accumulation_steps": 2,
            "warmup_steps": 20,
            "max_steps": max_steps,
            "learning_rate": 2.0e-4,
            "weight_decay": 0.01,
            "lr_scheduler": "cosine",
            "save_steps": save_steps,
            "save_total_limit": None,
        },
        "vllm": {
            "port": 8000,
            "gpu_memory_utilization": 0.85,
        },
        "timeouts": {
            "train_silence": 1800,
            "vllm_startup": 900,
            "vllm_poll_interval": 30,
            "eval_timeout": 3600,
        },
    }

    config_path = run_dir / "config.yaml"
    try:
        import yaml
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    except ImportError:
        # Fallback: write as JSON-compatible YAML
        with open(config_path, "w") as f:
            f.write(json.dumps(config, indent=2))

    return config_path


def train_adapter(config_path: Path, label: str) -> bool:
    """Run train.py with the given config. Returns True on success."""
    env = {
        **os.environ,
        "TRAINLLM_CONFIG": str(config_path),
        "HF_HOME": str(HF_HOME),
    }
    cmd = [str(UNSLOTH_PYTHON), str(TRAIN_SCRIPT)]
    print(f"\n{'='*60}")
    print(f"  Training: {label}")
    print(f"  Config:   {config_path}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, env=env, cwd=str(TRAINLLM_DIR))
    if result.returncode != 0:
        print(f"  FAILED: {label} (exit code {result.returncode})", file=sys.stderr)
        return False
    print(f"  DONE: {label}")
    return True


def promote_to_registry(endpoint: str, adapter_name: str, role: str, version: str = "v2") -> bool:
    """Copy trained adapter to registry and update route.json."""
    lora_dir = TRAINLLM_DIR / "lora" / adapter_name / "final"
    if not lora_dir.exists():
        lora_dir = TRAINLLM_DIR / "lora" / adapter_name
        # Find the latest checkpoint
        checkpoints = sorted(lora_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        if checkpoints:
            lora_dir = checkpoints[-1]
        else:
            print(f"  No adapter found for {adapter_name}", file=sys.stderr)
            return False

    if not (lora_dir / "adapter_model.safetensors").exists():
        print(f"  No adapter_model.safetensors in {lora_dir}", file=sys.stderr)
        return False

    route_key = f"videoamp/api/{endpoint}"
    dest = (REGISTRY_ROOT / "adapters" / "Qwen" / "Qwen2.5-Coder-1.5B-Instruct" /
            route_key / role / version)
    dest.mkdir(parents=True, exist_ok=True)

    # Copy adapter files
    for f in lora_dir.iterdir():
        if f.suffix in (".safetensors", ".json", ".jinja"):
            shutil.copy2(f, dest / f.name)

    # Write manifest
    manifest = {
        "schema_version": 1,
        "route_key": route_key,
        "role": role,
        "version": version,
        "status": "production",
        "base_model": {
            "id": BASE_MODEL,
            "family": BASE_MODEL,
            "vendor": "Qwen",
        },
        "adapter": {
            "format": "peft-lora",
            "path": str(dest),
            "source": "local",
        },
        "created_at": __import__("datetime").datetime.now().isoformat() + "Z",
        "notes": f"Retrained from updated training data.",
    }
    with open(dest / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  Promoted {adapter_name} → {dest}")
    return True


def scaffold_route(endpoint: str):
    """Create route.json for a new endpoint if it doesn't exist."""
    route_key = f"videoamp/api/{endpoint}"
    route_dir = REGISTRY_ROOT / "routes" / route_key
    route_json = route_dir / "route.json"

    if route_json.exists():
        return

    route_dir.mkdir(parents=True, exist_ok=True)
    route_data = {
        "route_key": route_key,
        "kind": "leaf",
        "roles": {
            "responder": {
                "base_model": BASE_MODEL,
                "base_family": BASE_MODEL,
                "serving_pool": "qwen25_coder_1p5b",
                "clarification_threshold": 0.8,
                "production": {"version": "v2"},
                "candidates": [],
            }
        },
        "children": [],
        "status": "active",
    }
    with open(route_json, "w") as f:
        json.dump(route_data, f, indent=2)
    print(f"  Scaffolded route: {route_key}")


def update_route_production(endpoint: str, role: str, version: str):
    """Update route.json to point production at the new version."""
    route_key = f"videoamp/api/{endpoint}"
    route_json = REGISTRY_ROOT / "routes" / route_key / "route.json"
    if not route_json.exists():
        return

    with open(route_json) as f:
        data = json.load(f)

    if role in data.get("roles", {}):
        data["roles"][role]["production"]["version"] = version
    else:
        data["roles"][role] = {
            "base_model": BASE_MODEL,
            "base_family": BASE_MODEL,
            "serving_pool": "qwen25_coder_1p5b",
            "clarification_threshold": 0.8,
            "production": {"version": version},
            "candidates": [],
        }

    with open(route_json, "w") as f:
        json.dump(data, f, indent=2)


def get_endpoints() -> list[str]:
    """List all endpoint directories with training data."""
    endpoints = []
    for d in sorted(RAW_DATA_DIR.iterdir()):
        if d.is_dir() and d.name != "router" and (d / "training.jsonl").exists():
            endpoints.append(d.name)
    return endpoints


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepare-only", action="store_true", help="Only prepare data, skip training")
    parser.add_argument("--endpoint", default=None, help="Train only this endpoint")
    parser.add_argument("--router-only", action="store_true", help="Train only the router")
    parser.add_argument("--skip-router", action="store_true", help="Skip router training")
    parser.add_argument("--skip-promote", action="store_true", help="Skip registry promotion")
    parser.add_argument("--version", default="v2", help="Version label for new adapters (default: v2)")
    args = parser.parse_args()

    endpoints = get_endpoints()
    if args.endpoint:
        if args.endpoint not in endpoints:
            sys.exit(f"Endpoint '{args.endpoint}' not found. Available: {endpoints}")
        endpoints = [args.endpoint]

    # --- Step 1: Prepare data ---
    print("\n" + "="*60)
    print("  STEP 1: Preparing training data")
    print("="*60)

    if not args.router_only:
        for ep in endpoints:
            train_n, holdout_n = prepare_endpoint(ep)
            print(f"  {ep:<25} train={train_n:>4}  holdout={holdout_n:>3}")

    if not args.skip_router:
        train_n, holdout_n = prepare_router()
        print(f"  {'router':<25} train={train_n:>4}  holdout={holdout_n:>3}")

    if args.prepare_only:
        print("\nData prepared. Use --endpoint or remove --prepare-only to train.")
        return

    # --- Step 2: Train responder adapters ---
    if not args.router_only:
        print("\n" + "="*60)
        print("  STEP 2: Training responder adapters")
        print("="*60)

        results = {}
        for ep in endpoints:
            adapter_name = f"videoamp-api-{ep}"
            train_data = PREPARED_DIR / ep / "training.jsonl"
            train_count = sum(1 for _ in open(train_data))

            config_path = write_run_config(ep, adapter_name, train_count)

            success = train_adapter(config_path, ep)
            results[ep] = success

            if success and not args.skip_promote:
                scaffold_route(ep)
                promote_to_registry(ep, adapter_name, "responder", args.version)
                update_route_production(ep, "responder", args.version)

        print("\n" + "="*60)
        print("  Responder Results:")
        for ep, ok in results.items():
            print(f"    {'OK' if ok else 'FAIL'}  {ep}")
        print("="*60)

    # --- Step 3: Train router adapter ---
    if not args.skip_router:
        print("\n" + "="*60)
        print("  STEP 3: Training router adapter")
        print("="*60)

        adapter_name = "videoamp-api-router"
        train_data = PREPARED_DIR / "router" / "training.jsonl"
        train_count = sum(1 for _ in open(train_data))

        config_path = write_run_config("router", adapter_name, train_count)
        success = train_adapter(config_path, "router")

        if success and not args.skip_promote:
            promote_to_registry("router", adapter_name, "router", args.version)
            # Router route uses "router" role, not "responder"
            route_json = REGISTRY_ROOT / "routes" / "videoamp" / "api" / "route.json"
            if route_json.exists():
                with open(route_json) as f:
                    data = json.load(f)
                if "router" in data.get("roles", {}):
                    data["roles"]["router"]["production"]["version"] = args.version
                    with open(route_json, "w") as f:
                        json.dump(data, f, indent=2)

    print("\n" + "="*60)
    print("  COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
