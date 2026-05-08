# vLLM Multi-LoRA Serving

This repository integrates with vLLM's multi-LoRA serving to host one base model with many LoRA adapters loaded simultaneously.

## Architecture

- One vLLM process per base model family
- One shared base model per process
- Many LoRA adapters registered at startup (router + all responders)
- Each adapter is addressable by its route key as the OpenAI `model` name
- The router adapter runs inside the same vLLM process as the responders

## Launch

Build and run the vLLM command from the registry:

```bash
python3 tools/serve_vllm.py <registry_root> <anchor_route> --port 8000
```

The anchor route (e.g. `acme/api/products`) identifies the base model family. All production adapters sharing that base model are auto-discovered and loaded.

Preview without launching:

```bash
python3 tools/serve_vllm.py <registry_root> <anchor_route> --print-only
```

Extra vLLM flags go after `--`:

```bash
python3 tools/serve_vllm.py <registry_root> <anchor_route> --port 8000 -- --enforce-eager
```

## Route Resolution

Resolve a production target from the registry:

```bash
python3 tools/resolve_route.py <registry_root> acme/api/products
```

Returns: route key, role, selected version, base model, serving pool, and adapter path.

## Orchestrator

The orchestrator handles the full dispatch loop: query -> router -> confidence check -> registry resolution -> vLLM dispatch.

```bash
python3 tools/orchestrate.py <registry_root> "List all products" --router lora
```

Router options:

- `--router lora` — uses the LoRA router adapter (recommended)
- `--router embedding` — uses sentence-transformer cosine similarity
- `--router hybrid` — LoRA first, embedding fallback
- `--router keyword` — simple keyword matching

Dry run (routes and resolves without calling vLLM):

```bash
python3 tools/orchestrate.py <registry_root> "Show me products" --dry-run
```

### Persistent Modes

**Interactive REPL** — loads the router once, accepts queries:

```bash
python3 tools/orchestrate.py <registry_root> --interactive --router lora
```

**HTTP server** — keeps the router warm, serves a web dashboard:

```bash
python3 tools/orchestrate.py <registry_root> --serve --serve-port 8080 --router lora
```

Query it:

```bash
curl -s http://127.0.0.1:8080 -d '{"prompt": "list all products"}' | python3 -m json.tool
```

## Testing

Test a single route:

```bash
python3 tools/test_vllm_chat.py <registry_root> acme/api/products "List 5 products"
```

Test all production routes:

```bash
python3 tools/test_all_routes.py <registry_root>
```

## Rollout Tools

Register a candidate version:

```bash
python3 tools/register_candidate.py <registry_root> acme/api/products v2 --weight 10
```

Promote a version to production:

```bash
python3 tools/promote.py <registry_root> acme/api/products v2
```

Roll back to a previous version:

```bash
python3 tools/rollback.py <registry_root> acme/api/products v1
```

## Merged Model Serving

As an alternative to multi-LoRA, adapters can be merged into a single full model via DARE-TIES (see trainLLM's `merge.py`). A merged model eliminates adapter swaps and enables KV cache sharing across multi-step chained calls.

### Registry format

Merged models use `"format": "merged-full"` in their manifest (vs `"peft-lora"` for adapters). The manifest also includes a `merge_config` block recording the merge method, density, and source adapters.

### Serving

`serve_vllm.py` auto-detects the format. If the resolved target is `merged-full`, it serves the model directly without `--enable-lora`:

```bash
python3 tools/serve_vllm.py <registry_root> videoamp/api-merged --port 8000
```

This produces a standard `vllm serve <model_path>` command with no LoRA flags.

### Route configuration

Merged routes use `"serving_mode": "merged"` in their `route.json` roleConfig and a separate `serving_pool` (e.g., `qwen25_coder_1p5b_merged`):

```json
{
  "route_key": "videoamp/api-merged",
  "kind": "leaf",
  "roles": {
    "responder": {
      "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
      "serving_pool": "qwen25_coder_1p5b_merged",
      "serving_mode": "merged",
      "production": {"version": "v1"},
      "candidates": []
    }
  }
}
```

### Batch retraining with merge

`retrain_all.py` supports an optional merge step after training all responder adapters:

```bash
python3 tools/retrain_all.py --merge --merge-density 0.9
```

This trains all adapters, then merges them via DARE-TIES and promotes the merged model to the registry.

### Multi-LoRA vs. merged tradeoffs

| | Multi-LoRA | Merged model |
|---|---|---|
| Serving | One base + many adapters | Single standalone model |
| KV cache | Invalidated on adapter swap | Shared across all endpoints |
| Quality | Per-endpoint specialization | Slight degradation from merge |
| Multi-step chains | Each step recomputes KV | Second step reuses KV (sublinear) |
| Adding endpoints | Train + register a new adapter | Retrain or re-merge |

Both paths coexist — the registry, schemas, and serving tools support either format.

## Importing Adapters

Import a trained adapter from trainLLM's `result.json` into the registry:

```bash
python3 tools/import_adapter.py <registry_root> <route_key> <version> \
  --result /path/to/result.json --promote
```

Example (LoRA adapter):

```bash
python3 tools/import_adapter.py ~/models acme/api/products v2 \
  --result ~/git_home/trainLLM/lora/acme-api-products/final/result.json --promote
```

Example (merged model):

```bash
python3 tools/import_adapter.py ~/models acme/api-merged v1 \
  --result ~/git_home/trainLLM/merged/default/result.json --promote
```

Without `--promote`, the version is registered as a candidate. Use `promote.py` to make it production later.

### Deployment spec

For batch retraining, `retrain_all.py` reads a deployment spec that locates trainLLM and per-endpoint configs:

```bash
python3 tools/retrain_all.py --spec deploy/spec.json
```

Copy `deploy/spec.example.json` to `deploy/spec.json` and fill in your paths. The spec is gitignored — each deployment has its own.

## Notes

- The registry is the source of truth for production route-to-version mapping.
- vLLM is the source of truth for inference execution.
- The orchestrator routes to logical route keys, not directly to filesystem paths.
- `HF_HOME` must be set to the expanded absolute path containing the cached model (tilde form does not propagate to vLLM's EngineCore subprocess).
- `--enforce-eager` may be needed if `torch.compile`/CUDA graphs cause issues with LoRA on your vLLM build.
