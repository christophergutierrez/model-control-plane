# vLLM

This repository integrates with `vLLM` by resolving logical routes from the registry into named static LoRA modules.

The intended shape is:

- one `vLLM` process per base model family
- one shared base model per process
- many static LoRA modules registered at startup
- requests routed by logical route key, using that same route key as the OpenAI `model`

## Current Integration

Launch and test scripts:

- `tools/launch_vllm.sh` — one-command server start with pinned environment
- `tools/serve_vllm.py` — builds the vLLM command from the registry
- `tools/test_vllm_chat.py` — sends a single route-keyed chat request
- `tools/test_all_routes.py` — exercises all production responder routes
- `tools/orchestrate.py` — full dispatch: query → router → confidence check → registry → vLLM

The current reference deployment on this machine uses:

- base model `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- production responder routes under `/home/chris/models`
- eager mode during bring-up because compile mode failed in the local `vLLM` environment
- `HF_HOME` must point to the expanded path where the model is cached

## Route Resolution

Resolve a production target:

```bash
python3 tools/resolve_route.py /home/chris/models videoamp/api/programs
```

This returns:

- route key
- role
- selected version
- base model
- serving pool
- adapter path

## Launching vLLM

Quickest path:

```bash
tools/launch_vllm.sh
```

This sets `HF_HOME`, activates the venv, and runs `serve_vllm.py` with the pinned flags.

Manual equivalent:

```bash
python3 tools/serve_vllm.py /home/chris/models videoamp/api/programs --role responder --port 8000 --dtype bfloat16 -- --enforce-eager
```

Preview without launching:

```bash
python3 tools/serve_vllm.py /home/chris/models videoamp/api/programs --role responder --print-only
```

This wrapper derives:

- the shared base model
- the full list of production responder adapters for that base model
- one `--lora-modules` entry per route, using the route key as the LoRA model name

## Testing a Route

Once `vLLM` is running:

```bash
python3 tools/test_vllm_chat.py /home/chris/models videoamp/api/programs "List 5 programs"
```

This sends a `v1/chat/completions` request with:

- `model` set to the logical route key, for example `videoamp/api/programs`
- OpenAI-compatible chat messages

## Orchestrator

The orchestrator handles the full dispatch loop: query → route classification → confidence check → registry resolution → vLLM dispatch.

```bash
python3 tools/orchestrate.py /home/chris/models "List all programs" --verbose
```

If the router is confident, the query is dispatched to the matched route's LoRA adapter. If confidence is below the route's threshold, the orchestrator asks for clarification instead of guessing.

Dry run (routes and resolves without calling vLLM):

```bash
python3 tools/orchestrate.py /home/chris/models "Show me audience exports" --dry-run --verbose
```

The current router is a keyword-matching stub (`tools/router.py`). It will be replaced by a trained model.

### Legacy Orchestrator Stub

The older `orchestrate_vllm_chat.py` assumes you already know the route key. It is still available for direct dispatch:

```bash
python3 tools/orchestrate_vllm_chat.py /home/chris/models videoamp/api/programs "List 5 programs" --print-payload
```

## Rollout Tools

Register a candidate version:

```bash
python3 tools/register_candidate.py /home/chris/models videoamp/api/programs v2 --weight 10
```

Promote a version to production:

```bash
python3 tools/promote.py /home/chris/models videoamp/api/programs v2
```

Roll back to a previous version:

```bash
python3 tools/rollback.py /home/chris/models videoamp/api/programs v1
```

## Notes

- The registry is the source of truth for production route-to-version mapping.
- `vLLM` is the source of truth for inference execution.
- The orchestrator should route to logical routes such as `videoamp/api/programs`, not directly to filesystem paths.
- The current `vLLM` environment on this machine expects `transformers >= 4.56.0`.
- The current launcher preloads production responder adapters for one base model family and exposes each route key as a named OpenAI model.
- `HF_HOME` must be set to the expanded path containing the cached model. The tilde form (`~`) does not propagate to vLLM's EngineCore subprocess. Use the full absolute path.
- The vLLM venv is at `/home/chris/vllm-install/.vllm`. vLLM version: `0.11.1rc4`.
