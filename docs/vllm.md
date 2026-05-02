# vLLM

This repository integrates with `vLLM` by resolving logical routes from the registry into named static LoRA modules.

The intended shape is:

- one `vLLM` process per base model family
- one shared base model per process
- many static LoRA modules registered at startup
- requests routed by logical route key, using that same route key as the OpenAI `model`

## Current Integration

The current integration uses three scripts:

- `tools/serve_vllm.py`
- `tools/test_vllm_chat.py`
- `tools/orchestrate_vllm_chat.py`

The current reference deployment on this machine uses:

- base model `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- production responder routes under `/home/chris/models`
- eager mode during bring-up because compile mode failed in the local `vLLM` environment

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

Launch vLLM for the base model implied by a route and preload all production responders for that base model:

```bash
python3 tools/serve_vllm.py /home/chris/models videoamp/api/programs --role responder --print-only
```

Typical execution:

```bash
python3 tools/serve_vllm.py /home/chris/models videoamp/api/programs --role responder --port 8000 --dtype bfloat16
```

Compatibility bring-up example:

```bash
python3 tools/serve_vllm.py /home/chris/models videoamp/api/programs --role responder --port 8000 --dtype bfloat16 -- --enforce-eager
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

## Orchestrator Stub

The orchestrator stub is intentionally narrow. It assumes you already know the logical route and want to exercise the dispatch path:

```bash
python3 tools/orchestrate_vllm_chat.py /home/chris/models videoamp/api/programs "List 5 programs" --print-payload
```

This script:

1. resolves the logical route through the registry
2. verifies that the requested target is the production responder
3. builds a `vLLM` chat request using the route key as the model name
4. sends the request to `vLLM`

## Notes

- The registry is the source of truth for production route-to-version mapping.
- `vLLM` is the source of truth for inference execution.
- The orchestrator should route to logical routes such as `videoamp/api/programs`, not directly to filesystem paths.
- The current `vLLM` environment on this machine expects `transformers >= 4.56.0`.
- The current launcher preloads production responder adapters for one base model family and exposes each route key as a named OpenAI model.
