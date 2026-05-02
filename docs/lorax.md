# LoRAX

This repository integrates with LoRAX by resolving logical routes from the registry into local adapter paths.

LoRAX is expected to serve:

- one base model family per process
- any number of compatible adapters loaded dynamically per request

## Current Integration

The current integration uses three scripts:

- `tools/resolve_route.py`
- `tools/serve_lorax.py`
- `tools/test_lorax_chat.py`

A minimal orchestrator stub is also provided:

- `tools/orchestrate_lorax_chat.py`

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

## Launching LoRAX

Launch LoRAX for the base model implied by a route:

```bash
python3 tools/serve_lorax.py /home/chris/models videoamp/api/programs --role responder --print-only
```

Typical execution:

```bash
python3 tools/serve_lorax.py /home/chris/models videoamp/api/programs --role responder --port 8080 --dtype bfloat16
```

This wrapper derives the base model from the registry and starts LoRAX with:

- `--model-id <base_model>`
- `--adapter-source local`

It does not preload adapters. Adapters are resolved dynamically from the registry and loaded per request by LoRAX.

## Testing a Route

Once LoRAX is running:

```bash
python3 tools/test_lorax_chat.py /home/chris/models videoamp/api/programs "List 5 programs"
```

This script:

1. resolves the route to its production adapter path
2. sends a `v1/chat/completions` request to LoRAX
3. sets:
   - `model` to the local adapter path
   - `adapter_source` to `local`

## Orchestrator Stub

The orchestrator stub is intentionally narrow. It assumes you already know the logical route and want to exercise the dispatch path:

```bash
python3 tools/orchestrate_lorax_chat.py /home/chris/models videoamp/api/programs "List 5 programs" --print-payload
```

This script:

1. resolves the logical route through the registry
2. resolves the production version for the requested role
3. builds a LoRAX chat request using the local adapter path
4. sends the request to LoRAX

It does not yet perform model-based routing or clarification.

## Notes

- The registry is the source of truth for production route-to-version mapping.
- LoRAX is the source of truth for dynamic adapter loading and inference execution.
- The orchestrator should route to logical routes such as `videoamp/api/programs`, not directly to filesystem paths.
