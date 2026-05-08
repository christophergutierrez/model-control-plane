# AI Getting Started

This file is for coding agents and operators using coding agents.

## Goal

Understand the repository fast enough to:

- locate the machine-local model registry
- validate it
- resolve a logical route to a concrete production adapter
- launch the serving backend

## Read Order

1. `README.md`
2. `docs/architecture.md`
3. `docs/operator-workflow.md`
4. `docs/vllm.md`

Do not start by reading every example or schema file unless you are changing the contract.

## Key Concepts

- A `route_key` is a logical identifier such as `acme/api/products`.
- The registry root is machine-local and configurable.
- Route metadata lives under `routes/`.
- Adapter artifacts live under `adapters/`.
- Version directories such as `v1` are immutable.
- Rollout changes happen in `route.json`, not by editing old artifacts.
- The router is itself a LoRA adapter served via vLLM multi-LoRA alongside responders.
- The router can determine multi-step execution plans, not just single-endpoint classification.

## Important Files

- `tools/registry_lib.py` — resolves a route key to its production target.
- `tools/resolve_route.py` — prints one resolved target from the registry.
- `tools/validate.py` — validates the registry contract.
- `tools/serve_vllm.py` — builds and runs the vLLM multi-LoRA launch command.
- `tools/orchestrate.py` — full dispatch: query -> router -> confidence check -> registry -> vLLM.

## Current Reference Flow

1. Registry stores route metadata and adapter versions.
2. `serve_vllm.py` launches vLLM with all production adapters (router + responders).
3. Orchestrator invokes the router adapter to classify the query to a route key.
4. Orchestrator resolves the route key to a versioned adapter via the registry.
5. Orchestrator dispatches to the matched responder adapter using its route key as the OpenAI `model` name.

Example:

- route key: `acme/api/products`
- base model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- adapter path: `<registry_root>/adapters/Qwen/Qwen2.5-Coder-1.5B-Instruct/acme/api/products/responder/v1`

## Minimal Commands

Validate a registry:

```bash
python3 tools/validate.py <registry_root>
```

Resolve one route:

```bash
python3 tools/resolve_route.py <registry_root> acme/api/products
```

Preview the vLLM launch command:

```bash
python3 tools/serve_vllm.py <registry_root> acme/api/products --print-only
```

Run the full orchestrator:

```bash
python3 tools/orchestrate.py <registry_root> "List 5 products" --router lora
```

## Rules of Thumb

- Do not rewrite route naming without checking the full hierarchy design.
- Do not mutate released version directories.
- Do not treat training output directories as the permanent source of truth.
- Prefer the registry contract over conversational assumptions.
- If serving fails, separate control-plane issues from runtime/kernel issues.

## Current Reality

- vLLM multi-LoRA is the active serving backend.
- `--enforce-eager` may be needed depending on your vLLM build and GPU.
- The router adapter runs inside the same vLLM process as all responders.

If you are picking up future work, start from the vLLM multi-LoRA path unless new evidence says otherwise.
