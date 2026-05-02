# AI Getting Started

This file is for coding agents and operators using coding agents.

## Goal

Understand the repository fast enough to:

- locate the machine-local model registry
- validate it
- resolve a logical route to a concrete production adapter
- launch the current reference serving backend

## Read Order

1. `README.md`
2. `docs/architecture.md`
3. `docs/operator-workflow.md`
4. `docs/vllm.md`

Do not start by reading every example or schema file unless you are changing the contract.

## Key Concepts

- A `route_key` is a logical identifier such as `videoamp/api/programs`.
- The registry root is machine-local and configurable.
- Route metadata lives under `routes/`.
- Adapter artifacts live under `adapters/`.
- Version directories such as `v1` are immutable.
- Rollout changes happen in `route.json`, not by editing old artifacts.

## Important Files

- `tools/registry_lib.py`
  Resolves a route key to its production target.
- `tools/resolve_route.py`
  Prints one resolved target from the registry.
- `tools/validate.py`
  Validates the registry contract.
- `tools/serve_vllm.py`
  Builds and runs the current `vLLM` multi-LoRA launch command.
- `tools/orchestrate_vllm_chat.py`
  Sends a route-key-based OpenAI chat request to the running server.

## Current Reference Flow

1. Registry stores route metadata and adapter versions.
2. Orchestrator resolves a route key from the registry.
3. `vLLM` serves one base model plus many named LoRA adapters.
4. The request `model` field is the route key itself.

Example:

- route key: `videoamp/api/programs`
- base model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- adapter path:
  `/home/chris/models/adapters/Qwen/Qwen2.5-Coder-1.5B-Instruct/videoamp/api/programs/responder/v1`

## Minimal Commands

Validate a registry:

```bash
python3 tools/validate.py /home/chris/models
```

Resolve one route:

```bash
python3 tools/resolve_route.py /home/chris/models videoamp/api/programs
```

Preview the current `vLLM` launch command:

```bash
python3 tools/serve_vllm.py /home/chris/models videoamp/api/programs --print-only
```

Send one request once the server is up:

```bash
python3 tools/orchestrate_vllm_chat.py /home/chris/models videoamp/api/programs "List 5 programs" --base-url http://127.0.0.1:8000 --print-payload
```

## Rules of Thumb

- Do not rewrite route naming without checking the full hierarchy design.
- Do not mutate released version directories.
- Do not treat training output directories as the permanent source of truth.
- Prefer the registry contract over conversational assumptions.
- If serving fails, separate control-plane issues from runtime/kernel issues.

## Current Reality

As of the current setup:

- `vLLM` is the active backend direction.
- `LoRAX` was explored and then removed because it was a poor fit for this host/runtime combination.
- `vLLM` on this machine may need `--enforce-eager` during bring-up.

If you are picking up future work, start from the `vLLM` path unless new evidence says otherwise.
