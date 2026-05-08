# model-control-plane

Control-plane contract for local model registries used to serve base models, LoRA adapters, and route metadata on a machine.

This repository does not store model weights. It defines:

- how a model registry on disk is structured
- how routes, versions, and rollout policy are represented
- how operators and coding agents bootstrap a registry on a server
- how to validate a registry before serving traffic

## What Lives Here

- documentation for the architecture and operator workflow
- JSON schemas for route and manifest files
- an example top-level `registry.yaml`
- examples showing minimal and hierarchical route layouts
- tools to scaffold, validate, launch, route, and manage rollout for a registry

## What Does Not Live Here

- base model weights
- adapter weights
- runtime secrets
- deployment-specific configuration
- generated runtime state that is specific to one machine

Those belong in a configurable registry root on the server.

## Core Ideas

- The registry root is configurable and machine-local.
- Logical routes are separate from physical adapter artifacts.
- Artifact versions are immutable.
- Rollout policy is mutable.
- A router adapter classifies user queries to route keys and can determine multi-step execution plans.
- The orchestrator decides clarification, version selection, and traffic split.

## Repository Layout

```text
docs/
schemas/
examples/
tools/
deployments/
```

## Server Registry Layout

At runtime, a model registry on a server is expected to look like:

```text
<registry_root>/
  registry.yaml
  routes/
  adapters/
```

Only create route and adapter directories that actually exist for that deployment.

## Quick Start

1. Read [docs/architecture.md](docs/architecture.md).
2. Read [docs/ai-getting-started.md](docs/ai-getting-started.md) if you are a coding agent or are using one.
3. Copy [examples/registry.yaml](examples/registry.yaml) to your target machine as `<registry_root>/registry.yaml`.
4. Use `tools/scaffold.py` to create the minimum route and adapter directories you need.
5. Place adapter artifacts into the scaffolded version directories.
6. Run `tools/validate.py <registry_root>` before wiring the registry into an orchestrator or serving pool.

## Serving Backend

The current backend is vLLM multi-LoRA serving:

- one base model per vLLM process
- many LoRA adapters loaded at startup (router + responders)
- logical route keys exposed directly as OpenAI `model` names
- a router adapter classifies user queries to route keys with confidence scores
- the router can plan multi-step endpoint chains, not just single classifications
- an orchestrator dispatches to the correct LoRA adapter or asks for clarification

See [docs/vllm.md](docs/vllm.md) for the concrete integration.

## Design Goals

- minimal required files and directories
- explicit machine-readable contracts
- easy for humans and coding agents to inspect
- no hardcoded assumption about one product hierarchy
- safe support for multiple base models and multiple rollout candidates

## Audience

This repository assumes an operator may use Claude Code, Codex, Gemini, or a human shell workflow to install and maintain a model registry on a server.
