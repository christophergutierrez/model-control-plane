# model-control-plane

Control-plane contract for local model registries used to serve base models, adapters, and route metadata on a machine.

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
- small tools to scaffold and validate a registry

## What Does Not Live Here

- base model weights
- adapter weights
- runtime secrets
- generated runtime state that is specific to one machine

Those belong in a configurable registry root on the server, for example:

- `/srv/models`
- `/opt/models`
- `/home/chris/models`

## Core Ideas

- The registry root is configurable and machine-local.
- Logical routes are separate from physical adapter artifacts.
- Artifact versions are immutable.
- Rollout policy is mutable.
- The router model predicts `route_key + confidence`.
- The orchestrator decides clarification, version selection, and traffic split.

## Repository Layout

```text
docs/
schemas/
examples/
tools/
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
2. Copy [examples/registry.yaml](examples/registry.yaml) to your target machine as `<registry_root>/registry.yaml`.
3. Use `tools/scaffold.py` to create the minimum route and adapter directories you need.
4. Place adapter artifacts into the scaffolded version directories.
5. Run `tools/validate.py <registry_root>` before wiring the registry into an orchestrator or serving pool.

## Design Goals

- minimal required files and directories
- explicit machine-readable contracts
- easy for humans and coding agents to inspect
- no hardcoded assumption about one product hierarchy
- safe support for multiple base models and multiple rollout candidates

## Audience

This repository assumes an operator may use Claude Code, Codex, Gemini, or a human shell workflow to install and maintain a model registry on a server.
