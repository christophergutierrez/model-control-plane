# Architecture

## Scope

This repository defines the control plane for a machine-local model registry.

It does not define:

- how a base model is trained
- how an adapter is trained
- how a serving runtime internally executes inference

It defines:

- how routes are named
- how routes map to versions
- how adapters are organized under base models
- how a router, orchestrator, and serving pool interact

## Components

### Router

A trained model that predicts a route target and a confidence score.

Expected output shape:

```json
{
  "route_key": "videoamp/api/programs",
  "confidence": 0.96
}
```

The router does not decide rollout, clarification wording, or serving backend.

### Orchestrator

A non-LLM control layer that:

- invokes the router
- applies confidence thresholds
- asks clarifying questions on low confidence
- resolves logical routes to concrete versions
- applies rollout policy
- dispatches to the correct serving pool

### Serving Pool

A runtime that serves one base model family and one or more compatible adapters.

Examples include:

- vLLM plus static or dynamic adapter logic

### Model Registry

A machine-local filesystem tree containing:

- global registry config
- route metadata
- adapter versions

## Route Model

Routes are logical identifiers. They are not file paths for weights.

Canonical example:

```text
videoamp/api/programs
```

Routes may be:

- branch routes
- leaf routes

Routes may expose one or more roles:

- `router`
- `responder`

Common convention:

- non-leaf routes usually expose `router`
- leaf routes usually expose `responder`

## Artifact Model

Physical adapter artifacts are stored by base model, then by route, then by role, then by immutable version.

Example:

```text
<registry_root>/adapters/Qwen/Qwen2.5-Coder-1.5B-Instruct/videoamp/api/programs/responder/v1/
```

Each version directory contains:

- adapter files
- `manifest.json`
- optional human-readable notes and eval artifacts

## Mutability Rules

Immutable:

- version directories such as `v1`, `v2`, `v3`
- files inside a released version directory

Mutable:

- `route.json`
- rollout policy
- production/candidate assignments

## Clarification

If router confidence is below the route threshold, the orchestrator should ask a clarifying question rather than silently falling back to a default route.

## Registry Root

The registry root is configurable. This repository does not require a specific path.

Examples:

- `/srv/models`
- `/opt/models`
- `/home/chris/models`
