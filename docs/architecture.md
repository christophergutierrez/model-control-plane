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

A trained LoRA adapter that classifies user queries to route keys. The router does more than simple classification — it can determine multi-step execution plans when a query requires chaining endpoints (e.g. fetching an ID before looking up details).

The router does not rewrite or improve the user query. The original query is passed verbatim to the matched responder adapter.

Three router implementations are available:

- **LoRA router** — a fine-tuned adapter served alongside responders via vLLM multi-LoRA. Outputs a route key directly as text. Supports multi-step chain detection.
- **Embedding router** — encodes route descriptions and user queries with a sentence-transformer, ranks by cosine similarity.
- **Hybrid router** — tries the LoRA router first, falls back to embedding on low confidence.

### Orchestrator

A non-LLM control layer that:

- invokes the router
- applies confidence thresholds
- asks clarifying questions on low confidence
- resolves logical routes to concrete versions
- applies rollout policy
- dispatches to the correct serving pool

The orchestrator can run as a one-shot CLI, an interactive REPL, or an HTTP server with a web dashboard.

### Serving Pool

A runtime that serves one base model family and one or more compatible adapters via vLLM multi-LoRA serving.

One vLLM process loads a single base model and registers all production LoRA adapters (both router and responders) at startup. Each adapter is addressable by its route key as the OpenAI `model` name.

### Model Registry

A machine-local filesystem tree containing:

- global registry config
- route metadata
- adapter versions

## Route Model

Routes are logical identifiers. They are not file paths for weights.

Canonical example:

```text
acme/api/products
```

Routes may be:

- branch routes — contain child routes and typically expose a `router` role
- leaf routes — terminal endpoints that expose a `responder` role

Routes may expose one or more roles:

- `router`
- `responder`

## Artifact Model

Physical adapter artifacts are stored by base model, then by route, then by role, then by immutable version.

Example:

```text
<registry_root>/adapters/Qwen/Qwen2.5-Coder-1.5B-Instruct/acme/api/products/responder/v1/
```

Each version directory contains:

- adapter files (`adapter_model.safetensors`, `adapter_config.json`, etc.)
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
- `~/models`
