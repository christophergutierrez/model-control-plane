# Operator Workflow

## Purpose

This document describes the expected workflow for a human operator or coding agent managing a machine-local model registry.

## Initial Setup

1. Choose a registry root on the server.
2. Create the top-level structure:
   - `registry.yaml`
   - `routes/`
   - `adapters/`
3. Define one or more serving pools in `registry.yaml`.
4. Add routes and version manifests only for models that actually exist.

For the current serving backend, see [vllm.md](vllm.md).

## Adding a Route

1. Decide the canonical `route_key`.
2. Create the route directory under `routes/`.
3. Add a `route.json`.
4. If the route has a model, create the matching adapter version directory and `manifest.json`.

## Adding a New Adapter Version

1. Pick an immutable version name such as `v2`.
2. Create the new version directory under the matching base model, route, and role.
3. Place adapter artifacts into that directory.
4. Add `manifest.json`.
5. Add evaluation artifacts if available.
6. Run validation.
7. Update the route rollout policy if needed.

## Retraining an Adapter

1. Prepare or verify training data.
2. Train the adapter (SFT). Optionally apply DPO for targeted corrections.
3. Promote the trained adapter to the registry as a new version.
4. Update the route's production pointer in `route.json`.
5. Restart vLLM to pick up the new weights.
6. Test the affected routes.

When promoting, be aware of the adapter selection order: if a `dpo_final/` directory exists from a previous DPO pass, the promote logic will prefer it over the fresh SFT adapter. Remove stale DPO directories before promoting if you want pure SFT weights.

## Promoting a Candidate

1. Validate the registry.
2. Update the route's production target in `route.json`.
3. Optionally keep the previous production version as a candidate or mark it deprecated.

## Low Confidence Behavior

The router returns a route key and confidence score.

If confidence is below the configured threshold:

- do not silently route to a default endpoint
- ask a clarifying question

## Operating Principle

The registry on disk is the deployed instance.

This repository defines the contract, but it does not replace judgment about:

- when a candidate should be promoted
- which serving pool should own a route
- what clarification experience is best for end users

## Agent Guidance

If a coding agent is operating this repository or the runtime registry:

1. Start with [architecture.md](architecture.md).
2. Then read [ai-getting-started.md](ai-getting-started.md).
3. Validate the registry before changing rollout policy.
4. Prefer changing route metadata over changing immutable version directories.
5. After retraining, always restart vLLM — it loads adapters at startup and does not hot-reload.
