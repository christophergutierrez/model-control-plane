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

## Promoting a Candidate

1. Validate the registry.
2. Update the route's production target in `route.json`.
3. Optionally keep the previous production version as a candidate or mark it deprecated.

## Low Confidence Behavior

The router returns `route_key + confidence`.

If confidence is below the configured threshold:

- do not silently route to a default endpoint
- ask a clarifying question

## Operating Principle

The registry on disk is the deployed instance.

This repository defines the contract, but it does not replace judgment about:

- when a candidate should be promoted
- which serving pool should own a route
- what clarification experience is best for end users
