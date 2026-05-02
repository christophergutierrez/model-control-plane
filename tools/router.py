#!/usr/bin/env python3
"""Route classification interface and keyword-matching stub.

The interface is: given a user query and a list of available route keys,
return ranked candidates with confidence scores.

The keyword stub is a placeholder. Swap it for an LLM-based or trained
classifier later — the orchestrator only depends on the return type.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class RouteCandidate:
    route_key: str
    confidence: float


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def _leaf_tokens(route_key: str) -> set[str]:
    leaf = route_key.rsplit("/", 1)[-1]
    return set(re.findall(r"[a-z]+", leaf.lower()))


def _stem(word: str) -> str:
    for suffix in ("es", "s", "ing", "tion", "ment"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _stem_set(tokens: set[str]) -> set[str]:
    return {_stem(t) for t in tokens}


def keyword_route(query: str, available_routes: list[str]) -> list[RouteCandidate]:
    """Score routes by keyword overlap between query and the leaf segment of the route key."""
    query_tokens = _tokenize(query)
    query_stems = _stem_set(query_tokens)

    scored: list[tuple[str, float, float]] = []
    for route_key in available_routes:
        leaf_tokens = _leaf_tokens(route_key)
        leaf_stems = _stem_set(leaf_tokens)

        exact = len(query_tokens & leaf_tokens)
        stem = len(query_stems & leaf_stems)
        overlap = exact + 0.5 * (stem - exact)

        if overlap > 0:
            score = min(overlap / max(len(leaf_tokens), 1), 1.0)
            scored.append((route_key, score, overlap))

    scored.sort(key=lambda x: (-x[1], -x[2]))

    if not scored:
        return [RouteCandidate(route_key=available_routes[0], confidence=0.0)] if available_routes else []

    return [RouteCandidate(route_key=r, confidence=round(s, 3)) for r, s, _ in scored]
