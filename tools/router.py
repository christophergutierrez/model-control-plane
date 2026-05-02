#!/usr/bin/env python3
"""Route classification: map a user query to a route key with confidence.

Two implementations:
  - keyword_route: fast keyword-matching fallback (no dependencies)
  - EmbeddingRouter: sentence-transformer cosine similarity (requires sentence-transformers)

The orchestrator depends only on the return type: list[RouteCandidate].
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RouteCandidate:
    route_key: str
    confidence: float


# ---------------------------------------------------------------------------
# Keyword fallback (no external dependencies)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Embedding router (requires sentence-transformers)
# ---------------------------------------------------------------------------

class EmbeddingRouter:
    """Route queries using cosine similarity against route description embeddings."""

    def __init__(
        self,
        descriptions_path: str | Path,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: str | Path | None = None,
    ):
        import numpy as np
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        desc_path = Path(descriptions_path)
        descriptions: dict[str, str] = json.loads(desc_path.read_text())
        self.route_keys = list(descriptions.keys())

        cache_path = self._cache_path(desc_path, model_name, cache_dir)
        if cache_path.exists():
            self.route_embeddings = np.load(cache_path)
        else:
            self.route_embeddings = self.model.encode(
                list(descriptions.values()), normalize_embeddings=True,
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, self.route_embeddings)

    @staticmethod
    def _cache_path(desc_path: Path, model_name: str, cache_dir: str | Path | None) -> Path:
        content_hash = hashlib.sha256(desc_path.read_bytes()).hexdigest()[:12]
        name = f"route_embeddings_{model_name.replace('/', '_')}_{content_hash}.npy"
        if cache_dir:
            return Path(cache_dir) / name
        return desc_path.parent / ".cache" / name

    def route(self, query: str) -> list[RouteCandidate]:
        query_emb = self.model.encode([query], normalize_embeddings=True)
        scores = (query_emb @ self.route_embeddings.T)[0]

        paired = sorted(zip(self.route_keys, scores.tolist()), key=lambda x: -x[1])

        # Calibrate raw cosine similarity to a 0-1 confidence range.
        # Floor ~0.15 (typical noise), ceiling ~0.50 (moderate semantic match).
        floor = 0.15
        ceiling = 0.50
        span = ceiling - floor

        candidates = []
        for route_key, raw in paired:
            confidence = max(0.0, min(1.0, (raw - floor) / span))
            candidates.append(RouteCandidate(route_key=route_key, confidence=round(confidence, 3)))

        return candidates
