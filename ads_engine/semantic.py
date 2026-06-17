"""Semantic similarity to known-bad terms.

Flags search terms that are semantically close to a curated list of negative / known-bad
terms BEFORE they accumulate enough volume to trigger the statistical tests. This is an
early-warning layer: it catches "kilimanjaro fatalities" as similar to a known bad
"death rate" term even on its first few clicks.

Uses ``sentence-transformers`` when the model is available; otherwise falls back to a
TF-IDF + cosine-similarity scheme (scikit-learn) so the module always runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# Default seed list of known-bad intents for an adventure-travel advertiser.
DEFAULT_BAD_TERMS = [
    "travel agent salary",
    "free travel jobs",
    "is it dangerous",
    "death rate statistics",
    "cheap flights only",
    "how to become a guide",
    "trekking gear discount code",
    "travel insurance complaints",
]


@dataclass
class SemanticConfig:
    threshold: float = 0.55
    model_name: str = "all-MiniLM-L6-v2"


def _try_sentence_transformer(model_name: str):
    """Return a loaded SentenceTransformer, or ``None`` if unavailable/offline."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        return SentenceTransformer(model_name)
    except Exception:
        return None


def _st_similarity(terms: list[str], bad_terms: list[str], model) -> np.ndarray:
    """Max cosine similarity of each term to any bad term, via embeddings."""
    emb_terms = model.encode(terms, normalize_embeddings=True)
    emb_bad = model.encode(bad_terms, normalize_embeddings=True)
    sims = np.asarray(emb_terms) @ np.asarray(emb_bad).T
    return sims.max(axis=1)


def _tfidf_similarity(terms: list[str], bad_terms: list[str]) -> np.ndarray:
    """TF-IDF cosine fallback: max similarity of each term to any bad term."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    matrix = vec.fit_transform(terms + bad_terms)
    term_m = matrix[: len(terms)]
    bad_m = matrix[len(terms) :]
    sims = cosine_similarity(term_m, bad_m)
    if sims.shape[1] == 0:
        return np.zeros(len(terms))
    return sims.max(axis=1)


def flag_semantic_negatives(
    df: pd.DataFrame,
    bad_terms: list[str] | None = None,
    cfg: SemanticConfig | None = None,
) -> pd.DataFrame:
    """Add ``semantic_score``, ``semantic_method`` and ``semantic_flag`` columns.

    ``semantic_flag`` is True when a term's max similarity to any known-bad term
    exceeds ``cfg.threshold`` - an early-warning negative candidate.
    """
    cfg = cfg or SemanticConfig()
    bad_terms = bad_terms or DEFAULT_BAD_TERMS
    terms = [str(t) for t in df["search_term"].tolist()]

    model = _try_sentence_transformer(cfg.model_name)
    if model is not None:
        scores = _st_similarity(terms, bad_terms, model)
        method = "sentence-transformers"
    else:
        scores = _tfidf_similarity(terms, bad_terms)
        method = "tfidf-fallback"

    out = df.copy()
    out["semantic_score"] = scores
    out["semantic_method"] = method
    out["semantic_flag"] = out["semantic_score"] >= cfg.threshold
    return out
