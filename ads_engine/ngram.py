"""N-gram performance analysis over search terms.

Tokenises every search term into 1/2/3-grams, aggregates clicks/cost/conversions per
n-gram, ranks them, and flags:

* harvesting candidates - high-converting n-grams that are NOT yet exact-match keywords.
* negative candidates   - high-cost, zero-conversion n-grams.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Common stop words that make noisy, meaningless n-grams.
_STOP = {
    "the", "a", "an", "to", "of", "in", "on", "for", "and", "or", "is", "are",
    "my", "me", "i", "you", "with", "best", "near",
}


def tokenize(term: str) -> list[str]:
    """Lower-case word tokeniser, stop-words removed."""
    return [t for t in _TOKEN_RE.findall(term.lower()) if t not in _STOP]


def ngrams(tokens: list[str], n: int) -> list[str]:
    """Return all contiguous n-grams of length ``n`` as space-joined strings."""
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


@dataclass
class NgramConfig:
    """Thresholds for harvesting / negative flagging."""

    max_n: int = 3
    min_clicks: int = 10
    harvest_min_conversions: float = 2.0
    harvest_min_cvr: float = 0.05
    negative_min_cost: float = 30.0


def build_ngram_table(df: pd.DataFrame, cfg: NgramConfig | None = None) -> pd.DataFrame:
    """Aggregate clicks/cost/conversions for every 1..max_n gram across all terms."""
    cfg = cfg or NgramConfig()
    rows: list[dict] = []
    for _, r in df.iterrows():
        toks = tokenize(str(r["search_term"]))
        seen: set[str] = set()
        for n in range(1, cfg.max_n + 1):
            for g in ngrams(toks, n):
                if g in seen:  # don't double-count a repeated gram within one term
                    continue
                seen.add(g)
                rows.append(
                    {
                        "ngram": g,
                        "n": n,
                        "clicks": r["clicks"],
                        "cost": r["cost"],
                        "conversions": r["conversions"],
                    }
                )
    if not rows:
        return pd.DataFrame(
            columns=["ngram", "n", "clicks", "cost", "conversions", "cvr", "cpa", "n_terms"]
        )

    long = pd.DataFrame(rows)
    agg = (
        long.groupby(["ngram", "n"], as_index=False)
        .agg(
            clicks=("clicks", "sum"),
            cost=("cost", "sum"),
            conversions=("conversions", "sum"),
            n_terms=("ngram", "size"),
        )
    )
    agg["cvr"] = agg.apply(lambda x: x["conversions"] / x["clicks"] if x["clicks"] else 0.0, axis=1)
    agg["cpa"] = agg.apply(
        lambda x: x["cost"] / x["conversions"] if x["conversions"] else float("inf"), axis=1
    )
    return agg.sort_values(["conversions", "clicks"], ascending=False).reset_index(drop=True)


def _existing_exact_keywords(df: pd.DataFrame) -> set[str]:
    """Set of search terms already running as EXACT match (lower-cased)."""
    if "match_type" not in df.columns:
        return set()
    exact = df[df["match_type"].astype(str).str.upper() == "EXACT"]
    return {str(t).lower() for t in exact["search_term"]}


def harvest_candidates(
    df: pd.DataFrame, ngram_table: pd.DataFrame, cfg: NgramConfig | None = None
) -> pd.DataFrame:
    """High-converting n-grams not yet captured as exact-match keywords."""
    cfg = cfg or NgramConfig()
    existing = _existing_exact_keywords(df)
    cand = ngram_table[
        (ngram_table["clicks"] >= cfg.min_clicks)
        & (ngram_table["conversions"] >= cfg.harvest_min_conversions)
        & (ngram_table["cvr"] >= cfg.harvest_min_cvr)
    ].copy()
    cand = cand[~cand["ngram"].isin(existing)]
    return cand.sort_values(["conversions", "cvr"], ascending=False).reset_index(drop=True)


def negative_candidates(ngram_table: pd.DataFrame, cfg: NgramConfig | None = None) -> pd.DataFrame:
    """High-cost, zero-conversion n-grams - candidates for campaign negatives."""
    cfg = cfg or NgramConfig()
    cand = ngram_table[
        (ngram_table["conversions"] == 0)
        & (ngram_table["cost"] >= cfg.negative_min_cost)
        & (ngram_table["clicks"] >= cfg.min_clicks)
    ].copy()
    return cand.sort_values("cost", ascending=False).reset_index(drop=True)
