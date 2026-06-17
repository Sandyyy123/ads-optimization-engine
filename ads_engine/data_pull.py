"""Pull keyword / search-term / conversion data from the Google Ads API via GAQL.

The real puller uses ``google.ads.googleads.client.GoogleAdsClient`` and is keyed on a
configurable *offline* conversion action (e.g. ``deposit_paid``) so that the engine
optimises toward a meaningful business event rather than raw "conversions".

A ``--demo`` mode generates a synthetic, deterministic search-terms DataFrame so the
whole pipeline runs end-to-end with no credentials present.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# GAQL query building
# --------------------------------------------------------------------------- #
@dataclass
class PullConfig:
    """Configuration for a search-term performance pull."""

    customer_id: str = "0000000000"
    conversion_action: str = "deposit_paid"
    date_range: str = "LAST_30_DAYS"
    # Only pull terms with at least this many impressions to keep the report sane.
    min_impressions: int = 1
    extra_segments: list[str] = field(default_factory=list)


def build_search_term_gaql(cfg: PullConfig) -> str:
    """Build the GAQL query for the search-term view.

    The conversion action is filtered in the metrics segment so that
    ``metrics.conversions`` reflects only the chosen offline action
    (``cfg.conversion_action``), not every configured conversion.
    """
    select_fields = [
        "search_term_view.search_term",
        "search_term_view.status",
        "segments.keyword.info.match_type",
        "campaign.id",
        "campaign.name",
        "ad_group.id",
        "ad_group.name",
        "metrics.impressions",
        "metrics.clicks",
        "metrics.cost_micros",
        "metrics.conversions",
        "metrics.conversions_value",
    ] + list(cfg.extra_segments)

    query = (
        "SELECT "
        + ", ".join(select_fields)
        + " FROM search_term_view"
        + f" WHERE segments.date DURING {cfg.date_range}"
        + f" AND metrics.impressions >= {cfg.min_impressions}"
        + f" AND segments.conversion_action_name = '{cfg.conversion_action}'"
    )
    return query


def build_keyword_gaql(cfg: PullConfig) -> str:
    """Build the GAQL query for current keyword status (used by write-back)."""
    return (
        "SELECT ad_group_criterion.criterion_id, ad_group_criterion.keyword.text, "
        "ad_group_criterion.keyword.match_type, ad_group_criterion.status, "
        "ad_group.id, campaign.id, metrics.conversions "
        "FROM keyword_view "
        f"WHERE segments.date DURING {cfg.date_range} "
        f"AND segments.conversion_action_name = '{cfg.conversion_action}'"
    )


# --------------------------------------------------------------------------- #
# Real puller
# --------------------------------------------------------------------------- #
def pull_search_terms(cfg: PullConfig, client: Optional[object] = None) -> pd.DataFrame:
    """Execute the search-term GAQL query against the Google Ads API.

    Parameters
    ----------
    cfg:
        Pull configuration (customer id, conversion action, date range).
    client:
        An initialised ``GoogleAdsClient``. If ``None`` we attempt to load one from the
        environment / ``google-ads.yaml`` via ``GoogleAdsClient.load_from_storage``.

    Returns
    -------
    DataFrame with one row per search term and the standard metric columns.
    """
    if client is None:
        # Imported lazily so the module imports cleanly without the dependency.
        from google.ads.googleads.client import GoogleAdsClient  # type: ignore

        client = GoogleAdsClient.load_from_storage()

    ga_service = client.get_service("GoogleAdsService")
    query = build_search_term_gaql(cfg)

    rows: list[dict] = []
    stream = ga_service.search_stream(customer_id=cfg.customer_id, query=query)
    for batch in stream:
        for row in batch.results:
            rows.append(
                {
                    "search_term": row.search_term_view.search_term,
                    "match_type": row.segments.keyword.info.match_type.name,
                    "campaign_id": row.campaign.id,
                    "campaign_name": row.campaign.name,
                    "ad_group_id": row.ad_group.id,
                    "ad_group_name": row.ad_group.name,
                    "impressions": row.metrics.impressions,
                    "clicks": row.metrics.clicks,
                    "cost": row.metrics.cost_micros / 1e6,
                    "conversions": row.metrics.conversions,
                    "conversions_value": row.metrics.conversions_value,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Demo / synthetic data
# --------------------------------------------------------------------------- #
def make_demo_search_terms(n_terms: int = 60, seed: int = 7) -> pd.DataFrame:
    """Generate a deterministic synthetic search-terms DataFrame.

    The synthetic data is built so that the downstream statistics have something
    real to find: a few clear winners, a few clear losers, and a long tail of
    low-volume terms where the Empirical-Bayes shrinkage matters.
    """
    rng = np.random.default_rng(seed)

    head = [
        "guided patagonia trek", "patagonia hiking tour", "kilimanjaro climb package",
        "amazon river expedition", "everest base camp trek", "antarctica cruise booking",
        "morocco desert tour", "iceland northern lights trip", "galapagos wildlife cruise",
        "safari tanzania luxury",
    ]
    bad = [
        "free travel jobs", "cheap flights only", "travel agent salary",
        "is patagonia trek dangerous", "kilimanjaro death rate", "trekking gear discount code",
    ]
    fillers = ["tour", "trip", "package", "deal", "guide", "cost", "review", "map", "weather", "best time"]
    dests = ["patagonia", "kilimanjaro", "amazon", "everest", "iceland", "morocco", "galapagos", "nepal", "peru", "chile"]

    terms: list[str] = []
    terms += head + bad
    while len(terms) < n_terms:
        terms.append(f"{rng.choice(dests)} {rng.choice(fillers)}")
    terms = terms[:n_terms]

    campaigns = ["Adventure-Search-BR", "Adventure-Search-NB"]
    ad_groups = ["AG-Trek", "AG-Cruise", "AG-Safari"]

    records = []
    for t in terms:
        is_head = t in head
        is_bad = t in bad
        impressions = int(rng.integers(400, 4000)) if is_head else int(rng.integers(5, 600))
        ctr = rng.uniform(0.04, 0.12) if is_head else rng.uniform(0.005, 0.06)
        clicks = max(0, int(impressions * ctr))
        # cost per click higher for competitive head terms
        cpc = rng.uniform(1.5, 4.5) if is_head else rng.uniform(0.4, 2.0)
        cost = round(clicks * cpc, 2)
        if is_bad:
            cvr = 0.0  # known losers: spend, no deposits
        elif is_head:
            cvr = rng.uniform(0.06, 0.14)
        else:
            cvr = rng.uniform(0.0, 0.05)
        conversions = float(np.round(rng.binomial(clicks, cvr) if clicks > 0 else 0, 0))
        records.append(
            {
                "search_term": t,
                "match_type": rng.choice(["BROAD", "PHRASE", "EXACT"], p=[0.6, 0.3, 0.1]),
                "campaign_id": rng.integers(10_000, 99_999),
                "campaign_name": rng.choice(campaigns),
                "ad_group_id": rng.integers(100_000, 999_999),
                "ad_group_name": rng.choice(ad_groups),
                "impressions": impressions,
                "clicks": clicks,
                "cost": cost,
                "conversions": conversions,
                "conversions_value": round(conversions * rng.uniform(800, 2500), 2),
            }
        )
    df = pd.DataFrame(records)
    return df


def get_search_terms(cfg: PullConfig, demo: bool = False) -> pd.DataFrame:
    """Front door: synthetic data in demo mode, real API pull otherwise."""
    if demo:
        return make_demo_search_terms()
    return pull_search_terms(cfg)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Pull Google Ads search-term data via GAQL.")
    parser.add_argument("--demo", action="store_true", help="Generate synthetic data (no credentials needed).")
    parser.add_argument("--conversion-action", default="deposit_paid")
    parser.add_argument("--customer-id", default="0000000000")
    parser.add_argument("--show-query", action="store_true", help="Print the GAQL query and exit.")
    args = parser.parse_args()

    cfg = PullConfig(customer_id=args.customer_id, conversion_action=args.conversion_action)
    if args.show_query:
        print(build_search_term_gaql(cfg))
        return

    df = get_search_terms(cfg, demo=args.demo)
    print(f"Pulled {len(df)} search terms (demo={args.demo}).")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    _cli()
