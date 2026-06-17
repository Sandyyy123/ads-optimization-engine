"""End-to-end runner for the Google Ads statistical optimization engine.

Demo:
    python main.py --demo

Flow:
    pull (synthetic) search terms
      -> statistical decisions (z-test, Wilson CI, Empirical-Bayes shrinkage)
      -> n-gram harvesting + negative candidates
      -> semantic early-warning flags
      -> write staged recommendations to the Sheet (PENDING)
      -> simulate human approval (PENDING -> APPROVED)
      -> write-back in demo mode (build mutate ops, append change_log.csv, mark EXECUTED)
      -> print a readable summary.
"""

from __future__ import annotations

import argparse

import pandas as pd

from ads_engine import ngram, semantic, sheet_io, stats, writeback
from ads_engine.data_pull import PullConfig, get_search_terms

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)


def _section(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def run(demo: bool, conversion_action: str, confidence: float) -> int:
    """Run the full pipeline. Returns process exit code."""
    _section(f"1. PULL search-term data (demo={demo}, conversion='{conversion_action}')")
    pull_cfg = PullConfig(conversion_action=conversion_action)
    df = get_search_terms(pull_cfg, demo=demo)
    print(f"Pulled {len(df)} search terms. Total clicks={int(df['clicks'].sum())}, "
          f"cost=€{df['cost'].sum():,.0f}, conversions={df['conversions'].sum():.0f}")

    _section(f"2. STATISTICS (z-test + Wilson CI + Empirical-Bayes shrinkage @ {confidence:.0%})")
    decided = stats.decide(df, stats.DecisionConfig(confidence=confidence))
    counts = decided["recommendation"].value_counts()
    print("Recommendation tally:")
    for rec, c in counts.items():
        print(f"  {rec:16s} {c}")
    print("\nGroup mean CVR: {:.2%}".format(decided["group_mean_cvr"].iloc[0]))
    show = decided[decided["recommendation"].isin(["PRUNE", "PROMOTE", "NEGATIVE"])]
    cols = ["search_term", "clicks", "cost", "conversions", "cvr", "shrunk_cvr",
            "p_value", "recommendation", "reason"]
    if not show.empty:
        print("\nActionable rows:")
        print(show[cols].to_string(index=False))

    _section("3. N-GRAM analysis (harvesting + negative candidates)")
    ng_table = ngram.build_ngram_table(df)
    harvest = ngram.harvest_candidates(df, ng_table)
    negs = ngram.negative_candidates(ng_table)
    print(f"Top n-grams by conversions:")
    print(ng_table.head(8)[["ngram", "n", "clicks", "cost", "conversions", "cvr"]].to_string(index=False))
    print(f"\nHarvesting candidates (high-converting, not yet exact): {len(harvest)}")
    if not harvest.empty:
        print(harvest.head(6)[["ngram", "clicks", "conversions", "cvr"]].to_string(index=False))
    print(f"\nNegative n-gram candidates (high-cost, zero-conv): {len(negs)}")
    if not negs.empty:
        print(negs.head(6)[["ngram", "clicks", "cost", "conversions"]].to_string(index=False))

    _section("4. SEMANTIC early-warning flags")
    sem = semantic.flag_semantic_negatives(decided)
    method = sem["semantic_method"].iloc[0]
    flagged = sem[sem["semantic_flag"]]
    print(f"Method: {method}. Terms flagged similar to known-bad: {len(flagged)}")
    if not flagged.empty:
        print(flagged.sort_values("semantic_score", ascending=False)
              .head(8)[["search_term", "semantic_score", "recommendation"]].to_string(index=False))
    # carry the semantic flag into the recommendations frame for staging.
    # sem is a column-wise superset of `decided` (same rows, same order), so assign the
    # column directly - a merge on search_term would fan out on duplicate terms.
    decided["semantic_flag"] = sem["semantic_flag"].to_numpy()

    _section("5. WRITE staged recommendations to the Sheet (STATUS=PENDING)")
    sheet_cfg = sheet_io.SheetConfig(demo_csv="staging_sheet.csv")
    staged = sheet_io.write_recommendations(decided, sheet_cfg, demo=demo)
    print(f"Staged {len(staged)} rows. Status breakdown: {staged['status'].value_counts().to_dict()}")

    _section("6. SIMULATE human approval (PENDING -> APPROVED)")
    n_appr = sheet_io.simulate_approval(sheet_cfg, demo=demo)
    print(f"Approved {n_appr} actionable rows (PRUNE / NEGATIVE / PROMOTE).")

    _section("7. WRITE-BACK (build mutate ops, append change_log.csv, mark EXECUTED)")
    wb_cfg = writeback.WriteBackConfig()
    log = writeback.execute_approved(wb_cfg, sheet_cfg, demo=demo)

    _section("SUMMARY")
    print(f"Search terms analysed : {len(df)}")
    print(f"Recommendations staged: {len(staged)}")
    print(f"Rows approved         : {n_appr}")
    print(f"Mutate ops executed   : {len(log)}")
    if not log.empty:
        print(f"Operations by type    : {log['operation'].value_counts().to_dict()}")
    print(f"Harvest candidates    : {len(harvest)}")
    print(f"Negative candidates   : {len(negs)}")
    print(f"Semantic warnings     : {len(flagged)}")
    print("\nAudit trail appended to change_log.csv; staging sheet -> staging_sheet.csv (demo).")
    print("Pipeline completed successfully.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Statistical Google Ads optimization engine.")
    parser.add_argument("--demo", action="store_true", help="Run with synthetic data, no credentials needed.")
    parser.add_argument("--conversion-action", default="deposit_paid",
                        help="Offline conversion action to optimise toward.")
    parser.add_argument("--confidence", type=float, default=0.95,
                        help="Confidence level for the decision function (e.g. 0.80 or 0.95).")
    args = parser.parse_args()

    if not args.demo:
        print("Running in LIVE mode requires Google Ads + gspread credentials. "
              "Use --demo to run with synthetic data.")
    raise SystemExit(run(args.demo, args.conversion_action, args.confidence))


if __name__ == "__main__":
    main()
