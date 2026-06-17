# Ads Optimization Engine

A statistical bid / keyword optimization engine for Google Ads, designed to be run
**headless** and controlled entirely from a **Google Sheet**. It pulls performance data
via GAQL, runs honest statistics (two-proportion tests, Wilson intervals,
Empirical-Bayes shrinkage), mines search terms with n-gram and semantic analysis, stages
recommendations for human approval in a Sheet, and writes approved changes back to the
Google Ads account with a permanent audit trail.

> **This is a demonstration repository with synthetic data.** Run `python main.py --demo`
> to exercise the full pipeline end-to-end with no credentials. The Google Ads API and
> gspread integration points are real (the code builds GAQL queries and mutate
> operations), but the demo path substitutes a deterministic synthetic dataset and prints
> the mutate payloads instead of sending them.

## Why this design

The classic failure mode of automated bid/keyword tools is **acting on noise**: pausing a
keyword after a 0-for-4 click run, or promoting a 1-for-3 fluke. This engine is built
around three guardrails:

1. **Two-proportion z-test** - is a search term's conversion rate *statistically*
   different from its ad group, or is it just variance?
2. **Wilson confidence intervals** - honest uncertainty for low-volume terms (the normal
   approximation lies when n is small or the rate is near 0).
3. **Empirical-Bayes (Beta-Binomial) shrinkage** - small-n terms are pulled toward the
   group prior, so a 1/3 term is not mistaken for a 33% converter.

A human stays in the loop: nothing changes the account until a person flips a row to
`APPROVED` in the Sheet.

## Architecture

```
                  ┌──────────────────────────────────────────────┐
                  │              Google Ads account               │
                  └───────────────┬───────────────▲───────────────┘
                       GAQL pull   │               │  mutate ops
                                   ▼               │  (pause / negative / harvest)
          ┌────────────────────────────────────────┴─────────────┐
          │                   ads_engine                          │
          │                                                       │
          │  data_pull.py   GAQL builder + GoogleAdsClient puller │
          │       │         (--demo -> synthetic search terms)    │
          │       ▼                                               │
          │  stats.py       z-test · Wilson CI · EB shrinkage ·   │
          │       │         decision fn -> PRUNE/PROMOTE/NEGATIVE/│
          │       │         HOLD-INCUBATING/LOW-VOLUME            │
          │       ├──► ngram.py    1/2/3-gram perf, harvest +     │
          │       │               negative candidates             │
          │       ├──► semantic.py sentence-transformers (TF-IDF  │
          │       │               fallback) early-warning flags   │
          │       ▼                                               │
          │  sheet_io.py    write PENDING ─► read APPROVED ─►      │
          │       │         mark EXECUTED (+ timestamp)            │
          │       ▼                                               │
          │  writeback.py   APPROVED -> mutate ops, append         │
          │                 change_log.csv (permanent audit)      │
          └───────────────────────────────────────────────────────┘
                                   ▲   │
                          APPROVE  │   │  recommendations (PENDING)
                                   │   ▼
                  ┌──────────────────────────────────────────────┐
                  │   Google Sheet  (the headless control panel)  │
                  │   STATUS column: PENDING ─► APPROVED ─► EXECUTED│
                  └──────────────────────────────────────────────┘
```

## The Sheet workflow: PENDING -> APPROVED -> EXECUTED

1. The engine writes one row per recommendation to the staging worksheet with
   `STATUS = PENDING` (search term, recommendation, reason, stats, semantic flag).
2. A human reviews the rows in the Sheet and changes `STATUS` to `APPROVED` for the
   changes they accept. Everything else stays `PENDING` and is never touched.
3. The write-back step reads only `APPROVED` rows, builds the Google Ads mutate
   operations, applies them, then sets `STATUS = EXECUTED` with a UTC timestamp and
   appends every change to `change_log.csv`.

This means the account is only ever changed by an explicit human decision, and every
change is auditable forever.

## Recommendation tags

| Tag               | Meaning                                                              |
|-------------------|----------------------------------------------------------------------|
| `PROMOTE`         | Significantly above group CVR and shrunk rate beats the mean -> harvest as exact-match keyword. |
| `PRUNE`           | Significantly below group CVR with enough volume -> pause / lower bid. |
| `NEGATIVE`        | Meaningful spend, zero conversions, statistically below group -> add as negative. |
| `HOLD-INCUBATING` | Spending but not yet enough signal -> keep watching.                 |
| `LOW-VOLUME`      | Too few clicks to act -> shrink toward prior, no action.             |

## Setup (live mode)

```bash
pip install -r requirements.txt
```

1. **Google Ads API**: create a `google-ads.yaml` (developer token, OAuth client,
   refresh token, login-customer-id). `data_pull.py` loads it via
   `GoogleAdsClient.load_from_storage()`.
2. **Google Sheet**: create a service account, share the staging Sheet with it, and set
   `GSPREAD_SA_KEYFILE` to the service-account JSON path. Put the spreadsheet id in
   `SheetConfig`.
3. Choose your **offline conversion action** (e.g. `deposit_paid` for a deposit-based
   travel funnel) - the engine optimises toward that action, not raw conversions.

None of these secrets are committed; `.gitignore` blocks `.env`, `google-ads.yaml`, and
all service-account / token files.

## Run the demo

```bash
python main.py --demo
```

This runs the entire pipeline on synthetic adventure-travel search terms and prints a
readable summary. Vary the confidence level to see the decision function act more or less
aggressively:

```bash
python main.py --demo --confidence 0.80     # act on weaker evidence
python main.py --demo --confidence 0.95      # conservative (default)
```

Inspect a single GAQL query without running anything:

```bash
python -m ads_engine.data_pull --show-query
python -m ads_engine.data_pull --demo        # print synthetic data
```

Demo artifacts written to the working directory:

- `staging_sheet.csv` - the staging "Sheet" (PENDING -> APPROVED -> EXECUTED).
- `change_log.csv` - the permanent audit trail of executed changes.

## Module map

| File                     | Responsibility                                                    |
|--------------------------|-------------------------------------------------------------------|
| `ads_engine/data_pull.py`| GAQL query builders + `GoogleAdsClient` puller; `--demo` synthetic data. |
| `ads_engine/stats.py`    | z-test, Wilson CI, Empirical-Bayes shrinkage, decision function.  |
| `ads_engine/ngram.py`    | 1/2/3-gram aggregation, harvesting + negative candidates.         |
| `ads_engine/semantic.py` | sentence-transformers / TF-IDF similarity to known-bad terms.     |
| `ads_engine/sheet_io.py` | gspread staging Sheet read/write; PENDING/APPROVED/EXECUTED.      |
| `ads_engine/writeback.py`| build mutate ops from APPROVED rows; append `change_log.csv`.     |
| `main.py`                | end-to-end runner (`--demo`).                                     |

## License

MIT.
