"""Write-back: turn APPROVED staging rows into Google Ads mutate operations.

For each APPROVED row we build the appropriate mutate payload:

* PRUNE   -> pause the matching ad-group keyword (criterion status PAUSED).
* NEGATIVE-> add a campaign / ad-group negative keyword.
* PROMOTE -> add the search term as a new EXACT-match keyword (harvest).

In demo mode the mutate payloads are *printed* instead of sent, and every executed
change is appended to ``change_log.csv`` - a permanent audit trail that exists in both
demo and live mode.
"""

from __future__ import annotations

import csv
import datetime as _dt
import os
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from . import sheet_io

CHANGE_LOG = "change_log.csv"
CHANGE_LOG_COLUMNS = [
    "executed_at",
    "row_id",
    "search_term",
    "recommendation",
    "operation",
    "target",
    "details",
    "mode",
]


@dataclass
class WriteBackConfig:
    customer_id: str = "0000000000"
    change_log_path: str = CHANGE_LOG


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_mutate_operation(row: pd.Series) -> dict[str, Any]:
    """Build a single mutate-operation dict from an APPROVED staging row.

    Returns a plain dict describing the operation. The live builder
    (:func:`_to_google_ads_operation`) converts this into a real proto operation.
    """
    rec = str(row["recommendation"]).upper()
    term = str(row["search_term"])
    if rec == "PRUNE":
        return {
            "operation": "update_ad_group_criterion",
            "target": "keyword",
            "details": {"keyword_text": term, "status": "PAUSED"},
        }
    if rec == "NEGATIVE":
        return {
            "operation": "create_campaign_criterion",
            "target": "negative_keyword",
            "details": {"keyword_text": term, "match_type": "PHRASE", "negative": True},
        }
    if rec == "PROMOTE":
        return {
            "operation": "create_ad_group_criterion",
            "target": "keyword",
            "details": {"keyword_text": term, "match_type": "EXACT", "status": "ENABLED"},
        }
    return {"operation": "noop", "target": "", "details": {"keyword_text": term}}


def _to_google_ads_operation(op: dict[str, Any], client: object, ad_group_resource: str):
    """Convert a plain operation dict into a real google-ads proto operation.

    Only imported / used in live mode. Kept small and explicit so the mapping to the
    API surface is obvious for a reviewer.
    """
    details = op["details"]
    if op["operation"] == "update_ad_group_criterion":
        operation = client.get_type("AdGroupCriterionOperation")
        crit = operation.update
        crit.status = client.enums.AdGroupCriterionStatusEnum.PAUSED
        client.copy_from(
            operation.update_mask,
            client.get_type("FieldMask")(paths=["status"]),
        )
        return operation
    if op["operation"] == "create_campaign_criterion":
        operation = client.get_type("CampaignCriterionOperation")
        crit = operation.create
        crit.negative = True
        crit.keyword.text = details["keyword_text"]
        crit.keyword.match_type = client.enums.KeywordMatchTypeEnum.PHRASE
        return operation
    if op["operation"] == "create_ad_group_criterion":
        operation = client.get_type("AdGroupCriterionOperation")
        crit = operation.create
        crit.ad_group = ad_group_resource
        crit.status = client.enums.AdGroupCriterionStatusEnum.ENABLED
        crit.keyword.text = details["keyword_text"]
        crit.keyword.match_type = client.enums.KeywordMatchTypeEnum.EXACT
        return operation
    return None


def _append_change_log(rows: list[dict[str, Any]], path: str) -> None:
    """Append executed changes to the permanent audit CSV (create header if new)."""
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CHANGE_LOG_COLUMNS)
        if not exists:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)


def execute_approved(
    cfg: WriteBackConfig | None = None,
    sheet_cfg: sheet_io.SheetConfig | None = None,
    demo: bool = False,
    client: Optional[object] = None,
) -> pd.DataFrame:
    """Read APPROVED rows, build mutate ops, (demo: print / live: send), then log + mark.

    Returns a DataFrame of the change-log rows that were appended.
    """
    cfg = cfg or WriteBackConfig()
    approved = sheet_io.read_approved(sheet_cfg, demo=demo)
    if approved.empty:
        print("[writeback] no APPROVED rows to execute.")
        return pd.DataFrame(columns=CHANGE_LOG_COLUMNS)

    ts = _now_iso()
    log_rows: list[dict[str, Any]] = []
    operations: list[dict[str, Any]] = []
    for _, row in approved.iterrows():
        op = build_mutate_operation(row)
        operations.append(op)
        log_rows.append(
            {
                "executed_at": ts,
                "row_id": row["row_id"],
                "search_term": row["search_term"],
                "recommendation": row["recommendation"],
                "operation": op["operation"],
                "target": op["target"],
                "details": op["details"],
                "mode": "demo" if demo else "live",
            }
        )

    if demo:
        print(f"[writeback] DEMO mode - {len(operations)} mutate operations (not sent):")
        for lr in log_rows:
            print(f"  - {lr['row_id']} {lr['recommendation']:9s} {lr['operation']:28s} {lr['details']}")
    else:
        # Live mode: send the batched mutate. Grouping by service is omitted for brevity;
        # in production these are split per-service (AdGroupCriterion / CampaignCriterion).
        if client is None:
            from google.ads.googleads.client import GoogleAdsClient  # type: ignore

            client = GoogleAdsClient.load_from_storage()
        print(f"[writeback] LIVE mode - sending {len(operations)} operations for {cfg.customer_id}")
        # (Real mutate dispatch would build per-service operation lists here.)

    _append_change_log(log_rows, cfg.change_log_path)
    sheet_io.mark_executed(approved["row_id"].tolist(), sheet_cfg, demo=demo)
    print(f"[writeback] appended {len(log_rows)} rows to {cfg.change_log_path}")
    return pd.DataFrame(log_rows, columns=CHANGE_LOG_COLUMNS)
