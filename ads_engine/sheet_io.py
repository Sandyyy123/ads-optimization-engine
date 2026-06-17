"""Google Sheet staging I/O for the human-in-the-loop approval workflow.

The Sheet is the control surface: the engine writes recommendations with STATUS=PENDING,
a human reviews and flips rows to APPROVED, then the write-back step reads APPROVED rows
and marks them EXECUTED with a timestamp.

In demo mode (no Google credentials) the staging sheet is a local CSV and EXECUTED marks
are written back to that CSV, so the full loop is exercised offline.
"""

from __future__ import annotations

import csv
import datetime as _dt
import os
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

STATUS_PENDING = "PENDING"
STATUS_APPROVED = "APPROVED"
STATUS_EXECUTED = "EXECUTED"

# Columns written to the staging sheet, in order.
STAGING_COLUMNS = [
    "row_id",
    "search_term",
    "campaign_name",
    "ad_group_name",
    "recommendation",
    "reason",
    "clicks",
    "cost",
    "conversions",
    "cvr",
    "shrunk_cvr",
    "p_value",
    "semantic_flag",
    "status",
    "executed_at",
]


@dataclass
class SheetConfig:
    """Where the staging sheet lives."""

    spreadsheet_id: str = ""
    worksheet: str = "staging"
    sa_keyfile: str = field(default_factory=lambda: os.environ.get("GSPREAD_SA_KEYFILE", ""))
    demo_csv: str = "staging_sheet.csv"


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _open_worksheet(cfg: SheetConfig):
    """Open the gspread worksheet (lazy import; real-mode only)."""
    import gspread  # type: ignore
    from google.oauth2.service_account import Credentials  # type: ignore

    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_file(cfg.sa_keyfile, scopes=scopes)
    client = gspread.authorize(creds)
    sh = client.open_by_key(cfg.spreadsheet_id)
    try:
        return sh.worksheet(cfg.worksheet)
    except Exception:
        return sh.add_worksheet(title=cfg.worksheet, rows=1000, cols=len(STAGING_COLUMNS))


def _staging_frame(recs: pd.DataFrame) -> pd.DataFrame:
    """Project a recommendations DataFrame into the staging-sheet schema."""
    out = pd.DataFrame()
    out["row_id"] = [f"R{i:04d}" for i in range(1, len(recs) + 1)]
    out["search_term"] = recs.get("search_term")
    out["campaign_name"] = recs.get("campaign_name")
    out["ad_group_name"] = recs.get("ad_group_name")
    out["recommendation"] = recs.get("recommendation")
    out["reason"] = recs.get("reason")
    out["clicks"] = recs.get("clicks")
    out["cost"] = recs.get("cost")
    out["conversions"] = recs.get("conversions")
    out["cvr"] = recs.get("cvr").round(4) if "cvr" in recs else None
    out["shrunk_cvr"] = recs.get("shrunk_cvr").round(4) if "shrunk_cvr" in recs else None
    out["p_value"] = recs.get("p_value").round(4) if "p_value" in recs else None
    out["semantic_flag"] = recs.get("semantic_flag")
    out["status"] = STATUS_PENDING
    out["executed_at"] = ""
    return out[STAGING_COLUMNS]


def write_recommendations(
    recs: pd.DataFrame, cfg: SheetConfig | None = None, demo: bool = False
) -> pd.DataFrame:
    """Write recommendations to the staging sheet with STATUS=PENDING.

    Returns the staging frame that was written.
    """
    cfg = cfg or SheetConfig()
    staging = _staging_frame(recs)

    if demo:
        staging.to_csv(cfg.demo_csv, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"[sheet_io] wrote {len(staging)} PENDING rows -> {cfg.demo_csv}")
        return staging

    ws = _open_worksheet(cfg)
    ws.clear()
    ws.update([STAGING_COLUMNS] + staging.astype(object).fillna("").values.tolist())
    print(f"[sheet_io] wrote {len(staging)} PENDING rows -> sheet {cfg.spreadsheet_id}/{cfg.worksheet}")
    return staging


def read_staging(cfg: SheetConfig | None = None, demo: bool = False) -> pd.DataFrame:
    """Read the full staging sheet back into a DataFrame."""
    cfg = cfg or SheetConfig()
    if demo:
        if not os.path.exists(cfg.demo_csv):
            return pd.DataFrame(columns=STAGING_COLUMNS)
        return pd.read_csv(cfg.demo_csv)
    ws = _open_worksheet(cfg)
    records = ws.get_all_records()
    return pd.DataFrame(records, columns=STAGING_COLUMNS) if records else pd.DataFrame(columns=STAGING_COLUMNS)


def read_approved(cfg: SheetConfig | None = None, demo: bool = False) -> pd.DataFrame:
    """Read only the rows a human has flipped to APPROVED."""
    df = read_staging(cfg, demo=demo)
    if df.empty:
        return df
    return df[df["status"].astype(str).str.upper() == STATUS_APPROVED].reset_index(drop=True)


def simulate_approval(
    cfg: SheetConfig | None = None, demo: bool = False, recommendations: Optional[list[str]] = None
) -> int:
    """Demo helper: flip PENDING rows whose recommendation is actionable to APPROVED.

    Mimics the human reviewer approving PRUNE / NEGATIVE / PROMOTE rows. Returns the
    number of rows approved. Only meaningful in demo mode.
    """
    cfg = cfg or SheetConfig()
    recommendations = recommendations or ["PRUNE", "NEGATIVE", "PROMOTE"]
    df = read_staging(cfg, demo=demo)
    if df.empty:
        return 0
    mask = (df["status"].astype(str).str.upper() == STATUS_PENDING) & (
        df["recommendation"].isin(recommendations)
    )
    df.loc[mask, "status"] = STATUS_APPROVED
    if demo:
        df.to_csv(cfg.demo_csv, index=False)
    else:
        ws = _open_worksheet(cfg)
        ws.clear()
        ws.update([STAGING_COLUMNS] + df.astype(object).fillna("").values.tolist())
    print(f"[sheet_io] simulated approval of {int(mask.sum())} rows")
    return int(mask.sum())


def mark_executed(
    row_ids: list[str], cfg: SheetConfig | None = None, demo: bool = False
) -> int:
    """Mark the given row_ids EXECUTED with a UTC timestamp.

    Returns the number of rows marked. The change-log append lives in writeback.py.
    """
    cfg = cfg or SheetConfig()
    df = read_staging(cfg, demo=demo)
    if df.empty:
        return 0
    ts = _now_iso()
    # Ensure the timestamp column is string-typed; when read back from CSV an all-empty
    # column comes in as float64 and a string assignment would raise/warn.
    df["executed_at"] = df["executed_at"].astype("object")
    mask = df["row_id"].isin(row_ids)
    df.loc[mask, "status"] = STATUS_EXECUTED
    df.loc[mask, "executed_at"] = ts
    if demo:
        df.to_csv(cfg.demo_csv, index=False)
    else:
        ws = _open_worksheet(cfg)
        ws.clear()
        ws.update([STAGING_COLUMNS] + df.astype(object).fillna("").values.tolist())
    print(f"[sheet_io] marked {int(mask.sum())} rows EXECUTED at {ts}")
    return int(mask.sum())
