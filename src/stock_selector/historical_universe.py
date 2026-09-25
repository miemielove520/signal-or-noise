"""Point-in-time universe membership — the mechanism for survivorship-bias-free
backtests.

A normal backtest evaluates *today's* index/universe members over history, which
silently inflates results (the losers that got delisted are missing). The honest
fix needs to know, for each historical date, which tickers were actually members
*then*. This module loads that membership and answers "who was in as of date X",
plus a status flag so any output can state whether it is survivorship-safe.

The MECHANISM is here and tested; the real historical-constituent DATA
(``data/historical_universe.csv``) must be sourced separately — until it exists,
``survivorship_status`` returns ``not_handled`` so the bias is surfaced, not hidden.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


MEMBERSHIP_COLUMNS = ("ticker", "start_date", "end_date")
DEFAULT_MEMBERSHIP_PATH = "data/historical_universe.csv"


def load_historical_membership(path: str | Path) -> pd.DataFrame:
    """Load a membership file: one row per (ticker, membership interval).

    Columns: ``ticker, start_date, end_date``. ``start_date`` = when it entered the
    universe; ``end_date`` = when it left (blank/empty = still a member). Returns an
    empty frame (never raises) when the file is missing or malformed."""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=list(MEMBERSHIP_COLUMNS))
    try:
        frame = pd.read_csv(p, comment="#")
    except Exception:
        return pd.DataFrame(columns=list(MEMBERSHIP_COLUMNS))
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    if "ticker" not in frame.columns or "start_date" not in frame.columns:
        return pd.DataFrame(columns=list(MEMBERSHIP_COLUMNS))
    if "end_date" not in frame.columns:
        frame["end_date"] = pd.NaT
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    frame["start_date"] = pd.to_datetime(frame["start_date"], errors="coerce")
    frame["end_date"] = pd.to_datetime(frame["end_date"], errors="coerce")
    frame = frame.dropna(subset=["ticker", "start_date"])
    return frame[list(MEMBERSHIP_COLUMNS)].reset_index(drop=True)


def members_as_of(membership: pd.DataFrame, as_of_date: str | pd.Timestamp) -> list[str]:
    """Tickers that were universe members on ``as_of_date`` — the set a
    survivorship-free backtest may consider at that historical point."""
    if membership is None or membership.empty:
        return []
    as_of = pd.Timestamp(as_of_date).normalize()
    started = membership["start_date"] <= as_of
    not_ended = membership["end_date"].isna() | (membership["end_date"] >= as_of)
    members = membership.loc[started & not_ended, "ticker"]
    return sorted(dict.fromkeys(members.tolist()))


def filter_to_members_as_of(
    tickers: list[str], membership: pd.DataFrame, as_of_date: str | pd.Timestamp
) -> list[str]:
    """Keep only the tickers that were members as of the date (order preserved).
    If no membership data exists, returns the input unchanged (cannot filter)."""
    if membership is None or membership.empty:
        return list(tickers)
    allowed = set(members_as_of(membership, as_of_date))
    return [t for t in tickers if str(t).upper().strip() in allowed]


def survivorship_status(membership: pd.DataFrame | None) -> dict[str, object]:
    """Honest flag for backtest/scan output: is the run survivorship-safe?"""
    if membership is None or membership.empty:
        return {
            "handled": False,
            "status": "not_handled",
            "status_zh": "未做幸存者偏差处理",
            "note": "No historical membership data — backtest uses today's survivors and is optimistically biased.",
            "note_zh": "没有历史成分股数据——回测用的是今天还活着的股票，结果偏乐观，请打折扣看。",
        }
    return {
        "handled": True,
        "status": "point_in_time",
        "status_zh": "已按时点成分股处理",
        "note": f"Point-in-time membership loaded ({membership['ticker'].nunique()} tickers).",
        "note_zh": f"已加载时点成分股（{membership['ticker'].nunique()}只），回测无幸存者偏差。",
    }
