from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import REQUIRED_PRICE_COLUMNS


@dataclass(frozen=True)
class AuditIssue:
    severity: str
    code: str
    message: str
    row_count: int = 0
    ticker: str | None = None


@dataclass(frozen=True)
class AuditReport:
    summary: dict[str, object]
    issues: list[AuditIssue]

    @property
    def passed(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def issues_frame(self) -> pd.DataFrame:
        return pd.DataFrame([issue.__dict__ for issue in self.issues])

    def to_text(self) -> str:
        lines = ["Data audit summary"]
        for key, value in self.summary.items():
            lines.append(f"{key}: {value}")

        if not self.issues:
            lines.append("No issues found.")
            return "\n".join(lines)

        lines.append("Issues")
        for issue in self.issues:
            ticker = f" ticker={issue.ticker}" if issue.ticker else ""
            count = f" rows={issue.row_count}" if issue.row_count else ""
            lines.append(f"[{issue.severity}] {issue.code}:{ticker}{count} {issue.message}")
        return "\n".join(lines)


def audit_price_csv(path: str | Path) -> AuditReport:
    prices = pd.read_csv(path)
    return audit_price_data(prices)


def audit_price_data(prices: pd.DataFrame) -> AuditReport:
    frame = prices.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    issues: list[AuditIssue] = []

    missing_columns = REQUIRED_PRICE_COLUMNS.difference(frame.columns)
    extra_columns = set(frame.columns).difference(REQUIRED_PRICE_COLUMNS)
    summary: dict[str, object] = {
        "rows": len(frame),
        "columns": len(frame.columns),
        "missing_required_columns": ",".join(sorted(missing_columns)) or "none",
        "extra_columns": ",".join(sorted(extra_columns)) or "none",
    }

    if missing_columns:
        issues.append(
            AuditIssue(
                severity="error",
                code="missing_columns",
                message=", ".join(sorted(missing_columns)),
                row_count=len(frame),
            )
        )
        return AuditReport(summary=summary, issues=issues)

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=False)
    frame["ticker"] = frame["ticker"].astype("string").str.upper().str.strip()
    frame["ticker"] = frame["ticker"].replace("", pd.NA)
    numeric_columns = ["open", "high", "low", "close", "adj_close", "volume"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    summary.update(
        {
            "tickers": frame["ticker"].nunique(dropna=True),
            "start_date": _format_date(frame["date"].min()),
            "end_date": _format_date(frame["date"].max()),
        }
    )

    _add_missing_value_issues(frame, issues)
    _add_duplicate_issues(frame, issues)
    _add_price_sanity_issues(frame, issues)
    _add_return_outlier_issues(frame, issues)
    _add_gap_issues(frame, issues)
    _add_stale_price_issues(frame, issues)

    return AuditReport(summary=summary, issues=issues)


def _format_date(value: pd.Timestamp | float) -> str:
    if pd.isna(value):
        return "none"
    return pd.Timestamp(value).date().isoformat()


def _add_missing_value_issues(frame: pd.DataFrame, issues: list[AuditIssue]) -> None:
    for column in ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]:
        missing_count = int(frame[column].isna().sum())
        if missing_count:
            issues.append(
                AuditIssue(
                    severity="error",
                    code=f"missing_{column}",
                    message=f"{column} contains missing or unparsable values",
                    row_count=missing_count,
                )
            )


def _add_duplicate_issues(frame: pd.DataFrame, issues: list[AuditIssue]) -> None:
    duplicate_mask = frame.duplicated(["date", "ticker"], keep=False)
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count:
        issues.append(
            AuditIssue(
                severity="error",
                code="duplicate_date_ticker",
                message="multiple rows share the same date and ticker",
                row_count=duplicate_count,
            )
        )


def _add_price_sanity_issues(frame: pd.DataFrame, issues: list[AuditIssue]) -> None:
    price_columns = ["open", "high", "low", "close", "adj_close"]
    non_positive_mask = (frame[price_columns] <= 0).any(axis=1)
    non_positive_count = int(non_positive_mask.sum())
    if non_positive_count:
        issues.append(
            AuditIssue(
                severity="error",
                code="non_positive_price",
                message="at least one price column is zero or negative",
                row_count=non_positive_count,
            )
        )

    negative_volume_count = int((frame["volume"] < 0).sum())
    if negative_volume_count:
        issues.append(
            AuditIssue(
                severity="error",
                code="negative_volume",
                message="volume is negative",
                row_count=negative_volume_count,
            )
        )

    ohlc_bad = (
        (frame["high"] < frame[["open", "close", "low"]].max(axis=1))
        | (frame["low"] > frame[["open", "close", "high"]].min(axis=1))
    )
    ohlc_bad_count = int(ohlc_bad.sum())
    if ohlc_bad_count:
        issues.append(
            AuditIssue(
                severity="error",
                code="invalid_ohlc_range",
                message="high/low are inconsistent with open/close",
                row_count=ohlc_bad_count,
            )
        )


def _add_return_outlier_issues(frame: pd.DataFrame, issues: list[AuditIssue]) -> None:
    clean = frame.dropna(subset=["date", "ticker", "adj_close"]).sort_values(["ticker", "date"])
    returns = clean.groupby("ticker")["adj_close"].pct_change()
    outlier_mask = returns.abs() > 0.5
    outliers = clean.loc[outlier_mask.fillna(False), ["ticker"]]
    for ticker, ticker_frame in outliers.groupby("ticker"):
        issues.append(
            AuditIssue(
                severity="warning",
                code="large_adjusted_return",
                message="absolute one-day adjusted return is greater than 50%",
                row_count=len(ticker_frame),
                ticker=str(ticker),
            )
        )


def _add_gap_issues(frame: pd.DataFrame, issues: list[AuditIssue]) -> None:
    clean = frame.dropna(subset=["date", "ticker"]).sort_values(["ticker", "date"])
    for ticker, ticker_frame in clean.groupby("ticker"):
        gaps = ticker_frame["date"].diff().dt.days
        long_gap_count = int((gaps > 10).sum())
        if long_gap_count:
            issues.append(
                AuditIssue(
                    severity="warning",
                    code="long_calendar_gap",
                    message="more than 10 calendar days between adjacent rows",
                    row_count=long_gap_count,
                    ticker=str(ticker),
                )
            )


def _add_stale_price_issues(frame: pd.DataFrame, issues: list[AuditIssue]) -> None:
    clean = frame.dropna(subset=["date", "ticker", "adj_close"]).sort_values(["ticker", "date"])
    for ticker, ticker_frame in clean.groupby("ticker"):
        unchanged = ticker_frame["adj_close"].diff().abs() < np.finfo(float).eps
        streak_id = unchanged.ne(unchanged.shift()).cumsum()
        streak_lengths = unchanged.groupby(streak_id).transform("sum")
        stale_rows = unchanged & (streak_lengths >= 5)
        stale_count = int(stale_rows.sum())
        if stale_count:
            issues.append(
                AuditIssue(
                    severity="warning",
                    code="stale_adjusted_close",
                    message="adjusted close is unchanged for at least 5 consecutive rows",
                    row_count=stale_count,
                    ticker=str(ticker),
                )
            )
