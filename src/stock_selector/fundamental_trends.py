"""Multi-period fundamental trend analysis.

The point-in-time SEC history (``sec_data.extract_sec_fundamental_history``) gives
one row per fiscal period keyed by ``report_date`` (the date the filing was public).
This module turns that history into *trends*: are margins expanding or contracting,
is revenue growth accelerating or decelerating, is cash flow improving? A single
snapshot cannot answer these — only a series can. All filtering is on ``report_date``
so no future filing leaks into an as-of date (lookahead-safe).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


FUNDAMENTAL_TREND_MIN_PERIODS = 3
# A margin slope (decimal margin points per period) beyond this is a real move.
_MARGIN_SLOPE_THRESHOLD = 0.005
# A change in the revenue growth rate per period beyond this is accel/decel.
_GROWTH_SLOPE_THRESHOLD = 0.01


@dataclass(frozen=True)
class FundamentalTrendContext:
    ticker: str
    status: str
    periods_used: int
    latest_period_end: str | None
    trend_score: float
    trend_direction: str
    trend_direction_zh: str
    gross_margin_trend: str
    operating_margin_trend: str
    net_margin_trend: str
    fcf_margin_trend: str
    revenue_growth_trend: str
    note: str
    note_zh: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_fundamental_trend_context(
    history: pd.DataFrame,
    ticker: str | None = None,
    as_of_date: str | pd.Timestamp | None = None,
) -> FundamentalTrendContext:
    resolved_ticker = str(ticker).upper().strip() if ticker else ""
    if history is None or history.empty:
        return _insufficient(resolved_ticker, "no fundamental history available")

    frame = history.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if "period_end" not in frame.columns or "revenue" not in frame.columns:
        return _insufficient(resolved_ticker, "history is missing period_end or revenue")

    if not resolved_ticker and "ticker" in frame.columns and not frame["ticker"].empty:
        resolved_ticker = str(frame["ticker"].iloc[-1]).upper().strip()

    frame["period_end"] = pd.to_datetime(frame["period_end"], errors="coerce")
    # Point-in-time: only periods the market could already see by as_of_date.
    if as_of_date is not None and "report_date" in frame.columns:
        cutoff = pd.Timestamp(as_of_date)
        frame["report_date"] = pd.to_datetime(frame["report_date"], errors="coerce")
        frame = frame[frame["report_date"].notna() & (frame["report_date"] <= cutoff)]

    frame = frame.dropna(subset=["period_end"]).sort_values("period_end")
    frame = frame.drop_duplicates("period_end", keep="last")
    if len(frame) < FUNDAMENTAL_TREND_MIN_PERIODS:
        return _insufficient(
            resolved_ticker,
            f"needs >= {FUNDAMENTAL_TREND_MIN_PERIODS} periods, has {len(frame)}",
        )

    revenue = _numeric(frame, "revenue")
    gross_margin = _safe_ratio(_numeric(frame, "gross_profit"), revenue)
    operating_margin = _safe_ratio(_numeric(frame, "operating_income"), revenue)
    net_margin = _safe_ratio(_numeric(frame, "net_income"), revenue)
    fcf = _numeric(frame, "operating_cash_flow") - _numeric(frame, "capital_expenditure")
    fcf_margin = _safe_ratio(fcf, revenue)
    revenue_growth = revenue.pct_change()

    gross_dir, gross_score = _classify_margin(gross_margin)
    operating_dir, operating_score = _classify_margin(operating_margin)
    net_dir, net_score = _classify_margin(net_margin)
    fcf_dir, fcf_score = _classify_margin(fcf_margin)
    growth_dir, growth_score = _classify_growth(revenue_growth)

    scores = [s for s in (gross_score, operating_score, net_score, fcf_score, growth_score) if s is not None]
    if not scores:
        return _insufficient(resolved_ticker, "no trend could be computed from history")
    trend_score = round(float(np.mean(scores)), 2)
    direction, direction_zh = _overall_direction(trend_score)
    latest_period = pd.Timestamp(frame["period_end"].iloc[-1]).date().isoformat()

    note = (
        f"{len(frame)} periods; gross margin {gross_dir}, operating margin {operating_dir}, "
        f"net margin {net_dir}, FCF margin {fcf_dir}, revenue growth {growth_dir}."
    )
    note_zh = (
        f"{len(frame)}期；毛利率{_ZH[gross_dir]}，营业利润率{_ZH[operating_dir]}，"
        f"净利率{_ZH[net_dir]}，自由现金流率{_ZH[fcf_dir]}，营收增速{_ZH_GROWTH[growth_dir]}。"
    )

    return FundamentalTrendContext(
        ticker=resolved_ticker,
        status="ok",
        periods_used=int(len(frame)),
        latest_period_end=latest_period,
        trend_score=trend_score,
        trend_direction=direction,
        trend_direction_zh=direction_zh,
        gross_margin_trend=gross_dir,
        operating_margin_trend=operating_dir,
        net_margin_trend=net_dir,
        fcf_margin_trend=fcf_dir,
        revenue_growth_trend=growth_dir,
        note=note,
        note_zh=note_zh,
    )


def fetch_fundamental_trend_context(
    ticker: str,
    data_root: str = "data",
    as_of_date: str | pd.Timestamp | None = None,
) -> FundamentalTrendContext:
    """Fetch point-in-time SEC history and analyze its trend. Requires network and
    a ``SEC_USER_AGENT``; returns an ``insufficient_history`` context on any failure
    so callers never have to special-case the SEC path being unavailable."""
    try:
        from .sec_data import fetch_sec_fundamental_history

        history = fetch_sec_fundamental_history(ticker, data_root=data_root)
    except Exception as error:  # network / missing user agent / unknown ticker
        return _insufficient(str(ticker).upper().strip(), f"SEC history fetch failed: {error}")
    return build_fundamental_trend_context(history, ticker=ticker, as_of_date=as_of_date)


_ZH = {"expanding": "扩张", "stable": "稳定", "contracting": "收缩", "unknown": "未知"}
_ZH_GROWTH = {"accelerating": "加速", "stable": "稳定", "decelerating": "放缓", "unknown": "未知"}


def _classify_margin(series: pd.Series) -> tuple[str, float | None]:
    slope = _slope(series)
    if slope is None:
        return "unknown", None
    if slope >= _MARGIN_SLOPE_THRESHOLD:
        return "expanding", 80.0
    if slope <= -_MARGIN_SLOPE_THRESHOLD:
        return "contracting", 30.0
    return "stable", 55.0


def _classify_growth(series: pd.Series) -> tuple[str, float | None]:
    slope = _slope(series)
    if slope is None:
        return "unknown", None
    if slope >= _GROWTH_SLOPE_THRESHOLD:
        return "accelerating", 78.0
    if slope <= -_GROWTH_SLOPE_THRESHOLD:
        return "decelerating", 32.0
    return "stable", 55.0


def _slope(series: pd.Series) -> float | None:
    """Least-squares slope per period over the non-null tail (up to 5 periods)."""
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2:
        return None
    tail = clean.tail(5)
    xs = np.arange(len(tail), dtype=float)
    ys = tail.to_numpy(dtype=float)
    variance = float(((xs - xs.mean()) ** 2).sum())
    if variance <= 0:
        return None
    covariance = float(((xs - xs.mean()) * (ys - ys.mean())).sum())
    return covariance / variance


def _overall_direction(score: float) -> tuple[str, str]:
    if score >= 65:
        return "improving", "改善中"
    if score <= 42:
        return "deteriorating", "恶化中"
    return "stable", "稳定"


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([np.nan] * len(frame), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace(0, np.nan)
    return (numerator / denom).replace([np.inf, -np.inf], np.nan)


def _insufficient(ticker: str, warning: str) -> FundamentalTrendContext:
    return FundamentalTrendContext(
        ticker=ticker,
        status="insufficient_history",
        periods_used=0,
        latest_period_end=None,
        trend_score=50.0,
        trend_direction="unknown",
        trend_direction_zh="未知",
        gross_margin_trend="unknown",
        operating_margin_trend="unknown",
        net_margin_trend="unknown",
        fcf_margin_trend="unknown",
        revenue_growth_trend="unknown",
        note=f"Fundamental trend unavailable: {warning}.",
        note_zh=f"基本面趋势不可用：{warning}。",
    )
