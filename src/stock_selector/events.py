from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EventRiskContext:
    ticker: str
    as_of_date: str
    next_earnings_date: str | None
    days_until_earnings: int | None
    last_earnings_date: str | None
    days_since_earnings: int | None
    event_risk_level: str
    event_risk_level_zh: str
    event_risk_score: float
    event_window: str
    event_window_zh: str
    event_block_new_entries: bool
    event_cooldown_active: bool
    event_risk_note: str
    event_risk_note_zh: str
    event_risk_warning: str
    source: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_event_risk_context(
    ticker: str,
    as_of_date: str | date | pd.Timestamp,
    earnings_dates: Iterable[object],
    source: str = "manual",
    warning: str = "",
) -> EventRiskContext:
    ticker = ticker.upper().strip()
    as_of = pd.Timestamp(as_of_date).normalize()
    parsed_dates = sorted(
        parsed for parsed in (_parse_date(value) for value in earnings_dates) if parsed is not None
    )
    future_dates = [parsed for parsed in parsed_dates if parsed >= as_of]
    past_dates = [parsed for parsed in parsed_dates if parsed < as_of]
    last_date = past_dates[-1] if past_dates else None
    days_since = int((as_of - last_date).days) if last_date is not None else None

    if days_since is not None and days_since <= 3:
        next_date = future_dates[0] if future_dates else None
        days_until = int((next_date - as_of).days) if next_date is not None else None
        return EventRiskContext(
            ticker=ticker,
            as_of_date=as_of.date().isoformat(),
            next_earnings_date=next_date.date().isoformat() if next_date is not None else None,
            days_until_earnings=days_until,
            last_earnings_date=last_date.date().isoformat(),
            days_since_earnings=days_since,
            event_risk_level="medium",
            event_risk_level_zh="中",
            event_risk_score=65.0,
            event_window="post_earnings_cooldown",
            event_window_zh="财报后冷却期",
            event_block_new_entries=True,
            event_cooldown_active=True,
            event_risk_note=(
                f"Earnings were {days_since} calendar days ago; wait for post-earnings price discovery."
            ),
            event_risk_note_zh=(
                f"距离上一次财报只有{days_since}个自然日；等待财报后价格消化。"
            ),
            event_risk_warning=warning,
            source=source,
        )

    if not future_dates:
        return EventRiskContext(
            ticker=ticker,
            as_of_date=as_of.date().isoformat(),
            next_earnings_date=None,
            days_until_earnings=None,
            last_earnings_date=last_date.date().isoformat() if last_date is not None else None,
            days_since_earnings=days_since,
            event_risk_level="unknown",
            event_risk_level_zh="未知",
            event_risk_score=50.0,
            event_window="unknown",
            event_window_zh="未知",
            event_block_new_entries=False,
            event_cooldown_active=False,
            event_risk_note=(
                "Upcoming earnings date is unavailable; no event-risk downgrade is applied."
            ),
            event_risk_note_zh="暂时无法取得下一次财报日期；不进行事件风险降级。",
            event_risk_warning=warning or "earnings date unavailable",
            source=source,
        )

    next_date = future_dates[0]
    days_until = int((next_date - as_of).days)
    level, level_zh, score, window, window_zh, block_new_entries = _classify_event_risk(days_until)
    cooldown_active = False
    if level == "high":
        note = (
            f"Upcoming earnings are in {days_until} calendar days; new entries are downgraded."
        )
        note_zh = f"距离下一次财报还有{days_until}个自然日；新的入场信号会被降级等待。"
    elif level == "medium":
        note = (
            f"Upcoming earnings are in {days_until} calendar days; position timing needs caution."
        )
        note_zh = f"距离下一次财报还有{days_until}个自然日；入场时机需要更谨慎。"
    else:
        note = (
            f"Next known earnings date is {days_until} calendar days away; event risk is low."
        )
        note_zh = f"已知下一次财报距离现在{days_until}个自然日；事件风险较低。"

    return EventRiskContext(
        ticker=ticker,
        as_of_date=as_of.date().isoformat(),
        next_earnings_date=next_date.date().isoformat(),
        days_until_earnings=days_until,
        last_earnings_date=last_date.date().isoformat() if last_date is not None else None,
        days_since_earnings=days_since,
        event_risk_level=level,
        event_risk_level_zh=level_zh,
        event_risk_score=score,
        event_window=window,
        event_window_zh=window_zh,
        event_block_new_entries=block_new_entries,
        event_cooldown_active=cooldown_active,
        event_risk_note=note,
        event_risk_note_zh=note_zh,
        event_risk_warning=warning,
        source=source,
    )


def fetch_yfinance_event_risk(
    ticker: str,
    as_of_date: str | date | pd.Timestamp,
) -> EventRiskContext:
    ticker = ticker.upper().strip()
    warnings: list[str] = []
    candidates: list[object] = []

    try:
        import yfinance as yf
    except ImportError:
        return build_event_risk_context(
            ticker=ticker,
            as_of_date=as_of_date,
            earnings_dates=[],
            source="yfinance",
            warning="yfinance is not installed",
        )

    try:
        stock = yf.Ticker(ticker)
        candidates.extend(_extract_dates_from_earnings_dates(stock))
        candidates.extend(_extract_dates_from_calendar(stock))
    except Exception as exc:
        warnings.append(f"yfinance event fetch failed: {exc}")

    return build_event_risk_context(
        ticker=ticker,
        as_of_date=as_of_date,
        earnings_dates=candidates,
        source="yfinance",
        warning="; ".join(warnings),
    )


def _classify_event_risk(days_until: int) -> tuple[str, str, float, str, str, bool]:
    if days_until <= 3:
        return "high", "高", 95.0, "earnings_imminent", "财报即将发布", True
    if days_until <= 7:
        return "high", "高", 90.0, "earnings_week", "财报周", True
    if days_until <= 14:
        return "medium", "中", 55.0, "earnings_two_weeks", "财报两周内", False
    return "low", "低", 20.0, "normal", "正常", False


def _extract_dates_from_earnings_dates(stock: Any) -> list[object]:
    dates: list[object] = []
    try:
        earnings = stock.get_earnings_dates(limit=12)
    except Exception:
        return dates

    if isinstance(earnings, pd.DataFrame):
        dates.extend(earnings.index.tolist())
        for column in earnings.columns:
            if "earn" in str(column).lower() or "date" in str(column).lower():
                dates.extend(earnings[column].tolist())
    return dates


def _extract_dates_from_calendar(stock: Any) -> list[object]:
    try:
        calendar = stock.calendar
    except Exception:
        return []

    return _walk_for_earnings_dates(calendar)


def _walk_for_earnings_dates(value: Any, parent_key: str = "") -> list[object]:
    dates: list[object] = []
    if isinstance(value, pd.DataFrame):
        for column in value.columns:
            if "earn" in str(column).lower() or "date" in str(column).lower():
                dates.extend(value[column].tolist())
        dates.extend(value.index.tolist())
    elif isinstance(value, pd.Series):
        for key, nested in value.items():
            dates.extend(_walk_for_earnings_dates(nested, str(key)))
    elif isinstance(value, dict):
        for key, nested in value.items():
            dates.extend(_walk_for_earnings_dates(nested, str(key)))
    elif isinstance(value, (list, tuple, set)):
        for nested in value:
            dates.extend(_walk_for_earnings_dates(nested, parent_key))
    elif "earn" in parent_key.lower() or "date" in parent_key.lower():
        dates.append(value)
    return dates


def _parse_date(value: object) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        if isinstance(value, float) and not np.isfinite(value):
            return None
        parsed = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(parsed):
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.tz_convert(None)
    return parsed.normalize()
