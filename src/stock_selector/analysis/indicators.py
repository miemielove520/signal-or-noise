"""Technical indicator primitives: ATR, moving-average slope, window returns, stops."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import (
    _is_finite,
    _last_valid,
    _safe_ratio,
)


def _average_true_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int,
) -> pd.Series:
    previous_close = close.shift(1)
    ranges = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    )
    true_range = ranges.max(axis=1)
    return true_range.rolling(window).mean()


def _moving_average_slope(close: pd.Series, window: int) -> float:
    moving_average = close.rolling(window).mean()
    lag = max(5, min(20, window // 2))
    current = _last_valid(moving_average)
    previous = _last_valid(moving_average.shift(lag))
    return _safe_ratio(current, previous) - 1.0


def _window_return(close: pd.Series, window: int) -> float:
    if len(close) <= window:
        return np.nan
    previous = float(close.iloc[-window - 1])
    current = float(close.iloc[-1])
    return _safe_ratio(current, previous) - 1.0


def _pullback_entry(
    latest_price: float,
    trend_ma: float,
    support: float,
    entry_buffer_pct: float,
) -> float:
    candidates = [value for value in [trend_ma, support * (1.0 + entry_buffer_pct)] if _is_finite(value)]
    if not candidates:
        return latest_price
    below_or_near = [value for value in candidates if value <= latest_price * 1.02]
    if below_or_near:
        return max(below_or_near)
    return min(candidates)


def _stop_loss(
    entry_price: float,
    support: float,
    atr: float,
    atr_multiple: float,
    entry_buffer_pct: float,
) -> float:
    fallback_risk = max(entry_price * 0.05, 0.01)
    atr_stop = entry_price - atr * atr_multiple if _is_finite(atr) and atr > 0 else entry_price - fallback_risk
    structure_stop = support * (1.0 - entry_buffer_pct) if _is_finite(support) else np.nan
    candidates = [value for value in [atr_stop, structure_stop] if _is_finite(value) and value < entry_price]
    if not candidates:
        return max(entry_price - fallback_risk, 0.01)
    return max(candidates)
