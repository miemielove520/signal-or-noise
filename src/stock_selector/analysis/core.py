"""Entry point: run every horizon for one ticker and assemble the analysis tables."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from ..screening_config import ScreeningThresholds
from ..screening_config import TradingRules

from .horizons import (
    HORIZON_SPECS,
    _effective_horizon_spec,
    normalize_horizons,
)
from .screening import (
    _add_calibrated_screening,
    _add_calibrated_watchlist_plan,
    _add_high_probability_screening,
    _add_probability_calibration_feedback,
    _add_signal_review_feedback,
    _add_threshold_calibration,
    _add_watchlist_plan,
)
from .signals import (
    _analyze_horizon,
    _latest_score_snapshot,
    _ticker_prices,
    _ticker_scores,
)
from .summaries import (
    _build_confidence_summary,
    _build_data_quality_summary,
    _build_decision_summary,
    _build_final_decision_summary,
    _build_horizon_alignment_summary,
    _build_priority_blocker_summary,
    _build_risk_breakdown_summary,
)


def analyze_ticker(
    scored: pd.DataFrame,
    prices: pd.DataFrame,
    ticker: str,
    horizons: str | Iterable[str] = "all",
    as_of_date: str | pd.Timestamp | None = None,
    entry_buffer_pct: float = 0.003,
    market_context: object | None = None,
    relative_strength_contexts: dict[str, object] | None = None,
    event_risk_context: object | None = None,
    fundamental_context: object | None = None,
    sentiment_context: object | None = None,
    analyst_context: object | None = None,
    valuation_context: object | None = None,
    sector_context: object | None = None,
    screening_thresholds: ScreeningThresholds | None = None,
    trading_rules: TradingRules | None = None,
    probability_calibration_context: object | None = None,
    signal_review_feedback_context: object | None = None,
) -> pd.DataFrame:
    if entry_buffer_pct < 0:
        raise ValueError("entry_buffer_pct cannot be negative.")

    ticker = ticker.upper().strip()
    if not ticker:
        raise ValueError("ticker cannot be empty.")

    selected_horizons = normalize_horizons(horizons)
    price_frame = _ticker_prices(prices, ticker, as_of_date)
    scored_frame = _ticker_scores(scored, ticker, as_of_date)
    score_snapshot = _latest_score_snapshot(scored, scored_frame, price_frame["date"].max())

    effective_trading_rules = trading_rules or TradingRules()
    rows = [
        _analyze_horizon(
            price_frame=price_frame,
            ticker=ticker,
            spec=_effective_horizon_spec(HORIZON_SPECS[horizon], effective_trading_rules),
            trading_rules=effective_trading_rules,
            score_snapshot=score_snapshot,
            entry_buffer_pct=entry_buffer_pct,
            market_context=market_context,
            relative_strength_context=(relative_strength_contexts or {}).get(horizon),
            event_risk_context=event_risk_context,
            fundamental_context=fundamental_context,
            sentiment_context=sentiment_context,
            analyst_context=analyst_context,
            valuation_context=valuation_context,
            sector_context=sector_context,
        )
        for horizon in selected_horizons
    ]
    analysis = pd.DataFrame(rows)
    decision_summary = _build_decision_summary(analysis)
    for key, value in decision_summary.items():
        analysis[key] = value
    horizon_alignment_summary = _build_horizon_alignment_summary(analysis)
    for key, value in horizon_alignment_summary.items():
        analysis[key] = value
    confidence_summary = _build_confidence_summary(
        analysis,
        focus_horizon=str(decision_summary["decision_focus_horizon"]),
    )
    for key, value in confidence_summary.items():
        analysis[key] = value
    risk_summary = _build_risk_breakdown_summary(
        analysis,
        focus_horizon=str(decision_summary["decision_focus_horizon"]),
    )
    for key, value in risk_summary.items():
        analysis[key] = value
    data_quality_summary = _build_data_quality_summary(analysis)
    for key, value in data_quality_summary.items():
        analysis[key] = value
    thresholds = screening_thresholds or ScreeningThresholds()
    analysis = _add_high_probability_screening(analysis, thresholds)
    analysis = _add_probability_calibration_feedback(
        analysis,
        probability_calibration_context,
    )
    analysis = _add_threshold_calibration(analysis, thresholds)
    analysis = _add_calibrated_screening(analysis, thresholds)
    analysis = _add_signal_review_feedback(analysis, signal_review_feedback_context)
    analysis = _add_calibrated_watchlist_plan(analysis, thresholds)
    analysis = _add_watchlist_plan(analysis, thresholds)
    final_summary = _build_final_decision_summary(analysis)
    for key, value in final_summary.items():
        analysis[key] = value
    priority_blocker_summary = _build_priority_blocker_summary(analysis)
    for key, value in priority_blocker_summary.items():
        analysis[key] = value
    return analysis
