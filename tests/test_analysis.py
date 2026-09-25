from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from stock_selector.analysis import (
    HORIZON_SPECS,
    _calibrated_screening_for_row,
    _calibrated_watchlist_plan_for_row,
    _backtest_decay_profile,
    _dynamic_slippage_profile,
    _entry_evidence_profile,
    _backtest_trust_profile,
    _high_probability_screening_for_row,
    _price_data_health_profile,
    _recent_backtest_profile,
    _regime_coverage_profile,
    _threshold_calibration_for_row,
    _trade_result,
    _watchlist_plan_for_row,
    analyze_ticker,
    normalize_horizons,
    render_ticker_analysis,
)
from stock_selector.screening_config import ScreeningThresholds, TradingRules


def make_price_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=170)
    rows = []
    for ticker, drift in [("AAA", 0.0012), ("BBB", 0.0002)]:
        price = 100.0 if ticker == "AAA" else 80.0
        for index, date in enumerate(dates):
            price *= 1.0 + drift + 0.001 * np.sin(index / 7)
            volume = 1_000_000 + index * 1_000
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": price * 0.995,
                    "high": price * 1.010,
                    "low": price * 0.990,
                    "close": price,
                    "adj_close": price,
                    "volume": volume,
                }
            )
    return pd.DataFrame(rows)


def make_scored_frame(prices: pd.DataFrame) -> pd.DataFrame:
    scored = prices[["date", "ticker", "adj_close", "volume"]].copy()
    scored["passes_universe"] = True
    scored["score"] = np.where(scored["ticker"] == "AAA", 0.8, 0.2)
    return scored


def make_high_probability_gate_row() -> pd.Series:
    return pd.Series(
        {
            "horizon": "short",
            "action": "watch_breakout_or_pullback",
            "entry_type": "stop_limit_breakout",
            "signal_score": 78.0,
            "confidence_score": 82.0,
            "data_quality_score": 90.0,
            "overall_risk_level": "low",
            "market_score": 75.0,
            "relative_strength_score": 65.0,
            "fundamental_score": 70.0,
            "sector_score": 60.0,
            "event_risk_level": "low",
            "event_block_new_entries": False,
            "sentiment_score": 70.0,
            "sentiment_risk_level": "low",
            "sentiment_block_new_entries": False,
            "analyst_score": 72.0,
            "analyst_risk_level": "low",
            "analyst_block_new_entries": False,
            "valuation_score": 68.0,
            "valuation_risk_level": "low",
            "valuation_block_new_entries": False,
            "chase_status": "entry_allowed_if_price_holds",
            "chase_status_zh": "价格守住突破区才可考虑",
            "entry_distance_pct": 0.01,
            "entry_price": 100.0,
            "stop_loss": 94.0,
            "take_profit": 112.0,
            "risk_reward": 2.0,
            "risk_per_share": 6.0,
            "breakout_trade_count": 18,
            "breakout_win_rate": 0.62,
            "breakout_stop_hit_rate": 0.32,
            "breakout_average_return": 0.045,
            "pullback_trade_count": 12,
            "pullback_win_rate": 0.58,
            "pullback_stop_hit_rate": 0.35,
            "pullback_average_return": 0.030,
            "backtest_avg_dollar_volume": 250_000_000.0,
            "backtest_slippage_pct": 0.001,
            "backtest_liquidity_label": "high_liquidity",
            "backtest_liquidity_label_zh": "流动性较高",
            "backtest_trust_score": 82.0,
            "backtest_trust_level": "high_trust",
            "backtest_trust_level_zh": "可信度较高",
            "recent_backtest_score": 78.0,
            "recent_backtest_level": "stable",
            "recent_backtest_level_zh": "近期稳定",
            "backtest_decay_score": 76.0,
            "backtest_decay_level": "acceptable",
            "backtest_decay_level_zh": "可接受",
            "backtest_sample_score": 85.0,
            "backtest_liquidity_score": 95.0,
            "backtest_slippage_score": 90.0,
            "backtest_return_evidence_score": 75.0,
            "backtest_trust_note": (
                "Trust score blends sample count, price health, liquidity, "
                "slippage, and return evidence."
            ),
            "backtest_trust_note_zh": "可信度分综合样本数、价格健康、流动性、滑点和收益证据。",
        }
    )


class AnalysisTest(unittest.TestCase):
    def test_analyze_ticker_outputs_trade_plan_for_each_horizon(self) -> None:
        prices = make_price_frame()
        scored = make_scored_frame(prices)

        result = analyze_ticker(
            scored=scored,
            prices=prices,
            ticker="AAA",
            horizons="all",
        )

        self.assertEqual(result["horizon"].to_list(), ["short", "medium", "long"])
        self.assertTrue((result["entry_price"] > 0).all())
        self.assertTrue((result["stop_loss"] < result["entry_price"]).all())
        self.assertTrue((result["take_profit"] > result["entry_price"]).all())
        self.assertTrue((result["signal_score"] >= 0).all())
        self.assertIn("rationale", result.columns)
        self.assertIn("plain_summary", result.columns)
        self.assertIn("plain_summary_zh", result.columns)
        self.assertIn("breakout_trade_count", result.columns)
        self.assertIn("breakout_sample_quality", result.columns)
        self.assertIn("breakout_sample_quality_zh", result.columns)
        self.assertIn("breakout_win_rate", result.columns)
        self.assertIn("breakout_target_hit_rate", result.columns)
        self.assertIn("breakout_stop_hit_rate", result.columns)
        self.assertIn("breakout_trailing_stop_hit_rate", result.columns)
        self.assertIn("breakout_average_gain", result.columns)
        self.assertIn("breakout_average_loss", result.columns)
        self.assertIn("breakout_average_return", result.columns)
        self.assertIn("backtest_time_stop_days", result.columns)
        self.assertIn("pullback_trade_count", result.columns)
        self.assertIn("pullback_sample_quality", result.columns)
        self.assertIn("pullback_sample_quality_zh", result.columns)
        self.assertIn("pullback_win_rate", result.columns)
        self.assertIn("pullback_target_hit_rate", result.columns)
        self.assertIn("pullback_stop_hit_rate", result.columns)
        self.assertIn("pullback_trailing_stop_hit_rate", result.columns)
        self.assertIn("pullback_average_gain", result.columns)
        self.assertIn("pullback_average_loss", result.columns)
        self.assertIn("pullback_average_return", result.columns)
        self.assertIn("event_risk_level", result.columns)
        self.assertIn("event_risk_score", result.columns)
        self.assertIn("event_window", result.columns)
        self.assertIn("event_window_zh", result.columns)
        self.assertIn("event_block_new_entries", result.columns)
        self.assertIn("event_cooldown_active", result.columns)
        self.assertIn("next_earnings_date", result.columns)
        self.assertIn("last_earnings_date", result.columns)
        self.assertIn("days_since_earnings", result.columns)
        self.assertIn("sentiment_score", result.columns)
        self.assertIn("sentiment_label", result.columns)
        self.assertIn("sentiment_label_zh", result.columns)
        self.assertIn("sentiment_risk_level", result.columns)
        self.assertIn("sentiment_risk_level_zh", result.columns)
        self.assertIn("sentiment_block_new_entries", result.columns)
        self.assertIn("positive_news_level", result.columns)
        self.assertIn("positive_news_level_zh", result.columns)
        self.assertIn("positive_news_score", result.columns)
        self.assertIn("positive_news_drivers_zh", result.columns)
        self.assertIn("sentiment_positive_count", result.columns)
        self.assertIn("sentiment_negative_count", result.columns)
        self.assertIn("sentiment_high_risk_count", result.columns)
        self.assertIn("sentiment_titles_used", result.columns)
        self.assertIn("sentiment_note_zh", result.columns)
        self.assertIn("analyst_score", result.columns)
        self.assertIn("analyst_label", result.columns)
        self.assertIn("analyst_label_zh", result.columns)
        self.assertIn("analyst_risk_level", result.columns)
        self.assertIn("analyst_risk_level_zh", result.columns)
        self.assertIn("analyst_block_new_entries", result.columns)
        self.assertIn("analyst_upside", result.columns)
        self.assertIn("recommendation_mean", result.columns)
        self.assertIn("recommendation_key", result.columns)
        self.assertIn("number_of_analysts", result.columns)
        self.assertIn("target_mean_price", result.columns)
        self.assertIn("analyst_note_zh", result.columns)
        self.assertIn("valuation_score", result.columns)
        self.assertIn("valuation_label", result.columns)
        self.assertIn("valuation_label_zh", result.columns)
        self.assertIn("valuation_risk_level", result.columns)
        self.assertIn("valuation_risk_level_zh", result.columns)
        self.assertIn("valuation_block_new_entries", result.columns)
        self.assertIn("valuation_forward_pe", result.columns)
        self.assertIn("valuation_peg_ratio", result.columns)
        self.assertIn("valuation_growth_reference", result.columns)
        self.assertIn("valuation_profit_margin", result.columns)
        self.assertIn("valuation_note_zh", result.columns)
        self.assertIn("fundamental_score", result.columns)
        self.assertIn("fundamental_quality", result.columns)
        self.assertIn("fundamental_note_zh", result.columns)
        self.assertIn("sector_score", result.columns)
        self.assertIn("sector_status", result.columns)
        self.assertIn("sector_etf", result.columns)
        self.assertIn("sector_note_zh", result.columns)
        self.assertIn("primary_decision", result.columns)
        self.assertIn("primary_decision_zh", result.columns)
        self.assertIn("decision_reason", result.columns)
        self.assertIn("decision_reason_zh", result.columns)
        self.assertIn("decision_wait_for", result.columns)
        self.assertIn("decision_wait_for_zh", result.columns)
        self.assertIn("decision_invalidation", result.columns)
        self.assertIn("decision_invalidation_zh", result.columns)
        self.assertIn("decision_focus_horizon", result.columns)
        self.assertIn("horizon_alignment_score", result.columns)
        self.assertIn("horizon_alignment_label", result.columns)
        self.assertIn("horizon_alignment_label_zh", result.columns)
        self.assertIn("constructive_horizon_count", result.columns)
        self.assertIn("weak_horizon_count", result.columns)
        self.assertIn("risk_wait_horizon_count", result.columns)
        self.assertIn("horizon_signal_score_spread", result.columns)
        self.assertIn("horizon_alignment_note", result.columns)
        self.assertIn("horizon_alignment_note_zh", result.columns)
        self.assertIn("confidence_score", result.columns)
        self.assertIn("confidence_level", result.columns)
        self.assertIn("confidence_level_zh", result.columns)
        self.assertIn("confidence_note", result.columns)
        self.assertIn("confidence_note_zh", result.columns)
        self.assertIn("entry_distance_pct", result.columns)
        self.assertIn("entry_distance_note", result.columns)
        self.assertIn("entry_distance_note_zh", result.columns)
        self.assertIn("chase_status", result.columns)
        self.assertIn("chase_status_zh", result.columns)
        self.assertIn("entry_plan", result.columns)
        self.assertIn("entry_plan_zh", result.columns)
        self.assertIn("backtest_execution_model", result.columns)
        self.assertIn("backtest_execution_model_zh", result.columns)
        self.assertIn("backtest_slippage_pct", result.columns)
        self.assertIn("backtest_avg_dollar_volume", result.columns)
        self.assertIn("backtest_atr_ratio", result.columns)
        self.assertIn("backtest_liquidity_label", result.columns)
        self.assertIn("backtest_liquidity_label_zh", result.columns)
        self.assertIn("backtest_volatility_label", result.columns)
        self.assertIn("backtest_volatility_label_zh", result.columns)
        self.assertIn("backtest_execution_note", result.columns)
        self.assertIn("backtest_execution_note_zh", result.columns)
        self.assertIn("overall_risk_score", result.columns)
        self.assertIn("overall_risk_level", result.columns)
        self.assertIn("overall_risk_level_zh", result.columns)
        self.assertIn("technical_risk_level", result.columns)
        self.assertIn("market_risk_level", result.columns)
        self.assertIn("relative_strength_risk_level", result.columns)
        self.assertIn("sector_risk_level", result.columns)
        self.assertIn("sector_risk_level_zh", result.columns)
        self.assertIn("fundamental_risk_level", result.columns)
        self.assertIn("event_risk_breakdown_level", result.columns)
        self.assertIn("sentiment_risk_breakdown_level", result.columns)
        self.assertIn("analyst_risk_breakdown_level", result.columns)
        self.assertIn("valuation_risk_breakdown_level", result.columns)
        self.assertIn("data_risk_level", result.columns)
        self.assertIn("risk_breakdown_note_zh", result.columns)
        self.assertIn("backtest_reliability_level", result.columns)
        self.assertIn("backtest_reliability_level_zh", result.columns)
        self.assertIn("backtest_reliability_note", result.columns)
        self.assertIn("backtest_reliability_note_zh", result.columns)
        self.assertIn("backtest_trust_score", result.columns)
        self.assertIn("backtest_trust_level", result.columns)
        self.assertIn("backtest_trust_level_zh", result.columns)
        self.assertIn("backtest_sample_score", result.columns)
        self.assertIn("backtest_liquidity_score", result.columns)
        self.assertIn("backtest_slippage_score", result.columns)
        self.assertIn("backtest_return_evidence_score", result.columns)
        self.assertIn("regime_coverage_score", result.columns)
        self.assertIn("regime_coverage_level", result.columns)
        self.assertIn("regime_coverage_level_zh", result.columns)
        self.assertIn("regime_coverage_regime_count", result.columns)
        self.assertIn("regime_coverage_dominant_regime", result.columns)
        self.assertIn("regime_coverage_dominant_share", result.columns)
        self.assertIn("regime_coverage_note", result.columns)
        self.assertIn("regime_coverage_note_zh", result.columns)
        self.assertIn("recent_backtest_score", result.columns)
        self.assertIn("recent_backtest_level", result.columns)
        self.assertIn("recent_backtest_level_zh", result.columns)
        self.assertIn("recent_backtest_trade_count", result.columns)
        self.assertIn("recent_backtest_win_rate", result.columns)
        self.assertIn("recent_backtest_average_return", result.columns)
        self.assertIn("recent_backtest_return_delta", result.columns)
        self.assertIn("recent_backtest_note", result.columns)
        self.assertIn("recent_backtest_note_zh", result.columns)
        self.assertIn("backtest_decay_score", result.columns)
        self.assertIn("backtest_decay_level", result.columns)
        self.assertIn("backtest_decay_level_zh", result.columns)
        self.assertIn("backtest_decay_early_trade_count", result.columns)
        self.assertIn("backtest_decay_late_trade_count", result.columns)
        self.assertIn("backtest_decay_early_win_rate", result.columns)
        self.assertIn("backtest_decay_late_win_rate", result.columns)
        self.assertIn("backtest_decay_early_average_return", result.columns)
        self.assertIn("backtest_decay_late_average_return", result.columns)
        self.assertIn("backtest_decay_average_return_delta", result.columns)
        self.assertIn("backtest_decay_note", result.columns)
        self.assertIn("backtest_decay_note_zh", result.columns)
        self.assertIn("backtest_trust_note", result.columns)
        self.assertIn("backtest_trust_note_zh", result.columns)
        self.assertIn("data_quality_score", result.columns)
        self.assertIn("data_quality_level", result.columns)
        self.assertIn("data_quality_level_zh", result.columns)
        self.assertIn("price_data_status", result.columns)
        self.assertIn("price_data_status_zh", result.columns)
        self.assertIn("price_health_score", result.columns)
        self.assertIn("price_health_level", result.columns)
        self.assertIn("price_health_level_zh", result.columns)
        self.assertIn("price_health_issue_count", result.columns)
        self.assertIn("price_max_calendar_gap_days", result.columns)
        self.assertIn("price_large_gap_count", result.columns)
        self.assertIn("price_zero_volume_days", result.columns)
        self.assertIn("price_missing_ohlcv_rows", result.columns)
        self.assertIn("price_extreme_return_count", result.columns)
        self.assertIn("price_health_note", result.columns)
        self.assertIn("price_health_note_zh", result.columns)
        self.assertIn("market_data_status", result.columns)
        self.assertIn("market_data_status_zh", result.columns)
        self.assertIn("relative_strength_data_status", result.columns)
        self.assertIn("relative_strength_data_status_zh", result.columns)
        self.assertIn("sector_data_status", result.columns)
        self.assertIn("sector_data_status_zh", result.columns)
        self.assertIn("fundamental_data_status", result.columns)
        self.assertIn("fundamental_data_status_zh", result.columns)
        self.assertIn("event_data_status", result.columns)
        self.assertIn("event_data_status_zh", result.columns)
        self.assertIn("sentiment_data_status", result.columns)
        self.assertIn("sentiment_data_status_zh", result.columns)
        self.assertIn("analyst_data_status", result.columns)
        self.assertIn("analyst_data_status_zh", result.columns)
        self.assertIn("valuation_data_status", result.columns)
        self.assertIn("valuation_data_status_zh", result.columns)
        self.assertIn("missing_fallback_count", result.columns)
        self.assertIn("data_quality_note", result.columns)
        self.assertIn("data_quality_note_zh", result.columns)
        self.assertIn("data_quality_weakest_layer", result.columns)
        self.assertIn("data_quality_weakest_layer_zh", result.columns)
        self.assertIn("data_quality_weak_layers", result.columns)
        self.assertIn("data_quality_weak_layers_zh", result.columns)
        self.assertIn("data_quality_repair_actions", result.columns)
        self.assertIn("data_quality_repair_actions_zh", result.columns)
        self.assertIn("data_quality_repair_priority", result.columns)
        self.assertIn("data_quality_repair_priority_zh", result.columns)
        self.assertIn("screening_action", result.columns)
        self.assertIn("screening_action_zh", result.columns)
        self.assertIn("high_probability_score", result.columns)
        self.assertIn("high_probability_level", result.columns)
        self.assertIn("high_probability_level_zh", result.columns)
        self.assertIn("calibrated_win_probability", result.columns)
        self.assertIn("calibrated_win_probability_raw", result.columns)
        self.assertIn("probability_calibration_adjustment", result.columns)
        self.assertIn("probability_calibration_source", result.columns)
        self.assertIn("probability_calibration_source_zh", result.columns)
        self.assertIn("probability_calibration_sample_count", result.columns)
        self.assertIn("probability_calibration_action", result.columns)
        self.assertIn("probability_calibration_action_zh", result.columns)
        self.assertIn("probability_calibration_note", result.columns)
        self.assertIn("probability_calibration_note_zh", result.columns)
        self.assertIn("calibrated_probability_level", result.columns)
        self.assertIn("calibrated_probability_level_zh", result.columns)
        self.assertIn("calibrated_probability_confidence", result.columns)
        self.assertIn("calibrated_probability_confidence_level", result.columns)
        self.assertIn("calibrated_probability_confidence_level_zh", result.columns)
        self.assertIn("calibrated_probability_note", result.columns)
        self.assertIn("calibrated_probability_note_zh", result.columns)
        self.assertIn("quality_gate_passed", result.columns)
        self.assertIn("quality_gate_fail_reasons", result.columns)
        self.assertIn("quality_gate_fail_reasons_zh", result.columns)
        self.assertIn("screening_backtest_entry_type", result.columns)
        self.assertIn("screening_backtest_trade_count", result.columns)
        self.assertIn("screening_backtest_win_rate", result.columns)
        self.assertIn("screening_backtest_stop_hit_rate", result.columns)
        self.assertIn("screening_backtest_average_return", result.columns)
        self.assertIn("sample_confidence_level", result.columns)
        self.assertIn("sample_confidence_level_zh", result.columns)
        self.assertIn("evidence_strength", result.columns)
        self.assertIn("evidence_strength_zh", result.columns)
        self.assertIn("evidence_note", result.columns)
        self.assertIn("evidence_note_zh", result.columns)
        self.assertIn("liquidity_filter_passed", result.columns)
        self.assertIn("liquidity_filter_reason", result.columns)
        self.assertIn("liquidity_filter_reason_zh", result.columns)
        self.assertIn("entry_readiness_gate_passed", result.columns)
        self.assertIn("entry_readiness_status", result.columns)
        self.assertIn("entry_readiness_status_zh", result.columns)
        self.assertIn("entry_readiness_score", result.columns)
        self.assertIn("entry_readiness_reason", result.columns)
        self.assertIn("entry_readiness_reason_zh", result.columns)
        self.assertIn("entry_readiness_note", result.columns)
        self.assertIn("entry_readiness_note_zh", result.columns)
        self.assertIn("trade_plan_quality_gate_passed", result.columns)
        self.assertIn("trade_plan_quality_status", result.columns)
        self.assertIn("trade_plan_quality_status_zh", result.columns)
        self.assertIn("trade_plan_quality_score", result.columns)
        self.assertIn("trade_plan_quality_reason", result.columns)
        self.assertIn("trade_plan_quality_reason_zh", result.columns)
        self.assertIn("trade_plan_quality_note", result.columns)
        self.assertIn("trade_plan_quality_note_zh", result.columns)
        self.assertIn("calibration_action", result.columns)
        self.assertIn("calibration_action_zh", result.columns)
        self.assertIn("calibration_level", result.columns)
        self.assertIn("calibration_level_zh", result.columns)
        self.assertIn("recommended_signal_threshold", result.columns)
        self.assertIn("market_regime", result.columns)
        self.assertIn("market_regime_zh", result.columns)
        self.assertIn("market_regime_note", result.columns)
        self.assertIn("market_regime_note_zh", result.columns)
        self.assertIn("market_regime_signal_delta", result.columns)
        self.assertIn("market_regime_confidence_delta", result.columns)
        self.assertIn("market_regime_sample_delta", result.columns)
        self.assertIn("market_regime_win_rate_delta", result.columns)
        self.assertIn("market_regime_average_return_delta", result.columns)
        self.assertIn("recommended_confidence_threshold", result.columns)
        self.assertIn("recommended_backtest_sample_min", result.columns)
        self.assertIn("recommended_backtest_win_rate_min", result.columns)
        self.assertIn("recommended_backtest_average_return_min", result.columns)
        self.assertIn("calibration_note", result.columns)
        self.assertIn("calibration_note_zh", result.columns)
        self.assertIn("calibrated_screening_action", result.columns)
        self.assertIn("calibrated_screening_action_zh", result.columns)
        self.assertIn("calibrated_quality_gate_passed", result.columns)
        self.assertIn("calibrated_high_probability_score", result.columns)
        self.assertIn("calibrated_high_probability_level", result.columns)
        self.assertIn("calibrated_quality_gate_fail_reasons", result.columns)
        self.assertIn("calibrated_quality_gate_fail_reasons_zh", result.columns)
        self.assertIn("signal_review_score", result.columns)
        self.assertIn("signal_review_adjustment", result.columns)
        self.assertIn("signal_review_level", result.columns)
        self.assertIn("signal_review_level_zh", result.columns)
        self.assertIn("signal_review_note", result.columns)
        self.assertIn("signal_review_note_zh", result.columns)
        self.assertIn("calibrated_watchlist_status", result.columns)
        self.assertIn("calibrated_watchlist_status_zh", result.columns)
        self.assertIn("calibrated_watchlist_gap_score", result.columns)
        self.assertIn("calibrated_watchlist_ready_items", result.columns)
        self.assertIn("calibrated_watchlist_ready_items_zh", result.columns)
        self.assertIn("calibrated_watchlist_missing_items", result.columns)
        self.assertIn("calibrated_watchlist_missing_items_zh", result.columns)
        self.assertIn("calibrated_watchlist_missing_count", result.columns)
        self.assertIn("calibrated_watchlist_trigger_price", result.columns)
        self.assertIn("calibrated_watchlist_recheck_reason", result.columns)
        self.assertIn("calibrated_watchlist_recheck_reason_zh", result.columns)
        self.assertIn("final_decision", result.columns)
        self.assertIn("final_decision_zh", result.columns)
        self.assertIn("final_focus_horizon", result.columns)
        self.assertIn("final_focus_horizon_zh", result.columns)
        self.assertIn("final_score", result.columns)
        self.assertIn("final_watchlist_status", result.columns)
        self.assertIn("final_watchlist_status_zh", result.columns)
        self.assertIn("final_reason", result.columns)
        self.assertIn("final_reason_zh", result.columns)
        self.assertIn("final_next_step", result.columns)
        self.assertIn("final_next_step_zh", result.columns)
        self.assertIn("primary_blocker", result.columns)
        self.assertIn("primary_blocker_zh", result.columns)
        self.assertIn("priority_blockers", result.columns)
        self.assertIn("priority_blockers_zh", result.columns)
        self.assertIn("priority_blocker_count", result.columns)
        self.assertIn("priority_blocker_note", result.columns)
        self.assertIn("priority_blocker_note_zh", result.columns)
        self.assertIn("primary_blocker_resolution", result.columns)
        self.assertIn("primary_blocker_resolution_zh", result.columns)
        self.assertIn("blocker_resolution_steps", result.columns)
        self.assertIn("blocker_resolution_steps_zh", result.columns)
        self.assertIn("blocker_recheck_trigger", result.columns)
        self.assertIn("blocker_recheck_trigger_zh", result.columns)
        self.assertIn("primary_blocker_progress", result.columns)
        self.assertIn("blocker_resolution_score", result.columns)

    def test_analyze_ticker_applies_walk_forward_probability_feedback(self) -> None:
        prices = make_price_frame()
        scored = make_scored_frame(prices)
        calibration = pd.DataFrame(
            [
                {
                    "probability_bucket": "overall",
                    "probability_bucket_zh": "整体",
                    "sample_count": 20,
                    "recommended_probability_adjustment": 0.10,
                    "formula_action": "raise_probability_estimate",
                    "formula_action_zh": "上调胜率估计",
                    "calibration_note": "Overall validation suggests raising probability.",
                    "calibration_note_zh": "整体验证建议上调概率。",
                }
            ]
        )

        result = analyze_ticker(
            scored=scored,
            prices=prices,
            ticker="AAA",
            horizons="all",
            probability_calibration_context=calibration,
        )

        self.assertTrue(
            (
                result["calibrated_win_probability"]
                >= result["calibrated_win_probability_raw"]
            ).all()
        )
        self.assertTrue(
            (
                result["calibrated_win_probability"]
                > result["calibrated_win_probability_raw"]
            ).any()
        )
        self.assertTrue((result["probability_calibration_adjustment"] <= 0.08).all())
        self.assertEqual(
            set(result["probability_calibration_action"]),
            {"raise_probability_estimate"},
        )
        self.assertIn("blocker_resolution_level", result.columns)
        self.assertIn("blocker_resolution_level_zh", result.columns)
        self.assertIn("blocker_resolution_gap", result.columns)
        self.assertIn("blocker_resolution_gap_zh", result.columns)
        self.assertIn("watchlist_status", result.columns)
        self.assertIn("watchlist_status_zh", result.columns)
        self.assertIn("watchlist_gap_score", result.columns)
        self.assertIn("watchlist_ready_items", result.columns)
        self.assertIn("watchlist_ready_items_zh", result.columns)
        self.assertIn("watchlist_missing_items", result.columns)
        self.assertIn("watchlist_missing_items_zh", result.columns)
        self.assertIn("watchlist_missing_count", result.columns)
        self.assertIn("watchlist_trigger_price", result.columns)
        self.assertIn("watchlist_recheck_reason", result.columns)
        self.assertIn("watchlist_recheck_reason_zh", result.columns)
        self.assertTrue(result["primary_decision"].notna().all())
        self.assertTrue(((result["confidence_score"] >= 0) & (result["confidence_score"] <= 100)).all())
        self.assertTrue(((result["overall_risk_score"] >= 0) & (result["overall_risk_score"] <= 100)).all())
        self.assertTrue(((result["data_quality_score"] >= 0) & (result["data_quality_score"] <= 100)).all())
        self.assertTrue(((result["price_health_score"] >= 0) & (result["price_health_score"] <= 100)).all())
        self.assertTrue(((result["backtest_trust_score"] >= 0) & (result["backtest_trust_score"] <= 100)).all())
        self.assertTrue(((result["regime_coverage_score"] >= 0) & (result["regime_coverage_score"] <= 100)).all())
        self.assertTrue(((result["recent_backtest_score"] >= 0) & (result["recent_backtest_score"] <= 100)).all())
        self.assertTrue(((result["backtest_decay_score"] >= 0) & (result["backtest_decay_score"] <= 100)).all())
        self.assertTrue(result["data_quality_weakest_layer"].notna().all())
        self.assertTrue(result["data_quality_repair_actions"].notna().all())
        self.assertTrue(((result["high_probability_score"] >= 0) & (result["high_probability_score"] <= 100)).all())
        self.assertTrue(((result["blocker_resolution_score"] >= 0) & (result["blocker_resolution_score"] <= 100)).all())
        self.assertTrue(np.isfinite(result["entry_distance_pct"]).all())

    def test_normalize_horizons_rejects_unknown_value(self) -> None:
        with self.assertRaises(ValueError):
            normalize_horizons(["swing"])

    def test_signal_review_feedback_adjusts_high_probability_score(self) -> None:
        prices = make_price_frame()
        scored = make_scored_frame(prices)
        neutral = analyze_ticker(scored=scored, prices=prices, ticker="AAA", horizons="short")
        weak = analyze_ticker(
            scored=scored,
            prices=prices,
            ticker="AAA",
            horizons="short",
            signal_review_feedback_context={
                "status": "ok",
                "score": 30.0,
                "adjustment": -4.4,
                "sample_count": 6,
                "focus_window": "20d",
                "win_rate": 0.25,
                "avg_return": -0.03,
                "median_return": -0.02,
                "level": "poor",
                "level_zh": "历史复盘较差",
                "note": "Prior signal review is weak.",
                "note_zh": "历史信号复盘偏弱。",
            },
        )

        self.assertLess(
            float(weak.iloc[0]["high_probability_score"]),
            float(neutral.iloc[0]["high_probability_score"]),
        )
        self.assertEqual(weak.iloc[0]["signal_review_level"], "poor")
        self.assertIn("历史信号复盘偏弱", weak.iloc[0]["calibrated_quality_gate_fail_reasons_zh"])

    def test_render_ticker_analysis_includes_signal_table(self) -> None:
        prices = make_price_frame()
        scored = make_scored_frame(prices)
        result = analyze_ticker(scored=scored, prices=prices, ticker="AAA", horizons="all")

        report = render_ticker_analysis(result)

        self.assertIn("# Ticker Analysis: AAA", report)
        self.assertIn("## Final Decision / 最终执行结论", report)
        self.assertIn("Final decision / 最终结论", report)
        self.assertIn("Final next step / 下一步", report)
        self.assertIn("## Priority Blockers / 主要卡点排序", report)
        self.assertIn("Primary blocker / 第一卡点", report)
        self.assertIn("Blocker note / 卡点说明", report)
        self.assertIn("Primary blocker resolution / 第一卡点解除条件", report)
        self.assertIn("Re-check trigger / 重新检查触发条件", report)
        self.assertIn("Blocker resolution score / 卡点解除分数", report)
        self.assertIn("Resolution gap / 解除差距", report)
        self.assertIn("## Horizon Alignment / 周期一致性", report)
        self.assertIn("Alignment label / 一致性标签", report)
        self.assertIn("Alignment note / 一致性说明", report)
        self.assertIn("## Primary Decision / 主要结论", report)
        self.assertIn("Decision / 结论", report)
        self.assertIn("What to wait for / 等什么", report)
        self.assertIn("Invalidation / 判断失效条件", report)
        self.assertIn("## High Probability Filter / 高概率筛选器", report)
        self.assertIn("Final screening / 最终筛选结论", report)
        self.assertIn("High probability score / 高概率分数", report)
        self.assertIn("Calibrated win probability / 校准后胜率估计", report)
        self.assertIn("Probability calibration feedback / 概率校准反馈", report)
        self.assertIn("Probability calibration action / 概率校准动作", report)
        self.assertIn("Probability confidence / 概率置信度", report)
        self.assertIn("Quality gate passed / 是否通过质量门槛", report)
        self.assertIn("Sample confidence / 样本置信度", report)
        self.assertIn("## Signal Review Feedback / 信号复盘反馈", report)
        self.assertIn("Score adjustment / 分数调整", report)
        self.assertIn("Evidence strength / 证据强度", report)
        self.assertIn("Evidence note / 证据说明", report)
        self.assertIn("Backtest trust gate / 回测可信度门槛", report)
        self.assertIn("Stability gates / 稳定性门槛", report)
        self.assertIn("Regime coverage score / 行情覆盖分", report)
        self.assertIn("Regime coverage note / 行情覆盖说明", report)
        self.assertIn("Recent backtest strength / 近期回测强度", report)
        self.assertIn("Recent backtest note / 近期回测说明", report)
        self.assertIn("Backtest decay check / 回测衰退检查", report)
        self.assertIn("Backtest decay note / 回测衰退说明", report)
        self.assertIn("Liquidity filter / 流动性过滤", report)
        self.assertIn("Entry readiness / 买点可执行性", report)
        self.assertIn("Entry readiness note / 买点可执行性说明", report)
        self.assertIn("Trade plan quality / 交易计划质量", report)
        self.assertIn("Trade plan quality note / 交易计划质量说明", report)
        self.assertIn("Gate result / 门槛结果", report)
        self.assertIn("## Profile Trading Rules / 大类交易规则", report)
        self.assertIn("Entry style / 买点风格", report)
        self.assertIn("Sell rule / 卖出规则", report)
        self.assertIn("## Threshold Calibration / 阈值校准", report)
        self.assertIn("Market regime / 市场状态分层", report)
        self.assertIn("Market regime adjustment / 市场状态门槛调整", report)
        self.assertIn("Recommended signal threshold / 建议信号分门槛", report)
        self.assertIn("Calibration note / 校准说明", report)
        self.assertIn("## Calibrated Screening / 校准后筛选", report)
        self.assertIn("Calibrated final screening / 校准后最终结论", report)
        self.assertIn("Calibrated gate result / 校准后门槛结果", report)
        self.assertIn("Calibrated watchlist status / 校准后观察状态", report)
        self.assertIn("Calibrated missing items / 校准后未达标条件", report)
        self.assertIn("## Watchlist Plan / 观察计划", report)
        self.assertIn("Watchlist status / 观察状态", report)
        self.assertIn("Missing items / 未达标条件", report)
        self.assertIn("Re-check reason / 重新检查原因", report)
        self.assertIn("## Confidence / 置信度", report)
        self.assertIn("Confidence score / 置信度分数", report)
        self.assertIn("Confidence note / 置信度说明", report)
        self.assertIn("## Risk Breakdown / 风险拆解", report)
        self.assertIn("Overall risk / 综合风险", report)
        self.assertIn("Technical risk / 技术风险", report)
        self.assertIn("Sector risk / 板块风险", report)
        self.assertIn("Risk note / 风险说明", report)
        self.assertIn("## Data Quality / 数据质量", report)
        self.assertIn("Data quality score / 数据质量分数", report)
        self.assertIn("Weakest data layer / 最弱数据层", report)
        self.assertIn("Weak data layers / 薄弱数据层", report)
        self.assertIn("Repair priority / 修复优先级", report)
        self.assertIn("Repair actions / 修复动作", report)
        self.assertIn("Data repair actions applied / 已自动修复", report)
        self.assertIn("Price data / 价格数据", report)
        self.assertIn("Price health score / 价格健康分", report)
        self.assertIn("Price health note / 价格健康说明", report)
        self.assertIn("Market data / 大盘数据", report)
        self.assertIn("Relative strength data / 相对强弱数据", report)
        self.assertIn("Sector data / 板块数据", report)
        self.assertIn("Fundamental data / 基本面数据", report)
        self.assertIn("Event data / 事件数据", report)
        self.assertIn("Missing fallback count / 缺失回退数量", report)
        self.assertIn("## Plain Summary / 简明结论", report)
        self.assertIn("## Entry Plan / 买点计划", report)
        self.assertIn("## Backtest Reliability / 回测可靠性", report)
        self.assertIn("Time stop / 时间止损", report)
        self.assertIn("Trailing stop / 移动止损", report)
        self.assertIn("backtest_reliability_note_zh", report)
        self.assertIn("## Sector Context / 板块环境", report)
        self.assertIn("Sector ETF / 板块ETF", report)
        self.assertIn("Sector score / 板块分数", report)
        self.assertIn("entry_distance_note_zh", report)
        self.assertIn("chase_status_zh", report)
        self.assertIn("entry_plan_zh", report)
        self.assertIn("plain_summary", report)
        self.assertIn("plain_summary_zh", report)
        self.assertIn("## Signal Plan", report)
        self.assertNotIn("## Capital Plan / 资金计划", report)
        self.assertIn("## Entry Backtest / 买点回测", report)
        self.assertIn("Execution model / 执行模型", report)
        self.assertIn("Backtest trust score / 回测可信度分", report)
        self.assertIn("Backtest trust note / 回测可信度说明", report)
        self.assertIn("next_open_dynamic_slippage", report)
        self.assertIn("Slippage / 滑点", report)
        self.assertIn("Slippage drivers / 滑点依据", report)
        self.assertIn("## Event Risk / 事件风险", report)
        self.assertIn("Event risk score / 事件风险分数", report)
        self.assertIn("Event window / 事件窗口", report)
        self.assertIn("Block new entries / 是否阻止新入场", report)
        self.assertIn("Cooldown active / 财报后冷却是否生效", report)
        self.assertIn("Last earnings date / 上一次财报日期", report)
        self.assertIn("Days since earnings / 距离上次财报天数", report)
        self.assertIn("## News Sentiment / 新闻情绪", report)
        self.assertIn("Sentiment score / 情绪分数", report)
        self.assertIn("Positive news level / 利好等级", report)
        self.assertIn("High-risk keywords / 高风险关键词数", report)
        self.assertIn("## Analyst Expectations / 分析师预期", report)
        self.assertIn("Analyst score / 分析师分数", report)
        self.assertIn("Analyst upside / 目标价上行空间", report)
        self.assertIn("## Valuation Risk / 估值风险", report)
        self.assertIn("Valuation score / 估值分数", report)
        self.assertIn("Growth reference / 增长参考", report)
        self.assertIn("## Fundamental Quality / 基本面质量", report)
        self.assertIn("Fundamental score / 基本面分数", report)
        self.assertIn("breakout_sample_quality", report)
        self.assertIn("breakout_sample_quality_zh", report)
        self.assertIn("breakout_win_rate", report)
        self.assertIn("breakout_target_hit_rate", report)
        self.assertIn("breakout_stop_hit_rate", report)
        self.assertIn("breakout_trailing_stop_hit_rate", report)
        self.assertIn("breakout_average_gain", report)
        self.assertIn("breakout_average_loss", report)
        self.assertIn("breakout_average_return", report)
        self.assertIn("pullback_win_rate", report)
        self.assertIn("pullback_sample_quality", report)
        self.assertIn("pullback_sample_quality_zh", report)
        self.assertIn("pullback_target_hit_rate", report)
        self.assertIn("pullback_stop_hit_rate", report)
        self.assertIn("pullback_trailing_stop_hit_rate", report)
        self.assertIn("pullback_average_gain", report)
        self.assertIn("pullback_average_loss", report)
        self.assertIn("pullback_average_return", report)
        self.assertIn("These labels describe analysis style, not fixed holding periods.", report)
        self.assertIn("short-term trading setup", report)
        self.assertIn("trend continuation setup", report)
        self.assertIn("long-term quality and trend setup", report)
        self.assertNotIn("0-3 months", report)
        self.assertNotIn("3-12 months", report)
        self.assertNotIn("12+ months", report)
        self.assertNotIn("position_value", report)
        self.assertNotIn("unused_cash", report)
        self.assertNotIn("capital_at_risk", report)
        self.assertNotIn("Capital assumption", report)
        self.assertIn("short", report)

    def test_render_final_next_step_is_string_not_tuple(self) -> None:
        prices = make_price_frame()
        scored = make_scored_frame(prices)
        result = analyze_ticker(scored=scored, prices=prices, ticker="AAA", horizons="short")

        report = render_ticker_analysis(result)

        prefix = "- Final next step / 下一步: "
        next_step_lines = [line for line in report.splitlines() if line.startswith(prefix)]
        self.assertEqual(len(next_step_lines), 1)
        payload = next_step_lines[0][len(prefix):]
        self.assertFalse(payload.startswith("("))
        first = result.iloc[0]
        self.assertEqual(payload, f"{first['final_next_step']} / {first['final_next_step_zh']}")

    def test_render_final_next_step_uses_position_next_step_for_holdings(self) -> None:
        prices = make_price_frame()
        scored = make_scored_frame(prices)
        result = analyze_ticker(scored=scored, prices=prices, ticker="AAA", horizons="short")
        result["position_is_held"] = True
        result["position_next_step"] = "Hold and track your stop level."
        result["position_next_step_zh"] = "继续持有并跟踪止损位。"

        report = render_ticker_analysis(result)

        self.assertIn(
            "- Final next step / 下一步: Hold and track your stop level. / 继续持有并跟踪止损位。",
            report,
        )
        self.assertIn("## Position Awareness / 持仓感知", report)

    def test_price_data_health_profile_flags_obvious_anomalies(self) -> None:
        frame = make_price_frame()
        aaa = frame[frame["ticker"] == "AAA"].head(20).copy().reset_index(drop=True)
        aaa.loc[5, "volume"] = 0
        aaa.loc[6, "adj_close"] = aaa.loc[5, "adj_close"] * 1.8
        aaa.loc[7, "high"] = aaa.loc[7, "low"] * 0.9
        aaa.loc[8, "high"] = np.nan

        profile = _price_data_health_profile(aaa)

        self.assertLess(profile["score"], 100.0)
        self.assertGreater(profile["issue_count"], 0)
        self.assertGreaterEqual(profile["zero_volume_days"], 1)
        self.assertGreaterEqual(profile["extreme_return_count"], 1)
        self.assertGreaterEqual(profile["high_low_inversion_count"], 1)
        self.assertIn("零成交量", profile["note_zh"])

    def test_backtest_trust_profile_combines_sample_liquidity_and_price_health(self) -> None:
        entry_backtest = {
            "breakout_trade_count": 30,
            "pullback_trade_count": 25,
            "breakout_average_return": 0.04,
            "pullback_average_return": 0.03,
            "backtest_avg_dollar_volume": 300_000_000.0,
            "backtest_slippage_pct": 0.001,
            "regime_coverage_score": 80.0,
            "recent_backtest_score": 75.0,
            "backtest_decay_score": 72.0,
        }
        price_health = {"score": 95.0}

        profile = _backtest_trust_profile(entry_backtest, price_health)

        self.assertGreaterEqual(profile["score"], 75.0)
        self.assertIn(profile["level"], {"usable", "high_trust"})
        self.assertGreaterEqual(profile["sample_score"], 100.0)
        self.assertEqual(profile["regime_coverage_score"], 80.0)
        self.assertEqual(profile["recent_backtest_score"], 75.0)
        self.assertEqual(profile["backtest_decay_score"], 72.0)
        self.assertIn("样本数", profile["note_zh"])
        self.assertIn("行情覆盖分", profile["note_zh"])
        self.assertIn("近期强度分", profile["note_zh"])
        self.assertIn("衰退检查分", profile["note_zh"])

    def test_regime_coverage_profile_penalizes_concentrated_samples(self) -> None:
        narrow = _regime_coverage_profile(["steady_uptrend"] * 20)
        broad = _regime_coverage_profile(
            ["steady_uptrend"] * 10 + ["strong_uptrend"] * 8 + ["volatile"] * 7
        )

        self.assertLess(narrow["score"], broad["score"])
        self.assertEqual(narrow["regime_count"], 1)
        self.assertEqual(broad["regime_count"], 3)
        self.assertIn("行情覆盖度", broad["note_zh"])

    def test_recent_backtest_profile_detects_recent_weakness(self) -> None:
        improving = _recent_backtest_profile(
            [(index, value) for index, value in enumerate([-0.02] * 10 + [0.04] * 6)]
        )
        weakening = _recent_backtest_profile(
            [(index, value) for index, value in enumerate([0.04] * 10 + [-0.02] * 6)]
        )

        self.assertGreater(improving["score"], weakening["score"])
        self.assertIn(improving["level"], {"stable", "improving"})
        self.assertIn(weakening["level"], {"weakening", "poor_recent_evidence"})
        self.assertIn("近期回测强度", improving["note_zh"])

    def test_backtest_decay_profile_detects_late_deterioration(self) -> None:
        durable = _backtest_decay_profile(
            [(index, value) for index, value in enumerate([0.02] * 8 + [0.03] * 8)]
        )
        decaying = _backtest_decay_profile(
            [(index, value) for index, value in enumerate([0.04] * 8 + [-0.03] * 8)]
        )

        self.assertGreater(durable["score"], decaying["score"])
        self.assertIn(durable["level"], {"acceptable", "durable"})
        self.assertIn(decaying["level"], {"decaying", "severe_decay"})
        self.assertLess(decaying["average_return_delta"], 0)
        self.assertIn("回测衰退检查", decaying["note_zh"])

    def test_high_event_risk_downgrades_action(self) -> None:
        prices = make_price_frame()
        scored = make_scored_frame(prices)

        result = analyze_ticker(
            scored=scored,
            prices=prices,
            ticker="AAA",
            horizons=["long"],
            event_risk_context={
                "event_risk_level": "high",
                "event_risk_level_zh": "高",
                "event_risk_score": 90.0,
                "event_window": "earnings_week",
                "event_window_zh": "财报周",
                "event_block_new_entries": True,
                "event_cooldown_active": False,
                "next_earnings_date": "2024-08-27",
                "days_until_earnings": 6,
                "last_earnings_date": "2024-05-27",
                "days_since_earnings": 92,
                "event_risk_note": "Upcoming earnings are close.",
                "event_risk_note_zh": "财报日期较近。",
            },
        )

        self.assertEqual(result.iloc[0]["action"], "wait_event_risk")
        self.assertEqual(result.iloc[0]["entry_type"], "deferred_event_risk_filter")
        self.assertFalse(bool(result.iloc[0]["quality_gate_passed"]))
        self.assertIn("event risk high", result.iloc[0]["quality_gate_fail_reasons"])
        self.assertIn("财报", result.iloc[0]["plain_summary_zh"])

    def test_event_block_downgrades_action_and_quality_gate(self) -> None:
        prices = make_price_frame()
        scored = make_scored_frame(prices)

        result = analyze_ticker(
            scored=scored,
            prices=prices,
            ticker="AAA",
            horizons=["long"],
            event_risk_context={
                "event_risk_level": "medium",
                "event_risk_level_zh": "中",
                "event_risk_score": 65.0,
                "event_window": "post_earnings_cooldown",
                "event_window_zh": "财报后冷却期",
                "event_block_new_entries": True,
                "event_cooldown_active": True,
                "next_earnings_date": "2024-11-27",
                "days_until_earnings": 92,
                "last_earnings_date": "2024-08-27",
                "days_since_earnings": 2,
                "event_risk_note": "Post-earnings cooldown is active.",
                "event_risk_note_zh": "财报后冷却期生效。",
            },
        )

        self.assertEqual(result.iloc[0]["action"], "wait_event_risk")
        self.assertEqual(result.iloc[0]["event_window"], "post_earnings_cooldown")
        self.assertTrue(bool(result.iloc[0]["event_block_new_entries"]))
        self.assertFalse(bool(result.iloc[0]["quality_gate_passed"]))
        self.assertIn("event window blocks new entries", result.iloc[0]["quality_gate_fail_reasons"])
        self.assertIn("事件窗口阻止新入场", result.iloc[0]["quality_gate_fail_reasons_zh"])

    def test_high_news_sentiment_risk_downgrades_action(self) -> None:
        prices = make_price_frame()
        scored = make_scored_frame(prices)

        result = analyze_ticker(
            scored=scored,
            prices=prices,
            ticker="AAA",
            horizons=["long"],
            sentiment_context={
                "sentiment_score": 20.0,
                "sentiment_label": "negative",
                "sentiment_label_zh": "负面",
                "sentiment_risk_level": "high",
                "sentiment_risk_level_zh": "高",
                "sentiment_block_new_entries": True,
                "sentiment_positive_count": 0,
                "sentiment_negative_count": 3,
                "sentiment_high_risk_count": 2,
                "sentiment_titles_used": 4,
                "sentiment_note": "High-risk news is present.",
                "sentiment_note_zh": "存在高风险新闻。",
            },
        )

        self.assertEqual(result.iloc[0]["action"], "wait_sentiment_risk")
        self.assertEqual(result.iloc[0]["entry_type"], "deferred_sentiment_filter")
        self.assertFalse(bool(result.iloc[0]["quality_gate_passed"]))
        self.assertIn("news sentiment risk high", result.iloc[0]["quality_gate_fail_reasons"])
        self.assertIn("新闻情绪风险过高", result.iloc[0]["quality_gate_fail_reasons_zh"])

    def test_high_analyst_risk_downgrades_action(self) -> None:
        prices = make_price_frame()
        scored = make_scored_frame(prices)

        result = analyze_ticker(
            scored=scored,
            prices=prices,
            ticker="AAA",
            horizons=["long"],
            analyst_context={
                "analyst_score": 22.0,
                "analyst_label": "negative",
                "analyst_label_zh": "负面",
                "analyst_risk_level": "high",
                "analyst_risk_level_zh": "高",
                "analyst_block_new_entries": True,
                "analyst_upside": -0.18,
                "recommendation_mean": 3.8,
                "recommendation_key": "sell",
                "number_of_analysts": 12,
                "target_mean_price": 70.0,
                "analyst_note": "Analyst expectations are weak.",
                "analyst_note_zh": "分析师预期偏弱。",
            },
        )

        self.assertEqual(result.iloc[0]["action"], "wait_analyst_weak")
        self.assertEqual(result.iloc[0]["entry_type"], "deferred_analyst_filter")
        self.assertFalse(bool(result.iloc[0]["quality_gate_passed"]))
        self.assertIn("analyst expectation risk high", result.iloc[0]["quality_gate_fail_reasons"])
        self.assertIn("分析师预期风险过高", result.iloc[0]["quality_gate_fail_reasons_zh"])

    def test_high_valuation_risk_downgrades_action(self) -> None:
        prices = make_price_frame()
        scored = make_scored_frame(prices)

        result = analyze_ticker(
            scored=scored,
            prices=prices,
            ticker="AAA",
            horizons=["long"],
            valuation_context={
                "valuation_score": 24.0,
                "valuation_label": "expensive",
                "valuation_label_zh": "偏贵",
                "valuation_risk_level": "high",
                "valuation_risk_level_zh": "高",
                "valuation_block_new_entries": True,
                "valuation_forward_pe": 75.0,
                "valuation_peg_ratio": 4.5,
                "valuation_growth_reference": 0.03,
                "valuation_profit_margin": 0.08,
                "valuation_note": "Valuation is rich.",
                "valuation_note_zh": "估值偏贵。",
            },
        )

        self.assertEqual(result.iloc[0]["action"], "wait_valuation_rich")
        self.assertEqual(result.iloc[0]["entry_type"], "deferred_valuation_filter")
        self.assertFalse(bool(result.iloc[0]["quality_gate_passed"]))
        self.assertIn("valuation risk high", result.iloc[0]["quality_gate_fail_reasons"])
        self.assertIn("估值风险过高", result.iloc[0]["quality_gate_fail_reasons_zh"])

    def test_high_probability_gate_passes_strict_quality_row(self) -> None:
        row = make_high_probability_gate_row()

        result = _high_probability_screening_for_row(row)

        self.assertTrue(result["quality_gate_passed"])
        self.assertEqual(result["screening_action"], "high_probability_watchlist_candidate")
        self.assertEqual(result["quality_gate_fail_reasons"], "all strict quality gates passed")
        self.assertGreaterEqual(result["high_probability_score"], 60)
        self.assertGreaterEqual(result["calibrated_win_probability"], 0.55)
        self.assertIn(result["calibrated_probability_level"], {"constructive", "high"})
        self.assertGreaterEqual(result["calibrated_probability_confidence"], 55)
        self.assertTrue(result["entry_readiness_gate_passed"])
        self.assertTrue(result["trade_plan_quality_gate_passed"])

        watchlist = _watchlist_plan_for_row(pd.Series({**row.to_dict(), **result}))
        self.assertEqual(watchlist["watchlist_status"], "ready_high_probability")
        self.assertEqual(watchlist["watchlist_missing_items"], "none")

    def test_entry_evidence_profile_marks_tiny_sample_as_weak(self) -> None:
        result = _entry_evidence_profile(
            trade_count=1,
            win_rate=1.0,
            average_return=0.20,
        )

        self.assertEqual(result["sample_confidence_level"], "very_weak")
        self.assertEqual(result["sample_confidence_level_zh"], "很弱")
        self.assertEqual(result["evidence_strength"], "weak")
        self.assertIn("sample=1", result["evidence_note"])
        self.assertIn("样本=1", result["evidence_note_zh"])
        self.assertIn("样本太少", result["evidence_note_zh"])

    def test_high_probability_gate_blocks_low_win_rate(self) -> None:
        row = make_high_probability_gate_row()
        row["breakout_win_rate"] = 0.40

        result = _high_probability_screening_for_row(row)
        watchlist = _watchlist_plan_for_row(pd.Series({**row.to_dict(), **result}))

        self.assertFalse(result["quality_gate_passed"])
        self.assertEqual(result["screening_action"], "not_high_probability_setup_now")
        self.assertIn("backtest win rate too low", result["quality_gate_fail_reasons"])
        self.assertIn("回测胜率不足", result["quality_gate_fail_reasons_zh"])
        self.assertIn("entry backtest win rate", watchlist["watchlist_missing_items"])

    def test_calibrated_win_probability_penalizes_weak_entry_evidence(self) -> None:
        row = make_high_probability_gate_row()
        strong = _high_probability_screening_for_row(row)
        # Weaken BOTH entry-type buckets: with the pooled-evidence fallback,
        # bad stats in one under-sampled bucket are averaged against the other
        # bucket's healthy samples instead of dominating the judgement.
        row["breakout_trade_count"] = 4
        row["breakout_win_rate"] = 0.35
        row["breakout_stop_hit_rate"] = 0.70
        row["breakout_average_return"] = -0.04
        row["pullback_trade_count"] = 5
        row["pullback_win_rate"] = 0.40
        row["pullback_stop_hit_rate"] = 0.62
        row["pullback_average_return"] = -0.015

        weak = _high_probability_screening_for_row(row)

        self.assertLess(
            weak["calibrated_win_probability"],
            strong["calibrated_win_probability"],
        )
        self.assertLess(
            weak["calibrated_probability_confidence"],
            strong["calibrated_probability_confidence"],
        )
        self.assertIn(weak["calibrated_probability_level"], {"low", "neutral"})

    def test_high_probability_gate_blocks_when_entry_is_not_executable(self) -> None:
        row = make_high_probability_gate_row()
        row["chase_status"] = "do_not_chase_wait_for_breakout"
        row["entry_distance_pct"] = 0.08

        result = _high_probability_screening_for_row(row)
        watchlist = _watchlist_plan_for_row(pd.Series({**row.to_dict(), **result}))

        self.assertFalse(result["quality_gate_passed"])
        self.assertFalse(result["entry_readiness_gate_passed"])
        self.assertIn("entry trigger not reached", result["quality_gate_fail_reasons"])
        self.assertIn("买点触发尚未到达", result["quality_gate_fail_reasons_zh"])
        self.assertIn("entry is executable now", watchlist["watchlist_missing_items"])

    def test_high_probability_gate_blocks_low_quality_trade_plan(self) -> None:
        row = make_high_probability_gate_row()
        row["risk_reward"] = 1.2
        row["take_profit"] = 107.2

        result = _high_probability_screening_for_row(row)
        watchlist = _watchlist_plan_for_row(pd.Series({**row.to_dict(), **result}))

        self.assertFalse(result["quality_gate_passed"])
        self.assertFalse(result["trade_plan_quality_gate_passed"])
        self.assertIn("risk-reward too low", result["quality_gate_fail_reasons"])
        self.assertIn("盈亏比不足", result["quality_gate_fail_reasons_zh"])
        self.assertIn("trade plan quality passed", watchlist["watchlist_missing_items"])

    def test_threshold_calibration_tightens_when_backtest_is_weak(self) -> None:
        row = make_high_probability_gate_row()
        row["breakout_win_rate"] = 0.40
        screening = _high_probability_screening_for_row(row)

        result = _threshold_calibration_for_row(
            pd.Series({**row.to_dict(), **screening}),
            ScreeningThresholds(),
        )

        self.assertEqual(result["calibration_action"], "tighten_thresholds")
        self.assertGreater(result["recommended_signal_threshold"], 65.0)
        self.assertGreater(result["recommended_backtest_win_rate_min"], 0.55)
        self.assertIn("提高门槛", result["calibration_action_zh"])

    def test_threshold_calibration_tightens_in_weak_market_regime(self) -> None:
        row = make_high_probability_gate_row()
        row["market_score"] = 35.0
        row["market_status"] = "weak"
        screening = _high_probability_screening_for_row(row)

        result = _threshold_calibration_for_row(
            pd.Series({**row.to_dict(), **screening}),
            ScreeningThresholds(),
        )

        self.assertEqual(result["market_regime"], "weak")
        self.assertGreaterEqual(result["recommended_signal_threshold"], 69.0)
        self.assertGreaterEqual(result["recommended_confidence_threshold"], 68.0)
        self.assertGreaterEqual(result["recommended_backtest_sample_min"], 15)
        self.assertGreater(result["recommended_backtest_win_rate_min"], 0.55)
        self.assertIn("弱势大盘", result["market_regime_note_zh"])

    def test_calibrated_screening_uses_recommended_thresholds(self) -> None:
        row = make_high_probability_gate_row()
        screening = _high_probability_screening_for_row(row)
        calibrated_row = pd.Series(
            {
                **row.to_dict(),
                **screening,
                "recommended_signal_threshold": 65.0,
                "recommended_confidence_threshold": 65.0,
                # Above the POOLED breakout+pullback total (18+12=30) so the
                # calibrated sample gate still flips even with pooled evidence.
                "recommended_backtest_sample_min": 40,
                "recommended_backtest_win_rate_min": 0.55,
                "recommended_backtest_average_return_min": 0.0,
            }
        )

        result = _calibrated_screening_for_row(calibrated_row, ScreeningThresholds())

        self.assertFalse(result["calibrated_quality_gate_passed"])
        self.assertEqual(
            result["calibrated_screening_action"],
            "not_calibrated_high_probability_now",
        )
        self.assertIn(
            "backtest sample too small",
            result["calibrated_quality_gate_fail_reasons"],
        )

    def test_calibrated_watchlist_uses_recommended_thresholds(self) -> None:
        row = make_high_probability_gate_row()
        screening = _high_probability_screening_for_row(row)
        calibrated_screening = _calibrated_screening_for_row(
            pd.Series(
                {
                    **row.to_dict(),
                    **screening,
                    "recommended_signal_threshold": 65.0,
                    "recommended_confidence_threshold": 65.0,
                    "recommended_backtest_sample_min": 25,
                    "recommended_backtest_win_rate_min": 0.55,
                    "recommended_backtest_average_return_min": 0.0,
                }
            ),
            ScreeningThresholds(),
        )
        full_row = pd.Series(
            {
                **row.to_dict(),
                **screening,
                **calibrated_screening,
                "recommended_signal_threshold": 65.0,
                "recommended_confidence_threshold": 65.0,
                "recommended_backtest_sample_min": 25,
                "recommended_backtest_win_rate_min": 0.55,
                "recommended_backtest_average_return_min": 0.0,
            }
        )

        result = _calibrated_watchlist_plan_for_row(full_row, ScreeningThresholds())

        self.assertIn(
            "entry backtest sample >= 25",
            result["calibrated_watchlist_missing_items"],
        )
        self.assertIn(
            "买点回测样本至少25笔",
            result["calibrated_watchlist_missing_items_zh"],
        )

    def test_high_probability_gate_blocks_low_liquidity(self) -> None:
        row = make_high_probability_gate_row()
        row["backtest_avg_dollar_volume"] = 2_000_000.0
        row["backtest_slippage_pct"] = 0.009
        row["backtest_liquidity_label"] = "very_low_liquidity"
        row["backtest_liquidity_label_zh"] = "流动性很低"

        result = _high_probability_screening_for_row(row)
        watchlist = _watchlist_plan_for_row(pd.Series({**row.to_dict(), **result}))

        self.assertFalse(result["quality_gate_passed"])
        self.assertFalse(result["liquidity_filter_passed"])
        self.assertIn("liquidity filter failed", result["quality_gate_fail_reasons"])
        self.assertIn("流动性过滤未通过", result["quality_gate_fail_reasons_zh"])
        self.assertIn("liquidity filter passed", watchlist["watchlist_missing_items"])

    def test_high_probability_gate_blocks_low_backtest_trust(self) -> None:
        row = make_high_probability_gate_row()
        row["backtest_trust_score"] = 45.0
        row["backtest_trust_level"] = "low_trust"
        row["backtest_trust_level_zh"] = "可信度偏低"

        result = _high_probability_screening_for_row(row)
        watchlist = _watchlist_plan_for_row(pd.Series({**row.to_dict(), **result}))

        self.assertFalse(result["quality_gate_passed"])
        self.assertIn("backtest trust score too low", result["quality_gate_fail_reasons"])
        self.assertIn("回测可信度分不足", result["quality_gate_fail_reasons_zh"])
        self.assertIn("backtest trust score >= 60", watchlist["watchlist_missing_items"])

    def test_high_probability_gate_blocks_weak_recent_backtest(self) -> None:
        row = make_high_probability_gate_row()
        row["recent_backtest_score"] = 30.0
        row["recent_backtest_level"] = "poor_recent_evidence"
        row["recent_backtest_level_zh"] = "近期证据较差"

        result = _high_probability_screening_for_row(row)
        watchlist = _watchlist_plan_for_row(pd.Series({**row.to_dict(), **result}))

        self.assertFalse(result["quality_gate_passed"])
        self.assertIn("recent backtest strength too weak", result["quality_gate_fail_reasons"])
        self.assertIn("近期回测强度不足", result["quality_gate_fail_reasons_zh"])
        self.assertIn("recent backtest score >= 45", watchlist["watchlist_missing_items"])

    def test_high_probability_gate_blocks_backtest_decay(self) -> None:
        row = make_high_probability_gate_row()
        row["backtest_decay_score"] = 25.0
        row["backtest_decay_level"] = "severe_decay"
        row["backtest_decay_level_zh"] = "明显衰退"

        result = _high_probability_screening_for_row(row)
        watchlist = _watchlist_plan_for_row(pd.Series({**row.to_dict(), **result}))

        self.assertFalse(result["quality_gate_passed"])
        self.assertIn("backtest decay check failed", result["quality_gate_fail_reasons"])
        self.assertIn("回测衰退检查未通过", result["quality_gate_fail_reasons_zh"])
        self.assertIn("backtest decay score >= 45", watchlist["watchlist_missing_items"])

    def test_high_probability_gate_blocks_high_stop_hit_rate(self) -> None:
        row = make_high_probability_gate_row()
        row["breakout_stop_hit_rate"] = 0.68

        result = _high_probability_screening_for_row(row)
        watchlist = _watchlist_plan_for_row(pd.Series({**row.to_dict(), **result}))

        self.assertFalse(result["quality_gate_passed"])
        self.assertEqual(result["screening_backtest_stop_hit_rate"], 0.68)
        self.assertIn(
            "breakout backtest stop-hit rate too high",
            result["quality_gate_fail_reasons"],
        )
        self.assertIn("突破买点回测止损命中率过高", result["quality_gate_fail_reasons_zh"])
        self.assertIn(
            "entry backtest stop-hit rate <= 50%",
            watchlist["watchlist_missing_items"],
        )

    def test_high_probability_gate_blocks_event_window(self) -> None:
        row = make_high_probability_gate_row()
        row["event_risk_level"] = "medium"
        row["event_block_new_entries"] = True

        result = _high_probability_screening_for_row(row)
        watchlist = _watchlist_plan_for_row(pd.Series({**row.to_dict(), **result}))

        self.assertFalse(result["quality_gate_passed"])
        self.assertIn("event window blocks new entries", result["quality_gate_fail_reasons"])
        self.assertIn("事件窗口阻止新入场", result["quality_gate_fail_reasons_zh"])
        self.assertIn("event risk does not block new entries", watchlist["watchlist_missing_items"])

    def test_high_probability_gate_blocks_news_sentiment(self) -> None:
        row = make_high_probability_gate_row()
        row["sentiment_risk_level"] = "high"
        row["sentiment_block_new_entries"] = True

        result = _high_probability_screening_for_row(row)
        watchlist = _watchlist_plan_for_row(pd.Series({**row.to_dict(), **result}))

        self.assertFalse(result["quality_gate_passed"])
        self.assertIn("news sentiment risk high", result["quality_gate_fail_reasons"])
        self.assertIn("新闻情绪风险过高", result["quality_gate_fail_reasons_zh"])
        self.assertIn("news sentiment does not block new entries", watchlist["watchlist_missing_items"])

    def test_high_probability_gate_blocks_analyst_risk(self) -> None:
        row = make_high_probability_gate_row()
        row["analyst_risk_level"] = "high"
        row["analyst_block_new_entries"] = True

        result = _high_probability_screening_for_row(row)
        watchlist = _watchlist_plan_for_row(pd.Series({**row.to_dict(), **result}))

        self.assertFalse(result["quality_gate_passed"])
        self.assertIn("analyst expectation risk high", result["quality_gate_fail_reasons"])
        self.assertIn("分析师预期风险过高", result["quality_gate_fail_reasons_zh"])
        self.assertIn("analyst expectations do not block new entries", watchlist["watchlist_missing_items"])

    def test_high_probability_gate_blocks_valuation_risk(self) -> None:
        row = make_high_probability_gate_row()
        row["valuation_risk_level"] = "high"
        row["valuation_block_new_entries"] = True

        result = _high_probability_screening_for_row(row)
        watchlist = _watchlist_plan_for_row(pd.Series({**row.to_dict(), **result}))

        self.assertFalse(result["quality_gate_passed"])
        self.assertIn("valuation risk high", result["quality_gate_fail_reasons"])
        self.assertIn("估值风险过高", result["quality_gate_fail_reasons_zh"])
        self.assertIn("valuation does not block new entries", watchlist["watchlist_missing_items"])

    def test_high_probability_gate_uses_custom_thresholds(self) -> None:
        row = make_high_probability_gate_row()
        row["breakout_win_rate"] = 0.40
        thresholds = ScreeningThresholds(backtest_win_rate_min=0.35)

        result = _high_probability_screening_for_row(row, thresholds)
        watchlist = _watchlist_plan_for_row(pd.Series({**row.to_dict(), **result}), thresholds)

        self.assertTrue(result["quality_gate_passed"])
        self.assertEqual(watchlist["watchlist_status"], "ready_high_probability")

    def test_weak_fundamentals_downgrade_long_action(self) -> None:
        prices = make_price_frame()
        scored = make_scored_frame(prices)

        result = analyze_ticker(
            scored=scored,
            prices=prices,
            ticker="AAA",
            horizons=["long"],
            fundamental_context={
                "fundamental_score": 25.0,
                "fundamental_quality": "weak",
                "fundamental_quality_zh": "弱",
                "fundamental_note": "Fundamentals are weak.",
                "fundamental_note_zh": "基本面偏弱。",
            },
        )

        self.assertEqual(result.iloc[0]["action"], "wait_fundamental_weak")
        self.assertEqual(result.iloc[0]["entry_type"], "deferred_fundamental_filter")
        self.assertIn("公司质量", result.iloc[0]["plain_summary_zh"])

    def test_analyze_ticker_omits_capital_fields(self) -> None:
        prices = make_price_frame()
        scored = make_scored_frame(prices)

        result = analyze_ticker(
            scored=scored,
            prices=prices,
            ticker="AAA",
            horizons=["short"],
        )

        for column in [
            "capital",
            "risk_budget",
            "position_shares",
            "position_value",
            "unused_cash",
            "capital_at_risk",
        ]:
            self.assertNotIn(column, result.columns)

    def test_analyze_ticker_applies_profile_trading_rules(self) -> None:
        prices = make_price_frame()
        scored = make_scored_frame(prices)
        rules = TradingRules(
            preferred_entry_style="pullback",
            preferred_entry_style_zh="回调优先",
            short_atr_stop_multiple=1.8,
            short_target_r_multiple=2.7,
            short_max_chase_pct=0.02,
            short_time_stop_days=12,
            sell_rule="custom_pullback_exit",
            sell_rule_zh="自定义回调退出规则",
        )

        result = analyze_ticker(
            scored=scored,
            prices=prices,
            ticker="AAA",
            horizons=["short"],
            trading_rules=rules,
        )
        row = result.iloc[0]

        self.assertEqual(row["trading_rule_entry_style"], "pullback")
        self.assertEqual(row["trading_rule_entry_style_zh"], "回调优先")
        self.assertEqual(row["trading_rule_atr_stop_multiple"], 1.8)
        self.assertEqual(row["trading_rule_target_r_multiple"], 2.7)
        self.assertEqual(row["trading_rule_max_chase_pct"], 0.02)
        self.assertEqual(row["trading_rule_time_stop_days"], 12)
        self.assertEqual(row["backtest_time_stop_days"], 12)
        self.assertEqual(row["backtest_trailing_stop_trigger_r"], 1.5)
        self.assertEqual(row["backtest_trailing_stop_lock_r"], 0.2)
        self.assertEqual(row["trading_rule_sell_rule"], "custom_pullback_exit")
        self.assertEqual(row["trading_rule_sell_rule_zh"], "自定义回调退出规则")
        self.assertAlmostEqual(row["risk_reward"], 2.7)
        self.assertIn(row["entry_type"], {"limit_pullback", "none"})

    def test_analyze_ticker_entry_backtest_counts_are_non_negative(self) -> None:
        prices = make_price_frame()
        scored = make_scored_frame(prices)

        result = analyze_ticker(scored=scored, prices=prices, ticker="AAA", horizons="all")

        self.assertTrue((result["breakout_trade_count"] >= 0).all())
        self.assertTrue((result["pullback_trade_count"] >= 0).all())
        quality_values = {"strong", "moderate", "weak", "insufficient"}
        self.assertTrue(set(result["breakout_sample_quality"]).issubset(quality_values))
        self.assertTrue(set(result["pullback_sample_quality"]).issubset(quality_values))
        for column in [
            "breakout_win_rate",
            "breakout_target_hit_rate",
            "breakout_stop_hit_rate",
            "breakout_trailing_stop_hit_rate",
            "pullback_win_rate",
            "pullback_target_hit_rate",
            "pullback_stop_hit_rate",
            "pullback_trailing_stop_hit_rate",
        ]:
            values = result[column].dropna()
            self.assertTrue(((values >= 0) & (values <= 1)).all())
        for column in ["breakout_average_gain", "pullback_average_gain"]:
            values = result[column].dropna()
            self.assertTrue((values > 0).all())
        for column in ["breakout_average_loss", "pullback_average_loss"]:
            values = result[column].dropna()
            self.assertTrue((values < 0).all())
        for column in ["breakout_average_return", "pullback_average_return"]:
            values = result[column].dropna()
            self.assertTrue(np.isfinite(values).all())

    def test_trade_result_uses_gap_stop_open_instead_of_ideal_stop(self) -> None:
        outcome, trade_return = _trade_result(
            entry_price=100.0,
            stop_loss=95.0,
            target_r_multiple=2.0,
            future_open=pd.Series([90.0]),
            future_high=pd.Series([93.0]),
            future_low=pd.Series([89.0]),
            future_close=pd.Series([92.0]),
            slippage_pct=0.001,
        )

        self.assertEqual(outcome, "gap_stop_hit")
        self.assertLess(trade_return, -0.05)

    def test_trade_result_applies_trailing_stop_after_trigger(self) -> None:
        outcome, trade_return = _trade_result(
            entry_price=100.0,
            stop_loss=95.0,
            target_r_multiple=3.0,
            future_open=pd.Series([101.0, 102.0]),
            future_high=pd.Series([106.0, 103.0]),
            future_low=pd.Series([100.0, 100.5]),
            future_close=pd.Series([105.0, 101.0]),
            slippage_pct=0.0,
            trailing_stop_trigger_r=1.0,
            trailing_stop_lock_r=0.2,
        )

        self.assertEqual(outcome, "trailing_stop_win")
        self.assertAlmostEqual(trade_return, 0.01)

    def test_dynamic_slippage_increases_for_low_liquidity(self) -> None:
        close = pd.Series([100.0] * 40)
        high = pd.Series([101.0] * 40)
        low = pd.Series([99.0] * 40)
        high_volume = pd.Series([20_000_000.0] * 40)
        low_volume = pd.Series([20_000.0] * 40)

        high_liquidity = _dynamic_slippage_profile(
            close=close,
            high=high,
            low=low,
            volume=high_volume,
            spec=HORIZON_SPECS["short"],
        )
        low_liquidity = _dynamic_slippage_profile(
            close=close,
            high=high,
            low=low,
            volume=low_volume,
            spec=HORIZON_SPECS["short"],
        )

        self.assertLess(high_liquidity["slippage_pct"], low_liquidity["slippage_pct"])
        self.assertEqual(high_liquidity["liquidity_label"], "very_high_liquidity")
        self.assertEqual(low_liquidity["liquidity_label"], "very_low_liquidity")


if __name__ == "__main__":
    unittest.main()
