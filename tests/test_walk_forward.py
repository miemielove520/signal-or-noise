from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from stock_selector.screening_config import default_screening_config
from stock_selector.universe import HistoricalUniverseMembership
from stock_selector.walk_forward import (
    _apply_pooled_entry_backtest,
    build_validation_market_regime_lookup,
    build_benchmark_aware_policy,
    build_benchmark_tightening_recommendations,
    build_minimum_sample_guard,
    build_sample_sufficiency_guidance,
    build_ticker_validation_ranking,
    build_threshold_sensitivity_grid,
    build_tightening_impact_validation,
    build_walk_forward_benchmark_comparison,
    build_walk_forward_portfolio_equity,
    calibrate_walk_forward_profiles,
    calibrate_walk_forward_rules,
    render_walk_forward_report,
    render_suggested_screening_config,
    run_walk_forward_validation,
    summarize_walk_forward_segments,
    summarize_probability_calibration,
    summarize_walk_forward_portfolios,
    summarize_walk_forward_events,
    summarize_walk_forward_profiles,
    summarize_market_regime_validation,
    build_market_regime_protection_policy,
    build_overfitting_risk_report,
    MIN_CALIBRATION_SAMPLE_COUNT,
)


class OverfittingRiskTest(unittest.TestCase):
    def test_conservative_default_threshold(self) -> None:
        # A trading-parameter calibration floor of 30 (not 5/10) for a live-money system.
        self.assertGreaterEqual(MIN_CALIBRATION_SAMPLE_COUNT, 30)

    def test_many_params_few_samples_is_high_risk(self) -> None:
        calibration = pd.DataFrame(
            [
                {"screening_profile": "saas_software", "rule": r, "sample_count": 6}
                for r in ["signal_score", "high_probability_score", "relative_strength_score"]
            ]
        )
        report = build_overfitting_risk_report(calibration)
        row = report[report["screening_profile"] == "saas_software"].iloc[0]
        self.assertEqual(row["overfit_risk_level"], "high")
        self.assertEqual(int(row["tunable_param_count"]), 3)
        self.assertEqual(int(row["adoptable_param_count"]), 0)  # none reach 30

    def test_enough_samples_is_low_risk(self) -> None:
        calibration = pd.DataFrame(
            [
                {"screening_profile": "mega_cap_tech", "rule": r, "sample_count": 45}
                for r in ["signal_score", "high_probability_score"]
            ]
        )
        report = build_overfitting_risk_report(calibration)
        row = report[report["screening_profile"] == "mega_cap_tech"].iloc[0]
        self.assertEqual(row["overfit_risk_level"], "low")
        self.assertEqual(int(row["adoptable_param_count"]), 2)

    def test_empty_calibration(self) -> None:
        self.assertTrue(build_overfitting_risk_report(pd.DataFrame()).empty)

    def test_low_sample_suggestion_blocked_by_default_gate(self) -> None:
        # 12 samples used to be written (gate was 5); now the default gate is 30 -> skipped.
        events = pd.DataFrame(
            [
                {
                    "screening_profile": "saas_software",
                    "screening_profile_zh": "SaaS软件规则",
                    "signal_score": 72.0,
                    "high_probability_score": 68.0,
                    "relative_strength_score": 54.0,
                    "market_score": 60.0,
                    "screening_backtest_win_rate": 0.6,
                    "forward_return_20d": 0.05,
                }
                for _ in range(12)
            ]
        )
        calibration = calibrate_walk_forward_profiles(events, target_window=20, min_sample_count=5)
        text = render_suggested_screening_config(
            screening_config=default_screening_config(),
            profile_calibration=calibration,
        )
        # With the conservative default, a 12-sample calibration is not adopted.
        self.assertIn("skipped", text.lower())


class WalkForwardTest(unittest.TestCase):
    def test_run_walk_forward_validation_outputs_events_summary_and_calibration(self) -> None:
        prices = make_price_frame()

        with tempfile.TemporaryDirectory() as directory:
            result = run_walk_forward_validation(
                prices=prices,
                tickers=["AAA"],
                forward_windows=(5, 20),
                step_days=30,
                min_history_days=120,
                output_dir=directory,
            )

            self.assertFalse(result.events.empty)
            self.assertFalse(result.summary.empty)
            self.assertFalse(result.ticker_ranking.empty)
            self.assertFalse(result.sample_sufficiency.empty)
            self.assertFalse(result.profile_summary.empty)
            self.assertFalse(result.segment_summary.empty)
            self.assertFalse(result.market_regime_summary.empty)
            self.assertFalse(result.market_regime_policy.empty)
            self.assertFalse(result.probability_calibration.empty)
            self.assertFalse(result.portfolio_summary.empty)
            self.assertFalse(result.portfolio_rebalances.empty)
            self.assertFalse(result.portfolio_equity_summary.empty)
            self.assertFalse(result.portfolio_equity_curve.empty)
            self.assertFalse(result.profile_calibration.empty)
            self.assertFalse(result.calibration.empty)
            self.assertIn("overall", result.win_rate_dashboard)
            self.assertFalse(result.win_rate_dashboard["overall"].empty)
            self.assertFalse(result.profile_health_dashboard.empty)
            self.assertFalse(result.profile_action_recommendations.empty)
            self.assertFalse(result.profile_blocker_dashboard.empty)
            self.assertFalse(result.historical_win_rate_gate.empty)
            self.assertFalse(result.historical_threshold_recommendations.empty)
            self.assertIn("validation_bucket", result.events.columns)
            self.assertIn("horizon_zh_label", result.events.columns)
            self.assertIn("screening_profile", result.events.columns)
            self.assertIn("calibrated_win_probability", result.events.columns)
            self.assertIn("calibrated_probability_confidence", result.events.columns)
            self.assertIn("forward_return_20d", result.events.columns)
            self.assertIn("screening_backtest_average_return", result.events.columns)
            self.assertIn("sample_confidence_level", result.events.columns)
            self.assertIn("evidence_strength", result.events.columns)
            self.assertIn("sentiment_score", result.events.columns)
            self.assertIn("sentiment_risk_level", result.events.columns)
            self.assertIn("sentiment_block_new_entries", result.events.columns)
            self.assertIn("analyst_score", result.events.columns)
            self.assertIn("analyst_risk_level", result.events.columns)
            self.assertIn("analyst_block_new_entries", result.events.columns)
            self.assertIn("valuation_score", result.events.columns)
            self.assertIn("valuation_risk_level", result.events.columns)
            self.assertIn("valuation_block_new_entries", result.events.columns)
            self.assertIn("quality_gate_fail_reasons_zh", result.events.columns)
            self.assertIn("watchlist_missing_items_zh", result.events.columns)
            self.assertIn("validation_market_regime", result.events.columns)
            self.assertIn("validation_market_regime_zh", result.events.columns)
            self.assertIn("ticker_ranking_score", result.ticker_ranking.columns)
            self.assertIn("ticker_decision_zh", result.ticker_ranking.columns)
            self.assertIn("sample_status_zh", result.sample_sufficiency.columns)
            self.assertIn("suggested_command_hint", result.sample_sufficiency.columns)
            self.assertTrue((Path(directory) / "walk_forward_events.csv").exists())
            self.assertTrue((Path(directory) / "ticker_validation_ranking.csv").exists())
            self.assertTrue((Path(directory) / "sample_sufficiency_guidance.csv").exists())
            self.assertTrue((Path(directory) / "profile_validation_summary.csv").exists())
            self.assertTrue((Path(directory) / "segment_validation_summary.csv").exists())
            self.assertTrue((Path(directory) / "market_regime_validation_summary.csv").exists())
            self.assertTrue((Path(directory) / "market_regime_policy.csv").exists())
            self.assertTrue((Path(directory) / "probability_calibration.csv").exists())
            self.assertTrue((Path(directory) / "portfolio_validation_summary.csv").exists())
            self.assertTrue((Path(directory) / "portfolio_rebalances.csv").exists())
            self.assertTrue((Path(directory) / "portfolio_equity_summary.csv").exists())
            self.assertTrue((Path(directory) / "portfolio_equity_curve.csv").exists())
            self.assertTrue((Path(directory) / "benchmark_comparison_summary.csv").exists())
            self.assertTrue((Path(directory) / "benchmark_comparison_curve.csv").exists())
            self.assertTrue((Path(directory) / "benchmark_policy.csv").exists())
            self.assertTrue((Path(directory) / "benchmark_tightening_recommendations.csv").exists())
            self.assertTrue((Path(directory) / "tightening_impact_validation.csv").exists())
            self.assertTrue((Path(directory) / "threshold_sensitivity_grid.csv").exists())
            self.assertTrue((Path(directory) / "minimum_sample_guard.csv").exists())
            self.assertTrue((Path(directory) / "win_rate_dashboard.md").exists())
            self.assertTrue((Path(directory) / "win_rate_dashboard.json").exists())
            self.assertTrue((Path(directory) / "win_rate_overall.csv").exists())
            self.assertTrue((Path(directory) / "win_rate_by_horizon.csv").exists())
            self.assertTrue((Path(directory) / "win_rate_by_entry_type.csv").exists())
            self.assertTrue((Path(directory) / "win_rate_by_profile.csv").exists())
            self.assertTrue((Path(directory) / "win_rate_by_quality_gate.csv").exists())
            self.assertTrue((Path(directory) / "profile_health_dashboard.md").exists())
            self.assertTrue((Path(directory) / "profile_health_dashboard.json").exists())
            self.assertTrue((Path(directory) / "profile_health_dashboard.csv").exists())
            self.assertTrue((Path(directory) / "profile_action_recommendations.md").exists())
            self.assertTrue((Path(directory) / "profile_action_recommendations.json").exists())
            self.assertTrue((Path(directory) / "profile_action_recommendations.csv").exists())
            self.assertTrue((Path(directory) / "profile_blocker_dashboard.md").exists())
            self.assertTrue((Path(directory) / "profile_blocker_dashboard.json").exists())
            self.assertTrue((Path(directory) / "profile_blocker_dashboard.csv").exists())
            self.assertTrue((Path(directory) / "historical_win_rate_gate.md").exists())
            self.assertTrue((Path(directory) / "historical_win_rate_gate.json").exists())
            self.assertTrue((Path(directory) / "historical_win_rate_gate.csv").exists())
            self.assertTrue(
                (Path(directory) / "historical_threshold_recommendations.md").exists()
            )
            self.assertTrue(
                (Path(directory) / "historical_threshold_recommendations.json").exists()
            )
            self.assertTrue(
                (Path(directory) / "historical_threshold_recommendations.csv").exists()
            )
            self.assertTrue((Path(directory) / "profile_rule_calibration.csv").exists())
            self.assertTrue((Path(directory) / "suggested_screening.toml").exists())
            self.assertTrue((Path(directory) / "walk_forward_report.md").exists())
            self.assertTrue((Path(directory) / "validation_result.json").exists())
            self.assertIn("Walk-Forward Validation / 滚动历史验证", result.report)
            self.assertIn("Ticker Validation Ranking / 个股验证排名", result.report)
            self.assertIn("Sample Sufficiency Guidance / 样本充分性建议", result.report)
            self.assertIn("Profile Summary / 分类规则表现", result.report)
            self.assertIn("Segment Validation Summary / 分层验证表现", result.report)
            self.assertIn("Market Regime Validation / 市场状态验证", result.report)
            self.assertIn("Market Regime Protection Policy / 市场状态保护规则", result.report)
            self.assertIn("Probability Calibration / 概率校准", result.report)
            self.assertIn("Portfolio Validation / 组合验证", result.report)
            self.assertIn("Portfolio Equity Curve / 组合逐日净值曲线", result.report)
            self.assertIn("Benchmark Comparison / 基准对比", result.report)
            self.assertIn("Benchmark-Aware Rule Policy / 基准感知规则建议", result.report)
            self.assertIn("Specific Tightening Recommendations / 具体收紧建议", result.report)
            self.assertIn("Tightening Impact Validation / 收紧效果验证", result.report)
            self.assertIn("Threshold Sensitivity Grid / 阈值敏感度网格", result.report)
            self.assertIn("Minimum Sample Guard / 最小样本保护", result.report)
            self.assertIn("Profile Rule Calibration / 分类规则阈值建议", result.report)
            self.assertIn("Historical Win-Rate Dashboard / 历史胜率统计面板", result.report)
            self.assertIn("Profile Health Dashboard / 分类规则健康面板", result.report)
            self.assertIn("Profile Action Recommendations / 分类规则行动建议", result.report)
            self.assertIn("Profile Blocker Dashboard / 分类规则卡点面板", result.report)
            self.assertIn("Historical Win-Rate Gate / 历史胜率部署门槛", result.report)
            self.assertIn("Historical Threshold Recommendations / 历史阈值建议", result.report)
            self.assertIn("recommended_probability_adjustment", result.probability_calibration.columns)
            self.assertIn("brier_score", result.probability_calibration.columns)
            self.assertIn("overall", result.probability_calibration["probability_bucket"].to_list())
            self.assertIn("top_calibrated_probability", result.portfolio_summary["portfolio_name"].to_list())
            self.assertIn("calibrated_win_probability", result.calibration["rule"].to_list())
            self.assertIn("suggestion_confidence", result.profile_calibration.columns)

    def test_run_walk_forward_validation_can_force_screening_profile(self) -> None:
        prices = make_price_frame()

        result = run_walk_forward_validation(
            prices=prices,
            tickers=["AAA"],
            forward_windows=(5, 20),
            step_days=30,
            min_history_days=120,
            screening_profile_name="semiconductor",
        )

        self.assertFalse(result.events.empty)
        self.assertEqual(set(result.events["screening_profile"]), {"semiconductor"})
        self.assertEqual(set(result.events["screening_profile_zh"]), {"半导体规则"})

    def test_run_walk_forward_validation_emits_ticker_progress(self) -> None:
        prices = make_price_frame()
        progress_events: list[tuple[str, str, int, int]] = []

        result = run_walk_forward_validation(
            prices=prices,
            tickers=["AAA", "MISSING"],
            forward_windows=(5, 20),
            step_days=30,
            min_history_days=120,
            progress_callback=lambda ticker, status, index, total: progress_events.append(
                (ticker, status, index, total)
            ),
        )

        self.assertFalse(result.events.empty)
        self.assertEqual(progress_events[0], ("AAA", "started", 1, 2))
        self.assertIn(("AAA", "completed", 1, 2), progress_events)
        self.assertIn(("MISSING", "started", 2, 2), progress_events)
        self.assertIn(("MISSING", "skipped_insufficient_history", 2, 2), progress_events)

    def test_run_walk_forward_validation_handles_point_in_time_delisted_ticker(self) -> None:
        dates = pd.bdate_range("2023-01-02", periods=155)
        rows = []
        for ticker, start_price, periods in [("DEAD", 50.0, 132), ("SPY", 400.0, 155), ("QQQ", 350.0, 155)]:
            price = start_price
            for index, date in enumerate(dates[:periods]):
                price *= 1.0 + 0.001 + 0.001 * np.sin(index / 7)
                rows.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "open": price * 0.995,
                        "high": price * 1.01,
                        "low": price * 0.99,
                        "close": price,
                        "adj_close": price,
                        "volume": 2_000_000,
                    }
                )
        membership = HistoricalUniverseMembership(
            pd.DataFrame(
                [
                    {
                        "ticker": "DEAD",
                        "start_date": pd.Timestamp("2023-01-02"),
                        "end_date": pd.Timestamp(dates[131]),
                        "as_of_date": pd.Timestamp("2023-01-02"),
                        "delisted_date": pd.Timestamp(dates[131]),
                        "delisting_return": -1.0,
                        "status": "delisted",
                    }
                ]
            )
        )

        result = run_walk_forward_validation(
            prices=pd.DataFrame(rows),
            tickers=["DEAD"],
            forward_windows=(5, 20),
            step_days=20,
            min_history_days=120,
            universe_membership=membership,
        )

        self.assertFalse(result.events.empty)
        self.assertTrue(result.survivorship_bias_report["survivorship_bias_handled"])
        self.assertTrue(result.survivorship_bias_report["contains_delisted_tickers"])
        self.assertIn("survivorship_bias_handled: `true`", result.report)
        self.assertTrue(result.events["delisted_during_forward_window"].any())
        self.assertTrue(result.events["forward_return_20d_delisting_adjusted"].any())
        self.assertLess(result.events["forward_return_20d"].min(), -0.90)

    def test_validation_market_regime_lookup_labels_benchmark_context(self) -> None:
        dates = pd.date_range("2023-01-01", periods=260, freq="D")
        prices = pd.DataFrame(
            [
                {
                    "date": date,
                    "ticker": ticker,
                    "adj_close": 100.0 + index * 0.5,
                }
                for ticker in ["SPY", "QQQ"]
                for index, date in enumerate(dates)
            ]
        )

        lookup = build_validation_market_regime_lookup(prices)
        summary = summarize_market_regime_validation(
            pd.DataFrame(
                [
                    {
                        "validation_market_regime": lookup[pd.Timestamp(dates[-1]).normalize()][
                            "validation_market_regime"
                        ],
                        "validation_market_regime_zh": lookup[pd.Timestamp(dates[-1]).normalize()][
                            "validation_market_regime_zh"
                        ],
                        "validation_bucket": "high_probability",
                        "quality_gate_passed": True,
                        "signal_score": 75.0,
                        "confidence_score": 72.0,
                        "high_probability_score": 80.0,
                        "calibrated_win_probability": 0.65,
                        "forward_return_20d": 0.04,
                        "max_drawdown_after_signal": -0.02,
                    }
                    for _ in range(12)
                ]
            ),
            forward_windows=(20,),
        )

        self.assertEqual(
            lookup[pd.Timestamp(dates[-1]).normalize()]["validation_market_regime"],
            "bull_uptrend",
        )
        self.assertIn("regime_decision_zh", summary.columns)
        self.assertEqual(summary.iloc[0]["regime_decision"], "robust_regime")

    def test_market_regime_policy_blocks_weak_high_volatility_entries(self) -> None:
        summary = pd.DataFrame(
            [
                {
                    "validation_market_regime": "high_volatility",
                    "validation_market_regime_zh": "高波动市场",
                    "sample_count": 18,
                    "win_rate_20d": 0.44,
                    "avg_return_20d": -0.02,
                    "avg_max_drawdown_after_signal": -0.18,
                    "regime_decision": "weak_regime",
                },
                {
                    "validation_market_regime": "bull_uptrend",
                    "validation_market_regime_zh": "牛市上升",
                    "sample_count": 25,
                    "win_rate_20d": 0.60,
                    "avg_return_20d": 0.04,
                    "avg_max_drawdown_after_signal": -0.05,
                    "regime_decision": "robust_regime",
                },
            ]
        )

        policy = build_market_regime_protection_policy(summary)
        high_vol = policy[policy["validation_market_regime"] == "high_volatility"].iloc[0]
        bull = policy[policy["validation_market_regime"] == "bull_uptrend"].iloc[0]

        self.assertEqual(high_vol["protection_action"], "block_new_entries")
        self.assertFalse(bool(high_vol["allow_new_entries"]))
        self.assertEqual(high_vol["position_scale"], 0.0)
        self.assertEqual(bull["protection_action"], "normal_rules")

    def test_ticker_validation_ranking_prioritizes_stronger_ticker(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "ticker": "GOOD",
                    "validation_bucket": "high_probability",
                    "quality_gate_passed": True,
                    "signal_score": 72.0,
                    "confidence_score": 70.0,
                    "high_probability_score": 76.0,
                    "calibrated_win_probability": 0.66,
                    "forward_return_20d": 0.05,
                    "max_drawdown_after_signal": -0.02,
                }
                for _ in range(6)
            ]
            + [
                {
                    "ticker": "WEAK",
                    "validation_bucket": "filtered_out",
                    "quality_gate_passed": False,
                    "signal_score": 48.0,
                    "confidence_score": 50.0,
                    "high_probability_score": 42.0,
                    "calibrated_win_probability": 0.45,
                    "forward_return_20d": -0.04,
                    "max_drawdown_after_signal": -0.10,
                }
                for _ in range(6)
            ]
        )

        ranking = build_ticker_validation_ranking(events, target_window=20)

        self.assertEqual(ranking.iloc[0]["ticker"], "GOOD")
        self.assertEqual(ranking.iloc[0]["rank"], 1)
        self.assertEqual(ranking.iloc[0]["ticker_decision"], "priority_candidate")
        self.assertEqual(ranking.iloc[1]["ticker_decision"], "deprioritize")

    def test_sample_sufficiency_guidance_suggests_more_history_and_frequency(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "validation_bucket": "filtered_out",
                    "forward_return_20d": -0.01,
                }
                for _ in range(3)
            ]
        )
        ticker_ranking = build_ticker_validation_ranking(events, target_window=20)

        guidance = build_sample_sufficiency_guidance(
            events=events,
            tickers=["AAA", "BBB"],
            ticker_ranking=ticker_ranking,
            step_days=120,
            min_history_days=180,
        )

        overall = guidance[guidance["scope"] == "overall"].iloc[0]
        missing = guidance[guidance["ticker"] == "BBB"].iloc[0]
        self.assertEqual(overall["sample_status"], "thin_samples")
        self.assertEqual(overall["suggested_step_days"], 20)
        self.assertEqual(overall["suggested_min_history_days"], 170)
        self.assertIn("--period 5y", overall["suggested_command_hint"])
        self.assertEqual(missing["sample_status"], "no_samples")
        self.assertIn("没有样本", missing["sample_status_zh"])

    def test_pooled_entry_backtest_can_upgrade_sample_limited_events(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "screening_profile": "semiconductor",
                    "screening_profile_zh": "半导体规则",
                    "horizon": "medium",
                    "screening_backtest_entry_type": "pullback",
                    "quality_gate_passed": False,
                    "validation_bucket": "near_watchlist",
                    "screening_backtest_trade_count": 4,
                    "screening_backtest_win_rate": 0.75,
                    "screening_backtest_average_return": 0.08,
                    "quality_gate_fail_reasons": "pullback backtest sample too small",
                    "quality_gate_fail_reasons_zh": "回调买点回测样本不足",
                },
                {
                    "screening_profile": "semiconductor",
                    "screening_profile_zh": "半导体规则",
                    "horizon": "medium",
                    "screening_backtest_entry_type": "pullback",
                    "quality_gate_passed": False,
                    "validation_bucket": "near_watchlist",
                    "screening_backtest_trade_count": 5,
                    "screening_backtest_win_rate": 0.80,
                    "screening_backtest_average_return": 0.07,
                    "quality_gate_fail_reasons": "pullback backtest sample too small",
                    "quality_gate_fail_reasons_zh": "回调买点回测样本不足",
                },
                {
                    "screening_profile": "semiconductor",
                    "screening_profile_zh": "半导体规则",
                    "horizon": "medium",
                    "screening_backtest_entry_type": "pullback",
                    "quality_gate_passed": False,
                    "validation_bucket": "near_watchlist",
                    "screening_backtest_trade_count": 4,
                    "screening_backtest_win_rate": 1.00,
                    "screening_backtest_average_return": 0.10,
                    "quality_gate_fail_reasons": "pullback backtest sample too small",
                    "quality_gate_fail_reasons_zh": "回调买点回测样本不足",
                },
            ]
        )

        pooled = _apply_pooled_entry_backtest(events, default_screening_config())

        self.assertTrue(pooled["pooled_entry_backtest_used"].all())
        self.assertTrue(pooled["quality_gate_passed"].all())
        self.assertEqual(set(pooled["validation_bucket"]), {"high_probability"})
        self.assertTrue((pooled["screening_backtest_trade_count"] == 13).all())
        self.assertEqual(set(pooled["sample_confidence_level"]), {"limited"})
        self.assertEqual(set(pooled["evidence_strength"]), {"limited"})
        self.assertIn("同类聚合买点回测", pooled["quality_gate_fail_reasons_zh"].iloc[0])

    def test_pooled_entry_backtest_does_not_upgrade_early_watchlist(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "screening_profile": "semiconductor",
                    "screening_profile_zh": "半导体规则",
                    "horizon": "short",
                    "screening_backtest_entry_type": "breakout",
                    "quality_gate_passed": False,
                    "validation_bucket": "early_watchlist",
                    "screening_backtest_trade_count": 6,
                    "screening_backtest_win_rate": 0.65,
                    "screening_backtest_average_return": 0.04,
                    "quality_gate_fail_reasons": "signal score below threshold; breakout backtest sample too small",
                    "quality_gate_fail_reasons_zh": "信号分低于门槛；突破买点回测样本不足",
                },
                {
                    "screening_profile": "semiconductor",
                    "screening_profile_zh": "半导体规则",
                    "horizon": "short",
                    "screening_backtest_entry_type": "breakout",
                    "quality_gate_passed": False,
                    "validation_bucket": "early_watchlist",
                    "screening_backtest_trade_count": 7,
                    "screening_backtest_win_rate": 0.70,
                    "screening_backtest_average_return": 0.05,
                    "quality_gate_fail_reasons": "signal score below threshold; breakout backtest sample too small",
                    "quality_gate_fail_reasons_zh": "信号分低于门槛；突破买点回测样本不足",
                },
            ]
        )

        pooled = _apply_pooled_entry_backtest(events, default_screening_config())

        self.assertFalse(pooled["pooled_entry_backtest_used"].any())
        self.assertFalse(pooled["quality_gate_passed"].any())
        self.assertEqual(set(pooled["validation_bucket"]), {"early_watchlist"})

    def test_summary_and_calibration_handle_manual_events(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "validation_bucket": "high_probability",
                    "screening_profile": "saas_software",
                    "screening_profile_zh": "SaaS软件规则",
                    "quality_gate_passed": True,
                    "signal_score": 70.0,
                    "high_probability_score": 72.0,
                    "calibrated_win_probability": 0.62,
                    "relative_strength_score": 55.0,
                    "market_score": 60.0,
                    "forward_return_20d": 0.05,
                    "max_drawdown_after_signal": -0.02,
                },
                {
                    "validation_bucket": "filtered_out",
                    "screening_profile": "saas_software",
                    "screening_profile_zh": "SaaS软件规则",
                    "quality_gate_passed": False,
                    "signal_score": 50.0,
                    "high_probability_score": 45.0,
                    "calibrated_win_probability": 0.48,
                    "relative_strength_score": 40.0,
                    "market_score": 50.0,
                    "forward_return_20d": -0.03,
                    "max_drawdown_after_signal": -0.05,
                },
            ]
        )

        summary = summarize_walk_forward_events(events, forward_windows=(20,))
        profile_summary = summarize_walk_forward_profiles(events, forward_windows=(20,))
        segment_summary = summarize_walk_forward_segments(events, forward_windows=(20,))
        market_regime_summary = summarize_market_regime_validation(
            events.assign(
                validation_market_regime="sideways_mixed",
                validation_market_regime_zh="震荡分化",
            ),
            forward_windows=(20,),
        )
        probability_calibration = summarize_probability_calibration(events, target_window=20)
        calibration = calibrate_walk_forward_rules(events, target_window=20, min_sample_count=1)
        profile_calibration = calibrate_walk_forward_profiles(
            events,
            target_window=20,
            min_sample_count=1,
        )
        report = render_walk_forward_report(
            events,
            summary,
            calibration,
            profile_summary=profile_summary,
            segment_summary=segment_summary,
            market_regime_summary=market_regime_summary,
            profile_calibration=profile_calibration,
            probability_calibration=probability_calibration,
            forward_windows=(20,),
        )

        self.assertIn("high_probability", summary["bucket"].to_list())
        self.assertIn("saas_software", profile_summary["screening_profile"].to_list())
        self.assertIn("validation_segment", segment_summary.columns)
        self.assertIn("segment_decision_zh", segment_summary.columns)
        self.assertIn("sideways_mixed", market_regime_summary["validation_market_regime"].to_list())
        self.assertIn("signal_score", calibration["rule"].to_list())
        self.assertIn("calibrated_win_probability", calibration["rule"].to_list())
        self.assertIn("50_to_55", probability_calibration["probability_bucket"].to_list())
        self.assertIn("60_to_65", probability_calibration["probability_bucket"].to_list())
        self.assertIn("overall", probability_calibration["probability_bucket"].to_list())
        self.assertIn("recommended_probability_adjustment", probability_calibration.columns)
        self.assertIn("formula_action", probability_calibration.columns)
        self.assertIn("brier_score", probability_calibration.columns)
        self.assertIn("saas_software", profile_calibration["screening_profile"].to_list())
        self.assertIn("relative_strength_score", profile_calibration["rule"].to_list())
        self.assertIn("suggestion_confidence", profile_calibration.columns)
        self.assertIn("Profile Summary / 分类规则表现", report)
        self.assertIn("Segment Validation Summary / 分层验证表现", report)
        self.assertIn("Market Regime Validation / 市场状态验证", report)
        self.assertIn("Probability Calibration / 概率校准", report)
        self.assertIn("Profile Rule Calibration / 分类规则阈值建议", report)
        self.assertIn("Rule Calibration / 规则校准", report)

    def test_probability_calibration_recommends_formula_direction(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "calibrated_win_probability": 0.40,
                    "forward_return_20d": 0.02,
                }
                for _ in range(8)
            ]
        )

        result = summarize_probability_calibration(events, target_window=20)
        overall = result[result["probability_bucket"] == "overall"].iloc[0]
        low_bucket = result[result["probability_bucket"] == "below_50"].iloc[0]

        self.assertEqual(overall["formula_action"], "raise_probability_estimate")
        self.assertEqual(low_bucket["formula_action"], "raise_probability_estimate")
        self.assertGreater(overall["recommended_probability_adjustment"], 0)
        self.assertLess(overall["adjusted_calibration_error_abs"], overall["calibration_error_abs"])

    def test_portfolio_validation_selects_top_ranked_equal_weight_basket(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "date": "2024-01-31",
                    "ticker": "AAA",
                    "validation_bucket": "high_probability",
                    "high_probability_score": 80.0,
                    "calibrated_win_probability": 0.70,
                    "forward_return_20d": 0.10,
                    "max_drawdown_after_signal": -0.02,
                },
                {
                    "date": "2024-01-31",
                    "ticker": "BBB",
                    "validation_bucket": "near_watchlist",
                    "high_probability_score": 70.0,
                    "calibrated_win_probability": 0.60,
                    "forward_return_20d": -0.04,
                    "max_drawdown_after_signal": -0.06,
                },
                {
                    "date": "2024-01-31",
                    "ticker": "CCC",
                    "validation_bucket": "filtered_out",
                    "high_probability_score": 40.0,
                    "calibrated_win_probability": 0.45,
                    "forward_return_20d": 0.30,
                    "max_drawdown_after_signal": -0.01,
                },
                {
                    "date": "2024-02-29",
                    "ticker": "AAA",
                    "validation_bucket": "high_probability",
                    "high_probability_score": 76.0,
                    "calibrated_win_probability": 0.66,
                    "forward_return_20d": 0.02,
                    "max_drawdown_after_signal": -0.03,
                },
                {
                    "date": "2024-02-29",
                    "ticker": "BBB",
                    "validation_bucket": "near_watchlist",
                    "high_probability_score": 74.0,
                    "calibrated_win_probability": 0.64,
                    "forward_return_20d": 0.04,
                    "max_drawdown_after_signal": -0.02,
                },
            ]
        )

        summary, rebalances = summarize_walk_forward_portfolios(
            events,
            target_window=20,
            top_n=2,
        )
        top_row = summary[
            summary["portfolio_name"] == "top_calibrated_probability"
        ].iloc[0]
        strict_row = summary[
            summary["portfolio_name"] == "strict_high_probability_only"
        ].iloc[0]

        self.assertEqual(int(top_row["rebalance_count"]), 2)
        self.assertAlmostEqual(float(top_row["avg_forward_return"]), 0.03)
        self.assertAlmostEqual(float(top_row["win_rate"]), 1.0)
        self.assertEqual(int(strict_row["rebalance_count"]), 2)
        self.assertIn("AAA, BBB", rebalances["selected_tickers"].to_list())

    def test_portfolio_equity_curve_builds_daily_returns_with_costs(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=8)
        prices = pd.DataFrame(
            [
                {
                    "date": date,
                    "ticker": ticker,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "adj_close": price,
                    "volume": 1_000_000,
                }
                for ticker, series in {
                    "AAA": [100, 102, 104, 106, 108, 110, 112, 114],
                    "BBB": [50, 51, 50, 52, 53, 54, 55, 56],
                    "CCC": [80, 79, 81, 82, 84, 83, 85, 86],
                }.items()
                for date, price in zip(dates, series)
            ]
        )
        rebalances = pd.DataFrame(
            [
                {
                    "date": dates[0],
                    "portfolio_name": "top_calibrated_probability",
                    "portfolio_name_zh": "校准胜率前N",
                    "selected_count": 2,
                    "selected_tickers": "AAA, BBB",
                },
                {
                    "date": dates[4],
                    "portfolio_name": "top_calibrated_probability",
                    "portfolio_name_zh": "校准胜率前N",
                    "selected_count": 2,
                    "selected_tickers": "AAA, CCC",
                },
            ]
        )

        summary, curve = build_walk_forward_portfolio_equity(
            prices=prices,
            portfolio_rebalances=rebalances,
            target_window=3,
            transaction_cost=0.001,
        )

        self.assertFalse(summary.empty)
        self.assertFalse(curve.empty)
        self.assertIn("equity", curve.columns)
        self.assertIn("drawdown", curve.columns)
        self.assertLess(float(curve["equity"].iloc[0]), 1.0)
        self.assertGreater(float(summary["daily_rows"].iloc[0]), 0)
        self.assertGreater(float(summary["total_return"].iloc[0]), 0)
        self.assertLessEqual(float(summary["max_drawdown"].iloc[0]), 0)

    def test_benchmark_comparison_aligns_portfolio_with_spy_and_qqq(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=6)
        prices = pd.DataFrame(
            [
                {
                    "date": date,
                    "ticker": ticker,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "adj_close": price,
                    "volume": 1_000_000,
                }
                for ticker, series in {
                    "SPY": [100, 101, 102, 103, 104, 105],
                    "QQQ": [200, 202, 204, 206, 208, 210],
                }.items()
                for date, price in zip(dates, series)
            ]
        )
        daily_returns = [0.0, 0.02, 0.01, -0.01, 0.03, 0.01]
        equity_values = [1.0, 1.02, 1.0302, 1.019898, 1.05049494, 1.0609998894]
        portfolio_curve = pd.DataFrame(
            [
                {
                    "date": date,
                    "portfolio_name": "top_calibrated_probability",
                    "portfolio_name_zh": "校准胜率前N",
                    "daily_return": daily_return,
                    "equity": equity,
                    "drawdown": min(equity / max(equity_values[: index + 1]) - 1.0, 0.0),
                }
                for index, (date, daily_return, equity) in enumerate(
                    zip(dates, daily_returns, equity_values)
                )
            ]
        )

        summary, curve = build_walk_forward_benchmark_comparison(
            prices=prices,
            portfolio_equity_curve=portfolio_curve,
        )

        self.assertFalse(summary.empty)
        self.assertFalse(curve.empty)
        self.assertIn("SPY", summary["benchmark_ticker"].to_list())
        self.assertIn("QQQ", summary["benchmark_ticker"].to_list())
        spy_row = summary[summary["benchmark_ticker"] == "SPY"].iloc[0]
        self.assertGreater(float(spy_row["excess_total_return"]), 0)
        self.assertIn("relative_equity", curve.columns)

    def test_benchmark_policy_recommends_tightening_when_portfolio_lags(self) -> None:
        benchmark_summary = pd.DataFrame(
            [
                {
                    "portfolio_name": "top_calibrated_probability",
                    "portfolio_name_zh": "校准胜率前N",
                    "benchmark_ticker": "SPY",
                    "excess_total_return": -0.04,
                    "excess_annualized_return": -0.10,
                    "drawdown_advantage": -0.03,
                    "daily_correlation": 0.70,
                },
                {
                    "portfolio_name": "top_calibrated_probability",
                    "portfolio_name_zh": "校准胜率前N",
                    "benchmark_ticker": "QQQ",
                    "excess_total_return": -0.01,
                    "excess_annualized_return": -0.03,
                    "drawdown_advantage": -0.01,
                    "daily_correlation": 0.80,
                },
            ]
        )

        policy = build_benchmark_aware_policy(benchmark_summary)
        row = policy.iloc[0]

        self.assertEqual(row["benchmark_policy_action"], "tighten_rules")
        self.assertFalse(bool(row["allow_relaxation"]))
        self.assertIn("收紧", row["benchmark_policy_action_zh"])

    def test_benchmark_tightening_recommends_specific_threshold_changes(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "screening_profile": "saas_software",
                    "screening_profile_zh": "SaaS软件规则",
                    "signal_score": 70.0,
                    "high_probability_score": 68.0,
                    "relative_strength_score": 54.0,
                    "market_score": 60.0,
                    "screening_backtest_win_rate": 0.58,
                    "forward_return_20d": 0.04,
                }
                for _ in range(12)
            ]
        )
        profile_calibration = calibrate_walk_forward_profiles(
            events,
            target_window=20,
            min_sample_count=5,
        )
        benchmark_policy = pd.DataFrame(
            [
                {
                    "portfolio_name": "top_calibrated_probability",
                    "benchmark_policy_action": "tighten_rules",
                    "allow_relaxation": False,
                }
            ]
        )

        tightening = build_benchmark_tightening_recommendations(
            events=events,
            screening_config=default_screening_config(),
            profile_calibration=profile_calibration,
            benchmark_policy=benchmark_policy,
            target_window=20,
        )

        self.assertFalse(tightening.empty)
        self.assertIn("signal_score_min", tightening["threshold_attr"].to_list())
        signal_row = tightening[tightening["threshold_attr"] == "signal_score_min"].iloc[0]
        self.assertGreater(float(signal_row["suggested_threshold"]), float(signal_row["current_threshold"]))
        self.assertIn(signal_row["priority"], {"high", "medium"})

    def test_tightening_impact_validation_compares_before_and_after_samples(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "screening_profile": "saas_software",
                    "screening_profile_zh": "SaaS软件规则",
                    "validation_bucket": "near_watchlist",
                    "relative_strength_score": 48.0,
                    "market_score": 56.0,
                    "forward_return_20d": -0.03,
                    "max_drawdown_after_signal": -0.07,
                },
                {
                    "screening_profile": "saas_software",
                    "screening_profile_zh": "SaaS软件规则",
                    "validation_bucket": "near_watchlist",
                    "relative_strength_score": 52.0,
                    "market_score": 62.0,
                    "forward_return_20d": 0.04,
                    "max_drawdown_after_signal": -0.03,
                },
                {
                    "screening_profile": "saas_software",
                    "screening_profile_zh": "SaaS软件规则",
                    "validation_bucket": "high_probability",
                    "relative_strength_score": 58.0,
                    "market_score": 66.0,
                    "forward_return_20d": 0.06,
                    "max_drawdown_after_signal": -0.02,
                },
                {
                    "screening_profile": "saas_software",
                    "screening_profile_zh": "SaaS软件规则",
                    "validation_bucket": "early_watchlist",
                    "relative_strength_score": 60.0,
                    "market_score": 70.0,
                    "forward_return_20d": 0.03,
                    "max_drawdown_after_signal": -0.01,
                },
                {
                    "screening_profile": "saas_software",
                    "screening_profile_zh": "SaaS软件规则",
                    "validation_bucket": "near_watchlist",
                    "relative_strength_score": 62.0,
                    "market_score": 68.0,
                    "forward_return_20d": 0.05,
                    "max_drawdown_after_signal": -0.02,
                },
                {
                    "screening_profile": "saas_software",
                    "screening_profile_zh": "SaaS软件规则",
                    "validation_bucket": "near_watchlist",
                    "relative_strength_score": 64.0,
                    "market_score": 72.0,
                    "forward_return_20d": 0.04,
                    "max_drawdown_after_signal": -0.02,
                },
            ]
        )
        tightening = pd.DataFrame(
            [
                {
                    "screening_profile": "saas_software",
                    "threshold_attr": "relative_strength_min",
                    "suggested_threshold": 50.0,
                    "priority": "medium",
                }
            ]
        )

        impact = build_tightening_impact_validation(
            events=events,
            benchmark_tightening=tightening,
            target_window=20,
        )
        row = impact.iloc[0]

        self.assertEqual(int(row["before_sample_count"]), 6)
        self.assertEqual(int(row["after_sample_count"]), 5)
        self.assertGreater(float(row["after_avg_return"]), float(row["before_avg_return"]))
        self.assertIn(row["impact_decision"], {"validated_improvement", "mixed_result"})

    def test_threshold_sensitivity_grid_finds_promising_thresholds(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "screening_profile": "saas_software",
                    "screening_profile_zh": "SaaS软件规则",
                    "validation_bucket": "near_watchlist",
                    "signal_score": 64.0,
                    "high_probability_score": 61.0,
                    "relative_strength_score": 44.0,
                    "market_score": 54.0,
                    "screening_backtest_win_rate": 0.54,
                    "forward_return_20d": -0.05,
                    "max_drawdown_after_signal": -0.08,
                },
                *[
                    {
                        "screening_profile": "saas_software",
                        "screening_profile_zh": "SaaS软件规则",
                        "validation_bucket": "near_watchlist",
                        "signal_score": 72.0 + index,
                        "high_probability_score": 68.0,
                        "relative_strength_score": 56.0,
                        "market_score": 62.0,
                        "screening_backtest_win_rate": 0.60,
                        "forward_return_20d": 0.04 + index * 0.002,
                        "max_drawdown_after_signal": -0.02,
                    }
                    for index in range(6)
                ],
            ]
        )

        grid = build_threshold_sensitivity_grid(
            events=events,
            screening_config=default_screening_config(),
            target_window=20,
        )

        self.assertFalse(grid.empty)
        self.assertIn("baseline", grid["sensitivity_decision"].to_list())
        self.assertIn("promising_threshold", grid["sensitivity_decision"].to_list())
        promising = grid[grid["sensitivity_decision"] == "promising_threshold"].iloc[0]
        self.assertGreater(float(promising["avg_return_change"]), 0)
        self.assertGreaterEqual(float(promising["win_rate_change"]), 0)

    def test_minimum_sample_guard_blocks_low_sample_thresholds(self) -> None:
        sensitivity = pd.DataFrame(
            [
                {
                    "screening_profile": "saas_software",
                    "screening_profile_zh": "SaaS软件规则",
                    "grid_type": "single_rule",
                    "threshold_expression": "signal_score_min>=70",
                    "threshold_expression_zh": "信号分>=70",
                    "sample_count": 4,
                    "baseline_sample_count": 20,
                    "sample_retention_rate": 0.20,
                    "sensitivity_decision": "promising_threshold",
                },
                {
                    "screening_profile": "saas_software",
                    "screening_profile_zh": "SaaS软件规则",
                    "grid_type": "single_rule",
                    "threshold_expression": "relative_strength_min>=50",
                    "threshold_expression_zh": "相对强弱>=50",
                    "sample_count": 12,
                    "baseline_sample_count": 20,
                    "sample_retention_rate": 0.60,
                    "sensitivity_decision": "promising_threshold",
                },
                {
                    "screening_profile": "saas_software",
                    "screening_profile_zh": "SaaS软件规则",
                    "grid_type": "single_rule",
                    "threshold_expression": "market_score_min>=60",
                    "threshold_expression_zh": "大盘环境分>=60",
                    "sample_count": 12,
                    "baseline_sample_count": 20,
                    "sample_retention_rate": 0.60,
                    "sensitivity_decision": "mixed_threshold",
                },
            ]
        )

        guard = build_minimum_sample_guard(sensitivity, min_sample_count=10)

        blocked = guard[guard["threshold_attr"] == "signal_score_min"].iloc[0]
        allowed = guard[guard["threshold_attr"] == "relative_strength_min"].iloc[0]
        watch = guard[guard["threshold_attr"] == "market_score_min"].iloc[0]
        self.assertEqual(blocked["guard_action"], "block_adoption")
        self.assertFalse(bool(blocked["allow_adoption"]))
        self.assertEqual(allowed["guard_action"], "allow_adoption")
        self.assertTrue(bool(allowed["allow_adoption"]))
        self.assertEqual(watch["guard_action"], "watch_only")

    def test_render_suggested_screening_config_skips_low_confidence_suggestions(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "validation_bucket": "early_watchlist",
                    "screening_profile": "saas_software",
                    "screening_profile_zh": "SaaS软件规则",
                    "quality_gate_passed": False,
                    "signal_score": 72.0,
                    "high_probability_score": 68.0,
                    "relative_strength_score": 56.0,
                    "market_score": 60.0,
                    "forward_return_20d": 0.04,
                    "max_drawdown_after_signal": -0.02,
                }
                for _ in range(5)
            ]
        )
        profile_calibration = calibrate_walk_forward_profiles(
            events,
            target_window=20,
            min_sample_count=5,
        )

        toml_text = render_suggested_screening_config(
            screening_config=default_screening_config(),
            profile_calibration=profile_calibration,
            min_sample_count=5,
        )

        self.assertIn("Suggested screening config", toml_text)
        self.assertIn("[profiles.saas_software]", toml_text)
        self.assertIn("confidence=low", toml_text)
        self.assertIn("signal_score_min = 67", toml_text)

    def test_render_suggested_screening_config_applies_medium_confidence_suggestions(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "validation_bucket": "early_watchlist",
                    "screening_profile": "saas_software",
                    "screening_profile_zh": "SaaS软件规则",
                    "quality_gate_passed": False,
                    "signal_score": 72.0,
                    "high_probability_score": 68.0,
                    "relative_strength_score": 56.0,
                    "market_score": 60.0,
                    "forward_return_20d": 0.04,
                    "max_drawdown_after_signal": -0.02,
                }
                for _ in range(15)
            ]
        )
        profile_calibration = calibrate_walk_forward_profiles(
            events,
            target_window=20,
            min_sample_count=5,
        )

        toml_text = render_suggested_screening_config(
            screening_config=default_screening_config(),
            profile_calibration=profile_calibration,
            min_sample_count=5,
        )

        self.assertIn("confidence=medium", toml_text)
        self.assertIn("signal_score_min = 70", toml_text)

    def test_render_suggested_screening_config_blocks_relaxation_when_benchmark_lags(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "validation_bucket": "early_watchlist",
                    "screening_profile": "saas_software",
                    "screening_profile_zh": "SaaS软件规则",
                    "quality_gate_passed": False,
                    "signal_score": 60.0,
                    "high_probability_score": 60.0,
                    "relative_strength_score": 50.0,
                    "market_score": 55.0,
                    "forward_return_20d": 0.04,
                    "max_drawdown_after_signal": -0.02,
                }
                for _ in range(15)
            ]
        )
        profile_calibration = calibrate_walk_forward_profiles(
            events,
            target_window=20,
            min_sample_count=5,
        )
        benchmark_policy = pd.DataFrame(
            [
                {
                    "portfolio_name": "top_calibrated_probability",
                    "benchmark_policy_action": "tighten_rules",
                    "allow_relaxation": False,
                    "policy_note": "Portfolio validation is not beating benchmarks.",
                }
            ]
        )

        toml_text = render_suggested_screening_config(
            screening_config=default_screening_config(),
            profile_calibration=profile_calibration,
            benchmark_policy=benchmark_policy,
            min_sample_count=5,
        )

        self.assertIn("Benchmark guard: tighten_rules", toml_text)
        self.assertIn("skipped relaxation by benchmark guard", toml_text)
        self.assertIn("signal_score_min = 67", toml_text)

    def test_render_suggested_screening_config_applies_specific_tightening(self) -> None:
        profile_calibration = pd.DataFrame()
        benchmark_policy = pd.DataFrame(
            [
                {
                    "portfolio_name": "top_calibrated_probability",
                    "benchmark_policy_action": "tighten_rules",
                    "allow_relaxation": False,
                    "policy_note": "Portfolio validation is not beating benchmarks.",
                }
            ]
        )
        benchmark_tightening = pd.DataFrame(
            [
                {
                    "screening_profile": "saas_software",
                    "threshold_attr": "signal_score_min",
                    "current_threshold": 67.0,
                    "suggested_threshold": 70.0,
                    "priority": "high",
                }
            ]
        )

        toml_text = render_suggested_screening_config(
            screening_config=default_screening_config(),
            profile_calibration=profile_calibration,
            benchmark_policy=benchmark_policy,
            benchmark_tightening=benchmark_tightening,
            min_sample_count=5,
        )

        self.assertIn("Benchmark tightening suggestions applied: 1", toml_text)
        self.assertIn("signal_score_min = 70", toml_text)

    def test_render_suggested_screening_config_respects_minimum_sample_guard(self) -> None:
        profile_calibration = pd.DataFrame()
        benchmark_policy = pd.DataFrame(
            [
                {
                    "portfolio_name": "top_calibrated_probability",
                    "benchmark_policy_action": "tighten_rules",
                    "allow_relaxation": False,
                    "policy_note": "Portfolio validation is not beating benchmarks.",
                }
            ]
        )
        benchmark_tightening = pd.DataFrame(
            [
                {
                    "screening_profile": "saas_software",
                    "threshold_attr": "signal_score_min",
                    "current_threshold": 67.0,
                    "suggested_threshold": 70.0,
                    "priority": "high",
                }
            ]
        )
        minimum_sample_guard = pd.DataFrame(
            [
                {
                    "screening_profile": "saas_software",
                    "threshold_attr": "signal_score_min",
                    "threshold_value": 70.0,
                    "allow_adoption": False,
                    "guard_action": "block_adoption",
                }
            ]
        )

        toml_text = render_suggested_screening_config(
            screening_config=default_screening_config(),
            profile_calibration=profile_calibration,
            benchmark_policy=benchmark_policy,
            benchmark_tightening=benchmark_tightening,
            minimum_sample_guard=minimum_sample_guard,
            min_sample_count=5,
        )

        self.assertIn("Benchmark tightening suggestions applied: 0", toml_text)
        self.assertIn("signal_score_min = 67", toml_text)

    def test_render_suggested_screening_config_applies_historical_threshold_recommendations(self) -> None:
        profile_calibration = pd.DataFrame()
        historical_recommendations = pd.DataFrame(
            [
                {
                    "scope": "screening_profile",
                    "group_value": "saas_software",
                    "threshold_attr": "signal_score_min",
                    "current_threshold": 67.0,
                    "suggested_threshold": 71.0,
                    "priority": "high",
                    "allow_auto_apply": True,
                },
                {
                    "scope": "overall",
                    "group_value": "all_events",
                    "threshold_attr": "market_score_min",
                    "current_threshold": 55.0,
                    "suggested_threshold": 60.0,
                    "priority": "high",
                    "allow_auto_apply": True,
                },
                {
                    "scope": "screening_profile",
                    "group_value": "saas_software",
                    "threshold_attr": "relative_strength_min",
                    "current_threshold": 45.0,
                    "suggested_threshold": 55.0,
                    "priority": "medium",
                    "allow_auto_apply": False,
                },
            ]
        )

        toml_text = render_suggested_screening_config(
            screening_config=default_screening_config(),
            profile_calibration=profile_calibration,
            historical_threshold_recommendations=historical_recommendations,
            min_sample_count=5,
        )

        self.assertIn("Historical threshold suggestions applied: 1", toml_text)
        self.assertIn("saas_software.signal_score_min: 67 -> 71", toml_text)
        self.assertIn("signal_score_min = 71", toml_text)
        self.assertIn("relative_strength_min = 45", toml_text)


def make_price_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=280)
    rows = []
    specs = {
        "AAA": (100.0, 0.0015),
        "SPY": (400.0, 0.0006),
        "QQQ": (350.0, 0.0008),
    }
    for ticker, (start_price, drift) in specs.items():
        price = start_price
        for index, date in enumerate(dates):
            price *= 1.0 + drift + 0.002 * np.sin(index / 8)
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": price * 0.995,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "adj_close": price,
                    "volume": 2_000_000,
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    unittest.main()
