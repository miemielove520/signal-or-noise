from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest

import pandas as pd

from stock_selector.screening_config import default_screening_config
from stock_selector.universe import BUILT_IN_UNIVERSES, load_universe_tickers
from stock_selector.universe_validation import (
    _diagnose_universe,
    build_best_universe_ranking,
    build_validation_coverage_plan,
    render_validation_coverage_plan,
    render_suggested_universe_screening_config,
    run_all_builtin_universe_validations,
)
from stock_selector.walk_forward import WalkForwardResult


class UniverseValidationTest(unittest.TestCase):
    def test_run_all_builtin_universe_validations_writes_aggregate_outputs(self) -> None:
        price_calls: list[list[str]] = []
        validation_calls: list[list[str]] = []

        def fake_price_loader(tickers, period, output_path):
            price_calls.append(list(tickers))
            return SimpleNamespace(
                prices=pd.DataFrame(
                    {
                        "date": pd.to_datetime(["2026-01-01"]),
                        "ticker": [tickers[0]],
                        "adj_close": [100.0],
                    }
                ),
                provider="fake_provider",
            )

        def fake_validation_runner(
            prices,
            tickers,
            step_days,
            min_history_days,
            output_dir,
            screening_config,
            screening_profile_name=None,
        ):
            validation_calls.append(list(tickers))
            events = pd.DataFrame(
                [
                    {
                        "validation_bucket": "high_probability",
                        "quality_gate_fail_reasons": "all strict quality gates passed",
                        "quality_gate_fail_reasons_zh": "所有严格质量门槛通过",
                        "watchlist_missing_items": "none",
                        "watchlist_missing_items_zh": "无",
                        "forward_return_20d": 0.06,
                        "max_drawdown_after_signal": -0.02,
                    },
                    {
                        "validation_bucket": "filtered_out",
                        "quality_gate_fail_reasons": "signal score too low; market not supportive",
                        "quality_gate_fail_reasons_zh": "信号分数不足；大盘环境不够支持",
                        "watchlist_missing_items": "signal score >= 65; market score >= 55",
                        "watchlist_missing_items_zh": "信号分数达到65分；大盘分数达到55分",
                        "forward_return_20d": -0.03,
                        "max_drawdown_after_signal": -0.05,
                    },
                ]
            )
            return WalkForwardResult(
                events=events,
                summary=pd.DataFrame(),
                profile_summary=pd.DataFrame(),
                segment_summary=pd.DataFrame(),
                market_regime_summary=pd.DataFrame(),
                market_regime_policy=pd.DataFrame(),
                profile_calibration=pd.DataFrame(),
                calibration=pd.DataFrame(),
                report="",
            )

        with tempfile.TemporaryDirectory() as directory:
            progress_events: list[tuple[str, str, int, int]] = []
            result = run_all_builtin_universe_validations(
                universe_names=["software", "cybersecurity"],
                period="1y",
                step_days=40,
                min_history_days=120,
                output_dir=directory,
                price_loader=fake_price_loader,
                validation_runner=fake_validation_runner,
                progress_callback=lambda name, status, index, total: progress_events.append(
                    (name, status, index, total)
                ),
            )

            root = Path(directory)
            self.assertTrue((root / "all_universe_validation_summary.csv").exists())
            self.assertTrue((root / "best_universe_ranking.csv").exists())
            self.assertTrue((root / "validation_coverage_plan.csv").exists())
            self.assertTrue((root / "validation_coverage_plan.md").exists())
            self.assertTrue((root / "all_universe_validation_report.md").exists())
            self.assertTrue((root / "all_universe_validation_result.json").exists())
            self.assertTrue((root / "all_universe_suggested_screening.toml").exists())
            payload = json.loads((root / "all_universe_validation_result.json").read_text())

        self.assertEqual(len(price_calls), 2)
        self.assertEqual(len(validation_calls), 2)
        self.assertIn("SPY", price_calls[0])
        self.assertIn("QQQ", price_calls[0])
        self.assertNotIn("SPY", validation_calls[0])
        self.assertNotIn("QQQ", validation_calls[0])
        self.assertEqual(progress_events[0], ("software", "started", 1, 2))
        self.assertEqual(progress_events[-1], ("cybersecurity", "completed", 2, 2))
        self.assertEqual(payload["completed_universe_count"], 2)
        self.assertEqual(result.failures, [])
        self.assertEqual(set(result.summary["universe"]), {"software", "cybersecurity"})
        self.assertTrue((result.summary["high_probability_sample_count"] == 1).all())
        self.assertIn("optimization_priority", result.summary.columns)
        self.assertIn("diagnostic_level_zh", result.summary.columns)
        self.assertIn("recommendation_zh", result.summary.columns)
        self.assertIn("top_quality_gate_failures_zh", result.summary.columns)
        self.assertIn("信号分数不足", result.summary["top_quality_gate_failures_zh"].iloc[0])
        self.assertIn("大盘环境不够支持", result.summary["top_quality_gate_failures_zh"].iloc[0])
        self.assertIn("suggested_threshold_changes_zh", result.summary.columns)
        self.assertIn("All Universe Walk-Forward Validation", result.report)
        self.assertIn("Primary Diagnostic / 主要诊断", result.report)
        self.assertIn("Best Candidate Universe Ranking / 最佳股票池排序", result.report)
        self.assertIn("Large sample universes / 大样本股票池", result.report)
        self.assertIn("ranking", payload)
        self.assertIn("coverage_plan", payload)
        self.assertIn("best_universe_ranking_csv", payload["output_files"])
        self.assertIn("validation_coverage_plan_csv", payload["output_files"])
        self.assertFalse(result.ranking.empty)
        self.assertIn("decision_zh", result.ranking.columns)
        self.assertIn("why_ranked_here_zh", result.ranking.columns)
        self.assertIn("Validation Coverage Plan / 验证覆盖计划", result.report)

    def test_validation_coverage_plan_maps_universes_to_safe_workflow(self) -> None:
        plan = build_validation_coverage_plan(["semiconductors", "healthcare-quality"])
        markdown = render_validation_coverage_plan(plan)

        self.assertEqual(plan.loc[plan["universe"] == "semiconductors", "mapped_profile"].iloc[0], "semiconductor")
        self.assertEqual(plan.loc[plan["universe"] == "healthcare-quality", "mapped_profile"].iloc[0], "default")
        self.assertIn("python3 validate.py --universe semiconductors", plan["deep_command"].iloc[0])
        self.assertIn("compare_runs.py", markdown)
        self.assertIn("安全配置流程", markdown)

    def test_research_core_universe_combines_built_in_sample_pools(self) -> None:
        tickers = load_universe_tickers(universe_name="research-core")

        self.assertIn("AAPL", tickers)
        self.assertIn("NVDA", tickers)
        self.assertIn("PANW", tickers)
        self.assertIn("COST", tickers)
        self.assertIn("AAON", tickers)
        self.assertGreater(len(tickers), len(BUILT_IN_UNIVERSES["mega-cap-tech"]))
        self.assertEqual(len(tickers), len(set(tickers)))

    def test_builtin_universes_use_expected_screening_profiles(self) -> None:
        profile_calls: list[str | None] = []

        def fake_price_loader(tickers, period, output_path):
            return SimpleNamespace(
                prices=pd.DataFrame(
                    {
                        "date": pd.to_datetime(["2026-01-01"]),
                        "ticker": [tickers[0]],
                        "adj_close": [100.0],
                    }
                ),
                provider="fake_provider",
            )

        def fake_validation_runner(
            prices,
            tickers,
            step_days,
            min_history_days,
            output_dir,
            screening_config,
            screening_profile_name=None,
        ):
            profile_calls.append(screening_profile_name)
            return WalkForwardResult(
                events=pd.DataFrame(
                    [
                        {
                            "validation_bucket": "filtered_out",
                            "quality_gate_fail_reasons": "signal score too low",
                            "quality_gate_fail_reasons_zh": "信号分数不足",
                            "watchlist_missing_items": "signal score >= 65",
                            "watchlist_missing_items_zh": "信号分数达到65分",
                            "forward_return_20d": -0.01,
                            "max_drawdown_after_signal": -0.03,
                        }
                    ]
                ),
                summary=pd.DataFrame(),
                profile_summary=pd.DataFrame(),
                segment_summary=pd.DataFrame(),
                market_regime_summary=pd.DataFrame(),
                market_regime_policy=pd.DataFrame(),
                profile_calibration=pd.DataFrame(),
                calibration=pd.DataFrame(),
                report="",
            )

        with tempfile.TemporaryDirectory() as directory:
            run_all_builtin_universe_validations(
                universe_names=["ai-infrastructure", "industrial-quality"],
                period="1y",
                output_dir=directory,
                price_loader=fake_price_loader,
                validation_runner=fake_validation_runner,
            )

        self.assertEqual(profile_calls, ["ai_infrastructure", "industrial_quality"])

    def test_diagnosis_prioritizes_strong_high_probability_universe(self) -> None:
        def fake_price_loader(tickers, period, output_path):
            return SimpleNamespace(
                prices=pd.DataFrame(
                    {
                        "date": pd.to_datetime(["2026-01-01"]),
                        "ticker": [tickers[0]],
                        "adj_close": [100.0],
                    }
                ),
                provider="fake_provider",
            )

        def fake_validation_runner(
            prices,
            tickers,
            step_days,
            min_history_days,
            output_dir,
            screening_config,
            screening_profile_name=None,
        ):
            events = pd.DataFrame(
                [
                    {
                        "validation_bucket": "high_probability",
                        "quality_gate_fail_reasons": "all strict quality gates passed",
                        "quality_gate_fail_reasons_zh": "所有严格质量门槛通过",
                        "watchlist_missing_items": "none",
                        "watchlist_missing_items_zh": "无",
                        "forward_return_20d": 0.04,
                        "max_drawdown_after_signal": -0.02,
                    }
                    for _ in range(12)
                ]
                + [
                    {
                        "validation_bucket": "filtered_out",
                        "quality_gate_fail_reasons": "confidence too low",
                        "quality_gate_fail_reasons_zh": "置信度不足",
                        "watchlist_missing_items": "confidence >= 65",
                        "watchlist_missing_items_zh": "置信度达到65分",
                        "forward_return_20d": -0.02,
                        "max_drawdown_after_signal": -0.04,
                    }
                    for _ in range(8)
                ]
            )
            return WalkForwardResult(
                events=events,
                summary=pd.DataFrame(),
                profile_summary=pd.DataFrame(),
                segment_summary=pd.DataFrame(),
                market_regime_summary=pd.DataFrame(),
                market_regime_policy=pd.DataFrame(),
                profile_calibration=pd.DataFrame(),
                calibration=pd.DataFrame(),
                report="",
            )

        with tempfile.TemporaryDirectory() as directory:
            result = run_all_builtin_universe_validations(
                universe_names=["software"],
                period="1y",
                output_dir=directory,
                price_loader=fake_price_loader,
                validation_runner=fake_validation_runner,
            )

        row = result.summary.iloc[0]
        self.assertEqual(row["optimization_priority"], 1)
        self.assertEqual(row["diagnostic_level"], "optimize_first")
        self.assertEqual(row["diagnostic_level_zh"], "优先优化")
        self.assertEqual(result.ranking.iloc[0]["decision"], "optimize_first")

    def test_best_universe_ranking_prioritizes_stronger_universe(self) -> None:
        summary = pd.DataFrame(
            [
                {
                    "universe": "weak",
                    "status": "ok",
                    "optimization_priority": 9,
                    "diagnostic_level": "deprioritize",
                    "diagnostic_level_zh": "降低优先级",
                    "recommendation": "Do not optimize this universe first.",
                    "recommendation_zh": "不要优先优化这个股票池。",
                    "ticker_count": 20,
                    "event_count": 80,
                    "all_sample_count": 80,
                    "all_20d_win_rate": 0.46,
                    "all_20d_avg_return": -0.01,
                    "all_avg_max_drawdown": -0.08,
                    "high_probability_sample_count": 0,
                    "high_probability_20d_win_rate": float("nan"),
                    "high_probability_20d_avg_return": float("nan"),
                    "output_dir": "outputs/weak",
                },
                {
                    "universe": "strong",
                    "status": "ok",
                    "optimization_priority": 1,
                    "diagnostic_level": "optimize_first",
                    "diagnostic_level_zh": "优先优化",
                    "recommendation": "Prioritize this universe for tighter thresholds.",
                    "recommendation_zh": "优先优化这个股票池。",
                    "ticker_count": 30,
                    "event_count": 120,
                    "all_sample_count": 120,
                    "all_20d_win_rate": 0.58,
                    "all_20d_avg_return": 0.03,
                    "all_avg_max_drawdown": -0.04,
                    "high_probability_sample_count": 18,
                    "high_probability_20d_win_rate": 0.67,
                    "high_probability_20d_avg_return": 0.05,
                    "output_dir": "outputs/strong",
                },
            ]
        )

        ranking = build_best_universe_ranking(summary)

        self.assertEqual(ranking.iloc[0]["rank"], 1)
        self.assertEqual(ranking.iloc[0]["universe"], "strong")
        self.assertEqual(ranking.iloc[0]["decision"], "optimize_first")
        self.assertEqual(ranking.iloc[1]["decision"], "deprioritize")

    def test_diagnosis_describes_small_high_probability_sample_as_too_few(self) -> None:
        diagnosis = _diagnose_universe(
            {
                "status": "ok",
                "event_count": 27,
                "high_probability_sample_count": 1,
                "near_watchlist_sample_count": 8,
                "early_watchlist_sample_count": 10,
                "all_20d_avg_return": 0.049,
                "all_20d_win_rate": 0.667,
                "high_probability_20d_avg_return": 0.309,
                "high_probability_20d_win_rate": 1.0,
            }
        )

        self.assertEqual(diagnosis["diagnostic_level"], "thresholds_too_strict")
        self.assertIn("Only 1 high-probability samples", diagnosis["diagnosis_reasons"])
        self.assertIn("目前只有 1 个高概率样本", diagnosis["diagnosis_reasons_zh"])
        self.assertNotIn("No strict high-probability samples", diagnosis["diagnosis_reasons"])

    def test_threshold_suggestions_use_common_blockers_when_validation_is_positive(self) -> None:
        def fake_price_loader(tickers, period, output_path):
            return SimpleNamespace(
                prices=pd.DataFrame(
                    {
                        "date": pd.to_datetime(["2026-01-01"]),
                        "ticker": [tickers[0]],
                        "adj_close": [100.0],
                    }
                ),
                provider="fake_provider",
            )

        profile_names: list[str | None] = []

        def fake_validation_runner(
            prices,
            tickers,
            step_days,
            min_history_days,
            output_dir,
            screening_config,
            screening_profile_name=None,
        ):
            profile_names.append(screening_profile_name)
            events = pd.DataFrame(
                [
                    {
                        "validation_bucket": "early_watchlist",
                        "quality_gate_fail_reasons": (
                            "signal score too low; confidence too low; "
                            "breakout backtest sample too small"
                        ),
                        "quality_gate_fail_reasons_zh": (
                            "信号分数不足；置信度不足；突破买点回测样本不足"
                        ),
                        "watchlist_missing_items": (
                            "signal score >= 68; confidence >= 65; entry backtest sample >= 12"
                        ),
                        "watchlist_missing_items_zh": (
                            "信号分数达到68分；置信度达到65分；买点回测样本至少12笔"
                        ),
                        "forward_return_20d": 0.03,
                        "max_drawdown_after_signal": -0.02,
                    }
                    for _ in range(20)
                ]
            )
            return WalkForwardResult(
                events=events,
                summary=pd.DataFrame(),
                profile_summary=pd.DataFrame(),
                segment_summary=pd.DataFrame(),
                market_regime_summary=pd.DataFrame(),
                market_regime_policy=pd.DataFrame(),
                profile_calibration=pd.DataFrame(),
                calibration=pd.DataFrame(),
                report="",
            )

        with tempfile.TemporaryDirectory() as directory:
            result = run_all_builtin_universe_validations(
                universe_names=["semiconductors"],
                period="1y",
                output_dir=directory,
                price_loader=fake_price_loader,
                validation_runner=fake_validation_runner,
            )

        row = result.summary.iloc[0]
        self.assertIn("signal_score_min: 68 -> 65", row["suggested_threshold_changes"])
        self.assertIn("confidence_min: 65 -> 62", row["suggested_threshold_changes"])
        self.assertIn("backtest_sample_min: 12 -> 10", row["suggested_threshold_changes"])
        self.assertNotIn("backtest_win_rate_min", row["suggested_threshold_changes"])
        self.assertIn("信号分数", row["suggestion_rationale_zh"])
        self.assertEqual(profile_names, ["semiconductor"])

        config_text = render_suggested_universe_screening_config(
            screening_config=default_screening_config(),
            summary=result.summary,
        )
        self.assertIn("[profiles.semiconductor]", config_text)
        self.assertIn("signal_score_min = 65", config_text)
        self.assertIn("confidence_min = 62", config_text)
        self.assertIn("backtest_sample_min = 10", config_text)


if __name__ == "__main__":
    unittest.main()
