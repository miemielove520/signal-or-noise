from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from stock_selector.signal_review import (
    build_signal_review_feedback_context,
    render_signal_review_section,
    scan_signal_review_due_items,
    summarize_signal_history,
    summarize_ticker_signal_review,
    write_signal_review,
)


class SignalReviewTest(unittest.TestCase):
    def test_write_signal_review_records_and_updates_forward_returns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prices_short = make_prices(periods=20)
            prices_long = make_prices(periods=90)
            analysis_short = make_analysis("AAA", prices_short["adj_close"].iloc[-1])

            first = write_signal_review(
                analysis=analysis_short,
                prices=prices_short,
                review_root=root,
                source_report_path="outputs/real_ticker/AAA/ticker_analysis.md",
            )
            self.assertTrue(first.history_path.exists())
            self.assertTrue(first.report_path.exists())
            self.assertEqual(len(first.ticker_history), 1)
            self.assertEqual(first.ticker_history.iloc[0]["outcome_status_5d"], "pending")
            self.assertEqual(int(first.summary.iloc[0]["pending_5d_count"]), 1)
            self.assertEqual(int(first.summary.iloc[0]["min_trading_days_remaining_5d"]), 5)
            self.assertNotEqual(first.summary.iloc[0]["next_estimated_review_date_5d"], "")

            analysis_long = make_analysis("AAA", prices_long["adj_close"].iloc[-1])
            second = write_signal_review(
                analysis=analysis_long,
                prices=prices_long,
                review_root=root,
                source_report_path="outputs/real_ticker/AAA/ticker_analysis.md",
            )

            old_signal = second.ticker_history.iloc[0]
            self.assertGreater(float(old_signal["forward_return_5d"]), 0.0)
            self.assertGreater(float(old_signal["forward_return_20d"]), 0.0)
            self.assertGreater(float(old_signal["forward_return_60d"]), 0.0)
            self.assertEqual(int(old_signal["trading_days_remaining_5d"]), 0)
            self.assertEqual(old_signal["outcome_status_5d"], "win")
            self.assertTrue((root / "signal_review_summary.csv").exists())

    def test_write_signal_review_deduplicates_same_ticker_date_horizon(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prices = make_prices(periods=30)
            analysis = make_analysis("AAA", prices["adj_close"].iloc[-1])

            write_signal_review(analysis=analysis, prices=prices, review_root=root)
            result = write_signal_review(analysis=analysis, prices=prices, review_root=root)

            self.assertEqual(len(result.ticker_history), 1)

    def test_summarize_signal_history_calculates_win_rate(self) -> None:
        history = pd.DataFrame(
            [
                {"ticker": "AAA", "forward_return_5d": 0.05},
                {"ticker": "AAA", "forward_return_5d": -0.02},
            ]
        )

        summary = summarize_signal_history(history)

        self.assertEqual(int(summary.iloc[0]["signal_count"]), 2)
        self.assertEqual(int(summary.iloc[0]["completed_5d_count"]), 2)
        self.assertEqual(int(summary.iloc[0]["pending_5d_count"]), 0)
        self.assertAlmostEqual(float(summary.iloc[0]["win_rate_5d"]), 0.5)
        self.assertIn("review_learning_score", summary.columns)
        self.assertEqual(summary.iloc[0]["review_learning_level"], "insufficient_history")

    def test_summarize_signal_history_counts_pending_outcomes(self) -> None:
        history = pd.DataFrame(
            [
                {"ticker": "AAA", "forward_return_5d": 0.05, "outcome_status_5d": "win"},
                {"ticker": "AAA", "forward_return_5d": None, "outcome_status_5d": "pending"},
                {"ticker": "AAA", "forward_return_5d": None, "outcome_status_5d": ""},
            ]
        )

        summary = summarize_signal_history(history)

        self.assertEqual(int(summary.iloc[0]["completed_5d_count"]), 1)
        self.assertEqual(int(summary.iloc[0]["pending_5d_count"]), 2)

    def test_summarize_signal_history_reports_next_pending_review_date(self) -> None:
        history = pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "forward_return_5d": None,
                    "outcome_status_5d": "pending",
                    "trading_days_remaining_5d": 3,
                    "estimated_review_date_5d": "2026-01-08",
                },
                {
                    "ticker": "AAA",
                    "forward_return_5d": None,
                    "outcome_status_5d": "pending",
                    "trading_days_remaining_5d": 1,
                    "estimated_review_date_5d": "2026-01-06",
                },
            ]
        )

        summary = summarize_signal_history(history)

        self.assertEqual(int(summary.iloc[0]["min_trading_days_remaining_5d"]), 1)
        self.assertEqual(summary.iloc[0]["next_estimated_review_date_5d"], "2026-01-06")

    def test_ticker_signal_review_summary_and_section_are_bilingual(self) -> None:
        ticker_history = pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "signal_date": "2026-01-01",
                    "forward_return_5d": 0.05,
                    "forward_return_20d": 0.08,
                    "forward_return_60d": -0.02,
                }
            ]
        )
        summary = summarize_signal_history(ticker_history)

        metrics = summarize_ticker_signal_review(ticker_history, summary)
        section = render_signal_review_section(
            ticker_history,
            summary,
            "outputs/signal_review/signal_review.md",
        )

        self.assertEqual(metrics["ticker_signal_count"], 1)
        self.assertEqual(metrics["completed_5d_count"], 1)
        self.assertEqual(metrics["pending_5d_count"], 0)
        self.assertEqual(metrics["min_trading_days_remaining_5d"], 0)
        self.assertEqual(metrics["win_rate_5d"], 1.0)
        self.assertIn(
            metrics["review_learning_level"],
            {"supportive", "slightly_supportive", "neutral", "weak", "poor", "insufficient_history"},
        )
        self.assertIn("Signal Review / 信号复盘", section)
        self.assertIn("Review learning score / 复盘学习分数", section)
        self.assertIn("Review learning level / 复盘学习等级", section)
        self.assertIn("5 trading days / 5个交易日", section)
        self.assertIn("pending samples / 等待中样本", section)
        self.assertIn("next review estimate / 预计下次可复盘", section)
        self.assertIn("Full review report / 完整复盘报告", section)

    def test_review_learning_state_scores_completed_signal_history(self) -> None:
        history = pd.DataFrame(
            [
                {"ticker": "AAA", "forward_return_5d": 0.04, "forward_return_20d": 0.07},
                {"ticker": "AAA", "forward_return_5d": 0.03, "forward_return_20d": 0.06},
                {"ticker": "AAA", "forward_return_5d": -0.01, "forward_return_20d": 0.02},
                {"ticker": "AAA", "forward_return_5d": 0.02, "forward_return_20d": 0.04},
            ]
        )

        summary = summarize_signal_history(history)
        row = summary.iloc[0]

        self.assertEqual(int(row["review_learning_sample_count"]), 8)
        self.assertGreater(float(row["review_learning_score"]), 50.0)
        self.assertGreater(float(row["review_learning_adjustment"]), 0.0)
        self.assertEqual(row["review_learning_focus_window"], "20d")
        self.assertIn(row["review_learning_level"], {"slightly_supportive", "supportive"})
        self.assertIn("复盘学习状态使用8个已完成结果", row["review_learning_note_zh"])

    def test_build_signal_review_feedback_context_scores_prior_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = pd.DataFrame(
                [
                    {"ticker": "AAA", "forward_return_5d": 0.04, "forward_return_20d": 0.08},
                    {"ticker": "AAA", "forward_return_5d": 0.03, "forward_return_20d": 0.06},
                    {"ticker": "AAA", "forward_return_5d": -0.01, "forward_return_20d": 0.02},
                    {"ticker": "AAA", "forward_return_5d": 0.02, "forward_return_20d": 0.04},
                ]
            )
            history.to_csv(root / "signal_history.csv", index=False)

            feedback = build_signal_review_feedback_context("AAA", review_root=root)

            self.assertEqual(feedback.status, "ok")
            self.assertGreater(feedback.score, 50.0)
            self.assertGreater(feedback.adjustment, 0.0)
            self.assertEqual(feedback.sample_count, 8)
            self.assertIn(feedback.level, {"slightly_supportive", "supportive"})

    def test_scan_signal_review_due_items_writes_due_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = pd.DataFrame(
                [
                    {
                        "ticker": "AAA",
                        "signal_date": "2026-01-01",
                        "focus_horizon": "short",
                        "estimated_review_date_5d": "2026-01-08",
                        "trading_days_remaining_5d": 0,
                        "outcome_status_5d": "pending",
                        "report_path": "outputs/real_ticker/AAA/ticker_analysis.md",
                    },
                    {
                        "ticker": "BBB",
                        "signal_date": "2026-01-01",
                        "focus_horizon": "long",
                        "estimated_review_date_5d": "2026-02-01",
                        "trading_days_remaining_5d": 10,
                        "outcome_status_5d": "pending",
                        "report_path": "outputs/real_ticker/BBB/ticker_analysis.md",
                    },
                    {
                        "ticker": "CCC",
                        "signal_date": "2026-01-01",
                        "focus_horizon": "medium",
                        "forward_return_5d": 0.05,
                        "outcome_status_5d": "win",
                    },
                ]
            )
            history.to_csv(root / "signal_history.csv", index=False)

            result = scan_signal_review_due_items(
                review_root=root,
                output_dir=root,
                as_of_date="2026-01-10",
            )

            self.assertTrue(result.csv_path.exists())
            self.assertTrue(result.report_path.exists())
            self.assertTrue(result.json_path.exists())
            due_now = result.due_items[result.due_items["review_status"] == "due_now"]
            pending = result.due_items[result.due_items["review_status"] == "pending"]
            pending_unknown = result.due_items[result.due_items["review_status"] == "pending_unknown"]
            self.assertEqual(len(due_now), 1)
            self.assertEqual(due_now.iloc[0]["ticker"], "AAA")
            self.assertEqual(
                due_now.iloc[0]["recommended_command"],
                "python3 run.py AAA --no-snapshot --no-peers",
            )
            self.assertEqual(len(pending), 1)
            self.assertGreaterEqual(len(pending_unknown), 1)
            self.assertEqual(int(result.summary.iloc[0]["due_now_count"]), 1)
            report = result.report_path.read_text()
            self.assertIn("Signal Review Due Scan / 信号复盘到期扫描", report)
            self.assertIn("recommended_command", report)


def make_prices(periods: int) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-01", periods=periods)
    values = [100.0 + index for index in range(periods)]
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ["AAA"] * periods,
            "open": values,
            "high": [value + 1 for value in values],
            "low": [value - 1 for value in values],
            "close": values,
            "adj_close": values,
            "volume": [1_000_000] * periods,
        }
    )


def make_analysis(ticker: str, latest_price: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": ticker,
                "horizon": "short",
                "final_focus_horizon": "short",
                "final_decision": "near_watchlist",
                "final_decision_zh": "接近机会",
                "calibrated_watchlist_status": "close_but_not_ready",
                "calibrated_screening_action": "not_high_probability_setup_now",
                "calibrated_quality_gate_passed": False,
                "latest_price": latest_price,
                "calibrated_high_probability_score": 62.0,
                "high_probability_score": 60.0,
                "signal_score": 64.0,
                "calibrated_win_probability": 0.52,
                "primary_blocker": "entry_not_executable",
                "calibrated_quality_gate_fail_reasons_zh": "买点当前不可执行",
            }
        ]
    )


if __name__ == "__main__":
    unittest.main()
