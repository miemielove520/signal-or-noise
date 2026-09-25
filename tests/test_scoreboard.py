from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from stock_selector.human_log import fill_human_trade_outcomes
from stock_selector.scoreboard import (
    build_head_to_head,
    compute_outcome_metrics,
    prepare_model_outcomes,
    render_head_to_head_markdown,
    summarize_human_scoreboard,
)


class ComputeOutcomeMetricsTests(unittest.TestCase):
    def test_known_series(self) -> None:
        returns = [0.05, -0.02, 0.03, -0.04, 0.01]
        metrics = compute_outcome_metrics(returns)
        self.assertEqual(metrics["completed_count"], 5)
        self.assertEqual(metrics["beat_count"], 3)
        self.assertEqual(metrics["lag_count"], 2)
        self.assertAlmostEqual(metrics["beat_rate"], 0.6, places=6)
        self.assertAlmostEqual(metrics["expectancy"], 0.006, places=6)
        self.assertAlmostEqual(metrics["avg_win"], 0.03, places=6)
        self.assertAlmostEqual(metrics["avg_loss"], -0.03, places=6)
        self.assertAlmostEqual(metrics["payoff_ratio"], 1.0, places=6)
        self.assertAlmostEqual(metrics["profit_factor"], 1.5, places=6)
        self.assertAlmostEqual(metrics["max_drawdown"], -0.04, places=6)

    def test_all_wins_has_no_drawdown(self) -> None:
        metrics = compute_outcome_metrics([0.01, 0.02, 0.03])
        self.assertEqual(metrics["lag_count"], 0)
        self.assertAlmostEqual(metrics["max_drawdown"], 0.0, places=6)
        self.assertTrue(np.isnan(metrics["payoff_ratio"]))
        self.assertTrue(np.isnan(metrics["profit_factor"]))

    def test_high_win_rate_can_still_lose(self) -> None:
        # 4 small wins, 1 huge loss -> positive win rate, negative expectancy.
        metrics = compute_outcome_metrics([0.01, 0.01, 0.01, 0.01, -0.20])
        self.assertAlmostEqual(metrics["beat_rate"], 0.8, places=6)
        self.assertLess(metrics["expectancy"], 0.0)
        self.assertLess(metrics["profit_factor"], 1.0)

    def test_order_affects_drawdown(self) -> None:
        returns = [0.05, -0.04, -0.03, 0.02]
        dates = ["2026-01-10", "2026-01-01", "2026-01-02", "2026-01-03"]
        # Time-ordered: -0.04, -0.03, 0.02, 0.05 -> trough at -0.07.
        metrics = compute_outcome_metrics(returns, order=dates)
        self.assertAlmostEqual(metrics["max_drawdown"], -0.07, places=6)

    def test_empty(self) -> None:
        metrics = compute_outcome_metrics([])
        self.assertEqual(metrics["completed_count"], 0)
        self.assertTrue(np.isnan(metrics["expectancy"]))


class HumanScoreboardTests(unittest.TestCase):
    def test_scoreboard_shape_and_values(self) -> None:
        trades = pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "excess_return_5d": 0.04,
                    "outcome_status_5d": "beat_benchmark",
                    "outcome_date_5d": "2026-01-10",
                    "excess_return_20d": np.nan,
                    "outcome_status_20d": "pending",
                },
                {
                    "ticker": "MSFT",
                    "excess_return_5d": -0.02,
                    "outcome_status_5d": "lagged_benchmark",
                    "outcome_date_5d": "2026-01-11",
                    "excess_return_20d": np.nan,
                    "outcome_status_20d": "pending",
                },
            ]
        )
        board = summarize_human_scoreboard(trades)
        self.assertListEqual(list(board["window"]), ["5d", "20d", "60d"])
        five = board[board["window"] == "5d"].iloc[0]
        self.assertEqual(five["completed_count"], 2)
        self.assertEqual(five["beat_count"], 1)
        self.assertAlmostEqual(five["expectancy"], 0.01, places=6)
        twenty = board[board["window"] == "20d"].iloc[0]
        self.assertEqual(twenty["completed_count"], 0)
        self.assertEqual(twenty["pending_count"], 2)

    def test_empty_trades(self) -> None:
        board = summarize_human_scoreboard(pd.DataFrame())
        self.assertEqual(len(board), 3)
        self.assertEqual(int(board.iloc[0]["completed_count"]), 0)


def _price_frame(dates, closes):
    return pd.DataFrame({"date": list(dates), "ticker": "X", "adj_close": list(closes)})


class HeadToHeadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dates = pd.date_range("2026-01-05", periods=11, freq="B")
        # Strong stock (+10% over 5d), weak stock (+1% over 5d), benchmark (+5%).
        self.strong = _price_frame(self.dates, [100 + 2 * i for i in range(11)])
        self.weak = _price_frame(self.dates, [100 + 0.2 * i for i in range(11)])
        self.qqq = _price_frame(self.dates, [200 + 2 * i for i in range(11)])
        self.price_lookup = {"STRONG": self.strong, "WEAK": self.weak}
        self.benchmark_lookup = {"QQQ": self.qqq}

    def _trade(self, ticker, signal_price):
        return {
            "signal_date": self.dates[0].date().isoformat(),
            "ticker": ticker,
            "signal_price": signal_price,
            "benchmark": "QQQ",
        }

    def test_human_beats_model_when_picking_stronger_stock(self) -> None:
        human = fill_human_trade_outcomes(
            pd.DataFrame([self._trade("STRONG", 100.0)]),
            self.price_lookup,
            self.benchmark_lookup,
        )
        # Model signal lacks a benchmark column on purpose -> prepare adds one.
        model_signals = pd.DataFrame(
            [{"signal_date": self.dates[0].date().isoformat(), "ticker": "WEAK", "signal_price": 100.0}]
        )
        model = prepare_model_outcomes(model_signals, self.price_lookup, self.benchmark_lookup)
        h2h = build_head_to_head(human, model)
        five = h2h[h2h["window"] == "5d"].iloc[0]
        self.assertEqual(five["human_completed"], 1)
        self.assertEqual(five["model_completed"], 1)
        self.assertEqual(five["leader"], "human")
        self.assertGreater(five["expectancy_edge"], 0.0)

    def test_leader_insufficient_without_both_samples(self) -> None:
        human = fill_human_trade_outcomes(
            pd.DataFrame([self._trade("STRONG", 100.0)]),
            self.price_lookup,
            self.benchmark_lookup,
        )
        h2h = build_head_to_head(human, pd.DataFrame())
        five = h2h[h2h["window"] == "5d"].iloc[0]
        self.assertEqual(five["leader"], "insufficient")

    def test_render_markdown_has_headline(self) -> None:
        human = fill_human_trade_outcomes(
            pd.DataFrame([self._trade("STRONG", 100.0)]),
            self.price_lookup,
            self.benchmark_lookup,
        )
        model = prepare_model_outcomes(
            pd.DataFrame([self._trade("WEAK", 100.0)]),
            self.price_lookup,
            self.benchmark_lookup,
        )
        text = render_head_to_head_markdown(build_head_to_head(human, model))
        self.assertIn("Human vs Model Scoreboard", text)
        self.assertIn("Verdict", text)

    def test_render_empty(self) -> None:
        text = render_head_to_head_markdown(build_head_to_head(pd.DataFrame(), pd.DataFrame()))
        self.assertIn("人机对照记分台", text)


if __name__ == "__main__":
    unittest.main()
