from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from stock_selector.human_log import (
    DEFAULT_BENCHMARK,
    append_human_trade,
    fill_human_trade_outcomes,
    load_human_trades,
)


def _price_frame(dates, closes):
    return pd.DataFrame({"date": list(dates), "ticker": "X", "adj_close": list(closes)})


class HumanLogTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.review_root = Path(self._tmp.name) / "signal_review"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_append_writes_normalized_row(self) -> None:
        frame = append_human_trade(
            ticker="aapl",
            action="buy",
            weight_pct=12.5,
            signal_price=190.25,
            signal_date="2026-07-01",
            reason="momentum breakout",
            review_root=self.review_root,
        )
        self.assertEqual(len(frame), 1)
        row = frame.iloc[0]
        self.assertEqual(row["ticker"], "AAPL")
        self.assertEqual(row["action"], "BUY")
        self.assertEqual(row["actor"], "human")
        self.assertEqual(row["benchmark"], DEFAULT_BENCHMARK)
        self.assertEqual(row["signal_date"], "2026-07-01")
        self.assertAlmostEqual(float(row["weight_pct"]), 12.5)
        self.assertAlmostEqual(float(row["signal_price"]), 190.25)
        self.assertTrue((self.review_root / "human_trades.csv").exists())

    def test_default_date_is_today(self) -> None:
        frame = append_human_trade(
            ticker="MSFT",
            action="ADD",
            weight_pct=5,
            signal_price=400,
            review_root=self.review_root,
        )
        today = pd.Timestamp.today().normalize().date().isoformat()
        self.assertEqual(frame.iloc[0]["signal_date"], today)

    def test_dedup_keeps_latest_same_key(self) -> None:
        append_human_trade(
            ticker="NVDA",
            action="BUY",
            weight_pct=10,
            signal_price=100,
            signal_date="2026-07-02",
            reason="first",
            review_root=self.review_root,
        )
        frame = append_human_trade(
            ticker="NVDA",
            action="BUY",
            weight_pct=20,
            signal_price=110,
            signal_date="2026-07-02",
            reason="corrected",
            review_root=self.review_root,
        )
        nvda = frame[frame["ticker"] == "NVDA"]
        self.assertEqual(len(nvda), 1)
        self.assertAlmostEqual(float(nvda.iloc[0]["weight_pct"]), 20.0)
        self.assertEqual(nvda.iloc[0]["reason"], "corrected")

    def test_distinct_action_is_separate_row(self) -> None:
        append_human_trade(
            ticker="NVDA",
            action="BUY",
            weight_pct=10,
            signal_price=100,
            signal_date="2026-07-02",
            review_root=self.review_root,
        )
        frame = append_human_trade(
            ticker="NVDA",
            action="SELL",
            weight_pct=0,
            signal_price=130,
            signal_date="2026-07-02",
            review_root=self.review_root,
        )
        self.assertEqual(len(frame[frame["ticker"] == "NVDA"]), 2)

    def test_load_empty_returns_core_schema(self) -> None:
        frame = load_human_trades(self.review_root)
        self.assertTrue(frame.empty)
        for column in ("ticker", "action", "weight_pct", "signal_price", "benchmark"):
            self.assertIn(column, frame.columns)

    def test_invalid_action_raises(self) -> None:
        with self.assertRaises(ValueError):
            append_human_trade(
                ticker="AAPL",
                action="hold-forever",
                weight_pct=10,
                signal_price=100,
                review_root=self.review_root,
            )

    def test_out_of_range_weight_raises(self) -> None:
        with self.assertRaises(ValueError):
            append_human_trade(
                ticker="AAPL",
                action="BUY",
                weight_pct=150,
                signal_price=100,
                review_root=self.review_root,
            )

    def test_non_positive_price_raises(self) -> None:
        with self.assertRaises(ValueError):
            append_human_trade(
                ticker="AAPL",
                action="BUY",
                weight_pct=10,
                signal_price=0,
                review_root=self.review_root,
            )

    def test_invalid_benchmark_raises(self) -> None:
        with self.assertRaises(ValueError):
            append_human_trade(
                ticker="AAPL",
                action="BUY",
                weight_pct=10,
                signal_price=100,
                benchmark="BTC",
                review_root=self.review_root,
            )

    def test_empty_ticker_raises(self) -> None:
        with self.assertRaises(ValueError):
            append_human_trade(
                ticker="   ",
                action="BUY",
                weight_pct=10,
                signal_price=100,
                review_root=self.review_root,
            )


class HumanOutcomeTests(unittest.TestCase):
    def setUp(self) -> None:
        # 11 business days; window 5 completes, windows 20/60 stay pending.
        self.dates = pd.date_range("2026-01-05", periods=11, freq="B")
        # index0=100 ... index5=110 -> 5d forward return = +10%.
        self.aapl = _price_frame(self.dates, [100 + 2 * i for i in range(11)])
        # index0=200 ... index5=210 -> benchmark 5d return = +5%.
        self.qqq = _price_frame(self.dates, [200 + 2 * i for i in range(11)])

    def _trade(self, ticker="AAPL", signal_price=100.0, benchmark="QQQ"):
        return pd.DataFrame(
            [
                {
                    "signal_date": self.dates[0].date().isoformat(),
                    "ticker": ticker,
                    "actor": "human",
                    "action": "BUY",
                    "weight_pct": 10.0,
                    "signal_price": signal_price,
                    "benchmark": benchmark,
                    "reason": "test",
                    "report_path": "",
                }
            ]
        )

    def test_excess_return_and_beat_benchmark(self) -> None:
        filled = fill_human_trade_outcomes(
            self._trade(),
            price_lookup={"AAPL": self.aapl},
            benchmark_lookup={"QQQ": self.qqq},
        )
        row = filled.iloc[0]
        self.assertAlmostEqual(float(row["forward_return_5d"]), 0.10, places=6)
        self.assertAlmostEqual(float(row["benchmark_return_5d"]), 0.05, places=6)
        self.assertAlmostEqual(float(row["excess_return_5d"]), 0.05, places=6)
        self.assertEqual(row["outcome_status_5d"], "beat_benchmark")

    def test_lagged_benchmark_when_stock_underperforms(self) -> None:
        # Stock rises only +2% over 5 days while QQQ rises +5% -> lagged.
        weak = _price_frame(self.dates, [100 + 0.4 * i for i in range(11)])
        filled = fill_human_trade_outcomes(
            self._trade(ticker="WEAK"),
            price_lookup={"WEAK": weak},
            benchmark_lookup={"QQQ": self.qqq},
        )
        row = filled.iloc[0]
        self.assertLess(float(row["excess_return_5d"]), 0.0)
        self.assertEqual(row["outcome_status_5d"], "lagged_benchmark")

    def test_far_windows_pending(self) -> None:
        filled = fill_human_trade_outcomes(
            self._trade(),
            price_lookup={"AAPL": self.aapl},
            benchmark_lookup={"QQQ": self.qqq},
        )
        row = filled.iloc[0]
        self.assertEqual(row["outcome_status_20d"], "pending")
        self.assertEqual(row["outcome_status_60d"], "pending")
        # 11 days observed (indices 0..10); 20d target needs 10 more trading days.
        self.assertEqual(int(row["trading_days_remaining_20d"]), 10)

    def test_missing_benchmark_falls_back_to_absolute(self) -> None:
        filled = fill_human_trade_outcomes(
            self._trade(),
            price_lookup={"AAPL": self.aapl},
            benchmark_lookup={},
        )
        row = filled.iloc[0]
        self.assertAlmostEqual(float(row["forward_return_5d"]), 0.10, places=6)
        self.assertTrue(pd.isna(row["excess_return_5d"]))
        self.assertEqual(row["outcome_status_5d"], "up_no_benchmark")

    def test_uses_recorded_entry_price_not_close(self) -> None:
        # Entered at 90 (better than the day's close of 100) -> larger forward return.
        filled = fill_human_trade_outcomes(
            self._trade(signal_price=90.0),
            price_lookup={"AAPL": self.aapl},
            benchmark_lookup={"QQQ": self.qqq},
        )
        row = filled.iloc[0]
        self.assertAlmostEqual(float(row["forward_return_5d"]), 110 / 90 - 1, places=6)

    def test_sell_and_trim_score_the_opposite_direction(self) -> None:
        trades = pd.concat(
            [
                self._trade(),
                self._trade(),
            ],
            ignore_index=True,
        )
        trades.loc[0, "action"] = "SELL"
        trades.loc[1, "action"] = "TRIM"

        filled = fill_human_trade_outcomes(
            trades,
            price_lookup={"AAPL": self.aapl},
            benchmark_lookup={"QQQ": self.qqq},
        )

        for _, row in filled.iterrows():
            self.assertAlmostEqual(float(row["underlying_forward_return_5d"]), 0.10, places=6)
            self.assertAlmostEqual(float(row["forward_return_5d"]), -0.10, places=6)
            self.assertAlmostEqual(float(row["excess_return_5d"]), -0.05, places=6)
            self.assertEqual(row["outcome_status_5d"], "lagged_benchmark")
            self.assertEqual(int(row["decision_direction"]), -1)


if __name__ == "__main__":
    unittest.main()
