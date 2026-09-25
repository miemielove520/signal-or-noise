from __future__ import annotations

import unittest

import pandas as pd

from stock_selector.paper_tracker import (
    mark_to_market,
    record_equity_point,
    summarize_paper_performance,
)


class MarkToMarketTests(unittest.TestCase):
    def test_cash_plus_positions(self) -> None:
        state = pd.DataFrame(
            [{"ticker": "AAPL", "quantity": 10}, {"ticker": "CASH", "quantity": 500.0}]
        )
        equity = mark_to_market(state, {"AAPL": 100.0})
        self.assertEqual(equity, 1500.0)  # 10*100 + 500

    def test_missing_price_uses_last_known_fallback(self) -> None:
        state = pd.DataFrame([{"ticker": "ZZZZ", "quantity": 5}, {"ticker": "CASH", "quantity": 100.0}])
        self.assertEqual(mark_to_market(state, {}, {"ZZZZ": 20.0}), 200.0)

    def test_missing_price_without_fallback_fails_loudly(self) -> None:
        state = pd.DataFrame([{"ticker": "ZZZZ", "quantity": 5}, {"ticker": "CASH", "quantity": 100.0}])
        with self.assertRaisesRegex(ValueError, "ZZZZ"):
            mark_to_market(state, {})

    def test_empty_state(self) -> None:
        self.assertEqual(mark_to_market(pd.DataFrame(), {"AAPL": 100.0}), 0.0)


class EquityHistoryTests(unittest.TestCase):
    def test_append_and_replace_same_date(self) -> None:
        h = record_equity_point(None, "2026-07-01", 100000.0, 500.0)
        h = record_equity_point(h, "2026-07-02", 101000.0, 505.0)
        self.assertEqual(len(h), 2)
        # Re-recording the same date replaces, not duplicates.
        h = record_equity_point(h, "2026-07-02", 101500.0, 506.0)
        self.assertEqual(len(h), 2)
        self.assertEqual(float(h[h["date"] == "2026-07-02"]["equity"].iloc[0]), 101500.0)


class PerformanceTests(unittest.TestCase):
    def _history(self, equities, benches):
        dates = [f"2026-07-{i + 1:02d}" for i in range(len(equities))]
        return pd.DataFrame({"date": dates, "equity": equities, "benchmark_close": benches})

    def test_insufficient_days_is_accumulating(self) -> None:
        perf = summarize_paper_performance(self._history([100000, 100500], [500, 501]), min_days=20)
        self.assertEqual(perf.readiness, "accumulating")
        self.assertEqual(perf.days_tracked, 2)

    def test_on_track_when_beating_benchmark(self) -> None:
        # 25 days, paper +5%, benchmark +2% -> excess positive -> on_track.
        equities = [100000 * (1 + 0.05 * i / 24) for i in range(25)]
        benches = [500 * (1 + 0.02 * i / 24) for i in range(25)]
        perf = summarize_paper_performance(self._history(equities, benches), min_days=20)
        self.assertEqual(perf.readiness, "on_track")
        self.assertGreater(perf.excess_return, 0)

    def test_not_ready_when_deep_drawdown(self) -> None:
        # Rises then crashes -30% -> not_ready.
        equities = [100000 + 200 * i for i in range(15)] + [103000 * (1 - 0.30 * i / 9) for i in range(1, 11)]
        benches = [500] * len(equities)
        perf = summarize_paper_performance(self._history(equities, benches), min_days=20)
        self.assertEqual(perf.readiness, "not_ready")
        self.assertLess(perf.max_drawdown, -0.25)

    def test_empty_history(self) -> None:
        self.assertEqual(summarize_paper_performance(pd.DataFrame()).readiness, "accumulating")


if __name__ == "__main__":
    unittest.main()


class BacktestGateTests(unittest.TestCase):
    def _perf(self, days, total_return):
        from stock_selector.paper_tracker import PaperPerformance
        return PaperPerformance(
            status="", status_zh="", days_tracked=days, start_date="a", latest_date="b",
            start_equity=1000.0, latest_equity=1000.0 * (1 + total_return),
            total_return=total_return, benchmark_return=None, excess_return=None,
            max_drawdown=-0.05, readiness="", readiness_zh="", note="", note_zh="",
        )

    def test_expected_annual_from_walk_forward(self) -> None:
        from stock_selector.paper_tracker import expected_annual_from_walk_forward
        summary = pd.DataFrame({"avg_return_20d": [0.02, 0.03], "sample_count": [10, 30]})
        exp = expected_annual_from_walk_forward(summary, window=20)
        # sample-weighted per-20d = (0.02*10+0.03*30)/40 = 0.0275, annualized
        self.assertAlmostEqual(exp, (1.0275) ** (252 / 20) - 1, places=4)

    def test_gate_divergent_when_backtest_positive_but_paper_losing(self) -> None:
        from stock_selector.paper_tracker import compare_paper_to_backtest
        r = compare_paper_to_backtest(self._perf(30, -0.05), expected_annual_return=0.30, min_days=20)
        self.assertEqual(r["gate"], "divergent")

    def test_gate_consistent_when_matching(self) -> None:
        from stock_selector.paper_tracker import compare_paper_to_backtest
        # 30 days +3% -> annualized ~29% vs expected 30% -> consistent
        r = compare_paper_to_backtest(self._perf(30, 0.03), expected_annual_return=0.30, min_days=20)
        self.assertEqual(r["gate"], "consistent")

    def test_gate_below_expectation(self) -> None:
        from stock_selector.paper_tracker import compare_paper_to_backtest
        # tiny positive, well under 40% of expected
        r = compare_paper_to_backtest(self._perf(60, 0.005), expected_annual_return=0.40, min_days=20)
        self.assertEqual(r["gate"], "below_expectation")

    def test_gate_accumulating_when_too_few_days(self) -> None:
        from stock_selector.paper_tracker import compare_paper_to_backtest
        r = compare_paper_to_backtest(self._perf(5, 0.02), expected_annual_return=0.30, min_days=20)
        self.assertEqual(r["gate"], "accumulating")

    def test_gate_no_backtest(self) -> None:
        from stock_selector.paper_tracker import compare_paper_to_backtest
        r = compare_paper_to_backtest(self._perf(30, 0.03), expected_annual_return=None)
        self.assertEqual(r["gate"], "no_backtest")
