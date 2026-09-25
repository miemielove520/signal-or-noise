from __future__ import annotations

import unittest

import pandas as pd

from stock_selector.backtest import run_backtest
from stock_selector.config import BacktestConfig


class BacktestExecutionTest(unittest.TestCase):
    def test_weights_and_costs_start_after_execution_lag(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=4)
        prices = pd.DataFrame(
            {
                "date": dates,
                "ticker": ["AAA"] * 4,
                "open": [100.0, 110.0, 121.0, 133.1],
                "high": [100.0, 110.0, 121.0, 133.1],
                "low": [100.0, 110.0, 121.0, 133.1],
                "close": [100.0, 110.0, 121.0, 133.1],
                "adj_close": [100.0, 110.0, 121.0, 133.1],
                "volume": [1_000_000] * 4,
            }
        )
        selections = pd.DataFrame(
            {
                "date": [dates[0]],
                "ticker": ["AAA"],
                "weight": [1.0],
                "score": [1.0],
                "rank": [1.0],
            }
        )

        curve, _metrics = run_backtest(
            prices,
            selections,
            BacktestConfig(
                initial_capital=100_000,
                transaction_cost_bps=10,
                execution_lag_days=1,
            ),
        )

        self.assertEqual(curve.loc[0, "gross_exposure"], 0.0)
        self.assertEqual(curve.loc[0, "turnover"], 0.0)
        self.assertEqual(curve.loc[1, "gross_exposure"], 1.0)
        self.assertEqual(curve.loc[1, "turnover"], 1.0)
        self.assertAlmostEqual(curve.loc[1, "gross_return"], 0.10)
        self.assertAlmostEqual(curve.loc[1, "transaction_cost"], 0.001)

    def test_rejects_same_day_execution(self) -> None:
        prices = pd.DataFrame(
            {
                "date": pd.bdate_range("2024-01-02", periods=2),
                "ticker": ["AAA", "AAA"],
                "open": [100.0, 101.0],
                "high": [100.0, 101.0],
                "low": [100.0, 101.0],
                "close": [100.0, 101.0],
                "adj_close": [100.0, 101.0],
                "volume": [1_000_000, 1_000_000],
            }
        )
        selections = pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-02")],
                "ticker": ["AAA"],
                "weight": [1.0],
            }
        )

        with self.assertRaises(ValueError):
            run_backtest(prices, selections, BacktestConfig(execution_lag_days=0))


if __name__ == "__main__":
    unittest.main()
