from __future__ import annotations

import unittest

import pandas as pd

from stock_selector.config import FactorConfig, UniverseConfig
from stock_selector.factors import add_basic_factors, add_fundamental_factors


class FactorFundamentalTest(unittest.TestCase):
    def test_sec_pit_fundamentals_merge_on_filing_date(self) -> None:
        prices = pd.DataFrame(
            [
                _price_row("2025-01-01", 100.0),
                _price_row("2025-01-02", 101.0),
                _price_row("2025-01-03", 102.0),
                _price_row("2025-01-06", 103.0),
                _price_row("2025-01-07", 104.0),
            ]
        )
        fundamentals = pd.DataFrame(
            [
                _fundamental_row("2025-01-03", "2024-09-30", 100.0, "sec_pit"),
                _fundamental_row("2025-01-07", "2025-09-30", 200.0, "sec_pit"),
            ]
        )

        result = add_fundamental_factors(prices, fundamentals)

        before_filing = result[result["date"] == pd.Timestamp("2025-01-02")].iloc[0]
        first_filing = result[result["date"] == pd.Timestamp("2025-01-03")].iloc[0]
        second_filing = result[result["date"] == pd.Timestamp("2025-01-07")].iloc[0]

        self.assertTrue(pd.isna(before_filing["report_date"]))
        self.assertEqual(first_filing["fundamentals_source"], "sec_pit")
        self.assertEqual(first_filing["report_date"].date().isoformat(), "2025-01-03")
        self.assertEqual(second_filing["report_date"].date().isoformat(), "2025-01-07")

    def test_missing_fundamental_source_defaults_to_yfinance_restated(self) -> None:
        prices = pd.DataFrame([_price_row("2025-01-03", 100.0)])
        fundamentals = pd.DataFrame([_fundamental_row("2025-01-03", "2024-09-30", 100.0, None)])
        fundamentals = fundamentals.drop(columns=["fundamentals_source"])

        result = add_fundamental_factors(prices, fundamentals)

        self.assertEqual(result["fundamentals_source"].iloc[0], "yfinance_restated")

    def test_sec_pit_source_is_preferred_over_restated_same_period(self) -> None:
        prices = pd.DataFrame([_price_row("2025-01-03", 100.0)])
        fundamentals = pd.DataFrame(
            [
                _fundamental_row("2024-09-30", "2024-09-30", 999.0, "yfinance_restated"),
                _fundamental_row("2025-01-03", "2024-09-30", 100.0, "sec_pit"),
            ]
        )

        result = add_fundamental_factors(prices, fundamentals)

        self.assertEqual(result["fundamentals_source"].iloc[0], "sec_pit")
        self.assertEqual(result["report_date"].iloc[0].date().isoformat(), "2025-01-03")


class MoneyFlowFactorTest(unittest.TestCase):
    def test_money_flow_matches_chaikin_formula(self) -> None:
        prices = pd.DataFrame(
            [
                _ohlc_row("2025-01-02", high=10.0, low=8.0, close=9.0, volume=100),
                _ohlc_row("2025-01-03", high=12.0, low=10.0, close=11.5, volume=200),
                _ohlc_row("2025-01-06", high=11.0, low=9.0, close=9.5, volume=300),
            ]
        )

        result = add_basic_factors(prices, FactorConfig(money_flow_window=3), UniverseConfig())

        last = result[result["date"] == pd.Timestamp("2025-01-06")].iloc[0]
        # Money flow multipliers: 0.0, +0.5, -0.5.
        # MFV sum = 0*100 + 0.5*200 + (-0.5)*300 = -50; volume sum = 600.
        self.assertAlmostEqual(last["money_flow_3"], -50.0 / 600.0)
        # The first two rows lack a full window and stay NaN.
        self.assertTrue(pd.isna(result["money_flow_3"].iloc[0]))
        self.assertTrue(pd.isna(result["money_flow_3"].iloc[1]))

    def test_money_flow_handles_flat_bars_and_stays_bounded(self) -> None:
        prices = pd.DataFrame(
            [
                _ohlc_row("2025-01-02", high=5.0, low=5.0, close=5.0, volume=100),
                _ohlc_row("2025-01-03", high=6.0, low=4.0, close=6.0, volume=100),
            ]
        )

        result = add_basic_factors(prices, FactorConfig(money_flow_window=2), UniverseConfig())

        last = result[result["date"] == pd.Timestamp("2025-01-03")].iloc[0]
        # Flat bar (high == low) contributes 0; the second bar closes at its high (+1).
        # CMF = (0 + 100) / (100 + 100) = 0.5, safely within [-1, 1].
        self.assertAlmostEqual(last["money_flow_2"], 0.5)
        self.assertLessEqual(abs(last["money_flow_2"]), 1.0)


def _ohlc_row(
    date: str, high: float, low: float, close: float, volume: int
) -> dict[str, object]:
    return {
        "date": pd.Timestamp(date),
        "ticker": "AAA",
        "open": close,
        "high": high,
        "low": low,
        "close": close,
        "adj_close": close,
        "volume": volume,
    }


def _price_row(date: str, close: float) -> dict[str, object]:
    return {
        "date": pd.Timestamp(date),
        "ticker": "AAA",
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "adj_close": close,
        "volume": 1_000_000,
    }


def _fundamental_row(
    report_date: str,
    period_end: str,
    revenue: float,
    source: str | None,
) -> dict[str, object]:
    row = {
        "report_date": pd.Timestamp(report_date),
        "period_end": pd.Timestamp(period_end),
        "ticker": "AAA",
        "revenue": revenue,
        "gross_profit": revenue * 0.5,
        "operating_income": revenue * 0.3,
        "net_income": revenue * 0.2,
        "book_value": 80.0,
        "total_assets": 200.0,
        "total_liabilities": 120.0,
        "operating_cash_flow": revenue * 0.25,
        "capital_expenditure": revenue * 0.05,
        "shares_outstanding": 10.0,
        "fundamentals_source": source,
    }
    return row


if __name__ == "__main__":
    unittest.main()
