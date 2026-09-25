from __future__ import annotations

import unittest

import pandas as pd

from stock_selector.market import (
    build_market_context,
    build_relative_strength_context,
    build_sector_context,
    choose_sector_etf,
)


def make_price_frame(ticker: str, start_price: float, end_price: float, rows: int = 140) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=rows)
    step = (end_price - start_price) / (rows - 1)
    prices = [start_price + step * index for index in range(rows)]
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": [ticker] * rows,
            "open": prices,
            "high": [price * 1.01 for price in prices],
            "low": [price * 0.99 for price in prices],
            "close": prices,
            "adj_close": prices,
            "volume": [1_000_000] * rows,
        }
    )


class MarketContextTest(unittest.TestCase):
    def test_market_score_uses_spy_qqq_and_vix(self) -> None:
        context = build_market_context(
            {
                "SPY": make_price_frame("SPY", 100, 125),
                "QQQ": make_price_frame("QQQ", 100, 130),
                "^VIX": make_price_frame("^VIX", 18, 14),
            }
        )

        self.assertGreater(context.market_score, 65)
        self.assertEqual(context.market_status, "supportive")
        self.assertGreater(context.spy_trend_score, 50)
        self.assertGreater(context.qqq_trend_score, 50)
        self.assertGreater(context.vix_risk_score, 50)

    def test_market_score_falls_back_to_neutral_when_benchmarks_are_missing(self) -> None:
        context = build_market_context({})

        self.assertEqual(context.market_score, 50.0)
        self.assertEqual(context.market_status, "neutral")
        self.assertIn("SPY", context.warning)
        self.assertIn("QQQ", context.warning)
        self.assertIn("VIX", context.warning)

    def test_relative_strength_score_rewards_benchmark_outperformance(self) -> None:
        context = build_relative_strength_context(
            target_prices=make_price_frame("AAA", 100, 170, rows=80),
            spy_prices=make_price_frame("SPY", 100, 104, rows=80),
            qqq_prices=make_price_frame("QQQ", 100, 105, rows=80),
            lookback_days=20,
        )

        self.assertGreater(context.score, 65)
        self.assertGreater(context.vs_spy_return, 0)
        self.assertGreater(context.vs_qqq_return, 0)

    def test_relative_strength_score_falls_back_to_neutral_when_missing(self) -> None:
        context = build_relative_strength_context(
            target_prices=make_price_frame("AAA", 100, 110, rows=80),
            spy_prices=None,
            qqq_prices=None,
            lookback_days=20,
        )

        self.assertEqual(context.score, 50.0)
        self.assertIn("SPY", context.warning)
        self.assertIn("QQQ", context.warning)

    def test_choose_sector_etf_maps_sector_and_industry(self) -> None:
        self.assertEqual(choose_sector_etf("Technology", "Consumer Electronics"), "XLK")
        self.assertEqual(choose_sector_etf("Technology", "Semiconductors"), "SMH")

    def test_sector_context_scores_supportive_sector(self) -> None:
        context = build_sector_context(
            target_prices=make_price_frame("AAA", 100, 130, rows=100),
            sector_prices=make_price_frame("XLK", 100, 125, rows=140),
            sector="Technology",
            industry="Software",
            sector_etf="XLK",
            lookback_days=20,
        )

        self.assertEqual(context.sector_etf, "XLK")
        self.assertGreater(context.sector_score, 55)
        self.assertIn(context.sector_status, {"supportive", "neutral"})

    def test_sector_context_falls_back_to_neutral_when_missing(self) -> None:
        context = build_sector_context(
            target_prices=make_price_frame("AAA", 100, 130, rows=100),
            sector_prices=None,
            sector="",
            industry="",
            sector_etf="",
        )

        self.assertEqual(context.sector_score, 50.0)
        self.assertEqual(context.sector_status, "unknown")


if __name__ == "__main__":
    unittest.main()
