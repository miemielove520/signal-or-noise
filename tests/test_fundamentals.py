from __future__ import annotations

import unittest

from stock_selector.fundamentals import build_fundamental_context


class FundamentalContextTest(unittest.TestCase):
    def test_strong_fundamentals_score_high(self) -> None:
        context = build_fundamental_context(
            "AAPL",
            {
                "data_source": "test",
                "revenue_growth": 0.12,
                "earnings_growth": 0.18,
                "profit_margin": 0.24,
                "return_on_equity": 0.35,
                "free_cash_flow": 100_000_000.0,
                "forward_pe": 22.0,
                "peg_ratio": 1.4,
                "debt_to_equity": 80.0,
            },
        )

        self.assertGreaterEqual(context.fundamental_score, 65)
        self.assertIn(context.fundamental_quality, {"good", "strong"})
        self.assertEqual(context.fundamental_warning, "")

    def test_missing_fundamentals_use_neutral_unknown(self) -> None:
        context = build_fundamental_context("XYZ", None)

        self.assertEqual(context.fundamental_score, 50.0)
        self.assertEqual(context.fundamental_quality, "unknown")
        self.assertEqual(context.fundamental_quality_zh, "未知")
        self.assertIn("snapshot unavailable", context.fundamental_warning)

    def test_weak_fundamentals_score_low(self) -> None:
        context = build_fundamental_context(
            "WEAK",
            {
                "data_source": "test",
                "revenue_growth": -0.10,
                "earnings_growth": -0.20,
                "profit_margin": -0.05,
                "return_on_equity": -0.15,
                "free_cash_flow": -10_000_000.0,
                "forward_pe": 80.0,
                "peg_ratio": 5.0,
                "debt_to_equity": 500.0,
            },
        )

        self.assertLessEqual(context.fundamental_score, 35)
        self.assertEqual(context.fundamental_quality, "weak")


    def test_weak_cash_conversion_lowers_score_vs_strong(self) -> None:
        base = {
            "data_source": "test",
            "revenue_growth": 0.12,
            "earnings_growth": 0.15,
            "profit_margin": 0.20,
            "return_on_equity": 0.25,
            "free_cash_flow": 50_000_000.0,
            "forward_pe": 22.0,
            "peg_ratio": 1.5,
            "debt_to_equity": 80.0,
            "revenue": 1_000_000_000.0,
            "net_income": 200_000_000.0,
        }
        # Same reported profit; one converts profit to cash, one does not.
        strong_cash = build_fundamental_context("GOOD", {**base, "operating_cash_flow": 240_000_000.0})
        weak_cash = build_fundamental_context("POOR", {**base, "operating_cash_flow": 40_000_000.0})

        self.assertGreater(strong_cash.cash_conversion, 1.0)
        self.assertLess(weak_cash.cash_conversion, 0.5)
        self.assertGreater(strong_cash.fundamental_score, weak_cash.fundamental_score)

    def test_net_cash_scores_better_than_net_debt(self) -> None:
        base = {
            "data_source": "test",
            "profit_margin": 0.15,
            "return_on_equity": 0.18,
            "free_cash_flow": 20_000_000.0,
            "revenue_growth": 0.08,
            "debt_to_equity": 60.0,
            "shareholders_equity": 500_000_000.0,
        }
        net_cash = build_fundamental_context(
            "CASH", {**base, "total_cash": 300_000_000.0, "total_debt": 100_000_000.0}
        )
        net_debt = build_fundamental_context(
            "DEBT", {**base, "total_cash": 50_000_000.0, "total_debt": 900_000_000.0}
        )
        self.assertLess(net_cash.net_debt, 0)  # net cash
        self.assertGreater(net_debt.net_debt_to_equity, 1.0)
        self.assertGreater(net_cash.fundamental_score, net_debt.fundamental_score)

    def test_depth_fields_default_when_absent(self) -> None:
        context = build_fundamental_context(
            "AAPL",
            {
                "data_source": "test",
                "profit_margin": 0.24,
                "return_on_equity": 0.35,
                "free_cash_flow": 100_000_000.0,
                "forward_pe": 22.0,
            },
        )
        # No revenue/OCF/cash-debt in snapshot -> depth metrics stay None, no crash.
        self.assertIsNone(context.cash_conversion)
        self.assertIsNone(context.net_debt)


if __name__ == "__main__":
    unittest.main()
