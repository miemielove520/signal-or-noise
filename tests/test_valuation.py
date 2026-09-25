from __future__ import annotations

import unittest

from stock_selector.valuation import build_valuation_context


class ValuationContextTest(unittest.TestCase):
    def test_reasonable_growth_valuation_scores_well(self) -> None:
        context = build_valuation_context(
            "GOOD",
            {
                "data_source": "test",
                "forward_pe": 24.0,
                "peg_ratio": 1.4,
                "revenue_growth": 0.16,
                "earnings_growth": 0.20,
                "profit_margin": 0.22,
            },
        )

        self.assertGreaterEqual(context.valuation_score, 65)
        self.assertIn(context.valuation_label, {"reasonable", "attractive"})
        self.assertEqual(context.valuation_risk_level, "low")
        self.assertFalse(context.valuation_block_new_entries)

    def test_expensive_low_growth_valuation_blocks_new_entries(self) -> None:
        context = build_valuation_context(
            "RICH",
            {
                "data_source": "test",
                "forward_pe": 75.0,
                "peg_ratio": 4.5,
                "revenue_growth": 0.04,
                "earnings_growth": 0.03,
                "profit_margin": 0.08,
            },
        )

        self.assertLessEqual(context.valuation_score, 35)
        self.assertEqual(context.valuation_risk_level, "high")
        self.assertTrue(context.valuation_block_new_entries)

    def test_missing_valuation_uses_unknown_neutral_context(self) -> None:
        context = build_valuation_context("XYZ", None)

        self.assertEqual(context.valuation_score, 50.0)
        self.assertEqual(context.valuation_label, "unknown")
        self.assertEqual(context.valuation_risk_level, "unknown")
        self.assertIn("snapshot unavailable", context.valuation_warning)


if __name__ == "__main__":
    unittest.main()
