from __future__ import annotations

import unittest

from stock_selector.analyst import build_analyst_context


class AnalystContextTest(unittest.TestCase):
    def test_positive_analyst_expectations_score_high(self) -> None:
        context = build_analyst_context(
            "AAPL",
            {
                "data_source": "test",
                "analyst_upside": 0.18,
                "recommendation_mean": 1.9,
                "recommendation_key": "buy",
                "number_of_analysts": 30,
                "target_mean_price": 250.0,
            },
        )

        self.assertGreaterEqual(context.analyst_score, 70)
        self.assertEqual(context.analyst_label, "positive")
        self.assertEqual(context.analyst_risk_level, "low")
        self.assertFalse(context.analyst_block_new_entries)

    def test_negative_analyst_expectations_block_new_entries(self) -> None:
        context = build_analyst_context(
            "WEAK",
            {
                "data_source": "test",
                "analyst_upside": -0.18,
                "recommendation_mean": 3.8,
                "recommendation_key": "sell",
                "number_of_analysts": 18,
                "target_mean_price": 50.0,
            },
        )

        self.assertEqual(context.analyst_risk_level, "high")
        self.assertTrue(context.analyst_block_new_entries)
        self.assertLessEqual(context.analyst_score, 40)

    def test_missing_analyst_data_uses_unknown_neutral_context(self) -> None:
        context = build_analyst_context("XYZ", None)

        self.assertEqual(context.analyst_score, 50.0)
        self.assertEqual(context.analyst_label, "unknown")
        self.assertEqual(context.analyst_risk_level, "unknown")
        self.assertIn("snapshot unavailable", context.analyst_warning)


if __name__ == "__main__":
    unittest.main()
