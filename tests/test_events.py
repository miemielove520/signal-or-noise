from __future__ import annotations

import unittest

from stock_selector.events import build_event_risk_context


class EventRiskTest(unittest.TestCase):
    def test_high_event_risk_when_earnings_are_close(self) -> None:
        context = build_event_risk_context(
            ticker="AAPL",
            as_of_date="2026-06-11",
            earnings_dates=["2026-06-17"],
        )

        self.assertEqual(context.event_risk_level, "high")
        self.assertEqual(context.event_risk_level_zh, "高")
        self.assertEqual(context.days_until_earnings, 6)
        self.assertEqual(context.event_window, "earnings_week")
        self.assertTrue(context.event_block_new_entries)
        self.assertGreaterEqual(context.event_risk_score, 80)

    def test_imminent_event_window_blocks_new_entries(self) -> None:
        context = build_event_risk_context(
            ticker="AAPL",
            as_of_date="2026-06-11",
            earnings_dates=["2026-06-13"],
        )

        self.assertEqual(context.event_risk_level, "high")
        self.assertEqual(context.event_window, "earnings_imminent")
        self.assertEqual(context.event_window_zh, "财报即将发布")
        self.assertTrue(context.event_block_new_entries)
        self.assertFalse(context.event_cooldown_active)

    def test_unknown_event_risk_when_earnings_are_missing(self) -> None:
        context = build_event_risk_context(
            ticker="AAPL",
            as_of_date="2026-06-11",
            earnings_dates=[],
        )

        self.assertEqual(context.event_risk_level, "unknown")
        self.assertIsNone(context.next_earnings_date)
        self.assertEqual(context.event_window, "unknown")
        self.assertEqual(context.event_risk_score, 50.0)
        self.assertIn("unavailable", context.event_risk_warning)

    def test_low_event_risk_when_earnings_are_far(self) -> None:
        context = build_event_risk_context(
            ticker="AAPL",
            as_of_date="2026-06-11",
            earnings_dates=["2026-08-01"],
        )

        self.assertEqual(context.event_risk_level, "low")
        self.assertEqual(context.event_window, "normal")
        self.assertFalse(context.event_block_new_entries)
        self.assertLessEqual(context.event_risk_score, 30)

    def test_post_earnings_cooldown_blocks_new_entries(self) -> None:
        context = build_event_risk_context(
            ticker="AAPL",
            as_of_date="2026-06-11",
            earnings_dates=["2026-06-09", "2026-09-01"],
        )

        self.assertEqual(context.event_risk_level, "medium")
        self.assertEqual(context.event_window, "post_earnings_cooldown")
        self.assertEqual(context.event_window_zh, "财报后冷却期")
        self.assertTrue(context.event_block_new_entries)
        self.assertTrue(context.event_cooldown_active)
        self.assertEqual(context.last_earnings_date, "2026-06-09")
        self.assertEqual(context.days_since_earnings, 2)


if __name__ == "__main__":
    unittest.main()
