from __future__ import annotations

import unittest

import pandas as pd

from stock_selector.fundamental_trends import build_fundamental_trend_context


def _history(margins, revenues=None, report_offsets_days=90):
    """Build a synthetic annual history where net_income = margin * revenue."""
    revenues = revenues or [1_000_000_000.0] * len(margins)
    rows = []
    for i, (margin, rev) in enumerate(zip(margins, revenues)):
        period_end = pd.Timestamp("2021-12-31") + pd.DateOffset(years=i)
        rows.append(
            {
                "period_end": period_end.date().isoformat(),
                "report_date": (period_end + pd.Timedelta(days=report_offsets_days)).date().isoformat(),
                "ticker": "XYZ",
                "revenue": rev,
                "gross_profit": rev * (margin + 0.30),
                "operating_income": rev * (margin + 0.05),
                "net_income": rev * margin,
                "operating_cash_flow": rev * (margin + 0.03),
                "capital_expenditure": rev * 0.02,
            }
        )
    return pd.DataFrame(rows)


class FundamentalTrendTests(unittest.TestCase):
    def test_expanding_margins_are_improving(self) -> None:
        ctx = build_fundamental_trend_context(_history([0.10, 0.14, 0.18, 0.22]), ticker="XYZ")
        self.assertEqual(ctx.status, "ok")
        self.assertEqual(ctx.net_margin_trend, "expanding")
        self.assertEqual(ctx.trend_direction, "improving")
        self.assertGreater(ctx.trend_score, 65)

    def test_contracting_margins_are_deteriorating(self) -> None:
        ctx = build_fundamental_trend_context(_history([0.22, 0.18, 0.13, 0.08]), ticker="XYZ")
        self.assertEqual(ctx.net_margin_trend, "contracting")
        self.assertEqual(ctx.trend_direction, "deteriorating")
        self.assertLess(ctx.trend_score, 42)

    def test_flat_margins_are_stable(self) -> None:
        ctx = build_fundamental_trend_context(_history([0.15, 0.151, 0.149, 0.15]), ticker="XYZ")
        self.assertEqual(ctx.net_margin_trend, "stable")
        self.assertEqual(ctx.trend_direction, "stable")

    def test_decelerating_revenue_growth(self) -> None:
        # Revenue grows but by a shrinking amount each year -> decelerating.
        ctx = build_fundamental_trend_context(
            _history([0.15] * 4, revenues=[1_000, 1_400, 1_650, 1_780])
        )
        self.assertEqual(ctx.revenue_growth_trend, "decelerating")

    def test_insufficient_history(self) -> None:
        ctx = build_fundamental_trend_context(_history([0.10, 0.20]), ticker="XYZ")
        self.assertEqual(ctx.status, "insufficient_history")
        self.assertEqual(ctx.trend_score, 50.0)

    def test_point_in_time_filter_excludes_future_filings(self) -> None:
        history = _history([0.10, 0.14, 0.18, 0.22])
        # As of just after the 2nd filing, only 2 periods are known -> insufficient.
        as_of = pd.Timestamp("2023-01-01")
        ctx = build_fundamental_trend_context(history, ticker="XYZ", as_of_date=as_of)
        self.assertEqual(ctx.status, "insufficient_history")

    def test_empty_history(self) -> None:
        ctx = build_fundamental_trend_context(pd.DataFrame(), ticker="XYZ")
        self.assertEqual(ctx.status, "insufficient_history")


if __name__ == "__main__":
    unittest.main()
