from __future__ import annotations

import unittest

import pandas as pd

from stock_selector.report import render_research_report


class ReportTest(unittest.TestCase):
    def test_render_research_report_includes_latest_sections(self) -> None:
        selections = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2024-01-31"),
                    "ticker": "AAA",
                    "weight": 0.5,
                    "score": 1.2,
                    "rank": 1,
                    "sector": "Technology",
                    "industry": "Software",
                }
            ]
        )
        risk_report = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2024-01-31"),
                    "gross_exposure": 1.0,
                    "cash_weight": 0.0,
                    "max_position_weight": 0.5,
                    "max_sector_weight": 0.5,
                    "largest_sector": "Technology",
                    "estimated_annual_volatility": 0.12,
                }
            ]
        )
        exposure_report = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2024-01-31"),
                    "sector": "Technology",
                    "weight": 0.5,
                    "selected_count": 1,
                }
            ]
        )

        report = render_research_report(
            title="Test Report",
            selections=selections,
            metrics={"sharpe": 1.5},
            risk_report=risk_report,
            exposure_report=exposure_report,
        )

        self.assertIn("# Test Report", report)
        self.assertIn("Latest Selections", report)
        self.assertIn("Latest Sector Exposure", report)
        self.assertIn("Technology", report)


if __name__ == "__main__":
    unittest.main()
