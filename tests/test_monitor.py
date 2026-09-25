from __future__ import annotations

import unittest

import pandas as pd

from stock_selector.config import MonitorConfig
from stock_selector.monitor import (
    build_daily_monitor,
    feature_drift_report,
    feature_missing_report,
    selection_turnover,
)


def prices() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": pd.Timestamp("2024-01-31"), "ticker": "AAA", "adj_close": 100.0},
            {"date": pd.Timestamp("2024-01-31"), "ticker": "BBB", "adj_close": 50.0},
        ]
    )


def scored() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-15"),
                "ticker": "AAA",
                "momentum": 0.02,
                "quality": 0.2,
            },
            {
                "date": pd.Timestamp("2024-01-15"),
                "ticker": "BBB",
                "momentum": 0.03,
                "quality": 0.3,
            },
            {
                "date": pd.Timestamp("2024-01-31"),
                "ticker": "AAA",
                "momentum": 0.1,
                "quality": 0.2,
            },
            {
                "date": pd.Timestamp("2024-01-31"),
                "ticker": "BBB",
                "momentum": None,
                "quality": 0.3,
            },
        ]
    )


def selections() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2023-12-29"),
                "ticker": "AAA",
                "weight": 0.4,
                "score": 0.9,
                "rank": 1,
                "sector": "Technology",
            },
            {
                "date": pd.Timestamp("2024-01-31"),
                "ticker": "AAA",
                "weight": 0.5,
                "score": 1.1,
                "rank": 1,
                "sector": "Technology",
            },
            {
                "date": pd.Timestamp("2024-01-31"),
                "ticker": "BBB",
                "weight": 0.3,
                "score": 0.8,
                "rank": 2,
                "sector": "Healthcare",
            },
        ]
    )


class MonitorTest(unittest.TestCase):
    def test_feature_missing_report_uses_latest_date(self) -> None:
        report = feature_missing_report(scored(), ("momentum", "quality"))

        self.assertEqual(report.loc[0, "feature"], "momentum")
        self.assertEqual(report.loc[0, "missing_rate"], 0.5)

    def test_feature_drift_report_uses_history_window(self) -> None:
        report = feature_drift_report(scored(), ("momentum", "quality"), lookback_days=30)

        self.assertIn("drift_zscore", report.columns)
        self.assertEqual(set(report["feature"]), {"momentum", "quality"})

    def test_selection_turnover_compares_latest_two_rebalances(self) -> None:
        self.assertAlmostEqual(selection_turnover(selections()), 0.2)

    def test_build_daily_monitor_outputs_candidates_checks_and_markdown(self) -> None:
        risk_report = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2024-01-31"),
                    "gross_exposure": 0.8,
                    "cash_weight": 0.2,
                    "max_sector_weight": 0.5,
                    "largest_sector": "Technology",
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

        result = build_daily_monitor(
            prices=prices(),
            scored=scored(),
            selections=selections(),
            risk_report=risk_report,
            exposure_report=exposure_report,
            feature_columns=("momentum", "quality"),
            as_of_date="2024-02-02",
            monitor_config=MonitorConfig(max_single_feature_missing_rate=0.25),
        )

        self.assertEqual(len(result.candidates), 2)
        self.assertIn("data_age_days", result.checks["metric"].to_list())
        self.assertIn("candidate_overlap", result.checks["metric"].to_list())
        self.assertIn("max_feature_drift_zscore", result.checks["metric"].to_list())
        self.assertIn("Daily Stock Selection Monitor", result.report)

    def test_monitor_thresholds_can_trigger_warning(self) -> None:
        risk_report = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2024-01-31"),
                    "gross_exposure": 0.8,
                    "cash_weight": 0.2,
                    "max_sector_weight": 0.5,
                    "largest_sector": "Technology",
                }
            ]
        )
        result = build_daily_monitor(
            prices=prices(),
            scored=scored(),
            selections=selections(),
            risk_report=risk_report,
            exposure_report=pd.DataFrame(),
            feature_columns=("momentum", "quality"),
            as_of_date="2024-03-31",
            monitor_config=MonitorConfig(max_data_age_days=7),
        )

        data_age = result.checks[result.checks["metric"] == "data_age_days"].iloc[0]
        self.assertEqual(data_age["status"], "warning")


if __name__ == "__main__":
    unittest.main()
