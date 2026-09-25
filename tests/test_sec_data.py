from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from stock_selector.sec_data import (
    extract_sec_fundamental_history,
    fetch_sec_fundamental_snapshot,
)


class SecDataTest(unittest.TestCase):
    def test_fetch_sec_fundamental_snapshot_extracts_and_caches_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            with patch(
                "stock_selector.sec_data._load_or_fetch_json",
                side_effect=[_company_tickers_payload(), _companyfacts_payload()],
            ):
                result = fetch_sec_fundamental_snapshot(
                    "AAPL",
                    data_root=root,
                )

            fields = result.fields
            self.assertEqual(result.cik, "0000320193")
            self.assertEqual(fields["company_name"], "Apple Inc.")
            self.assertAlmostEqual(float(fields["revenue_growth"]), 0.10, places=4)
            self.assertAlmostEqual(float(fields["earnings_growth"]), 0.25, places=4)
            self.assertAlmostEqual(float(fields["profit_margin"]), 0.2273, places=4)
            self.assertAlmostEqual(float(fields["return_on_equity"]), 0.625, places=4)
            self.assertAlmostEqual(float(fields["free_cash_flow"]), 90_000_000_000.0, places=2)
            self.assertAlmostEqual(float(fields["debt_to_equity"]), 68.75, places=4)
            self.assertGreaterEqual(result.data_coverage, 0.8)
            self.assertTrue(Path(result.extracted_cache_path).exists())

    def test_extract_sec_fundamental_history_uses_filing_date_as_report_date(self) -> None:
        history = extract_sec_fundamental_history(
            ticker="AAPL",
            cik="0000320193",
            facts_payload=_companyfacts_payload(),
        )

        self.assertFalse(history.empty)
        self.assertEqual(history["fundamentals_source"].unique().tolist(), ["sec_pit"])
        latest = history.sort_values("period_end").iloc[-1]
        self.assertEqual(latest["period_end"].date().isoformat(), "2024-09-30")
        self.assertEqual(latest["report_date"].date().isoformat(), "2025-01-01")
        self.assertAlmostEqual(float(latest["revenue"]), 110_000_000_000.0)
        self.assertAlmostEqual(float(latest["book_value"]), 40_000_000_000.0)


def _company_tickers_payload() -> dict[str, dict[str, object]]:
    return {
        "0": {
            "cik_str": 320193,
            "ticker": "AAPL",
            "title": "Apple Inc.",
        }
    }


def _companyfacts_payload() -> dict[str, object]:
    return {
        "entityName": "Apple Inc.",
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {
                        "USD": [
                            _duration("2023-09-30", 100_000_000_000),
                            _duration("2024-09-30", 110_000_000_000),
                        ]
                    }
                },
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            _duration("2023-09-30", 20_000_000_000),
                            _duration("2024-09-30", 25_000_000_000),
                        ]
                    }
                },
                "NetCashProvidedByUsedInOperatingActivities": {
                    "units": {"USD": [_duration("2024-09-30", 95_000_000_000)]}
                },
                "PaymentsToAcquirePropertyPlantAndEquipment": {
                    "units": {"USD": [_duration("2024-09-30", 5_000_000_000)]}
                },
                "StockholdersEquity": {
                    "units": {"USD": [_instant("2024-09-30", 40_000_000_000)]}
                },
                "LongTermDebtCurrent": {
                    "units": {"USD": [_instant("2024-09-30", 2_500_000_000)]}
                },
                "LongTermDebtNoncurrent": {
                    "units": {"USD": [_instant("2024-09-30", 25_000_000_000)]}
                },
            }
        },
    }


def _duration(end: str, value: float) -> dict[str, object]:
    return {
        "end": end,
        "fy": int(end[:4]),
        "fp": "FY",
        "form": "10-K",
        "filed": f"{int(end[:4]) + 1}-01-01",
        "val": value,
    }


def _instant(end: str, value: float) -> dict[str, object]:
    return {
        "end": end,
        "fy": int(end[:4]),
        "fp": "FY",
        "form": "10-K",
        "filed": f"{int(end[:4]) + 1}-01-01",
        "val": value,
    }


if __name__ == "__main__":
    unittest.main()
