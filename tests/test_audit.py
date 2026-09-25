from __future__ import annotations

import unittest

import pandas as pd

from stock_selector.audit import audit_price_data


def valid_prices() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "ticker": "AAA",
                "open": 10.0,
                "high": 10.5,
                "low": 9.8,
                "close": 10.2,
                "adj_close": 10.2,
                "volume": 100_000,
            },
            {
                "date": "2024-01-03",
                "ticker": "AAA",
                "open": 10.2,
                "high": 10.8,
                "low": 10.1,
                "close": 10.7,
                "adj_close": 10.7,
                "volume": 120_000,
            },
            {
                "date": "2024-01-02",
                "ticker": "BBB",
                "open": 20.0,
                "high": 20.4,
                "low": 19.7,
                "close": 20.1,
                "adj_close": 20.1,
                "volume": 130_000,
            },
        ]
    )


class AuditTest(unittest.TestCase):
    def test_valid_prices_pass_audit(self) -> None:
        report = audit_price_data(valid_prices())

        self.assertTrue(report.passed)
        self.assertEqual(report.summary["tickers"], 2)
        self.assertEqual(report.issues, [])

    def test_audit_detects_duplicate_and_bad_price(self) -> None:
        prices = valid_prices()
        prices = pd.concat([prices, prices.iloc[[0]]], ignore_index=True)
        prices.loc[1, "adj_close"] = -1

        report = audit_price_data(prices)
        codes = {issue.code for issue in report.issues}

        self.assertFalse(report.passed)
        self.assertIn("duplicate_date_ticker", codes)
        self.assertIn("non_positive_price", codes)

    def test_audit_detects_missing_required_columns(self) -> None:
        prices = valid_prices().drop(columns=["volume"])

        report = audit_price_data(prices)

        self.assertFalse(report.passed)
        self.assertEqual(report.issues[0].code, "missing_columns")

    def test_audit_detects_blank_ticker(self) -> None:
        prices = valid_prices()
        prices.loc[0, "ticker"] = ""

        report = audit_price_data(prices)
        codes = {issue.code for issue in report.issues}

        self.assertFalse(report.passed)
        self.assertIn("missing_ticker", codes)


if __name__ == "__main__":
    unittest.main()
