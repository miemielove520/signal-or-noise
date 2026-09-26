from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from stock_selector.universe import (
    BUILT_IN_UNIVERSES,
    load_historical_universe_membership,
    load_universe_tickers,
)
from stock_selector.validation_cli import _validation_stem


class UniverseTest(unittest.TestCase):
    def test_load_builtin_universe_and_manual_tickers(self) -> None:
        tickers = load_universe_tickers(["appl"], universe_name="software")

        self.assertIn("NOW", tickers)
        self.assertIn("AAPL", tickers)
        self.assertEqual(len(tickers), len(set(tickers)))

    def test_new_profile_universes_are_available(self) -> None:
        expected = {
            "ai-infrastructure": "ANET",
            "cybersecurity": "CRWD",
            "fintech-high-beta": "COIN",
            "defensive-quality": "COST",
            "industrial-quality": "AAON",
            "saas-software": "NOW",
            "healthcare-quality": "LLY",
            "financial-quality": "JPM",
            "consumer-discretionary": "HD",
            "consumer-staples": "PG",
            "energy-industrials": "XOM",
            "small-mid-growth": "ALAB",
            "sector-core": "JPM",
        }

        for universe_name, ticker in expected.items():
            self.assertIn(universe_name, BUILT_IN_UNIVERSES)
            self.assertIn(ticker, load_universe_tickers(universe_name=universe_name))

    def test_load_universe_from_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "universe.csv"
            pd.DataFrame({"symbol": ["aapl", "msft"]}).to_csv(path, index=False)

            tickers = load_universe_tickers(universe_file=path)

        self.assertEqual(tickers, ["AAPL", "MSFT"])

    def test_load_historical_universe_membership_filters_by_signal_date(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "historical_universe.csv"
            pd.DataFrame(
                [
                    {
                        "ticker": "aaa",
                        "start_date": "2020-01-01",
                        "end_date": "2020-12-31",
                    },
                    {
                        "ticker": "dead",
                        "start_date": "2020-01-01",
                        "delisted_date": "2020-06-30",
                        "delisting_return": -1.0,
                        "status": "delisted",
                    },
                ]
            ).to_csv(path, index=False)

            membership = load_historical_universe_membership(path)

        self.assertEqual(membership.tickers(), ["AAA", "DEAD"])
        self.assertIsNotNone(membership.active_record("AAA", "2020-06-01"))
        self.assertIsNone(membership.active_record("AAA", "2021-01-01"))
        self.assertTrue(membership.contains_delisted_tickers)
        report = membership.survivorship_report(["AAA", "DEAD"])
        self.assertTrue(report["survivorship_bias_handled"])
        self.assertTrue(report["contains_delisted_tickers"])

    def test_validation_stem_uses_universe_name_or_compact_tickers(self) -> None:
        self.assertEqual(_validation_stem(["AAPL", "MSFT"], "software"), "software")
        self.assertEqual(_validation_stem(["AAPL", "MSFT"], None), "AAPL_MSFT")
        self.assertEqual(
            _validation_stem(["AAPL", "MSFT", "NVDA", "NOW", "CRM", "ADBE", "ORCL"], None),
            "AAPL_MSFT_NVDA_NOW_CRM_ADBE_7tickers",
        )


if __name__ == "__main__":
    unittest.main()
