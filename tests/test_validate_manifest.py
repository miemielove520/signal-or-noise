from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import tempfile
import unittest

import pandas as pd

import validate


class ValidateManifestTest(unittest.TestCase):
    def test_validate_writes_run_manifest(self) -> None:
        prices = _price_frame(["AAPL", "SPY", "QQQ"])

        def fake_download(tickers, period, output_path):
            return SimpleNamespace(
                prices=prices[prices["ticker"].isin(tickers)].copy(),
                provider="fake_provider",
                attempts=("fake_provider",),
                warnings=("fake warning",),
                missing_tickers=("MSFT",),
            )

        with tempfile.TemporaryDirectory() as directory:
            with patch("stock_selector.validation_cli.download_prices_for_period_multi_source", side_effect=fake_download):
                exit_code = validate.main(
                    [
                        "AAPL",
                        "MSFT",
                        "--period",
                        "1y",
                        "--step-days",
                        "120",
                        "--min-history-days",
                        "80",
                        "--output-dir",
                        directory,
                    ]
                )

            manifest_path = Path(directory) / "run_manifest.json"
            payload = json.loads(manifest_path.read_text())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["period"], "1y")
        self.assertEqual(payload["step_days"], 120)
        self.assertEqual(payload["price_provider"], "fake_provider")
        self.assertEqual(payload["price_warnings"], ["fake warning"])
        self.assertEqual(payload["missing_price_tickers"], ["MSFT"])
        self.assertEqual(payload["requested_tickers"], ["AAPL", "MSFT"])
        self.assertIn("walk_forward_report", payload["output_files"])


def _price_frame(tickers: list[str]) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=180)
    rows = []
    for ticker_index, ticker in enumerate(tickers):
        for index, date in enumerate(dates):
            close = 100.0 + index * 0.2 + ticker_index
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": close - 0.2,
                    "high": close + 0.5,
                    "low": close - 0.5,
                    "close": close,
                    "adj_close": close,
                    "volume": 1_000_000,
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    unittest.main()
