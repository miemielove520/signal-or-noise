from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from stock_selector.data import (
    download_prices_for_period_multi_source,
    download_yfinance_prices,
    load_fundamental_csv,
    load_metadata_csv,
    normalize_price_frame,
)


class DataDownloadTest(unittest.TestCase):
    def test_download_yfinance_prices_normalizes_output(self) -> None:
        index = pd.DatetimeIndex(["2024-01-02", "2024-01-03"], name="Date")
        raw = pd.DataFrame(
            {
                ("AAPL", "Open"): [100.0, 101.0],
                ("AAPL", "High"): [102.0, 103.0],
                ("AAPL", "Low"): [99.0, 100.0],
                ("AAPL", "Close"): [101.0, 102.0],
                ("AAPL", "Adj Close"): [101.0, 102.0],
                ("AAPL", "Volume"): [1_000_000, 1_100_000],
            },
            index=index,
        )

        download_kwargs: list[dict[str, object]] = []

        def fake_download(**kwargs: object) -> pd.DataFrame:
            download_kwargs.append(kwargs)
            return raw

        previous = sys.modules.get("yfinance")
        sys.modules["yfinance"] = types.SimpleNamespace(download=fake_download)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / "prices.csv"
                prices = download_yfinance_prices(["aapl"], "2024-01-01", output_path=output_path)

                self.assertTrue(output_path.exists())
                self.assertEqual(download_kwargs[0]["timeout"], 30)
                self.assertEqual(prices["ticker"].to_list(), ["AAPL", "AAPL"])
                self.assertEqual(
                    prices.columns.to_list(),
                    ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"],
                )
        finally:
            if previous is None:
                sys.modules.pop("yfinance", None)
            else:
                sys.modules["yfinance"] = previous

    def test_multi_source_falls_back_to_yfinance_when_professional_keys_are_missing(self) -> None:
        index = pd.DatetimeIndex(["2024-01-02", "2024-01-03"], name="Date")
        raw = pd.DataFrame(
            {
                ("AAPL", "Open"): [100.0, 101.0],
                ("AAPL", "High"): [102.0, 103.0],
                ("AAPL", "Low"): [99.0, 100.0],
                ("AAPL", "Close"): [101.0, 102.0],
                ("AAPL", "Adj Close"): [101.0, 102.0],
                ("AAPL", "Volume"): [1_000_000, 1_100_000],
            },
            index=index,
        )

        def fake_download(**_kwargs: object) -> pd.DataFrame:
            return raw

        previous = sys.modules.get("yfinance")
        sys.modules["yfinance"] = types.SimpleNamespace(download=fake_download)
        try:
            with patch.dict(
                "os.environ",
                {
                    "POLYGON_API_KEY": "",
                    "ALPACA_API_KEY_ID": "",
                    "ALPACA_API_SECRET_KEY": "",
                },
                clear=False,
            ):
                result = download_prices_for_period_multi_source(
                    ["aapl"],
                    period="1mo",
                    provider_order=("polygon", "alpaca", "yfinance"),
                )

            self.assertEqual(result.provider, "yfinance")
            self.assertEqual(result.prices["ticker"].to_list(), ["AAPL", "AAPL"])
            self.assertIn("polygon", result.attempts)
            self.assertIn("alpaca", result.attempts)
            self.assertIn("yfinance", result.attempts)
            self.assertTrue(any("POLYGON_API_KEY" in warning for warning in result.warnings))
            self.assertEqual(result.source_validation["status"], "single_source_available")
        finally:
            if previous is None:
                sys.modules.pop("yfinance", None)
            else:
                sys.modules["yfinance"] = previous

    def test_multi_source_cross_validation_passes_when_prices_match(self) -> None:
        primary = _price_frame("AAPL", close=100.0, volume=1_000_000)
        comparison = _price_frame("AAPL", close=100.2, volume=1_020_000)

        with patch("stock_selector.data.download_polygon_prices_for_period", return_value=primary):
            with patch("stock_selector.data.download_yfinance_prices_for_period", return_value=comparison):
                result = download_prices_for_period_multi_source(
                    ["AAPL"],
                    period="1mo",
                    provider_order=("polygon", "yfinance"),
                )

        self.assertEqual(result.provider, "polygon")
        self.assertEqual(result.source_validation["status"], "validated")
        self.assertLess(result.source_validation["max_close_diff_pct"], 0.005)

    def test_multi_source_cross_validation_warns_when_prices_conflict(self) -> None:
        primary = _price_frame("AAPL", close=100.0, volume=1_000_000)
        comparison = _price_frame("AAPL", close=110.0, volume=1_500_000)

        with patch("stock_selector.data.download_polygon_prices_for_period", return_value=primary):
            with patch("stock_selector.data.download_yfinance_prices_for_period", return_value=comparison):
                result = download_prices_for_period_multi_source(
                    ["AAPL"],
                    period="1mo",
                    provider_order=("polygon", "yfinance"),
                )

        self.assertEqual(result.provider, "polygon")
        self.assertEqual(result.source_validation["status"], "conflict_warning")
        self.assertTrue(any("price source validation conflict" in warning for warning in result.warnings))

    def test_multi_source_retries_provider_and_records_missing_tickers(self) -> None:
        index = pd.DatetimeIndex(["2024-01-02", "2024-01-03"], name="Date")
        raw = pd.DataFrame(
            {
                ("AAPL", "Open"): [100.0, 101.0],
                ("AAPL", "High"): [102.0, 103.0],
                ("AAPL", "Low"): [99.0, 100.0],
                ("AAPL", "Close"): [101.0, 102.0],
                ("AAPL", "Adj Close"): [101.0, 102.0],
                ("AAPL", "Volume"): [1_000_000, 1_100_000],
            },
            index=index,
        )
        calls = {"count": 0}
        download_kwargs: list[dict[str, object]] = []

        def fake_download(**kwargs: object) -> pd.DataFrame:
            calls["count"] += 1
            download_kwargs.append(kwargs)
            if calls["count"] == 1:
                raise TimeoutError("temporary timeout")
            return raw

        previous = sys.modules.get("yfinance")
        sys.modules["yfinance"] = types.SimpleNamespace(download=fake_download)
        try:
            result = download_prices_for_period_multi_source(
                ["aapl", "msft"],
                period="1mo",
                provider_order=("yfinance",),
                timeout_seconds=7,
                max_attempts=2,
            )

            self.assertEqual(calls["count"], 2)
            self.assertEqual(download_kwargs[0]["timeout"], 7)
            self.assertEqual(result.provider, "yfinance")
            self.assertEqual(result.missing_tickers, ("MSFT",))
            self.assertTrue(any("attempt 1/2" in warning for warning in result.warnings))
            self.assertTrue(any("missing price rows for MSFT" in warning for warning in result.warnings))
        finally:
            if previous is None:
                sys.modules.pop("yfinance", None)
            else:
                sys.modules["yfinance"] = previous

    def test_normalize_price_frame_accepts_close_without_adjusted_close(self) -> None:
        prices = normalize_price_frame(
            pd.DataFrame(
                {
                    "date": ["2024-01-02"],
                    "ticker": ["msft"],
                    "open": [100],
                    "high": [102],
                    "low": [99],
                    "close": [101],
                    "volume": [1_000_000],
                }
            )
        )

        self.assertEqual(prices["ticker"].iloc[0], "MSFT")
        self.assertEqual(prices["adj_close"].iloc[0], 101)

    def test_load_metadata_csv_normalizes_optional_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_path = Path(tmpdir) / "metadata.csv"
            pd.DataFrame(
                [
                    {"ticker": "aapl", "sector": "Technology"},
                    {"ticker": "msft", "sector": "Technology", "industry": "Software"},
                ]
            ).to_csv(metadata_path, index=False)

            metadata = load_metadata_csv(metadata_path)

            self.assertEqual(metadata["ticker"].to_list(), ["AAPL", "MSFT"])
            self.assertIn("industry", metadata.columns)
            self.assertIn("country", metadata.columns)
            self.assertEqual(metadata.loc[0, "industry"], "Unknown")

    def test_load_fundamental_csv_defaults_source_to_yfinance_restated(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fundamentals.csv"
            pd.DataFrame(
                [
                    {
                        "report_date": "2025-01-01",
                        "period_end": "2024-09-30",
                        "ticker": "aapl",
                        "revenue": 100,
                        "gross_profit": 50,
                        "operating_income": 30,
                        "net_income": 20,
                        "book_value": 80,
                        "total_assets": 200,
                        "total_liabilities": 120,
                        "operating_cash_flow": 25,
                        "capital_expenditure": 5,
                        "shares_outstanding": 10,
                    }
                ]
            ).to_csv(path, index=False)

            fundamentals = load_fundamental_csv(path)

        self.assertEqual(fundamentals["ticker"].iloc[0], "AAPL")
        self.assertEqual(fundamentals["fundamentals_source"].iloc[0], "yfinance_restated")


def _price_frame(ticker: str, close: float, volume: int) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=3)
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": [ticker] * len(dates),
            "open": [close] * len(dates),
            "high": [close + 1.0] * len(dates),
            "low": [close - 1.0] * len(dates),
            "close": [close] * len(dates),
            "adj_close": [close] * len(dates),
            "volume": [volume] * len(dates),
        }
    )


if __name__ == "__main__":
    unittest.main()
