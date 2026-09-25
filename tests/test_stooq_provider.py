from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from stock_selector import data


STOOQ_CSV = """Date,Open,High,Low,Close,Volume
2024-01-02,100.0,102.0,99.0,101.0,1000000
2024-01-03,101.0,103.0,100.0,102.0,1100000
"""


class StooqSymbolTests(unittest.TestCase):
    def test_us_equity_gets_us_suffix(self) -> None:
        self.assertEqual(data._stooq_symbol("AAPL"), "aapl.us")

    def test_index_symbol_keeps_caret(self) -> None:
        self.assertEqual(data._stooq_symbol("^VIX"), "^vix")


class StooqParseTests(unittest.TestCase):
    def test_parses_valid_csv(self) -> None:
        frame = data._parse_stooq_csv(STOOQ_CSV, "AAPL")
        self.assertEqual(len(frame), 2)
        self.assertListEqual(list(frame["ticker"].unique()), ["AAPL"])
        self.assertIn("adj_close", frame.columns)
        self.assertEqual(frame["adj_close"].tolist(), frame["close"].tolist())

    def test_no_data_returns_empty(self) -> None:
        self.assertTrue(data._parse_stooq_csv("No data", "AAPL").empty)
        self.assertTrue(data._parse_stooq_csv("", "AAPL").empty)

    def test_malformed_returns_empty(self) -> None:
        self.assertTrue(data._parse_stooq_csv("<html>error</html>", "AAPL").empty)


class StooqDownloadTests(unittest.TestCase):
    def test_download_uses_get_text_and_normalizes(self) -> None:
        with patch("stock_selector.data._get_text", return_value=STOOQ_CSV):
            frame = data.download_stooq_prices_for_period(["AAPL"], period="1mo")
        self.assertEqual(list(frame.columns[:2]), ["date", "ticker"])
        self.assertEqual(len(frame), 2)

    def test_all_empty_raises(self) -> None:
        with patch("stock_selector.data._get_text", return_value="No data"):
            with self.assertRaises(ValueError):
                data.download_stooq_prices_for_period(["ZZZZ"], period="1mo")


TIINGO_JSON = """[
  {"date":"2024-01-02T00:00:00.000Z","close":101.0,"high":102.0,"low":99.0,"open":100.0,"volume":1000000,"adjClose":101.0,"adjHigh":102.0,"adjLow":99.0,"adjOpen":100.0,"adjVolume":1000000},
  {"date":"2024-01-03T00:00:00.000Z","close":102.0,"high":103.0,"low":100.0,"open":101.0,"volume":1100000,"adjClose":102.0,"adjHigh":103.0,"adjLow":100.0,"adjOpen":101.0,"adjVolume":1100000}
]"""


class TiingoTests(unittest.TestCase):
    def test_parse_json(self) -> None:
        frame = data._parse_tiingo_json(TIINGO_JSON, "AAPL")
        self.assertEqual(len(frame), 2)
        self.assertEqual(frame["date"].tolist(), ["2024-01-02", "2024-01-03"])
        self.assertListEqual(list(frame["ticker"].unique()), ["AAPL"])

    def test_parse_non_list_returns_empty(self) -> None:
        self.assertTrue(data._parse_tiingo_json('{"detail":"Not authorized"}', "AAPL").empty)
        self.assertTrue(data._parse_tiingo_json("boom", "AAPL").empty)

    def test_download_requires_token(self) -> None:
        with patch.dict("os.environ", {"TIINGO_API_TOKEN": "", "TIINGO_API_KEY": ""}, clear=False):
            with self.assertRaises(RuntimeError):
                data.download_tiingo_prices_for_period(["AAPL"], period="1mo")

    def test_download_with_token_parses_and_normalizes(self) -> None:
        with patch.dict("os.environ", {"TIINGO_API_TOKEN": "dummy"}, clear=False):
            with patch("stock_selector.data._get_text", return_value=TIINGO_JSON):
                frame = data.download_tiingo_prices_for_period(["AAPL"], period="1mo")
        self.assertEqual(len(frame), 2)
        self.assertEqual(list(frame.columns[:2]), ["date", "ticker"])

    def test_rejects_benchmark_only(self) -> None:
        with patch.dict("os.environ", {"TIINGO_API_TOKEN": "dummy"}, clear=False):
            with self.assertRaises(ValueError):
                data.download_tiingo_prices_for_period(["^VIX"], period="1mo")

    def test_becomes_primary_when_token_set(self) -> None:
        with patch.dict(
            "os.environ",
            {"TIINGO_API_TOKEN": "dummy", "POLYGON_API_KEY": "", "ALPACA_API_KEY_ID": ""},
            clear=False,
        ):
            order = data._available_default_providers()
        self.assertEqual(order[0], "tiingo")
        self.assertIn("yfinance", order)


class ProviderRegistryTests(unittest.TestCase):
    def test_registry_contains_expected_providers(self) -> None:
        self.assertEqual(
            set(data.PRICE_PROVIDERS), {"polygon", "alpaca", "tiingo", "yfinance", "stooq"}
        )

    def test_is_known_provider(self) -> None:
        self.assertTrue(data._is_known_provider("stooq"))
        self.assertFalse(data._is_known_provider("nasdaq_data_link"))

    def test_defaults_include_free_second_source(self) -> None:
        with patch.dict("os.environ", {"POLYGON_API_KEY": "", "ALPACA_API_KEY_ID": "", "STOCK_SELECTOR_DISABLE_STOOQ": ""}, clear=False):
            self.assertEqual(data._available_default_providers(), ("yfinance", "stooq"))

    def test_stooq_can_be_disabled(self) -> None:
        with patch.dict("os.environ", {"STOCK_SELECTOR_DISABLE_STOOQ": "1"}, clear=False):
            self.assertNotIn("stooq", data._available_default_providers())

    def test_fetch_late_binds_for_patching(self) -> None:
        provider = data.PRICE_PROVIDERS["stooq"]
        with patch("stock_selector.data.download_stooq_prices_for_period", return_value=pd.DataFrame({"x": [1]})) as mock:
            provider.fetch(["AAPL"], period="1mo", timeout_seconds=10)
            mock.assert_called_once()


class CrossValidationActivationTests(unittest.TestCase):
    def _frame(self, close: float) -> pd.DataFrame:
        return data.normalize_price_frame(
            pd.DataFrame(
                {
                    "date": ["2024-01-02", "2024-01-03"],
                    "ticker": ["AAPL", "AAPL"],
                    "open": [close, close],
                    "high": [close, close],
                    "low": [close, close],
                    "close": [close, close],
                    "adj_close": [close, close],
                    "volume": [1_000_000, 1_000_000],
                }
            )
        )

    def test_yfinance_plus_stooq_activates_validation(self) -> None:
        # No paid keys: default order is (yfinance, stooq). Both succeed and agree,
        # so cross-source validation runs instead of "single source".
        with patch("stock_selector.data.download_yfinance_prices_for_period", return_value=self._frame(100.0)):
            with patch("stock_selector.data.download_stooq_prices_for_period", return_value=self._frame(100.1)):
                with patch.dict("os.environ", {"POLYGON_API_KEY": "", "ALPACA_API_KEY_ID": ""}, clear=False):
                    result = data.download_prices_for_period_multi_source(["AAPL"], period="1mo")
        self.assertEqual(result.provider, "yfinance")
        self.assertEqual(result.source_validation["status"], "validated")


if __name__ == "__main__":
    unittest.main()
