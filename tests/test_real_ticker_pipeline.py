"""Offline end-to-end test of run_real_ticker_analysis with every network call stubbed."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from stock_selector.data import PriceDownloadResult
from stock_selector.events import build_event_risk_context
from stock_selector.real_data import run_real_ticker_analysis


def _synthetic_prices(tickers: list[str], days: int = 520) -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=days)
    rows = []
    for offset, ticker in enumerate(tickers):
        rng = np.random.default_rng(offset + 7)
        close = 100.0 * np.cumprod(1.0 + 0.0006 + rng.normal(0, 0.015, days))
        for date, price in zip(dates, close):
            rows.append({
                "date": date, "ticker": ticker, "open": price * 0.996, "high": price * 1.012,
                "low": price * 0.988, "close": price, "adj_close": price, "volume": 2_000_000,
            })
    return pd.DataFrame(rows)


def _fake_download(tickers, period, output_path=None, **_kwargs):
    return PriceDownloadResult(
        prices=_synthetic_prices(list(tickers)),
        provider="synthetic",
        attempts=("synthetic",),
        warnings=(),
    )


def _fake_event_risk(ticker, as_of_date):
    return build_event_risk_context(
        ticker=ticker, as_of_date=as_of_date, earnings_dates=[], source="test", warning="offline test",
    )


class RealTickerPipelineTest(unittest.TestCase):
    def test_offline_run_writes_every_output(self) -> None:
        with tempfile.TemporaryDirectory() as root, \
                patch("stock_selector.real_data.download_prices_for_period_multi_source", side_effect=_fake_download), \
                patch("stock_selector.real_data.fetch_yfinance_event_risk", side_effect=_fake_event_risk):
            result = run_real_ticker_analysis(
                "TEST",
                period="2y",
                output_root=Path(root) / "real_ticker",
                data_root=Path(root) / "prices",
                include_snapshot=False,
                include_peer_comparison=False,
                auto_period_upgrade=False,
            )
            out = Path(root) / "real_ticker" / "TEST"
            expected = [
                "ticker_analysis.md", "ticker_analysis.csv", "analysis_result.json", "cache_metadata.json",
                "data_sources.json", "data_readiness.json", "data_readiness.csv", "data_readiness.md",
                "event_risk.json", "fundamental_quality.json", "sentiment_risk.json",
                "analyst_expectations.json", "valuation_risk.json",
            ]
            missing = [name for name in expected if not (out / name).exists()]
            report = (out / "ticker_analysis.md").read_text(encoding="utf-8")
            payload = json.loads((out / "analysis_result.json").read_text(encoding="utf-8"))
            cache = json.loads((out / "cache_metadata.json").read_text(encoding="utf-8"))

        self.assertEqual(missing, [])
        self.assertEqual(result.ticker, "TEST")
        self.assertEqual(set(result.analysis["horizon"]), {"short", "medium", "long"})
        self.assertEqual(result.data_sources["TEST"], "synthetic")
        self.assertIn("# Ticker Analysis: TEST", report)
        self.assertIn("## Final Decision / 最终执行结论", report)
        self.assertIn("# Data Source Readiness / 数据源准备度", report)
        self.assertEqual(payload["ticker"], "TEST")
        self.assertIn(cache["signal_review"]["status"], {"ok", "failed"})
        self.assertFalse((out / "external_snapshot.json").exists())


if __name__ == "__main__":
    unittest.main()
