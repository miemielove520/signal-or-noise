from __future__ import annotations

from types import SimpleNamespace
import unittest

import pandas as pd

from stock_selector.data_sources import (
    build_data_readiness_report,
    render_data_readiness_report,
)


class DataSourcesTest(unittest.TestCase):
    def test_single_source_price_status_is_available(self) -> None:
        report = build_data_readiness_report(
            ticker="AAPL",
            prices=_price_frame("AAPL"),
            price_provider="yfinance",
            price_attempts=("yfinance",),
            price_warnings=(),
            price_missing_tickers=(),
            price_source_validation=None,
            benchmark_prices={
                "SPY": _price_frame("SPY"),
                "QQQ": _price_frame("QQQ"),
                "^VIX": _price_frame("^VIX"),
            },
            benchmark_sources={"SPY": "yfinance", "QQQ": "yfinance", "^VIX": "yfinance"},
            benchmark_warnings={},
            sector_etf="XLK",
            sector_prices=_price_frame("XLK"),
            sector_source="yfinance",
            sector_warnings=(),
            snapshot={
                "company_name": "Apple",
                "sector": "Technology",
                "industry": "Consumer Electronics",
                "forward_pe": 25.0,
                "peg_ratio": 2.0,
                "target_mean_price": 220.0,
                "news_titles": ["Apple product demand rises"],
            },
            contexts=_contexts(),
        )

        price_layer = next(layer for layer in report.layers if layer.layer == "price")
        self.assertEqual(price_layer.status, "single_source_available")
        self.assertTrue(price_layer.usable_for_scoring)
        self.assertEqual(report.source_validation["status"], "single_source_available")

    def test_price_conflict_blocks_scoring_and_report_mentions_readiness(self) -> None:
        report = build_data_readiness_report(
            ticker="AAPL",
            prices=_price_frame("AAPL"),
            price_provider="polygon",
            price_attempts=("polygon", "yfinance"),
            price_warnings=("price source validation conflict detected",),
            price_missing_tickers=(),
            price_source_validation={
                "status": "conflict_warning",
                "status_zh": "多源价格存在冲突",
                "comparison_provider_count": 1,
                "max_close_diff_pct": 0.08,
                "max_volume_diff_pct": 0.30,
                "max_missing_date_count": 0,
                "comparisons": [],
                "warnings": [],
            },
            benchmark_prices={
                "SPY": _price_frame("SPY"),
                "QQQ": _price_frame("QQQ"),
                "^VIX": _price_frame("^VIX"),
            },
            benchmark_sources={"SPY": "yfinance", "QQQ": "yfinance", "^VIX": "yfinance"},
            benchmark_warnings={},
            sector_etf="XLK",
            sector_prices=_price_frame("XLK"),
            sector_source="yfinance",
            sector_warnings=(),
            snapshot={"company_name": "Apple"},
            contexts=_contexts(),
        )

        price_layer = next(layer for layer in report.layers if layer.layer == "price")
        markdown = render_data_readiness_report(report)

        self.assertEqual(price_layer.status, "conflict_warning")
        self.assertFalse(price_layer.usable_for_scoring)
        self.assertEqual(report.overall_status, "conflict_warning")
        self.assertIn("Data Source Readiness / 数据源准备度", markdown)
        self.assertIn("conflict_warning", markdown)

    def test_missing_layers_are_not_usable_for_scoring(self) -> None:
        report = build_data_readiness_report(
            ticker="AAPL",
            prices=pd.DataFrame(),
            price_provider="unavailable",
            price_attempts=("yfinance",),
            price_warnings=("No rows returned",),
            price_missing_tickers=("AAPL",),
            price_source_validation=None,
            benchmark_prices={},
            benchmark_sources={},
            benchmark_warnings={},
            sector_etf="",
            sector_prices=pd.DataFrame(),
            sector_source="unavailable",
            sector_warnings=(),
            snapshot=None,
            contexts={},
        )

        self.assertEqual(report.overall_status, "insufficient")
        self.assertIn("price", report.primary_blockers)
        self.assertIn("价格", report.primary_blockers_zh)

    def test_fundamental_source_marks_sec_point_in_time(self) -> None:
        contexts = _contexts()
        contexts["fundamental"] = SimpleNamespace(source="sec_pit", data_coverage=1.0)

        report = build_data_readiness_report(
            ticker="AAPL",
            prices=_price_frame("AAPL"),
            price_provider="yfinance",
            price_attempts=("yfinance",),
            price_warnings=(),
            price_missing_tickers=(),
            price_source_validation=None,
            benchmark_prices={
                "SPY": _price_frame("SPY"),
                "QQQ": _price_frame("QQQ"),
                "^VIX": _price_frame("^VIX"),
            },
            benchmark_sources={"SPY": "yfinance", "QQQ": "yfinance", "^VIX": "yfinance"},
            benchmark_warnings={},
            sector_etf="XLK",
            sector_prices=_price_frame("XLK"),
            sector_source="yfinance",
            sector_warnings=(),
            snapshot={"company_name": "Apple"},
            contexts=contexts,
        )
        markdown = render_data_readiness_report(report)
        fundamental_layer = next(layer for layer in report.layers if layer.layer == "fundamental")

        self.assertEqual(report.fundamentals_source, "sec_pit")
        self.assertEqual(fundamental_layer.freshness, "point_in_time_filing_date")
        self.assertIn("Fundamentals source / 基本面来源", markdown)
        self.assertIn("sec_pit", markdown)


def _price_frame(ticker: str) -> pd.DataFrame:
    dates = pd.bdate_range("2026-06-01", periods=30)
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": [ticker] * len(dates),
            "open": [100.0] * len(dates),
            "high": [101.0] * len(dates),
            "low": [99.0] * len(dates),
            "close": [100.0] * len(dates),
            "adj_close": [100.0] * len(dates),
            "volume": [1_000_000] * len(dates),
        }
    )


def _contexts() -> dict[str, object]:
    context = SimpleNamespace(source="yfinance_snapshot", data_coverage=1.0)
    event = SimpleNamespace(
        source="yfinance_calendar",
        data_coverage=1.0,
        event_risk_level="low",
    )
    return {
        "fundamental": context,
        "valuation": context,
        "sentiment": context,
        "event": event,
        "analyst": context,
    }


if __name__ == "__main__":
    unittest.main()
