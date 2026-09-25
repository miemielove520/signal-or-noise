from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
from unittest.mock import patch

import pandas as pd

from stock_selector.real_data import (
    _apply_data_readiness_to_analysis,
    _auto_period_upgrade_decision,
    _period_is_shorter_than_5y,
    _repair_snapshot_from_cache,
    _repair_snapshot_financial_fields,
    _repair_snapshot_classification,
    _save_snapshot_cache,
    SNAPSHOT_CACHE_MAX_AGE_DAYS,
    build_single_ticker_scored_frame,
    normalize_ticker,
)
from stock_selector.data_sources import DataLayerReadiness, DataReadinessReport


class RealDataTest(unittest.TestCase):
    def test_build_single_ticker_scored_frame_uses_price_only_data(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=80)
        prices = pd.DataFrame(
            {
                "date": dates,
                "ticker": ["aapl"] * len(dates),
                "open": [100.0] * len(dates),
                "high": [101.0] * len(dates),
                "low": [99.0] * len(dates),
                "close": [100.0] * len(dates),
                "adj_close": [100.0] * len(dates),
                "volume": [1_000_000] * len(dates),
            }
        )

        scored = build_single_ticker_scored_frame(prices)

        self.assertEqual(scored["ticker"].unique().tolist(), ["AAPL"])
        self.assertIn("passes_universe", scored.columns)
        self.assertTrue(scored["passes_universe"].iloc[-1])
        self.assertTrue(scored["score"].isna().all())

    def test_normalize_ticker_corrects_common_apple_typo(self) -> None:
        self.assertEqual(normalize_ticker("appl"), "AAPL")
        self.assertEqual(normalize_ticker("nvda"), "NVDA")

    def test_period_is_shorter_than_5y_handles_yfinance_periods(self) -> None:
        self.assertTrue(_period_is_shorter_than_5y("6mo"))
        self.assertTrue(_period_is_shorter_than_5y("1y"))
        self.assertTrue(_period_is_shorter_than_5y("2y"))
        self.assertFalse(_period_is_shorter_than_5y("5y"))
        self.assertFalse(_period_is_shorter_than_5y("10y"))
        self.assertFalse(_period_is_shorter_than_5y("max"))

    def test_auto_period_upgrade_decision_uses_low_data_quality(self) -> None:
        analysis = pd.DataFrame(
            [
                {
                    "high_probability_score": 20.0,
                    "data_quality_score": 45.0,
                }
            ]
        )

        should_upgrade, reason, reason_zh = _auto_period_upgrade_decision(
            requested_period="1y",
            effective_period="1y",
            analysis=analysis,
        )

        self.assertTrue(should_upgrade)
        self.assertIn("5y", reason)
        self.assertIn("5年", reason_zh)

    def test_auto_period_upgrade_decision_keeps_5y(self) -> None:
        analysis = pd.DataFrame(
            [
                {
                    "high_probability_score": 20.0,
                    "data_quality_score": 45.0,
                }
            ]
        )

        should_upgrade, _, _ = _auto_period_upgrade_decision(
            requested_period="5y",
            effective_period="5y",
            analysis=analysis,
        )

        self.assertFalse(should_upgrade)

    def test_data_readiness_conflict_blocks_high_probability_gate(self) -> None:
        analysis = pd.DataFrame(
            [
                {
                    "quality_gate_passed": True,
                    "screening_action": "high_probability_watchlist_candidate",
                    "screening_action_zh": "高概率观察候选",
                    "calibrated_quality_gate_passed": True,
                    "calibrated_screening_action": "calibrated_high_probability_candidate",
                    "calibrated_screening_action_zh": "校准后高概率候选",
                    "quality_gate_fail_reasons": "all strict quality gates passed",
                    "quality_gate_fail_reasons_zh": "所有严格质量门槛通过",
                    "calibrated_quality_gate_fail_reasons": "all strict quality gates passed",
                    "calibrated_quality_gate_fail_reasons_zh": "所有严格质量门槛通过",
                }
            ]
        )
        readiness = DataReadinessReport(
            overall_status="conflict_warning",
            overall_status_zh="存在数据冲突",
            overall_score=55.0,
            repair_priority="high",
            repair_priority_zh="高",
            primary_blockers=("price",),
            primary_blockers_zh=("价格",),
            layers=(
                DataLayerReadiness(
                    layer="price",
                    layer_zh="价格",
                    source="polygon",
                    status="conflict_warning",
                    status_zh="价格源冲突",
                    freshness="fresh",
                    freshness_zh="新鲜",
                    coverage=1.0,
                    warning_count=1,
                    warnings=("price source validation conflict detected",),
                    usable_for_scoring=False,
                    note="conflict",
                    note_zh="冲突",
                ),
            ),
            source_validation={
                "status": "conflict_warning",
                "status_zh": "多源价格存在冲突",
            },
            sec_status="planned_not_enabled",
            sec_status_zh="已规划但尚未启用",
            fred_status="planned_not_enabled",
            fred_status_zh="已规划但尚未启用",
        )

        result = _apply_data_readiness_to_analysis(analysis, readiness)

        self.assertFalse(bool(result["quality_gate_passed"].iloc[0]))
        self.assertFalse(bool(result["calibrated_quality_gate_passed"].iloc[0]))
        self.assertEqual(result["data_readiness_level"].iloc[0], "conflict_warning")
        self.assertTrue(bool(result["data_needs_repair"].iloc[0]))
        self.assertIn("data source conflict warning", result["quality_gate_fail_reasons"].iloc[0])

    def test_repair_snapshot_classification_fills_common_ticker_sector(self) -> None:
        repaired, actions, actions_zh = _repair_snapshot_classification("AAPL", None)

        self.assertIsNotNone(repaired)
        assert repaired is not None
        self.assertEqual(repaired["sector"], "Technology")
        self.assertEqual(repaired["industry"], "Consumer Electronics")
        self.assertIn("filled missing sector", "; ".join(actions))
        self.assertIn("补齐缺失的板块", "；".join(actions_zh))

    def test_repair_snapshot_classification_preserves_existing_values(self) -> None:
        repaired, actions, _ = _repair_snapshot_classification(
            "AAPL",
            {"sector": "Custom Sector", "industry": "Custom Industry"},
        )

        self.assertEqual(repaired["sector"], "Custom Sector")
        self.assertEqual(repaired["industry"], "Custom Industry")
        self.assertEqual(actions, ())

    def test_repair_snapshot_financial_fields_retries_and_fills_missing_values(self) -> None:
        initial_snapshot = {
            "ticker": "AAPL",
            "sector": "Technology",
            "industry": "Consumer Electronics",
        }
        retry_snapshot = {
            "ticker": "AAPL",
            "revenue_growth": 0.08,
            "profit_margin": 0.22,
            "forward_pe": 28.0,
            "peg_ratio": 2.1,
            "target_mean_price": 210.0,
        }

        with patch("stock_selector.real_data.fetch_yfinance_snapshot", return_value=retry_snapshot):
            repaired, actions, actions_zh = _repair_snapshot_financial_fields(
                ticker="AAPL",
                snapshot=initial_snapshot,
                allow_fetch=True,
            )

        self.assertEqual(repaired["forward_pe"], 28.0)
        self.assertEqual(repaired["peg_ratio"], 2.1)
        self.assertEqual(repaired["target_mean_price"], 210.0)
        self.assertEqual(repaired["financial_repair_source"], "yfinance_snapshot_retry")
        self.assertIn("filled missing financial fields", actions[0])
        self.assertIn("通过yfinance二次拉取补齐财务字段", actions_zh[0])

    def test_repair_snapshot_financial_fields_respects_no_snapshot_mode(self) -> None:
        with patch("stock_selector.real_data.fetch_yfinance_snapshot") as fetch_mock:
            repaired, actions, actions_zh = _repair_snapshot_financial_fields(
                ticker="AAPL",
                snapshot=None,
                allow_fetch=False,
            )

        self.assertIsNone(repaired)
        self.assertEqual(actions, ())
        self.assertEqual(actions_zh, ())
        fetch_mock.assert_not_called()

    def test_snapshot_cache_repairs_missing_company_fields(self) -> None:
        cached_snapshot = {
            "ticker": "AAON",
            "company_name": "AAON Inc.",
            "sector": "Industrials",
            "industry": "Building Products",
            "forward_pe": 30.5,
            "peg_ratio": 2.1,
            "revenue_growth": 0.08,
            "profit_margin": 0.18,
            "target_mean_price": 120.0,
            "news_titles": ["AAON raises outlook"],
        }

        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            _save_snapshot_cache("AAON", cached_snapshot, data_root)

            repaired, actions, actions_zh = _repair_snapshot_from_cache(
                ticker="AAON",
                snapshot={"ticker": "AAON", "sector": "Industrials"},
                data_root=data_root,
            )

        assert repaired is not None
        self.assertEqual(repaired["company_name"], "AAON Inc.")
        self.assertEqual(repaired["forward_pe"], 30.5)
        self.assertEqual(repaired["news_titles"], ["AAON raises outlook"])
        self.assertIn("local cache", actions[0])
        self.assertIn("本地快照缓存", actions_zh[0])

    def test_snapshot_cache_ignores_mismatched_ticker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            _save_snapshot_cache(
                "MSFT",
                {"ticker": "MSFT", "company_name": "Microsoft", "forward_pe": 31.0},
                data_root,
            )
            cache_path = data_root / "real_snapshots" / "AAPL_snapshot.json"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                '{"ticker": "MSFT", "company_name": "Wrong cache"}',
                encoding="utf-8",
            )

            repaired, actions, _ = _repair_snapshot_from_cache(
                ticker="AAPL",
                snapshot={"ticker": "AAPL"},
                data_root=data_root,
            )

        self.assertEqual(repaired, {"ticker": "AAPL"})
        self.assertEqual(actions, ())

    def test_snapshot_cache_ignores_stale_snapshot(self) -> None:
        stale_time = (
            datetime.now(timezone.utc)
            - timedelta(days=SNAPSHOT_CACHE_MAX_AGE_DAYS + 1)
        ).isoformat()

        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            cache_path = data_root / "real_snapshots" / "AAON_snapshot.json"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                (
                    '{"ticker": "AAON", "snapshot_cache_saved_at_utc": "'
                    + stale_time
                    + '", "forward_pe": 30.5}'
                ),
                encoding="utf-8",
            )

            repaired, actions, _ = _repair_snapshot_from_cache(
                ticker="AAON",
                snapshot={"ticker": "AAON"},
                data_root=data_root,
            )

        self.assertEqual(repaired, {"ticker": "AAON"})
        self.assertEqual(actions, ())


if __name__ == "__main__":
    unittest.main()
