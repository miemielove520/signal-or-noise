from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from stock_selector.dashboard import (
    _build_page_banner,
    build_dashboard,
    collect_candidates,
    collect_ticker_summaries,
    recent_returns,
    render_dashboard_html,
)


class DashboardTests(unittest.TestCase):
    def _write_result(self, root: Path, ticker: str, decision: str) -> None:
        d = root / "real_ticker" / ticker
        d.mkdir(parents=True, exist_ok=True)
        (d / "analysis_result.json").write_text(
            json.dumps(
                {
                    "ticker": ticker,
                    "company_name": f"{ticker} Inc.",
                    "latest_price": 310.66,
                    "date": "2026-07-07",
                    "final_decision": decision,
                    "final_decision_zh": "观望",
                    "final_score": 70.0,
                    "final_watchlist_status_zh": "观察",
                    "final_focus_horizon_zh": "长期",
                    "fundamental_score": 75.4,
                    "fundamental_trend_direction_zh": "改善中",
                    "primary_blocker_zh": "等待突破",
                    "final_next_step_zh": "等待入场",
                    "generated_at_utc": "2026-07-08T00:00:00Z",
                }
            ),
            encoding="utf-8",
        )

    def test_collect_and_render(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_result(root, "AAPL", "watch_breakout_or_pullback")
            self._write_result(root, "NVDA", "avoid_or_wait_downtrend")
            summaries = collect_ticker_summaries(root / "real_ticker")
            self.assertEqual(len(summaries), 2)
            html = render_dashboard_html(summaries, generated_at="2026-07-08")
            self.assertIn("AAPL", html)
            self.assertIn("NVDA", html)
            self.assertIn("改善中", html)
            self.assertIn("<!doctype html>", html)

    def test_build_dashboard_writes_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_result(root, "AAPL", "buy")
            index_path = build_dashboard(outputs_dir=root, generated_at="2026-07-08 10:00")
            self.assertTrue(index_path.exists())
            self.assertIn("Stock Dashboard", index_path.read_text(encoding="utf-8"))

    def test_empty_outputs_is_graceful(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            index_path = build_dashboard(outputs_dir=Path(tmp), generated_at="now")
            self.assertTrue(index_path.exists())
            self.assertIn("还没有分析结果", index_path.read_text(encoding="utf-8"))

    def test_banner_surfaces_partial_scan_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scan_dir = root / "scans" / "latest"
            scan_dir.mkdir(parents=True)
            (scan_dir / "scan_result.json").write_text(
                json.dumps({"failures": [{"ticker": "CFLT", "error": "no data"}]}),
                encoding="utf-8",
            )

            banner = _build_page_banner([], [], date(2026, 7, 10), root)

            self.assertIn("CFLT", banner)
            self.assertIn("没有进入排序", banner)

    def test_watchlist_filter_excludes_leftover_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_result(root, "AAPL", "buy")
            self._write_result(root, "AAON", "avoid_or_wait_downtrend")  # leftover
            only_aapl = collect_ticker_summaries(root / "real_ticker", tickers=["AAPL"])
            self.assertEqual([s["ticker"] for s in only_aapl], ["AAPL"])
            # Order follows the watchlist, not the filesystem.
            both = collect_ticker_summaries(root / "real_ticker", tickers=["AAPL", "AAON"])
            self.assertEqual([s["ticker"] for s in both], ["AAPL", "AAON"])

    def test_company_name_and_date_render(self) -> None:
        summaries = [{"ticker": "AAPL", "company_name": "Apple Inc.", "date": "2026-07-07"}]
        html = render_dashboard_html(summaries)
        self.assertIn("Apple Inc.", html)
        self.assertIn("2026-07-07", html)

    def test_position_line_on_holding_card(self) -> None:
        summaries = [
            {"ticker": "AVGO", "final_decision": "hold", "pos_pl_pct": 0.0109,
             "pos_weight": 37.95, "pos_conc": "extreme", "pos_conc_zh": "极高"}
        ]
        html = render_dashboard_html(summaries)
        self.assertIn("持仓", html)
        self.assertIn("+1.09%", html)
        self.assertIn("极高", html)
        self.assertIn("pl-up", html)  # green for profit

    def test_no_position_line_when_not_held(self) -> None:
        html = render_dashboard_html([{"ticker": "XYZ", "final_decision": "watch"}])
        self.assertNotIn('class="position"', html)

    def test_news_block_shows_tags_and_english_headlines(self) -> None:
        summaries = [
            {
                "ticker": "AVGO",
                "final_decision": "hold",
                "positive_news_level_zh": "重大利好",
                "risk_news_level_zh": "无明显风险",
                "event_risk_level_zh": "高",
                "news_titles": ["Broadcom wins $30 billion Apple chip deal", "Analyst raises target"],
            }
        ]
        html = render_dashboard_html(summaries)
        self.assertIn("利好·重大利好", html)  # Chinese quick-read tag kept
        self.assertIn("事件风险高", html)
        self.assertIn("Broadcom wins $30 billion Apple chip deal", html)  # real headline
        self.assertNotIn("无明显风险", html)  # trivial risk level is not shown as a tag

    def test_no_news_block_when_empty(self) -> None:
        html = render_dashboard_html([{"ticker": "XYZ", "final_decision": "watch"}])
        self.assertNotIn('<div class="newsblock">', html)  # CSS defines .newsblock; no card uses it

    def test_paper_section_renders(self) -> None:
        paper = {"days_tracked": 25, "latest_equity": 105000.0, "total_return": 0.05,
                 "benchmark_return": 0.02, "excess_return": 0.03, "max_drawdown": -0.06,
                 "readiness_zh": "达标", "note_zh": "站得住"}
        html = render_dashboard_html([], paper=paper)
        self.assertIn("模拟盘", html)
        self.assertIn("$105,000.00", html)
        self.assertIn("达标", html)
        self.assertIn("+5.00%", html)

    def test_paper_section_empty(self) -> None:
        html = render_dashboard_html([], paper=None)
        self.assertIn("模拟盘还没启动", html)

    def test_decision_tone_colors(self) -> None:
        summaries = [
            {"ticker": "A", "final_decision": "avoid_or_wait_downtrend"},
            {"ticker": "B", "final_decision": "buy"},
        ]
        html = render_dashboard_html(summaries)
        self.assertIn("card bad", html)  # avoid -> red
        self.assertIn("card good", html)  # buy -> green


class MarketAndFreshnessTests(unittest.TestCase):
    def test_recent_returns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            closes = [100.0 + i for i in range(25)]  # steadily rising
            pd.DataFrame({"date": range(25), "adj_close": closes}).to_csv(
                root / "VOO_5y.csv", index=False
            )
            r = recent_returns("VOO", prices_dir=root)
            self.assertIsNotNone(r)
            self.assertAlmostEqual(r["r1"], 124 / 123 - 1, places=6)
            self.assertAlmostEqual(r["r5"], 124 / 119 - 1, places=6)
            self.assertEqual(r["latest_close"], 124.0)

    def test_recent_returns_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(recent_returns("NOPE", prices_dir=Path(tmp)))

    def test_stale_badge_uses_business_days(self) -> None:
        # 2026-06-01 是周一。周二看周一的收盘=落后1个交易日（正常，当天管线还没跑）；
        # 周三还停在周一=落后2个交易日，才算真的漏跑。
        summaries = [{"ticker": "AVGO", "date": "2026-06-01", "final_decision": "hold"}]
        fresh = render_dashboard_html(summaries, today=date(2026, 6, 2))
        stale = render_dashboard_html(summaries, today=date(2026, 6, 3))
        self.assertNotIn("数据滞后", fresh)
        self.assertIn("数据滞后", stale)

    def test_weekend_gap_is_not_stale(self) -> None:
        # 2026-06-05 是周五。周一看周五的收盘是正常状态，日历差3天不该误报。
        summaries = [{"ticker": "AVGO", "date": "2026-06-05", "final_decision": "hold"}]
        monday = render_dashboard_html(summaries, today=date(2026, 6, 8))
        self.assertNotIn("数据滞后", monday)

    def test_market_section_renders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame({"date": range(25), "adj_close": [100.0 + i for i in range(25)]}).to_csv(
                root / "VOO_5y.csv", index=False
            )
            html = render_dashboard_html(
                [],
                market=[{"ticker": "VOO", "company_name": "Vanguard S&P 500", "current_price": 500.0}],
                prices_dir=root,
            )
            self.assertIn("VOO", html)
            self.assertIn("大盘概况", html)
            self.assertIn("标普500", html)


class CandidatesTests(unittest.TestCase):
    def test_collect_candidates_sorts_and_excludes_holdings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv = Path(tmp) / "scan.csv"
            pd.DataFrame(
                {
                    "ticker": ["ABC", "AVGO", "XYZ"],
                    "high_probability_score": [40.0, 90.0, 55.0],
                    "watchlist_status_zh": ["接近", "就绪", "接近"],
                }
            ).to_csv(csv, index=False)
            got = collect_candidates(csv, exclude=["AVGO"], top_n=5)
            self.assertEqual([c["ticker"] for c in got], ["XYZ", "ABC"])  # AVGO excluded, sorted

    def test_collect_candidates_missing_file(self) -> None:
        self.assertEqual(collect_candidates("/nope/scan.csv"), [])

    def test_candidates_section_renders(self) -> None:
        html = render_dashboard_html(
            [],
            candidates=[{"ticker": "XYZ", "high_probability_score": 61.0, "watchlist_status_zh": "接近"}],
        )
        self.assertIn("潜力股候选", html)
        self.assertIn("XYZ", html)


if __name__ == "__main__":
    unittest.main()


class HolderNextStepTests(unittest.TestCase):
    def test_negatives_plus_concentration_leans_trim(self) -> None:
        from stock_selector.dashboard import _holder_next_step
        s = {"pos_pl_pct": -0.13, "pos_conc": "high", "pos_weight": 30.0,
             "fundamental_trend_direction_zh": "恶化中", "risk_news_level_zh": "增发稀释"}
        txt = _holder_next_step(s)
        self.assertIn("优先考虑减仓", txt)
        self.assertIn("基本面恶化", txt)

    def test_positive_but_concentrated_leans_lock_and_trim(self) -> None:
        from stock_selector.dashboard import _holder_next_step
        s = {"pos_pl_pct": 0.01, "pos_conc": "extreme", "pos_weight": 38.0,
             "fundamental_trend_direction_zh": "改善中", "positive_news_level_zh": "重大利好"}
        txt = _holder_next_step(s)
        self.assertIn("锁部分利润", txt)
        self.assertIn("集中度", txt)

    def test_not_a_holding_returns_none(self) -> None:
        from stock_selector.dashboard import _holder_next_step
        self.assertIsNone(_holder_next_step({"ticker": "XYZ"}))


class SectorAndFlowTests(unittest.TestCase):
    def _ohlcv(self, root, ticker, closes, vols):
        pd.DataFrame({
            "date": range(len(closes)),
            "high": [c + 1 for c in closes], "low": [c - 1 for c in closes],
            "close": closes, "adj_close": closes, "volume": vols,
        }).to_csv(Path(root) / f"{ticker}_5y.csv", index=False)

    def test_money_flow_index_range(self) -> None:
        from stock_selector.dashboard import money_flow_index
        with tempfile.TemporaryDirectory() as tmp:
            # steadily rising price -> high MFI (inflow)
            self._ohlcv(tmp, "XLK", [100 + i for i in range(30)], [1_000_000] * 30)
            mfi = money_flow_index("XLK", prices_dir=tmp)
            self.assertIsNotNone(mfi)
            self.assertGreater(mfi, 60)  # accumulation

    def test_collect_sector_returns_sorted(self) -> None:
        from stock_selector.dashboard import collect_sector_returns
        with tempfile.TemporaryDirectory() as tmp:
            self._ohlcv(tmp, "XLE", [100 + 2 * i for i in range(25)], [1e6] * 25)  # strong
            self._ohlcv(tmp, "XLU", [100 - i * 0.2 for i in range(25)], [1e6] * 25)  # weak
            rows = collect_sector_returns(["XLE", "XLU"], prices_dir=tmp)
            self.assertEqual([r["ticker"] for r in rows], ["XLE", "XLU"])  # strong first
            self.assertEqual(rows[0]["name"], "能源")

    def test_sector_section_renders_in_page(self) -> None:
        sectors = [{"ticker": "XLK", "name": "科技", "r1": 0.01, "r5": 0.03, "r20": 0.05, "mfi": 72.0}]
        html = render_dashboard_html([], sectors=sectors)
        self.assertIn("板块表现与资金流", html)
        self.assertIn("科技", html)
        self.assertIn("流入", html)  # MFI 72 -> 流入
        self.assertIn("非机构级真实流向", html)  # honest caveat present
