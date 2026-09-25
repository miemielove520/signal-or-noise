from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from stock_selector.config import PaperTradingConfig
from stock_selector.paper import (
    PaperTargetConfig,
    build_paper_targets_from_analysis,
    load_portfolio_state,
    run_paper_rebalance,
)


def prices() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-31"),
                "ticker": "AAA",
                "adj_close": 100.0,
            },
            {
                "date": pd.Timestamp("2024-01-31"),
                "ticker": "BBB",
                "adj_close": 50.0,
            },
        ]
    )


def selections() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-31"),
                "ticker": "AAA",
                "weight": 0.60,
                "score": 1.0,
                "rank": 1,
            },
            {
                "date": pd.Timestamp("2024-01-31"),
                "ticker": "BBB",
                "weight": 0.30,
                "score": 0.5,
                "rank": 2,
            },
        ]
    )


class PaperTradingTest(unittest.TestCase):
    def test_missing_state_initializes_cash(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = load_portfolio_state(Path(tmpdir) / "missing.csv", initial_cash=12_345)

            self.assertEqual(state.loc[0, "ticker"], "CASH")
            self.assertEqual(state.loc[0, "quantity"], 12_345)

    def test_paper_rebalance_generates_orders_and_post_trade_state(self) -> None:
        state = pd.DataFrame([{"ticker": "CASH", "quantity": 10_000.0}])
        result = run_paper_rebalance(
            selections=selections(),
            prices=prices(),
            state=state,
            config=PaperTradingConfig(
                initial_cash=10_000,
                min_trade_value=1,
                trade_buffer_pct=0.0,
                slippage_bps=0.0,
            ),
        )

        self.assertEqual(len(result.orders), 2)
        self.assertIn("AAA", result.post_trade_state["ticker"].to_list())
        self.assertGreater(result.summary["post_trade_cash"], 0)
        self.assertIn("Paper Trade Rebalance Report", result.report)

    def test_analysis_rows_build_capped_paper_targets(self) -> None:
        analysis = pd.DataFrame(
            [
                {
                    "date": "2024-01-31",
                    "ticker": "AAA",
                    "calibrated_quality_gate_passed": True,
                    "calibrated_high_probability_score": 82.0,
                    "calibrated_win_probability": 0.66,
                    "backtest_trust_score": 75.0,
                    "signal_score": 78.0,
                    "confidence_score": 80.0,
                    "data_quality_score": 85.0,
                    "overall_risk_level": "medium",
                    "latest_price": 100.0,
                    "final_focus_horizon": "medium",
                    "screening_profile": "software",
                },
                {
                    "date": "2024-01-31",
                    "ticker": "BBB",
                    "calibrated_quality_gate_passed": True,
                    "calibrated_high_probability_score": 76.0,
                    "calibrated_win_probability": 0.61,
                    "backtest_trust_score": 70.0,
                    "signal_score": 72.0,
                    "confidence_score": 74.0,
                    "data_quality_score": 78.0,
                    "overall_risk_level": "medium",
                    "latest_price": 50.0,
                    "final_focus_horizon": "short",
                    "screening_profile": "semiconductor",
                },
            ]
        )

        targets = build_paper_targets_from_analysis(
            analysis,
            PaperTargetConfig(max_positions=2, max_position_weight=0.30, cash_reserve_weight=0.20),
        )

        self.assertEqual(targets["ticker"].to_list(), ["AAA", "BBB"])
        self.assertTrue((targets["paper_target_weight"] <= 0.30).all())
        self.assertLessEqual(targets["paper_target_weight"].sum(), 0.60)
        self.assertIn("通过严格高概率筛选", targets.loc[0, "paper_reason_zh"])

    def test_analysis_rows_filter_weak_or_high_risk_names(self) -> None:
        analysis = pd.DataFrame(
            [
                {
                    "date": "2024-01-31",
                    "ticker": "AAA",
                    "quality_gate_passed": True,
                    "high_probability_score": 80.0,
                    "confidence_score": 80.0,
                    "data_quality_score": 80.0,
                    "overall_risk_level": "high",
                    "latest_price": 100.0,
                },
                {
                    "date": "2024-01-31",
                    "ticker": "BBB",
                    "quality_gate_passed": True,
                    "high_probability_score": 80.0,
                    "confidence_score": 60.0,
                    "data_quality_score": 80.0,
                    "overall_risk_level": "medium",
                    "latest_price": 50.0,
                },
            ]
        )

        targets = build_paper_targets_from_analysis(analysis)

        self.assertTrue(targets.empty)

    def test_market_regime_policy_blocks_weak_market_targets(self) -> None:
        analysis = pd.DataFrame(
            [
                {
                    "date": "2024-01-31",
                    "ticker": "AAA",
                    "quality_gate_passed": True,
                    "high_probability_score": 82.0,
                    "signal_score": 78.0,
                    "confidence_score": 80.0,
                    "data_quality_score": 85.0,
                    "overall_risk_level": "medium",
                    "market_regime": "weak",
                    "latest_price": 100.0,
                }
            ]
        )
        policy = pd.DataFrame(
            [
                {
                    "validation_market_regime": "bear_downtrend",
                    "protection_action": "block_new_entries",
                    "protection_action_zh": "暂停新开仓",
                    "allow_new_entries": False,
                    "allowed_signal_bucket": "none",
                    "signal_score_delta": 8.0,
                    "confidence_score_delta": 8.0,
                    "high_probability_score_delta": 10.0,
                    "position_scale": 0.0,
                    "policy_note_zh": "熊市环境暂停新开仓。",
                }
            ]
        )

        targets = build_paper_targets_from_analysis(analysis, market_regime_policy=policy)

        self.assertTrue(targets.empty)

    def test_market_regime_policy_scales_high_volatility_targets(self) -> None:
        analysis = pd.DataFrame(
            [
                {
                    "date": "2024-01-31",
                    "ticker": "AAA",
                    "quality_gate_passed": True,
                    "high_probability_score": 82.0,
                    "signal_score": 78.0,
                    "confidence_score": 80.0,
                    "data_quality_score": 85.0,
                    "overall_risk_level": "medium",
                    "market_regime": "high_volatility",
                    "latest_price": 100.0,
                }
            ]
        )
        policy = pd.DataFrame(
            [
                {
                    "validation_market_regime": "high_volatility",
                    "protection_action": "strict_only",
                    "protection_action_zh": "只允许严格高概率",
                    "allow_new_entries": True,
                    "allowed_signal_bucket": "strict_high_probability_only",
                    "signal_score_delta": 5.0,
                    "confidence_score_delta": 5.0,
                    "high_probability_score_delta": 7.0,
                    "position_scale": 0.5,
                    "policy_note_zh": "高波动环境半仓模拟。",
                }
            ]
        )

        targets = build_paper_targets_from_analysis(
            analysis,
            config=PaperTargetConfig(
                max_positions=1,
                max_position_weight=0.40,
                cash_reserve_weight=0.0,
            ),
            market_regime_policy=policy,
        )

        self.assertEqual(targets.loc[0, "ticker"], "AAA")
        self.assertAlmostEqual(float(targets.loc[0, "paper_target_weight"]), 0.20)
        self.assertEqual(targets.loc[0, "market_regime_policy_action"], "strict_only")


if __name__ == "__main__":
    unittest.main()


class UnpricedHoldingTests(unittest.TestCase):
    """持仓掉出候选后必须仍可估值/卖出；无法定价时必须显式亮出，不能静默按 0 算。"""

    def test_engine_flags_held_ticker_without_price(self) -> None:
        state = pd.DataFrame(
            [
                {"ticker": "CASH", "quantity": 599.8},
                {"ticker": "ANET", "quantity": 1.0818},  # 持仓，但今天没有报价
            ]
        )
        targets = pd.DataFrame(
            [{"date": pd.Timestamp("2026-07-10"), "ticker": "NEWT", "weight": 0.2}]
        )
        only_target_prices = pd.DataFrame(
            [{"date": pd.Timestamp("2026-07-10"), "ticker": "NEWT", "adj_close": 50.0}]
        )

        result = run_paper_rebalance(
            selections=targets,
            prices=only_target_prices,
            state=state,
            config=PaperTradingConfig(min_trade_value=100, trade_buffer_pct=0.001),
        )

        self.assertEqual(result.summary.get("unpriced_positions"), "ANET")
        self.assertIn("unpriced_positions", result.report)
        # 仍然留在仓位里（没有被偷偷丢掉），但也没有生成它的订单。
        self.assertIn("ANET", result.post_trade_state["ticker"].to_list())
        if not result.orders.empty:
            self.assertNotIn("ANET", result.orders["ticker"].to_list())

    def test_engine_no_flag_when_all_priced(self) -> None:
        state = pd.DataFrame([{"ticker": "CASH", "quantity": 10_000.0}])
        result = run_paper_rebalance(
            selections=selections(),
            prices=prices(),
            state=state,
            config=PaperTradingConfig(min_trade_value=1, trade_buffer_pct=0.0),
        )
        self.assertNotIn("unpriced_positions", result.summary)


class SellClampTests(unittest.TestCase):
    def test_full_liquidation_never_oversells_into_negative_position(self) -> None:
        """卖出成交价低于收盘（半价差）→ 按差额算出的股数会略超持仓；必须钳制，
        否则全清仓后留下负持仓（意外做空）。"""
        state = pd.DataFrame(
            [
                {"ticker": "CASH", "quantity": 599.8},
                {"ticker": "AAA", "quantity": 1.0818138348532826},
            ]
        )
        no_targets = pd.DataFrame(columns=["date", "ticker", "weight"])
        price_table = pd.DataFrame(
            [{"date": pd.Timestamp("2026-07-10"), "ticker": "AAA", "adj_close": 184.69}]
        )
        result = run_paper_rebalance(
            selections=no_targets,
            prices=price_table,
            state=state,
            config=PaperTradingConfig(
                min_trade_value=100, trade_buffer_pct=0.001, spread_bps=10.0, slippage_bps=5.0
            ),
        )

        sell = result.orders.iloc[0]
        self.assertEqual(sell["action"], "SELL")
        self.assertLessEqual(float(sell["quantity"]), 1.0818138348532826 + 1e-12)
        post = result.post_trade_state
        self.assertNotIn("AAA", post["ticker"].to_list())  # 清干净，无负残渣
        self.assertTrue((post["quantity"] >= 0).all())


class CliPriceTableTests(unittest.TestCase):
    """CLI 价格表：analysis ∪ targets 并集 + 持仓补价（修复"掉出候选就卖不掉"）。"""

    @staticmethod
    def _cli():
        import importlib
        import sys as _sys
        from pathlib import Path as _Path

        root = str(_Path(__file__).resolve().parent.parent)
        if root not in _sys.path:
            _sys.path.insert(0, root)
        return importlib.import_module("paper")

    def _args(self):
        return self._cli().build_parser().parse_args([])

    def test_price_table_unions_analysis_and_targets(self) -> None:
        cli = self._cli()
        analysis = pd.DataFrame(
            [
                {"date": "2026-07-10", "ticker": "AAA", "latest_price": 10.0},
                {"date": "2026-07-10", "ticker": "BBB", "latest_price": 20.0},
            ]
        )
        targets = pd.DataFrame(
            [{"date": "2026-07-10", "ticker": "BBB", "latest_price": 21.0, "weight": 0.2}]
        )

        table = cli._load_prices(self._args(), analysis, targets, state=None)

        tickers = set(table["ticker"])
        self.assertEqual(tickers, {"AAA", "BBB"})  # AAA 不再因只看 targets 而消失
        bbb = float(table.loc[table["ticker"] == "BBB", "adj_close"].iloc[0])
        self.assertEqual(bbb, 21.0)  # targets 排在后面，覆盖 analysis 的旧价

    def test_stale_dated_ticker_still_priced_at_latest_date(self) -> None:
        cli = self._cli()
        analysis = pd.DataFrame(
            [
                {"date": "2026-07-09", "ticker": "OLD", "latest_price": 5.0},  # 数据滞后一天
                {"date": "2026-07-10", "ticker": "NEW", "latest_price": 8.0},
            ]
        )
        table = cli._load_prices(self._args(), analysis, analysis.iloc[0:0], state=None)
        self.assertEqual(set(table["ticker"]), {"OLD", "NEW"})
        self.assertEqual(table["date"].nunique(), 1)  # 全部压到最新日期，滞后票不再隐形

    def test_held_ticker_missing_from_scan_gets_live_price(self) -> None:
        cli = self._cli()
        analysis = pd.DataFrame([{"date": "2026-07-10", "ticker": "NEWT", "latest_price": 50.0}])
        state = pd.DataFrame(
            [
                {"ticker": "CASH", "quantity": 599.8},
                {"ticker": "ANET", "quantity": 1.0818},  # 掉出扫描池的持仓
            ]
        )
        table = cli._load_prices(
            self._args(),
            analysis,
            analysis.iloc[0:0],
            state=state,
            fetch_close_fn=lambda ticker: 185.0 if ticker == "ANET" else None,
        )
        anet = table.loc[table["ticker"] == "ANET"]
        self.assertEqual(len(anet), 1)
        self.assertEqual(float(anet["adj_close"].iloc[0]), 185.0)

    def test_held_ticker_unfetchable_is_skipped_without_crash(self) -> None:
        cli = self._cli()
        analysis = pd.DataFrame([{"date": "2026-07-10", "ticker": "NEWT", "latest_price": 50.0}])
        state = pd.DataFrame([{"ticker": "GONE", "quantity": 3.0}])
        table = cli._load_prices(
            self._args(), analysis, analysis.iloc[0:0], state=state,
            fetch_close_fn=lambda ticker: None,
        )
        self.assertNotIn("GONE", set(table["ticker"]))  # 补价失败：跳过并警告（引擎侧会亮 unpriced_positions）


class SpreadCostTests(unittest.TestCase):
    def test_buy_fills_above_close_by_half_spread(self) -> None:
        import pandas as pd
        from stock_selector.config import PaperTradingConfig
        from stock_selector.paper import _rebalance_to_targets
        targets = pd.DataFrame({"ticker": ["AAA"], "target_weight": [1.0]})
        prices = pd.DataFrame({"ticker": ["AAA"], "price": [100.0]})
        state = pd.DataFrame({"ticker": ["CASH"], "quantity": [1000.0]})
        cfg = PaperTradingConfig(spread_bps=10.0, slippage_bps=0.0, commission_bps=0.0)
        orders, post, summary = _rebalance_to_targets(targets, prices, state, cfg)
        row = orders.iloc[0]
        self.assertEqual(row["action"], "BUY")
        self.assertAlmostEqual(row["price"], 100.0 * (1 + 0.0010), places=4)  # fill above close
        self.assertAlmostEqual(row["close_price"], 100.0, places=4)
        self.assertGreater(float(row["spread_cost"]), 0.0)
        self.assertEqual(float(row["cash_before"]), 1000.0)
        self.assertAlmostEqual(float(row["cash_after"]), float(post.loc[post["ticker"] == "CASH", "quantity"].iloc[0]))
        self.assertAlmostEqual(
            float(row["total_transaction_cost"]),
            float(row["spread_cost"] + row["slippage_cost"] + row["commission_cost"]),
        )
        # Marked at close, the freshly-bought position is worth slightly less than paid.
        self.assertLess(float(summary["post_trade_value"]), 1000.0)
