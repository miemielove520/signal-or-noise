from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from stock_selector.position import build_position_context


def _portfolio(tmp: str) -> Path:
    path = Path(tmp) / "portfolio.csv"
    pd.DataFrame(
        [
            {"ticker": "AVGO", "shares": 6.881, "avg_cost": 385.65, "current_price": 389.86,
             "market_value": 2682.62, "weight_pct": 37.95},
            {"ticker": "MRVL", "shares": 9.037, "avg_cost": 268.33, "current_price": 233.80,
             "market_value": 2112.85, "weight_pct": 29.89},
            {"ticker": "CASH", "shares": None, "avg_cost": None, "current_price": None,
             "market_value": 719.58, "weight_pct": 10.18},
        ]
    ).to_csv(path, index=False)
    return path


class PositionContextTests(unittest.TestCase):
    def test_held_position_profit_and_concentration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = build_position_context("AVGO", latest_price=389.86, portfolio_path=_portfolio(tmp))
            self.assertTrue(ctx.is_held)
            self.assertAlmostEqual(ctx.unrealized_pl_pct, 389.86 / 385.65 - 1, places=4)
            self.assertGreater(ctx.unrealized_pl, 0)  # in profit
            self.assertEqual(ctx.concentration_level, "extreme")  # 38% > 30%

    def test_held_position_loss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = build_position_context("MRVL", latest_price=233.80, portfolio_path=_portfolio(tmp))
            self.assertTrue(ctx.is_held)
            self.assertLess(ctx.unrealized_pl_pct, 0)  # underwater
            self.assertEqual(ctx.concentration_level, "high")  # ~30%

    def test_prefers_fresh_analysis_price_over_stored_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # 当日分析价（每次运行都会刷新）优先于 csv 里的手工快照价（隔天就过期）：
            # P/L 按今天的价算，csv 的 current_price 只在没有新价时兜底。
            ctx = build_position_context("AVGO", latest_price=401.11, portfolio_path=_portfolio(tmp))
            self.assertAlmostEqual(ctx.unrealized_pl_pct, 401.11 / 385.65 - 1, places=4)

    def test_falls_back_to_stored_price_without_fresh_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = build_position_context("AVGO", portfolio_path=_portfolio(tmp))
            self.assertAlmostEqual(ctx.unrealized_pl_pct, 389.86 / 385.65 - 1, places=4)

    def test_weight_override_recomputes_concentration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = build_position_context(
                "AVGO", latest_price=401.11, portfolio_path=_portfolio(tmp),
                weight_pct_override=41.2,
            )
            self.assertAlmostEqual(ctx.weight_pct, 41.2)
            self.assertEqual(ctx.concentration_level, "extreme")

    def test_falls_back_to_latest_when_no_stored_price(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.csv"
            pd.DataFrame([{"ticker": "AVGO", "shares": 6.881, "avg_cost": 385.65, "weight_pct": 38.0}]).to_csv(path, index=False)
            ctx = build_position_context("AVGO", latest_price=420.0, portfolio_path=path)
            self.assertAlmostEqual(ctx.unrealized_pl_pct, 420.0 / 385.65 - 1, places=4)

    def test_not_held(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = build_position_context("NVDA", portfolio_path=_portfolio(tmp))
            self.assertFalse(ctx.is_held)
            self.assertIn("entry", ctx.position_note.lower())

    def test_cash_is_not_a_position(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(build_position_context("CASH", portfolio_path=_portfolio(tmp)).is_held)

    def test_next_step_is_holder_oriented_not_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = build_position_context("AVGO", latest_price=389.86, portfolio_path=_portfolio(tmp))
            self.assertIn("方案参考", ctx.next_step_zh)  # concrete options, not just facts
            self.assertIn("降集中度", ctx.next_step_zh)  # extreme concentration option
            self.assertIn("385", ctx.next_step_zh)  # cost line referenced
            self.assertNotIn("收在", ctx.next_step_zh)  # not an entry trigger

    def test_deep_loss_prompts_thesis_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.csv"
            pd.DataFrame([{"ticker": "NVTS", "shares": 20.0, "avg_cost": 16.94,
                           "current_price": 13.0, "weight_pct": 3.8}]).to_csv(path, index=False)
            ctx = build_position_context("NVTS", portfolio_path=path)
            self.assertLess(ctx.unrealized_pl_pct, -0.15)
            self.assertIn("重新检视", ctx.next_step_zh)

    def test_missing_file(self) -> None:
        self.assertFalse(build_position_context("AVGO", portfolio_path="/nope/x.csv").is_held)


if __name__ == "__main__":
    unittest.main()
