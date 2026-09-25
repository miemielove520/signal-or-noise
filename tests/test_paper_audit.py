from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from stock_selector.paper_audit import (
    new_audit_identity,
    record_rebalance_failure,
    record_rebalance_success,
    record_valuation,
)


class PaperAuditTest(unittest.TestCase):
    def test_successful_rebalance_writes_immutable_snapshot_and_ledgers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_id, started_at = new_audit_identity(
                "rebalance",
                datetime(2026, 7, 13, 22, 0, tzinfo=timezone.utc),
            )
            analysis = pd.DataFrame([{"date": "2026-07-13", "ticker": "AAA", "score": 80}])
            prices = pd.DataFrame(
                [
                    {
                        "date": "2026-07-13",
                        "source_date": "2026-07-13",
                        "ticker": "AAA",
                        "adj_close": 100.0,
                        "price_source": "analysis_input",
                    }
                ]
            )
            targets = pd.DataFrame([{"date": "2026-07-13", "ticker": "AAA", "weight": 0.5}])
            orders = pd.DataFrame(
                [
                    {
                        "ticker": "AAA",
                        "action": "BUY",
                        "quantity": 5.0,
                        "signed_quantity": 5.0,
                        "price": 100.1,
                        "close_price": 100.0,
                        "notional": 500.5,
                        "spread_cost": 0.5,
                        "slippage_cost": 0.25,
                        "commission_cost": 0.0,
                        "estimated_cost": 0.25,
                        "total_transaction_cost": 0.75,
                        "target_weight": 0.5,
                        "current_weight": 0.0,
                        "target_value": 500.0,
                        "current_value": 0.0,
                        "position_quantity_before": 0.0,
                        "position_quantity_after": 5.0,
                        "cash_before": 1000.0,
                        "cash_after": 499.25,
                    }
                ]
            )
            pre_state = pd.DataFrame([{"ticker": "CASH", "quantity": 1000.0}])
            post_state = pd.DataFrame(
                [
                    {"ticker": "CASH", "quantity": 499.25},
                    {"ticker": "AAA", "quantity": 5.0},
                ]
            )
            summary = {
                "pre_trade_value": 1000.0,
                "post_trade_value": 999.25,
                "post_trade_cash": 499.25,
                "cash_weight": 0.4996,
                "total_spread_cost": 0.5,
                "total_slippage_cost": 0.25,
                "total_commission_cost": 0.0,
                "total_transaction_cost": 0.75,
            }
            payload = {"as_of_date": "2026-07-13", "summary": summary}

            snapshot = record_rebalance_success(
                root,
                run_id=run_id,
                started_at=started_at,
                finished_at="2026-07-13T22:00:05Z",
                state_updated=True,
                analysis=analysis,
                prices=prices,
                targets=targets,
                orders=orders,
                pre_trade_state=pre_state,
                post_trade_state=post_state,
                result_payload=payload,
                report="# report\n",
                config_snapshot={"initial_cash": 1000},
                source_snapshot={"scan_csv": "scan.csv", "state_csv": "state.csv"},
                project_root=Path(tmpdir),
            )

            self.assertTrue((snapshot / "run_manifest.json").exists())
            self.assertTrue((snapshot / "analysis_input.csv").exists())
            self.assertTrue((root / "paper_runs.csv").exists())
            self.assertTrue((root / "paper_orders.csv").exists())
            self.assertTrue((root / "paper_positions.csv").exists())
            self.assertTrue((root / "paper_journal.md").exists())
            manifest = json.loads((snapshot / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "success")
            self.assertEqual(manifest["counts"]["orders"], 1)
            order_ledger = pd.read_csv(root / "paper_orders.csv")
            self.assertEqual(order_ledger.loc[0, "cash_after"], 499.25)
            positions = pd.read_csv(root / "paper_positions.csv")
            aaa = positions[(positions["stage"] == "post_trade") & (positions["ticker"] == "AAA")]
            self.assertEqual(float(aaa.iloc[0]["market_value"]), 500.0)

    def test_failed_rebalance_is_not_silently_lost(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_id, started_at = new_audit_identity("rebalance")
            snapshot = record_rebalance_failure(
                root,
                run_id=run_id,
                started_at=started_at,
                finished_at="2026-07-13T22:00:05Z",
                state_updated=True,
                error="price feed failed",
                config_snapshot={},
                source_snapshot={},
                project_root=root,
            )
            manifest = json.loads((snapshot / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            runs = pd.read_csv(root / "paper_runs.csv")
            self.assertEqual(runs.loc[0, "error"], "price feed failed")
            self.assertFalse(bool(runs.loc[0, "state_updated"]))

    def test_valuation_records_current_and_fallback_prices(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            valuation_id, executed_at = new_audit_identity("valuation")
            state = pd.DataFrame(
                [
                    {"ticker": "CASH", "quantity": 100.0},
                    {"ticker": "AAA", "quantity": 2.0},
                    {"ticker": "BBB", "quantity": 3.0},
                ]
            )
            snapshot = record_valuation(
                root,
                valuation_id=valuation_id,
                executed_at=executed_at,
                date="2026-07-13",
                status="success",
                state=state,
                current_prices={"AAA": 10.0},
                fallback_prices={"BBB": 20.0},
                benchmark_ticker="QQQ",
                benchmark_close=600.0,
                warnings=["BBB fallback"],
                performance={"total_return": 0.01, "readiness": "accumulating"},
                backtest_gate={"gate": "accumulating"},
                equity=180.0,
            )
            prices = pd.read_csv(snapshot / "valuation_prices.csv")
            status = dict(zip(prices["ticker"], prices["price_status"]))
            self.assertEqual(status["AAA"], "current")
            self.assertEqual(status["BBB"], "fallback")
            ledger = pd.read_csv(root / "paper_valuations.csv")
            self.assertEqual(int(ledger.loc[0, "fallback_price_count"]), 1)


if __name__ == "__main__":
    unittest.main()
