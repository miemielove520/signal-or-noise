from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from stock_selector.model_log import update_model_trades_from_paper_result


class ModelLogTests(unittest.TestCase):
    def test_state_updating_paper_orders_are_logged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = root / "paper_trade_result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "as_of_date": "2026-07-10T00:00:00",
                        "state_updated": True,
                        "summary": {"pre_trade_value": 1000.0},
                        "orders": [
                            {
                                "ticker": "AAPL",
                                "action": "BUY",
                                "price": 200.0,
                                "notional": 180.0,
                                "quantity": 0.9,
                                "estimated_cost": 0.1,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            trades = update_model_trades_from_paper_result(result_path, root / "review")

            self.assertEqual(len(trades), 1)
            self.assertEqual(trades.iloc[0]["actor"], "model")
            self.assertEqual(trades.iloc[0]["action"], "BUY")
            self.assertAlmostEqual(float(trades.iloc[0]["weight_pct"]), 18.0)
            self.assertTrue((root / "review" / "model_trades.csv").exists())

    def test_dry_run_orders_are_not_logged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = root / "paper_trade_result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "as_of_date": "2026-07-10",
                        "state_updated": False,
                        "orders": [{"ticker": "AAPL", "action": "BUY", "price": 200.0}],
                    }
                ),
                encoding="utf-8",
            )

            trades = update_model_trades_from_paper_result(result_path, root / "review")

            self.assertTrue(trades.empty)
            self.assertFalse((root / "review" / "model_trades.csv").exists())


if __name__ == "__main__":
    unittest.main()
