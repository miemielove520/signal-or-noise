from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

import paper as paper_cli


class PaperCliTest(unittest.TestCase):
    def test_paper_cli_writes_simulation_outputs_from_scan_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scan_csv = root / "scan.csv"
            output_dir = root / "paper_output"
            pd.DataFrame(
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
                    }
                ]
            ).to_csv(scan_csv, index=False)

            exit_code = paper_cli.main(
                [
                    "--scan-csv",
                    str(scan_csv),
                    "--output-dir",
                    str(output_dir),
                    "--state-csv",
                    str(root / "paper_state.csv"),
                    "--initial-cash",
                    "10000",
                    "--min-trade-value",
                    "1",
                    "--trade-buffer-pct",
                    "0",
                    "--slippage-bps",
                    "0",
                    "--ignore-market-regime-policy",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "paper_targets.csv").exists())
            self.assertTrue((output_dir / "orders.csv").exists())
            self.assertTrue((output_dir / "paper_trade_report.md").exists())
            self.assertTrue((output_dir / "paper_trade_result.json").exists())
            self.assertTrue((output_dir / "prices_used.csv").exists())
            self.assertTrue((output_dir / "history" / "paper_runs.csv").exists())
            self.assertTrue((output_dir / "history" / "paper_orders.csv").exists())
            self.assertTrue((output_dir / "history" / "paper_journal.md").exists())
            orders = pd.read_csv(output_dir / "orders.csv")
            self.assertEqual(orders.loc[0, "ticker"], "AAA")

    def test_paper_cli_applies_market_regime_policy_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scan_csv = root / "scan.csv"
            policy_csv = root / "market_regime_policy.csv"
            output_dir = root / "paper_output"
            pd.DataFrame(
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
            ).to_csv(scan_csv, index=False)
            pd.DataFrame(
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
            ).to_csv(policy_csv, index=False)

            exit_code = paper_cli.main(
                [
                    "--scan-csv",
                    str(scan_csv),
                    "--market-regime-policy",
                    str(policy_csv),
                    "--output-dir",
                    str(output_dir),
                    "--state-csv",
                    str(root / "paper_state.csv"),
                    "--initial-cash",
                    "10000",
                    "--min-trade-value",
                    "1",
                ]
            )

            self.assertEqual(exit_code, 0)
            targets = pd.read_csv(output_dir / "paper_targets.csv")
            self.assertTrue(targets.empty)
            self.assertTrue((output_dir / "orders.csv").exists())


if __name__ == "__main__":
    unittest.main()
