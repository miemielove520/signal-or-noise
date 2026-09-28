"""Smoke tests for every `stock-selector` subcommand that can run offline.

Commands run against the bundled synthetic sample data; commands that would
download prices get the same stubs as the end-to-end pipeline test.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from stock_selector import cli
from test_real_ticker_pipeline import _fake_download, _fake_event_risk

CONFIG = str(Path(__file__).resolve().parents[1] / "configs" / "default.toml")


def run_cli(*argv: str) -> tuple[int, str]:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = cli.main(list(argv))
    return code, buffer.getvalue()


class SampleDataCommandsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.out = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_run_backtest(self) -> None:
        code, text = run_cli("run", "--config", CONFIG, "--output-dir", str(self.out / "run"))
        self.assertEqual(code, 0)
        self.assertIn("Backtest metrics", text)
        self.assertTrue((self.out / "run" / "equity_curve.csv").exists())

    def test_ml_run(self) -> None:
        code, _ = run_cli("ml-run", "--config", CONFIG, "--output-dir", str(self.out / "ml"))
        self.assertEqual(code, 0)

    def test_paper_trade_and_daily_report(self) -> None:
        code, _ = run_cli("paper-trade", "--config", CONFIG, "--output-dir", str(self.out / "paper"))
        self.assertEqual(code, 0)
        code, _ = run_cli("daily-report", "--config", CONFIG, "--output-dir", str(self.out / "daily"))
        self.assertEqual(code, 0)

    def test_analyze_ticker(self) -> None:
        code, text = run_cli("analyze-ticker", "--config", CONFIG, "--ticker", "ALFA",
                             "--output-dir", str(self.out / "ticker"))
        self.assertEqual(code, 0)
        self.assertIn("ALFA", text)

    def test_audit(self) -> None:
        code, _ = run_cli("audit", "--config", CONFIG, "--issues-csv", str(self.out / "issues.csv"))
        self.assertEqual(code, 0)


class NetworkCommandsTest(unittest.TestCase):
    """Commands that normally download data, with the downloads stubbed."""

    def test_real_prints_the_shared_console_report(self) -> None:
        with tempfile.TemporaryDirectory() as root, \
                patch("stock_selector.real_data.download_prices_for_period_multi_source", side_effect=_fake_download), \
                patch("stock_selector.real_data.fetch_yfinance_event_risk", side_effect=_fake_event_risk):
            code, text = run_cli("real", "TEST", "--period", "2y", "--no-snapshot", "--no-peers",
                                 "--output-root", str(Path(root) / "real_ticker"))
        self.assertEqual(code, 0)
        for section in ("Real ticker analysis completed.", "Data source readiness / 数据源准备度",
                        "Final decision / 最终执行结论", "High probability filter / 高概率筛选器",
                        "Entry backtest / 买点回测"):
            self.assertIn(section, text)

    def test_validate_then_compare_runs(self) -> None:
        with tempfile.TemporaryDirectory() as root, \
                patch("stock_selector.validation_cli.download_prices_for_period_multi_source",
                      side_effect=_fake_download):
            first, second = Path(root) / "first", Path(root) / "second"
            for folder, step in ((first, "60"), (second, "40")):
                code, text = run_cli("validate", "AAA", "BBB", "--period", "2y", "--step-days", step,
                                     "--min-history-days", "120", "--output-dir", str(folder))
                self.assertEqual(code, 0)
            manifest = json.loads((second / "run_manifest.json").read_text())
            code, compare_text = run_cli("compare-runs", str(first), str(second),
                                         "--output-dir", str(Path(root) / "compare"))
        self.assertIn("Walk-forward validation completed.", text)
        self.assertEqual(manifest["step_days"], 40)
        self.assertEqual(code, 0)
        self.assertTrue(compare_text.strip())

    def test_validate_rejects_conflicting_arguments(self) -> None:
        with self.assertRaises(ValueError):
            run_cli("validate", "--all-universes", "AAPL")

    def test_review_due_with_no_history(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            code, _ = run_cli("review-due", "--review-root", str(Path(root) / "signal_review"),
                              "--output-dir", str(Path(root) / "out"))
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
