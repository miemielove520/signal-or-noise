from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from stock_selector.journal import compare_scan_snapshots, write_daily_journal


class JournalTest(unittest.TestCase):
    def test_compare_scan_snapshots_detects_upgrade(self) -> None:
        previous = pd.DataFrame(
            [
                make_scan_row("AAA", "early_watch", 50.0),
            ]
        )
        current = pd.DataFrame(
            [
                make_scan_row("AAA", "close_but_not_ready", 63.0),
            ]
        )

        changes = compare_scan_snapshots(current, previous)

        self.assertEqual(changes.iloc[0]["change_type"], "upgraded")

    def test_write_daily_journal_compares_previous_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = pd.DataFrame([make_scan_row("AAA", "early_watch", 50.0)])
            second = pd.DataFrame([make_scan_row("AAA", "close_but_not_ready", 63.0)])

            write_daily_journal(first, journal_root=root, run_date=date(2026, 1, 2))
            result = write_daily_journal(second, journal_root=root, run_date=date(2026, 1, 3))

            self.assertTrue(result.snapshot_path.exists())
            self.assertTrue(result.report_path.exists())
            self.assertTrue((root / "journal_result.json").exists())
            payload = json.loads((root / "journal_result.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["journal_date"], "2026-01-03")
            self.assertIn("changes", payload)
            self.assertEqual(result.changes.iloc[0]["change_type"], "upgraded")


def make_scan_row(ticker: str, status: str, score: float) -> dict[str, object]:
    return {
        "ticker": ticker,
        "watchlist_status": status,
        "high_probability_score": score,
        "quality_gate_passed": status == "ready_high_probability",
        "watchlist_trigger_price": 100.0,
        "watchlist_missing_items_zh": "信号分数不足",
        "report_path": f"outputs/{ticker}/ticker_analysis.md",
    }


if __name__ == "__main__":
    unittest.main()
