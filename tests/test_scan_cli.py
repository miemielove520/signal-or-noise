from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

import scan as scan_cli


class ScanCliTests(unittest.TestCase):
    def test_partial_ticker_failures_return_nonzero(self) -> None:
        result = SimpleNamespace(
            tickers=("AAA", "MISSING"),
            results=(object(),),
            output_dir=Path("outputs/scans/latest"),
            top_candidates=pd.DataFrame(),
            summary=pd.DataFrame(),
            failures=({"ticker": "MISSING", "error": "no price data"},),
            journal=None,
        )
        with patch("scan.load_universe_tickers", return_value=["AAA", "MISSING"]), patch(
            "scan.load_screening_config", return_value=object()
        ), patch("scan.run_high_probability_scan", return_value=result):
            exit_code = scan_cli.main(["AAA", "MISSING"])

        self.assertEqual(exit_code, 1)

    def test_successful_scan_returns_zero(self) -> None:
        result = SimpleNamespace(
            tickers=("AAA",),
            results=(object(),),
            output_dir=Path("outputs/scans/latest"),
            top_candidates=pd.DataFrame(),
            summary=pd.DataFrame(),
            failures=(),
            journal=None,
        )
        with patch("scan.load_universe_tickers", return_value=["AAA"]), patch(
            "scan.load_screening_config", return_value=object()
        ), patch("scan.run_high_probability_scan", return_value=result):
            exit_code = scan_cli.main(["AAA"])

        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
