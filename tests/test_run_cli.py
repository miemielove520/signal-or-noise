from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

import run


class RunCliTest(unittest.TestCase):
    def test_format_user_error_suggests_common_ticker_match(self) -> None:
        message = run.format_user_error(
            "APPPL",
            ValueError("No price data returned from configured providers."),
        )

        self.assertIn("Analysis did not run / 分析没有成功", message)
        self.assertIn("Possible ticker match / 可能想输入的是: AAPL", message)
        self.assertIn("Data note / 数据说明", message)
        self.assertIn("python3 run.py AAPL", message)

    def test_main_prints_user_facing_error_and_returns_nonzero(self) -> None:
        stderr = io.StringIO()
        with patch(
            "run.run_real_ticker_analysis",
            side_effect=ValueError("No usable ticker data returned from yfinance."),
        ):
            with redirect_stderr(stderr):
                exit_code = run.main(["BADTICKER", "--no-snapshot", "--no-peers"])

        self.assertEqual(exit_code, 2)
        output = stderr.getvalue()
        self.assertIn("Analysis did not run / 分析没有成功", output)
        self.assertIn("BADTICKER", output)
        self.assertIn("Do not type the ticker alone", output)

    def test_main_handles_empty_ticker(self) -> None:
        stderr = io.StringIO()
        with patch("builtins.input", return_value=""):
            with redirect_stderr(stderr):
                exit_code = run.main([])

        self.assertEqual(exit_code, 2)
        self.assertIn("Ticker cannot be empty", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
