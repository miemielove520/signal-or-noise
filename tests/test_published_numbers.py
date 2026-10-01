"""The README and the report quote the numbers that research/forward_test.py actually produced.

If the data are refreshed and summary.json changes, this test fails until the prose is updated,
so the published text can never silently drift from the analysis.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = json.loads((ROOT / "results" / "summary.json").read_text(encoding="utf-8"))
README = (ROOT / "README.md").read_text(encoding="utf-8")
REPORT = (ROOT / "results" / "REPORT.md").read_text(encoding="utf-8")


def pct(value: float, digits: int = 1, sign: bool = True) -> str:
    """Format a fraction the way the documents do, e.g. -0.24274 -> '−24.3 %'."""
    text = f"{value * 100:{'+' if sign else ''}.{digits}f} %"
    return text.replace("-", "−")


class PublishedNumbersTest(unittest.TestCase):
    def setUp(self) -> None:
        self.forward = SUMMARY["forward_test"]
        self.luck = SUMMARY["luck_test"]
        self.impl = SUMMARY["implementation"]
        self.signals = SUMMARY["signal_tests"]

    def assertQuoted(self, text: str, *needles: str) -> None:
        for needle in needles:
            self.assertIn(needle, text, f"{needle!r} not found; update the prose to match summary.json")

    def test_headline_numbers_in_both_documents(self) -> None:
        low, high = self.forward["mean_daily_excess_ci95"]
        for doc in (README, REPORT):
            self.assertQuoted(
                doc,
                pct(self.forward["model_return"]),
                pct(self.forward["qqq_return"]),
                pct(self.forward["universe_equal_weight_return"]),
                pct(self.luck["backtest_total_return"], 0),
                pct(self.luck["backtest_qqq_return"], 0),
                f"{pct(low, 2)} to {pct(high, 2)}",
                f"{self.impl['orders']}",
                f"{self.impl['turnover_multiple_of_capital']:.1f}×",
            )

    def test_report_details(self) -> None:
        f, luck, impl, sig = self.forward, self.luck, self.impl, self.signals
        low, high = f["mean_daily_excess_ci95"]
        self.assertQuoted(
            REPORT,
            f"[{pct(low, 2)}, {pct(high, 2)}]",
            pct(f["relative_excess_vs_qqq"]),
            pct(f["max_drawdown"]),
            pct(f["prespecified_check"]["model_return"]),
            pct(f["prespecified_check"]["qqq_return"]),
            f"{f['trading_days']} trading sessions",
            pct(luck["p_value_as_bad_as_observed"], 2, sign=False),
            pct(luck["simulated_excess_median"]),
            pct(luck["simulated_excess_5th_percentile"]),
            f"{luck['backtest_t_stat_daily_excess']:.2f} (p = {luck['backtest_p_value_daily_excess']:.2f})",
            pct(impl["transaction_costs_pct_of_capital"], 2, sign=False),
            f"{impl['closed_round_trips']} completed trades",
            f"{impl['counterfactual']['trades_with_full_window']} completed trades",
            f"{sig['ic_mean']:+.3f}",
            pct(sig["top5_minus_universe_mean_20d"]),
            f"{sig['brier_skill_score']:.2f}".replace("-", "−"),
            f"{sig['distinct_date_ticker_outcomes']:,}",
            f"**{sig['dates_needed_for_ic']['0.05']}**",
            f"**{f['power']['observations_to_detect_modest_edge']:,}**",
            f"**{round(sig['effective_independent_outcomes'])} independent observations**",
        )


if __name__ == "__main__":
    unittest.main()
