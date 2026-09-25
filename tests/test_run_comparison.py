from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

import pandas as pd

from stock_selector.run_comparison import compare_validation_runs


class RunComparisonTest(unittest.TestCase):
    def test_compare_validation_runs_writes_summary_and_ticker_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = root / "previous"
            current = root / "current"
            output = root / "comparison"
            _write_run(
                previous,
                event_count=10,
                win_rate=0.50,
                avg_return=0.01,
                ticker_rows=[
                    {
                        "rank": 1,
                        "ticker": "AAA",
                        "ticker_decision": "watchlist_candidate",
                        "ticker_decision_zh": "观察候选",
                        "ticker_ranking_score": 60.0,
                        "sample_count": 10,
                    },
                    {
                        "rank": 2,
                        "ticker": "BBB",
                        "ticker_decision": "deprioritize",
                        "ticker_decision_zh": "降低优先级",
                        "ticker_ranking_score": 30.0,
                        "sample_count": 10,
                    },
                ],
            )
            _write_run(
                current,
                event_count=20,
                win_rate=0.60,
                avg_return=0.03,
                ticker_rows=[
                    {
                        "rank": 1,
                        "ticker": "BBB",
                        "ticker_decision": "priority_candidate",
                        "ticker_decision_zh": "优先候选",
                        "ticker_ranking_score": 78.0,
                        "sample_count": 14,
                    },
                    {
                        "rank": 2,
                        "ticker": "AAA",
                        "ticker_decision": "watchlist_candidate",
                        "ticker_decision_zh": "观察候选",
                        "ticker_ranking_score": 64.0,
                        "sample_count": 12,
                    },
                ],
            )

            result = compare_validation_runs(previous, current, output_dir=output)

            summary = result.summary.set_index("metric")
            tickers = result.ticker_comparison.set_index("ticker")
            adoption = result.adoption_decision.set_index("check_name")
            self.assertTrue((output / "run_comparison_summary.csv").exists())
            self.assertTrue((output / "run_comparison_tickers.csv").exists())
            self.assertTrue((output / "config_adoption_decision.csv").exists())
            self.assertTrue((output / "run_comparison.md").exists())
            self.assertTrue((output / "run_comparison.json").exists())

        self.assertEqual(summary.loc["event_count", "assessment"], "improved")
        self.assertGreater(summary.loc["weighted_20d_win_rate", "change"], 0)
        self.assertEqual(
            adoption.loc["final_decision", "status"],
            "candidate_config_passed_review",
        )
        self.assertEqual(tickers.loc["BBB", "rank_change"], 1)
        self.assertGreater(tickers.loc["BBB", "score_change"], 0)
        self.assertIn("Validation Run Comparison", result.report)
        self.assertIn("Config Adoption Decision / 配置采用结论", result.report)

    def test_compare_validation_runs_rejects_worse_candidate_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = root / "previous"
            current = root / "current"
            output = root / "comparison"
            _write_run(
                previous,
                event_count=40,
                win_rate=0.62,
                avg_return=0.04,
                ticker_rows=[
                    {
                        "rank": 1,
                        "ticker": "AAA",
                        "ticker_decision": "priority_candidate",
                        "ticker_decision_zh": "优先候选",
                        "ticker_ranking_score": 75.0,
                        "sample_count": 20,
                    }
                ],
            )
            _write_run(
                current,
                event_count=12,
                win_rate=0.50,
                avg_return=-0.01,
                ticker_rows=[
                    {
                        "rank": 1,
                        "ticker": "AAA",
                        "ticker_decision": "deprioritize",
                        "ticker_decision_zh": "降低优先级",
                        "ticker_ranking_score": 30.0,
                        "sample_count": 5,
                    }
                ],
            )

            result = compare_validation_runs(previous, current, output_dir=output)

        adoption = result.adoption_decision.set_index("check_name")
        self.assertEqual(
            adoption.loc["final_decision", "status"],
            "reject_candidate_config",
        )
        self.assertEqual(adoption.loc["win_rate_not_worse", "status"], "failed")


def _write_run(
    run_dir: Path,
    event_count: int,
    win_rate: float,
    avg_return: float,
    ticker_rows: list[dict[str, object]],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "started_at_utc": "2026-01-01T00:00:00+00:00",
                "completed_at_utc": "2026-01-01T00:00:01+00:00",
                "duration_seconds": 1.0,
                "period": "1y",
                "step_days": 20,
                "min_history_days": 80,
                "price_provider": "fake",
                "price_warnings": [],
                "missing_price_tickers": [],
                "event_count": event_count,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "bucket": "filtered_out",
                "sample_count": event_count,
                "win_rate_20d": win_rate,
                "avg_return_20d": avg_return,
            }
        ]
    ).to_csv(run_dir / "walk_forward_summary.csv", index=False)
    pd.DataFrame(ticker_rows).to_csv(run_dir / "ticker_validation_ranking.csv", index=False)
    pd.DataFrame(
        [
            {
                "scope": "overall",
                "ticker": "ALL",
                "sample_status": "enough_samples",
            }
        ]
    ).to_csv(run_dir / "sample_sufficiency_guidance.csv", index=False)


if __name__ == "__main__":
    unittest.main()
