"""Export the frozen experiment's raw results into ``results/data/``.

The daily pipeline writes its outputs to a local, git-ignored ``outputs/``
folder. This script copies the pieces the analysis needs, keeps only
research columns (no local paths, no personal holdings), and downloads the
benchmark / universe prices for the forward window, so that
``research/forward_test.py`` can be re-run by anyone from the repo alone.

Usage:
    python scripts/export_results.py --pipeline-dir ../stock-selector-latest
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "data"

WALK_FORWARD_RUN = "outputs/walk_forward/v2_baseline_2026-07-10"
SIGNAL_COLUMNS = [
    "date", "ticker", "horizon", "screening_profile", "validation_bucket",
    "high_probability_score", "calibrated_win_probability", "signal_score",
    "forward_return_5d", "forward_return_20d", "forward_return_60d",
    "point_in_time_universe_member",
]
ORDER_COLUMNS = [
    "as_of_date", "order_sequence", "ticker", "action", "quantity", "close_price",
    "fill_price", "notional", "total_transaction_cost", "position_quantity_after",
]
BACKTEST_CURVE_COLUMNS = [
    "date", "portfolio_name", "benchmark_ticker", "portfolio_daily_return",
    "benchmark_daily_return", "excess_daily_return",
]


def _write(frame: pd.DataFrame, name: str) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DATA / name, index=False)
    print(f"wrote results/data/{name} ({len(frame)} rows)")


def export_local(pipeline: Path) -> None:
    wf = pipeline / WALK_FORWARD_RUN
    _write(pd.read_csv(wf / "walk_forward_events.csv", usecols=SIGNAL_COLUMNS, low_memory=False),
           "walk_forward_signals.csv")
    _write(pd.read_csv(wf / "benchmark_comparison_curve.csv", usecols=BACKTEST_CURVE_COLUMNS),
           "backtest_portfolio_vs_benchmark.csv")
    _write(pd.read_csv(wf / "probability_calibration.csv"), "backtest_probability_calibration.csv")
    _write(pd.read_csv(pipeline / "outputs/paper_latest/paper_equity_history.csv"), "paper_equity_history.csv")
    _write(pd.read_csv(pipeline / "outputs/paper_history/paper_orders.csv", usecols=ORDER_COLUMNS),
           "paper_orders.csv")
    universe = [
        line.strip() for line in (pipeline / "candidates_universe.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    _write(pd.DataFrame({"ticker": universe}), "universe.csv")


def download_prices(start: str, end: str) -> None:
    import yfinance as yf

    universe = pd.read_csv(DATA / "universe.csv")["ticker"].tolist()
    closes = yf.download(universe + ["QQQ", "SPY"], start=start, end=end,
                         auto_adjust=True, progress=False)["Close"]
    closes.index.name = "date"
    _write(closes.reset_index(), "forward_window_prices.csv")
    qqq = yf.download(["QQQ"], start="2025-01-01", end=end, auto_adjust=True, progress=False)["Close"]
    qqq.index.name = "date"
    _write(qqq.reset_index(), "qqq_history.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pipeline-dir", type=Path, required=True,
                        help="Checkout whose outputs/ holds the running experiment.")
    parser.add_argument("--start", default="2026-07-01")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    export_local(args.pipeline_dir.expanduser().resolve())
    if not args.skip_download:
        download_prices(args.start, args.end)


if __name__ == "__main__":
    main()
