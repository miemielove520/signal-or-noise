from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd  # noqa: E402

from stock_selector.human_log import (  # noqa: E402
    FORWARD_WINDOWS,
    load_human_trades,
    update_human_trade_outcomes,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fill forward and excess-vs-benchmark outcomes for your logged trades. "
        "为你记录的交易填入前瞻收益和相对基准的超额收益。",
    )
    parser.add_argument(
        "--review-root",
        default="outputs/signal_review",
        help="Where signal-review files live.",
    )
    parser.add_argument(
        "--period",
        default="2y",
        help="Price lookback period used to fetch outcomes.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    trades = load_human_trades(args.review_root)
    if trades.empty:
        print("No human trades logged yet. Use log_trade.py first. / 还没有记录，请先用 log_trade.py。")
        return 0

    tickers = sorted({str(t).upper().strip() for t in trades["ticker"] if str(t).strip()})
    benchmarks = sorted({str(b).upper().strip() for b in trades.get("benchmark", []) if str(b).strip()})

    print(f"Fetching prices for {len(tickers)} tickers and {len(benchmarks)} benchmarks...")
    price_lookup = _download_many(tickers, args.period)
    benchmark_lookup = _download_many(benchmarks, args.period)

    filled = update_human_trade_outcomes(
        price_lookup=price_lookup,
        benchmark_lookup=benchmark_lookup,
        review_root=args.review_root,
    )

    _print_summary(filled)
    print(f"Updated / 已更新: {Path(args.review_root) / 'human_trades.csv'}")
    return 0


def _download_many(symbols: list[str], period: str) -> dict[str, pd.DataFrame]:
    from stock_selector.data import download_prices_for_period_multi_source

    lookup: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            result = download_prices_for_period_multi_source([symbol], period=period)
            lookup[symbol] = result.prices
        except Exception as error:
            print(f"- {symbol}: price fetch failed ({error})", file=sys.stderr)
            lookup[symbol] = pd.DataFrame()
    return lookup


def _print_summary(filled: pd.DataFrame) -> None:
    print()
    print("Outcome summary / 结果汇总")
    for window in FORWARD_WINDOWS:
        status_col = f"outcome_status_{window}d"
        excess_col = f"excess_return_{window}d"
        if status_col not in filled.columns:
            continue
        status = filled[status_col].astype(str)
        beat = int((status == "beat_benchmark").sum())
        lagged = int((status == "lagged_benchmark").sum())
        pending = int((status == "pending").sum())
        completed = beat + lagged
        avg_excess = (
            pd.to_numeric(filled[excess_col], errors="coerce").mean()
            if excess_col in filled.columns
            else float("nan")
        )
        win_rate = (beat / completed) if completed else float("nan")
        print(
            f"- {window}d: completed={completed} (beat={beat}, lagged={lagged}), "
            f"pending={pending}, beat_rate={_pct(win_rate)}, avg_excess={_pct(avg_excess)}"
        )


def _pct(value: object) -> str:
    try:
        if value != value:  # NaN
            return "N/A"
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "N/A"


if __name__ == "__main__":
    raise SystemExit(main())
