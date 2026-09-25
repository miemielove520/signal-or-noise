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
    DEFAULT_BENCHMARK,
    load_human_trades,
    update_human_trade_outcomes,
)
from stock_selector.model_log import (  # noqa: E402
    MODEL_TRADES_FILENAME,
    update_model_trades_from_paper_result,
)
from stock_selector.scoreboard import (  # noqa: E402
    build_head_to_head,
    prepare_model_outcomes,
    render_head_to_head_markdown,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the human-vs-model scoreboard report. 生成人机对照记分台报表。",
    )
    parser.add_argument("--review-root", default="outputs/signal_review")
    parser.add_argument("--period", default="2y", help="Price lookback period for outcomes.")
    parser.add_argument(
        "--benchmark",
        default=DEFAULT_BENCHMARK,
        help="Benchmark applied to model signals (QQQ or SPY).",
    )
    parser.add_argument(
        "--paper-result",
        default="outputs/paper_latest/paper_trade_result.json",
        help="State-updating paper result used as the model's executed decisions.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    review_root = Path(args.review_root)

    human_trades = load_human_trades(review_root)
    model_signals = update_model_trades_from_paper_result(
        paper_result_path=args.paper_result,
        review_root=review_root,
        benchmark=args.benchmark,
    )

    if human_trades.empty and model_signals.empty:
        print("No human trades and no model signals yet. / 还没有人类记录，也没有模型信号。")
        return 0

    tickers, benchmarks = _collect_symbols(human_trades, model_signals, args.benchmark)
    print(f"Fetching prices for {len(tickers)} tickers and {len(benchmarks)} benchmarks...")
    price_lookup = _download_many(tickers, args.period)
    benchmark_lookup = _download_many(benchmarks, args.period)

    human_filled = update_human_trade_outcomes(price_lookup, benchmark_lookup, review_root)
    model_filled = prepare_model_outcomes(
        model_signals, price_lookup, benchmark_lookup, benchmark=args.benchmark
    )
    if not model_filled.empty:
        review_root.mkdir(parents=True, exist_ok=True)
        model_filled.to_csv(review_root / MODEL_TRADES_FILENAME, index=False)

    head_to_head = build_head_to_head(human_filled, model_filled)
    report = render_head_to_head_markdown(head_to_head)

    review_root.mkdir(parents=True, exist_ok=True)
    md_path = review_root / "human_vs_model.md"
    csv_path = review_root / "human_vs_model.csv"
    md_path.write_text(report, encoding="utf-8")
    head_to_head.to_csv(csv_path, index=False)

    print(report)
    print(f"Report / 报表: {md_path}")
    print(f"Table / 表格: {csv_path}")
    return 0


def _collect_symbols(
    human_trades: pd.DataFrame,
    model_signals: pd.DataFrame,
    benchmark: str,
) -> tuple[list[str], list[str]]:
    tickers: set[str] = set()
    benchmarks: set[str] = {str(benchmark).upper().strip()}
    for frame in (human_trades, model_signals):
        if frame is None or frame.empty or "ticker" not in frame.columns:
            continue
        tickers.update(str(t).upper().strip() for t in frame["ticker"] if str(t).strip())
    if "benchmark" in human_trades.columns:
        benchmarks.update(
            str(b).upper().strip() for b in human_trades["benchmark"] if str(b).strip()
        )
    return sorted(tickers), sorted(benchmarks)


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


if __name__ == "__main__":
    raise SystemExit(main())
