from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stock_selector.human_log import (  # noqa: E402
    DEFAULT_BENCHMARK,
    VALID_ACTIONS,
    VALID_BENCHMARKS,
    append_human_trade,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Log your own trade decision into the human-vs-model scoreboard. "
        "记录你自己的交易决策，用于人机对照。",
    )
    parser.add_argument("ticker", nargs="?", help="Ticker symbol, e.g. AAPL.")
    parser.add_argument("--action", help=f"One of {', '.join(VALID_ACTIONS)}.")
    parser.add_argument("--weight", type=float, help="Portfolio weight percent, 0-100.")
    parser.add_argument("--price", type=float, help="Decision price. Fetched automatically if omitted.")
    parser.add_argument("--date", help="Decision date YYYY-MM-DD. Defaults to today.")
    parser.add_argument("--reason", default="", help="Why you made this decision.")
    parser.add_argument(
        "--benchmark",
        default=DEFAULT_BENCHMARK,
        help=f"Benchmark to score against: {', '.join(VALID_BENCHMARKS)}.",
    )
    parser.add_argument(
        "--review-root",
        default="outputs/signal_review",
        help="Where signal-review files live.",
    )
    parser.add_argument(
        "--period",
        default="5d",
        help="Price lookback period used only to fetch the latest close.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    ticker = (args.ticker or _prompt("Ticker / 股票代码: ")).upper().strip()
    if not ticker:
        print("Ticker cannot be empty. / 股票代码不能为空。", file=sys.stderr)
        return 2

    action = (args.action or _prompt(f"Action {VALID_ACTIONS} / 操作: ")).upper().strip()
    weight = args.weight if args.weight is not None else _prompt_float("Weight percent 0-100 / 仓位百分比: ")
    benchmark = (args.benchmark or DEFAULT_BENCHMARK).upper().strip()

    price = args.price
    if price is None:
        price = _fetch_latest_close(ticker, args.period)
        if price is not None:
            print(f"Fetched latest close / 自动获取最新收盘价: {price:.4f}")
        else:
            price = _prompt_float("Could not fetch price. Enter decision price / 手动输入决策价: ")

    reason = args.reason or _prompt("Reason (optional) / 理由（可选）: ")

    try:
        frame = append_human_trade(
            ticker=ticker,
            action=action,
            weight_pct=weight,
            signal_price=price,
            signal_date=args.date,
            reason=reason,
            benchmark=benchmark,
            review_root=args.review_root,
        )
    except ValueError as error:
        print(f"Not logged / 未记录: {error}", file=sys.stderr)
        return 2

    path = Path(args.review_root) / "human_trades.csv"
    latest = frame.iloc[-1]
    print()
    print("Logged your decision / 已记录你的决策")
    print(f"- {latest['signal_date']} {latest['ticker']} {latest['action']} "
          f"weight={latest['weight_pct']}% @ {latest['signal_price']} vs {latest['benchmark']}")
    print(f"- Total human decisions on file / 累计记录数: {len(frame)}")
    print(f"- File / 文件: {path}")
    return 0


def _fetch_latest_close(ticker: str, period: str) -> float | None:
    """Best-effort latest close via the existing multi-source downloader."""
    try:
        from stock_selector.data import download_prices_for_period_multi_source

        result = download_prices_for_period_multi_source([ticker], period=period)
        prices = result.prices
        if prices is None or prices.empty:
            return None
        rows = prices[prices["ticker"].astype(str).str.upper() == ticker]
        rows = rows if not rows.empty else prices
        rows = rows.sort_values("date")
        close = float(rows.iloc[-1]["adj_close"])
        return close if close > 0 else None
    except Exception:
        return None


def _prompt(message: str) -> str:
    try:
        return input(message).strip()
    except EOFError:
        return ""


def _prompt_float(message: str) -> float:
    raw = _prompt(message)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
