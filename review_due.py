from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stock_selector.real_data import run_real_ticker_analysis  # noqa: E402
from stock_selector.signal_review import scan_signal_review_due_items  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan signal review history for due outcomes.")
    parser.add_argument(
        "--review-root",
        default="outputs/signal_review",
        help="Folder containing signal_history.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/signal_review",
        help="Output folder for due scan CSV, Markdown, and JSON.",
    )
    parser.add_argument(
        "--as-of-date",
        help="Optional scan date, YYYY-MM-DD. Defaults to today.",
    )
    parser.add_argument(
        "--refresh-due",
        action="store_true",
        help="Refresh tickers with due review items, then regenerate the due scan.",
    )
    parser.add_argument("--period", default="5y", help="Price history period for refresh runs.")
    parser.add_argument(
        "--include-snapshot",
        action="store_true",
        help="Fetch current company snapshot during refresh runs.",
    )
    parser.add_argument(
        "--include-peers",
        action="store_true",
        help="Run peer comparison during refresh runs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = scan_signal_review_due_items(
        review_root=args.review_root,
        output_dir=args.output_dir,
        as_of_date=args.as_of_date,
    )
    if args.refresh_due:
        due_tickers = _due_tickers(result)
        if due_tickers:
            print("Refreshing due tickers / 正在刷新已到期股票")
            for ticker in due_tickers:
                print(f"- {ticker}")
                run_real_ticker_analysis(
                    ticker=ticker,
                    period=args.period,
                    include_snapshot=args.include_snapshot,
                    include_peer_comparison=args.include_peers,
                )
            result = scan_signal_review_due_items(
                review_root=args.review_root,
                output_dir=args.output_dir,
                as_of_date=args.as_of_date,
            )
        else:
            print("No due tickers to refresh. / 没有需要刷新的到期股票。")
    summary = result.summary.iloc[0] if not result.summary.empty else None

    print()
    print("Signal review due scan completed. / 信号复盘到期扫描完成。")
    print(f"Review root / 复盘目录: {result.review_root}")
    print(f"Output folder / 输出文件夹: {result.output_dir}")
    if summary is not None:
        print(f"Signal count / 信号数量: {int(summary.signal_count)}")
        print(f"Due now / 已到复盘时间: {int(summary.due_now_count)}")
        print(f"Pending / 等待中: {int(summary.pending_count)}")
        print(f"Pending unknown / 等待信息不足: {int(summary.pending_unknown_count)}")
        if str(summary.next_due_date):
            print(f"Next due / 下一次预计复盘: {summary.next_due_date} ({summary.next_due_ticker})")

    due_now = (
        result.due_items[result.due_items["review_status"] == "due_now"]
        if not result.due_items.empty
        else result.due_items
    )
    if due_now.empty:
        print("No due review items now. / 当前没有已到期复盘项。")
    else:
        print()
        print("Due review items / 已到期复盘项")
        for row in due_now.head(20).itertuples(index=False):
            print(
                f"- {row.ticker} {row.review_window}: "
                f"signal_date={row.signal_date}, "
                f"estimated_review_date={row.estimated_review_date}, "
                f"report={row.report_path}"
            )

    print()
    print(f"Open report: {result.report_path}")
    return 0


def _due_tickers(result) -> list[str]:
    if result.due_items.empty:
        return []
    due_now = result.due_items[result.due_items["review_status"] == "due_now"]
    if due_now.empty:
        return []
    return sorted({str(ticker).upper().strip() for ticker in due_now["ticker"] if str(ticker).strip()})


if __name__ == "__main__":
    raise SystemExit(main())
