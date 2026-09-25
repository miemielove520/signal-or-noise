from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stock_selector.scanner import run_high_probability_scan  # noqa: E402
from stock_selector.screening_config import load_screening_config  # noqa: E402
from stock_selector.universe import load_universe_tickers  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan multiple real tickers for high-probability setups.")
    parser.add_argument("tickers", nargs="*", help="Ticker symbols, e.g. AAPL MSFT NVDA NOW.")
    parser.add_argument(
        "--universe",
        help="Built-in universe name, e.g. software, ai-infrastructure, cybersecurity.",
    )
    parser.add_argument(
        "--universe-file",
        help="CSV/TXT file containing tickers. CSV may use ticker or symbol column.",
    )
    parser.add_argument("--period", default="5y", help="Price history period, e.g. 1y, 2y, 5y.")
    parser.add_argument(
        "--output-dir",
        default="outputs/scans/latest",
        help="Output folder for scan CSV and Markdown report.",
    )
    parser.add_argument(
        "--no-snapshot",
        action="store_true",
        help="Skip current company/analyst/news snapshot.",
    )
    parser.add_argument(
        "--journal",
        action="store_true",
        help="Write daily scan journal and compare against the previous snapshot.",
    )
    parser.add_argument(
        "--journal-root",
        default="outputs/journal",
        help="Folder for daily scan journal snapshots and report.",
    )
    parser.add_argument(
        "--screening-config",
        help="Optional TOML file for high-probability screening thresholds.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    tickers = load_universe_tickers(
        tickers=args.tickers,
        universe_name=args.universe,
        universe_file=args.universe_file,
    )
    screening_config = load_screening_config(args.screening_config)
    result = run_high_probability_scan(
        tickers=tickers,
        period=args.period,
        output_dir=args.output_dir,
        include_snapshot=not args.no_snapshot,
        write_journal=args.journal,
        journal_root=args.journal_root,
        screening_config=screening_config,
    )

    print()
    print("High probability scan completed. / 高概率扫描完成。")
    print(f"Tickers requested / 请求股票数: {len(result.tickers)}")
    print(f"Tickers analyzed / 完成分析数: {len(result.results)}")
    print(f"Output folder / 输出文件夹: {result.output_dir}")
    print(f"Top candidates / 最佳候选总结: {result.output_dir / 'top_candidates.md'}")
    if result.journal is not None:
        print(f"Journal report / 每日复盘: {result.journal.report_path}")
    print()

    if result.summary.empty:
        print("No tickers were analyzed successfully. / 没有股票成功完成分析。")
    else:
        passed = result.summary[result.summary["quality_gate_passed"] == True]  # noqa: E712
        near = result.summary[
            (result.summary["quality_gate_passed"] == False)  # noqa: E712
            & (result.summary["watchlist_status"] == "close_but_not_ready")
        ]
        early = result.summary[
            (result.summary["quality_gate_passed"] == False)  # noqa: E712
            & (result.summary["watchlist_status"] == "early_watch")
        ]
        filtered = result.summary[
            (result.summary["quality_gate_passed"] == False)  # noqa: E712
            & (~result.summary["watchlist_status"].isin(["close_but_not_ready", "early_watch"]))
        ]
        _print_group("High probability candidates / 高概率候选", passed)
        _print_group("Near watchlist / 接近机会", near)
        _print_group("Early watchlist / 早期观察", early)
        _print_group("Filtered out / 被过滤", filtered)
        _print_top_candidates(result.top_candidates)

    if result.failures:
        print()
        print("Failures / 失败项")
        for failure in result.failures:
            print(f"- {failure['ticker']}: {failure['error']}")

    print()
    print(f"Open report: {result.output_dir / 'high_probability_scan.md'}")
    return 1 if result.failures else 0


def _print_group(title: str, frame) -> None:
    print(title)
    if frame.empty:
        print("- None / 暂无")
        print()
        return
    for row in frame.itertuples(index=False):
        print(
            f"- {row.ticker}: score={row.high_probability_score:.2f}, "
            f"horizon={row.focus_horizon}, passed={row.quality_gate_passed}, "
            f"profile={row.screening_profile_zh}, "
            f"watchlist={row.watchlist_status_zh}, "
            f"missing={row.watchlist_missing_items_zh}"
        )
    print()


def _print_top_candidates(frame) -> None:
    print("Top candidates / 最佳候选总结")
    if frame.empty:
        print("- None / 暂无")
        print()
        return
    for row in frame.head(5).itertuples(index=False):
        print(
            f"- #{row.top_rank} {row.ticker}: "
            f"{row.candidate_category} / {row.candidate_category_zh}, "
            f"attention={row.attention_score:.2f}, "
            f"score={row.calibrated_high_probability_score:.2f}, "
            f"win_prob={row.calibrated_win_probability:.2%}, "
            f"blocker={row.primary_blocker_zh}"
        )
    print()


if __name__ == "__main__":
    raise SystemExit(main())
