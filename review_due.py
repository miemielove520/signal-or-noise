from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stock_selector.cli import review_due_command  # noqa: E402


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
    args = build_parser().parse_args(argv)
    return review_due_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
