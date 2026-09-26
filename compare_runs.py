from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stock_selector.cli import compare_runs_command  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two validation run folders.")
    parser.add_argument("previous_run", help="Previous validation output folder.")
    parser.add_argument("current_run", help="Current validation output folder.")
    parser.add_argument(
        "--output-dir",
        help="Output folder for comparison files. Defaults to the current run folder.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return compare_runs_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
