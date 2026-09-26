from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stock_selector.validation_cli import run_validation  # noqa: E402
from stock_selector.validation_presets import (  # noqa: E402
    apply_validation_preset,
    validation_preset_choices,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run walk-forward validation for ticker screening rules.")
    parser.add_argument("tickers", nargs="*", help="Ticker symbols, e.g. AAPL MSFT NVDA NOW.")
    parser.add_argument(
        "--universe",
        help=(
            "Built-in universe name, e.g. software, ai-infrastructure, cybersecurity, "
            "growth-core, balanced-core, research-core."
        ),
    )
    parser.add_argument(
        "--universe-file",
        help="CSV/TXT file containing tickers. CSV may use ticker or symbol column.",
    )
    parser.add_argument(
        "--historical-universe-file",
        help=(
            "CSV with point-in-time universe membership. Supports ticker,start_date,end_date,"
            "delisted_date,delisting_return or complete as_of_date snapshots."
        ),
    )
    parser.add_argument(
        "--all-universes",
        action="store_true",
        help="Validate all built-in universes and write an aggregate comparison report.",
    )
    parser.add_argument(
        "--preset",
        choices=validation_preset_choices(),
        help="Validation preset: quick, standard, or deep.",
    )
    parser.add_argument("--period", help="Price history period, e.g. 2y, 5y, 10y.")
    parser.add_argument("--step-days", type=int, help="Spacing between validation signal dates.")
    parser.add_argument(
        "--min-history-days",
        type=int,
        help="Minimum history before the first validation signal.",
    )
    parser.add_argument(
        "--output-dir",
        help="Output folder for validation CSV and Markdown reports.",
    )
    parser.add_argument(
        "--screening-config",
        help="Optional TOML file for high-probability screening thresholds.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = apply_validation_preset(parser.parse_args(argv))
    return run_validation(args, parser.error)


if __name__ == "__main__":
    raise SystemExit(main())
