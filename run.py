from __future__ import annotations

import argparse
import difflib
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stock_selector.console_report import print_ticker_report  # noqa: E402
from stock_selector.real_data import normalize_ticker, run_real_ticker_analysis  # noqa: E402
from stock_selector.screening_config import load_screening_config  # noqa: E402
from stock_selector.universe import BUILT_IN_UNIVERSES  # noqa: E402


PROJECT_COMMAND = f"cd {PROJECT_ROOT}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze one real stock ticker.")
    parser.add_argument("ticker", nargs="?", help="Ticker symbol, e.g. AAPL or NVDA.")
    parser.add_argument("--period", default="5y", help="yfinance period, e.g. 1y, 2y, 5y.")
    parser.add_argument(
        "--no-snapshot",
        action="store_true",
        help="Skip current company/analyst/news snapshot.",
    )
    parser.add_argument(
        "--no-peers",
        action="store_true",
        help="Skip automatic peer comparison.",
    )
    parser.add_argument(
        "--peer-limit",
        type=int,
        default=6,
        help="Maximum number of peer tickers to compare.",
    )
    parser.add_argument(
        "--screening-config",
        help="Optional TOML file for high-probability screening thresholds.",
    )
    parser.add_argument(
        "--probability-calibration",
        help="Optional probability_calibration.csv from walk-forward validation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    ticker = args.ticker or input("Enter ticker symbol: ").strip().upper()
    if not ticker:
        print(format_user_error(ticker, ValueError("Ticker cannot be empty.")), file=sys.stderr)
        return 2
    try:
        screening_config = load_screening_config(args.screening_config)
        result = run_real_ticker_analysis(
            ticker=ticker,
            period=args.period,
            include_snapshot=not args.no_snapshot,
            include_peer_comparison=not args.no_peers,
            peer_limit=max(args.peer_limit, 0),
            screening_config=screening_config,
            probability_calibration_path=args.probability_calibration,
        )
    except Exception as error:
        print(format_user_error(ticker, error), file=sys.stderr)
        return 2

    print_ticker_report(result)
    return 0


def format_user_error(ticker: str, error: Exception) -> str:
    normalized = normalize_ticker(ticker)
    error_message = str(error).strip() or error.__class__.__name__
    suggestions = _ticker_suggestions(normalized)
    example_ticker = suggestions[0] if suggestions else normalized or "AAPL"

    lines = [
        "Analysis did not run / 分析没有成功",
        "----------------------------------------",
        f"Ticker entered / 你输入的股票代码: {ticker or 'N/A'}",
        f"Normalized ticker / 系统识别为: {normalized or 'N/A'}",
        f"Reason / 原因: {error_message}",
        "",
        "What to do next / 下一步怎么做",
        f"1. Make sure VS Code opened the latest project / 确认VS Code打开新版项目: {PROJECT_COMMAND}",
        f"2. Run with python / 用python运行: python3 run.py {example_ticker}",
        "3. Do not type the ticker alone in Terminal / 不要在终端只输入股票代码本身。",
    ]

    if suggestions and normalized not in suggestions:
        lines.append(
            "Possible ticker match / 可能想输入的是: "
            + ", ".join(suggestions[:5])
        )
    if _looks_like_price_data_error(error_message):
        lines.extend(
            [
                "",
                "Data note / 数据说明",
                "The data provider returned no usable daily price rows.",
                "数据源没有返回可用的日线价格。",
                "Check whether the ticker needs a suffix, is delisted, is OTC, or is outside yfinance coverage.",
                "请检查它是否需要交易所后缀、已经退市、属于OTC，或不在yfinance覆盖范围内。",
            ]
        )
    return "\n".join(lines)


def _ticker_suggestions(ticker: str) -> list[str]:
    candidates = sorted(
        {
            symbol
            for tickers in BUILT_IN_UNIVERSES.values()
            for symbol in tickers
        }
        | {"AAPL", "MSFT", "NVDA", "AVGO", "AAON", "NOW", "CRM"}
    )
    if not ticker:
        return []
    if ticker in candidates:
        return [ticker]

    prioritized: list[str] = []
    for index in range(len(ticker)):
        candidate = normalize_ticker(ticker[:index] + ticker[index + 1 :])
        if candidate in candidates and candidate not in prioritized:
            prioritized.append(candidate)

    fuzzy = difflib.get_close_matches(ticker, candidates, n=5, cutoff=0.55)
    for candidate in fuzzy:
        if candidate not in prioritized:
            prioritized.append(candidate)
    return prioritized[:5]


def _looks_like_price_data_error(error_message: str) -> bool:
    lowered = error_message.lower()
    return any(
        phrase in lowered
        for phrase in [
            "no price data",
            "no usable ticker data",
            "no rows returned",
            "missing price rows",
            "possibly delisted",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
