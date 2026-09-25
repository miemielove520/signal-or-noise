from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stock_selector.json_io import write_json  # noqa: E402
from stock_selector.paper import load_portfolio_state  # noqa: E402
from stock_selector.paper_audit import (  # noqa: E402
    new_audit_identity,
    record_valuation,
)
from stock_selector.paper_tracker import (  # noqa: E402
    CASH_TICKER,
    EQUITY_HISTORY_COLUMNS,
    compare_paper_to_backtest,
    expected_annual_from_walk_forward,
    mark_to_market,
    record_equity_point,
    summarize_paper_performance,
)

DEFAULT_WALK_FORWARD_SUMMARY = (
    PROJECT_ROOT / "outputs" / "walk_forward" / "latest" / "walk_forward_summary.csv"
)


DEFAULT_STATE = PROJECT_ROOT / "outputs" / "paper_latest" / "paper_state.csv"
DEFAULT_HISTORY = PROJECT_ROOT / "outputs" / "paper_latest" / "paper_equity_history.csv"
DEFAULT_PAPER_RESULT = PROJECT_ROOT / "outputs" / "paper_latest" / "paper_trade_result.json"
DEFAULT_HISTORY_DIR = PROJECT_ROOT / "outputs" / "paper_history"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mark the paper portfolio to market, record its equity curve, and "
        "report readiness. 给模拟组合盯市、记录净值曲线、给出是否可进下一步的判断。",
    )
    parser.add_argument("--state-csv", default=str(DEFAULT_STATE))
    parser.add_argument("--history-csv", default=str(DEFAULT_HISTORY))
    parser.add_argument("--audit-history-dir", default=str(DEFAULT_HISTORY_DIR))
    parser.add_argument("--benchmark", default="QQQ")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    valuation_id, executed_at = new_audit_identity("valuation")
    today = datetime.now().date().isoformat()
    audit_history_dir = Path(args.audit_history_dir)
    state_path = Path(args.state_csv)
    if not state_path.exists():
        print(
            "No paper portfolio yet. Initialize it first with:\n"
            "  python3 paper.py --update-state\n"
            "还没有模拟组合。先运行 python3 paper.py --update-state 初始化。"
        )
        try:
            record_valuation(
                audit_history_dir,
                valuation_id=valuation_id,
                executed_at=executed_at,
                date=today,
                status="skipped",
                state=pd.DataFrame(columns=["ticker", "quantity"]),
                current_prices={},
                fallback_prices={},
                benchmark_ticker=args.benchmark.upper(),
                benchmark_close=None,
                warnings=[],
                error="Paper state does not exist.",
            )
        except Exception as audit_error:
            print(f"Paper audit failed / 模拟盘审计失败: {audit_error}", file=sys.stderr)
            return 2
        return 0

    state = load_portfolio_state(state_path, initial_cash=1_000.0)
    tickers = [
        str(t).upper().strip()
        for t in state["ticker"]
        if str(t).upper().strip() and str(t).upper().strip() != CASH_TICKER
    ]
    price_lookup = _latest_prices(tickers)
    fallback_prices = _last_known_prices(DEFAULT_PAPER_RESULT)
    valuation_warnings = [
        f"{ticker}: current price unavailable; used last known paper price"
        for ticker in tickers
        if ticker not in price_lookup and ticker in fallback_prices
    ]
    for warning in valuation_warnings:
        print(f"Valuation warning / 估值警告: {warning}", file=sys.stderr)
    benchmark_close = _latest_prices([args.benchmark]).get(args.benchmark.upper())

    try:
        equity = mark_to_market(state, price_lookup, fallback_price_lookup=fallback_prices)
    except ValueError as error:
        try:
            record_valuation(
                audit_history_dir,
                valuation_id=valuation_id,
                executed_at=executed_at,
                date=today,
                status="failed",
                state=state,
                current_prices=price_lookup,
                fallback_prices=fallback_prices,
                benchmark_ticker=args.benchmark.upper(),
                benchmark_close=benchmark_close,
                warnings=valuation_warnings,
                error=str(error),
            )
        except Exception as audit_error:
            print(f"Paper audit failed / 模拟盘审计失败: {audit_error}", file=sys.stderr)
        print(f"Paper valuation failed / 模拟盘估值失败: {error}", file=sys.stderr)
        return 2
    history = _load_history(Path(args.history_csv))
    history = record_equity_point(history, today, equity, benchmark_close)
    Path(args.history_csv).parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(args.history_csv, index=False)

    perf = summarize_paper_performance(history)

    # Gate: compare the forward paper run to what the backtest predicted.
    expected_annual = None
    wf_path = Path(DEFAULT_WALK_FORWARD_SUMMARY)
    if wf_path.exists():
        try:
            expected_annual = expected_annual_from_walk_forward(pd.read_csv(wf_path))
        except Exception:
            expected_annual = None
    backtest_gate = compare_paper_to_backtest(perf, expected_annual)

    payload = {
        **perf.to_dict(),
        "backtest_gate": backtest_gate,
        "valuation_warnings": valuation_warnings,
    }
    write_json(Path(args.history_csv).with_name("paper_performance.json"), payload)
    try:
        audit_path = record_valuation(
            audit_history_dir,
            valuation_id=valuation_id,
            executed_at=executed_at,
            date=today,
            status="success",
            state=state,
            current_prices=price_lookup,
            fallback_prices=fallback_prices,
            benchmark_ticker=args.benchmark.upper(),
            benchmark_close=benchmark_close,
            warnings=valuation_warnings,
            performance=perf.to_dict(),
            backtest_gate=backtest_gate,
            equity=equity,
        )
    except Exception as audit_error:
        print(f"Paper audit failed / 模拟盘审计失败: {audit_error}", file=sys.stderr)
        return 2
    _print(perf, equity)
    print(f"Backtest gate / 回测对比闸门: {backtest_gate['gate_zh']} — {backtest_gate['note_zh']}")
    print(f"Valuation audit / 估值审计快照: {audit_path}")
    print(f"Audit journal / 完整流水总表: {audit_history_dir / 'paper_journal.md'}")
    return 0


def _latest_prices(tickers: list[str]) -> dict[str, float]:
    if not tickers:
        return {}
    from stock_selector.data import download_prices_for_period_multi_source

    prices: dict[str, float] = {}
    for ticker in tickers:
        try:
            result = download_prices_for_period_multi_source([ticker], period="5d")
            frame = result.prices
            if not frame.empty:
                prices[ticker.upper()] = float(frame.sort_values("date")["adj_close"].iloc[-1])
        except Exception as error:
            print(f"- {ticker}: price fetch failed ({error})", file=sys.stderr)
    return prices


def _last_known_prices(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    prices: dict[str, float] = {}
    for row in payload.get("paper_targets") or []:
        _add_known_price(prices, row, ("latest_price", "current_price"))
    for row in payload.get("orders") or []:
        _add_known_price(prices, row, ("close_price", "price"))
    return prices


def _add_known_price(
    prices: dict[str, float],
    row: object,
    columns: tuple[str, ...],
) -> None:
    if not isinstance(row, dict):
        return
    ticker = str(row.get("ticker") or "").upper().strip()
    if not ticker:
        return
    for column in columns:
        try:
            value = float(row.get(column))
        except (TypeError, ValueError):
            continue
        if value > 0 and value == value:
            prices[ticker] = value
            return


def _load_history(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    return pd.DataFrame(columns=list(EQUITY_HISTORY_COLUMNS))


def _print(perf, equity: float) -> None:
    print(f"Paper equity / 模拟盘净值: ${equity:,.2f}")
    print(f"Days tracked / 已追踪天数: {perf.days_tracked}")
    if perf.total_return is not None:
        print(f"Return / 收益: {perf.total_return:+.2%} | benchmark / 基准: {_pct(perf.benchmark_return)} "
              f"| excess / 超额: {_pct(perf.excess_return)} | maxDD / 最大回撤: {_pct(perf.max_drawdown)}")
    print(f"Readiness / 是否可进下一步: {perf.readiness} / {perf.readiness_zh}")
    print(f"  {perf.note_zh}")


def _pct(value: object) -> str:
    try:
        return f"{float(value):+.2%}"
    except (TypeError, ValueError):
        return "N/A"


if __name__ == "__main__":
    raise SystemExit(main())
