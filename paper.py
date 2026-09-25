from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stock_selector.config import PaperTradingConfig  # noqa: E402
from stock_selector.json_io import dataframe_records, write_json  # noqa: E402
from stock_selector.paper import (  # noqa: E402
    PaperTargetConfig,
    build_paper_targets_from_analysis,
    load_portfolio_state,
    run_paper_rebalance,
)
from stock_selector.paper_audit import (  # noqa: E402
    new_audit_identity,
    record_rebalance_failure,
    record_rebalance_success,
    utc_timestamp,
)


DEFAULT_SCAN_CSV = PROJECT_ROOT / "outputs" / "scans" / "latest" / "high_probability_scan.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "paper_latest"
DEFAULT_HISTORY_DIR = PROJECT_ROOT / "outputs" / "paper_history"
DEFAULT_MARKET_REGIME_POLICY_CSV = (
    PROJECT_ROOT / "outputs" / "walk_forward" / "latest" / "market_regime_policy.csv"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate local paper-trading targets and simulated orders from model outputs."
    )
    parser.add_argument(
        "--scan-csv",
        default=str(DEFAULT_SCAN_CSV),
        help="Scan CSV to convert into paper targets.",
    )
    parser.add_argument(
        "--analysis-json",
        nargs="*",
        default=None,
        help="Optional single-ticker analysis_result.json files.",
    )
    parser.add_argument(
        "--analysis-csv",
        nargs="*",
        default=None,
        help="Optional ticker_analysis.csv files.",
    )
    parser.add_argument(
        "--prices-csv",
        help="Optional price CSV with date,ticker,adj_close columns.",
    )
    parser.add_argument(
        "--market-regime-policy",
        default=str(DEFAULT_MARKET_REGIME_POLICY_CSV),
        help=(
            "Optional market_regime_policy.csv from validation. If present, paper targets "
            "are scaled or blocked by market-regime protection rules."
        ),
    )
    parser.add_argument(
        "--ignore-market-regime-policy",
        action="store_true",
        help="Ignore market-regime protection policy and use normal paper target rules.",
    )
    parser.add_argument(
        "--state-csv",
        default=str(DEFAULT_OUTPUT_DIR / "paper_state.csv"),
        help="Paper portfolio state CSV. Created only when --update-state is used.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output folder for paper targets, orders, states, and report.",
    )
    parser.add_argument(
        "--history-dir",
        default=None,
        help=(
            "Append-only paper audit folder. Defaults to outputs/paper_history for the "
            "normal output folder, or <output-dir>/history for custom output folders."
        ),
    )
    parser.add_argument("--initial-cash", type=float, default=100_000.0)
    parser.add_argument("--min-trade-value", type=float, default=100.0)
    parser.add_argument("--trade-buffer-pct", type=float, default=0.001)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--commission-bps", type=float, default=0.0)
    parser.add_argument("--spread-bps", type=float, default=10.0)
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--max-position-weight", type=float, default=0.20)
    parser.add_argument("--cash-reserve-weight", type=float, default=0.10)
    parser.add_argument("--min-target-weight", type=float, default=0.02)
    parser.add_argument(
        "--allow-near-watchlist",
        action="store_true",
        help="Allow close-but-not-ready names into paper targets.",
    )
    parser.add_argument(
        "--whole-shares",
        action="store_true",
        help="Use whole-share simulated orders instead of fractional shares.",
    )
    parser.add_argument(
        "--update-state",
        action="store_true",
        help="Write post-trade state back to --state-csv. Default is dry run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_dir = (
        Path(args.history_dir)
        if args.history_dir
        else (
            DEFAULT_HISTORY_DIR
            if output_dir.resolve() == DEFAULT_OUTPUT_DIR.resolve()
            else output_dir / "history"
        )
    )
    run_id, started_at = new_audit_identity("rebalance")
    analysis: pd.DataFrame | None = None
    prices: pd.DataFrame | None = None
    targets: pd.DataFrame | None = None
    target_config: PaperTargetConfig | None = None
    trade_config: PaperTradingConfig | None = None
    market_regime_policy: pd.DataFrame | None = None
    source_snapshot = _source_snapshot(args, output_dir, history_dir)
    config_snapshot: dict[str, object] = {
        "cli_arguments": vars(args).copy(),
        "resolved_history_dir": str(history_dir),
    }

    try:
        analysis = _load_analysis_inputs(args)
        target_config = PaperTargetConfig(
            max_positions=args.max_positions,
            max_position_weight=args.max_position_weight,
            min_target_weight=args.min_target_weight,
            cash_reserve_weight=args.cash_reserve_weight,
            allow_near_watchlist=args.allow_near_watchlist,
        )
        config_snapshot["target_config"] = asdict(target_config)
        market_regime_policy = _load_market_regime_policy(args)
        targets = build_paper_targets_from_analysis(
            analysis,
            config=target_config,
            market_regime_policy=market_regime_policy,
        )
        state = load_portfolio_state(args.state_csv, initial_cash=args.initial_cash)
        prices = _load_prices(args, analysis, targets, state=state)
        trade_config = PaperTradingConfig(
            state_csv=Path(args.state_csv),
            initial_cash=args.initial_cash,
            min_trade_value=args.min_trade_value,
            trade_buffer_pct=args.trade_buffer_pct,
            slippage_bps=args.slippage_bps,
            commission_bps=args.commission_bps,
            spread_bps=args.spread_bps,
            allow_fractional_shares=not args.whole_shares,
        )
        config_snapshot["trade_config"] = asdict(trade_config)
        result = run_paper_rebalance(
            selections=targets,
            prices=prices,
            state=state,
            config=trade_config,
        )
    except Exception as error:
        finished_at = utc_timestamp()
        audit_path = _record_failed_run(
            history_dir=history_dir,
            run_id=run_id,
            started_at=started_at,
            finished_at=finished_at,
            args=args,
            error=error,
            config_snapshot=config_snapshot,
            source_snapshot=source_snapshot,
            analysis=analysis,
            prices=prices,
            targets=targets,
        )
        print(f"Paper trading simulation failed / 纸面模拟失败: {error}", file=sys.stderr)
        if audit_path is not None:
            print(f"Failure audit / 失败审计: {audit_path}", file=sys.stderr)
        return 2

    finished_at = utc_timestamp()
    audit_snapshot_dir = history_dir / "runs" / run_id
    result_payload = {
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "as_of_date": result.as_of_date,
        "paper_targets": dataframe_records(targets),
        "orders": dataframe_records(result.orders),
        "pre_trade_state": dataframe_records(result.pre_trade_state),
        "post_trade_state": dataframe_records(result.post_trade_state),
        "summary": result.summary,
        "state_updated": bool(args.update_state),
        "audit_snapshot_dir": str(audit_snapshot_dir),
        "market_regime_policy_path": None
        if args.ignore_market_regime_policy
        else str(args.market_regime_policy),
        "market_regime_policy_loaded": bool(
            market_regime_policy is not None and not market_regime_policy.empty
        ),
    }
    try:
        targets.to_csv(output_dir / "paper_targets.csv", index=False)
        prices.to_csv(output_dir / "prices_used.csv", index=False)
        result.orders.to_csv(output_dir / "orders.csv", index=False)
        result.pre_trade_state.to_csv(output_dir / "pre_trade_state.csv", index=False)
        result.post_trade_state.to_csv(output_dir / "post_trade_state.csv", index=False)
        (output_dir / "paper_trade_report.md").write_text(result.report, encoding="utf-8")
        write_json(output_dir / "paper_trade_result.json", result_payload)
        if args.update_state:
            result.post_trade_state.to_csv(args.state_csv, index=False)
        audit_path = record_rebalance_success(
            history_dir,
            run_id=run_id,
            started_at=started_at,
            finished_at=finished_at,
            state_updated=bool(args.update_state),
            analysis=analysis,
            prices=prices,
            targets=targets,
            orders=result.orders,
            pre_trade_state=result.pre_trade_state,
            post_trade_state=result.post_trade_state,
            result_payload=result_payload,
            report=result.report,
            config_snapshot=config_snapshot,
            source_snapshot=source_snapshot,
            project_root=PROJECT_ROOT,
        )
    except Exception as error:
        print(f"Paper audit persistence failed / 模拟盘审计保存失败: {error}", file=sys.stderr)
        return 2

    _print_summary(
        output_dir,
        targets,
        result,
        update_state=bool(args.update_state),
        audit_path=audit_path,
        history_dir=history_dir,
    )
    return 0


def _source_snapshot(
    args: argparse.Namespace,
    output_dir: Path,
    history_dir: Path,
) -> dict[str, object]:
    return {
        "scan_csv": str(args.scan_csv),
        "analysis_json": list(args.analysis_json or []),
        "analysis_csv": list(args.analysis_csv or []),
        "prices_csv": args.prices_csv,
        "state_csv": str(args.state_csv),
        "output_dir": str(output_dir),
        "history_dir": str(history_dir),
        "market_regime_policy": None
        if args.ignore_market_regime_policy
        else str(args.market_regime_policy),
    }


def _record_failed_run(
    *,
    history_dir: Path,
    run_id: str,
    started_at: str,
    finished_at: str,
    args: argparse.Namespace,
    error: Exception,
    config_snapshot: dict[str, object],
    source_snapshot: dict[str, object],
    analysis: pd.DataFrame | None,
    prices: pd.DataFrame | None,
    targets: pd.DataFrame | None,
) -> Path | None:
    try:
        return record_rebalance_failure(
            history_dir,
            run_id=run_id,
            started_at=started_at,
            finished_at=finished_at,
            state_updated=bool(args.update_state),
            error=str(error),
            config_snapshot=config_snapshot,
            source_snapshot=source_snapshot,
            project_root=PROJECT_ROOT,
            analysis=analysis,
            prices=prices,
            targets=targets,
        )
    except Exception as audit_error:
        print(
            f"Failure audit could not be saved / 失败审计无法保存: {audit_error}",
            file=sys.stderr,
        )
        return None


def _load_market_regime_policy(args: argparse.Namespace) -> pd.DataFrame | None:
    if args.ignore_market_regime_policy:
        return None
    path = Path(args.market_regime_policy)
    if not path.exists():
        return None
    policy = pd.read_csv(path)
    if policy.empty:
        return None
    return policy


def _load_analysis_inputs(args: argparse.Namespace) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if args.analysis_json:
        frames.extend(_read_analysis_json(Path(path)) for path in args.analysis_json)
    if args.analysis_csv:
        frames.extend(pd.read_csv(path) for path in args.analysis_csv)
    if not frames:
        scan_path = Path(args.scan_csv)
        if not scan_path.exists():
            raise FileNotFoundError(
                f"Scan CSV not found: {scan_path}. Run python3 scan.py first or pass --analysis-json."
            )
        frames.append(pd.read_csv(scan_path))
    frame = pd.concat(frames, ignore_index=True)
    if "ticker" not in frame.columns:
        raise ValueError("Analysis input must contain a ticker column.")
    return frame


def _read_analysis_json(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ["analysis_rows", "analysis", "summary"]:
            value = payload.get(key)
            if isinstance(value, list):
                return pd.DataFrame(value)
        row = {
            key: value
            for key, value in payload.items()
            if not isinstance(value, (list, dict))
        }
        if "ticker" in row:
            return pd.DataFrame([row])
    raise ValueError(f"Could not find analysis rows in {path}")


def _load_prices(
    args: argparse.Namespace,
    analysis: pd.DataFrame,
    targets: pd.DataFrame,
    state: pd.DataFrame | None = None,
    fetch_close_fn=None,
) -> pd.DataFrame:
    if args.prices_csv:
        prices = pd.read_csv(args.prices_csv)
        missing = {"date", "ticker", "adj_close"}.difference(prices.columns)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"Price CSV missing columns: {missing_text}")
        prices["source_date"] = prices["date"]
        prices["price_source"] = "prices_csv"
        return prices
    # Use the union of the full scan and targets so a holding that drops out of
    # today's target list can still be valued and sold.
    rows: list[dict[str, object]] = []
    for source_name, source in (("analysis_input", analysis), ("paper_target", targets)):
        if source is None or source.empty:
            continue
        for row in source.to_dict(orient="records"):
            ticker = str(row.get("ticker", "")).upper().strip()
            if not ticker:
                continue
            price = _first_numeric(
                row,
                ["latest_price", "current_price", "close", "adj_close", "watchlist_trigger_price"],
            )
            if price is None or price <= 0:
                continue
            date_value = _first_present(row, ["date", "analysis_date", "latest_date"])
            rows.append(
                {
                    "date": pd.Timestamp(date_value).normalize()
                    if date_value is not None
                    else pd.Timestamp.today().normalize(),
                    "ticker": ticker,
                    "adj_close": float(price),
                    "price_source": source_name,
                }
            )
    if not rows:
        raise ValueError("No usable price data found. Pass --prices-csv with date,ticker,adj_close.")
    prices = pd.DataFrame(rows)
    # Normalize each latest known quote to one valuation date while retaining its
    # original source date for audit purposes.
    latest_date = prices["date"].max()
    prices = (
        prices.sort_values(["ticker", "date"])
        .drop_duplicates(subset=["ticker"], keep="last")
        .assign(source_date=lambda frame: frame["date"], date=latest_date)
        .reset_index(drop=True)
    )
    return _augment_prices_with_held_tickers(prices, state, latest_date, fetch_close_fn)


def _augment_prices_with_held_tickers(
    prices: pd.DataFrame,
    state: pd.DataFrame | None,
    price_date: pd.Timestamp,
    fetch_close_fn=None,
) -> pd.DataFrame:
    """Fetch a current close for held tickers absent from the scan price table."""
    if state is None or state.empty:
        return prices
    held = {str(t).upper().strip() for t in state["ticker"].tolist()} - {"CASH"}
    priced = set(prices["ticker"].astype(str).str.upper())
    missing = sorted(held - priced)
    if not missing:
        return prices
    if fetch_close_fn is None:
        def fetch_close_fn(ticker: str) -> float | None:  # pragma: no cover - network
            from stock_selector.data import download_prices_for_period_multi_source

            result = download_prices_for_period_multi_source([ticker], period="5d")
            frame = result.prices
            if frame is None or frame.empty:
                return None
            return float(frame.sort_values("date")["adj_close"].iloc[-1])
    extra: list[dict[str, object]] = []
    for ticker in missing:
        close = None
        try:
            close = fetch_close_fn(ticker)
        except Exception as error:
            print(f"- {ticker}: live price fetch failed ({error})", file=sys.stderr)
        if close is not None and close > 0:
            extra.append(
                {
                    "date": price_date,
                    "source_date": price_date,
                    "ticker": ticker,
                    "adj_close": float(close),
                    "price_source": "held_ticker_live_fetch",
                }
            )
            print(
                f"- {ticker}: held but absent from today's scan; using live close for pricing"
                f" / 持仓已不在当日扫描结果中，已补实时收盘价参与估值"
            )
        else:
            print(
                f"- {ticker}: held but NO price available — cannot be valued or sold this run"
                f" / 持仓无法定价，本次无法估值或卖出，请检查数据源",
                file=sys.stderr,
            )
    if extra:
        prices = pd.concat([prices, pd.DataFrame(extra)], ignore_index=True)
    return prices


def _first_numeric(row: dict[str, object], columns: list[str]) -> float | None:
    for column in columns:
        value = row.get(column)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if pd.notna(numeric):
            return numeric
    return None


def _first_present(row: dict[str, object], columns: list[str]) -> object | None:
    for column in columns:
        value = row.get(column)
        try:
            missing = pd.isna(value)
        except (TypeError, ValueError):
            missing = False
        if value is not None and not bool(missing):
            return value
    return None


def _print_summary(
    output_dir: Path,
    targets: pd.DataFrame,
    result,
    update_state: bool,
    audit_path: Path,
    history_dir: Path,
) -> None:
    print()
    print("Paper trading simulation completed. / 纸面模拟交易完成。")
    print("This did not send live broker orders. / 这一步没有发送真实券商订单。")
    print(f"Output folder / 输出文件夹: {output_dir}")
    print(f"As of date / 模拟日期: {result.as_of_date.date().isoformat()}")
    print(f"Targets / 模拟目标数: {len(targets)}")
    print(f"Orders / 模拟订单数: {len(result.orders)}")
    print(f"State updated / 是否更新模拟仓位: {update_state}")
    print()
    print("Summary / 摘要")
    for key, value in result.summary.items():
        if isinstance(value, float):
            print(f"- {key}: {value:.4f}")
        else:
            print(f"- {key}: {value}")
    print()
    print(f"Open report: {output_dir / 'paper_trade_report.md'}")
    print(f"Immutable audit snapshot / 不可覆盖审计快照: {audit_path}")
    print(f"Audit journal / 完整流水总表: {history_dir / 'paper_journal.md'}")


if __name__ == "__main__":
    raise SystemExit(main())
