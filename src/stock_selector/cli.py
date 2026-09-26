from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NoReturn

from .analysis import analyze_ticker, render_ticker_analysis
from .audit import audit_price_csv
from .config import load_config
from .console_report import print_ticker_report
from .data import download_yfinance_prices
from .monitor import build_daily_monitor, feature_drift_report, feature_missing_report
from .paper import load_portfolio_state, run_paper_rebalance
from .pipeline import run_ml_pipeline, run_research_pipeline
from .real_data import run_real_ticker_analysis
from .report import render_research_report
from .run_comparison import compare_validation_runs
from .scanner import run_high_probability_scan
from .screening_config import load_screening_config
from .signal_review import scan_signal_review_due_items
from .snapshot import fetch_yfinance_snapshot
from .universe import load_universe_tickers
from .validation_cli import run_validation
from .validation_presets import apply_validation_preset, validation_preset_choices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run stock selection research pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a configured research backtest.")
    run_parser.add_argument("--config", required=True, help="Path to TOML config.")
    run_parser.add_argument(
        "--output-dir",
        default="outputs/latest",
        help="Directory for selections and equity curve CSV files.",
    )

    ml_parser = subparsers.add_parser(
        "ml-run",
        help="Run rolling out-of-sample ML prediction and backtest.",
    )
    ml_parser.add_argument("--config", required=True, help="Path to TOML config.")
    ml_parser.add_argument(
        "--output-dir",
        default="outputs/ml_latest",
        help="Directory for ML predictions, feature importance, and backtest outputs.",
    )

    paper_parser = subparsers.add_parser(
        "paper-trade",
        help="Generate local paper-trading rebalance orders from target weights.",
    )
    paper_parser.add_argument("--config", required=True, help="Path to TOML config.")
    paper_parser.add_argument(
        "--mode",
        choices=["factor", "ml"],
        default="factor",
        help="Use factor scores or ML predictions as the target portfolio.",
    )
    paper_parser.add_argument(
        "--output-dir",
        default="outputs/paper_latest",
        help="Directory for paper orders and portfolio state outputs.",
    )
    paper_parser.add_argument(
        "--state-csv",
        help="Override portfolio state CSV. Uses [paper].state_csv by default.",
    )
    paper_parser.add_argument(
        "--as-of-date",
        help="Optional latest price date cutoff, YYYY-MM-DD.",
    )
    paper_parser.add_argument(
        "--update-state",
        action="store_true",
        help="Write the post-trade state back to the configured state CSV.",
    )

    daily_parser = subparsers.add_parser(
        "daily-report",
        help="Generate latest candidate list and monitoring report.",
    )
    daily_parser.add_argument("--config", required=True, help="Path to TOML config.")
    daily_parser.add_argument(
        "--mode",
        choices=["factor", "ml"],
        default="factor",
        help="Use factor scores or ML predictions for the candidate list.",
    )
    daily_parser.add_argument(
        "--output-dir",
        default="outputs/daily_latest",
        help="Directory for daily candidates and monitoring outputs.",
    )
    daily_parser.add_argument(
        "--as-of-date",
        help="Optional monitor date, YYYY-MM-DD. Defaults to today.",
    )

    analyze_parser = subparsers.add_parser(
        "analyze-ticker",
        help="Generate short, medium, or long horizon trade plans for one ticker.",
    )
    analyze_parser.add_argument("--config", required=True, help="Path to TOML config.")
    analyze_parser.add_argument("--ticker", required=True, help="Ticker symbol to analyze.")
    analyze_parser.add_argument(
        "--horizon",
        nargs="+",
        choices=["short", "medium", "long", "all"],
        default=["all"],
        help="One or more horizons to analyze. Defaults to all.",
    )
    analyze_parser.add_argument(
        "--entry-buffer-pct",
        type=float,
        default=0.003,
        help="Breakout/support buffer used for entry and stop calculations.",
    )
    analyze_parser.add_argument(
        "--as-of-date",
        help="Optional analysis date cutoff, YYYY-MM-DD.",
    )
    analyze_parser.add_argument(
        "--output-dir",
        default="outputs/ticker_latest",
        help="Directory for ticker analysis CSV and Markdown report.",
    )
    analyze_parser.add_argument(
        "--include-external-snapshot",
        action="store_true",
        help=(
            "Fetch current yfinance company/analyst/news snapshot and save it separately. "
            "The snapshot is not used in historical backtests."
        ),
    )

    real_parser = subparsers.add_parser(
        "real",
        help="Download real yfinance data and analyze one ticker.",
    )
    real_parser.add_argument("ticker", nargs="?", help="Ticker symbol, e.g. AAPL or NVDA.")
    real_parser.add_argument("--period", default="5y", help="yfinance period, e.g. 1y, 2y, 5y.")
    real_parser.add_argument(
        "--horizon",
        nargs="+",
        choices=["short", "medium", "long", "all"],
        default=["all"],
        help="One or more horizons to analyze.",
    )
    real_parser.add_argument(
        "--no-snapshot",
        action="store_true",
        help="Skip the current external company/analyst/news snapshot.",
    )
    real_parser.add_argument(
        "--no-peers",
        action="store_true",
        help="Skip automatic peer comparison.",
    )
    real_parser.add_argument(
        "--peer-limit",
        type=int,
        default=6,
        help="Maximum number of peer tickers to compare.",
    )
    real_parser.add_argument(
        "--output-root",
        default="outputs/real_ticker",
        help="Root directory for real ticker analysis outputs.",
    )
    real_parser.add_argument(
        "--screening-config",
        help="Optional TOML file for high-probability screening thresholds.",
    )
    real_parser.add_argument(
        "--probability-calibration",
        help="Optional probability_calibration.csv from walk-forward validation.",
    )

    scan_parser = subparsers.add_parser(
        "scan",
        help="Scan multiple real tickers for high-probability setups.",
    )
    scan_parser.add_argument("tickers", nargs="*", help="Ticker symbols to scan.")
    scan_parser.add_argument(
        "--universe",
        help=(
            "Built-in universe name, e.g. software, ai-infrastructure, cybersecurity, "
            "growth-core, balanced-core, research-core."
        ),
    )
    scan_parser.add_argument("--universe-file", help="CSV/TXT file containing tickers.")
    scan_parser.add_argument("--period", default="5y", help="Price history period.")
    scan_parser.add_argument(
        "--output-dir",
        default="outputs/scans/latest",
        help="Output directory for scan CSV and Markdown report.",
    )
    scan_parser.add_argument(
        "--no-snapshot",
        action="store_true",
        help="Skip current company/analyst/news snapshot.",
    )
    scan_parser.add_argument("--journal", action="store_true", help="Write daily scan journal.")
    scan_parser.add_argument(
        "--journal-root",
        default="outputs/journal",
        help="Folder for journal snapshots and report.",
    )
    scan_parser.add_argument(
        "--screening-config",
        help="Optional TOML file for high-probability screening thresholds.",
    )

    validate_parser = subparsers.add_parser(
        "validate",
        help="Run walk-forward validation and rule calibration.",
    )
    validate_parser.add_argument("tickers", nargs="*", help="Ticker symbols to validate.")
    validate_parser.add_argument(
        "--universe",
        help=(
            "Built-in universe name, e.g. software, ai-infrastructure, cybersecurity, "
            "growth-core, balanced-core, research-core."
        ),
    )
    validate_parser.add_argument("--universe-file", help="CSV/TXT file containing tickers.")
    validate_parser.add_argument(
        "--historical-universe-file",
        help=(
            "CSV with point-in-time universe membership. Supports ticker,start_date,end_date,"
            "delisted_date,delisting_return or complete as_of_date snapshots."
        ),
    )
    validate_parser.add_argument(
        "--all-universes",
        action="store_true",
        help="Validate all built-in universes and write an aggregate comparison report.",
    )
    validate_parser.add_argument(
        "--preset",
        choices=validation_preset_choices(),
        help="Validation preset: quick, standard, or deep.",
    )
    validate_parser.add_argument("--period", help="Price history period.")
    validate_parser.add_argument("--step-days", type=int)
    validate_parser.add_argument("--min-history-days", type=int)
    validate_parser.add_argument(
        "--output-dir",
        help="Output directory for validation reports.",
    )
    validate_parser.add_argument(
        "--screening-config",
        help="Optional TOML file for high-probability screening thresholds.",
    )

    compare_parser = subparsers.add_parser(
        "compare-runs",
        help="Compare two walk-forward validation run folders.",
    )
    compare_parser.add_argument("previous_run", help="Previous validation output folder.")
    compare_parser.add_argument("current_run", help="Current validation output folder.")
    compare_parser.add_argument(
        "--output-dir",
        help="Output folder for comparison files. Defaults to the current run folder.",
    )

    audit_parser = subparsers.add_parser("audit", help="Audit local OHLCV price data.")
    audit_parser.add_argument("--config", help="Path to TOML config.")
    audit_parser.add_argument("--prices-csv", help="Path to a price CSV file.")
    audit_parser.add_argument(
        "--issues-csv",
        help="Optional path to write audit issues as CSV.",
    )

    download_parser = subparsers.add_parser(
        "download-yfinance",
        help="Download daily OHLCV prices with yfinance for research prototyping.",
    )
    download_parser.add_argument("--tickers", nargs="+", required=True)
    download_parser.add_argument("--start", required=True, help="Start date, YYYY-MM-DD.")
    download_parser.add_argument("--end", help="Optional end date, YYYY-MM-DD.")
    download_parser.add_argument("--output", required=True, help="Output CSV path.")

    review_due_parser = subparsers.add_parser(
        "review-due",
        help="Scan signal review history for due and pending outcome windows.",
    )
    review_due_parser.add_argument(
        "--review-root",
        default="outputs/signal_review",
        help="Folder containing signal_history.csv.",
    )
    review_due_parser.add_argument(
        "--output-dir",
        default="outputs/signal_review",
        help="Output folder for due scan CSV, Markdown, and JSON.",
    )
    review_due_parser.add_argument(
        "--as-of-date",
        help="Optional scan date, YYYY-MM-DD. Defaults to today.",
    )
    review_due_parser.add_argument(
        "--refresh-due",
        action="store_true",
        help="Refresh tickers with due review items, then regenerate the due scan.",
    )
    review_due_parser.add_argument("--period", default="5y", help="Price history period for refresh runs.")
    review_due_parser.add_argument(
        "--include-snapshot",
        action="store_true",
        help="Fetch current company snapshot during refresh runs.",
    )
    review_due_parser.add_argument(
        "--include-peers",
        action="store_true",
        help="Run peer comparison during refresh runs.",
    )
    return parser


def run_command(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    result = run_research_pipeline(config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.selections.to_csv(output_dir / "selections.csv", index=False)
    result.risk_report.to_csv(output_dir / "risk_report.csv", index=False)
    result.exposure_report.to_csv(output_dir / "exposure_report.csv", index=False)
    result.equity_curve.to_csv(output_dir / "equity_curve.csv", index=False)
    (output_dir / "summary_report.md").write_text(
        render_research_report(
            title="Stock Selection Research Report",
            selections=result.selections,
            metrics=result.metrics,
            risk_report=result.risk_report,
            exposure_report=result.exposure_report,
        ),
        encoding="utf-8",
    )

    print("Backtest metrics")
    for key, value in result.metrics.items():
        print(f"{key}: {value:.4f}")

    latest_date = result.selections["date"].max() if not result.selections.empty else None
    if latest_date is not None:
        latest = result.selections[result.selections["date"] == latest_date]
        tickers = ", ".join(latest["ticker"].to_list())
        print(f"Latest selection date: {latest_date.date()}")
        print(f"Selected tickers: {tickers}")
    else:
        print("No tickers passed the universe and factor filters.")

    if not result.risk_report.empty:
        latest_risk = result.risk_report.iloc[-1]
        print(
            "Latest risk: "
            f"gross_exposure={latest_risk.gross_exposure:.3f}, "
            f"estimated_annual_volatility={latest_risk.estimated_annual_volatility:.3f}, "
            f"cash_weight={latest_risk.cash_weight:.3f}"
        )

    print(f"Wrote outputs to {output_dir}")
    return 0


def audit_command(args: argparse.Namespace) -> int:
    if not args.config and not args.prices_csv:
        raise SystemExit("Provide either --config or --prices-csv.")

    if args.prices_csv:
        prices_csv = Path(args.prices_csv)
    else:
        prices_csv = load_config(args.config).data.prices_csv

    report = audit_price_csv(prices_csv)
    print(report.to_text())

    if args.issues_csv:
        issues_path = Path(args.issues_csv)
        issues_path.parent.mkdir(parents=True, exist_ok=True)
        report.issues_frame().to_csv(issues_path, index=False)
        print(f"Wrote audit issues to {issues_path}")

    return 0 if report.passed else 1


def ml_run_command(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    result = run_ml_pipeline(config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.ml.dataset.to_csv(output_dir / "ml_dataset.csv", index=False)
    result.ml.predictions.to_csv(output_dir / "ml_predictions.csv", index=False)
    result.ml.feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)
    result.selections.to_csv(output_dir / "ml_selections.csv", index=False)
    result.risk_report.to_csv(output_dir / "ml_risk_report.csv", index=False)
    result.exposure_report.to_csv(output_dir / "ml_exposure_report.csv", index=False)
    result.equity_curve.to_csv(output_dir / "ml_equity_curve.csv", index=False)
    (output_dir / "ml_summary_report.md").write_text(
        render_research_report(
            title="ML Stock Selection Research Report",
            selections=result.selections,
            metrics=result.backtest_metrics,
            risk_report=result.risk_report,
            exposure_report=result.exposure_report,
            prediction_metrics=result.ml.metrics,
            feature_importance=result.ml.feature_importance,
        ),
        encoding="utf-8",
    )

    print("ML prediction metrics")
    for key, value in result.ml.metrics.items():
        print(f"{key}: {value:.4f}")

    print("ML backtest metrics")
    for key, value in result.backtest_metrics.items():
        print(f"{key}: {value:.4f}")

    if not result.ml.feature_importance.empty:
        top_features = result.ml.feature_importance.head(5)
        rendered = ", ".join(
            f"{row.feature}={row.importance:.3f}" for row in top_features.itertuples()
        )
        print(f"Top features: {rendered}")

    latest_date = result.selections["date"].max() if not result.selections.empty else None
    if latest_date is not None:
        latest = result.selections[result.selections["date"] == latest_date]
        tickers = ", ".join(latest["ticker"].to_list())
        print(f"Latest ML selection date: {latest_date.date()}")
        print(f"Selected tickers: {tickers}")
    else:
        print("No ML selections were produced. Check min_train_rows and feature coverage.")

    if not result.risk_report.empty:
        latest_risk = result.risk_report.iloc[-1]
        print(
            "Latest ML risk: "
            f"gross_exposure={latest_risk.gross_exposure:.3f}, "
            f"estimated_annual_volatility={latest_risk.estimated_annual_volatility:.3f}, "
            f"cash_weight={latest_risk.cash_weight:.3f}"
        )

    print(f"Wrote outputs to {output_dir}")
    return 0


def paper_trade_command(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    pipeline_result = run_ml_pipeline(config) if args.mode == "ml" else run_research_pipeline(config)
    state_path = Path(args.state_csv) if args.state_csv else config.paper.state_csv
    state = load_portfolio_state(state_path, initial_cash=config.paper.initial_cash)
    paper_result = run_paper_rebalance(
        selections=pipeline_result.selections,
        prices=pipeline_result.prices,
        state=state,
        config=config.paper,
        as_of_date=args.as_of_date,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paper_result.pre_trade_state.to_csv(output_dir / "pre_trade_state.csv", index=False)
    paper_result.orders.to_csv(output_dir / "orders.csv", index=False)
    paper_result.post_trade_state.to_csv(output_dir / "post_trade_state.csv", index=False)
    (output_dir / "paper_trade_report.md").write_text(
        paper_result.report,
        encoding="utf-8",
    )

    if args.update_state:
        if state_path is None:
            raise SystemExit("Cannot update state because no state path is configured.")
        state_path.parent.mkdir(parents=True, exist_ok=True)
        paper_result.post_trade_state.to_csv(state_path, index=False)
        print(f"Updated paper state at {state_path}")

    print("Paper trade summary")
    for key, value in paper_result.summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print(f"Orders: {len(paper_result.orders)}")
    print(f"Wrote outputs to {output_dir}")
    return 0


def daily_report_command(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.mode == "ml":
        result = run_ml_pipeline(config)
        monitor = build_daily_monitor(
            prices=result.prices,
            scored=result.scored,
            selections=result.selections,
            risk_report=result.risk_report,
            exposure_report=result.exposure_report,
            feature_columns=config.ml.feature_columns,
            mode="ml",
            as_of_date=args.as_of_date,
            prediction_metrics=result.ml.metrics,
            monitor_config=config.monitor,
        )
    else:
        result = run_research_pipeline(config)
        monitor = build_daily_monitor(
            prices=result.prices,
            scored=result.scored,
            selections=result.selections,
            risk_report=result.risk_report,
            exposure_report=result.exposure_report,
            feature_columns=tuple(config.scoring.factor_weights.keys()),
            mode="factor",
            as_of_date=args.as_of_date,
            monitor_config=config.monitor,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    monitor.candidates.to_csv(output_dir / "daily_candidates.csv", index=False)
    monitor.checks.to_csv(output_dir / "monitor_checks.csv", index=False)
    feature_columns = (
        config.ml.feature_columns
        if args.mode == "ml"
        else tuple(config.scoring.factor_weights.keys())
    )
    feature_missing_report(result.scored, feature_columns).to_csv(
        output_dir / "feature_missing_report.csv",
        index=False,
    )
    feature_drift_report(
        result.scored,
        feature_columns,
        lookback_days=config.monitor.drift_lookback_days,
    ).to_csv(output_dir / "feature_drift_report.csv", index=False)
    (output_dir / "daily_report.md").write_text(monitor.report, encoding="utf-8")

    warning_count = int((monitor.checks["status"] == "warning").sum())
    print("Daily monitor summary")
    print(f"Candidates: {len(monitor.candidates)}")
    print(f"Warnings: {warning_count}")
    print(f"Wrote outputs to {output_dir}")
    return 0


def analyze_ticker_command(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    pipeline_result = run_research_pipeline(config)
    analysis = analyze_ticker(
        scored=pipeline_result.scored,
        prices=pipeline_result.prices,
        ticker=args.ticker,
        horizons=args.horizon,
        as_of_date=args.as_of_date,
        entry_buffer_pct=args.entry_buffer_pct,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis.to_csv(output_dir / "ticker_analysis.csv", index=False)
    report = render_ticker_analysis(analysis)
    (output_dir / "ticker_analysis.md").write_text(report, encoding="utf-8")
    if args.include_external_snapshot:
        snapshot = fetch_yfinance_snapshot(args.ticker)
        (output_dir / "external_snapshot.json").write_text(
            json.dumps(snapshot, indent=2),
            encoding="utf-8",
        )

    print("Ticker analysis summary")
    for row in analysis.itertuples(index=False):
        print(
            f"{row.ticker} {row.horizon}: "
            f"action={row.action}, "
            f"score={row.signal_score:.2f}, "
            f"entry={row.entry_price:.2f}, "
            f"stop={row.stop_loss:.2f}, "
            f"target={row.take_profit:.2f}, "
            f"risk_reward={row.risk_reward:.2f}"
        )
    if args.include_external_snapshot:
        print("External current snapshot saved separately and was not used in the backtest.")
    print(f"Wrote outputs to {output_dir}")
    return 0


def real_ticker_command(args: argparse.Namespace) -> int:
    ticker = args.ticker or input("Enter ticker symbol: ").strip().upper()
    screening_config = load_screening_config(args.screening_config)
    result = run_real_ticker_analysis(
        ticker=ticker,
        period=args.period,
        horizons=tuple(args.horizon),
        output_root=args.output_root,
        include_snapshot=not args.no_snapshot,
        include_peer_comparison=not args.no_peers,
        peer_limit=max(args.peer_limit, 0),
        screening_config=screening_config,
        probability_calibration_path=args.probability_calibration,
    )

    print_ticker_report(result)
    return 0


def scan_command(args: argparse.Namespace) -> int:
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
    print("High probability scan summary / 高概率扫描摘要")
    print(f"Tickers requested / 请求股票数: {len(result.tickers)}")
    print(f"Tickers analyzed / 完成分析数: {len(result.results)}")
    if result.summary.empty:
        print("No tickers were analyzed successfully. / 没有股票成功完成分析。")
    else:
        for row in result.summary.head(20).itertuples(index=False):
            print(
                f"{row.scan_rank}. {row.ticker}: "
                f"final={row.final_decision} / {row.final_decision_zh}, "
                f"final_score={row.final_score:.2f}, "
                f"period={row.requested_period}->{row.analysis_period}, "
                f"data_layer={row.data_quality_weakest_layer} / {row.data_quality_weakest_layer_zh}, "
                f"price_health={row.price_health_score:.2f} / {row.price_health_level_zh}, "
                f"backtest_trust={row.backtest_trust_score:.2f} / {row.backtest_trust_level_zh}, "
                f"repair={row.data_repair_actions_applied_zh}, "
                f"horizon={row.final_focus_horizon}, "
                f"watchlist={row.final_watchlist_status_zh}, "
                f"next={row.final_next_step_zh}"
            )
        recheck_queue = result.summary[result.summary["recheck_rank"] > 0].sort_values("recheck_rank")
        if not recheck_queue.empty:
            print("Re-check queue / 重新检查队列")
            for row in recheck_queue.head(10).itertuples(index=False):
                print(
                    f"{int(row.recheck_rank)}. {row.ticker}: "
                    f"recheck_score={row.recheck_priority_score:.2f}, "
                    f"level={row.recheck_priority_level} / {row.recheck_priority_level_zh}, "
                    f"price={row.latest_price:.2f}->{row.recheck_trigger_price:.2f}, "
                    f"distance={_format_optional_percent(row.recheck_trigger_distance_pct)} "
                    f"/ {row.recheck_trigger_distance_label_zh}, "
                    f"action={row.recheck_action_type} / {row.recheck_action_type_zh}, "
                    f"period={row.requested_period}->{row.analysis_period}, "
                    f"data={row.data_remediation_action} / {row.data_remediation_action_zh}, "
                    f"weak_layer={row.data_quality_weakest_layer_zh}, "
                    f"blocker={row.primary_blocker_zh}, "
                    f"trigger={row.recheck_reason_zh}"
                )
            print("Re-check action breakdown / 重新检查动作分类")
            action_summary = (
                recheck_queue.groupby(["recheck_action_type", "recheck_action_type_zh"])
                .agg(
                    ticker_count=("ticker", "count"),
                    avg_recheck_score=("recheck_priority_score", "mean"),
                )
                .reset_index()
                .sort_values(["ticker_count", "avg_recheck_score"], ascending=[False, False])
            )
            for row in action_summary.itertuples(index=False):
                print(
                    f"- {row.recheck_action_type} / {row.recheck_action_type_zh}: "
                    f"{int(row.ticker_count)} tickers, "
                    f"avg_score={float(row.avg_recheck_score):.2f}"
                )
    if result.failures:
        print("Failures / 失败项")
        for failure in result.failures:
            print(f"- {failure['ticker']}: {failure['error']}")
    if result.journal is not None:
        print(f"Journal report / 每日复盘: {result.journal.report_path}")
    print(f"Wrote outputs to {result.output_dir}")
    return 0


def validate_command(args: argparse.Namespace) -> int:
    return run_validation(apply_validation_preset(args), _raise_argument_error)


def _raise_argument_error(message: str) -> NoReturn:
    raise ValueError(message)


def _format_optional_percent(value: object) -> str:
    try:
        if value != value:
            return "N/A"
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "N/A"


def download_yfinance_command(args: argparse.Namespace) -> int:
    prices = download_yfinance_prices(
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        output_path=args.output,
    )
    print(f"Wrote {len(prices)} rows for {prices['ticker'].nunique()} tickers to {args.output}")
    return 0


def compare_runs_command(args: argparse.Namespace) -> int:
    result = compare_validation_runs(
        previous_dir=args.previous_run,
        current_dir=args.current_run,
        output_dir=args.output_dir,
    )
    print("Validation run comparison completed. / 验证运行对比完成。")
    print(f"Previous run / 上一次运行: {args.previous_run}")
    print(f"Current run / 当前运行: {args.current_run}")
    print(f"Output folder / 输出文件夹: {result.output_dir}")
    print()
    if not result.adoption_decision.empty:
        final = result.adoption_decision[
            result.adoption_decision["check_name"] == "final_decision"
        ]
        if not final.empty:
            row = final.iloc[0]
            print("Config adoption decision / 配置采用结论")
            print(f"- {row['status']} / {row['status_zh']}: {row['detail_zh']}")
            print()
    for row in result.summary.itertuples(index=False):
        print(
            f"- {row.metric} / {row.metric_zh}: "
            f"{row.previous_value} -> {row.current_value}, "
            f"change={row.change}, {row.assessment_zh}"
        )
    print()
    print(f"Open report: {result.output_files['markdown_report']}")
    return 0


def review_due_command(args: argparse.Namespace) -> int:
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return run_command(args)
    if args.command == "ml-run":
        return ml_run_command(args)
    if args.command == "paper-trade":
        return paper_trade_command(args)
    if args.command == "daily-report":
        return daily_report_command(args)
    if args.command == "analyze-ticker":
        return analyze_ticker_command(args)
    if args.command == "real":
        return real_ticker_command(args)
    if args.command == "scan":
        return scan_command(args)
    if args.command == "validate":
        return validate_command(args)
    if args.command == "compare-runs":
        return compare_runs_command(args)
    if args.command == "audit":
        return audit_command(args)
    if args.command == "download-yfinance":
        return download_yfinance_command(args)
    if args.command == "review-due":
        return review_due_command(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
