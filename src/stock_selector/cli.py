from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .analysis import analyze_ticker, render_ticker_analysis
from .audit import audit_price_csv
from .config import load_config
from .data import download_prices_for_period_multi_source, download_yfinance_prices
from .json_io import write_json
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
from .universe import load_historical_universe_membership, load_universe_tickers
from .universe_validation import run_all_builtin_universe_validations
from .validation_presets import apply_validation_preset, validation_preset_choices
from .walk_forward import DEFAULT_BENCHMARK_TICKERS, run_walk_forward_validation


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

    print("Real ticker analysis summary")
    print(f"Requested period / 请求周期: {result.requested_period}")
    print(f"Effective period / 实际使用周期: {result.effective_period}")
    print(f"Auto period upgraded / 自动周期升级: {result.auto_period_upgraded}")
    print(f"Period note / 周期说明: {result.auto_period_upgrade_reason_zh}")
    print("Data sources / 数据源")
    for symbol, provider in result.data_sources.items():
        print(f"- {symbol}: {provider}")
    warnings = [
        f"{symbol}: {'; '.join(items)}"
        for symbol, items in result.data_source_warnings.items()
        if items
    ]
    if warnings:
        print("Data source warnings / 数据源提示")
        for warning in warnings:
            print(f"- {warning}")
    print("Horizon definition / 周期定义")
    print("These labels describe analysis style, not fixed holding periods.")
    print("这些分类描述的是分析风格，不是固定持有时间。")
    print("- short / 短期: short-term trading setup / 短线交易节奏")
    print("- medium / 中期: trend continuation setup / 趋势延续判断")
    print("- long / 长期: long-term quality and trend setup / 长期质量与趋势判断")
    first = result.analysis.iloc[0]
    print("Final decision / 最终执行结论")
    print(f"Final decision / 最终结论: {first.final_decision} / {first.final_decision_zh}")
    print(f"Final focus horizon / 最终重点周期: {first.final_focus_horizon} / {first.final_focus_horizon_zh}")
    print(f"Final score / 最终分数: {first.final_score:.2f}")
    print(
        "Final watchlist status / 最终观察状态: "
        f"{first.final_watchlist_status} / {first.final_watchlist_status_zh}"
    )
    print(f"Final reason / 最终原因: {first.final_reason_zh}")
    print(f"Final next step / 下一步: {first.final_next_step_zh}")
    print("Priority blockers / 主要卡点排序")
    print(f"Primary blocker / 第一卡点: {first.primary_blocker} / {first.primary_blocker_zh}")
    print(f"Priority blockers / 主要卡点: {first.priority_blockers_zh}")
    print(f"Blocker count / 卡点数量: {int(first.priority_blocker_count)}")
    print(f"Blocker note / 卡点说明: {first.priority_blocker_note_zh}")
    print(f"Primary blocker resolution / 第一卡点解除条件: {first.primary_blocker_resolution_zh}")
    print(f"Re-check trigger / 重新检查触发条件: {first.blocker_recheck_trigger_zh}")
    print(
        "Blocker resolution score / 卡点解除分数: "
        f"{first.blocker_resolution_score:.2f}/100, "
        f"{first.blocker_resolution_level} / {first.blocker_resolution_level_zh}"
    )
    print(f"Resolution gap / 解除差距: {first.blocker_resolution_gap_zh}")
    print("Horizon alignment / 周期一致性")
    print(
        "Alignment / 一致性: "
        f"{first.horizon_alignment_label} / {first.horizon_alignment_label_zh}"
    )
    print(f"Alignment score / 一致性分数: {first.horizon_alignment_score:.2f}")
    print(
        "Horizon counts / 周期数量: "
        f"constructive={int(first.constructive_horizon_count)}, "
        f"weak={int(first.weak_horizon_count)}, "
        f"risk_wait={int(first.risk_wait_horizon_count)}"
    )
    print(f"Score spread / 分数差距: {first.horizon_signal_score_spread:.2f}")
    print(f"Alignment note / 一致性说明: {first.horizon_alignment_note_zh}")
    print("Primary decision / 主要结论")
    print(f"Decision / 结论: {first.primary_decision} / {first.primary_decision_zh}")
    print(f"Focus horizon / 重点周期: {first.decision_focus_horizon}")
    print(f"Why / 为什么: {first.decision_reason_zh}")
    print(f"Wait for / 等什么: {first.decision_wait_for_zh}")
    print(f"Invalidation / 判断失效: {first.decision_invalidation_zh}")
    screening_focus = result.analysis.sort_values("high_probability_score", ascending=False).iloc[0]
    print("High probability filter / 高概率筛选器")
    print(
        "Screening profile / 筛选规则: "
        f"{screening_focus.screening_profile} / {screening_focus.screening_profile_zh}"
    )
    print(
        "Final screening / 最终筛选结论: "
        f"{screening_focus.screening_action} / {screening_focus.screening_action_zh}"
    )
    print(f"Focus horizon / 重点周期: {screening_focus.horizon} / {screening_focus.horizon_zh_label}")
    print(f"High probability score / 高概率分数: {screening_focus.high_probability_score:.2f}")
    print(
        "High probability level / 高概率等级: "
        f"{screening_focus.high_probability_level} / {screening_focus.high_probability_level_zh}"
    )
    print(
        "Calibrated win probability / 校准后胜率估计: "
        f"{_format_optional_percent(screening_focus.calibrated_win_probability)}, "
        f"{screening_focus.calibrated_probability_level} / "
        f"{screening_focus.calibrated_probability_level_zh}"
    )
    print(
        "Probability calibration feedback / 概率校准反馈: "
        f"raw={_format_optional_percent(screening_focus.calibrated_win_probability_raw)}, "
        f"adjustment={_format_optional_percent(screening_focus.probability_calibration_adjustment)}, "
        f"source={screening_focus.probability_calibration_source_zh}, "
        f"sample={int(screening_focus.probability_calibration_sample_count)}"
    )
    print(
        "Probability calibration action / 概率校准动作: "
        f"{screening_focus.probability_calibration_action} / "
        f"{screening_focus.probability_calibration_action_zh}"
    )
    print(
        "Probability confidence / 概率置信度: "
        f"{screening_focus.calibrated_probability_confidence:.2f}, "
        f"{screening_focus.calibrated_probability_confidence_level} / "
        f"{screening_focus.calibrated_probability_confidence_level_zh}"
    )
    print(f"Probability note / 概率说明: {screening_focus.calibrated_probability_note_zh}")
    print(f"Quality gate passed / 是否通过质量门槛: {screening_focus.quality_gate_passed}")
    print(
        "Backtest used / 使用的买点回测: "
        f"{screening_focus.screening_backtest_entry_type}, "
        f"sample={screening_focus.screening_backtest_trade_count}, "
        f"win_rate={_format_optional_percent(screening_focus.screening_backtest_win_rate)}, "
        f"stop_hit={_format_optional_percent(screening_focus.screening_backtest_stop_hit_rate)}, "
        f"avg_return={_format_optional_percent(screening_focus.screening_backtest_average_return)}"
    )
    print(
        "Backtest trust gate / 回测可信度门槛: "
        f"{screening_focus.backtest_trust_score:.2f}, "
        f"{screening_focus.backtest_trust_level} / {screening_focus.backtest_trust_level_zh}"
    )
    print(
        "Stability gates / 稳定性门槛: "
        f"recent={screening_focus.recent_backtest_score:.2f} / "
        f"{screening_focus.recent_backtest_level_zh}, "
        f"decay={screening_focus.backtest_decay_score:.2f} / "
        f"{screening_focus.backtest_decay_level_zh}"
    )
    print(
        "Sample confidence / 样本置信度: "
        f"{screening_focus.sample_confidence_level} / {screening_focus.sample_confidence_level_zh}"
    )
    print(
        "Evidence strength / 证据强度: "
        f"{screening_focus.evidence_strength} / {screening_focus.evidence_strength_zh}"
    )
    print(f"Evidence note / 证据说明: {screening_focus.evidence_note_zh}")
    print(
        "Liquidity filter / 流动性过滤: "
        f"{screening_focus.liquidity_filter_passed}, "
        f"{screening_focus.liquidity_filter_reason_zh}"
    )
    print(
        "Entry readiness / 买点可执行性: "
        f"{screening_focus.entry_readiness_gate_passed}, "
        f"{screening_focus.entry_readiness_status} / "
        f"{screening_focus.entry_readiness_status_zh}, "
        f"score={screening_focus.entry_readiness_score:.2f}"
    )
    print(f"Entry readiness note / 买点可执行性说明: {screening_focus.entry_readiness_note_zh}")
    print(
        "Trade plan quality / 交易计划质量: "
        f"{screening_focus.trade_plan_quality_gate_passed}, "
        f"{screening_focus.trade_plan_quality_status} / "
        f"{screening_focus.trade_plan_quality_status_zh}, "
        f"score={screening_focus.trade_plan_quality_score:.2f}"
    )
    print(f"Trade plan quality note / 交易计划质量说明: {screening_focus.trade_plan_quality_note_zh}")
    print(f"Gate result / 门槛结果: {screening_focus.quality_gate_fail_reasons_zh}")
    print("Threshold calibration / 阈值校准")
    print(
        "Calibration action / 校准动作: "
        f"{screening_focus.calibration_action} / {screening_focus.calibration_action_zh}"
    )
    print(
        "Market regime / 市场状态分层: "
        f"{screening_focus.market_regime} / {screening_focus.market_regime_zh}"
    )
    print(
        "Market regime adjustment / 市场状态门槛调整: "
        f"signal={screening_focus.market_regime_signal_delta:.2f}, "
        f"confidence={screening_focus.market_regime_confidence_delta:.2f}, "
        f"sample={int(screening_focus.market_regime_sample_delta)}, "
        f"win_rate={_format_optional_percent(screening_focus.market_regime_win_rate_delta)}, "
        f"avg_return={_format_optional_percent(screening_focus.market_regime_average_return_delta)}"
    )
    print(f"Market regime note / 市场状态说明: {screening_focus.market_regime_note_zh}")
    print(
        "Recommended signal threshold / 建议信号分门槛: "
        f"{screening_focus.recommended_signal_threshold:.2f}"
    )
    print(
        "Recommended confidence threshold / 建议置信度门槛: "
        f"{screening_focus.recommended_confidence_threshold:.2f}"
    )
    print(
        "Recommended backtest sample min / 建议回测样本下限: "
        f"{int(screening_focus.recommended_backtest_sample_min)}"
    )
    print(
        "Recommended win rate min / 建议胜率下限: "
        f"{_format_optional_percent(screening_focus.recommended_backtest_win_rate_min)}"
    )
    print(f"Calibration note / 校准说明: {screening_focus.calibration_note_zh}")
    print("Calibrated screening / 校准后筛选")
    print(
        "Calibrated final screening / 校准后最终结论: "
        f"{screening_focus.calibrated_screening_action} / "
        f"{screening_focus.calibrated_screening_action_zh}"
    )
    print(
        "Calibrated quality gate passed / 校准后是否通过质量门槛: "
        f"{screening_focus.calibrated_quality_gate_passed}"
    )
    print(
        "Calibrated high probability score / 校准后高概率分数: "
        f"{screening_focus.calibrated_high_probability_score:.2f}"
    )
    print(
        "Calibrated gate result / 校准后门槛结果: "
        f"{screening_focus.calibrated_quality_gate_fail_reasons_zh}"
    )
    print("Calibrated watchlist plan / 校准后观察计划")
    print(
        "Calibrated watchlist status / 校准后观察状态: "
        f"{screening_focus.calibrated_watchlist_status} / "
        f"{screening_focus.calibrated_watchlist_status_zh}"
    )
    print(
        "Calibrated watchlist gap score / 校准后观察差距分: "
        f"{screening_focus.calibrated_watchlist_gap_score:.2f}"
    )
    print(
        "Calibrated trigger price / 校准后重新检查触发价: "
        f"{screening_focus.calibrated_watchlist_trigger_price:.2f}"
    )
    print(f"Calibrated missing items / 校准后未达标条件: {screening_focus.calibrated_watchlist_missing_items_zh}")
    print(f"Calibrated re-check reason / 校准后重新检查原因: {screening_focus.calibrated_watchlist_recheck_reason_zh}")
    print("Watchlist plan / 观察计划")
    print(
        "Watchlist status / 观察状态: "
        f"{screening_focus.watchlist_status} / {screening_focus.watchlist_status_zh}"
    )
    print(f"Watchlist gap score / 观察差距分: {screening_focus.watchlist_gap_score:.2f}")
    print(f"Trigger price / 重新检查触发价: {screening_focus.watchlist_trigger_price:.2f}")
    print(f"Ready items / 已达标条件: {screening_focus.watchlist_ready_items_zh}")
    print(f"Missing items / 未达标条件: {screening_focus.watchlist_missing_items_zh}")
    print(f"Re-check reason / 重新检查原因: {screening_focus.watchlist_recheck_reason_zh}")
    print("Confidence / 置信度")
    print(f"Confidence score / 置信度分数: {first.confidence_score:.2f}")
    print(f"Confidence level / 置信度等级: {first.confidence_level} / {first.confidence_level_zh}")
    print(f"Confidence note / 置信度说明: {first.confidence_note_zh}")
    print("Risk breakdown / 风险拆解")
    print(f"Overall risk / 综合风险: {first.overall_risk_level} / {first.overall_risk_level_zh}")
    print(f"Technical risk / 技术风险: {first.technical_risk_level} / {first.technical_risk_level_zh}")
    print(f"Market risk / 大盘风险: {first.market_risk_level} / {first.market_risk_level_zh}")
    print(
        "Relative strength risk / 相对强弱风险: "
        f"{first.relative_strength_risk_level} / {first.relative_strength_risk_level_zh}"
    )
    print(f"Sector risk / 板块风险: {first.sector_risk_level} / {first.sector_risk_level_zh}")
    print(f"Fundamental risk / 基本面风险: {first.fundamental_risk_level} / {first.fundamental_risk_level_zh}")
    print(f"Event risk / 事件风险: {first.event_risk_breakdown_level} / {first.event_risk_breakdown_level_zh}")
    print(
        "News sentiment risk / 新闻情绪风险: "
        f"{first.sentiment_risk_breakdown_level} / {first.sentiment_risk_breakdown_level_zh}"
    )
    print(
        "Analyst expectation risk / 分析师预期风险: "
        f"{first.analyst_risk_breakdown_level} / {first.analyst_risk_breakdown_level_zh}"
    )
    print(f"Valuation risk / 估值风险: {first.valuation_risk_breakdown_level} / {first.valuation_risk_breakdown_level_zh}")
    print(f"Data risk / 数据风险: {first.data_risk_level} / {first.data_risk_level_zh}")
    print(f"Risk note / 风险说明: {first.risk_breakdown_note_zh}")
    print("Data quality / 数据质量")
    print(f"Data quality score / 数据质量分数: {first.data_quality_score:.2f}")
    print(f"Data quality level / 数据质量等级: {first.data_quality_level} / {first.data_quality_level_zh}")
    print(
        "Weakest data layer / 最弱数据层: "
        f"{first.data_quality_weakest_layer} / {first.data_quality_weakest_layer_zh}"
    )
    print(f"Weak data layers / 薄弱数据层: {first.data_quality_weak_layers_zh}")
    print(
        "Repair priority / 修复优先级: "
        f"{first.data_quality_repair_priority} / {first.data_quality_repair_priority_zh}"
    )
    print(f"Repair actions / 修复动作: {first.data_quality_repair_actions_zh}")
    print(f"Data repair actions applied / 已自动修复: {first.data_repair_actions_applied_zh}")
    print(f"Price data / 价格数据: {first.price_data_status} / {first.price_data_status_zh}")
    print(
        "Price health / 价格健康: "
        f"{first.price_health_score:.2f}, "
        f"{first.price_health_level} / {first.price_health_level_zh}"
    )
    print(
        "Price anomalies / 价格异常: "
        f"max_gap_days={int(first.price_max_calendar_gap_days)}, "
        f"large_gaps={int(first.price_large_gap_count)}, "
        f"zero_volume={int(first.price_zero_volume_days)}, "
        f"missing_ohlcv={int(first.price_missing_ohlcv_rows)}, "
        f"extreme_returns={int(first.price_extreme_return_count)}"
    )
    print(f"Price health note / 价格健康说明: {first.price_health_note_zh}")
    print(f"Market data / 大盘数据: {first.market_data_status} / {first.market_data_status_zh}")
    print(
        "Relative strength data / 相对强弱数据: "
        f"{first.relative_strength_data_status} / {first.relative_strength_data_status_zh}"
    )
    print(f"Sector data / 板块数据: {first.sector_data_status} / {first.sector_data_status_zh}")
    print(
        "Fundamental data / 基本面数据: "
        f"{first.fundamental_data_status} / {first.fundamental_data_status_zh}"
    )
    print(f"Event data / 事件数据: {first.event_data_status} / {first.event_data_status_zh}")
    print(
        "News sentiment data / 新闻情绪数据: "
        f"{first.sentiment_data_status} / {first.sentiment_data_status_zh}"
    )
    print(f"Analyst data / 分析师数据: {first.analyst_data_status} / {first.analyst_data_status_zh}")
    print(f"Valuation data / 估值数据: {first.valuation_data_status} / {first.valuation_data_status_zh}")
    print(f"Missing fallback count / 缺失回退数量: {first.missing_fallback_count}")
    print(f"Data quality note / 数据质量说明: {first.data_quality_note_zh}")
    print(f"Market score / 大盘分数: {first.market_score:.2f}")
    print(f"Market status / 大盘状态: {first.market_status}")
    print("Sector context / 板块环境")
    print(f"Sector / 板块: {first.sector or 'N/A'}")
    print(f"Industry / 行业: {first.industry or 'N/A'}")
    print(f"Sector ETF / 板块ETF: {first.sector_etf or 'N/A'}")
    print(f"Sector score / 板块分数: {first.sector_score:.2f}")
    print(f"Sector status / 板块状态: {first.sector_status}")
    print(f"Note / 说明: {first.sector_note_zh}")
    if not result.peer_comparison.empty:
        target_peer_row = result.peer_comparison[result.peer_comparison["ticker"] == result.ticker]
        if not target_peer_row.empty:
            target_peer = target_peer_row.iloc[0]
            print("Peer comparison / 同业比较")
            print(
                f"Peer rank / 同业排名: {int(target_peer.peer_rank)} "
                f"of {len(result.peer_comparison)}"
            )
            print(f"Peer score / 同业高概率分数: {target_peer.high_probability_score:.2f}")
            print(
                "Peer screening / 同业筛选结论: "
                f"{target_peer.screening_action} / {target_peer.screening_action_zh}"
            )
            print(
                "Peer valuation / 同业估值: "
                f"{target_peer.peer_valuation_label} / {target_peer.peer_valuation_label_zh}"
            )
            print(
                f"Valuation rank / 估值排名: {int(target_peer.valuation_peer_rank)} "
                f"of {len(result.peer_comparison)}"
            )
            print(
                "Forward PE vs peer median / 预期市盈率相对同业中位数: "
                f"{_format_optional_percent(target_peer.forward_pe_vs_peer_median_pct)}"
            )
            print(
                "PEG vs peer median / PEG相对同业中位数: "
                f"{_format_optional_percent(target_peer.peg_vs_peer_median_pct)}"
            )
            print("Top peers / 同业前几名:")
            for row in result.peer_comparison.head(5).itertuples(index=False):
                print(
                    f"- {row.peer_rank}. {row.ticker}: "
                    f"score={row.high_probability_score:.2f}, "
                    f"passed={row.quality_gate_passed}, "
                    f"calibrated_passed={row.calibrated_quality_gate_passed}, "
                    f"horizon={row.focus_horizon}"
                )
            print(f"Peer report / 同业报告: {result.output_dir / 'peer_comparison.md'}")
    print("Fundamental quality / 基本面质量")
    print(f"Fundamental score / 基本面分数: {first.fundamental_score:.2f}")
    print(
        f"Quality / 质量: {first.fundamental_quality} / "
        f"{first.fundamental_quality_zh}"
    )
    print(f"Note / 说明: {first.fundamental_note_zh}")
    print("Analyst expectations / 分析师预期")
    print(f"Analyst score / 分析师分数: {first.analyst_score:.2f}")
    print(f"Analyst label / 分析师标签: {first.analyst_label} / {first.analyst_label_zh}")
    print(f"Analyst risk / 分析师风险: {first.analyst_risk_level} / {first.analyst_risk_level_zh}")
    print(f"Block new entries / 是否阻止新入场: {bool(first.analyst_block_new_entries)}")
    print(f"Analyst upside / 目标价上行空间: {_format_optional_percent(first.analyst_upside)}")
    print(f"Recommendation mean / 推荐均值: {_format_optional_number(first.recommendation_mean)}")
    print(f"Number of analysts / 覆盖分析师数: {_format_optional_number(first.number_of_analysts)}")
    print(f"Target mean price / 平均目标价: {_format_optional_number(first.target_mean_price)}")
    print(f"Note / 说明: {first.analyst_note_zh}")
    print("Valuation risk / 估值风险")
    print(f"Valuation score / 估值分数: {first.valuation_score:.2f}")
    print(f"Valuation label / 估值标签: {first.valuation_label} / {first.valuation_label_zh}")
    print(f"Valuation risk / 估值风险: {first.valuation_risk_level} / {first.valuation_risk_level_zh}")
    print(f"Block new entries / 是否阻止新入场: {bool(first.valuation_block_new_entries)}")
    print(f"Forward PE / 预期市盈率: {_format_optional_number(first.valuation_forward_pe)}")
    print(f"PEG ratio / PEG: {_format_optional_number(first.valuation_peg_ratio)}")
    print(f"Growth reference / 增长参考: {_format_optional_percent(first.valuation_growth_reference)}")
    print(f"Profit margin / 净利率: {_format_optional_percent(first.valuation_profit_margin)}")
    print(f"Note / 说明: {first.valuation_note_zh}")
    print("News sentiment / 新闻情绪")
    print(f"Sentiment score / 情绪分数: {first.sentiment_score:.2f}")
    print(f"Sentiment label / 情绪标签: {first.sentiment_label} / {first.sentiment_label_zh}")
    print(
        "Sentiment risk / 情绪风险: "
        f"{first.sentiment_risk_level} / {first.sentiment_risk_level_zh}"
    )
    print(f"Block new entries / 是否阻止新入场: {bool(first.sentiment_block_new_entries)}")
    print(
        "Keywords / 关键词: "
        f"positive={int(first.sentiment_positive_count)}, "
        f"negative={int(first.sentiment_negative_count)}, "
        f"high_risk={int(first.sentiment_high_risk_count)}"
    )
    print(f"Note / 说明: {first.sentiment_note_zh}")
    print("Event risk / 事件风险")
    print(
        f"Risk level / 风险等级: {first.event_risk_level} / "
        f"{first.event_risk_level_zh}"
    )
    print(f"Event window / 事件窗口: {first.event_window} / {first.event_window_zh}")
    print(f"Block new entries / 是否阻止新入场: {bool(first.event_block_new_entries)}")
    print(f"Cooldown active / 财报后冷却是否生效: {bool(first.event_cooldown_active)}")
    print(f"Next earnings / 下一次财报: {first.next_earnings_date or 'N/A'}")
    days_until = "N/A" if first.days_until_earnings != first.days_until_earnings else int(first.days_until_earnings)
    print(f"Days until earnings / 距离财报天数: {days_until}")
    print(f"Last earnings / 上一次财报: {first.last_earnings_date or 'N/A'}")
    days_since = "N/A" if first.days_since_earnings != first.days_since_earnings else int(first.days_since_earnings)
    print(f"Days since earnings / 距离上次财报天数: {days_since}")
    print(f"Note / 说明: {first.event_risk_note_zh}")
    print("Plain summary / 简明结论")
    for row in result.analysis.itertuples(index=False):
        print(f"- {row.ticker} {row.horizon} / {row.horizon_zh_label}: {row.plain_summary_zh}")
    print("Entry plan / 买点计划")
    for row in result.analysis.itertuples(index=False):
        print(
            f"- {row.ticker} {row.horizon} / {row.horizon_zh_label}: "
            f"{row.entry_distance_note_zh} {row.chase_status_zh} {row.entry_plan_zh}"
        )
    print("Backtest reliability / 回测可靠性")
    for row in result.analysis.itertuples(index=False):
        print(
            f"- {row.ticker} {row.horizon} / {row.horizon_zh_label}: "
            f"trust={row.backtest_trust_score:.2f} / {row.backtest_trust_level_zh}；"
            f"regime={row.regime_coverage_score:.2f} / {row.regime_coverage_level_zh}；"
            f"recent={row.recent_backtest_score:.2f} / {row.recent_backtest_level_zh}；"
            f"decay={row.backtest_decay_score:.2f} / {row.backtest_decay_level_zh}；"
            f"{row.backtest_reliability_note_zh}"
        )
    print("Entry backtest / 买点回测")
    print("How to read / 怎么看:")
    print(
        "execution model / 执行模型: "
        f"{first.backtest_execution_model} / {first.backtest_execution_model_zh}"
    )
    print(f"slippage / 滑点: {_format_optional_percent(first.backtest_slippage_pct)}")
    print(
        "slippage drivers / 滑点依据: "
        f"avg_dollar_volume={_format_optional_number(first.backtest_avg_dollar_volume)}, "
        f"atr_ratio={_format_optional_percent(first.backtest_atr_ratio)}, "
        f"liquidity={first.backtest_liquidity_label} / {first.backtest_liquidity_label_zh}, "
        f"volatility={first.backtest_volatility_label} / {first.backtest_volatility_label_zh}"
    )
    print(f"execution note / 执行说明: {first.backtest_execution_note_zh}")
    print(
        "backtest trust / 回测可信度: "
        f"{first.backtest_trust_score:.2f}, "
        f"{first.backtest_trust_level} / {first.backtest_trust_level_zh}"
    )
    print(f"trust note / 可信度说明: {first.backtest_trust_note_zh}")
    print(
        "regime coverage / 行情覆盖: "
        f"{first.regime_coverage_score:.2f}, "
        f"{first.regime_coverage_level} / {first.regime_coverage_level_zh}, "
        f"dominant={first.regime_coverage_dominant_regime} / "
        f"{first.regime_coverage_dominant_regime_zh}, "
        f"share={_format_optional_percent(first.regime_coverage_dominant_share)}"
    )
    print(f"regime coverage note / 行情覆盖说明: {first.regime_coverage_note_zh}")
    print(
        "recent backtest strength / 近期回测强度: "
        f"{first.recent_backtest_score:.2f}, "
        f"{first.recent_backtest_level} / {first.recent_backtest_level_zh}, "
        f"recent_win={_format_optional_percent(first.recent_backtest_win_rate)}, "
        f"recent_avg={_format_optional_percent(first.recent_backtest_average_return)}"
    )
    print(f"recent backtest note / 近期回测说明: {first.recent_backtest_note_zh}")
    print(
        "backtest decay / 回测衰退: "
        f"{first.backtest_decay_score:.2f}, "
        f"{first.backtest_decay_level} / {first.backtest_decay_level_zh}, "
        f"late_win={_format_optional_percent(first.backtest_decay_late_win_rate)}, "
        f"late_avg={_format_optional_percent(first.backtest_decay_late_average_return)}"
    )
    print(f"backtest decay note / 回测衰退说明: {first.backtest_decay_note_zh}")
    print("- win rate: historical similar entries that ended profitable / 过去类似买点最终赚钱的比例")
    print("- target: reached target before stop / 先到目标价的比例")
    print("- stop: reached stop before target / 先到止损价的比例")
    print("- avg return: average historical return for that entry type / 该类买点的历史平均收益")
    print("- quality: sample reliability based on sample count / 根据样本数判断统计可靠性")
    for row in result.analysis.itertuples(index=False):
        _print_entry_backtest_row(row)
    for row in result.analysis.itertuples(index=False):
        print(
            f"{row.ticker} {row.horizon} ({row.horizon_time_range}): "
            f"action={row.action}, "
            f"score={row.signal_score:.2f}, "
            f"entry={row.entry_price:.2f}, "
            f"stop={row.stop_loss:.2f}, "
            f"target={row.take_profit:.2f}, "
            f"risk_reward={row.risk_reward:.2f}"
        )
    print(f"Price rows: {len(result.prices)}")
    print(f"Wrote outputs to {result.output_dir}")
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
    run_started_at = datetime.now(timezone.utc)
    args = apply_validation_preset(args)
    if args.all_universes:
        if args.tickers or args.universe_file or args.historical_universe_file:
            raise ValueError(
                "--all-universes cannot be combined with manual tickers, --universe-file, "
                "or --historical-universe-file."
            )
        if args.preset:
            print(f"Preset / 预设: {args.preset} - {args.preset_description_zh}")
        screening_config = load_screening_config(args.screening_config)
        output_dir = (
            Path("outputs/walk_forward/all_universes")
            if args.output_dir == "outputs/walk_forward/latest"
            else Path(args.output_dir)
        )
        result = run_all_builtin_universe_validations(
            universe_names=[args.universe] if args.universe else None,
            period=args.period,
            step_days=args.step_days,
            min_history_days=args.min_history_days,
            output_dir=output_dir,
            screening_config=screening_config,
            progress_callback=_print_all_universe_progress,
        )
        print("All-universe validation summary / 全部股票池验证摘要")
        if not result.summary.empty:
            first = result.summary.iloc[0]
            print(
                "Primary diagnostic / 主要诊断: "
                f"{first['universe']} ({first['diagnostic_level_zh']}) - "
                f"{first['recommendation_zh']}; "
                f"主要卡点: {first['top_quality_gate_failures_zh']}; "
                f"建议阈值调整: {first['suggested_threshold_changes_zh']}"
            )
        if not result.ranking.empty:
            print("Best candidate universe ranking / 最佳股票池排序")
            for row in result.ranking.head(5).itertuples(index=False):
                print(
                    f"- #{row.rank} {row.universe}: {row.decision_zh}, "
                    f"score={row.ranking_score:.2f}, events={row.event_count}, "
                    f"high_probability_sample={row.high_probability_sample_count}, "
                    f"recommendation={row.recommendation_zh}"
                )
        for row in result.summary.itertuples(index=False):
            print(
                f"- {row.universe}: status={row.status}, tickers={row.ticker_count}, "
                f"priority={row.optimization_priority}, diagnostic={row.diagnostic_level_zh}, "
                f"events={row.event_count}, all_20d_win_rate={_format_optional_percent(row.all_20d_win_rate)}, "
                "high_probability_sample="
                f"{row.high_probability_sample_count}, "
                "high_probability_20d_win_rate="
                f"{_format_optional_percent(row.high_probability_20d_win_rate)}, "
                f"recommendation={row.recommendation_zh}, "
                f"top_blockers={row.top_quality_gate_failures_zh}, "
                f"suggested_changes={row.suggested_threshold_changes_zh}"
            )
        if result.failures:
            print("Failures / 失败项")
            for failure in result.failures:
                print(f"- {failure['universe']}: {failure['error']}")
        print(f"Wrote outputs to {result.output_dir}")
        return 0

    historical_membership = (
        load_historical_universe_membership(args.historical_universe_file)
        if args.historical_universe_file
        else None
    )
    if historical_membership is not None and not args.tickers and not args.universe and not args.universe_file:
        tickers = historical_membership.tickers()
    else:
        tickers = load_universe_tickers(
            tickers=args.tickers,
            universe_name=args.universe,
            universe_file=args.universe_file,
        )
    screening_config = load_screening_config(args.screening_config)
    download_tickers = _with_benchmark_tickers(tickers)
    print(
        "Downloading price data / 正在下载价格数据: "
        f"{len(download_tickers)} tickers, period={args.period}, "
        "timeout=30s, max_attempts=2",
        flush=True,
    )
    price_result = download_prices_for_period_multi_source(
        download_tickers,
        period=args.period,
        output_path=Path("data/real_prices")
        / f"walk_forward_{_validation_stem(tickers, args.universe)}_{args.period}.csv",
    )
    _print_price_download_result(price_result)
    print("Running walk-forward validation / 正在运行滚动验证...", flush=True)
    result = run_walk_forward_validation(
        prices=price_result.prices,
        tickers=tickers,
        step_days=args.step_days,
        min_history_days=args.min_history_days,
        output_dir=args.output_dir,
        screening_config=screening_config,
        universe_membership=historical_membership,
        progress_callback=_print_walk_forward_progress,
    )
    print("Walk-forward validation summary / 滚动历史验证摘要")
    if args.preset:
        print(f"Preset / 预设: {args.preset} - {args.preset_description_zh}")
    print(f"Tickers / 股票: {', '.join(tickers)}")
    print(f"Benchmarks / 基准: {', '.join(DEFAULT_BENCHMARK_TICKERS)}")
    print(
        "Validation settings / 验证参数: "
        f"period={args.period}, step_days={args.step_days}, "
        f"min_history_days={args.min_history_days}"
    )
    print(f"Event rows / 信号样本行数: {len(result.events)}")
    survivorship = result.survivorship_bias_report
    print(
        "Survivorship bias / 幸存者偏差: "
        f"handled={str(bool(survivorship.get('survivorship_bias_handled'))).lower()}, "
        f"delisted_sample={str(bool(survivorship.get('contains_delisted_tickers'))).lower()}, "
        f"source={survivorship.get('source', 'none')}"
    )
    if not result.summary.empty:
        for row in result.summary.itertuples(index=False):
            win_rate = getattr(row, "win_rate_20d", None)
            avg_return = getattr(row, "avg_return_20d", None)
            print(
                f"- {row.bucket}: sample={row.sample_count}, "
                f"20d_win_rate={_format_optional_percent(win_rate)}, "
                f"20d_avg_return={_format_optional_percent(avg_return)}"
            )
    if not result.ticker_ranking.empty:
        print("Ticker validation ranking / 个股验证排名")
        for row in result.ticker_ranking.head(10).itertuples(index=False):
            win_rate = getattr(row, "win_rate_20d", None)
            avg_return = getattr(row, "avg_return_20d", None)
            print(
                f"- #{row.rank} {row.ticker}: {row.ticker_decision_zh}, "
                f"score={row.ticker_ranking_score:.2f}, sample={row.sample_count}, "
                f"20d_win_rate={_format_optional_percent(win_rate)}, "
                f"20d_avg_return={_format_optional_percent(avg_return)}"
            )
    if not result.sample_sufficiency.empty:
        print("Sample sufficiency guidance / 样本充分性建议")
        guidance_rows = result.sample_sufficiency[
            result.sample_sufficiency["sample_status"] != "enough_samples"
        ].head(10)
        if guidance_rows.empty:
            print("- Samples are sufficient for the current validation settings. / 当前验证样本充足。")
        else:
            for row in guidance_rows.itertuples(index=False):
                print(
                    f"- {row.scope} {row.ticker}: {row.sample_status_zh}, "
                    f"sample={row.sample_count}/{row.target_sample_count}, "
                    f"{row.guidance_action_zh}, {row.suggested_command_hint_zh}"
                )
    if not result.calibration.empty:
        print("Rule calibration / 规则校准")
        for row in result.calibration.itertuples(index=False):
            print(
                f"- {row.rule}: current={row.current_threshold}, "
                f"suggested={row.suggested_threshold}, {row.recommendation_zh}"
            )
    if not result.profile_summary.empty:
        print("Profile summary / 分类规则表现")
        for row in result.profile_summary.itertuples(index=False):
            win_rate = getattr(row, "win_rate_20d", None)
            avg_return = getattr(row, "avg_return_20d", None)
            print(
                f"- {row.screening_profile} / {row.screening_profile_zh} "
                f"[{row.validation_bucket}]: sample={row.sample_count}, "
                f"20d_win_rate={_format_optional_percent(win_rate)}, "
                f"20d_avg_return={_format_optional_percent(avg_return)}"
            )
    if not result.probability_calibration.empty:
        print("Probability calibration / 概率校准")
        for row in result.probability_calibration.itertuples(index=False):
            actual_win_rate = getattr(row, "actual_win_rate_20d", None)
            avg_estimated = getattr(row, "avg_estimated_probability", None)
            error_abs = getattr(row, "calibration_error_abs", None)
            adjustment = getattr(row, "recommended_probability_adjustment", None)
            print(
                f"- {row.probability_bucket} / {row.probability_bucket_zh}: "
                f"sample={row.sample_count}, "
                f"estimated={_format_optional_percent(avg_estimated)}, "
                f"actual_20d={_format_optional_percent(actual_win_rate)}, "
                f"error={_format_optional_percent(error_abs)}, "
                f"adjustment={_format_optional_percent(adjustment)}, "
                f"action={row.formula_action_zh}, "
                f"quality={row.calibration_quality_zh}"
            )
    if not result.portfolio_summary.empty:
        print("Portfolio validation / 组合验证")
        for row in result.portfolio_summary.itertuples(index=False):
            print(
                f"- {row.portfolio_name} / {row.portfolio_name_zh}: "
                f"rebalances={row.rebalance_count}, "
                f"avg_positions={row.avg_position_count:.2f}, "
                f"win_rate={_format_optional_percent(row.win_rate)}, "
                f"avg_return={_format_optional_percent(row.avg_forward_return)}, "
                f"compounded={_format_optional_percent(row.compounded_forward_return)}"
            )
    if not result.portfolio_equity_summary.empty:
        print("Portfolio equity curve / 组合逐日净值曲线")
        for row in result.portfolio_equity_summary.itertuples(index=False):
            print(
                f"- {row.portfolio_name} / {row.portfolio_name_zh}: "
                f"days={row.daily_rows}, "
                f"total_return={_format_optional_percent(row.total_return)}, "
                f"max_drawdown={_format_optional_percent(row.max_drawdown)}, "
                f"sharpe={_format_optional_number(row.sharpe)}"
            )
    if not result.benchmark_summary.empty:
        print("Benchmark comparison / 基准对比")
        for row in result.benchmark_summary.itertuples(index=False):
            print(
                f"- {row.portfolio_name} vs {row.benchmark_ticker}: "
                f"portfolio={_format_optional_percent(row.portfolio_total_return)}, "
                f"benchmark={_format_optional_percent(row.benchmark_total_return)}, "
                f"excess={_format_optional_percent(row.excess_total_return)}, "
                f"corr={_format_optional_number(row.daily_correlation)}"
            )
    if not result.benchmark_policy.empty:
        print("Benchmark-aware rule policy / 基准感知规则建议")
        for row in result.benchmark_policy.itertuples(index=False):
            print(
                f"- {row.portfolio_name} / {row.portfolio_name_zh}: "
                f"action={row.benchmark_policy_action} / {row.benchmark_policy_action_zh}, "
                f"bias={row.threshold_bias} / {row.threshold_bias_zh}, "
                f"avg_excess={_format_optional_percent(row.avg_excess_total_return)}, "
                f"note={row.policy_note_zh}"
            )
    if not result.benchmark_tightening.empty:
        print("Specific tightening recommendations / 具体收紧建议")
        for row in result.benchmark_tightening.head(10).itertuples(index=False):
            print(
                f"- {row.screening_profile} / {row.screening_profile_zh}: "
                f"{row.threshold_attr} {row.current_threshold:g}->{row.suggested_threshold:g}, "
                f"priority={row.priority_zh}, reason={row.recommendation_reason_zh}"
            )
    if not result.tightening_impact.empty:
        print("Tightening impact validation / 收紧效果验证")
        for row in result.tightening_impact.itertuples(index=False):
            print(
                f"- {row.screening_profile} / {row.screening_profile_zh}: "
                f"samples={row.before_sample_count}->{row.after_sample_count}, "
                f"win_rate={_format_optional_percent(row.before_win_rate)}"
                f"->{_format_optional_percent(row.after_win_rate)}, "
                f"avg_return={_format_optional_percent(row.before_avg_return)}"
                f"->{_format_optional_percent(row.after_avg_return)}, "
                f"decision={row.impact_decision_zh}"
            )
    if not result.threshold_sensitivity.empty:
        print("Threshold sensitivity grid / 阈值敏感度网格")
        top_rows = result.threshold_sensitivity[
            result.threshold_sensitivity["sensitivity_decision"] != "baseline"
        ].head(10)
        for row in top_rows.itertuples(index=False):
            print(
                f"- {row.screening_profile} / {row.screening_profile_zh}: "
                f"{row.threshold_expression_zh}, "
                f"samples={row.sample_count}, "
                f"win_change={_format_optional_percent(row.win_rate_change)}, "
                f"return_change={_format_optional_percent(row.avg_return_change)}, "
                f"decision={row.sensitivity_decision_zh}"
            )
    if not result.minimum_sample_guard.empty:
        print("Minimum sample guard / 最小样本保护")
        for row in result.minimum_sample_guard.head(10).itertuples(index=False):
            print(
                f"- {row.screening_profile} / {row.screening_profile_zh}: "
                f"{row.threshold_expression_zh}, samples={row.sample_count}, "
                f"guard={row.guard_action_zh}, reason={row.guard_reason_zh}"
            )
    if not result.profile_calibration.empty:
        print("Profile rule calibration / 分类规则阈值建议")
        for row in result.profile_calibration.itertuples(index=False):
            print(
                f"- {row.screening_profile} / {row.screening_profile_zh} "
                f"{row.rule}: current={row.current_threshold}, "
                f"suggested={row.suggested_threshold}, "
                f"confidence={row.suggestion_confidence_zh}, {row.recommendation_zh}"
            )
    manifest_path = _write_validation_run_manifest(
        output_dir=Path(args.output_dir),
        args=args,
        tickers=tickers,
        download_tickers=download_tickers,
        price_result=price_result,
        result=result,
        started_at=run_started_at,
    )
    print(f"Run manifest / 运行记录: {manifest_path}")
    print(f"Wrote outputs to {args.output_dir}")
    return 0


def _format_optional_percent(value: object) -> str:
    try:
        if value != value:
            return "N/A"
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "N/A"


def _format_optional_number(value: object) -> str:
    try:
        if value != value:
            return "N/A"
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if abs(number) >= 1_000_000_000:
        return f"{number / 1_000_000_000:.2f}B"
    if abs(number) >= 1_000_000:
        return f"{number / 1_000_000:.2f}M"
    return f"{number:.2f}"


def _with_benchmark_tickers(tickers: list[str]) -> list[str]:
    combined = [*tickers, *DEFAULT_BENCHMARK_TICKERS]
    return list(dict.fromkeys(str(ticker).upper().strip() for ticker in combined if str(ticker).strip()))


def _print_all_universe_progress(universe_name: str, status: str, index: int, total: int) -> None:
    status_zh = {
        "started": "开始",
        "completed": "完成",
        "failed": "失败",
    }.get(status, status)
    print(
        f"[{index}/{total}] {universe_name}: {status} / {status_zh}",
        flush=True,
    )


def _print_price_download_result(price_result) -> None:
    print(
        "Price download result / 价格下载结果: "
        f"provider={price_result.provider}, attempts={', '.join(price_result.attempts)}",
        flush=True,
    )
    missing_tickers = getattr(price_result, "missing_tickers", ())
    if missing_tickers:
        print(
            "Missing price tickers / 缺失价格股票: "
            f"{', '.join(missing_tickers)}",
            flush=True,
        )
    for warning in getattr(price_result, "warnings", ()):
        print(f"Data warning / 数据警告: {warning}", flush=True)


def _write_validation_run_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    tickers: list[str],
    download_tickers: list[str],
    price_result,
    result,
    started_at: datetime,
) -> Path:
    completed_at = datetime.now(timezone.utc)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = {
        "walk_forward_report": str(output_dir / "walk_forward_report.md"),
        "validation_result_json": str(output_dir / "validation_result.json"),
        "events_csv": str(output_dir / "walk_forward_events.csv"),
        "summary_csv": str(output_dir / "walk_forward_summary.csv"),
        "ticker_ranking_csv": str(output_dir / "ticker_validation_ranking.csv"),
        "sample_sufficiency_csv": str(output_dir / "sample_sufficiency_guidance.csv"),
        "profile_summary_csv": str(output_dir / "profile_validation_summary.csv"),
        "probability_calibration_csv": str(output_dir / "probability_calibration.csv"),
        "portfolio_summary_csv": str(output_dir / "portfolio_validation_summary.csv"),
    }
    manifest_path = output_dir / "run_manifest.json"
    write_json(
        manifest_path,
        {
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": completed_at.isoformat(),
            "duration_seconds": (completed_at - started_at).total_seconds(),
            "preset": getattr(args, "preset", None),
            "preset_description": getattr(args, "preset_description", ""),
            "preset_description_zh": getattr(args, "preset_description_zh", ""),
            "universe": getattr(args, "universe", None),
            "universe_file": getattr(args, "universe_file", None),
            "historical_universe_file": getattr(args, "historical_universe_file", None),
            "all_universes": bool(getattr(args, "all_universes", False)),
            "period": args.period,
            "step_days": args.step_days,
            "min_history_days": args.min_history_days,
            "screening_config": getattr(args, "screening_config", None),
            "requested_tickers": tickers,
            "download_tickers": download_tickers,
            "benchmark_tickers": list(DEFAULT_BENCHMARK_TICKERS),
            "price_provider": price_result.provider,
            "price_provider_attempts": list(price_result.attempts),
            "price_warnings": list(price_result.warnings),
            "missing_price_tickers": list(getattr(price_result, "missing_tickers", ())),
            "survivorship_bias_report": getattr(result, "survivorship_bias_report", {}),
            "event_count": len(result.events),
            "summary_rows": len(result.summary),
            "ticker_ranking_rows": len(result.ticker_ranking),
            "sample_sufficiency_rows": len(result.sample_sufficiency),
            "output_dir": str(output_dir),
            "output_files": output_files,
        },
    )
    return manifest_path


def _print_walk_forward_progress(ticker: str, status: str, index: int, total: int) -> None:
    status_zh = {
        "started": "开始",
        "completed": "完成",
        "skipped_insufficient_history": "历史数据不足，跳过",
    }.get(status, status)
    print(f"[{index}/{total}] {ticker}: {status} / {status_zh}", flush=True)


def _validation_stem(tickers: list[str], universe_name: str | None) -> str:
    if universe_name:
        return universe_name.lower().strip().replace(" ", "_")
    if len(tickers) <= 6:
        return "_".join(str(ticker).upper().strip() for ticker in tickers)
    return f"{'_'.join(str(ticker).upper().strip() for ticker in tickers[:6])}_{len(tickers)}tickers"


def _print_entry_backtest_row(row: object) -> None:
    print(f"\n{row.ticker} {row.horizon} / {row.horizon_zh_label} ({row.horizon_time_range})")
    print(
        "  regime coverage / 行情覆盖: "
        f"{row.regime_coverage_score:.2f}, "
        f"{row.regime_coverage_level} / {row.regime_coverage_level_zh}, "
        f"dominant={row.regime_coverage_dominant_regime} / "
        f"{row.regime_coverage_dominant_regime_zh}, "
        f"share={_format_optional_percent(row.regime_coverage_dominant_share)}"
    )
    print(
        "  recent strength / 近期强度: "
        f"{row.recent_backtest_score:.2f}, "
        f"{row.recent_backtest_level} / {row.recent_backtest_level_zh}, "
        f"recent_win={_format_optional_percent(row.recent_backtest_win_rate)}, "
        f"recent_avg={_format_optional_percent(row.recent_backtest_average_return)}"
    )
    print(
        "  decay check / 衰退检查: "
        f"{row.backtest_decay_score:.2f}, "
        f"{row.backtest_decay_level} / {row.backtest_decay_level_zh}, "
        f"late_win={_format_optional_percent(row.backtest_decay_late_win_rate)}, "
        f"late_avg={_format_optional_percent(row.backtest_decay_late_average_return)}"
    )
    _print_entry_type_backtest(
        label="Breakout / 突破买点",
        trade_count=row.breakout_trade_count,
        sample_quality=row.breakout_sample_quality,
        sample_quality_zh=row.breakout_sample_quality_zh,
        win_rate=row.breakout_win_rate,
        target_hit_rate=row.breakout_target_hit_rate,
        stop_hit_rate=row.breakout_stop_hit_rate,
        average_gain=row.breakout_average_gain,
        average_loss=row.breakout_average_loss,
        average_return=row.breakout_average_return,
    )
    _print_entry_type_backtest(
        label="Pullback / 回调买点",
        trade_count=row.pullback_trade_count,
        sample_quality=row.pullback_sample_quality,
        sample_quality_zh=row.pullback_sample_quality_zh,
        win_rate=row.pullback_win_rate,
        target_hit_rate=row.pullback_target_hit_rate,
        stop_hit_rate=row.pullback_stop_hit_rate,
        average_gain=row.pullback_average_gain,
        average_loss=row.pullback_average_loss,
        average_return=row.pullback_average_return,
    )


def _print_entry_type_backtest(
    label: str,
    trade_count: int,
    sample_quality: str,
    sample_quality_zh: str,
    win_rate: object,
    target_hit_rate: object,
    stop_hit_rate: object,
    average_gain: object,
    average_loss: object,
    average_return: object,
) -> None:
    print(f"  {label}")
    print(f"    sample / 样本数: {trade_count}")
    print(f"    quality / 可靠性: {sample_quality} / {sample_quality_zh}")
    print(f"    win rate / 胜率: {_format_optional_percent(win_rate)}")
    print(f"    target hit / 目标价命中: {_format_optional_percent(target_hit_rate)}")
    print(f"    stop hit / 止损命中: {_format_optional_percent(stop_hit_rate)}")
    print(f"    avg gain / 平均盈利: {_format_optional_percent(average_gain)}")
    print(f"    avg loss / 平均亏损: {_format_optional_percent(average_loss)}")
    print(f"    avg return / 平均收益: {_format_optional_percent(average_return)}")


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
    for row in result.summary.itertuples(index=False):
        print(
            f"- {row.metric} / {row.metric_zh}: "
            f"{row.previous_value} -> {row.current_value}, "
            f"change={row.change}, {row.assessment_zh}"
        )
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
    due_now = result.due_items[result.due_items["review_status"] == "due_now"] if not result.due_items.empty else result.due_items
    if due_now.empty:
        print("No due review items now. / 当前没有已到期复盘项。")
    else:
        print("Due review items / 已到期复盘项")
        for row in due_now.head(20).itertuples(index=False):
            print(
                f"- {row.ticker} {row.review_window}: "
                f"signal_date={row.signal_date}, "
                f"estimated_review_date={row.estimated_review_date}, "
                f"report={row.report_path}"
            )
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
