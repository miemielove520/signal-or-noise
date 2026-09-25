from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stock_selector.data import download_prices_for_period_multi_source  # noqa: E402
from stock_selector.json_io import write_json  # noqa: E402
from stock_selector.real_data import normalize_ticker  # noqa: E402
from stock_selector.screening_config import load_screening_config  # noqa: E402
from stock_selector.universe import load_historical_universe_membership, load_universe_tickers  # noqa: E402
from stock_selector.universe_validation import run_all_builtin_universe_validations  # noqa: E402
from stock_selector.validation_presets import (  # noqa: E402
    apply_validation_preset,
    validation_preset_choices,
)
from stock_selector.walk_forward import DEFAULT_BENCHMARK_TICKERS, run_walk_forward_validation  # noqa: E402


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
    args = parser.parse_args(argv)
    args = apply_validation_preset(args)
    run_started_at = datetime.now(timezone.utc)
    if args.all_universes:
        if args.tickers or args.universe_file or args.historical_universe_file:
            parser.error(
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
        _print_all_universe_validation_result(result)
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
    output_dir = Path(args.output_dir)
    data_path = Path("data/real_prices") / f"walk_forward_{_validation_stem(tickers, args.universe)}_{args.period}.csv"

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
        output_path=data_path,
    )
    _print_price_download_result(price_result)
    print("Running walk-forward validation / 正在运行滚动验证...", flush=True)
    result = run_walk_forward_validation(
        prices=price_result.prices,
        tickers=tickers,
        step_days=args.step_days,
        min_history_days=args.min_history_days,
        output_dir=output_dir,
        screening_config=screening_config,
        universe_membership=historical_membership,
        progress_callback=_print_walk_forward_progress,
    )

    print()
    print("Walk-forward validation completed. / 滚动历史验证完成。")
    if args.preset:
        print(f"Preset / 预设: {args.preset} - {args.preset_description_zh}")
    print(f"Tickers / 股票: {', '.join(tickers)}")
    print(f"Benchmarks / 基准: {', '.join(DEFAULT_BENCHMARK_TICKERS)}")
    print(f"Price provider / 价格数据源: {price_result.provider}")
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
    print(f"Output folder / 输出文件夹: {output_dir}")
    print()
    if result.summary.empty:
        print("No validation events were produced. / 没有生成验证样本。")
    else:
        print("Performance summary / 表现摘要")
        for row in result.summary.itertuples(index=False):
            win_rate = getattr(row, "win_rate_20d", None)
            avg_return = getattr(row, "avg_return_20d", None)
            print(
                f"- {row.bucket}: sample={row.sample_count}, "
                f"20d_win_rate={_format_optional_percent(win_rate)}, "
                f"20d_avg_return={_format_optional_percent(avg_return)}"
            )
    if not result.ticker_ranking.empty:
        print()
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
        print()
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
        print()
        print("Rule calibration / 规则校准")
        for row in result.calibration.itertuples(index=False):
            print(
                f"- {row.rule}: current={row.current_threshold}, "
                f"suggested={row.suggested_threshold}, "
                f"{row.recommendation_zh}"
            )
    if not result.profile_summary.empty:
        print()
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
    _print_win_rate_dashboard(result)
    _print_profile_health_dashboard(result)
    _print_profile_action_recommendations(result)
    _print_profile_blocker_dashboard(result)
    _print_historical_win_rate_gate(result)
    _print_historical_threshold_recommendations(result)
    if not result.probability_calibration.empty:
        print()
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
        print()
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
        print()
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
        print()
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
        print()
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
        print()
        print("Specific tightening recommendations / 具体收紧建议")
        for row in result.benchmark_tightening.head(10).itertuples(index=False):
            print(
                f"- {row.screening_profile} / {row.screening_profile_zh}: "
                f"{row.threshold_attr} {row.current_threshold:g}->{row.suggested_threshold:g}, "
                f"priority={row.priority_zh}, reason={row.recommendation_reason_zh}"
            )
    if not result.tightening_impact.empty:
        print()
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
        print()
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
        print()
        print("Minimum sample guard / 最小样本保护")
        for row in result.minimum_sample_guard.head(10).itertuples(index=False):
            print(
                f"- {row.screening_profile} / {row.screening_profile_zh}: "
                f"{row.threshold_expression_zh}, samples={row.sample_count}, "
                f"guard={row.guard_action_zh}, reason={row.guard_reason_zh}"
            )
    if not result.profile_calibration.empty:
        print()
        print("Profile rule calibration / 分类规则阈值建议")
        for row in result.profile_calibration.itertuples(index=False):
            print(
                f"- {row.screening_profile} / {row.screening_profile_zh} "
                f"{row.rule}: current={row.current_threshold}, "
                f"suggested={row.suggested_threshold}, "
                f"confidence={row.suggestion_confidence_zh}, {row.recommendation_zh}"
            )
    print()
    print(f"Open report: {output_dir / 'walk_forward_report.md'}")
    manifest_path = _write_validation_run_manifest(
        output_dir=output_dir,
        args=args,
        tickers=tickers,
        download_tickers=download_tickers,
        price_result=price_result,
        result=result,
        started_at=run_started_at,
    )
    print(f"Run manifest / 运行记录: {manifest_path}")
    return 0


def _print_all_universe_validation_result(result) -> None:
    print()
    print("All-universe validation completed. / 全部股票池验证完成。")
    print(f"Universe rows / 股票池数量: {len(result.summary)}")
    print(f"Output folder / 输出文件夹: {result.output_dir}")
    print()
    if result.summary.empty:
        print("No universe validation rows were produced. / 没有生成股票池验证结果。")
    else:
        first = result.summary.iloc[0]
        print("Primary diagnostic / 主要诊断")
        print(
            f"- Best next research target / 下一步最值得研究: {first['universe']} "
            f"({first['diagnostic_level_zh']})"
        )
        print(f"- Recommendation / 建议: {first['recommendation_zh']}")
        print(f"- Top blockers / 主要卡点: {first['top_quality_gate_failures_zh']}")
        print(f"- Suggested threshold changes / 建议阈值调整: {first['suggested_threshold_changes_zh']}")
        print()
        if not result.ranking.empty:
            print("Best candidate universe ranking / 最佳股票池排序")
            for row in result.ranking.head(5).itertuples(index=False):
                print(
                    f"- #{row.rank} {row.universe}: {row.decision_zh}, "
                    f"score={row.ranking_score:.2f}, events={row.event_count}, "
                    f"high_probability_sample={row.high_probability_sample_count}, "
                    f"recommendation={row.recommendation_zh}"
                )
            print()
        print("Universe comparison / 股票池对比")
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
        print()
        print("Failures / 失败项")
        for failure in result.failures:
            print(f"- {failure['universe']}: {failure['error']}")
    print()
    print(f"Open report: {result.output_dir / 'all_universe_validation_report.md'}")
    print(f"Open ranking: {result.output_dir / 'best_universe_ranking.csv'}")


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
    validation = getattr(price_result, "source_validation", None) or {}
    if validation:
        print(
            "Price source validation / 价格源验证: "
            f"{validation.get('status', 'unknown')} / "
            f"{validation.get('status_zh', '未知')}",
            flush=True,
        )


def _write_validation_run_manifest(
    output_dir: Path,
    args,
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
        "segment_summary_csv": str(output_dir / "segment_validation_summary.csv"),
        "market_regime_summary_csv": str(
            output_dir / "market_regime_validation_summary.csv"
        ),
        "market_regime_policy_csv": str(output_dir / "market_regime_policy.csv"),
        "probability_calibration_csv": str(output_dir / "probability_calibration.csv"),
        "win_rate_dashboard_md": str(output_dir / "win_rate_dashboard.md"),
        "win_rate_dashboard_json": str(output_dir / "win_rate_dashboard.json"),
        "win_rate_overall_csv": str(output_dir / "win_rate_overall.csv"),
        "win_rate_by_horizon_csv": str(output_dir / "win_rate_by_horizon.csv"),
        "win_rate_by_entry_type_csv": str(output_dir / "win_rate_by_entry_type.csv"),
        "win_rate_by_profile_csv": str(output_dir / "win_rate_by_profile.csv"),
        "win_rate_by_quality_gate_csv": str(output_dir / "win_rate_by_quality_gate.csv"),
        "profile_health_dashboard_md": str(output_dir / "profile_health_dashboard.md"),
        "profile_health_dashboard_json": str(output_dir / "profile_health_dashboard.json"),
        "profile_health_dashboard_csv": str(output_dir / "profile_health_dashboard.csv"),
        "profile_action_recommendations_md": str(
            output_dir / "profile_action_recommendations.md"
        ),
        "profile_action_recommendations_json": str(
            output_dir / "profile_action_recommendations.json"
        ),
        "profile_action_recommendations_csv": str(
            output_dir / "profile_action_recommendations.csv"
        ),
        "profile_blocker_dashboard_md": str(output_dir / "profile_blocker_dashboard.md"),
        "profile_blocker_dashboard_json": str(output_dir / "profile_blocker_dashboard.json"),
        "profile_blocker_dashboard_csv": str(output_dir / "profile_blocker_dashboard.csv"),
        "historical_win_rate_gate_md": str(output_dir / "historical_win_rate_gate.md"),
        "historical_win_rate_gate_json": str(output_dir / "historical_win_rate_gate.json"),
        "historical_win_rate_gate_csv": str(output_dir / "historical_win_rate_gate.csv"),
        "historical_threshold_recommendations_md": str(
            output_dir / "historical_threshold_recommendations.md"
        ),
        "historical_threshold_recommendations_json": str(
            output_dir / "historical_threshold_recommendations.json"
        ),
        "historical_threshold_recommendations_csv": str(
            output_dir / "historical_threshold_recommendations.csv"
        ),
        "portfolio_summary_csv": str(output_dir / "portfolio_validation_summary.csv"),
        "data_source_validation_json": str(output_dir / "data_source_validation.json"),
    }
    write_json(
        output_dir / "data_source_validation.json",
        {
            "generated_at_utc": completed_at.isoformat(),
            "price_provider": price_result.provider,
            "price_provider_attempts": list(price_result.attempts),
            "price_warnings": list(price_result.warnings),
            "missing_price_tickers": list(getattr(price_result, "missing_tickers", ())),
            "source_validation": getattr(price_result, "source_validation", None) or {},
        },
    )
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
            "price_source_validation": getattr(price_result, "source_validation", None) or {},
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


def _print_win_rate_dashboard(result) -> None:
    dashboard = getattr(result, "win_rate_dashboard", {}) or {}
    overall = dashboard.get("overall")
    if overall is None or overall.empty:
        return
    print()
    print("Historical win-rate dashboard / 历史胜率统计面板")
    for row in overall.itertuples(index=False):
        print(
            f"- {row.forward_window_days}d: sample={row.sample_count}, "
            f"win_rate={_format_optional_percent(row.win_rate)}, "
            f"avg_return={_format_optional_percent(row.avg_return)}, "
            f"avg_gain={_format_optional_percent(row.avg_gain)}, "
            f"avg_loss={_format_optional_percent(row.avg_loss)}, "
            f"quality={row.sample_quality_zh}, result={row.result_label_zh}"
        )
    print("- Full dashboard / 完整面板: win_rate_dashboard.md")


def _print_profile_health_dashboard(result) -> None:
    profile_health = getattr(result, "profile_health_dashboard", None)
    if profile_health is None or profile_health.empty:
        return
    print()
    print("Profile health dashboard / 分类规则健康面板")
    for row in profile_health.head(8).itertuples(index=False):
        print(
            f"- {row.screening_profile} / {row.screening_profile_zh}: "
            f"health={row.profile_health_level_zh}, "
            f"action={row.profile_action_zh}, "
            f"sample={row.sample_count}, "
            f"win_rate={_format_optional_percent(row.win_rate)}, "
            f"avg_return={_format_optional_percent(row.avg_return)}, "
            f"reason={row.profile_reason_zh}"
        )
    print("- Full profile health / 完整规则健康面板: profile_health_dashboard.md")


def _print_profile_action_recommendations(result) -> None:
    recommendations = getattr(result, "profile_action_recommendations", None)
    if recommendations is None or recommendations.empty:
        return
    print()
    print("Profile action recommendations / 分类规则行动建议")
    for row in recommendations.head(8).itertuples(index=False):
        print(
            f"- {row.screening_profile} / {row.screening_profile_zh}: "
            f"{row.recommendation_action_zh}, priority={row.priority_zh}, "
            f"next={row.suggested_next_step_zh}"
        )
    print("- Full profile actions / 完整行动建议: profile_action_recommendations.md")


def _print_profile_blocker_dashboard(result) -> None:
    blockers = getattr(result, "profile_blocker_dashboard", None)
    if blockers is None or blockers.empty:
        return
    print()
    print("Profile blocker dashboard / 分类规则卡点面板")
    for row in blockers.head(8).itertuples(index=False):
        print(
            f"- {row.screening_profile} / {row.screening_profile_zh}: "
            f"#{row.blocker_rank} {row.blocker_zh}, "
            f"count={row.occurrence_count}, "
            f"rate={_format_optional_percent(row.blocker_rate)}, "
            f"fix={row.recommended_fix_zh}"
        )
    print("- Full profile blockers / 完整卡点面板: profile_blocker_dashboard.md")


def _print_historical_win_rate_gate(result) -> None:
    gate = getattr(result, "historical_win_rate_gate", None)
    if gate is None or gate.empty:
        return
    print()
    print("Historical win-rate gate / 历史胜率部署门槛")
    focus = gate[
        gate["scope"].isin(["overall", "validation_bucket", "quality_gate"])
    ].head(5)
    if focus.empty:
        focus = gate.head(5)
    for row in focus.itertuples(index=False):
        print(
            f"- {row.scope}/{row.group_value}: "
            f"action={row.deployment_gate_action_zh}, "
            f"sample={row.sample_count}, "
            f"win_rate={_format_optional_percent(row.win_rate)}, "
            f"avg_return={_format_optional_percent(row.avg_return)}, "
            f"reason={row.deployment_gate_reason_zh}"
        )
    print("- Full gate report / 完整门槛报告: historical_win_rate_gate.md")


def _print_historical_threshold_recommendations(result) -> None:
    recommendations = getattr(result, "historical_threshold_recommendations", None)
    if recommendations is None or recommendations.empty:
        return
    print()
    print("Historical threshold recommendations / 历史阈值建议")
    focus = recommendations.head(8)
    for row in focus.itertuples(index=False):
        if str(row.threshold_attr):
            threshold_text = (
                f"{row.threshold_attr}: "
                f"{_format_optional_number(row.current_threshold)}"
                f"->{_format_optional_number(row.suggested_threshold)}"
            )
        else:
            threshold_text = row.recommendation_action
        print(
            f"- {row.scope}/{row.group_value}: "
            f"{row.recommendation_action_zh}, {threshold_text}, "
            f"priority={row.priority_zh}, reason={row.recommendation_reason_zh}"
        )
    print(
        "- Full recommendations / 完整建议: "
        "historical_threshold_recommendations.md"
    )


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


def _print_walk_forward_progress(ticker: str, status: str, index: int, total: int) -> None:
    status_zh = {
        "started": "开始",
        "completed": "完成",
        "skipped_insufficient_history": "历史数据不足，跳过",
    }.get(status, status)
    print(f"[{index}/{total}] {ticker}: {status} / {status_zh}", flush=True)


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
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "N/A"


def _with_benchmark_tickers(tickers: list[str]) -> list[str]:
    combined = [*tickers, *DEFAULT_BENCHMARK_TICKERS]
    return list(dict.fromkeys(normalize_ticker(ticker) for ticker in combined if ticker.strip()))


def _validation_stem(tickers: list[str], universe_name: str | None) -> str:
    if universe_name:
        return universe_name.lower().strip().replace(" ", "_")
    if len(tickers) <= 6:
        return "_".join(normalize_ticker(ticker) for ticker in tickers)
    return f"{'_'.join(normalize_ticker(ticker) for ticker in tickers[:6])}_{len(tickers)}tickers"


if __name__ == "__main__":
    raise SystemExit(main())
