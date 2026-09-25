"""Entry point: replay historical signals, measure forward returns and write every validation table."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..analysis import analyze_ticker
from ..analysis import normalize_horizons
from ..json_io import dataframe_records
from ..json_io import write_json
from ..real_data import build_single_ticker_scored_frame
from ..real_data import normalize_ticker
from ..screening_config import ScreeningConfig
from ..screening_config import ScreeningThresholds
from ..screening_config import default_screening_config
from ..universe import HistoricalUniverseMembership
from ..universe import survivorship_report_without_historical_membership
from ..win_rate_dashboard import build_historical_threshold_recommendations
from ..win_rate_dashboard import build_historical_win_rate_gate
from ..win_rate_dashboard import build_profile_action_recommendations
from ..win_rate_dashboard import build_profile_blocker_dashboard
from ..win_rate_dashboard import build_profile_health_dashboard
from ..win_rate_dashboard import build_win_rate_dashboard
from ..win_rate_dashboard import render_historical_threshold_recommendations
from ..win_rate_dashboard import render_historical_win_rate_gate
from ..win_rate_dashboard import render_profile_action_recommendations
from ..win_rate_dashboard import render_profile_blocker_dashboard
from ..win_rate_dashboard import render_profile_health_dashboard
from ..win_rate_dashboard import render_win_rate_dashboard
from ..win_rate_dashboard import win_rate_dashboard_payload

from .calibration import (
    build_overfitting_risk_report,
    calibrate_walk_forward_profiles,
    calibrate_walk_forward_rules,
    render_suggested_screening_config,
)
from .config import (
    DEFAULT_FORWARD_WINDOWS,
    WalkForwardProgressCallback,
)
from .events import (
    _apply_pooled_entry_backtest,
    _event_row,
    _membership_record_for_signal,
    _normalize_prices,
    _profile_by_name,
    _target_window,
    _ticker_allows_truncated_forward,
    _validation_analyst_context,
    _validation_event_context,
    _validation_fundamental_context,
    _validation_market_context,
    _validation_relative_strength_contexts,
    _validation_sector_context,
    _validation_sentiment_context,
    _validation_valuation_context,
    build_validation_market_regime_lookup,
)
from .policy import (
    build_benchmark_aware_policy,
    build_benchmark_tightening_recommendations,
    build_minimum_sample_guard,
    build_threshold_sensitivity_grid,
    build_tightening_impact_validation,
)
from .portfolio import (
    build_walk_forward_benchmark_comparison,
    build_walk_forward_portfolio_equity,
    summarize_walk_forward_portfolios,
)
from .report import (
    render_walk_forward_report,
)
from .summaries import (
    build_market_regime_protection_policy,
    build_sample_sufficiency_guidance,
    build_ticker_validation_ranking,
    summarize_market_regime_validation,
    summarize_probability_calibration,
    summarize_walk_forward_events,
    summarize_walk_forward_profiles,
    summarize_walk_forward_segments,
)


@dataclass(frozen=True)
class WalkForwardResult:
    events: pd.DataFrame
    summary: pd.DataFrame
    profile_summary: pd.DataFrame
    segment_summary: pd.DataFrame
    market_regime_summary: pd.DataFrame
    market_regime_policy: pd.DataFrame
    profile_calibration: pd.DataFrame
    calibration: pd.DataFrame
    report: str
    ticker_ranking: pd.DataFrame = field(default_factory=pd.DataFrame)
    sample_sufficiency: pd.DataFrame = field(default_factory=pd.DataFrame)
    probability_calibration: pd.DataFrame = field(default_factory=pd.DataFrame)
    portfolio_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    portfolio_rebalances: pd.DataFrame = field(default_factory=pd.DataFrame)
    portfolio_equity_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    portfolio_equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    benchmark_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    benchmark_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    benchmark_policy: pd.DataFrame = field(default_factory=pd.DataFrame)
    benchmark_tightening: pd.DataFrame = field(default_factory=pd.DataFrame)
    tightening_impact: pd.DataFrame = field(default_factory=pd.DataFrame)
    threshold_sensitivity: pd.DataFrame = field(default_factory=pd.DataFrame)
    minimum_sample_guard: pd.DataFrame = field(default_factory=pd.DataFrame)
    win_rate_dashboard: dict[str, pd.DataFrame] = field(default_factory=dict)
    win_rate_dashboard_report: str = ""
    historical_win_rate_gate: pd.DataFrame = field(default_factory=pd.DataFrame)
    historical_win_rate_gate_report: str = ""
    historical_threshold_recommendations: pd.DataFrame = field(default_factory=pd.DataFrame)
    historical_threshold_recommendations_report: str = ""
    profile_health_dashboard: pd.DataFrame = field(default_factory=pd.DataFrame)
    profile_health_dashboard_report: str = ""
    profile_action_recommendations: pd.DataFrame = field(default_factory=pd.DataFrame)
    profile_action_recommendations_report: str = ""
    profile_blocker_dashboard: pd.DataFrame = field(default_factory=pd.DataFrame)
    profile_blocker_dashboard_report: str = ""
    survivorship_bias_report: dict[str, object] = field(default_factory=dict)


def run_walk_forward_validation(
    prices: pd.DataFrame,
    tickers: Iterable[str],
    horizons: str | Iterable[str] = "all",
    forward_windows: tuple[int, ...] = DEFAULT_FORWARD_WINDOWS,
    step_days: int = 20,
    min_history_days: int = 170,
    output_dir: str | Path | None = None,
    screening_thresholds: ScreeningThresholds | None = None,
    screening_config: ScreeningConfig | None = None,
    screening_profile_name: str | None = None,
    universe_membership: HistoricalUniverseMembership | None = None,
    progress_callback: WalkForwardProgressCallback | None = None,
) -> WalkForwardResult:
    if step_days <= 0:
        raise ValueError("step_days must be positive.")
    if not forward_windows:
        raise ValueError("At least one forward window is required.")

    normalized_tickers = tuple(dict.fromkeys(normalize_ticker(ticker) for ticker in tickers if ticker.strip()))
    if not normalized_tickers:
        raise ValueError("At least one ticker is required.")

    prices = _normalize_prices(prices)
    scored = build_single_ticker_scored_frame(prices)
    selected_horizons = normalize_horizons(horizons)
    config = screening_config or default_screening_config()
    forced_profile = _profile_by_name(config, screening_profile_name) if screening_profile_name else None
    max_forward_window = max(forward_windows)
    market_regime_lookup = build_validation_market_regime_lookup(prices)
    survivorship_bias_report = (
        universe_membership.survivorship_report(list(normalized_tickers))
        if universe_membership is not None
        else survivorship_report_without_historical_membership()
    )
    rows: list[dict[str, object]] = []

    for ticker_index, ticker in enumerate(normalized_tickers, start=1):
        _emit_walk_forward_progress(
            progress_callback,
            ticker,
            "started",
            ticker_index,
            len(normalized_tickers),
        )
        if screening_thresholds is not None:
            profile = None
        elif forced_profile is not None:
            profile = forced_profile
        else:
            profile = config.resolve_profile(ticker=ticker)
        ticker_thresholds = screening_thresholds or profile.thresholds
        ticker_trading_rules = None if profile is None else profile.trading_rules
        profile_name = "custom" if profile is None else profile.name
        profile_name_zh = "自定义规则" if profile is None else profile.name_zh
        ticker_prices = prices[prices["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        allow_truncated_forward = _ticker_allows_truncated_forward(ticker, universe_membership)
        min_required_rows = min_history_days + (1 if allow_truncated_forward else max_forward_window)
        if len(ticker_prices) <= min_required_rows:
            _emit_walk_forward_progress(
                progress_callback,
                ticker,
                "skipped_insufficient_history",
                ticker_index,
                len(normalized_tickers),
            )
            continue
        signal_stop = len(ticker_prices) if allow_truncated_forward else len(ticker_prices) - max_forward_window
        signal_indices = range(min_history_days, signal_stop, step_days)
        for signal_index in signal_indices:
            signal_date = ticker_prices.loc[signal_index, "date"]
            membership_record = _membership_record_for_signal(
                ticker=ticker,
                signal_date=signal_date,
                universe_membership=universe_membership,
            )
            if universe_membership is not None and membership_record is None:
                continue
            try:
                analysis = analyze_ticker(
                    scored=scored,
                    prices=prices,
                    ticker=ticker,
                    horizons=selected_horizons,
                    as_of_date=signal_date,
                    market_context=_validation_market_context(),
                    relative_strength_contexts=_validation_relative_strength_contexts(selected_horizons),
                    event_risk_context=_validation_event_context(),
                    fundamental_context=_validation_fundamental_context(ticker),
                    sentiment_context=_validation_sentiment_context(ticker),
                    analyst_context=_validation_analyst_context(ticker),
                    valuation_context=_validation_valuation_context(ticker),
                    sector_context=_validation_sector_context(),
                    screening_thresholds=ticker_thresholds,
                    trading_rules=ticker_trading_rules,
                )
                analysis["screening_profile"] = profile_name
                analysis["screening_profile_zh"] = profile_name_zh
            except Exception:
                continue
            # One event per horizon row. All horizons of a (date, ticker) share the same
            # forward returns, so pooled counts over-state independent samples by up to
            # 3x (see results/REPORT.md, section 6).
            for row in analysis.itertuples(index=False):
                rows.append(
                    _event_row(
                        ticker_prices=ticker_prices,
                        signal_index=signal_index,
                        analysis_row=row,
                        forward_windows=forward_windows,
                        max_forward_window=max_forward_window,
                        market_regime_lookup=market_regime_lookup,
                        membership_record=membership_record,
                    )
                )
        _emit_walk_forward_progress(
            progress_callback,
            ticker,
            "completed",
            ticker_index,
            len(normalized_tickers),
        )

    events = _apply_pooled_entry_backtest(pd.DataFrame(rows), config)
    summary = summarize_walk_forward_events(events, forward_windows)
    ticker_ranking = build_ticker_validation_ranking(
        events,
        target_window=_target_window(forward_windows),
    )
    sample_sufficiency = build_sample_sufficiency_guidance(
        events=events,
        tickers=normalized_tickers,
        ticker_ranking=ticker_ranking,
        step_days=step_days,
        min_history_days=min_history_days,
    )
    profile_summary = summarize_walk_forward_profiles(events, forward_windows)
    segment_summary = summarize_walk_forward_segments(events, forward_windows)
    market_regime_summary = summarize_market_regime_validation(events, forward_windows)
    market_regime_policy = build_market_regime_protection_policy(market_regime_summary)
    probability_calibration = summarize_probability_calibration(
        events,
        target_window=_target_window(forward_windows),
    )
    portfolio_summary, portfolio_rebalances = summarize_walk_forward_portfolios(
        events,
        target_window=_target_window(forward_windows),
    )
    portfolio_equity_summary, portfolio_equity_curve = build_walk_forward_portfolio_equity(
        prices=prices,
        portfolio_rebalances=portfolio_rebalances,
        target_window=_target_window(forward_windows),
    )
    benchmark_summary, benchmark_curve = build_walk_forward_benchmark_comparison(
        prices=prices,
        portfolio_equity_curve=portfolio_equity_curve,
    )
    benchmark_policy = build_benchmark_aware_policy(benchmark_summary)
    calibration = calibrate_walk_forward_rules(events, target_window=_target_window(forward_windows))
    profile_calibration = calibrate_walk_forward_profiles(
        events=events,
        screening_config=config,
        target_window=_target_window(forward_windows),
    )
    overfitting_risk = build_overfitting_risk_report(profile_calibration)
    benchmark_tightening = build_benchmark_tightening_recommendations(
        events=events,
        screening_config=config,
        profile_calibration=profile_calibration,
        benchmark_policy=benchmark_policy,
        target_window=_target_window(forward_windows),
    )
    tightening_impact = build_tightening_impact_validation(
        events=events,
        benchmark_tightening=benchmark_tightening,
        target_window=_target_window(forward_windows),
    )
    threshold_sensitivity = build_threshold_sensitivity_grid(
        events=events,
        screening_config=config,
        target_window=_target_window(forward_windows),
    )
    minimum_sample_guard = build_minimum_sample_guard(threshold_sensitivity)
    win_rate_dashboard = build_win_rate_dashboard(events, forward_windows)
    win_rate_dashboard_report = render_win_rate_dashboard(win_rate_dashboard)
    profile_health_dashboard = build_profile_health_dashboard(
        events,
        target_window=_target_window(forward_windows),
    )
    profile_health_dashboard_report = render_profile_health_dashboard(profile_health_dashboard)
    profile_action_recommendations = build_profile_action_recommendations(
        profile_health_dashboard
    )
    profile_action_recommendations_report = render_profile_action_recommendations(
        profile_action_recommendations
    )
    profile_blocker_dashboard = build_profile_blocker_dashboard(events)
    profile_blocker_dashboard_report = render_profile_blocker_dashboard(
        profile_blocker_dashboard
    )
    historical_win_rate_gate = build_historical_win_rate_gate(
        events,
        target_window=_target_window(forward_windows),
    )
    historical_win_rate_gate_report = render_historical_win_rate_gate(historical_win_rate_gate)
    historical_threshold_recommendations = build_historical_threshold_recommendations(
        historical_win_rate_gate,
        screening_config=config,
    )
    historical_threshold_recommendations_report = (
        render_historical_threshold_recommendations(historical_threshold_recommendations)
    )
    report = render_walk_forward_report(
        events=events,
        summary=summary,
        ticker_ranking=ticker_ranking,
        sample_sufficiency=sample_sufficiency,
        profile_summary=profile_summary,
        segment_summary=segment_summary,
        market_regime_summary=market_regime_summary,
        market_regime_policy=market_regime_policy,
        profile_calibration=profile_calibration,
        probability_calibration=probability_calibration,
        portfolio_summary=portfolio_summary,
        portfolio_rebalances=portfolio_rebalances,
        portfolio_equity_summary=portfolio_equity_summary,
        portfolio_equity_curve=portfolio_equity_curve,
        benchmark_summary=benchmark_summary,
        benchmark_curve=benchmark_curve,
        benchmark_policy=benchmark_policy,
        benchmark_tightening=benchmark_tightening,
        tightening_impact=tightening_impact,
        threshold_sensitivity=threshold_sensitivity,
        minimum_sample_guard=minimum_sample_guard,
        calibration=calibration,
        forward_windows=forward_windows,
        survivorship_bias_report=survivorship_bias_report,
    )
    report = (
        report.rstrip()
        + "\n\n"
        + win_rate_dashboard_report
        + "\n"
        + profile_health_dashboard_report
        + "\n"
        + profile_action_recommendations_report
        + "\n"
        + profile_blocker_dashboard_report
        + "\n"
        + historical_win_rate_gate_report
        + "\n"
        + historical_threshold_recommendations_report
    )

    result = WalkForwardResult(
            events=events,
            summary=summary,
            ticker_ranking=ticker_ranking,
            sample_sufficiency=sample_sufficiency,
            profile_summary=profile_summary,
            segment_summary=segment_summary,
            market_regime_summary=market_regime_summary,
            market_regime_policy=market_regime_policy,
            profile_calibration=profile_calibration,
            calibration=calibration,
            report=report,
            probability_calibration=probability_calibration,
            portfolio_summary=portfolio_summary,
            portfolio_rebalances=portfolio_rebalances,
            portfolio_equity_summary=portfolio_equity_summary,
            portfolio_equity_curve=portfolio_equity_curve,
            benchmark_summary=benchmark_summary,
            benchmark_curve=benchmark_curve,
            benchmark_policy=benchmark_policy,
            benchmark_tightening=benchmark_tightening,
            tightening_impact=tightening_impact,
            threshold_sensitivity=threshold_sensitivity,
            minimum_sample_guard=minimum_sample_guard,
            win_rate_dashboard=win_rate_dashboard,
            win_rate_dashboard_report=win_rate_dashboard_report,
            historical_win_rate_gate=historical_win_rate_gate,
            historical_win_rate_gate_report=historical_win_rate_gate_report,
            historical_threshold_recommendations=historical_threshold_recommendations,
            historical_threshold_recommendations_report=(
                historical_threshold_recommendations_report
            ),
            profile_health_dashboard=profile_health_dashboard,
            profile_health_dashboard_report=profile_health_dashboard_report,
            profile_action_recommendations=profile_action_recommendations,
            profile_action_recommendations_report=profile_action_recommendations_report,
            profile_blocker_dashboard=profile_blocker_dashboard,
            profile_blocker_dashboard_report=profile_blocker_dashboard_report,
            survivorship_bias_report=survivorship_bias_report,
        )
    if output_dir is not None:
        _write_walk_forward_outputs(
            result,
            output_dir,
            config=config,
            overfitting_risk=overfitting_risk,
            normalized_tickers=normalized_tickers,
            selected_horizons=selected_horizons,
            forward_windows=forward_windows,
            step_days=step_days,
            min_history_days=min_history_days,
            screening_thresholds=screening_thresholds,
        )
    return result


def _write_walk_forward_outputs(
    result: WalkForwardResult,
    output_dir: str | Path,
    *,
    config: ScreeningConfig,
    overfitting_risk: pd.DataFrame,
    normalized_tickers: tuple[str, ...],
    selected_horizons: tuple[str, ...],
    forward_windows: tuple[int, ...],
    step_days: int,
    min_history_days: int,
    screening_thresholds: ScreeningThresholds | None,
) -> None:
    """Write every validation table, report and the run manifest under ``output_dir``."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    result.events.to_csv(path / "walk_forward_events.csv", index=False)
    result.summary.to_csv(path / "walk_forward_summary.csv", index=False)
    result.ticker_ranking.to_csv(path / "ticker_validation_ranking.csv", index=False)
    result.sample_sufficiency.to_csv(path / "sample_sufficiency_guidance.csv", index=False)
    result.profile_summary.to_csv(path / "profile_validation_summary.csv", index=False)
    overfitting_risk.to_csv(path / "overfitting_risk_report.csv", index=False)
    result.segment_summary.to_csv(path / "segment_validation_summary.csv", index=False)
    result.market_regime_summary.to_csv(path / "market_regime_validation_summary.csv", index=False)
    result.market_regime_policy.to_csv(path / "market_regime_policy.csv", index=False)
    result.probability_calibration.to_csv(path / "probability_calibration.csv", index=False)
    result.portfolio_summary.to_csv(path / "portfolio_validation_summary.csv", index=False)
    result.portfolio_rebalances.to_csv(path / "portfolio_rebalances.csv", index=False)
    result.portfolio_equity_summary.to_csv(path / "portfolio_equity_summary.csv", index=False)
    result.portfolio_equity_curve.to_csv(path / "portfolio_equity_curve.csv", index=False)
    result.benchmark_summary.to_csv(path / "benchmark_comparison_summary.csv", index=False)
    result.benchmark_curve.to_csv(path / "benchmark_comparison_curve.csv", index=False)
    result.benchmark_policy.to_csv(path / "benchmark_policy.csv", index=False)
    result.benchmark_tightening.to_csv(path / "benchmark_tightening_recommendations.csv", index=False)
    result.tightening_impact.to_csv(path / "tightening_impact_validation.csv", index=False)
    result.threshold_sensitivity.to_csv(path / "threshold_sensitivity_grid.csv", index=False)
    result.minimum_sample_guard.to_csv(path / "minimum_sample_guard.csv", index=False)
    for name, frame in result.win_rate_dashboard.items():
        frame.to_csv(path / f"win_rate_{name}.csv", index=False)
    result.profile_health_dashboard.to_csv(path / "profile_health_dashboard.csv", index=False)
    result.profile_action_recommendations.to_csv(
        path / "profile_action_recommendations.csv",
        index=False,
    )
    result.profile_blocker_dashboard.to_csv(path / "profile_blocker_dashboard.csv", index=False)
    result.historical_win_rate_gate.to_csv(path / "historical_win_rate_gate.csv", index=False)
    result.historical_threshold_recommendations.to_csv(
        path / "historical_threshold_recommendations.csv",
        index=False,
    )
    result.profile_calibration.to_csv(path / "profile_rule_calibration.csv", index=False)
    result.calibration.to_csv(path / "rule_calibration.csv", index=False)
    suggested_config_text = render_suggested_screening_config(
        screening_config=config,
        profile_calibration=result.profile_calibration,
        benchmark_policy=result.benchmark_policy,
        benchmark_tightening=result.benchmark_tightening,
        minimum_sample_guard=result.minimum_sample_guard,
        historical_threshold_recommendations=result.historical_threshold_recommendations,
    )
    (path / "suggested_screening.toml").write_text(
        suggested_config_text,
        encoding="utf-8",
    )
    (path / "walk_forward_report.md").write_text(result.report, encoding="utf-8")
    (path / "win_rate_dashboard.md").write_text(
        result.win_rate_dashboard_report,
        encoding="utf-8",
    )
    write_json(
        path / "win_rate_dashboard.json",
        win_rate_dashboard_payload(result.win_rate_dashboard),
    )
    (path / "profile_health_dashboard.md").write_text(
        result.profile_health_dashboard_report,
        encoding="utf-8",
    )
    write_json(
        path / "profile_health_dashboard.json",
        {"rows": dataframe_records(result.profile_health_dashboard)},
    )
    (path / "profile_action_recommendations.md").write_text(
        result.profile_action_recommendations_report,
        encoding="utf-8",
    )
    write_json(
        path / "profile_action_recommendations.json",
        {"rows": dataframe_records(result.profile_action_recommendations)},
    )
    (path / "profile_blocker_dashboard.md").write_text(
        result.profile_blocker_dashboard_report,
        encoding="utf-8",
    )
    write_json(
        path / "profile_blocker_dashboard.json",
        {"rows": dataframe_records(result.profile_blocker_dashboard)},
    )
    (path / "historical_win_rate_gate.md").write_text(
        result.historical_win_rate_gate_report,
        encoding="utf-8",
    )
    write_json(
        path / "historical_win_rate_gate.json",
        {"rows": dataframe_records(result.historical_win_rate_gate)},
    )
    (path / "historical_threshold_recommendations.md").write_text(
        result.historical_threshold_recommendations_report,
        encoding="utf-8",
    )
    write_json(
        path / "historical_threshold_recommendations.json",
        {"rows": dataframe_records(result.historical_threshold_recommendations)},
    )
    write_json(
        path / "validation_result.json",
        {
            "tickers": list(normalized_tickers),
            "horizons": list(selected_horizons),
            "forward_windows": list(forward_windows),
            "step_days": step_days,
            "min_history_days": min_history_days,
            "event_count": len(result.events),
            "events": dataframe_records(result.events),
            "summary": dataframe_records(result.summary),
            "ticker_ranking": dataframe_records(result.ticker_ranking),
            "sample_sufficiency": dataframe_records(result.sample_sufficiency),
            "profile_summary": dataframe_records(result.profile_summary),
            "segment_summary": dataframe_records(result.segment_summary),
            "market_regime_summary": dataframe_records(result.market_regime_summary),
            "market_regime_policy": dataframe_records(result.market_regime_policy),
            "probability_calibration": dataframe_records(result.probability_calibration),
            "portfolio_summary": dataframe_records(result.portfolio_summary),
            "portfolio_rebalances": dataframe_records(result.portfolio_rebalances),
            "portfolio_equity_summary": dataframe_records(result.portfolio_equity_summary),
            "portfolio_equity_curve": dataframe_records(result.portfolio_equity_curve),
            "benchmark_summary": dataframe_records(result.benchmark_summary),
            "benchmark_curve": dataframe_records(result.benchmark_curve),
            "benchmark_policy": dataframe_records(result.benchmark_policy),
            "benchmark_tightening": dataframe_records(result.benchmark_tightening),
            "tightening_impact": dataframe_records(result.tightening_impact),
            "threshold_sensitivity": dataframe_records(result.threshold_sensitivity),
            "minimum_sample_guard": dataframe_records(result.minimum_sample_guard),
            "win_rate_dashboard": win_rate_dashboard_payload(result.win_rate_dashboard),
            "profile_health_dashboard": dataframe_records(result.profile_health_dashboard),
            "profile_action_recommendations": dataframe_records(
                result.profile_action_recommendations
            ),
            "profile_blocker_dashboard": dataframe_records(result.profile_blocker_dashboard),
            "historical_win_rate_gate": dataframe_records(result.historical_win_rate_gate),
            "historical_threshold_recommendations": dataframe_records(
                result.historical_threshold_recommendations
            ),
            "profile_calibration": dataframe_records(result.profile_calibration),
            "calibration": dataframe_records(result.calibration),
            "screening_thresholds": (screening_thresholds or ScreeningThresholds()).to_dict(),
            "screening_config": config.to_dict(),
            "survivorship_bias_report": result.survivorship_bias_report,
            "output_files": {
                "markdown_report": str(path / "walk_forward_report.md"),
                "events_csv": str(path / "walk_forward_events.csv"),
                "summary_csv": str(path / "walk_forward_summary.csv"),
                "ticker_ranking_csv": str(path / "ticker_validation_ranking.csv"),
                "sample_sufficiency_csv": str(path / "sample_sufficiency_guidance.csv"),
                "profile_summary_csv": str(path / "profile_validation_summary.csv"),
                "segment_summary_csv": str(path / "segment_validation_summary.csv"),
                "market_regime_summary_csv": str(
                    path / "market_regime_validation_summary.csv"
                ),
                "market_regime_policy_csv": str(path / "market_regime_policy.csv"),
                "probability_calibration_csv": str(path / "probability_calibration.csv"),
                "portfolio_summary_csv": str(path / "portfolio_validation_summary.csv"),
                "portfolio_rebalances_csv": str(path / "portfolio_rebalances.csv"),
                "portfolio_equity_summary_csv": str(path / "portfolio_equity_summary.csv"),
                "portfolio_equity_curve_csv": str(path / "portfolio_equity_curve.csv"),
                "benchmark_summary_csv": str(path / "benchmark_comparison_summary.csv"),
                "benchmark_curve_csv": str(path / "benchmark_comparison_curve.csv"),
                "benchmark_policy_csv": str(path / "benchmark_policy.csv"),
                "benchmark_tightening_csv": str(
                    path / "benchmark_tightening_recommendations.csv"
                ),
                "tightening_impact_csv": str(path / "tightening_impact_validation.csv"),
                "threshold_sensitivity_csv": str(path / "threshold_sensitivity_grid.csv"),
                "minimum_sample_guard_csv": str(path / "minimum_sample_guard.csv"),
                "win_rate_dashboard_md": str(path / "win_rate_dashboard.md"),
                "win_rate_dashboard_json": str(path / "win_rate_dashboard.json"),
                "win_rate_overall_csv": str(path / "win_rate_overall.csv"),
                "win_rate_by_horizon_csv": str(path / "win_rate_by_horizon.csv"),
                "win_rate_by_entry_type_csv": str(path / "win_rate_by_entry_type.csv"),
                "win_rate_by_profile_csv": str(path / "win_rate_by_profile.csv"),
                "win_rate_by_quality_gate_csv": str(
                    path / "win_rate_by_quality_gate.csv"
                ),
                "profile_health_dashboard_md": str(
                    path / "profile_health_dashboard.md"
                ),
                "profile_health_dashboard_json": str(
                    path / "profile_health_dashboard.json"
                ),
                "profile_health_dashboard_csv": str(
                    path / "profile_health_dashboard.csv"
                ),
                "profile_action_recommendations_md": str(
                    path / "profile_action_recommendations.md"
                ),
                "profile_action_recommendations_json": str(
                    path / "profile_action_recommendations.json"
                ),
                "profile_action_recommendations_csv": str(
                    path / "profile_action_recommendations.csv"
                ),
                "profile_blocker_dashboard_md": str(
                    path / "profile_blocker_dashboard.md"
                ),
                "profile_blocker_dashboard_json": str(
                    path / "profile_blocker_dashboard.json"
                ),
                "profile_blocker_dashboard_csv": str(
                    path / "profile_blocker_dashboard.csv"
                ),
                "historical_win_rate_gate_md": str(
                    path / "historical_win_rate_gate.md"
                ),
                "historical_win_rate_gate_json": str(
                    path / "historical_win_rate_gate.json"
                ),
                "historical_win_rate_gate_csv": str(
                    path / "historical_win_rate_gate.csv"
                ),
                "historical_threshold_recommendations_md": str(
                    path / "historical_threshold_recommendations.md"
                ),
                "historical_threshold_recommendations_json": str(
                    path / "historical_threshold_recommendations.json"
                ),
                "historical_threshold_recommendations_csv": str(
                    path / "historical_threshold_recommendations.csv"
                ),
                "profile_calibration_csv": str(path / "profile_rule_calibration.csv"),
                "suggested_screening_toml": str(path / "suggested_screening.toml"),
                "calibration_csv": str(path / "rule_calibration.csv"),
                "json": str(path / "validation_result.json"),
            },
        },
    )


def _emit_walk_forward_progress(
    progress_callback: WalkForwardProgressCallback | None,
    ticker: str,
    status: str,
    index: int,
    total: int,
) -> None:
    if progress_callback is not None:
        progress_callback(ticker, status, index, total)
