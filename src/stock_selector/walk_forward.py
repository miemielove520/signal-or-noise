from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .analysis import (
    HORIZON_SPECS,
    analyze_ticker,
    normalize_horizons,
)
from .analysis.screening import (
    _entry_evidence_profile,
)
from .json_io import dataframe_records, write_json
from .real_data import build_single_ticker_scored_frame, normalize_ticker
from .screening_config import (
    ScreeningConfig,
    ScreeningThresholds,
    default_screening_config,
    render_screening_config_toml,
)
from .universe import HistoricalUniverseMembership, survivorship_report_without_historical_membership
from .win_rate_dashboard import (
    build_profile_action_recommendations,
    build_profile_blocker_dashboard,
    build_historical_threshold_recommendations,
    build_profile_health_dashboard,
    build_win_rate_dashboard,
    build_historical_win_rate_gate,
    render_profile_action_recommendations,
    render_profile_blocker_dashboard,
    render_historical_threshold_recommendations,
    render_historical_win_rate_gate,
    render_profile_health_dashboard,
    render_win_rate_dashboard,
    win_rate_dashboard_payload,
)


DEFAULT_FORWARD_WINDOWS = (5, 20, 60)
DEFAULT_SIGNAL_THRESHOLDS = (55, 60, 65, 70, 75)
DEFAULT_HIGH_PROBABILITY_THRESHOLDS = (50, 55, 60, 65, 70)
DEFAULT_PROBABILITY_THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70)
DEFAULT_RELATIVE_STRENGTH_THRESHOLDS = (40, 45, 50, 55, 60)
DEFAULT_MARKET_THRESHOLDS = (50, 55, 60, 65, 70)
DEFAULT_BENCHMARK_TICKERS = ("SPY", "QQQ")

# Minimum completed walk-forward samples before a calibrated threshold is allowed to
# be WRITTEN into an adoptable config. Calibrating a trading threshold on 5-10 samples
# is statistically meaningless (huge confidence interval) and produces parameters that
# look smart but are overfit. 30 is a conservative floor for a system headed to live money.
MIN_CALIBRATION_SAMPLE_COUNT = 30

WalkForwardProgressCallback = Callable[[str, str, int, int], None]


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

    if output_dir is not None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        events.to_csv(path / "walk_forward_events.csv", index=False)
        summary.to_csv(path / "walk_forward_summary.csv", index=False)
        ticker_ranking.to_csv(path / "ticker_validation_ranking.csv", index=False)
        sample_sufficiency.to_csv(path / "sample_sufficiency_guidance.csv", index=False)
        profile_summary.to_csv(path / "profile_validation_summary.csv", index=False)
        overfitting_risk.to_csv(path / "overfitting_risk_report.csv", index=False)
        segment_summary.to_csv(path / "segment_validation_summary.csv", index=False)
        market_regime_summary.to_csv(path / "market_regime_validation_summary.csv", index=False)
        market_regime_policy.to_csv(path / "market_regime_policy.csv", index=False)
        probability_calibration.to_csv(path / "probability_calibration.csv", index=False)
        portfolio_summary.to_csv(path / "portfolio_validation_summary.csv", index=False)
        portfolio_rebalances.to_csv(path / "portfolio_rebalances.csv", index=False)
        portfolio_equity_summary.to_csv(path / "portfolio_equity_summary.csv", index=False)
        portfolio_equity_curve.to_csv(path / "portfolio_equity_curve.csv", index=False)
        benchmark_summary.to_csv(path / "benchmark_comparison_summary.csv", index=False)
        benchmark_curve.to_csv(path / "benchmark_comparison_curve.csv", index=False)
        benchmark_policy.to_csv(path / "benchmark_policy.csv", index=False)
        benchmark_tightening.to_csv(path / "benchmark_tightening_recommendations.csv", index=False)
        tightening_impact.to_csv(path / "tightening_impact_validation.csv", index=False)
        threshold_sensitivity.to_csv(path / "threshold_sensitivity_grid.csv", index=False)
        minimum_sample_guard.to_csv(path / "minimum_sample_guard.csv", index=False)
        for name, frame in win_rate_dashboard.items():
            frame.to_csv(path / f"win_rate_{name}.csv", index=False)
        profile_health_dashboard.to_csv(path / "profile_health_dashboard.csv", index=False)
        profile_action_recommendations.to_csv(
            path / "profile_action_recommendations.csv",
            index=False,
        )
        profile_blocker_dashboard.to_csv(path / "profile_blocker_dashboard.csv", index=False)
        historical_win_rate_gate.to_csv(path / "historical_win_rate_gate.csv", index=False)
        historical_threshold_recommendations.to_csv(
            path / "historical_threshold_recommendations.csv",
            index=False,
        )
        profile_calibration.to_csv(path / "profile_rule_calibration.csv", index=False)
        calibration.to_csv(path / "rule_calibration.csv", index=False)
        suggested_config_text = render_suggested_screening_config(
            screening_config=config,
            profile_calibration=profile_calibration,
            benchmark_policy=benchmark_policy,
            benchmark_tightening=benchmark_tightening,
            minimum_sample_guard=minimum_sample_guard,
            historical_threshold_recommendations=historical_threshold_recommendations,
        )
        (path / "suggested_screening.toml").write_text(
            suggested_config_text,
            encoding="utf-8",
        )
        (path / "walk_forward_report.md").write_text(report, encoding="utf-8")
        (path / "win_rate_dashboard.md").write_text(
            win_rate_dashboard_report,
            encoding="utf-8",
        )
        write_json(
            path / "win_rate_dashboard.json",
            win_rate_dashboard_payload(win_rate_dashboard),
        )
        (path / "profile_health_dashboard.md").write_text(
            profile_health_dashboard_report,
            encoding="utf-8",
        )
        write_json(
            path / "profile_health_dashboard.json",
            {"rows": dataframe_records(profile_health_dashboard)},
        )
        (path / "profile_action_recommendations.md").write_text(
            profile_action_recommendations_report,
            encoding="utf-8",
        )
        write_json(
            path / "profile_action_recommendations.json",
            {"rows": dataframe_records(profile_action_recommendations)},
        )
        (path / "profile_blocker_dashboard.md").write_text(
            profile_blocker_dashboard_report,
            encoding="utf-8",
        )
        write_json(
            path / "profile_blocker_dashboard.json",
            {"rows": dataframe_records(profile_blocker_dashboard)},
        )
        (path / "historical_win_rate_gate.md").write_text(
            historical_win_rate_gate_report,
            encoding="utf-8",
        )
        write_json(
            path / "historical_win_rate_gate.json",
            {"rows": dataframe_records(historical_win_rate_gate)},
        )
        (path / "historical_threshold_recommendations.md").write_text(
            historical_threshold_recommendations_report,
            encoding="utf-8",
        )
        write_json(
            path / "historical_threshold_recommendations.json",
            {"rows": dataframe_records(historical_threshold_recommendations)},
        )
        write_json(
            path / "validation_result.json",
            {
                "tickers": list(normalized_tickers),
                "horizons": list(selected_horizons),
                "forward_windows": list(forward_windows),
                "step_days": step_days,
                "min_history_days": min_history_days,
                "event_count": len(events),
                "events": dataframe_records(events),
                "summary": dataframe_records(summary),
                "ticker_ranking": dataframe_records(ticker_ranking),
                "sample_sufficiency": dataframe_records(sample_sufficiency),
                "profile_summary": dataframe_records(profile_summary),
                "segment_summary": dataframe_records(segment_summary),
                "market_regime_summary": dataframe_records(market_regime_summary),
                "market_regime_policy": dataframe_records(market_regime_policy),
                "probability_calibration": dataframe_records(probability_calibration),
                "portfolio_summary": dataframe_records(portfolio_summary),
                "portfolio_rebalances": dataframe_records(portfolio_rebalances),
                "portfolio_equity_summary": dataframe_records(portfolio_equity_summary),
                "portfolio_equity_curve": dataframe_records(portfolio_equity_curve),
                "benchmark_summary": dataframe_records(benchmark_summary),
                "benchmark_curve": dataframe_records(benchmark_curve),
                "benchmark_policy": dataframe_records(benchmark_policy),
                "benchmark_tightening": dataframe_records(benchmark_tightening),
                "tightening_impact": dataframe_records(tightening_impact),
                "threshold_sensitivity": dataframe_records(threshold_sensitivity),
                "minimum_sample_guard": dataframe_records(minimum_sample_guard),
                "win_rate_dashboard": win_rate_dashboard_payload(win_rate_dashboard),
                "profile_health_dashboard": dataframe_records(profile_health_dashboard),
                "profile_action_recommendations": dataframe_records(
                    profile_action_recommendations
                ),
                "profile_blocker_dashboard": dataframe_records(profile_blocker_dashboard),
                "historical_win_rate_gate": dataframe_records(historical_win_rate_gate),
                "historical_threshold_recommendations": dataframe_records(
                    historical_threshold_recommendations
                ),
                "profile_calibration": dataframe_records(profile_calibration),
                "calibration": dataframe_records(calibration),
                "screening_thresholds": (screening_thresholds or ScreeningThresholds()).to_dict(),
                "screening_config": config.to_dict(),
                "survivorship_bias_report": survivorship_bias_report,
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

    return WalkForwardResult(
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


def _emit_walk_forward_progress(
    progress_callback: WalkForwardProgressCallback | None,
    ticker: str,
    status: str,
    index: int,
    total: int,
) -> None:
    if progress_callback is not None:
        progress_callback(ticker, status, index, total)


def summarize_walk_forward_events(
    events: pd.DataFrame,
    forward_windows: tuple[int, ...] = DEFAULT_FORWARD_WINDOWS,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["bucket", "sample_count"])

    rows: list[dict[str, object]] = []
    for bucket, group in events.groupby("validation_bucket", dropna=False):
        row: dict[str, object] = {"bucket": bucket, "sample_count": len(group)}
        for window in forward_windows:
            returns = group[f"forward_return_{window}d"].dropna()
            row[f"win_rate_{window}d"] = float((returns > 0).mean()) if len(returns) else np.nan
            row[f"avg_return_{window}d"] = float(returns.mean()) if len(returns) else np.nan
            row[f"median_return_{window}d"] = float(returns.median()) if len(returns) else np.nan
            row[f"worst_return_{window}d"] = float(returns.min()) if len(returns) else np.nan
        drawdowns = group["max_drawdown_after_signal"].dropna()
        row["avg_max_drawdown_after_signal"] = float(drawdowns.mean()) if len(drawdowns) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("bucket").reset_index(drop=True)


def build_ticker_validation_ranking(
    events: pd.DataFrame,
    target_window: int = 20,
) -> pd.DataFrame:
    columns = [
        "rank",
        "ticker",
        "ticker_decision",
        "ticker_decision_zh",
        "ticker_ranking_score",
        "sample_count",
        "high_probability_sample_count",
        "watchlist_or_better_sample_count",
        f"win_rate_{target_window}d",
        f"avg_return_{target_window}d",
        f"median_return_{target_window}d",
        f"worst_return_{target_window}d",
        "avg_max_drawdown_after_signal",
        "quality_gate_pass_rate",
        "avg_signal_score",
        "avg_confidence_score",
        "avg_high_probability_score",
        "avg_calibrated_win_probability",
        "ticker_reason",
        "ticker_reason_zh",
    ]
    return_column = f"forward_return_{target_window}d"
    required = {"ticker", "validation_bucket", return_column}
    if events.empty or not required.issubset(events.columns):
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for ticker, group in events.groupby("ticker", dropna=False):
        returns = pd.to_numeric(group[return_column], errors="coerce").dropna()
        sample_count = len(returns)
        high_probability_sample_count = int(
            (group["validation_bucket"] == "high_probability").sum()
        )
        watchlist_or_better_sample_count = int(
            group["validation_bucket"].isin(["high_probability", "near_watchlist"]).sum()
        )
        avg_drawdown = _numeric_mean(group, "max_drawdown_after_signal")
        win_rate = float((returns > 0).mean()) if sample_count else np.nan
        avg_return = float(returns.mean()) if sample_count else np.nan
        row = {
            "rank": 0,
            "ticker": str(ticker),
            "_target_window": target_window,
            "ticker_decision": "",
            "ticker_decision_zh": "",
            "ticker_ranking_score": 0.0,
            "sample_count": sample_count,
            "high_probability_sample_count": high_probability_sample_count,
            "watchlist_or_better_sample_count": watchlist_or_better_sample_count,
            f"win_rate_{target_window}d": win_rate,
            f"avg_return_{target_window}d": avg_return,
            f"median_return_{target_window}d": float(returns.median()) if sample_count else np.nan,
            f"worst_return_{target_window}d": float(returns.min()) if sample_count else np.nan,
            "avg_max_drawdown_after_signal": avg_drawdown,
            "quality_gate_pass_rate": _numeric_mean(group, "quality_gate_passed"),
            "avg_signal_score": _numeric_mean(group, "signal_score"),
            "avg_confidence_score": _numeric_mean(group, "confidence_score"),
            "avg_high_probability_score": _numeric_mean(group, "high_probability_score"),
            "avg_calibrated_win_probability": _numeric_mean(
                group,
                "calibrated_win_probability",
            ),
        }
        score = _ticker_ranking_score(row)
        decision, decision_zh = _ticker_ranking_decision(row, score)
        reason, reason_zh = _ticker_ranking_reason(row, score, target_window)
        row["ticker_ranking_score"] = score
        row["ticker_decision"] = decision
        row["ticker_decision_zh"] = decision_zh
        row["ticker_reason"] = reason
        row["ticker_reason_zh"] = reason_zh
        rows.append(row)

    ranking = pd.DataFrame(rows)
    decision_rank = {
        "priority_candidate": 0,
        "watchlist_candidate": 1,
        "research_only": 2,
        "need_more_samples": 3,
        "deprioritize": 4,
    }
    ranking["_decision_rank"] = ranking["ticker_decision"].map(decision_rank).fillna(9)
    ranking = (
        ranking.sort_values(
            ["_decision_rank", "ticker_ranking_score", "sample_count"],
            ascending=[True, False, False],
            na_position="last",
        )
        .drop(columns=["_decision_rank"])
        .reset_index(drop=True)
    )
    ranking["rank"] = range(1, len(ranking) + 1)
    return ranking[columns]


def _ticker_ranking_score(row: dict[str, object]) -> float:
    target_window = _safe_int(row.get("_target_window")) or 20
    sample_count = _safe_int(row.get("sample_count"))
    high_probability_sample_count = _safe_int(row.get("high_probability_sample_count"))
    watchlist_or_better_sample_count = _safe_int(row.get("watchlist_or_better_sample_count"))
    win_rate = _safe_float(row.get(f"win_rate_{target_window}d"))
    avg_return = _safe_float(row.get(f"avg_return_{target_window}d"))
    avg_drawdown = _safe_float(row.get("avg_max_drawdown_after_signal"))
    quality_gate_pass_rate = _safe_float(row.get("quality_gate_pass_rate"))
    avg_signal_score = _safe_float(row.get("avg_signal_score"))
    avg_confidence_score = _safe_float(row.get("avg_confidence_score"))
    avg_high_probability_score = _safe_float(row.get("avg_high_probability_score"))
    avg_calibrated_win_probability = _safe_float(row.get("avg_calibrated_win_probability"))

    if sample_count <= 0:
        return 0.0

    score = 35.0
    score += min(sample_count / 20.0, 1.0) * 10.0
    score += min(high_probability_sample_count / 5.0, 1.0) * 10.0
    score += min(watchlist_or_better_sample_count / 10.0, 1.0) * 5.0

    if _is_finite(win_rate):
        score += (win_rate - 0.50) * 55.0
    if _is_finite(avg_return):
        score += avg_return * 180.0
    if _is_finite(avg_drawdown):
        score -= min(abs(avg_drawdown), 0.30) * 35.0
    if _is_finite(quality_gate_pass_rate):
        score += quality_gate_pass_rate * 8.0
    if _is_finite(avg_signal_score):
        score += (avg_signal_score - 60.0) * 0.15
    if _is_finite(avg_confidence_score):
        score += (avg_confidence_score - 60.0) * 0.12
    if _is_finite(avg_high_probability_score):
        score += (avg_high_probability_score - 60.0) * 0.12
    if _is_finite(avg_calibrated_win_probability):
        score += (avg_calibrated_win_probability - 0.50) * 25.0

    if sample_count < 5:
        score = min(score, 50.0)

    return round(_clamp(score, 0.0, 100.0), 2)


def _ticker_ranking_decision(
    row: dict[str, object],
    score: float,
) -> tuple[str, str]:
    target_window = _safe_int(row.get("_target_window")) or 20
    sample_count = _safe_int(row.get("sample_count"))
    high_probability_sample_count = _safe_int(row.get("high_probability_sample_count"))
    win_rate = _safe_float(row.get(f"win_rate_{target_window}d"))
    avg_return = _safe_float(row.get(f"avg_return_{target_window}d"))

    if sample_count < 5:
        return "need_more_samples", "样本不足"
    if (
        score >= 72
        and high_probability_sample_count >= 3
        and _is_finite(win_rate)
        and win_rate >= 0.55
        and _is_finite(avg_return)
        and avg_return > 0
    ):
        return "priority_candidate", "优先候选"
    if score >= 58 and _is_finite(avg_return) and avg_return > 0:
        return "watchlist_candidate", "观察候选"
    if score >= 45:
        return "research_only", "仅研究"
    return "deprioritize", "降低优先级"


def _ticker_ranking_reason(
    row: dict[str, object],
    score: float,
    target_window: int,
) -> tuple[str, str]:
    sample_count = _safe_int(row.get("sample_count"))
    high_probability_sample_count = _safe_int(row.get("high_probability_sample_count"))
    win_rate_text = _format_percent(row.get(f"win_rate_{target_window}d"))
    avg_return_text = _format_percent(row.get(f"avg_return_{target_window}d"))
    drawdown_text = _format_percent(row.get("avg_max_drawdown_after_signal"))
    reason = (
        f"Samples={sample_count}, high-probability samples={high_probability_sample_count}, "
        f"{target_window}d win rate={win_rate_text}, avg return={avg_return_text}, "
        f"avg drawdown={drawdown_text}, ranking score={score:.2f}."
    )
    reason_zh = (
        f"样本={sample_count}，高概率样本={high_probability_sample_count}，"
        f"{target_window}日胜率={win_rate_text}，平均收益={avg_return_text}，"
        f"平均回撤={drawdown_text}，排名分数={score:.2f}。"
    )
    return reason, reason_zh


def build_sample_sufficiency_guidance(
    events: pd.DataFrame,
    tickers: Iterable[str],
    ticker_ranking: pd.DataFrame | None = None,
    step_days: int = 20,
    min_history_days: int = 170,
    min_total_samples: int | None = None,
    min_ticker_samples: int = 10,
) -> pd.DataFrame:
    columns = [
        "scope",
        "ticker",
        "sample_count",
        "target_sample_count",
        "sample_status",
        "sample_status_zh",
        "guidance_action",
        "guidance_action_zh",
        "current_step_days",
        "suggested_step_days",
        "current_min_history_days",
        "suggested_min_history_days",
        "suggested_period",
        "suggested_command_hint",
        "suggested_command_hint_zh",
        "reason",
        "reason_zh",
    ]
    normalized_tickers = [normalize_ticker(ticker) for ticker in tickers if str(ticker).strip()]
    target_total = min_total_samples or max(30, len(normalized_tickers) * 5)
    ranking = ticker_ranking if ticker_ranking is not None else build_ticker_validation_ranking(events)

    rows: list[dict[str, object]] = []
    total_sample_count = int(len(events))
    insufficient_ticker_count = 0
    if not ranking.empty and "sample_count" in ranking.columns:
        insufficient_ticker_count = int((ranking["sample_count"] < min_ticker_samples).sum())
    missing_tickers = set(normalized_tickers)
    if not ranking.empty and "ticker" in ranking.columns:
        missing_tickers -= set(ranking["ticker"].astype(str).str.upper())
    insufficient_ticker_count += len(missing_tickers)

    rows.append(
        _sample_guidance_row(
            scope="overall",
            ticker="ALL",
            sample_count=total_sample_count,
            target_sample_count=target_total,
            step_days=step_days,
            min_history_days=min_history_days,
            reason_suffix=(
                f"{insufficient_ticker_count} tickers are below the ticker-level sample target."
                if insufficient_ticker_count
                else "Ticker-level sample coverage is acceptable."
            ),
            reason_suffix_zh=(
                f"{insufficient_ticker_count} 只股票低于个股样本目标。"
                if insufficient_ticker_count
                else "个股层面的样本覆盖可以接受。"
            ),
        )
    )

    if not ranking.empty:
        for row in ranking.itertuples(index=False):
            sample_count = _safe_int(getattr(row, "sample_count", 0))
            if sample_count >= min_ticker_samples:
                continue
            rows.append(
                _sample_guidance_row(
                    scope="ticker",
                    ticker=str(getattr(row, "ticker", "")),
                    sample_count=sample_count,
                    target_sample_count=min_ticker_samples,
                    step_days=step_days,
                    min_history_days=min_history_days,
                    reason_suffix="Ticker ranking should not be trusted until more signals are collected.",
                    reason_suffix_zh="收集更多信号前，不应过度信任这只股票的排名。",
                )
            )

    for ticker in sorted(missing_tickers):
        rows.append(
            _sample_guidance_row(
                scope="ticker",
                ticker=ticker,
                sample_count=0,
                target_sample_count=min_ticker_samples,
                step_days=step_days,
                min_history_days=min_history_days,
                reason_suffix="No validation signals were generated for this ticker.",
                reason_suffix_zh="这只股票没有生成验证信号。",
            )
        )

    return pd.DataFrame(rows, columns=columns)


def _sample_guidance_row(
    scope: str,
    ticker: str,
    sample_count: int,
    target_sample_count: int,
    step_days: int,
    min_history_days: int,
    reason_suffix: str,
    reason_suffix_zh: str,
) -> dict[str, object]:
    status, status_zh = _sample_status(sample_count, target_sample_count)
    suggested_step_days = _suggested_step_days(step_days, sample_count, target_sample_count)
    suggested_min_history_days = _suggested_min_history_days(min_history_days, sample_count, target_sample_count)
    suggested_period = _suggested_period(sample_count, target_sample_count)
    if sample_count >= target_sample_count:
        action = "keep_settings"
        action_zh = "保持当前设置"
    elif sample_count == 0:
        action = "expand_history_and_lower_step_days"
        action_zh = "扩大历史数据并降低信号间隔"
    else:
        action = "increase_history_or_frequency"
        action_zh = "增加历史周期或提高取样频率"

    command_hint = (
        f"--period {suggested_period} --step-days {suggested_step_days} "
        f"--min-history-days {suggested_min_history_days}"
    )
    reason = (
        f"{scope} sample count is {sample_count}, target is {target_sample_count}. "
        f"{reason_suffix}"
    )
    reason_zh = (
        f"{scope} 当前样本数为 {sample_count}，目标样本数为 {target_sample_count}。"
        f"{reason_suffix_zh}"
    )
    return {
        "scope": scope,
        "ticker": ticker,
        "sample_count": sample_count,
        "target_sample_count": target_sample_count,
        "sample_status": status,
        "sample_status_zh": status_zh,
        "guidance_action": action,
        "guidance_action_zh": action_zh,
        "current_step_days": step_days,
        "suggested_step_days": suggested_step_days,
        "current_min_history_days": min_history_days,
        "suggested_min_history_days": suggested_min_history_days,
        "suggested_period": suggested_period,
        "suggested_command_hint": command_hint,
        "suggested_command_hint_zh": f"建议参数：{command_hint}",
        "reason": reason,
        "reason_zh": reason_zh,
    }


def _sample_status(sample_count: int, target_sample_count: int) -> tuple[str, str]:
    if sample_count <= 0:
        return "no_samples", "没有样本"
    if sample_count < target_sample_count:
        return "thin_samples", "样本偏少"
    return "enough_samples", "样本充足"


def _suggested_step_days(step_days: int, sample_count: int, target_sample_count: int) -> int:
    if sample_count >= target_sample_count:
        return step_days
    if step_days > 60:
        return 20
    if step_days > 20:
        return max(20, step_days // 2)
    return step_days


def _suggested_min_history_days(
    min_history_days: int,
    sample_count: int,
    target_sample_count: int,
) -> int:
    if sample_count >= target_sample_count:
        return min_history_days
    if min_history_days > 170:
        return 170
    if min_history_days > 120:
        return 120
    return min_history_days


def _suggested_period(sample_count: int, target_sample_count: int) -> str:
    if sample_count >= target_sample_count:
        return "current"
    return "5y"


def summarize_walk_forward_profiles(
    events: pd.DataFrame,
    forward_windows: tuple[int, ...] = DEFAULT_FORWARD_WINDOWS,
) -> pd.DataFrame:
    columns = [
        "screening_profile",
        "screening_profile_zh",
        "validation_bucket",
        "sample_count",
    ]
    if events.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    group_columns = ["screening_profile", "screening_profile_zh", "validation_bucket"]
    for keys, group in events.groupby(group_columns, dropna=False):
        profile, profile_zh, bucket = keys
        row: dict[str, object] = {
            "screening_profile": profile,
            "screening_profile_zh": profile_zh,
            "validation_bucket": bucket,
            "sample_count": len(group),
            "quality_gate_pass_rate": float(group["quality_gate_passed"].mean())
            if "quality_gate_passed" in group
            else np.nan,
            "avg_high_probability_score": float(group["high_probability_score"].mean())
            if "high_probability_score" in group
            else np.nan,
            "avg_calibrated_win_probability": float(
                group["calibrated_win_probability"].mean()
            )
            if "calibrated_win_probability" in group
            else np.nan,
        }
        for window in forward_windows:
            returns = group[f"forward_return_{window}d"].dropna()
            row[f"win_rate_{window}d"] = float((returns > 0).mean()) if len(returns) else np.nan
            row[f"avg_return_{window}d"] = float(returns.mean()) if len(returns) else np.nan
            row[f"median_return_{window}d"] = float(returns.median()) if len(returns) else np.nan
            row[f"worst_return_{window}d"] = float(returns.min()) if len(returns) else np.nan
        drawdowns = group["max_drawdown_after_signal"].dropna()
        row["avg_max_drawdown_after_signal"] = float(drawdowns.mean()) if len(drawdowns) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["screening_profile", "validation_bucket"]
    ).reset_index(drop=True)


def summarize_walk_forward_segments(
    events: pd.DataFrame,
    forward_windows: tuple[int, ...] = DEFAULT_FORWARD_WINDOWS,
) -> pd.DataFrame:
    columns = [
        "validation_segment",
        "validation_segment_zh",
        "screening_profile",
        "screening_profile_zh",
        "horizon",
        "horizon_zh_label",
        "entry_type",
        "validation_bucket",
        "sample_count",
        "quality_gate_pass_rate",
        "avg_signal_score",
        "avg_confidence_score",
        "avg_high_probability_score",
        "avg_calibrated_win_probability",
        "avg_max_drawdown_after_signal",
        "segment_decision",
        "segment_decision_zh",
        "segment_note",
        "segment_note_zh",
    ]
    for window in forward_windows:
        columns.extend(
            [
                f"win_rate_{window}d",
                f"avg_return_{window}d",
                f"median_return_{window}d",
                f"worst_return_{window}d",
            ]
        )
    if events.empty:
        return pd.DataFrame(columns=columns)

    frame = events.copy()
    for column, default in [
        ("screening_profile", "default"),
        ("screening_profile_zh", "默认规则"),
        ("horizon", "unknown"),
        ("horizon_zh_label", "未知周期"),
        ("screening_backtest_entry_type", "unknown"),
        ("validation_bucket", "unknown"),
    ]:
        if column not in frame.columns:
            frame[column] = default
        frame[column] = frame[column].fillna(default).astype(str)

    group_columns = [
        "screening_profile",
        "screening_profile_zh",
        "horizon",
        "horizon_zh_label",
        "screening_backtest_entry_type",
        "validation_bucket",
    ]
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        profile, profile_zh, horizon, horizon_zh, entry_type, bucket = keys
        row: dict[str, object] = {
            "validation_segment": f"{profile}/{horizon}/{entry_type}/{bucket}",
            "validation_segment_zh": f"{profile_zh}/{horizon_zh}/{entry_type}/{bucket}",
            "screening_profile": profile,
            "screening_profile_zh": profile_zh,
            "horizon": horizon,
            "horizon_zh_label": horizon_zh,
            "entry_type": entry_type,
            "validation_bucket": bucket,
            "sample_count": int(len(group)),
            "quality_gate_pass_rate": _numeric_mean(group, "quality_gate_passed"),
            "avg_signal_score": _numeric_mean(group, "signal_score"),
            "avg_confidence_score": _numeric_mean(group, "confidence_score"),
            "avg_high_probability_score": _numeric_mean(group, "high_probability_score"),
            "avg_calibrated_win_probability": _numeric_mean(
                group,
                "calibrated_win_probability",
            ),
            "avg_max_drawdown_after_signal": _numeric_mean(
                group,
                "max_drawdown_after_signal",
            ),
        }
        for window in forward_windows:
            returns = pd.to_numeric(
                group.get(f"forward_return_{window}d", pd.Series(dtype=float)),
                errors="coerce",
            ).dropna()
            row[f"win_rate_{window}d"] = float((returns > 0).mean()) if len(returns) else np.nan
            row[f"avg_return_{window}d"] = float(returns.mean()) if len(returns) else np.nan
            row[f"median_return_{window}d"] = float(returns.median()) if len(returns) else np.nan
            row[f"worst_return_{window}d"] = float(returns.min()) if len(returns) else np.nan
        decision, decision_zh, note, note_zh = _segment_decision(row, _target_window(forward_windows))
        row["segment_decision"] = decision
        row["segment_decision_zh"] = decision_zh
        row["segment_note"] = note
        row["segment_note_zh"] = note_zh
        rows.append(row)

    result = pd.DataFrame(rows)
    result["_decision_rank"] = result["segment_decision"].map(
        {
            "strong_segment": 0,
            "watch_segment": 1,
            "thin_sample": 2,
            "weak_segment": 3,
        }
    ).fillna(9)
    result = result.sort_values(
        ["_decision_rank", "sample_count", f"win_rate_{_target_window(forward_windows)}d"],
        ascending=[True, False, False],
        na_position="last",
    ).drop(columns=["_decision_rank"])
    return result[columns].reset_index(drop=True)


def summarize_market_regime_validation(
    events: pd.DataFrame,
    forward_windows: tuple[int, ...] = DEFAULT_FORWARD_WINDOWS,
) -> pd.DataFrame:
    columns = [
        "validation_market_regime",
        "validation_market_regime_zh",
        "sample_count",
        "high_probability_sample_count",
        "quality_gate_pass_rate",
        "avg_signal_score",
        "avg_confidence_score",
        "avg_high_probability_score",
        "avg_calibrated_win_probability",
        "avg_max_drawdown_after_signal",
        "regime_decision",
        "regime_decision_zh",
        "regime_note",
        "regime_note_zh",
    ]
    for window in forward_windows:
        columns.extend(
            [
                f"win_rate_{window}d",
                f"avg_return_{window}d",
                f"median_return_{window}d",
                f"worst_return_{window}d",
            ]
        )
    if events.empty:
        return pd.DataFrame(columns=columns)

    frame = events.copy()
    if "validation_market_regime" not in frame.columns:
        frame["validation_market_regime"] = "unknown"
    if "validation_market_regime_zh" not in frame.columns:
        frame["validation_market_regime_zh"] = "未知市场状态"
    frame["validation_market_regime"] = frame["validation_market_regime"].fillna("unknown").astype(str)
    frame["validation_market_regime_zh"] = (
        frame["validation_market_regime_zh"].fillna("未知市场状态").astype(str)
    )

    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(
        ["validation_market_regime", "validation_market_regime_zh"],
        dropna=False,
    ):
        regime, regime_zh = keys
        row: dict[str, object] = {
            "validation_market_regime": regime,
            "validation_market_regime_zh": regime_zh,
            "sample_count": int(len(group)),
            "high_probability_sample_count": int(
                (group.get("validation_bucket", pd.Series(dtype=str)) == "high_probability").sum()
            ),
            "quality_gate_pass_rate": _numeric_mean(group, "quality_gate_passed"),
            "avg_signal_score": _numeric_mean(group, "signal_score"),
            "avg_confidence_score": _numeric_mean(group, "confidence_score"),
            "avg_high_probability_score": _numeric_mean(group, "high_probability_score"),
            "avg_calibrated_win_probability": _numeric_mean(
                group,
                "calibrated_win_probability",
            ),
            "avg_max_drawdown_after_signal": _numeric_mean(
                group,
                "max_drawdown_after_signal",
            ),
        }
        for window in forward_windows:
            returns = pd.to_numeric(
                group.get(f"forward_return_{window}d", pd.Series(dtype=float)),
                errors="coerce",
            ).dropna()
            row[f"win_rate_{window}d"] = float((returns > 0).mean()) if len(returns) else np.nan
            row[f"avg_return_{window}d"] = float(returns.mean()) if len(returns) else np.nan
            row[f"median_return_{window}d"] = float(returns.median()) if len(returns) else np.nan
            row[f"worst_return_{window}d"] = float(returns.min()) if len(returns) else np.nan
        decision, decision_zh, note, note_zh = _market_regime_validation_decision(
            row,
            _target_window(forward_windows),
        )
        row["regime_decision"] = decision
        row["regime_decision_zh"] = decision_zh
        row["regime_note"] = note
        row["regime_note_zh"] = note_zh
        rows.append(row)

    result = pd.DataFrame(rows)
    result["_decision_rank"] = result["regime_decision"].map(
        {
            "robust_regime": 0,
            "usable_regime": 1,
            "thin_sample": 2,
            "weak_regime": 3,
        }
    ).fillna(9)
    result = result.sort_values(
        ["_decision_rank", "sample_count", f"win_rate_{_target_window(forward_windows)}d"],
        ascending=[True, False, False],
        na_position="last",
    ).drop(columns=["_decision_rank"])
    return result[columns].reset_index(drop=True)


def _market_regime_validation_decision(
    row: dict[str, object],
    target_window: int,
) -> tuple[str, str, str, str]:
    sample_count = _safe_int(row.get("sample_count"))
    win_rate = _safe_float(row.get(f"win_rate_{target_window}d"))
    avg_return = _safe_float(row.get(f"avg_return_{target_window}d"))
    drawdown = _safe_float(row.get("avg_max_drawdown_after_signal"))
    if sample_count < 10:
        return (
            "thin_sample",
            "样本不足",
            f"Only {sample_count} samples in this market regime.",
            f"这个市场状态下只有 {sample_count} 个样本。",
        )
    if (
        _is_finite(win_rate)
        and win_rate >= 0.55
        and _is_finite(avg_return)
        and avg_return > 0
        and (not _is_finite(drawdown) or drawdown > -0.12)
    ):
        return (
            "robust_regime",
            "稳健市场状态",
            "Signals are historically robust in this market regime.",
            "该市场状态下信号历史表现较稳健。",
        )
    if _is_finite(avg_return) and avg_return > 0:
        return (
            "usable_regime",
            "可用但需谨慎",
            "Average return is positive, but regime evidence is not strong enough.",
            "平均收益为正，但该市场状态证据还不够强。",
        )
    return (
        "weak_regime",
        "弱市场状态",
        "Signals do not show positive evidence in this market regime.",
        "该市场状态下信号没有显示正向证据。",
    )


def build_market_regime_protection_policy(
    market_regime_summary: pd.DataFrame,
    target_window: int = 20,
) -> pd.DataFrame:
    columns = [
        "validation_market_regime",
        "validation_market_regime_zh",
        "sample_count",
        "win_rate",
        "avg_return",
        "avg_max_drawdown_after_signal",
        "regime_decision",
        "protection_action",
        "protection_action_zh",
        "allow_new_entries",
        "allowed_signal_bucket",
        "signal_score_delta",
        "confidence_score_delta",
        "high_probability_score_delta",
        "position_scale",
        "policy_severity",
        "policy_note",
        "policy_note_zh",
    ]
    if market_regime_summary.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for row in market_regime_summary.to_dict(orient="records"):
        regime = str(row.get("validation_market_regime", "unknown"))
        sample_count = _safe_int(row.get("sample_count"))
        win_rate = _safe_float(row.get(f"win_rate_{target_window}d"))
        avg_return = _safe_float(row.get(f"avg_return_{target_window}d"))
        drawdown = _safe_float(row.get("avg_max_drawdown_after_signal"))
        decision = str(row.get("regime_decision", ""))
        policy = _market_regime_policy_decision(
            regime=regime,
            sample_count=sample_count,
            win_rate=win_rate,
            avg_return=avg_return,
            drawdown=drawdown,
            regime_decision=decision,
        )
        rows.append(
            {
                "validation_market_regime": regime,
                "validation_market_regime_zh": row.get(
                    "validation_market_regime_zh",
                    "",
                ),
                "sample_count": sample_count,
                "win_rate": win_rate,
                "avg_return": avg_return,
                "avg_max_drawdown_after_signal": drawdown,
                "regime_decision": decision,
                **policy,
            }
        )
    result = pd.DataFrame(rows, columns=columns)
    result["_severity_rank"] = result["policy_severity"].map(
        {"critical": 0, "high": 1, "medium": 2, "info": 3}
    ).fillna(9)
    return result.sort_values(
        ["_severity_rank", "sample_count"],
        ascending=[True, False],
    ).drop(columns=["_severity_rank"]).reset_index(drop=True)


def _market_regime_policy_decision(
    regime: str,
    sample_count: int,
    win_rate: float,
    avg_return: float,
    drawdown: float,
    regime_decision: str,
) -> dict[str, object]:
    weak_evidence = (
        (_is_finite(win_rate) and win_rate < 0.50)
        or (_is_finite(avg_return) and avg_return <= 0)
        or (_is_finite(drawdown) and drawdown <= -0.15)
        or regime_decision == "weak_regime"
    )
    risk_regime = regime in {"bear_downtrend", "high_volatility"}
    if sample_count < 10:
        return {
            "protection_action": "collect_more_samples",
            "protection_action_zh": "继续收集样本",
            "allow_new_entries": False if risk_regime else True,
            "allowed_signal_bucket": "strict_high_probability_only" if risk_regime else "normal",
            "signal_score_delta": 3.0 if risk_regime else 0.0,
            "confidence_score_delta": 3.0 if risk_regime else 0.0,
            "high_probability_score_delta": 5.0 if risk_regime else 0.0,
            "position_scale": 0.50 if risk_regime else 1.0,
            "policy_severity": "medium" if risk_regime else "info",
            "policy_note": "Market-regime sample is thin; use conservative defaults until more evidence exists.",
            "policy_note_zh": "该市场状态样本不足；证据更多前使用保守默认规则。",
        }
    if risk_regime and weak_evidence:
        return {
            "protection_action": "block_new_entries",
            "protection_action_zh": "暂停新开仓",
            "allow_new_entries": False,
            "allowed_signal_bucket": "none",
            "signal_score_delta": 8.0,
            "confidence_score_delta": 8.0,
            "high_probability_score_delta": 10.0,
            "position_scale": 0.0,
            "policy_severity": "critical",
            "policy_note": "Bear or high-volatility regime shows weak evidence; new entries should be blocked until conditions improve.",
            "policy_note_zh": "熊市或高波动环境下历史证据偏弱；条件改善前应暂停新开仓。",
        }
    if risk_regime:
        return {
            "protection_action": "strict_only",
            "protection_action_zh": "只允许严格高概率",
            "allow_new_entries": True,
            "allowed_signal_bucket": "strict_high_probability_only",
            "signal_score_delta": 5.0,
            "confidence_score_delta": 5.0,
            "high_probability_score_delta": 7.0,
            "position_scale": 0.50,
            "policy_severity": "high",
            "policy_note": "Risk regime is usable but requires stricter thresholds and smaller position size.",
            "policy_note_zh": "风险市场状态仍可用，但需要更严格门槛和更小仓位。",
        }
    if weak_evidence:
        return {
            "protection_action": "tighten_entries",
            "protection_action_zh": "收紧入场",
            "allow_new_entries": True,
            "allowed_signal_bucket": "high_probability_or_near_only",
            "signal_score_delta": 3.0,
            "confidence_score_delta": 3.0,
            "high_probability_score_delta": 5.0,
            "position_scale": 0.75,
            "policy_severity": "medium",
            "policy_note": "Regime evidence is weak; allow only stronger setups.",
            "policy_note_zh": "该市场状态证据偏弱；只允许更强的机会。",
        }
    return {
        "protection_action": "normal_rules",
        "protection_action_zh": "使用正常规则",
        "allow_new_entries": True,
        "allowed_signal_bucket": "normal",
        "signal_score_delta": 0.0,
        "confidence_score_delta": 0.0,
        "high_probability_score_delta": 0.0,
        "position_scale": 1.0,
        "policy_severity": "info",
        "policy_note": "Market-regime validation is acceptable; use base rules.",
        "policy_note_zh": "该市场状态验证可接受；使用基础规则。",
    }


def _segment_decision(
    row: dict[str, object],
    target_window: int,
) -> tuple[str, str, str, str]:
    sample_count = _safe_int(row.get("sample_count"))
    win_rate = _safe_float(row.get(f"win_rate_{target_window}d"))
    avg_return = _safe_float(row.get(f"avg_return_{target_window}d"))
    drawdown = _safe_float(row.get("avg_max_drawdown_after_signal"))
    if sample_count < 10:
        return (
            "thin_sample",
            "样本不足",
            f"Only {sample_count} samples; do not tune this segment yet.",
            f"只有 {sample_count} 个样本；暂时不要根据该分层单独调参。",
        )
    if (
        _is_finite(win_rate)
        and win_rate >= 0.55
        and _is_finite(avg_return)
        and avg_return > 0
        and (not _is_finite(drawdown) or drawdown > -0.12)
    ):
        return (
            "strong_segment",
            "强分层",
            "Segment has enough samples, positive win rate, positive average return, and controlled drawdown.",
            "该分层样本足够，胜率、平均收益为正，回撤可控。",
        )
    if _is_finite(avg_return) and avg_return > 0:
        return (
            "watch_segment",
            "观察分层",
            "Segment average return is positive, but at least one quality metric is not strong enough.",
            "该分层平均收益为正，但至少一个质量指标还不够强。",
        )
    return (
        "weak_segment",
        "弱分层",
        "Segment does not yet show positive historical evidence.",
        "该分层还没有显示正向历史证据。",
    )


def summarize_probability_calibration(
    events: pd.DataFrame,
    target_window: int = 20,
) -> pd.DataFrame:
    columns = [
        "probability_bucket",
        "probability_bucket_zh",
        "sample_count",
        "avg_estimated_probability",
        f"actual_win_rate_{target_window}d",
        f"avg_return_{target_window}d",
        "calibration_error",
        "calibration_error_abs",
        "recommended_probability_adjustment",
        "adjusted_estimated_probability",
        "adjusted_calibration_error_abs",
        "brier_score",
        "expected_calibration_error_component",
        "formula_action",
        "formula_action_zh",
        "calibration_quality",
        "calibration_quality_zh",
        "calibration_note",
        "calibration_note_zh",
    ]
    required = {"calibrated_win_probability", f"forward_return_{target_window}d"}
    if events.empty or not required.issubset(events.columns):
        return pd.DataFrame(columns=columns)

    frame = events.copy()
    frame["calibrated_win_probability"] = pd.to_numeric(
        frame["calibrated_win_probability"],
        errors="coerce",
    )
    frame[f"forward_return_{target_window}d"] = pd.to_numeric(
        frame[f"forward_return_{target_window}d"],
        errors="coerce",
    )
    frame = frame.dropna(subset=["calibrated_win_probability", f"forward_return_{target_window}d"])
    if frame.empty:
        return pd.DataFrame(columns=columns)

    bins = [-np.inf, 0.50, 0.55, 0.60, 0.65, np.inf]
    labels = ["below_50", "50_to_55", "55_to_60", "60_to_65", "65_plus"]
    labels_zh = {
        "below_50": "低于50%",
        "50_to_55": "50%-55%",
        "55_to_60": "55%-60%",
        "60_to_65": "60%-65%",
        "65_plus": "65%以上",
    }
    frame["probability_bucket"] = pd.cut(
        frame["calibrated_win_probability"],
        bins=bins,
        labels=labels,
        right=False,
    ).astype("string")

    rows: list[dict[str, object]] = []
    return_column = f"forward_return_{target_window}d"
    total_samples = len(frame)
    for bucket in labels:
        group = frame[frame["probability_bucket"] == bucket]
        returns = group[return_column].dropna()
        if returns.empty:
            rows.append(
                {
                    "probability_bucket": bucket,
                    "probability_bucket_zh": labels_zh[bucket],
                    "sample_count": 0,
                    "avg_estimated_probability": np.nan,
                    f"actual_win_rate_{target_window}d": np.nan,
                    f"avg_return_{target_window}d": np.nan,
                    "calibration_error": np.nan,
                    "calibration_error_abs": np.nan,
                    "recommended_probability_adjustment": np.nan,
                    "adjusted_estimated_probability": np.nan,
                    "adjusted_calibration_error_abs": np.nan,
                    "brier_score": np.nan,
                    "expected_calibration_error_component": np.nan,
                    "formula_action": "collect_more_samples",
                    "formula_action_zh": "继续收集样本",
                    "calibration_quality": "insufficient",
                    "calibration_quality_zh": "样本不足",
                    "calibration_note": "No samples in this probability bucket.",
                    "calibration_note_zh": "这个概率分组没有样本。",
                }
            )
            continue
        estimated = float(group["calibrated_win_probability"].mean())
        actual = float((returns > 0).mean())
        avg_return = float(returns.mean())
        error = actual - estimated
        adjustment = min(max(error, -0.10), 0.10)
        adjusted_estimated = min(max(estimated + adjustment, 0.05), 0.90)
        adjusted_error_abs = abs(actual - adjusted_estimated)
        outcomes = (returns > 0).astype(float)
        brier_score = float(((outcomes - group.loc[returns.index, "calibrated_win_probability"]) ** 2).mean())
        ece_component = abs(error) * len(returns) / total_samples if total_samples else np.nan
        quality, quality_zh = _probability_calibration_quality(
            sample_count=len(returns),
            error_abs=abs(error),
        )
        formula_action, formula_action_zh = _probability_formula_action(
            sample_count=len(returns),
            error=error,
            error_abs=abs(error),
        )
        rows.append(
            {
                "probability_bucket": bucket,
                "probability_bucket_zh": labels_zh[bucket],
                "sample_count": int(len(returns)),
                "avg_estimated_probability": estimated,
                f"actual_win_rate_{target_window}d": actual,
                f"avg_return_{target_window}d": avg_return,
                "calibration_error": error,
                "calibration_error_abs": abs(error),
                "recommended_probability_adjustment": adjustment,
                "adjusted_estimated_probability": adjusted_estimated,
                "adjusted_calibration_error_abs": adjusted_error_abs,
                "brier_score": brier_score,
                "expected_calibration_error_component": ece_component,
                "formula_action": formula_action,
                "formula_action_zh": formula_action_zh,
                "calibration_quality": quality,
                "calibration_quality_zh": quality_zh,
                "calibration_note": (
                    f"Estimated win probability averaged {estimated:.2%}; "
                    f"actual {target_window}d win rate was {actual:.2%}. "
                    f"Suggested probability adjustment is {adjustment:+.2%}."
                ),
                "calibration_note_zh": (
                    f"估计胜率平均为{estimated:.2%}；"
                    f"未来{target_window}日实际胜率为{actual:.2%}。"
                    f"建议概率修正为{adjustment:+.2%}。"
                ),
            }
        )
    rows.append(_overall_probability_calibration_row(frame, target_window, total_samples))
    return pd.DataFrame(rows, columns=columns)


def summarize_walk_forward_portfolios(
    events: pd.DataFrame,
    target_window: int = 20,
    top_n: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_columns = [
        "portfolio_name",
        "portfolio_name_zh",
        "target_window_days",
        "top_n",
        "rebalance_count",
        "avg_position_count",
        "avg_forward_return",
        "median_forward_return",
        "win_rate",
        "best_forward_return",
        "worst_forward_return",
        "compounded_forward_return",
        "avg_selected_probability",
        "avg_selected_high_probability_score",
        "avg_selected_drawdown",
        "portfolio_note",
        "portfolio_note_zh",
    ]
    rebalance_columns = [
        "date",
        "portfolio_name",
        "portfolio_name_zh",
        "selected_count",
        "selected_tickers",
        f"portfolio_forward_return_{target_window}d",
        "portfolio_max_drawdown_after_signal",
        "avg_selected_probability",
        "avg_selected_high_probability_score",
    ]
    required = {
        "date",
        "ticker",
        f"forward_return_{target_window}d",
        "high_probability_score",
    }
    if events.empty or not required.issubset(events.columns):
        return (
            pd.DataFrame(columns=summary_columns),
            pd.DataFrame(columns=rebalance_columns),
        )

    strategies = [
        {
            "portfolio_name": "top_calibrated_probability",
            "portfolio_name_zh": "校准胜率前N",
            "score_column": "calibrated_win_probability",
            "buckets": None,
            "note": "Selects the highest calibrated win-probability candidates each signal date.",
            "note_zh": "每个信号日选择校准后胜率最高的候选。",
        },
        {
            "portfolio_name": "top_high_probability_score",
            "portfolio_name_zh": "高概率分数前N",
            "score_column": "high_probability_score",
            "buckets": None,
            "note": "Selects the highest high-probability-score candidates each signal date.",
            "note_zh": "每个信号日选择高概率分数最高的候选。",
        },
        {
            "portfolio_name": "strict_high_probability_only",
            "portfolio_name_zh": "只选严格高概率",
            "score_column": "calibrated_win_probability",
            "buckets": {"high_probability"},
            "note": "Selects only candidates that passed the strict high-probability gate.",
            "note_zh": "只选择通过严格高概率门槛的候选。",
        },
        {
            "portfolio_name": "watchlist_or_better",
            "portfolio_name_zh": "观察名单及以上",
            "score_column": "calibrated_win_probability",
            "buckets": {"high_probability", "near_watchlist"},
            "note": "Selects high-probability and near-watchlist candidates.",
            "note_zh": "选择高概率候选和接近观察名单的候选。",
        },
    ]

    all_rebalances: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for strategy in strategies:
        score_column = str(strategy["score_column"])
        if score_column not in events.columns:
            continue
        strategy_rebalances = _portfolio_rebalance_rows(
            events=events,
            target_window=target_window,
            top_n=top_n,
            portfolio_name=strategy["portfolio_name"],
            portfolio_name_zh=strategy["portfolio_name_zh"],
            score_column=score_column,
            buckets=strategy["buckets"],
        )
        all_rebalances.extend(strategy_rebalances)
        summary_rows.append(
            _portfolio_summary_row(
                strategy_rebalances,
                portfolio_name=strategy["portfolio_name"],
                portfolio_name_zh=strategy["portfolio_name_zh"],
                target_window=target_window,
                top_n=top_n,
                note=strategy["note"],
                note_zh=strategy["note_zh"],
            )
        )

    return (
        pd.DataFrame(summary_rows, columns=summary_columns),
        pd.DataFrame(all_rebalances, columns=rebalance_columns),
    )


def _portfolio_rebalance_rows(
    events: pd.DataFrame,
    target_window: int,
    top_n: int,
    portfolio_name: object,
    portfolio_name_zh: object,
    score_column: str,
    buckets: set[str] | None,
) -> list[dict[str, object]]:
    frame = events.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame[f"forward_return_{target_window}d"] = pd.to_numeric(
        frame[f"forward_return_{target_window}d"],
        errors="coerce",
    )
    frame[score_column] = pd.to_numeric(frame[score_column], errors="coerce")
    frame["high_probability_score"] = pd.to_numeric(
        frame["high_probability_score"],
        errors="coerce",
    )
    if "calibrated_win_probability" in frame.columns:
        frame["calibrated_win_probability"] = pd.to_numeric(
            frame["calibrated_win_probability"],
            errors="coerce",
        )
    if buckets is not None:
        frame = frame[frame["validation_bucket"].isin(buckets)].copy()
    frame = frame.dropna(subset=["date", "ticker", f"forward_return_{target_window}d", score_column])
    if frame.empty:
        return []

    rows: list[dict[str, object]] = []
    return_column = f"forward_return_{target_window}d"
    for date, group in frame.groupby("date", dropna=False):
        selected = (
            group.sort_values([score_column, "high_probability_score"], ascending=False)
            .drop_duplicates("ticker", keep="first")
            .head(top_n)
        )
        if selected.empty:
            continue
        drawdown = (
            float(pd.to_numeric(selected["max_drawdown_after_signal"], errors="coerce").mean())
            if "max_drawdown_after_signal" in selected.columns
            else np.nan
        )
        rows.append(
            {
                "date": date,
                "portfolio_name": portfolio_name,
                "portfolio_name_zh": portfolio_name_zh,
                "selected_count": int(len(selected)),
                "selected_tickers": ", ".join(selected["ticker"].astype(str).to_list()),
                f"portfolio_forward_return_{target_window}d": float(selected[return_column].mean()),
                "portfolio_max_drawdown_after_signal": drawdown,
                "avg_selected_probability": float(
                    selected["calibrated_win_probability"].mean()
                )
                if "calibrated_win_probability" in selected.columns
                else np.nan,
                "avg_selected_high_probability_score": float(
                    selected["high_probability_score"].mean()
                ),
            }
        )
    return rows


def _portfolio_summary_row(
    rebalance_rows: list[dict[str, object]],
    portfolio_name: object,
    portfolio_name_zh: object,
    target_window: int,
    top_n: int,
    note: object,
    note_zh: object,
) -> dict[str, object]:
    return_column = f"portfolio_forward_return_{target_window}d"
    if not rebalance_rows:
        return {
            "portfolio_name": portfolio_name,
            "portfolio_name_zh": portfolio_name_zh,
            "target_window_days": int(target_window),
            "top_n": int(top_n),
            "rebalance_count": 0,
            "avg_position_count": 0.0,
            "avg_forward_return": np.nan,
            "median_forward_return": np.nan,
            "win_rate": np.nan,
            "best_forward_return": np.nan,
            "worst_forward_return": np.nan,
            "compounded_forward_return": np.nan,
            "avg_selected_probability": np.nan,
            "avg_selected_high_probability_score": np.nan,
            "avg_selected_drawdown": np.nan,
            "portfolio_note": note,
            "portfolio_note_zh": note_zh,
        }
    frame = pd.DataFrame(rebalance_rows)
    returns = pd.to_numeric(frame[return_column], errors="coerce").dropna()
    return {
        "portfolio_name": portfolio_name,
        "portfolio_name_zh": portfolio_name_zh,
        "target_window_days": int(target_window),
        "top_n": int(top_n),
        "rebalance_count": int(len(returns)),
        "avg_position_count": float(pd.to_numeric(frame["selected_count"], errors="coerce").mean()),
        "avg_forward_return": float(returns.mean()) if len(returns) else np.nan,
        "median_forward_return": float(returns.median()) if len(returns) else np.nan,
        "win_rate": float((returns > 0).mean()) if len(returns) else np.nan,
        "best_forward_return": float(returns.max()) if len(returns) else np.nan,
        "worst_forward_return": float(returns.min()) if len(returns) else np.nan,
        "compounded_forward_return": float((1.0 + returns).prod() - 1.0)
        if len(returns)
        else np.nan,
        "avg_selected_probability": float(
            pd.to_numeric(frame["avg_selected_probability"], errors="coerce").mean()
        ),
        "avg_selected_high_probability_score": float(
            pd.to_numeric(frame["avg_selected_high_probability_score"], errors="coerce").mean()
        ),
        "avg_selected_drawdown": float(
            pd.to_numeric(frame["portfolio_max_drawdown_after_signal"], errors="coerce").mean()
        ),
        "portfolio_note": note,
        "portfolio_note_zh": note_zh,
    }


def build_walk_forward_portfolio_equity(
    prices: pd.DataFrame,
    portfolio_rebalances: pd.DataFrame,
    target_window: int = 20,
    transaction_cost: float = 0.001,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_columns = [
        "portfolio_name",
        "portfolio_name_zh",
        "daily_rows",
        "start_date",
        "end_date",
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "max_drawdown",
        "positive_day_rate",
        "avg_daily_return",
        "best_daily_return",
        "worst_daily_return",
        "avg_position_count",
        "avg_turnover",
        "transaction_cost",
        "equity_note",
        "equity_note_zh",
    ]
    curve_columns = [
        "date",
        "portfolio_name",
        "portfolio_name_zh",
        "selected_tickers",
        "selected_count",
        "daily_return_before_cost",
        "turnover",
        "transaction_cost_impact",
        "daily_return",
        "equity",
        "drawdown",
    ]
    required_rebalances = {"date", "portfolio_name", "portfolio_name_zh", "selected_tickers"}
    required_prices = {"date", "ticker", "adj_close"}
    if (
        prices.empty
        or portfolio_rebalances.empty
        or not required_rebalances.issubset(portfolio_rebalances.columns)
        or not required_prices.issubset(prices.columns)
    ):
        return (
            pd.DataFrame(columns=summary_columns),
            pd.DataFrame(columns=curve_columns),
        )

    price_frame = _normalize_prices(prices)
    price_pivot = (
        price_frame.pivot_table(
            index="date",
            columns="ticker",
            values="adj_close",
            aggfunc="last",
        )
        .sort_index()
        .astype(float)
    )
    daily_returns = price_pivot.pct_change().replace([np.inf, -np.inf], np.nan)
    if daily_returns.empty:
        return (
            pd.DataFrame(columns=summary_columns),
            pd.DataFrame(columns=curve_columns),
        )

    rebalance_frame = portfolio_rebalances.copy()
    rebalance_frame["date"] = pd.to_datetime(rebalance_frame["date"], errors="coerce")
    rebalance_frame = rebalance_frame.dropna(subset=["date", "portfolio_name", "selected_tickers"])
    if rebalance_frame.empty:
        return (
            pd.DataFrame(columns=summary_columns),
            pd.DataFrame(columns=curve_columns),
        )

    all_curve_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    group_columns = ["portfolio_name", "portfolio_name_zh"]
    for keys, group in rebalance_frame.groupby(group_columns, dropna=False):
        portfolio_name, portfolio_name_zh = keys
        curve_rows = _portfolio_equity_rows(
            daily_returns=daily_returns,
            rebalances=group.sort_values("date"),
            portfolio_name=portfolio_name,
            portfolio_name_zh=portfolio_name_zh,
            target_window=target_window,
            transaction_cost=transaction_cost,
        )
        all_curve_rows.extend(curve_rows)
        summary_rows.append(
            _portfolio_equity_summary_row(
                curve_rows=curve_rows,
                portfolio_name=portfolio_name,
                portfolio_name_zh=portfolio_name_zh,
                transaction_cost=transaction_cost,
            )
        )

    return (
        pd.DataFrame(summary_rows, columns=summary_columns),
        pd.DataFrame(all_curve_rows, columns=curve_columns),
    )


def _portfolio_equity_rows(
    daily_returns: pd.DataFrame,
    rebalances: pd.DataFrame,
    portfolio_name: object,
    portfolio_name_zh: object,
    target_window: int,
    transaction_cost: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rebalance_rows = list(rebalances.itertuples(index=False))
    if not rebalance_rows:
        return rows

    equity = 1.0
    previous_weights: dict[str, float] = {}
    last_written_date: pd.Timestamp | None = None
    available_dates = pd.Index(daily_returns.index)

    for index, rebalance in enumerate(rebalance_rows):
        rebalance_date = pd.Timestamp(rebalance.date)
        selected_tickers = _parse_selected_tickers(getattr(rebalance, "selected_tickers", ""))
        selected_tickers = [ticker for ticker in selected_tickers if ticker in daily_returns.columns]
        if not selected_tickers:
            continue
        start_position = int(available_dates.searchsorted(rebalance_date, side="left"))
        if start_position >= len(available_dates):
            continue
        if index + 1 < len(rebalance_rows):
            next_date = pd.Timestamp(rebalance_rows[index + 1].date)
            end_position = int(available_dates.searchsorted(next_date, side="left"))
        else:
            end_position = min(start_position + target_window + 1, len(available_dates))
        if end_position <= start_position:
            continue

        current_weights = {ticker: 1.0 / len(selected_tickers) for ticker in selected_tickers}
        turnover = _portfolio_turnover(previous_weights, current_weights)
        previous_weights = current_weights
        dates = available_dates[start_position:end_position]
        for date_offset, date in enumerate(dates):
            date = pd.Timestamp(date)
            if last_written_date is not None and date <= last_written_date:
                continue
            selected_returns = daily_returns.loc[date, selected_tickers].dropna()
            daily_return_before_cost = (
                float(selected_returns.mean())
                if len(selected_returns) and date_offset > 0
                else 0.0
            )
            cost_impact = turnover * transaction_cost if date_offset == 0 else 0.0
            daily_return = daily_return_before_cost - cost_impact
            equity *= 1.0 + daily_return
            rows.append(
                {
                    "date": date,
                    "portfolio_name": portfolio_name,
                    "portfolio_name_zh": portfolio_name_zh,
                    "selected_tickers": ", ".join(selected_tickers),
                    "selected_count": int(len(selected_tickers)),
                    "daily_return_before_cost": daily_return_before_cost,
                    "turnover": float(turnover) if date_offset == 0 else 0.0,
                    "transaction_cost_impact": float(cost_impact),
                    "daily_return": float(daily_return),
                    "equity": float(equity),
                    "drawdown": np.nan,
                }
            )
            last_written_date = date

    if rows:
        equity_series = pd.Series([row["equity"] for row in rows], dtype=float)
        drawdowns = equity_series / equity_series.cummax() - 1.0
        for row, drawdown in zip(rows, drawdowns):
            row["drawdown"] = float(drawdown)
    return rows


def _portfolio_equity_summary_row(
    curve_rows: list[dict[str, object]],
    portfolio_name: object,
    portfolio_name_zh: object,
    transaction_cost: float,
) -> dict[str, object]:
    if not curve_rows:
        return {
            "portfolio_name": portfolio_name,
            "portfolio_name_zh": portfolio_name_zh,
            "daily_rows": 0,
            "start_date": pd.NaT,
            "end_date": pd.NaT,
            "total_return": np.nan,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "positive_day_rate": np.nan,
            "avg_daily_return": np.nan,
            "best_daily_return": np.nan,
            "worst_daily_return": np.nan,
            "avg_position_count": np.nan,
            "avg_turnover": np.nan,
            "transaction_cost": transaction_cost,
            "equity_note": "No daily portfolio equity curve could be built.",
            "equity_note_zh": "没有生成逐日组合净值曲线。",
        }
    frame = pd.DataFrame(curve_rows)
    returns = pd.to_numeric(frame["daily_return"], errors="coerce").fillna(0.0)
    equity = pd.to_numeric(frame["equity"], errors="coerce").dropna()
    total_return = float(equity.iloc[-1] - 1.0) if len(equity) else np.nan
    day_count = len(returns)
    annualized_return = (
        float((1.0 + total_return) ** (252 / day_count) - 1.0)
        if day_count > 0 and _is_finite(total_return) and total_return > -1.0
        else np.nan
    )
    volatility = float(returns.std(ddof=0) * np.sqrt(252)) if day_count > 1 else np.nan
    sharpe = (
        float((returns.mean() / returns.std(ddof=0)) * np.sqrt(252))
        if day_count > 1 and returns.std(ddof=0) > 0
        else np.nan
    )
    return {
        "portfolio_name": portfolio_name,
        "portfolio_name_zh": portfolio_name_zh,
        "daily_rows": int(day_count),
        "start_date": frame["date"].min(),
        "end_date": frame["date"].max(),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": float(pd.to_numeric(frame["drawdown"], errors="coerce").min()),
        "positive_day_rate": float((returns > 0).mean()) if day_count else np.nan,
        "avg_daily_return": float(returns.mean()) if day_count else np.nan,
        "best_daily_return": float(returns.max()) if day_count else np.nan,
        "worst_daily_return": float(returns.min()) if day_count else np.nan,
        "avg_position_count": float(pd.to_numeric(frame["selected_count"], errors="coerce").mean()),
        "avg_turnover": float(pd.to_numeric(frame["turnover"], errors="coerce").mean()),
        "transaction_cost": transaction_cost,
        "equity_note": (
            "Daily curve holds each equal-weight basket until the next rebalance "
            "or the target validation window, including transaction-cost drag."
        ),
        "equity_note_zh": (
            "逐日曲线按等权组合持有到下一次调仓或目标验证窗口，"
            "并计入交易成本拖累。"
        ),
    }


def _parse_selected_tickers(value: object) -> list[str]:
    return [
        str(item).strip().upper()
        for item in str(value).split(",")
        if str(item).strip()
    ]


def _portfolio_turnover(
    previous_weights: dict[str, float],
    current_weights: dict[str, float],
) -> float:
    tickers = set(previous_weights) | set(current_weights)
    if not tickers:
        return 0.0
    return float(
        0.5
        * sum(abs(current_weights.get(ticker, 0.0) - previous_weights.get(ticker, 0.0)) for ticker in tickers)
    )


def build_walk_forward_benchmark_comparison(
    prices: pd.DataFrame,
    portfolio_equity_curve: pd.DataFrame,
    benchmark_tickers: tuple[str, ...] = DEFAULT_BENCHMARK_TICKERS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_columns = [
        "portfolio_name",
        "portfolio_name_zh",
        "benchmark_ticker",
        "daily_rows",
        "start_date",
        "end_date",
        "portfolio_total_return",
        "benchmark_total_return",
        "excess_total_return",
        "portfolio_annualized_return",
        "benchmark_annualized_return",
        "excess_annualized_return",
        "portfolio_max_drawdown",
        "benchmark_max_drawdown",
        "drawdown_advantage",
        "portfolio_sharpe",
        "benchmark_sharpe",
        "daily_correlation",
        "benchmark_note",
        "benchmark_note_zh",
    ]
    curve_columns = [
        "date",
        "portfolio_name",
        "portfolio_name_zh",
        "benchmark_ticker",
        "portfolio_daily_return",
        "benchmark_daily_return",
        "excess_daily_return",
        "portfolio_equity",
        "benchmark_equity",
        "relative_equity",
        "portfolio_drawdown",
        "benchmark_drawdown",
    ]
    required_curve = {"date", "portfolio_name", "portfolio_name_zh", "daily_return", "equity", "drawdown"}
    required_prices = {"date", "ticker", "adj_close"}
    if (
        prices.empty
        or portfolio_equity_curve.empty
        or not required_curve.issubset(portfolio_equity_curve.columns)
        or not required_prices.issubset(prices.columns)
    ):
        return pd.DataFrame(columns=summary_columns), pd.DataFrame(columns=curve_columns)

    price_frame = _normalize_prices(prices)
    benchmark_tickers = tuple(dict.fromkeys(str(ticker).upper().strip() for ticker in benchmark_tickers if str(ticker).strip()))
    price_frame = price_frame[price_frame["ticker"].isin(benchmark_tickers)].copy()
    if price_frame.empty:
        return pd.DataFrame(columns=summary_columns), pd.DataFrame(columns=curve_columns)

    benchmark_prices = (
        price_frame.pivot_table(
            index="date",
            columns="ticker",
            values="adj_close",
            aggfunc="last",
        )
        .sort_index()
        .astype(float)
    )
    benchmark_returns = benchmark_prices.pct_change().replace([np.inf, -np.inf], np.nan)

    curve_frame = portfolio_equity_curve.copy()
    curve_frame["date"] = pd.to_datetime(curve_frame["date"], errors="coerce")
    curve_frame["daily_return"] = pd.to_numeric(curve_frame["daily_return"], errors="coerce")
    curve_frame["equity"] = pd.to_numeric(curve_frame["equity"], errors="coerce")
    curve_frame["drawdown"] = pd.to_numeric(curve_frame["drawdown"], errors="coerce")
    curve_frame = curve_frame.dropna(subset=["date", "portfolio_name", "daily_return", "equity"])
    if curve_frame.empty:
        return pd.DataFrame(columns=summary_columns), pd.DataFrame(columns=curve_columns)

    all_curve_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    group_columns = ["portfolio_name", "portfolio_name_zh"]
    for keys, group in curve_frame.groupby(group_columns, dropna=False):
        portfolio_name, portfolio_name_zh = keys
        group = group.sort_values("date")
        for benchmark_ticker in benchmark_tickers:
            if benchmark_ticker not in benchmark_returns.columns:
                summary_rows.append(
                    _missing_benchmark_summary_row(
                        portfolio_name,
                        portfolio_name_zh,
                        benchmark_ticker,
                    )
                )
                continue
            comparison_rows = _benchmark_comparison_rows(
                portfolio_group=group,
                benchmark_returns=benchmark_returns[benchmark_ticker],
                portfolio_name=portfolio_name,
                portfolio_name_zh=portfolio_name_zh,
                benchmark_ticker=benchmark_ticker,
            )
            all_curve_rows.extend(comparison_rows)
            summary_rows.append(
                _benchmark_summary_row(
                    comparison_rows=comparison_rows,
                    portfolio_name=portfolio_name,
                    portfolio_name_zh=portfolio_name_zh,
                    benchmark_ticker=benchmark_ticker,
                )
            )

    return (
        pd.DataFrame(summary_rows, columns=summary_columns),
        pd.DataFrame(all_curve_rows, columns=curve_columns),
    )


def _benchmark_comparison_rows(
    portfolio_group: pd.DataFrame,
    benchmark_returns: pd.Series,
    portfolio_name: object,
    portfolio_name_zh: object,
    benchmark_ticker: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    benchmark_equity = 1.0
    benchmark_values: list[float] = []
    for index, row in enumerate(portfolio_group.itertuples(index=False)):
        date = pd.Timestamp(row.date)
        if date not in benchmark_returns.index:
            continue
        portfolio_daily_return = float(row.daily_return)
        benchmark_daily_return = 0.0 if index == 0 else float(benchmark_returns.loc[date])
        if not _is_finite(benchmark_daily_return):
            benchmark_daily_return = 0.0
        benchmark_equity *= 1.0 + benchmark_daily_return
        benchmark_values.append(benchmark_equity)
        benchmark_series = pd.Series(benchmark_values, dtype=float)
        benchmark_drawdown = float(benchmark_series.iloc[-1] / benchmark_series.cummax().iloc[-1] - 1.0)
        rows.append(
            {
                "date": date,
                "portfolio_name": portfolio_name,
                "portfolio_name_zh": portfolio_name_zh,
                "benchmark_ticker": benchmark_ticker,
                "portfolio_daily_return": portfolio_daily_return,
                "benchmark_daily_return": benchmark_daily_return,
                "excess_daily_return": portfolio_daily_return - benchmark_daily_return,
                "portfolio_equity": float(row.equity),
                "benchmark_equity": float(benchmark_equity),
                "relative_equity": float(row.equity / benchmark_equity - 1.0)
                if benchmark_equity > 0
                else np.nan,
                "portfolio_drawdown": float(row.drawdown)
                if _is_finite(row.drawdown)
                else np.nan,
                "benchmark_drawdown": benchmark_drawdown,
            }
        )
    return rows


def _benchmark_summary_row(
    comparison_rows: list[dict[str, object]],
    portfolio_name: object,
    portfolio_name_zh: object,
    benchmark_ticker: str,
) -> dict[str, object]:
    if not comparison_rows:
        return _missing_benchmark_summary_row(
            portfolio_name,
            portfolio_name_zh,
            benchmark_ticker,
        )
    frame = pd.DataFrame(comparison_rows)
    portfolio_returns = pd.to_numeric(frame["portfolio_daily_return"], errors="coerce").fillna(0.0)
    benchmark_returns = pd.to_numeric(frame["benchmark_daily_return"], errors="coerce").fillna(0.0)
    daily_rows = len(frame)
    portfolio_total_return = float(frame["portfolio_equity"].iloc[-1] - 1.0)
    benchmark_total_return = float(frame["benchmark_equity"].iloc[-1] - 1.0)
    excess_total_return = portfolio_total_return - benchmark_total_return
    portfolio_annualized = _annualized_return(portfolio_total_return, daily_rows)
    benchmark_annualized = _annualized_return(benchmark_total_return, daily_rows)
    correlation = (
        float(portfolio_returns.corr(benchmark_returns))
        if daily_rows > 2 and portfolio_returns.std(ddof=0) > 0 and benchmark_returns.std(ddof=0) > 0
        else np.nan
    )
    return {
        "portfolio_name": portfolio_name,
        "portfolio_name_zh": portfolio_name_zh,
        "benchmark_ticker": benchmark_ticker,
        "daily_rows": int(daily_rows),
        "start_date": frame["date"].min(),
        "end_date": frame["date"].max(),
        "portfolio_total_return": portfolio_total_return,
        "benchmark_total_return": benchmark_total_return,
        "excess_total_return": excess_total_return,
        "portfolio_annualized_return": portfolio_annualized,
        "benchmark_annualized_return": benchmark_annualized,
        "excess_annualized_return": portfolio_annualized - benchmark_annualized
        if _is_finite(portfolio_annualized) and _is_finite(benchmark_annualized)
        else np.nan,
        "portfolio_max_drawdown": float(pd.to_numeric(frame["portfolio_drawdown"], errors="coerce").min()),
        "benchmark_max_drawdown": float(pd.to_numeric(frame["benchmark_drawdown"], errors="coerce").min()),
        "drawdown_advantage": float(
            pd.to_numeric(frame["portfolio_drawdown"], errors="coerce").min()
            - pd.to_numeric(frame["benchmark_drawdown"], errors="coerce").min()
        ),
        "portfolio_sharpe": _sharpe(portfolio_returns),
        "benchmark_sharpe": _sharpe(benchmark_returns),
        "daily_correlation": correlation,
        "benchmark_note": (
            f"Compared daily portfolio equity against {benchmark_ticker} over matching dates."
        ),
        "benchmark_note_zh": (
            f"在相同日期上，将组合逐日净值与{benchmark_ticker}进行对比。"
        ),
    }


def _missing_benchmark_summary_row(
    portfolio_name: object,
    portfolio_name_zh: object,
    benchmark_ticker: str,
) -> dict[str, object]:
    return {
        "portfolio_name": portfolio_name,
        "portfolio_name_zh": portfolio_name_zh,
        "benchmark_ticker": benchmark_ticker,
        "daily_rows": 0,
        "start_date": pd.NaT,
        "end_date": pd.NaT,
        "portfolio_total_return": np.nan,
        "benchmark_total_return": np.nan,
        "excess_total_return": np.nan,
        "portfolio_annualized_return": np.nan,
        "benchmark_annualized_return": np.nan,
        "excess_annualized_return": np.nan,
        "portfolio_max_drawdown": np.nan,
        "benchmark_max_drawdown": np.nan,
        "drawdown_advantage": np.nan,
        "portfolio_sharpe": np.nan,
        "benchmark_sharpe": np.nan,
        "daily_correlation": np.nan,
        "benchmark_note": f"{benchmark_ticker} benchmark price data is unavailable.",
        "benchmark_note_zh": f"缺少{benchmark_ticker}基准价格数据。",
    }


def _annualized_return(total_return: float, day_count: int) -> float:
    if day_count <= 0 or not _is_finite(total_return) or total_return <= -1.0:
        return np.nan
    return float((1.0 + total_return) ** (252 / day_count) - 1.0)


def _sharpe(daily_returns: pd.Series) -> float:
    returns = pd.to_numeric(daily_returns, errors="coerce").dropna()
    if len(returns) < 2 or returns.std(ddof=0) <= 0:
        return np.nan
    return float((returns.mean() / returns.std(ddof=0)) * np.sqrt(252))


def build_benchmark_aware_policy(benchmark_summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "portfolio_name",
        "portfolio_name_zh",
        "benchmark_count",
        "avg_excess_total_return",
        "worst_excess_total_return",
        "avg_excess_annualized_return",
        "avg_drawdown_advantage",
        "worst_drawdown_advantage",
        "avg_daily_correlation",
        "benchmark_passed",
        "benchmark_policy_action",
        "benchmark_policy_action_zh",
        "threshold_bias",
        "threshold_bias_zh",
        "allow_relaxation",
        "policy_note",
        "policy_note_zh",
    ]
    required = {
        "portfolio_name",
        "portfolio_name_zh",
        "benchmark_ticker",
        "excess_total_return",
        "excess_annualized_return",
        "drawdown_advantage",
        "daily_correlation",
    }
    if benchmark_summary.empty or not required.issubset(benchmark_summary.columns):
        return pd.DataFrame(columns=columns)

    frame = benchmark_summary.copy()
    for column in [
        "excess_total_return",
        "excess_annualized_return",
        "drawdown_advantage",
        "daily_correlation",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["portfolio_name", "benchmark_ticker"])
    if frame.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(["portfolio_name", "portfolio_name_zh"], dropna=False):
        portfolio_name, portfolio_name_zh = keys
        valid = group.dropna(subset=["excess_total_return", "drawdown_advantage"])
        if valid.empty:
            rows.append(
                _benchmark_policy_row(
                    portfolio_name=portfolio_name,
                    portfolio_name_zh=portfolio_name_zh,
                    benchmark_count=0,
                    avg_excess_total_return=np.nan,
                    worst_excess_total_return=np.nan,
                    avg_excess_annualized_return=np.nan,
                    avg_drawdown_advantage=np.nan,
                    worst_drawdown_advantage=np.nan,
                    avg_daily_correlation=np.nan,
                    action="insufficient_benchmark_data",
                    action_zh="基准数据不足",
                    threshold_bias="hold",
                    threshold_bias_zh="暂不调整",
                    allow_relaxation=False,
                    note="Benchmark comparison is unavailable or incomplete.",
                    note_zh="基准对比数据缺失或不完整。",
                )
            )
            continue
        avg_excess = float(valid["excess_total_return"].mean())
        worst_excess = float(valid["excess_total_return"].min())
        avg_excess_annualized = float(valid["excess_annualized_return"].mean())
        avg_drawdown_advantage = float(valid["drawdown_advantage"].mean())
        worst_drawdown_advantage = float(valid["drawdown_advantage"].min())
        avg_correlation = float(valid["daily_correlation"].mean())
        action, action_zh, threshold_bias, threshold_bias_zh, allow_relaxation, note, note_zh = (
            _benchmark_policy_decision(
                avg_excess=avg_excess,
                worst_excess=worst_excess,
                avg_drawdown_advantage=avg_drawdown_advantage,
                worst_drawdown_advantage=worst_drawdown_advantage,
            )
        )
        rows.append(
            _benchmark_policy_row(
                portfolio_name=portfolio_name,
                portfolio_name_zh=portfolio_name_zh,
                benchmark_count=int(valid["benchmark_ticker"].nunique()),
                avg_excess_total_return=avg_excess,
                worst_excess_total_return=worst_excess,
                avg_excess_annualized_return=avg_excess_annualized,
                avg_drawdown_advantage=avg_drawdown_advantage,
                worst_drawdown_advantage=worst_drawdown_advantage,
                avg_daily_correlation=avg_correlation,
                action=action,
                action_zh=action_zh,
                threshold_bias=threshold_bias,
                threshold_bias_zh=threshold_bias_zh,
                allow_relaxation=allow_relaxation,
                note=note,
                note_zh=note_zh,
            )
        )
    return pd.DataFrame(rows, columns=columns)


def _benchmark_policy_decision(
    avg_excess: float,
    worst_excess: float,
    avg_drawdown_advantage: float,
    worst_drawdown_advantage: float,
) -> tuple[str, str, str, str, bool, str, str]:
    if (
        avg_excess < 0
        or worst_excess < -0.02
        or avg_drawdown_advantage < -0.02
        or worst_drawdown_advantage < -0.04
    ):
        return (
            "tighten_rules",
            "收紧规则",
            "tighten",
            "偏向收紧",
            False,
            "Portfolio validation is not beating benchmarks with acceptable drawdown.",
            "组合验证没有在可接受回撤下跑赢基准，优先收紧筛选门槛。",
        )
    if avg_excess >= 0.02 and worst_excess >= 0 and avg_drawdown_advantage >= 0:
        return (
            "allow_selective_relaxation",
            "允许谨慎放宽",
            "selective_relax",
            "可谨慎放宽",
            True,
            "Portfolio validation beats benchmarks and drawdown is not worse.",
            "组合验证跑赢基准且回撤不更差，可以谨慎接受高置信度放宽建议。",
        )
    return (
        "keep_rules",
        "维持规则",
        "hold",
        "维持",
        False,
        "Benchmark evidence is mixed, so avoid loosening thresholds.",
        "基准证据不够强，暂时不要放宽筛选门槛。",
    )


def _benchmark_policy_row(
    portfolio_name: object,
    portfolio_name_zh: object,
    benchmark_count: int,
    avg_excess_total_return: float,
    worst_excess_total_return: float,
    avg_excess_annualized_return: float,
    avg_drawdown_advantage: float,
    worst_drawdown_advantage: float,
    avg_daily_correlation: float,
    action: str,
    action_zh: str,
    threshold_bias: str,
    threshold_bias_zh: str,
    allow_relaxation: bool,
    note: str,
    note_zh: str,
) -> dict[str, object]:
    return {
        "portfolio_name": portfolio_name,
        "portfolio_name_zh": portfolio_name_zh,
        "benchmark_count": benchmark_count,
        "avg_excess_total_return": avg_excess_total_return,
        "worst_excess_total_return": worst_excess_total_return,
        "avg_excess_annualized_return": avg_excess_annualized_return,
        "avg_drawdown_advantage": avg_drawdown_advantage,
        "worst_drawdown_advantage": worst_drawdown_advantage,
        "avg_daily_correlation": avg_daily_correlation,
        "benchmark_passed": action == "allow_selective_relaxation",
        "benchmark_policy_action": action,
        "benchmark_policy_action_zh": action_zh,
        "threshold_bias": threshold_bias,
        "threshold_bias_zh": threshold_bias_zh,
        "allow_relaxation": bool(allow_relaxation),
        "policy_note": note,
        "policy_note_zh": note_zh,
    }


def build_benchmark_tightening_recommendations(
    events: pd.DataFrame,
    screening_config: ScreeningConfig,
    profile_calibration: pd.DataFrame,
    benchmark_policy: pd.DataFrame,
    target_window: int = 20,
) -> pd.DataFrame:
    columns = [
        "screening_profile",
        "screening_profile_zh",
        "rule",
        "threshold_attr",
        "current_threshold",
        "suggested_threshold",
        "tightening_amount",
        "priority",
        "priority_zh",
        "benchmark_policy_action",
        "sample_count",
        f"win_rate_{target_window}d",
        f"avg_return_{target_window}d",
        "recommendation_reason",
        "recommendation_reason_zh",
    ]
    if (
        events.empty
        or benchmark_policy.empty
        or "benchmark_policy_action" not in benchmark_policy.columns
        or not (benchmark_policy["benchmark_policy_action"].astype(str) == "tighten_rules").any()
    ):
        return pd.DataFrame(columns=columns)

    config_thresholds = {
        "default": screening_config.default_thresholds,
        **{profile.name: profile.thresholds for profile in screening_config.profiles},
    }
    profile_names_zh = {
        "default": "默认规则",
        **{profile.name: profile.name_zh for profile in screening_config.profiles},
    }
    specs = [
        {
            "rule": "signal_score",
            "threshold_attr": "signal_score_min",
            "candidates": DEFAULT_SIGNAL_THRESHOLDS,
            "reason": "Raise the minimum technical signal score before accepting candidates.",
            "reason_zh": "提高最低技术信号分，减少弱信号进入候选。",
        },
        {
            "rule": "high_probability_score",
            "threshold_attr": "high_probability_target_score",
            "candidates": DEFAULT_HIGH_PROBABILITY_THRESHOLDS,
            "reason": "Raise the final high-probability score requirement.",
            "reason_zh": "提高最终高概率分数要求。",
        },
        {
            "rule": "relative_strength_score",
            "threshold_attr": "relative_strength_min",
            "candidates": DEFAULT_RELATIVE_STRENGTH_THRESHOLDS,
            "reason": "Require stronger performance versus SPY and QQQ.",
            "reason_zh": "要求股票相对SPY和QQQ表现更强。",
        },
        {
            "rule": "market_score",
            "threshold_attr": "market_score_min",
            "candidates": DEFAULT_MARKET_THRESHOLDS,
            "reason": "Require a healthier market backdrop before new candidates pass.",
            "reason_zh": "要求更健康的大盘环境才允许候选通过。",
        },
        {
            "rule": "screening_backtest_win_rate",
            "threshold_attr": "backtest_win_rate_min",
            "candidates": (0.55, 0.57, 0.60, 0.62, 0.65),
            "reason": "Require stronger historical entry win-rate evidence.",
            "reason_zh": "要求历史买点回测胜率更高。",
        },
    ]

    calibration_lookup = _profile_calibration_lookup(profile_calibration)
    rows: list[dict[str, object]] = []
    grouped_profiles = events.groupby("screening_profile", dropna=False)
    for profile_name, group in grouped_profiles:
        profile_key = str(profile_name)
        thresholds = config_thresholds.get(profile_key, screening_config.default_thresholds)
        profile_zh = profile_names_zh.get(
            profile_key,
            _first_text(group, "screening_profile_zh", "默认规则"),
        )
        for spec in specs:
            threshold_attr = str(spec["threshold_attr"])
            current_threshold = float(getattr(thresholds, threshold_attr))
            suggested_threshold = _specific_tightening_threshold(
                current_threshold=current_threshold,
                candidates=tuple(spec["candidates"]),
                calibration_row=calibration_lookup.get((profile_key, str(spec["rule"]))),
            )
            if not _is_finite(suggested_threshold) or suggested_threshold <= current_threshold:
                continue
            sample_count, win_rate, avg_return = _rule_validation_metrics(
                group=group,
                rule=str(spec["rule"]),
                threshold=suggested_threshold,
                target_window=target_window,
            )
            priority, priority_zh = _tightening_priority(
                rule=str(spec["rule"]),
                sample_count=sample_count,
                win_rate=win_rate,
                avg_return=avg_return,
            )
            rows.append(
                {
                    "screening_profile": profile_key,
                    "screening_profile_zh": profile_zh,
                    "rule": spec["rule"],
                    "threshold_attr": threshold_attr,
                    "current_threshold": current_threshold,
                    "suggested_threshold": suggested_threshold,
                    "tightening_amount": suggested_threshold - current_threshold,
                    "priority": priority,
                    "priority_zh": priority_zh,
                    "benchmark_policy_action": "tighten_rules",
                    "sample_count": sample_count,
                    f"win_rate_{target_window}d": win_rate,
                    f"avg_return_{target_window}d": avg_return,
                    "recommendation_reason": spec["reason"],
                    "recommendation_reason_zh": spec["reason_zh"],
                }
            )

    result = pd.DataFrame(rows, columns=columns)
    if result.empty:
        return result
    priority_order = {"high": 0, "medium": 1, "low": 2}
    result["_priority_order"] = result["priority"].map(priority_order).fillna(3)
    result = result.sort_values(
        ["screening_profile", "_priority_order", "tightening_amount"],
        ascending=[True, True, False],
    ).drop(columns=["_priority_order"])
    return result.reset_index(drop=True)


def _profile_calibration_lookup(profile_calibration: pd.DataFrame) -> dict[tuple[str, str], object]:
    if profile_calibration.empty or not {"screening_profile", "rule"}.issubset(profile_calibration.columns):
        return {}
    return {
        (str(row.screening_profile), str(row.rule)): row
        for row in profile_calibration.itertuples(index=False)
    }


def _specific_tightening_threshold(
    current_threshold: float,
    candidates: tuple[float, ...],
    calibration_row: object | None,
) -> float:
    calibrated = np.nan
    if calibration_row is not None:
        calibrated = float(getattr(calibration_row, "suggested_threshold", np.nan))
    if _is_finite(calibrated) and calibrated > current_threshold:
        return calibrated
    stricter = [float(candidate) for candidate in candidates if float(candidate) > current_threshold]
    if stricter:
        return min(stricter)
    if current_threshold < 1:
        return min(current_threshold + 0.02, 0.75)
    return min(current_threshold + 2.0, 95.0)


def _rule_validation_metrics(
    group: pd.DataFrame,
    rule: str,
    threshold: float,
    target_window: int,
) -> tuple[int, float, float]:
    return_column = f"forward_return_{target_window}d"
    if return_column not in group.columns:
        return 0, np.nan, np.nan
    if rule == "screening_backtest_win_rate":
        column = "screening_backtest_win_rate"
    else:
        column = rule
    if column not in group.columns:
        return 0, np.nan, np.nan
    frame = group.copy()
    frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame[return_column] = pd.to_numeric(frame[return_column], errors="coerce")
    subset = frame[frame[column] >= threshold]
    returns = subset[return_column].dropna()
    if returns.empty:
        return 0, np.nan, np.nan
    return int(len(returns)), float((returns > 0).mean()), float(returns.mean())


def _tightening_priority(
    rule: str,
    sample_count: int,
    win_rate: float,
    avg_return: float,
) -> tuple[str, str]:
    core_rules = {"signal_score", "high_probability_score", "relative_strength_score"}
    if sample_count >= 10 and _is_finite(avg_return) and avg_return > 0 and rule in core_rules:
        return "high", "高"
    if sample_count >= 5 and _is_finite(win_rate):
        return "medium", "中"
    return "low", "低"


def build_tightening_impact_validation(
    events: pd.DataFrame,
    benchmark_tightening: pd.DataFrame,
    target_window: int = 20,
    active_priorities: tuple[str, ...] = ("high", "medium"),
) -> pd.DataFrame:
    columns = [
        "screening_profile",
        "screening_profile_zh",
        "active_rule_count",
        "active_rules",
        "active_rules_zh",
        "before_sample_count",
        "after_sample_count",
        "sample_retention_rate",
        "before_win_rate",
        "after_win_rate",
        "win_rate_change",
        "before_avg_return",
        "after_avg_return",
        "avg_return_change",
        "before_avg_drawdown",
        "after_avg_drawdown",
        "drawdown_change",
        "impact_decision",
        "impact_decision_zh",
        "impact_note",
        "impact_note_zh",
    ]
    return_column = f"forward_return_{target_window}d"
    if (
        events.empty
        or benchmark_tightening.empty
        or return_column not in events.columns
        or "screening_profile" not in events.columns
    ):
        return pd.DataFrame(columns=columns)

    tightening = benchmark_tightening.copy()
    tightening = tightening[tightening["priority"].astype(str).isin(active_priorities)].copy()
    if tightening.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for profile_name, rules in tightening.groupby("screening_profile", dropna=False):
        profile_key = str(profile_name)
        profile_events = events[events["screening_profile"].astype(str) == profile_key].copy()
        if profile_events.empty:
            continue
        actionable_events = _actionable_validation_events(profile_events)
        after_events = _apply_tightening_rules(actionable_events, rules)
        before_metrics = _validation_subset_metrics(actionable_events, target_window)
        after_metrics = _validation_subset_metrics(after_events, target_window)
        decision = _tightening_impact_decision(
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            active_rule_count=len(rules),
        )
        active_rules = [
            f"{row.threshold_attr}>={float(row.suggested_threshold):g}"
            for row in rules.itertuples(index=False)
        ]
        active_rules_zh = [
            f"{_threshold_attr_zh(str(row.threshold_attr))}>={float(row.suggested_threshold):g}"
            for row in rules.itertuples(index=False)
        ]
        rows.append(
            {
                "screening_profile": profile_key,
                "screening_profile_zh": _first_text(profile_events, "screening_profile_zh", profile_key),
                "active_rule_count": int(len(rules)),
                "active_rules": "; ".join(active_rules),
                "active_rules_zh": "；".join(active_rules_zh),
                "before_sample_count": before_metrics["sample_count"],
                "after_sample_count": after_metrics["sample_count"],
                "sample_retention_rate": (
                    after_metrics["sample_count"] / before_metrics["sample_count"]
                    if before_metrics["sample_count"]
                    else np.nan
                ),
                "before_win_rate": before_metrics["win_rate"],
                "after_win_rate": after_metrics["win_rate"],
                "win_rate_change": _metric_change(after_metrics["win_rate"], before_metrics["win_rate"]),
                "before_avg_return": before_metrics["avg_return"],
                "after_avg_return": after_metrics["avg_return"],
                "avg_return_change": _metric_change(after_metrics["avg_return"], before_metrics["avg_return"]),
                "before_avg_drawdown": before_metrics["avg_drawdown"],
                "after_avg_drawdown": after_metrics["avg_drawdown"],
                "drawdown_change": _metric_change(after_metrics["avg_drawdown"], before_metrics["avg_drawdown"]),
                "impact_decision": decision["decision"],
                "impact_decision_zh": decision["decision_zh"],
                "impact_note": decision["note"],
                "impact_note_zh": decision["note_zh"],
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _actionable_validation_events(events: pd.DataFrame) -> pd.DataFrame:
    if "validation_bucket" not in events.columns:
        return events.copy()
    actionable_buckets = {"high_probability", "near_watchlist", "early_watchlist"}
    subset = events[events["validation_bucket"].isin(actionable_buckets)].copy()
    return subset if not subset.empty else events.copy()


def _apply_tightening_rules(events: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    result = events.copy()
    for row in rules.itertuples(index=False):
        threshold_attr = str(row.threshold_attr)
        event_column = _event_column_for_threshold_attr(threshold_attr)
        if event_column not in result.columns:
            continue
        threshold = float(row.suggested_threshold)
        result[event_column] = pd.to_numeric(result[event_column], errors="coerce")
        result = result[result[event_column] >= threshold].copy()
        if result.empty:
            break
    return result


def _event_column_for_threshold_attr(threshold_attr: str) -> str:
    return {
        "signal_score_min": "signal_score",
        "high_probability_target_score": "high_probability_score",
        "relative_strength_min": "relative_strength_score",
        "market_score_min": "market_score",
        "backtest_win_rate_min": "screening_backtest_win_rate",
    }.get(threshold_attr, threshold_attr)


def _threshold_attr_zh(threshold_attr: str) -> str:
    return {
        "signal_score_min": "信号分",
        "high_probability_target_score": "高概率分",
        "relative_strength_min": "相对强弱",
        "market_score_min": "大盘环境分",
        "backtest_win_rate_min": "买点回测胜率",
    }.get(threshold_attr, threshold_attr)


def _validation_subset_metrics(events: pd.DataFrame, target_window: int) -> dict[str, float | int]:
    return_column = f"forward_return_{target_window}d"
    if events.empty or return_column not in events.columns:
        return {
            "sample_count": 0,
            "win_rate": np.nan,
            "avg_return": np.nan,
            "avg_drawdown": np.nan,
        }
    returns = pd.to_numeric(events[return_column], errors="coerce").dropna()
    drawdowns = (
        pd.to_numeric(events["max_drawdown_after_signal"], errors="coerce").dropna()
        if "max_drawdown_after_signal" in events.columns
        else pd.Series(dtype=float)
    )
    return {
        "sample_count": int(len(returns)),
        "win_rate": float((returns > 0).mean()) if len(returns) else np.nan,
        "avg_return": float(returns.mean()) if len(returns) else np.nan,
        "avg_drawdown": float(drawdowns.mean()) if len(drawdowns) else np.nan,
    }


def _tightening_impact_decision(
    before_metrics: dict[str, float | int],
    after_metrics: dict[str, float | int],
    active_rule_count: int,
) -> dict[str, str]:
    before_count = int(before_metrics["sample_count"])
    after_count = int(after_metrics["sample_count"])
    if active_rule_count == 0:
        return {
            "decision": "no_active_rules",
            "decision_zh": "没有有效收紧规则",
            "note": "No medium/high priority tightening rules were active.",
            "note_zh": "没有中高优先级的有效收紧规则。",
        }
    if before_count == 0 or after_count == 0:
        return {
            "decision": "insufficient_after_samples",
            "decision_zh": "收紧后样本不足",
            "note": "Tightening removed too many validation samples.",
            "note_zh": "收紧后剩余验证样本太少，暂时不能证明有效。",
        }
    if after_count < 5:
        return {
            "decision": "sample_too_small",
            "decision_zh": "样本偏少",
            "note": "After-tightening sample count is small; treat results as directional only.",
            "note_zh": "收紧后样本偏少，只能作为方向参考。",
        }
    win_change = _metric_change(after_metrics["win_rate"], before_metrics["win_rate"])
    return_change = _metric_change(after_metrics["avg_return"], before_metrics["avg_return"])
    drawdown_change = _metric_change(after_metrics["avg_drawdown"], before_metrics["avg_drawdown"])
    if (
        _is_finite(win_change)
        and _is_finite(return_change)
        and win_change >= 0
        and return_change > 0
        and (not _is_finite(drawdown_change) or drawdown_change >= -0.01)
    ):
        return {
            "decision": "validated_improvement",
            "decision_zh": "验证显示改善",
            "note": "Tightening improved average return without lowering win rate.",
            "note_zh": "收紧后平均收益改善，且胜率没有下降。",
        }
    if _is_finite(return_change) and return_change <= 0 and _is_finite(win_change) and win_change < 0:
        return {
            "decision": "not_improved",
            "decision_zh": "未改善",
            "note": "Tightening reduced both win rate and average return in validation.",
            "note_zh": "收紧后胜率和平均收益都下降，暂不应直接采用。",
        }
    return {
        "decision": "mixed_result",
        "decision_zh": "结果混合",
        "note": "Tightening changed the sample, but improvement is not decisive.",
        "note_zh": "收紧改变了样本，但改善证据还不够明确。",
    }


def _metric_change(after_value: object, before_value: object) -> float:
    if not _is_finite(after_value) or not _is_finite(before_value):
        return np.nan
    return float(after_value) - float(before_value)


def build_threshold_sensitivity_grid(
    events: pd.DataFrame,
    screening_config: ScreeningConfig,
    target_window: int = 20,
) -> pd.DataFrame:
    columns = [
        "screening_profile",
        "screening_profile_zh",
        "grid_type",
        "rule_set",
        "rule_set_zh",
        "threshold_expression",
        "threshold_expression_zh",
        "baseline_sample_count",
        "sample_count",
        "sample_retention_rate",
        "win_rate",
        "avg_return",
        "avg_drawdown",
        "win_rate_change",
        "avg_return_change",
        "drawdown_change",
        "sensitivity_score",
        "sensitivity_decision",
        "sensitivity_decision_zh",
        "sensitivity_note",
        "sensitivity_note_zh",
    ]
    return_column = f"forward_return_{target_window}d"
    if events.empty or return_column not in events.columns or "screening_profile" not in events.columns:
        return pd.DataFrame(columns=columns)

    config_thresholds = {
        "default": screening_config.default_thresholds,
        **{profile.name: profile.thresholds for profile in screening_config.profiles},
    }
    profile_names_zh = {
        "default": "默认规则",
        **{profile.name: profile.name_zh for profile in screening_config.profiles},
    }
    specs = _threshold_sensitivity_specs()
    rows: list[dict[str, object]] = []

    for profile_name, group in events.groupby("screening_profile", dropna=False):
        profile_key = str(profile_name)
        thresholds = config_thresholds.get(profile_key, screening_config.default_thresholds)
        profile_zh = profile_names_zh.get(
            profile_key,
            _first_text(group, "screening_profile_zh", profile_key),
        )
        actionable = _actionable_validation_events(group)
        baseline_metrics = _validation_subset_metrics(actionable, target_window)
        if int(baseline_metrics["sample_count"]) == 0:
            continue

        rows.append(
            _threshold_sensitivity_row(
                profile_name=profile_key,
                profile_name_zh=profile_zh,
                grid_type="baseline",
                rule_set="baseline",
                rule_set_zh="基准",
                threshold_expression="current_rules",
                threshold_expression_zh="当前规则",
                baseline_metrics=baseline_metrics,
                subset_metrics=baseline_metrics,
                note="Current actionable validation sample before additional threshold tests.",
                note_zh="当前可操作验证样本，尚未额外提高门槛。",
            )
        )

        single_rule_filters: list[tuple[str, str, float]] = []
        for spec in specs:
            threshold_attr = str(spec["threshold_attr"])
            current_threshold = float(getattr(thresholds, threshold_attr))
            candidates = _candidate_values_above_current(tuple(spec["candidates"]), current_threshold)
            for candidate in candidates:
                filters = [(threshold_attr, str(spec["event_column"]), candidate)]
                subset = _apply_threshold_filters(actionable, filters)
                metrics = _validation_subset_metrics(subset, target_window)
                rows.append(
                    _threshold_sensitivity_row(
                        profile_name=profile_key,
                        profile_name_zh=profile_zh,
                        grid_type="single_rule",
                        rule_set=str(spec["rule"]),
                        rule_set_zh=str(spec["rule_zh"]),
                        threshold_expression=f"{threshold_attr}>={candidate:g}",
                        threshold_expression_zh=f"{spec['threshold_attr_zh']}>={candidate:g}",
                        baseline_metrics=baseline_metrics,
                        subset_metrics=metrics,
                        note=str(spec["note"]),
                        note_zh=str(spec["note_zh"]),
                    )
                )
                single_rule_filters.append((threshold_attr, str(spec["event_column"]), candidate))

        pair_specs = [
            ("signal_score_min", "relative_strength_min"),
            ("signal_score_min", "market_score_min"),
            ("relative_strength_min", "market_score_min"),
            ("high_probability_target_score", "relative_strength_min"),
        ]
        spec_by_attr = {str(spec["threshold_attr"]): spec for spec in specs}
        for first_attr, second_attr in pair_specs:
            first_spec = spec_by_attr[first_attr]
            second_spec = spec_by_attr[second_attr]
            first_candidates = _candidate_values_above_current(
                tuple(first_spec["candidates"]),
                float(getattr(thresholds, first_attr)),
            )[:2]
            second_candidates = _candidate_values_above_current(
                tuple(second_spec["candidates"]),
                float(getattr(thresholds, second_attr)),
            )[:2]
            for first_candidate in first_candidates:
                for second_candidate in second_candidates:
                    filters = [
                        (first_attr, str(first_spec["event_column"]), first_candidate),
                        (second_attr, str(second_spec["event_column"]), second_candidate),
                    ]
                    subset = _apply_threshold_filters(actionable, filters)
                    metrics = _validation_subset_metrics(subset, target_window)
                    rows.append(
                        _threshold_sensitivity_row(
                            profile_name=profile_key,
                            profile_name_zh=profile_zh,
                            grid_type="pair_rules",
                            rule_set=f"{first_spec['rule']}+{second_spec['rule']}",
                            rule_set_zh=f"{first_spec['rule_zh']}+{second_spec['rule_zh']}",
                            threshold_expression=(
                                f"{first_attr}>={first_candidate:g}; "
                                f"{second_attr}>={second_candidate:g}"
                            ),
                            threshold_expression_zh=(
                                f"{first_spec['threshold_attr_zh']}>={first_candidate:g}；"
                                f"{second_spec['threshold_attr_zh']}>={second_candidate:g}"
                            ),
                            baseline_metrics=baseline_metrics,
                            subset_metrics=metrics,
                            note="Tests a combined threshold filter.",
                            note_zh="测试两个门槛同时收紧的组合效果。",
                        )
                    )

    result = pd.DataFrame(rows, columns=columns)
    if result.empty:
        return result
    result["_decision_order"] = result["sensitivity_decision"].map(
        {
            "promising_threshold": 0,
            "baseline": 1,
            "mixed_threshold": 2,
            "insufficient_samples": 3,
            "reject_threshold": 4,
        }
    ).fillna(5)
    result = result.sort_values(
        ["screening_profile", "_decision_order", "sensitivity_score", "sample_count"],
        ascending=[True, True, False, False],
    ).drop(columns=["_decision_order"])
    return result.reset_index(drop=True)


def _threshold_sensitivity_specs() -> list[dict[str, object]]:
    return [
        {
            "rule": "signal_score",
            "rule_zh": "信号分",
            "threshold_attr": "signal_score_min",
            "threshold_attr_zh": "信号分",
            "event_column": "signal_score",
            "candidates": DEFAULT_SIGNAL_THRESHOLDS,
            "note": "Tests stricter technical signal score requirements.",
            "note_zh": "测试更严格的技术信号分要求。",
        },
        {
            "rule": "high_probability_score",
            "rule_zh": "高概率分",
            "threshold_attr": "high_probability_target_score",
            "threshold_attr_zh": "高概率分",
            "event_column": "high_probability_score",
            "candidates": DEFAULT_HIGH_PROBABILITY_THRESHOLDS,
            "note": "Tests stricter final high-probability score requirements.",
            "note_zh": "测试更严格的最终高概率分要求。",
        },
        {
            "rule": "relative_strength_score",
            "rule_zh": "相对强弱",
            "threshold_attr": "relative_strength_min",
            "threshold_attr_zh": "相对强弱",
            "event_column": "relative_strength_score",
            "candidates": DEFAULT_RELATIVE_STRENGTH_THRESHOLDS,
            "note": "Tests stronger relative performance filters.",
            "note_zh": "测试更强的相对表现过滤。",
        },
        {
            "rule": "market_score",
            "rule_zh": "大盘环境",
            "threshold_attr": "market_score_min",
            "threshold_attr_zh": "大盘环境分",
            "event_column": "market_score",
            "candidates": DEFAULT_MARKET_THRESHOLDS,
            "note": "Tests stricter market backdrop filters.",
            "note_zh": "测试更严格的大盘环境过滤。",
        },
        {
            "rule": "screening_backtest_win_rate",
            "rule_zh": "买点回测胜率",
            "threshold_attr": "backtest_win_rate_min",
            "threshold_attr_zh": "买点回测胜率",
            "event_column": "screening_backtest_win_rate",
            "candidates": (0.55, 0.57, 0.60, 0.62, 0.65),
            "note": "Tests stricter historical entry win-rate requirements.",
            "note_zh": "测试更严格的历史买点胜率要求。",
        },
    ]


def _candidate_values_above_current(candidates: tuple[float, ...], current_threshold: float) -> list[float]:
    values = [float(candidate) for candidate in candidates if float(candidate) > current_threshold]
    if values:
        return values
    if current_threshold < 1:
        return [min(current_threshold + 0.02, 0.75)]
    return [min(current_threshold + 2.0, 95.0)]


def _apply_threshold_filters(
    events: pd.DataFrame,
    filters: list[tuple[str, str, float]],
) -> pd.DataFrame:
    result = events.copy()
    for _threshold_attr, event_column, threshold in filters:
        if event_column not in result.columns:
            return result.iloc[0:0].copy()
        result[event_column] = pd.to_numeric(result[event_column], errors="coerce")
        result = result[result[event_column] >= threshold].copy()
        if result.empty:
            break
    return result


def _threshold_sensitivity_row(
    profile_name: str,
    profile_name_zh: str,
    grid_type: str,
    rule_set: str,
    rule_set_zh: str,
    threshold_expression: str,
    threshold_expression_zh: str,
    baseline_metrics: dict[str, float | int],
    subset_metrics: dict[str, float | int],
    note: str,
    note_zh: str,
) -> dict[str, object]:
    baseline_count = int(baseline_metrics["sample_count"])
    sample_count = int(subset_metrics["sample_count"])
    win_rate_change = _metric_change(subset_metrics["win_rate"], baseline_metrics["win_rate"])
    avg_return_change = _metric_change(subset_metrics["avg_return"], baseline_metrics["avg_return"])
    drawdown_change = _metric_change(subset_metrics["avg_drawdown"], baseline_metrics["avg_drawdown"])
    score = _threshold_sensitivity_score(
        win_rate_change=win_rate_change,
        avg_return_change=avg_return_change,
        drawdown_change=drawdown_change,
        sample_count=sample_count,
        baseline_count=baseline_count,
    )
    decision = _threshold_sensitivity_decision(
        grid_type=grid_type,
        sample_count=sample_count,
        win_rate_change=win_rate_change,
        avg_return_change=avg_return_change,
        drawdown_change=drawdown_change,
    )
    return {
        "screening_profile": profile_name,
        "screening_profile_zh": profile_name_zh,
        "grid_type": grid_type,
        "rule_set": rule_set,
        "rule_set_zh": rule_set_zh,
        "threshold_expression": threshold_expression,
        "threshold_expression_zh": threshold_expression_zh,
        "baseline_sample_count": baseline_count,
        "sample_count": sample_count,
        "sample_retention_rate": sample_count / baseline_count if baseline_count else np.nan,
        "win_rate": subset_metrics["win_rate"],
        "avg_return": subset_metrics["avg_return"],
        "avg_drawdown": subset_metrics["avg_drawdown"],
        "win_rate_change": win_rate_change,
        "avg_return_change": avg_return_change,
        "drawdown_change": drawdown_change,
        "sensitivity_score": score,
        "sensitivity_decision": decision["decision"],
        "sensitivity_decision_zh": decision["decision_zh"],
        "sensitivity_note": decision["note"] or note,
        "sensitivity_note_zh": decision["note_zh"] or note_zh,
    }


def _threshold_sensitivity_score(
    win_rate_change: float,
    avg_return_change: float,
    drawdown_change: float,
    sample_count: int,
    baseline_count: int,
) -> float:
    if sample_count <= 0:
        return np.nan
    retention = sample_count / baseline_count if baseline_count else 0.0
    score = 0.0
    score += (win_rate_change if _is_finite(win_rate_change) else 0.0) * 100.0
    score += (avg_return_change if _is_finite(avg_return_change) else 0.0) * 100.0
    score += (drawdown_change if _is_finite(drawdown_change) else 0.0) * 50.0
    score += min(retention, 1.0) * 5.0
    return float(score)


def _threshold_sensitivity_decision(
    grid_type: str,
    sample_count: int,
    win_rate_change: float,
    avg_return_change: float,
    drawdown_change: float,
) -> dict[str, str]:
    if grid_type == "baseline":
        return {
            "decision": "baseline",
            "decision_zh": "当前基准",
            "note": "Current actionable validation sample.",
            "note_zh": "当前可操作验证样本。",
        }
    if sample_count < 5:
        return {
            "decision": "insufficient_samples",
            "decision_zh": "样本不足",
            "note": "Threshold leaves too few validation samples.",
            "note_zh": "该门槛留下的验证样本太少。",
        }
    if (
        _is_finite(avg_return_change)
        and avg_return_change > 0
        and _is_finite(win_rate_change)
        and win_rate_change >= 0
        and (not _is_finite(drawdown_change) or drawdown_change >= -0.01)
    ):
        return {
            "decision": "promising_threshold",
            "decision_zh": "值得进一步验证",
            "note": "Threshold improved average return without reducing win rate.",
            "note_zh": "该门槛提高了平均收益，且没有降低胜率。",
        }
    if (
        _is_finite(avg_return_change)
        and avg_return_change <= 0
        and _is_finite(win_rate_change)
        and win_rate_change < 0
    ):
        return {
            "decision": "reject_threshold",
            "decision_zh": "不建议采用",
            "note": "Threshold reduced both win rate and average return.",
            "note_zh": "该门槛同时降低胜率和平均收益。",
        }
    return {
        "decision": "mixed_threshold",
        "decision_zh": "结果混合",
        "note": "Threshold changes are mixed and need more evidence.",
        "note_zh": "该门槛结果混合，需要更多证据。",
    }


def build_minimum_sample_guard(
    threshold_sensitivity: pd.DataFrame,
    min_sample_count: int = 10,
    min_retention_rate: float = 0.30,
) -> pd.DataFrame:
    columns = [
        "screening_profile",
        "screening_profile_zh",
        "threshold_attr",
        "threshold_value",
        "threshold_expression",
        "threshold_expression_zh",
        "sample_count",
        "baseline_sample_count",
        "sample_retention_rate",
        "sensitivity_decision",
        "guard_action",
        "guard_action_zh",
        "allow_adoption",
        "guard_reason",
        "guard_reason_zh",
    ]
    required = {
        "screening_profile",
        "screening_profile_zh",
        "grid_type",
        "threshold_expression",
        "threshold_expression_zh",
        "sample_count",
        "baseline_sample_count",
        "sample_retention_rate",
        "sensitivity_decision",
    }
    if threshold_sensitivity.empty or not required.issubset(threshold_sensitivity.columns):
        return pd.DataFrame(columns=columns)

    frame = threshold_sensitivity[
        threshold_sensitivity["grid_type"].isin(["single_rule", "pair_rules"])
    ].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        sample_count = int(getattr(row, "sample_count", 0))
        baseline_count = int(getattr(row, "baseline_sample_count", 0))
        retention = float(getattr(row, "sample_retention_rate", np.nan))
        decision = str(getattr(row, "sensitivity_decision", ""))
        guard = _minimum_sample_guard_decision(
            sample_count=sample_count,
            retention=retention,
            sensitivity_decision=decision,
            min_sample_count=min_sample_count,
            min_retention_rate=min_retention_rate,
        )
        threshold_attr, threshold_value = _first_threshold_from_expression(
            str(getattr(row, "threshold_expression", ""))
        )
        rows.append(
            {
                "screening_profile": row.screening_profile,
                "screening_profile_zh": row.screening_profile_zh,
                "threshold_attr": threshold_attr,
                "threshold_value": threshold_value,
                "threshold_expression": row.threshold_expression,
                "threshold_expression_zh": row.threshold_expression_zh,
                "sample_count": sample_count,
                "baseline_sample_count": baseline_count,
                "sample_retention_rate": retention,
                "sensitivity_decision": decision,
                "guard_action": guard["action"],
                "guard_action_zh": guard["action_zh"],
                "allow_adoption": guard["allow_adoption"],
                "guard_reason": guard["reason"],
                "guard_reason_zh": guard["reason_zh"],
            }
        )
    result = pd.DataFrame(rows, columns=columns)
    action_order = {"allow_adoption": 0, "watch_only": 1, "block_adoption": 2}
    result["_action_order"] = result["guard_action"].map(action_order).fillna(3)
    result = result.sort_values(
        ["screening_profile", "_action_order", "sample_count"],
        ascending=[True, True, False],
    ).drop(columns=["_action_order"])
    return result.reset_index(drop=True)


def _minimum_sample_guard_decision(
    sample_count: int,
    retention: float,
    sensitivity_decision: str,
    min_sample_count: int,
    min_retention_rate: float,
) -> dict[str, object]:
    if sample_count < min_sample_count:
        return {
            "action": "block_adoption",
            "action_zh": "禁止采用",
            "allow_adoption": False,
            "reason": f"Sample count is below the minimum guard of {min_sample_count}.",
            "reason_zh": f"样本数低于最小保护要求{min_sample_count}。",
        }
    if _is_finite(retention) and retention < min_retention_rate:
        return {
            "action": "block_adoption",
            "action_zh": "禁止采用",
            "allow_adoption": False,
            "reason": f"Sample retention is below {min_retention_rate:.0%}.",
            "reason_zh": f"样本保留率低于{min_retention_rate:.0%}。",
        }
    if sensitivity_decision == "promising_threshold":
        return {
            "action": "allow_adoption",
            "action_zh": "允许采用",
            "allow_adoption": True,
            "reason": "Sample count is sufficient and validation improvement is promising.",
            "reason_zh": "样本数足够，且验证改善较明确。",
        }
    return {
        "action": "watch_only",
        "action_zh": "仅观察",
        "allow_adoption": False,
        "reason": "Sample count is sufficient, but improvement is not decisive.",
        "reason_zh": "样本数足够，但改善证据不够明确。",
    }


def _first_threshold_from_expression(expression: str) -> tuple[str, float]:
    first = expression.split(";")[0].strip()
    if ">=" not in first:
        return first, np.nan
    key, value = first.split(">=", 1)
    try:
        return key.strip(), float(value.strip())
    except ValueError:
        return key.strip(), np.nan


def _probability_calibration_quality(sample_count: int, error_abs: float) -> tuple[str, str]:
    if sample_count < 5 or not _is_finite(error_abs):
        return "insufficient", "样本不足"
    if error_abs <= 0.05:
        return "well_calibrated", "校准较好"
    if error_abs <= 0.10:
        return "acceptable", "可接受"
    return "miscalibrated", "偏差较大"


def _probability_formula_action(
    sample_count: int,
    error: float,
    error_abs: float,
) -> tuple[str, str]:
    if sample_count < 5 or not _is_finite(error) or not _is_finite(error_abs):
        return "collect_more_samples", "继续收集样本"
    if error > 0.08:
        return "raise_probability_estimate", "上调胜率估计"
    if error < -0.08:
        return "lower_probability_estimate", "下调胜率估计"
    if error_abs <= 0.05:
        return "keep_probability_formula", "保持概率公式"
    return "monitor_probability_formula", "继续观察概率公式"


def _overall_probability_calibration_row(
    frame: pd.DataFrame,
    target_window: int,
    total_samples: int,
) -> dict[str, object]:
    return_column = f"forward_return_{target_window}d"
    returns = frame[return_column].dropna()
    if returns.empty:
        return {
            "probability_bucket": "overall",
            "probability_bucket_zh": "整体",
            "sample_count": 0,
            "avg_estimated_probability": np.nan,
            f"actual_win_rate_{target_window}d": np.nan,
            f"avg_return_{target_window}d": np.nan,
            "calibration_error": np.nan,
            "calibration_error_abs": np.nan,
            "recommended_probability_adjustment": np.nan,
            "adjusted_estimated_probability": np.nan,
            "adjusted_calibration_error_abs": np.nan,
            "brier_score": np.nan,
            "expected_calibration_error_component": np.nan,
            "formula_action": "collect_more_samples",
            "formula_action_zh": "继续收集样本",
            "calibration_quality": "insufficient",
            "calibration_quality_zh": "样本不足",
            "calibration_note": "No validation samples are available.",
            "calibration_note_zh": "没有可用验证样本。",
        }

    estimated = float(frame.loc[returns.index, "calibrated_win_probability"].mean())
    actual = float((returns > 0).mean())
    avg_return = float(returns.mean())
    error = actual - estimated
    adjustment = min(max(error, -0.10), 0.10)
    adjusted_estimated = min(max(estimated + adjustment, 0.05), 0.90)
    adjusted_error_abs = abs(actual - adjusted_estimated)
    outcomes = (returns > 0).astype(float)
    brier_score = float(
        ((outcomes - frame.loc[returns.index, "calibrated_win_probability"]) ** 2).mean()
    )
    quality, quality_zh = _probability_calibration_quality(
        sample_count=len(returns),
        error_abs=abs(error),
    )
    formula_action, formula_action_zh = _probability_formula_action(
        sample_count=len(returns),
        error=error,
        error_abs=abs(error),
    )
    return {
        "probability_bucket": "overall",
        "probability_bucket_zh": "整体",
        "sample_count": int(len(returns)),
        "avg_estimated_probability": estimated,
        f"actual_win_rate_{target_window}d": actual,
        f"avg_return_{target_window}d": avg_return,
        "calibration_error": error,
        "calibration_error_abs": abs(error),
        "recommended_probability_adjustment": adjustment,
        "adjusted_estimated_probability": adjusted_estimated,
        "adjusted_calibration_error_abs": adjusted_error_abs,
        "brier_score": brier_score,
        "expected_calibration_error_component": abs(error)
        * len(returns)
        / total_samples
        if total_samples
        else np.nan,
        "formula_action": formula_action,
        "formula_action_zh": formula_action_zh,
        "calibration_quality": quality,
        "calibration_quality_zh": quality_zh,
        "calibration_note": (
            f"Overall estimated win probability averaged {estimated:.2%}; "
            f"actual {target_window}d win rate was {actual:.2%}. "
            f"Suggested global probability adjustment is {adjustment:+.2%}."
        ),
        "calibration_note_zh": (
            f"整体估计胜率平均为{estimated:.2%}；"
            f"未来{target_window}日实际胜率为{actual:.2%}。"
            f"建议整体概率修正为{adjustment:+.2%}。"
        ),
    }


def calibrate_walk_forward_rules(
    events: pd.DataFrame,
    target_window: int = 20,
    min_sample_count: int = 5,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["rule", "current_threshold", "suggested_threshold"])

    specs = [
        ("signal_score", 65.0, DEFAULT_SIGNAL_THRESHOLDS, "higher"),
        ("high_probability_score", 65.0, DEFAULT_HIGH_PROBABILITY_THRESHOLDS, "higher"),
        ("calibrated_win_probability", 0.58, DEFAULT_PROBABILITY_THRESHOLDS, "higher"),
        ("relative_strength_score", 45.0, DEFAULT_RELATIVE_STRENGTH_THRESHOLDS, "higher"),
        ("market_score", 55.0, DEFAULT_MARKET_THRESHOLDS, "higher"),
    ]
    rows: list[dict[str, object]] = []
    for column, current_threshold, candidates, direction in specs:
        best = _best_threshold(
            events=events,
            column=column,
            current_threshold=current_threshold,
            candidates=candidates,
            target_window=target_window,
            min_sample_count=min_sample_count,
            direction=direction,
        )
        rows.append(
            {
                "rule": column,
                "current_threshold": current_threshold,
                "suggested_threshold": best["threshold"],
                "sample_count": best["sample_count"],
                f"win_rate_{target_window}d": best["win_rate"],
                f"avg_return_{target_window}d": best["avg_return"],
                "recommendation": best["recommendation"],
                "recommendation_zh": best["recommendation_zh"],
            }
        )
    return pd.DataFrame(rows)


def calibrate_walk_forward_profiles(
    events: pd.DataFrame,
    screening_config: ScreeningConfig | None = None,
    target_window: int = 20,
    min_sample_count: int = 5,
) -> pd.DataFrame:
    columns = [
        "screening_profile",
        "screening_profile_zh",
        "rule",
        "current_threshold",
        "suggested_threshold",
        "sample_count",
        f"win_rate_{target_window}d",
        f"avg_return_{target_window}d",
        "recommendation",
        "recommendation_zh",
        "suggestion_confidence",
        "suggestion_confidence_zh",
        "suggestion_note",
        "suggestion_note_zh",
    ]
    if events.empty or "screening_profile" not in events.columns:
        return pd.DataFrame(columns=columns)

    config = screening_config or default_screening_config()
    profile_thresholds = {
        "default": config.default_thresholds,
        **{profile.name: profile.thresholds for profile in config.profiles},
    }
    profile_names_zh = {
        "default": "默认规则",
        **{profile.name: profile.name_zh for profile in config.profiles},
    }
    specs = [
        (
            "signal_score",
            "signal_score_min",
            DEFAULT_SIGNAL_THRESHOLDS,
            "higher",
        ),
        (
            "high_probability_score",
            "high_probability_target_score",
            DEFAULT_HIGH_PROBABILITY_THRESHOLDS,
            "higher",
        ),
        (
            "calibrated_win_probability",
            None,
            DEFAULT_PROBABILITY_THRESHOLDS,
            "higher",
        ),
        (
            "relative_strength_score",
            "relative_strength_min",
            DEFAULT_RELATIVE_STRENGTH_THRESHOLDS,
            "higher",
        ),
        (
            "market_score",
            "market_score_min",
            DEFAULT_MARKET_THRESHOLDS,
            "higher",
        ),
    ]
    rows: list[dict[str, object]] = []
    for profile_name, group in events.groupby("screening_profile", dropna=False):
        profile_key = str(profile_name)
        thresholds = profile_thresholds.get(profile_key, config.default_thresholds)
        profile_zh = profile_names_zh.get(
            profile_key,
            _first_text(group, "screening_profile_zh", "默认规则"),
        )
        for column, threshold_attr, candidates, direction in specs:
            if column not in group.columns:
                continue
            current_threshold = (
                0.58 if threshold_attr is None else float(getattr(thresholds, threshold_attr))
            )
            best = _best_threshold(
                events=group,
                column=column,
                current_threshold=current_threshold,
                candidates=candidates,
                target_window=target_window,
                min_sample_count=min_sample_count,
                direction=direction,
            )
            confidence = _suggestion_confidence(
                sample_count=int(best["sample_count"]),
                win_rate=best["win_rate"],
                avg_return=best["avg_return"],
            )
            rows.append(
                {
                    "screening_profile": profile_key,
                    "screening_profile_zh": profile_zh,
                    "rule": column,
                    "current_threshold": current_threshold,
                    "suggested_threshold": best["threshold"],
                    "sample_count": best["sample_count"],
                    f"win_rate_{target_window}d": best["win_rate"],
                    f"avg_return_{target_window}d": best["avg_return"],
                    "recommendation": best["recommendation"],
                    "recommendation_zh": best["recommendation_zh"],
                    "suggestion_confidence": confidence["level"],
                    "suggestion_confidence_zh": confidence["level_zh"],
                    "suggestion_note": confidence["note"],
                    "suggestion_note_zh": confidence["note_zh"],
                }
            )
    return pd.DataFrame(rows, columns=columns)


def build_overfitting_risk_report(
    profile_calibration: pd.DataFrame,
    min_sample_count: int = MIN_CALIBRATION_SAMPLE_COUNT,
) -> pd.DataFrame:
    """Per-profile overfitting exposure: how many thresholds are being tuned vs. how
    much evidence supports them. A profile tuning many thresholds on few samples is
    the classic overfitting trap."""
    columns = [
        "screening_profile",
        "tunable_param_count",
        "adoptable_param_count",
        "median_sample_count",
        "sample_to_param_ratio",
        "overfit_risk_level",
        "overfit_risk_level_zh",
        "note",
        "note_zh",
    ]
    if (
        profile_calibration is None
        or profile_calibration.empty
        or "screening_profile" not in profile_calibration.columns
    ):
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for profile, group in profile_calibration.groupby("screening_profile"):
        samples = pd.to_numeric(group.get("sample_count"), errors="coerce").fillna(0.0)
        tunable = int(len(group))
        adoptable = int((samples >= min_sample_count).sum())
        median_samples = float(samples.median()) if len(samples) else 0.0
        ratio = round(median_samples / tunable, 2) if tunable else 0.0
        level, level_zh = _overfit_risk_level(adoptable, tunable, median_samples, min_sample_count)
        rows.append(
            {
                "screening_profile": profile,
                "tunable_param_count": tunable,
                "adoptable_param_count": adoptable,
                "median_sample_count": round(median_samples, 1),
                "sample_to_param_ratio": ratio,
                "overfit_risk_level": level,
                "overfit_risk_level_zh": level_zh,
                "note": (
                    f"{adoptable}/{tunable} thresholds have >= {min_sample_count} samples; "
                    f"median sample count {median_samples:.0f}."
                ),
                "note_zh": (
                    f"{tunable}个可调阈值中{adoptable}个样本达到{min_sample_count}；"
                    f"样本中位数{median_samples:.0f}。"
                ),
            }
        )
    result = pd.DataFrame(rows, columns=columns)
    order = {"high": 0, "medium": 1, "low": 2}
    result["_r"] = result["overfit_risk_level"].map(order).fillna(9)
    return result.sort_values(["_r", "screening_profile"]).drop(columns="_r").reset_index(drop=True)


def _overfit_risk_level(
    adoptable: int, tunable: int, median_samples: float, min_sample_count: int
) -> tuple[str, str]:
    if adoptable == 0 or median_samples < min_sample_count / 2:
        return "high", "过拟合风险高"
    if adoptable < tunable or median_samples < min_sample_count:
        return "medium", "过拟合风险中"
    return "low", "过拟合风险低"


def render_suggested_screening_config(
    screening_config: ScreeningConfig,
    profile_calibration: pd.DataFrame,
    benchmark_policy: pd.DataFrame | None = None,
    benchmark_tightening: pd.DataFrame | None = None,
    minimum_sample_guard: pd.DataFrame | None = None,
    historical_threshold_recommendations: pd.DataFrame | None = None,
    min_sample_count: int = MIN_CALIBRATION_SAMPLE_COUNT,
) -> str:
    suggested_thresholds = {
        profile.name: profile.thresholds for profile in screening_config.profiles
    }
    suggested_thresholds["default"] = screening_config.default_thresholds
    applied_notes: list[str] = []
    skipped_notes: list[str] = []
    benchmark_guard = _benchmark_relaxation_guard(benchmark_policy)
    benchmark_tightening_notes = _apply_benchmark_tightening_recommendations(
        suggested_thresholds=suggested_thresholds,
        benchmark_tightening=benchmark_tightening,
        minimum_sample_guard=minimum_sample_guard,
    )
    applied_notes.extend(benchmark_tightening_notes)
    historical_threshold_notes = _apply_historical_threshold_recommendations(
        suggested_thresholds=suggested_thresholds,
        recommendations=historical_threshold_recommendations,
    )
    applied_notes.extend(historical_threshold_notes)

    for row in profile_calibration.itertuples(index=False):
        profile_name = str(row.screening_profile)
        threshold_attr = _threshold_attr_for_rule(str(row.rule))
        if threshold_attr is None:
            continue
        sample_count = int(row.sample_count)
        suggested_value = float(row.suggested_threshold)
        confidence = str(getattr(row, "suggestion_confidence", "insufficient"))
        if (
            sample_count < min_sample_count
            or confidence not in {"medium", "high"}
            or not _is_finite(suggested_value)
        ):
            skipped_notes.append(
                f"{profile_name}.{threshold_attr}: skipped; "
                f"sample_count={sample_count}; confidence={confidence}"
            )
            continue
        current = suggested_thresholds.get(profile_name, screening_config.default_thresholds)
        current_value = float(getattr(current, threshold_attr))
        if suggested_value < current_value and benchmark_guard["block_relaxation"]:
            skipped_notes.append(
                f"{profile_name}.{threshold_attr}: skipped relaxation by benchmark guard; "
                f"{current_value:g} -> {suggested_value:g}; policy={benchmark_guard['policy']}"
            )
            continue
        updated = ScreeningThresholds.from_mapping(
            {
                **current.to_dict(),
                threshold_attr: suggested_value,
            }
        )
        suggested_thresholds[profile_name] = updated
        applied_notes.append(
            f"{profile_name}.{threshold_attr}: {getattr(current, threshold_attr):g} -> "
            f"{suggested_value:g}; confidence={confidence}"
        )

    suggested_config = screening_config.with_profile_thresholds(suggested_thresholds)
    header = [
        "Suggested screening config generated from walk-forward validation.",
        "Review manually before replacing configs/screening.toml.",
        f"Benchmark guard: {benchmark_guard['policy']}",
        f"Benchmark guard note: {benchmark_guard['note']}",
        f"Benchmark tightening suggestions applied: {len(benchmark_tightening_notes)}",
        f"Historical threshold suggestions applied: {len(historical_threshold_notes)}",
        f"Minimum sample guard: {_minimum_sample_guard_summary(minimum_sample_guard)}",
        f"Applied suggestions: {len(applied_notes)}",
        f"Skipped suggestions: {len(skipped_notes)}",
    ]
    if applied_notes:
        header.extend(["", "Applied:"])
        header.extend(f"- {note}" for note in applied_notes)
    if skipped_notes:
        header.extend(["", "Skipped:"])
        header.extend(f"- {note}" for note in skipped_notes[:20])
    return render_screening_config_toml(suggested_config, header_lines=header)


def _apply_historical_threshold_recommendations(
    suggested_thresholds: dict[str, ScreeningThresholds],
    recommendations: pd.DataFrame | None,
) -> list[str]:
    if recommendations is None or recommendations.empty:
        return []
    required = {
        "scope",
        "group_value",
        "threshold_attr",
        "suggested_threshold",
        "priority",
        "allow_auto_apply",
    }
    if not required.issubset(recommendations.columns):
        return []

    notes: list[str] = []
    for row in recommendations.itertuples(index=False):
        if str(row.scope) != "screening_profile":
            continue
        if str(row.priority) not in {"high", "medium"}:
            continue
        if not _coerce_bool(getattr(row, "allow_auto_apply", False)):
            continue
        profile_name = str(row.group_value)
        threshold_attr = str(row.threshold_attr)
        suggested_value = float(row.suggested_threshold)
        current = suggested_thresholds.get(profile_name)
        if current is None or not hasattr(current, threshold_attr) or not _is_finite(suggested_value):
            continue
        current_value = float(getattr(current, threshold_attr))
        if not _historical_suggestion_is_tighter(
            threshold_attr=threshold_attr,
            current_value=current_value,
            suggested_value=suggested_value,
        ):
            continue
        updated = ScreeningThresholds.from_mapping(
            {
                **current.to_dict(),
                threshold_attr: suggested_value,
            }
        )
        suggested_thresholds[profile_name] = updated
        notes.append(
            f"{profile_name}.{threshold_attr}: {current_value:g} -> {suggested_value:g}; "
            f"historical win-rate gate; priority={row.priority}"
        )
    return notes


def _historical_suggestion_is_tighter(
    threshold_attr: str,
    current_value: float,
    suggested_value: float,
) -> bool:
    lower_is_tighter = {"max_backtest_stop_hit_rate", "max_backtest_slippage_pct"}
    if threshold_attr in lower_is_tighter:
        return suggested_value < current_value
    return suggested_value > current_value


def _coerce_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", ""}:
        return False
    return default


def _apply_benchmark_tightening_recommendations(
    suggested_thresholds: dict[str, ScreeningThresholds],
    benchmark_tightening: pd.DataFrame | None,
    minimum_sample_guard: pd.DataFrame | None = None,
) -> list[str]:
    if benchmark_tightening is None or benchmark_tightening.empty:
        return []
    required = {"screening_profile", "threshold_attr", "suggested_threshold", "priority"}
    if not required.issubset(benchmark_tightening.columns):
        return []

    notes: list[str] = []
    for row in benchmark_tightening.itertuples(index=False):
        priority = str(getattr(row, "priority", "low"))
        if priority not in {"high", "medium"}:
            continue
        profile_name = str(row.screening_profile)
        threshold_attr = str(row.threshold_attr)
        suggested_value = float(row.suggested_threshold)
        if not _minimum_sample_guard_allows(
            minimum_sample_guard=minimum_sample_guard,
            profile_name=profile_name,
            threshold_attr=threshold_attr,
            suggested_value=suggested_value,
        ):
            continue
        current = suggested_thresholds.get(profile_name)
        if current is None or not hasattr(current, threshold_attr) or not _is_finite(suggested_value):
            continue
        current_value = float(getattr(current, threshold_attr))
        if suggested_value <= current_value:
            continue
        updated = ScreeningThresholds.from_mapping(
            {
                **current.to_dict(),
                threshold_attr: suggested_value,
            }
        )
        suggested_thresholds[profile_name] = updated
        notes.append(
            f"{profile_name}.{threshold_attr}: {current_value:g} -> {suggested_value:g}; "
            f"benchmark tightening; priority={priority}"
        )
    return notes


def _minimum_sample_guard_allows(
    minimum_sample_guard: pd.DataFrame | None,
    profile_name: str,
    threshold_attr: str,
    suggested_value: float,
) -> bool:
    if minimum_sample_guard is None or minimum_sample_guard.empty:
        return True
    required = {"screening_profile", "threshold_attr", "threshold_value", "allow_adoption"}
    if not required.issubset(minimum_sample_guard.columns):
        return True
    frame = minimum_sample_guard[
        (minimum_sample_guard["screening_profile"].astype(str) == profile_name)
        & (minimum_sample_guard["threshold_attr"].astype(str) == threshold_attr)
    ].copy()
    if frame.empty:
        return False
    frame["threshold_value"] = pd.to_numeric(frame["threshold_value"], errors="coerce")
    matching = frame[(frame["threshold_value"] - suggested_value).abs() < 1e-9]
    if matching.empty:
        return False
    return bool(matching["allow_adoption"].fillna(False).astype(bool).any())


def _minimum_sample_guard_summary(minimum_sample_guard: pd.DataFrame | None) -> str:
    if minimum_sample_guard is None or minimum_sample_guard.empty or "guard_action" not in minimum_sample_guard.columns:
        return "not_available"
    counts = minimum_sample_guard["guard_action"].astype(str).value_counts().to_dict()
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def _benchmark_relaxation_guard(benchmark_policy: pd.DataFrame | None) -> dict[str, object]:
    if benchmark_policy is None or benchmark_policy.empty or "allow_relaxation" not in benchmark_policy.columns:
        return {
            "block_relaxation": False,
            "policy": "no_benchmark_policy",
            "note": "Benchmark policy was not available.",
        }
    allow_relaxation = benchmark_policy["allow_relaxation"].fillna(False).astype(bool)
    actions = ";".join(benchmark_policy["benchmark_policy_action"].astype(str).unique())
    notes = "; ".join(benchmark_policy["policy_note"].astype(str).dropna().unique()[:3])
    if allow_relaxation.any():
        return {
            "block_relaxation": False,
            "policy": actions,
            "note": notes or "At least one portfolio policy allows selective relaxation.",
        }
    return {
        "block_relaxation": True,
        "policy": actions,
        "note": notes or "Benchmark evidence does not support relaxing thresholds.",
    }


def _threshold_attr_for_rule(rule: str) -> str | None:
    return {
        "signal_score": "signal_score_min",
        "high_probability_score": "high_probability_target_score",
        "relative_strength_score": "relative_strength_min",
        "market_score": "market_score_min",
    }.get(rule)


def _suggestion_confidence(
    sample_count: int,
    win_rate: float,
    avg_return: float,
) -> dict[str, str]:
    if sample_count < 5 or not _is_finite(win_rate) or not _is_finite(avg_return):
        return {
            "level": "insufficient",
            "level_zh": "样本不足",
            "note": "Too few validation samples to trust this suggestion.",
            "note_zh": "验证样本太少，暂时不能信任该建议。",
        }
    if sample_count >= 30 and float(avg_return) > 0 and float(win_rate) >= 0.55:
        return {
            "level": "high",
            "level_zh": "高",
            "note": "Sample count, win rate, and average return are supportive.",
            "note_zh": "样本数、胜率和平均收益都支持该建议。",
        }
    if sample_count >= 15 and float(avg_return) > 0:
        return {
            "level": "medium",
            "level_zh": "中",
            "note": "Sample count is acceptable and average return is positive.",
            "note_zh": "样本数基本可用，平均收益为正。",
        }
    if sample_count >= 5:
        return {
            "level": "low",
            "level_zh": "低",
            "note": "Suggestion is based on limited or weak validation performance.",
            "note_zh": "该建议基于有限样本或表现不够强，只能参考。",
        }
    return {
        "level": "insufficient",
        "level_zh": "样本不足",
        "note": "Too few validation samples to trust this suggestion.",
        "note_zh": "验证样本太少，暂时不能信任该建议。",
    }


def render_walk_forward_report(
    events: pd.DataFrame,
    summary: pd.DataFrame,
    calibration: pd.DataFrame,
    ticker_ranking: pd.DataFrame | None = None,
    sample_sufficiency: pd.DataFrame | None = None,
    profile_summary: pd.DataFrame | None = None,
    segment_summary: pd.DataFrame | None = None,
    market_regime_summary: pd.DataFrame | None = None,
    market_regime_policy: pd.DataFrame | None = None,
    profile_calibration: pd.DataFrame | None = None,
    probability_calibration: pd.DataFrame | None = None,
    portfolio_summary: pd.DataFrame | None = None,
    portfolio_rebalances: pd.DataFrame | None = None,
    portfolio_equity_summary: pd.DataFrame | None = None,
    portfolio_equity_curve: pd.DataFrame | None = None,
    benchmark_summary: pd.DataFrame | None = None,
    benchmark_curve: pd.DataFrame | None = None,
    benchmark_policy: pd.DataFrame | None = None,
    benchmark_tightening: pd.DataFrame | None = None,
    tightening_impact: pd.DataFrame | None = None,
    threshold_sensitivity: pd.DataFrame | None = None,
    minimum_sample_guard: pd.DataFrame | None = None,
    forward_windows: tuple[int, ...] = DEFAULT_FORWARD_WINDOWS,
    survivorship_bias_report: dict[str, object] | None = None,
) -> str:
    survivorship = survivorship_bias_report or survivorship_report_without_historical_membership()
    lines = [
        "# Walk-Forward Validation / 滚动历史验证",
        "",
        "This report replays historical signal dates using only price data available up to each signal date.",
        "本报告按历史信号日期回放，只使用每个信号日以前可见的价格数据。",
        "",
        "Validation mode uses neutral historical-safe market, sector, fundamental, and event contexts.",
        "验证模式使用中性、历史安全的大盘、板块、基本面和事件上下文，避免把当前快照回填到历史。",
        "",
        f"- Event rows / 信号样本行数: `{len(events)}`",
        f"- Forward windows / 未来观察窗口: `{', '.join(str(window) + 'd' for window in forward_windows)}`",
        f"- survivorship_bias_handled: `{str(bool(survivorship.get('survivorship_bias_handled'))).lower()}`",
        f"- contains_delisted_tickers / 样本包含退市股票: `{str(bool(survivorship.get('contains_delisted_tickers'))).lower()}`",
        f"- historical_universe_source / 历史股票池来源: `{survivorship.get('source', 'none')}`",
        "",
        "## Survivorship Bias / 幸存者偏差",
        "",
        (
            "This section states whether the validation used point-in-time universe "
            "membership instead of today's surviving ticker list."
        ),
        "本区块说明验证是否使用了历史时点成分股，而不是今天仍然存在的股票列表。",
        "",
        f"- survivorship_bias_handled: `{str(bool(survivorship.get('survivorship_bias_handled'))).lower()}`",
        f"- point_in_time_universe: `{str(bool(survivorship.get('point_in_time_universe'))).lower()}`",
        f"- contains_delisted_tickers: `{str(bool(survivorship.get('contains_delisted_tickers'))).lower()}`",
        f"- historical_constituent_count: `{int(survivorship.get('historical_constituent_count') or 0)}`",
        f"- delisted_ticker_count: `{int(survivorship.get('delisted_ticker_count') or 0)}`",
        f"- source: `{survivorship.get('source', 'none')}`",
        "",
        *[f"- warning: {warning}" for warning in survivorship.get("warnings", [])],
        *[f"- 警告: {warning}" for warning in survivorship.get("warnings_zh", [])],
        "",
        "## Performance Summary / 表现摘要",
        "",
    ]
    lines.extend(_markdown_table(_format_metric_frame(summary)))
    lines.extend(
        [
            "",
            "## Ticker Validation Ranking / 个股验证排名",
            "",
            (
                "This section ranks tickers inside the validated universe using "
                "historical signal quality, win rate, average return, and drawdown."
            ),
            "本区块按历史信号质量、胜率、平均收益和回撤，对股票池内部 ticker 排名。",
            "",
        ]
    )
    ticker_ranking_frame = ticker_ranking if ticker_ranking is not None else pd.DataFrame()
    if not ticker_ranking_frame.empty:
        lines.extend(_markdown_table(_format_metric_frame(ticker_ranking_frame.head(25))))
    else:
        lines.extend(_markdown_table(ticker_ranking_frame))
    lines.extend(
        [
            "",
            "## Sample Sufficiency Guidance / 样本充分性建议",
            "",
            (
                "This section explains whether the validation has enough samples and "
                "which parameters to adjust when the ranking is too thin."
            ),
            "本区块说明验证样本是否足够；当排名样本太少时，给出下一次验证应调整的参数。",
            "",
        ]
    )
    sample_sufficiency_frame = (
        sample_sufficiency if sample_sufficiency is not None else pd.DataFrame()
    )
    if not sample_sufficiency_frame.empty:
        lines.extend(_markdown_table(_format_metric_frame(sample_sufficiency_frame.head(30))))
    else:
        lines.extend(_markdown_table(sample_sufficiency_frame))
    lines.extend(
        [
            "",
            "## Profile Summary / 分类规则表现",
            "",
        ]
    )
    profile_frame = profile_summary if profile_summary is not None else pd.DataFrame()
    lines.extend(_markdown_table(_format_metric_frame(profile_frame)))
    lines.extend(
        [
            "",
            "## Segment Validation Summary / 分层验证表现",
            "",
            (
                "This section breaks validation down by profile, horizon, entry type, "
                "and validation bucket so profile-specific rules are not judged only by averages."
            ),
            "本区块按规则大类、周期、买点类型和验证结果拆分表现，避免只看总体平均值。",
            "",
        ]
    )
    segment_frame = segment_summary if segment_summary is not None else pd.DataFrame()
    if not segment_frame.empty:
        lines.extend(_markdown_table(_format_metric_frame(segment_frame.head(40))))
    else:
        lines.extend(_markdown_table(segment_frame))
    lines.extend(
        [
            "",
            "## Market Regime Validation / 市场状态验证",
            "",
            (
                "This section splits historical signals by market regime using benchmark "
                "data available at each historical signal date."
            ),
            "本区块用每个历史信号日当时可见的基准数据，把信号拆成牛市、熊市、震荡和高波动环境。",
            "",
        ]
    )
    regime_frame = market_regime_summary if market_regime_summary is not None else pd.DataFrame()
    lines.extend(_markdown_table(_format_metric_frame(regime_frame)))
    lines.extend(
        [
            "",
            "## Market Regime Protection Policy / 市场状态保护规则",
            "",
            (
                "This section converts market-regime validation into conservative "
                "entry protection rules for bear and high-volatility regimes."
            ),
            "本区块把市场状态验证结果转换成保护规则，尤其针对熊市和高波动环境。",
            "",
        ]
    )
    regime_policy_frame = market_regime_policy if market_regime_policy is not None else pd.DataFrame()
    lines.extend(_markdown_table(_format_metric_frame(regime_policy_frame)))
    lines.extend(
        [
            "",
            "## Probability Calibration / 概率校准",
            "",
            (
                "This section compares estimated win probability against actual "
                "walk-forward win rate by probability bucket."
            ),
            "本区块按概率分组，对比模型估计胜率与滚动验证中的实际胜率。",
            "",
        ]
    )
    probability_frame = (
        probability_calibration if probability_calibration is not None else pd.DataFrame()
    )
    lines.extend(_markdown_table(_format_metric_frame(probability_frame)))
    lines.extend(
        [
            "",
            "## Portfolio Validation / 组合验证",
            "",
            (
                "This section forms equal-weight baskets from the top-ranked candidates "
                "on each historical signal date."
            ),
            "本区块在每个历史信号日从排名最高的候选中构建等权组合。",
            "",
        ]
    )
    portfolio_frame = portfolio_summary if portfolio_summary is not None else pd.DataFrame()
    lines.extend(_markdown_table(_format_metric_frame(portfolio_frame)))
    lines.extend(
        [
            "",
            "## Portfolio Equity Curve / 组合逐日净值曲线",
            "",
            (
                "This section simulates daily equal-weight portfolio returns from "
                "historical rebalance selections, including transaction-cost drag."
            ),
            "本区块根据历史调仓选择模拟逐日等权组合收益，并计入交易成本拖累。",
            "",
        ]
    )
    equity_summary_frame = (
        portfolio_equity_summary if portfolio_equity_summary is not None else pd.DataFrame()
    )
    lines.extend(_markdown_table(_format_metric_frame(equity_summary_frame)))
    lines.extend(
        [
            "",
            "## Portfolio Equity Curve Sample / 组合净值曲线样本",
            "",
        ]
    )
    equity_curve_frame = portfolio_equity_curve if portfolio_equity_curve is not None else pd.DataFrame()
    if not equity_curve_frame.empty:
        lines.extend(_markdown_table(_format_metric_frame(equity_curve_frame.head(25))))
    else:
        lines.extend(_markdown_table(equity_curve_frame))
    lines.extend(
        [
            "",
            "## Benchmark Comparison / 基准对比",
            "",
            (
                "This section compares portfolio equity against SPY and QQQ "
                "on the same daily dates."
            ),
            "本区块在相同每日日期上，把组合净值与SPY和QQQ进行对比。",
            "",
        ]
    )
    benchmark_summary_frame = benchmark_summary if benchmark_summary is not None else pd.DataFrame()
    lines.extend(_markdown_table(_format_metric_frame(benchmark_summary_frame)))
    lines.extend(
        [
            "",
            "## Benchmark-Aware Rule Policy / 基准感知规则建议",
            "",
            (
                "This section converts benchmark comparison into a threshold policy: "
                "tighten, keep, or selectively relax."
            ),
            "本区块把基准对比转换成阈值策略：收紧、维持或谨慎放宽。",
            "",
        ]
    )
    benchmark_policy_frame = benchmark_policy if benchmark_policy is not None else pd.DataFrame()
    lines.extend(_markdown_table(_format_metric_frame(benchmark_policy_frame)))
    lines.extend(
        [
            "",
            "## Specific Tightening Recommendations / 具体收紧建议",
            "",
            (
                "This section lists concrete threshold changes when benchmark-aware "
                "policy says the model should tighten."
            ),
            "当基准感知策略要求收紧时，本区块列出具体应该提高的门槛。",
            "",
        ]
    )
    benchmark_tightening_frame = benchmark_tightening if benchmark_tightening is not None else pd.DataFrame()
    lines.extend(_markdown_table(_format_metric_frame(benchmark_tightening_frame)))
    lines.extend(
        [
            "",
            "## Tightening Impact Validation / 收紧效果验证",
            "",
            (
                "This section compares validation metrics before and after applying "
                "medium/high priority tightening recommendations."
            ),
            "本区块对比应用中高优先级收紧建议前后的验证指标。",
            "",
        ]
    )
    tightening_impact_frame = tightening_impact if tightening_impact is not None else pd.DataFrame()
    lines.extend(_markdown_table(_format_metric_frame(tightening_impact_frame)))
    lines.extend(
        [
            "",
            "## Threshold Sensitivity Grid / 阈值敏感度网格",
            "",
            (
                "This section tests multiple threshold values and threshold pairs "
                "to find filters that actually improve validation metrics."
            ),
            "本区块测试多个门槛值和门槛组合，寻找真正改善验证指标的过滤条件。",
            "",
        ]
    )
    threshold_sensitivity_frame = threshold_sensitivity if threshold_sensitivity is not None else pd.DataFrame()
    if not threshold_sensitivity_frame.empty:
        lines.extend(_markdown_table(_format_metric_frame(threshold_sensitivity_frame.head(30))))
    else:
        lines.extend(_markdown_table(threshold_sensitivity_frame))
    lines.extend(
        [
            "",
            "## Minimum Sample Guard / 最小样本保护",
            "",
            (
                "This section blocks threshold changes that leave too few validation "
                "samples or do not show clear improvement."
            ),
            "本区块阻止样本太少或改善不明确的门槛被自动采用。",
            "",
        ]
    )
    guard_frame = minimum_sample_guard if minimum_sample_guard is not None else pd.DataFrame()
    if not guard_frame.empty:
        lines.extend(_markdown_table(_format_metric_frame(guard_frame.head(30))))
    else:
        lines.extend(_markdown_table(guard_frame))
    lines.extend(
        [
            "",
            "## Benchmark Comparison Sample / 基准对比样本",
            "",
        ]
    )
    benchmark_curve_frame = benchmark_curve if benchmark_curve is not None else pd.DataFrame()
    if not benchmark_curve_frame.empty:
        lines.extend(_markdown_table(_format_metric_frame(benchmark_curve_frame.head(25))))
    else:
        lines.extend(_markdown_table(benchmark_curve_frame))
    lines.extend(
        [
            "",
            "## Portfolio Rebalances / 组合历史调仓",
            "",
        ]
    )
    rebalance_frame = portfolio_rebalances if portfolio_rebalances is not None else pd.DataFrame()
    if not rebalance_frame.empty:
        lines.extend(_markdown_table(_format_metric_frame(rebalance_frame.head(25))))
    else:
        lines.extend(_markdown_table(rebalance_frame))
    lines.extend(
        [
            "",
            "## Profile Rule Calibration / 分类规则阈值建议",
            "",
        ]
    )
    profile_calibration_frame = (
        profile_calibration if profile_calibration is not None else pd.DataFrame()
    )
    lines.extend(_markdown_table(_format_metric_frame(profile_calibration_frame)))
    lines.extend(
        [
            "",
            "## Rule Calibration / 规则校准",
            "",
        ]
    )
    lines.extend(_markdown_table(_format_metric_frame(calibration)))
    return "\n".join(lines).rstrip() + "\n"


def _event_row(
    ticker_prices: pd.DataFrame,
    signal_index: int,
    analysis_row: object,
    forward_windows: tuple[int, ...],
    max_forward_window: int,
    market_regime_lookup: dict[pd.Timestamp, dict[str, object]] | None = None,
    membership_record: dict[str, object] | None = None,
) -> dict[str, object]:
    signal_price = float(ticker_prices.loc[signal_index, "adj_close"])
    signal_date = pd.Timestamp(ticker_prices.loc[signal_index, "date"]).normalize()
    market_regime = _lookup_market_regime(signal_date, market_regime_lookup or {})
    delisted_date = _optional_timestamp((membership_record or {}).get("delisted_date"))
    delisting_return = _safe_optional_float((membership_record or {}).get("delisting_return"))
    row = {
        "date": ticker_prices.loc[signal_index, "date"],
        "ticker": analysis_row.ticker,
        "horizon": analysis_row.horizon,
        "horizon_zh_label": getattr(
            analysis_row,
            "horizon_zh_label",
            HORIZON_SPECS[analysis_row.horizon].zh_label,
        ),
        "action": analysis_row.action,
        "screening_action": analysis_row.screening_action,
        "screening_profile": getattr(analysis_row, "screening_profile", "default"),
        "screening_profile_zh": getattr(analysis_row, "screening_profile_zh", "默认规则"),
        "quality_gate_passed": bool(analysis_row.quality_gate_passed),
        "watchlist_status": analysis_row.watchlist_status,
        "validation_bucket": _validation_bucket(analysis_row),
        "high_probability_score": float(analysis_row.high_probability_score),
        "calibrated_win_probability": float(analysis_row.calibrated_win_probability)
        if _is_finite(analysis_row.calibrated_win_probability)
        else np.nan,
        "calibrated_probability_level": analysis_row.calibrated_probability_level,
        "calibrated_probability_level_zh": analysis_row.calibrated_probability_level_zh,
        "calibrated_probability_confidence": float(
            analysis_row.calibrated_probability_confidence
        )
        if _is_finite(analysis_row.calibrated_probability_confidence)
        else np.nan,
        "calibrated_probability_confidence_level": (
            analysis_row.calibrated_probability_confidence_level
        ),
        "calibrated_probability_confidence_level_zh": (
            analysis_row.calibrated_probability_confidence_level_zh
        ),
        "signal_score": float(analysis_row.signal_score),
        "confidence_score": float(analysis_row.confidence_score),
        "market_score": float(analysis_row.market_score),
        "relative_strength_score": float(analysis_row.relative_strength_score),
        "event_risk_level": analysis_row.event_risk_level,
        "event_window": analysis_row.event_window,
        "event_block_new_entries": bool(analysis_row.event_block_new_entries),
        "event_cooldown_active": bool(analysis_row.event_cooldown_active),
        "sentiment_score": float(analysis_row.sentiment_score),
        "sentiment_risk_level": analysis_row.sentiment_risk_level,
        "sentiment_block_new_entries": bool(analysis_row.sentiment_block_new_entries),
        "analyst_score": float(analysis_row.analyst_score),
        "analyst_risk_level": analysis_row.analyst_risk_level,
        "analyst_block_new_entries": bool(analysis_row.analyst_block_new_entries),
        "valuation_score": float(analysis_row.valuation_score),
        "valuation_risk_level": analysis_row.valuation_risk_level,
        "valuation_block_new_entries": bool(analysis_row.valuation_block_new_entries),
        "screening_backtest_entry_type": analysis_row.screening_backtest_entry_type,
        "screening_backtest_win_rate": float(analysis_row.screening_backtest_win_rate)
        if _is_finite(analysis_row.screening_backtest_win_rate)
        else np.nan,
        "screening_backtest_average_return": float(analysis_row.screening_backtest_average_return)
        if _is_finite(analysis_row.screening_backtest_average_return)
        else np.nan,
        "screening_backtest_trade_count": int(analysis_row.screening_backtest_trade_count),
        "sample_confidence_level": analysis_row.sample_confidence_level,
        "sample_confidence_level_zh": analysis_row.sample_confidence_level_zh,
        "evidence_strength": analysis_row.evidence_strength,
        "evidence_strength_zh": analysis_row.evidence_strength_zh,
        "evidence_note": analysis_row.evidence_note,
        "evidence_note_zh": analysis_row.evidence_note_zh,
        "liquidity_filter_passed": bool(analysis_row.liquidity_filter_passed),
        "liquidity_filter_reason": analysis_row.liquidity_filter_reason,
        "liquidity_filter_reason_zh": analysis_row.liquidity_filter_reason_zh,
        "quality_gate_fail_reasons": analysis_row.quality_gate_fail_reasons,
        "quality_gate_fail_reasons_zh": analysis_row.quality_gate_fail_reasons_zh,
        "watchlist_missing_items": analysis_row.watchlist_missing_items,
        "watchlist_missing_items_zh": analysis_row.watchlist_missing_items_zh,
        "validation_market_regime": market_regime["validation_market_regime"],
        "validation_market_regime_zh": market_regime["validation_market_regime_zh"],
        "validation_spy_trend_score": market_regime["validation_spy_trend_score"],
        "validation_qqq_trend_score": market_regime["validation_qqq_trend_score"],
        "validation_benchmark_60d_return": market_regime[
            "validation_benchmark_60d_return"
        ],
        "validation_benchmark_20d_volatility": market_regime[
            "validation_benchmark_20d_volatility"
        ],
        "point_in_time_universe_member": membership_record is not None,
        "universe_start_date": _format_optional_date((membership_record or {}).get("start_date")),
        "universe_end_date": _format_optional_date((membership_record or {}).get("end_date")),
        "delisted_date": _format_optional_date(delisted_date),
        "delisting_return_used": np.nan,
        "delisted_during_forward_window": False,
    }
    for window in forward_windows:
        future_index = signal_index + window
        delisting_adjusted = False
        if future_index < len(ticker_prices):
            future_price = float(ticker_prices.loc[future_index, "adj_close"])
        else:
            last_price = float(ticker_prices["adj_close"].iloc[-1])
            if delisting_return is not None:
                future_price = last_price * (1.0 + delisting_return)
                row["delisting_return_used"] = delisting_return
            else:
                future_price = last_price
            delisting_adjusted = True
        row[f"forward_return_{window}d_delisting_adjusted"] = delisting_adjusted
        row["delisted_during_forward_window"] = (
            bool(row["delisted_during_forward_window"]) or delisting_adjusted
        )
        row[f"forward_return_{window}d"] = future_price / signal_price - 1.0
    future_prices = ticker_prices.loc[
        signal_index : min(signal_index + max_forward_window, len(ticker_prices) - 1),
        "adj_close",
    ].astype(float)
    row["max_drawdown_after_signal"] = float((future_prices / signal_price - 1.0).min())
    return row


def _membership_record_for_signal(
    ticker: str,
    signal_date: object,
    universe_membership: HistoricalUniverseMembership | None,
) -> dict[str, object] | None:
    if universe_membership is None:
        return None
    return universe_membership.active_record(ticker, signal_date)


def _ticker_allows_truncated_forward(
    ticker: str,
    universe_membership: HistoricalUniverseMembership | None,
) -> bool:
    if universe_membership is None or universe_membership.records.empty:
        return False
    clean = normalize_ticker(str(ticker))
    records = universe_membership.records[universe_membership.records["ticker"] == clean]
    if records.empty:
        return False
    if "delisted_date" in records.columns and records["delisted_date"].notna().any():
        return True
    if "status" in records.columns:
        return bool(records["status"].fillna("").astype(str).str.lower().str.contains("delist").any())
    return False


def _profile_by_name(config: ScreeningConfig, profile_name: str | None):
    if not profile_name:
        return None
    for profile in config.profiles:
        if profile.name == profile_name:
            return profile
    raise ValueError(f"Unknown screening profile: {profile_name}")


def _apply_pooled_entry_backtest(events: pd.DataFrame, config: ScreeningConfig) -> pd.DataFrame:
    if events.empty:
        return events
    required = {
        "screening_profile",
        "horizon",
        "screening_backtest_entry_type",
        "screening_backtest_trade_count",
        "screening_backtest_win_rate",
        "screening_backtest_average_return",
        "quality_gate_fail_reasons",
        "quality_gate_fail_reasons_zh",
        "validation_bucket",
    }
    if not required.issubset(events.columns):
        return events

    result = events.copy()
    result["original_quality_gate_passed"] = result["quality_gate_passed"]
    result["original_validation_bucket"] = result["validation_bucket"]
    result["pooled_entry_backtest_used"] = False
    result["pooled_screening_backtest_trade_count"] = result["screening_backtest_trade_count"]
    result["pooled_screening_backtest_win_rate"] = result["screening_backtest_win_rate"]
    result["pooled_screening_backtest_average_return"] = result["screening_backtest_average_return"]
    result["pooled_quality_gate_fail_reasons"] = result["quality_gate_fail_reasons"]
    result["pooled_quality_gate_fail_reasons_zh"] = result["quality_gate_fail_reasons_zh"]

    thresholds_by_profile = {
        "default": config.default_thresholds,
        **{profile.name: profile.thresholds for profile in config.profiles},
    }
    group_columns = ["screening_profile", "horizon", "screening_backtest_entry_type"]
    pooled_metrics = {
        keys: _pooled_backtest_metrics(group)
        for keys, group in result.groupby(group_columns, dropna=False)
    }

    for index, row in result.iterrows():
        profile_name = str(row["screening_profile"])
        thresholds = thresholds_by_profile.get(profile_name, config.default_thresholds)
        original_bucket = str(row["validation_bucket"])
        own_count = _safe_int(row["screening_backtest_trade_count"])
        keys = (
            row["screening_profile"],
            row["horizon"],
            row["screening_backtest_entry_type"],
        )
        pooled = pooled_metrics.get(keys, _empty_pooled_metrics())
        if own_count >= thresholds.backtest_sample_min:
            continue
        if int(pooled["trade_count"]) < thresholds.backtest_sample_min:
            continue
        if original_bucket != "near_watchlist":
            continue

        fail_reasons = _split_reason_text(row["quality_gate_fail_reasons"])
        fail_reasons_zh = _split_reason_text(row["quality_gate_fail_reasons_zh"])
        fail_reasons = _remove_entry_backtest_reasons(fail_reasons)
        fail_reasons_zh = _remove_entry_backtest_reasons_zh(fail_reasons_zh)

        if not _is_finite(pooled["win_rate"]) or float(pooled["win_rate"]) < thresholds.backtest_win_rate_min:
            fail_reasons.append("pooled entry backtest win rate too low")
            fail_reasons_zh.append("同类聚合买点回测胜率不足")
        if (
            not _is_finite(pooled["average_return"])
            or float(pooled["average_return"]) <= thresholds.backtest_average_return_min
        ):
            fail_reasons.append("pooled entry backtest average return not positive")
            fail_reasons_zh.append("同类聚合买点回测平均收益不是正数")

        result.at[index, "pooled_entry_backtest_used"] = True
        result.at[index, "pooled_screening_backtest_trade_count"] = int(pooled["trade_count"])
        result.at[index, "pooled_screening_backtest_win_rate"] = float(pooled["win_rate"])
        result.at[index, "pooled_screening_backtest_average_return"] = float(pooled["average_return"])

        if fail_reasons:
            result.at[index, "pooled_quality_gate_fail_reasons"] = "; ".join(fail_reasons)
            result.at[index, "pooled_quality_gate_fail_reasons_zh"] = "；".join(fail_reasons_zh)
        else:
            evidence = _entry_evidence_profile(
                trade_count=int(pooled["trade_count"]),
                win_rate=float(pooled["win_rate"]),
                average_return=float(pooled["average_return"]),
            )
            result.at[index, "pooled_quality_gate_fail_reasons"] = "all strict quality gates passed with pooled entry backtest"
            result.at[index, "pooled_quality_gate_fail_reasons_zh"] = "使用同类聚合买点回测后，所有严格质量门槛通过"
            result.at[index, "quality_gate_passed"] = True
            result.at[index, "validation_bucket"] = "high_probability"
            result.at[index, "quality_gate_fail_reasons"] = result.at[index, "pooled_quality_gate_fail_reasons"]
            result.at[index, "quality_gate_fail_reasons_zh"] = result.at[index, "pooled_quality_gate_fail_reasons_zh"]
            result.at[index, "screening_backtest_trade_count"] = int(pooled["trade_count"])
            result.at[index, "screening_backtest_win_rate"] = float(pooled["win_rate"])
            result.at[index, "screening_backtest_average_return"] = float(pooled["average_return"])
            for key, value in evidence.items():
                result.at[index, key] = value

    return result


def _pooled_backtest_metrics(group: pd.DataFrame) -> dict[str, float | int]:
    valid = group[
        (pd.to_numeric(group["screening_backtest_trade_count"], errors="coerce") > 0)
        & pd.to_numeric(group["screening_backtest_win_rate"], errors="coerce").notna()
        & pd.to_numeric(group["screening_backtest_average_return"], errors="coerce").notna()
    ].copy()
    if valid.empty:
        return _empty_pooled_metrics()
    counts = pd.to_numeric(valid["screening_backtest_trade_count"], errors="coerce").astype(float)
    win_rates = pd.to_numeric(valid["screening_backtest_win_rate"], errors="coerce").astype(float)
    average_returns = pd.to_numeric(valid["screening_backtest_average_return"], errors="coerce").astype(float)
    total_count = int(counts.sum())
    if total_count <= 0:
        return _empty_pooled_metrics()
    return {
        "trade_count": total_count,
        "win_rate": float((win_rates * counts).sum() / total_count),
        "average_return": float((average_returns * counts).sum() / total_count),
    }


def _empty_pooled_metrics() -> dict[str, float | int]:
    return {"trade_count": 0, "win_rate": np.nan, "average_return": np.nan}


def _split_reason_text(value: object) -> list[str]:
    if value is None:
        return []
    text = str(value).replace("；", ";")
    return [
        item.strip()
        for item in text.split(";")
        if item.strip()
        and item.strip().lower() not in {"none", "nan", "all strict quality gates passed"}
        and item.strip() not in {"无", "所有严格质量门槛通过"}
    ]


def _remove_entry_backtest_reasons(reasons: list[str]) -> list[str]:
    blocked = (
        "backtest sample too small",
        "backtest win rate too low",
        "backtest average return not positive",
    )
    return [reason for reason in reasons if not any(text in reason for text in blocked)]


def _remove_entry_backtest_reasons_zh(reasons: list[str]) -> list[str]:
    blocked = (
        "回测样本不足",
        "回测胜率不足",
        "回测平均收益不是正数",
    )
    return [reason for reason in reasons if not any(text in reason for text in blocked)]


def _validation_bucket(row: object) -> str:
    if bool(row.quality_gate_passed):
        return "high_probability"
    status = str(row.watchlist_status)
    if status == "close_but_not_ready":
        return "near_watchlist"
    if status == "early_watch":
        return "early_watchlist"
    return "filtered_out"


def _best_threshold(
    events: pd.DataFrame,
    column: str,
    current_threshold: float,
    candidates: tuple[int, ...],
    target_window: int,
    min_sample_count: int,
    direction: str,
) -> dict[str, object]:
    return_column = f"forward_return_{target_window}d"
    candidate_rows: list[dict[str, object]] = []
    for threshold in candidates:
        if direction == "higher":
            subset = events[events[column] >= threshold]
        else:
            subset = events[events[column] <= threshold]
        returns = subset[return_column].dropna()
        sample_count = len(returns)
        win_rate = float((returns > 0).mean()) if sample_count else np.nan
        avg_return = float(returns.mean()) if sample_count else np.nan
        candidate_rows.append(
            {
                "threshold": float(threshold),
                "sample_count": sample_count,
                "win_rate": win_rate,
                "avg_return": avg_return,
            }
        )

    eligible = [row for row in candidate_rows if row["sample_count"] >= min_sample_count]
    if not eligible:
        return {
            "threshold": np.nan,
            "sample_count": 0,
            "win_rate": np.nan,
            "avg_return": np.nan,
            "recommendation": "insufficient validation samples",
            "recommendation_zh": "验证样本不足，暂不建议调整",
        }

    best = sorted(
        eligible,
        key=lambda row: (
            -1 if not _is_finite(row["win_rate"]) else row["win_rate"],
            -1 if not _is_finite(row["avg_return"]) else row["avg_return"],
            row["sample_count"],
        ),
        reverse=True,
    )[0]
    if best["threshold"] > current_threshold:
        recommendation = "tighten threshold"
        recommendation_zh = "建议提高阈值"
    elif best["threshold"] < current_threshold:
        recommendation = "loosen threshold"
        recommendation_zh = "建议放宽阈值"
    else:
        recommendation = "keep threshold"
        recommendation_zh = "建议维持阈值"
    return {
        "threshold": best["threshold"],
        "sample_count": best["sample_count"],
        "win_rate": best["win_rate"],
        "avg_return": best["avg_return"],
        "recommendation": recommendation,
        "recommendation_zh": recommendation_zh,
    }


def _first_text(frame: pd.DataFrame, column: str, default: str) -> str:
    if column not in frame.columns:
        return default
    values = frame[column].dropna()
    if values.empty:
        return default
    return str(values.iloc[0])


def _target_window(forward_windows: tuple[int, ...]) -> int:
    return 20 if 20 in forward_windows else sorted(forward_windows)[len(forward_windows) // 2]


def build_validation_market_regime_lookup(prices: pd.DataFrame) -> dict[pd.Timestamp, dict[str, object]]:
    if prices.empty:
        return {}
    frame = _normalize_prices(prices)
    spy = frame[frame["ticker"] == "SPY"][["date", "adj_close"]].copy()
    qqq = frame[frame["ticker"] == "QQQ"][["date", "adj_close"]].copy()
    if spy.empty and qqq.empty:
        return {}

    benchmark = spy if not spy.empty else qqq
    benchmark = benchmark.sort_values("date").drop_duplicates("date", keep="last")
    benchmark["benchmark_return"] = benchmark["adj_close"].pct_change()
    benchmark["benchmark_60d_return"] = benchmark["adj_close"].pct_change(60)
    benchmark["benchmark_20d_volatility"] = (
        benchmark["benchmark_return"].rolling(20).std() * np.sqrt(252)
    )
    benchmark["benchmark_ma_200"] = benchmark["adj_close"].rolling(200).mean()

    if qqq.empty:
        qqq_trend = pd.DataFrame(columns=["date", "qqq_trend_score"])
    else:
        qqq_trend = _benchmark_trend_frame(qqq, "qqq_trend_score")
    spy_trend = _benchmark_trend_frame(spy, "spy_trend_score") if not spy.empty else pd.DataFrame(
        columns=["date", "spy_trend_score"]
    )
    regime_frame = benchmark.merge(spy_trend, on="date", how="left").merge(
        qqq_trend,
        on="date",
        how="left",
    )

    lookup: dict[pd.Timestamp, dict[str, object]] = {}
    for row in regime_frame.itertuples(index=False):
        date = pd.Timestamp(row.date).normalize()
        regime, regime_zh = _validation_market_regime_label(
            close=float(row.adj_close),
            ma_200=_safe_float(getattr(row, "benchmark_ma_200", np.nan)),
            return_60d=_safe_float(getattr(row, "benchmark_60d_return", np.nan)),
            volatility_20d=_safe_float(getattr(row, "benchmark_20d_volatility", np.nan)),
        )
        lookup[date] = {
            "validation_market_regime": regime,
            "validation_market_regime_zh": regime_zh,
            "validation_spy_trend_score": _safe_float(
                getattr(row, "spy_trend_score", np.nan)
            ),
            "validation_qqq_trend_score": _safe_float(
                getattr(row, "qqq_trend_score", np.nan)
            ),
            "validation_benchmark_60d_return": _safe_float(
                getattr(row, "benchmark_60d_return", np.nan)
            ),
            "validation_benchmark_20d_volatility": _safe_float(
                getattr(row, "benchmark_20d_volatility", np.nan)
            ),
        }
    return lookup


def _benchmark_trend_frame(prices: pd.DataFrame, output_column: str) -> pd.DataFrame:
    frame = prices[["date", "adj_close"]].copy().sort_values("date")
    frame = frame.drop_duplicates("date", keep="last")
    frame["return_60d"] = frame["adj_close"].pct_change(60)
    frame["ma_200"] = frame["adj_close"].rolling(200).mean()
    scores: list[float] = []
    for row in frame.itertuples(index=False):
        score = 50.0
        close = _safe_float(row.adj_close)
        ma_200 = _safe_float(row.ma_200)
        return_60d = _safe_float(row.return_60d)
        if _is_finite(close) and _is_finite(ma_200):
            score += 20.0 if close >= ma_200 else -20.0
        if _is_finite(return_60d):
            score += max(-20.0, min(20.0, return_60d * 200.0))
        scores.append(round(float(_clamp(score, 0.0, 100.0)), 2))
    return frame[["date"]].assign(**{output_column: scores})


def _validation_market_regime_label(
    close: float,
    ma_200: float,
    return_60d: float,
    volatility_20d: float,
) -> tuple[str, str]:
    if not _is_finite(close) or not _is_finite(ma_200):
        return "unknown", "未知市场状态"
    if _is_finite(volatility_20d) and volatility_20d >= 0.28:
        return "high_volatility", "高波动市场"
    if close >= ma_200 and _is_finite(return_60d) and return_60d > 0:
        return "bull_uptrend", "牛市上升"
    if close < ma_200 and _is_finite(return_60d) and return_60d < 0:
        return "bear_downtrend", "熊市下跌"
    return "sideways_mixed", "震荡分化"


def _lookup_market_regime(
    signal_date: pd.Timestamp,
    lookup: dict[pd.Timestamp, dict[str, object]],
) -> dict[str, object]:
    default = {
        "validation_market_regime": "unknown",
        "validation_market_regime_zh": "未知市场状态",
        "validation_spy_trend_score": np.nan,
        "validation_qqq_trend_score": np.nan,
        "validation_benchmark_60d_return": np.nan,
        "validation_benchmark_20d_volatility": np.nan,
    }
    if not lookup:
        return default
    date = pd.Timestamp(signal_date).normalize()
    if date in lookup:
        return lookup[date]
    available_dates = [candidate for candidate in lookup if candidate <= date]
    if not available_dates:
        return default
    return lookup[max(available_dates)]


def _normalize_prices(prices: pd.DataFrame) -> pd.DataFrame:
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=False)
    frame["ticker"] = frame["ticker"].astype("string").str.upper().str.strip()
    numeric_columns = ["open", "high", "low", "close", "adj_close", "volume"]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["date", "ticker", "adj_close"]).sort_values(["ticker", "date"]).reset_index(drop=True)


def _safe_int(value: object) -> int:
    try:
        if pd.isna(value):
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def _safe_optional_float(value: object) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _optional_timestamp(value: object) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(timestamp):
        return None
    return timestamp.normalize()


def _format_optional_date(value: object) -> str:
    timestamp = _optional_timestamp(value)
    if timestamp is None:
        return ""
    return timestamp.date().isoformat()


def _numeric_mean(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if len(values) else np.nan


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _validation_market_context() -> SimpleNamespace:
    return SimpleNamespace(
        market_score=60.0,
        market_status="neutral_validation",
        note="Validation uses neutral market context.",
        note_zh="验证使用中性大盘环境。",
        warning="",
    )


def _validation_relative_strength_contexts(horizons: tuple[str, ...]) -> dict[str, SimpleNamespace]:
    return {
        horizon: SimpleNamespace(
            score=50.0,
            note="Validation uses neutral benchmark-relative context.",
            note_zh="验证使用中性相对强弱环境。",
            warning="",
            vs_spy_return=np.nan,
            vs_qqq_return=np.nan,
        )
        for horizon in horizons
    }


def _validation_event_context() -> SimpleNamespace:
    return SimpleNamespace(
        event_risk_level="low",
        event_risk_level_zh="低",
        event_risk_score=20.0,
        next_earnings_date="validation_neutral",
        days_until_earnings=999,
        last_earnings_date="validation_neutral",
        days_since_earnings=999,
        event_window="normal",
        event_window_zh="正常",
        event_block_new_entries=False,
        event_cooldown_active=False,
        event_risk_note="Validation uses neutral event-risk context.",
        event_risk_note_zh="验证使用中性事件风险。",
        event_risk_warning="",
    )


def _validation_fundamental_context(ticker: str) -> SimpleNamespace:
    return SimpleNamespace(
        ticker=ticker,
        fundamental_score=60.0,
        fundamental_quality="neutral",
        fundamental_quality_zh="中性",
        fundamental_note="Validation uses neutral fundamental context.",
        fundamental_note_zh="验证使用中性基本面环境。",
        fundamental_warning="",
        data_coverage=1.0,
        revenue_growth=np.nan,
        earnings_growth=np.nan,
        profit_margin=np.nan,
        return_on_equity=np.nan,
        free_cash_flow=np.nan,
        forward_pe=np.nan,
        peg_ratio=np.nan,
        debt_to_equity=np.nan,
    )


def _validation_sentiment_context(ticker: str) -> SimpleNamespace:
    return SimpleNamespace(
        ticker=ticker,
        sentiment_score=50.0,
        sentiment_label="neutral",
        sentiment_label_zh="中性",
        sentiment_risk_level="low",
        sentiment_risk_level_zh="低",
        sentiment_block_new_entries=False,
        sentiment_positive_count=0,
        sentiment_negative_count=0,
        sentiment_high_risk_count=0,
        sentiment_titles_used=0,
        sentiment_note="Validation uses neutral news-sentiment context.",
        sentiment_note_zh="验证使用中性新闻情绪环境。",
        sentiment_warning="",
    )


def _validation_analyst_context(ticker: str) -> SimpleNamespace:
    return SimpleNamespace(
        ticker=ticker,
        analyst_score=50.0,
        analyst_label="neutral",
        analyst_label_zh="中性",
        analyst_risk_level="low",
        analyst_risk_level_zh="低",
        analyst_block_new_entries=False,
        analyst_upside=np.nan,
        recommendation_mean=np.nan,
        recommendation_key="validation_neutral",
        number_of_analysts=np.nan,
        target_mean_price=np.nan,
        analyst_note="Validation uses neutral analyst-expectation context.",
        analyst_note_zh="验证使用中性分析师预期环境。",
        analyst_warning="",
        data_coverage=1.0,
    )


def _validation_valuation_context(ticker: str) -> SimpleNamespace:
    return SimpleNamespace(
        ticker=ticker,
        valuation_score=50.0,
        valuation_label="reasonable",
        valuation_label_zh="合理",
        valuation_risk_level="low",
        valuation_risk_level_zh="低",
        valuation_block_new_entries=False,
        valuation_forward_pe=np.nan,
        valuation_peg_ratio=np.nan,
        valuation_growth_reference=np.nan,
        valuation_profit_margin=np.nan,
        valuation_note="Validation uses neutral valuation context.",
        valuation_note_zh="验证使用中性估值环境。",
        valuation_warning="",
        data_coverage=1.0,
    )


def _validation_sector_context() -> SimpleNamespace:
    return SimpleNamespace(
        sector="validation_neutral",
        industry="validation_neutral",
        sector_etf="VALIDATION",
        sector_score=50.0,
        sector_status="neutral",
        sector_note="Validation uses neutral sector context.",
        sector_note_zh="验证使用中性板块环境。",
        sector_warning="",
        sector_trend_score=50.0,
        sector_relative_strength=np.nan,
    )


def _format_metric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    for column in display.columns:
        if column in {"probability_bucket", "probability_bucket_zh"}:
            continue
        if any(
            token in column
            for token in ["return", "win_rate", "drawdown", "probability", "calibration_error"]
        ):
            display[column] = display[column].map(_format_percent)
    return display


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["No rows."]
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False):
        values = [_format_number(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _format_number(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _format_percent(value: object) -> str:
    if pd.isna(value):
        return ""
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return str(value)


def _is_finite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False
