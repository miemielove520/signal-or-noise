from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .data import download_prices_for_period_multi_source
from .json_io import dataframe_records, write_json
from .screening_config import (
    ScreeningConfig,
    ScreeningThresholds,
    default_screening_config,
    render_screening_config_toml,
)
from .universe import BUILT_IN_UNIVERSES, load_universe_tickers
from .walk_forward import DEFAULT_BENCHMARK_TICKERS, WalkForwardResult, run_walk_forward_validation


@dataclass(frozen=True)
class UniverseValidationBatchResult:
    summary: pd.DataFrame
    ranking: pd.DataFrame
    report: str
    output_dir: Path
    failures: list[dict[str, str]]


PriceLoader = Callable[..., object]
ValidationRunner = Callable[..., WalkForwardResult]
ProgressCallback = Callable[[str, str, int, int], None]


UNIVERSE_PROFILE_MAP = {
    "ai-infrastructure": "ai_infrastructure",
    "cybersecurity": "cybersecurity",
    "saas-software": "saas_software",
    "software": "saas_software",
    "fintech-high-beta": "fintech_high_beta",
    "defensive-quality": "defensive_quality",
    "consumer-staples": "defensive_quality",
    "industrial-quality": "industrial_quality",
    "energy-industrials": "industrial_quality",
    "semiconductors": "semiconductor",
    "mega-cap-tech": "mega_cap_tech",
    "healthcare-quality": "default",
    "financial-quality": "default",
    "consumer-discretionary": "default",
    "small-mid-growth": "default",
    "growth-core": "default",
    "balanced-core": "default",
    "research-core": "default",
    "sector-core": "default",
}


def run_all_builtin_universe_validations(
    universe_names: Iterable[str] | None = None,
    period: str = "5y",
    step_days: int = 20,
    min_history_days: int = 170,
    output_dir: str | Path = "outputs/walk_forward/all_universes",
    screening_config: ScreeningConfig | None = None,
    price_loader: PriceLoader = download_prices_for_period_multi_source,
    validation_runner: ValidationRunner = run_walk_forward_validation,
    progress_callback: ProgressCallback | None = None,
) -> UniverseValidationBatchResult:
    selected_universes = _normalize_universe_names(universe_names)
    effective_config = screening_config or default_screening_config()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []

    for index, universe_name in enumerate(selected_universes, start=1):
        tickers = load_universe_tickers(universe_name=universe_name)
        universe_output_dir = root / universe_name
        data_path = Path("data/real_prices") / f"walk_forward_{universe_name}_{period}.csv"
        _emit_progress(progress_callback, universe_name, "started", index, len(selected_universes))
        try:
            price_result = price_loader(
                _with_benchmark_tickers(tickers),
                period=period,
                output_path=data_path,
            )
            result = validation_runner(
                prices=price_result.prices,
                tickers=tickers,
                step_days=step_days,
                min_history_days=min_history_days,
                output_dir=universe_output_dir,
                screening_config=effective_config,
                screening_profile_name=UNIVERSE_PROFILE_MAP.get(universe_name),
            )
            rows.append(
                _with_universe_diagnosis(
                    _universe_summary_row(
                        universe_name=universe_name,
                        tickers=tickers,
                        provider=str(getattr(price_result, "provider", "unknown")),
                        output_dir=universe_output_dir,
                        result=result,
                        screening_config=effective_config,
                        status="ok",
                        error="",
                    )
                )
            )
            _emit_progress(progress_callback, universe_name, "completed", index, len(selected_universes))
        except Exception as error:
            message = str(error)
            failures.append({"universe": universe_name, "error": message})
            rows.append(
                _with_universe_diagnosis(
                    _failed_universe_summary_row(
                        universe_name=universe_name,
                        tickers=tickers,
                        output_dir=universe_output_dir,
                        screening_config=effective_config,
                        error=message,
                    )
                )
            )
            _emit_progress(progress_callback, universe_name, "failed", index, len(selected_universes))

        _write_batch_outputs(
            rows=rows,
            root=root,
            period=period,
            step_days=step_days,
            min_history_days=min_history_days,
            selected_universes=selected_universes,
            failures=failures,
            screening_config=effective_config,
        )

    summary = _sorted_summary(rows)
    ranking = build_best_universe_ranking(summary)
    coverage_plan = build_validation_coverage_plan(selected_universes)
    report = render_all_universe_validation_report(
        summary,
        period,
        step_days,
        min_history_days,
        ranking=ranking,
        coverage_plan=coverage_plan,
    )

    return UniverseValidationBatchResult(
        summary=summary,
        ranking=ranking,
        report=report,
        output_dir=root,
        failures=failures,
    )


def _emit_progress(
    progress_callback: ProgressCallback | None,
    universe_name: str,
    status: str,
    index: int,
    total: int,
) -> None:
    if progress_callback is not None:
        progress_callback(universe_name, status, index, total)


def _write_batch_outputs(
    rows: list[dict[str, object]],
    root: Path,
    period: str,
    step_days: int,
    min_history_days: int,
    selected_universes: list[str],
    failures: list[dict[str, str]],
    screening_config: ScreeningConfig,
) -> None:
    summary = _sorted_summary(rows)
    ranking = build_best_universe_ranking(summary)
    coverage_plan = build_validation_coverage_plan(selected_universes)
    report = render_all_universe_validation_report(
        summary,
        period,
        step_days,
        min_history_days,
        ranking=ranking,
        coverage_plan=coverage_plan,
    )
    suggested_config_text = render_suggested_universe_screening_config(
        screening_config=screening_config,
        summary=summary,
    )
    summary.to_csv(root / "all_universe_validation_summary.csv", index=False)
    ranking.to_csv(root / "best_universe_ranking.csv", index=False)
    coverage_plan.to_csv(root / "validation_coverage_plan.csv", index=False)
    (root / "validation_coverage_plan.md").write_text(
        render_validation_coverage_plan(coverage_plan),
        encoding="utf-8",
    )
    (root / "all_universe_validation_report.md").write_text(report, encoding="utf-8")
    (root / "all_universe_suggested_screening.toml").write_text(
        suggested_config_text,
        encoding="utf-8",
    )
    write_json(
        root / "all_universe_validation_result.json",
        {
            "period": period,
            "step_days": step_days,
            "min_history_days": min_history_days,
            "universe_count": len(selected_universes),
            "completed_universe_count": len(rows),
            "summary": dataframe_records(summary),
            "ranking": dataframe_records(ranking),
            "coverage_plan": dataframe_records(coverage_plan),
            "failures": failures,
            "output_files": {
                "summary_csv": str(root / "all_universe_validation_summary.csv"),
                "best_universe_ranking_csv": str(root / "best_universe_ranking.csv"),
                "validation_coverage_plan_csv": str(root / "validation_coverage_plan.csv"),
                "validation_coverage_plan_md": str(root / "validation_coverage_plan.md"),
                "markdown_report": str(root / "all_universe_validation_report.md"),
                "suggested_screening_toml": str(root / "all_universe_suggested_screening.toml"),
                "json": str(root / "all_universe_validation_result.json"),
            },
        },
    )


def render_suggested_universe_screening_config(
    screening_config: ScreeningConfig,
    summary: pd.DataFrame,
) -> str:
    suggested_thresholds = {
        profile.name: profile.thresholds for profile in screening_config.profiles
    }
    suggested_thresholds["default"] = screening_config.default_thresholds
    applied_notes: list[str] = []
    skipped_notes: list[str] = []
    applied_profiles: set[str] = set()

    for row in summary.itertuples(index=False):
        universe_name = str(getattr(row, "universe", ""))
        profile_name = UNIVERSE_PROFILE_MAP.get(universe_name)
        changes_text = str(getattr(row, "suggested_threshold_changes", "none"))
        if not profile_name or changes_text.lower() == "none":
            continue
        if profile_name in applied_profiles:
            skipped_notes.append(
                f"{universe_name}: skipped because profile {profile_name} already has suggestions."
            )
            continue
        current = suggested_thresholds.get(profile_name)
        if current is None:
            skipped_notes.append(f"{universe_name}: skipped because profile {profile_name} is missing.")
            continue
        overrides = _threshold_overrides_from_suggestions(changes_text)
        if not overrides:
            skipped_notes.append(f"{universe_name}: skipped because suggestions could not be parsed.")
            continue
        updated = ScreeningThresholds.from_mapping({**current.to_dict(), **overrides})
        suggested_thresholds[profile_name] = updated
        applied_profiles.add(profile_name)
        applied_notes.append(f"{profile_name}: {changes_text}")

    suggested_config = screening_config.with_profile_thresholds(suggested_thresholds)
    header = [
        "Experimental screening config generated from all-universe validation.",
        "Do not replace configs/screening.toml until this config passes a separate validation run.",
        "This file applies only parsed suggested_threshold_changes from the aggregate report.",
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


def _threshold_overrides_from_suggestions(changes_text: str) -> dict[str, object]:
    overrides: dict[str, object] = {}
    for raw_change in changes_text.split(";"):
        change = raw_change.strip()
        if not change or ":" not in change or "->" not in change:
            continue
        key, value_text = change.split(":", 1)
        _, suggested_text = value_text.split("->", 1)
        key = key.strip()
        suggested_text = suggested_text.strip()
        try:
            number = float(suggested_text)
        except ValueError:
            continue
        if key == "backtest_sample_min":
            overrides[key] = int(round(number))
        else:
            overrides[key] = number
    return overrides


def _sorted_summary(rows: list[dict[str, object]]) -> pd.DataFrame:
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary["_status_rank"] = summary["status"].map({"ok": 0}).fillna(1)
    return (
        summary.sort_values(
            [
                "_status_rank",
                "optimization_priority",
                "high_probability_20d_win_rate",
                "all_20d_win_rate",
            ],
            ascending=[True, True, False, False],
            na_position="last",
        )
        .drop(columns=["_status_rank"])
        .reset_index(drop=True)
    )


def build_best_universe_ranking(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "rank",
        "universe",
        "decision",
        "decision_zh",
        "ranking_score",
        "optimization_priority",
        "diagnostic_level",
        "diagnostic_level_zh",
        "recommendation_zh",
        "ticker_count",
        "event_count",
        "all_sample_count",
        "all_20d_win_rate",
        "all_20d_avg_return",
        "high_probability_sample_count",
        "high_probability_20d_win_rate",
        "high_probability_20d_avg_return",
        "why_ranked_here",
        "why_ranked_here_zh",
        "output_dir",
    ]
    if summary.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for raw_row in summary.to_dict("records"):
        ranking_score = _candidate_ranking_score(raw_row)
        decision, decision_zh = _candidate_ranking_decision(raw_row, ranking_score)
        why, why_zh = _candidate_ranking_reason(raw_row, ranking_score, decision)
        rows.append(
            {
                "rank": 0,
                "universe": raw_row.get("universe", ""),
                "decision": decision,
                "decision_zh": decision_zh,
                "ranking_score": ranking_score,
                "optimization_priority": raw_row.get("optimization_priority", np.nan),
                "diagnostic_level": raw_row.get("diagnostic_level", ""),
                "diagnostic_level_zh": raw_row.get("diagnostic_level_zh", ""),
                "recommendation_zh": raw_row.get("recommendation_zh", ""),
                "ticker_count": raw_row.get("ticker_count", 0),
                "event_count": raw_row.get("event_count", 0),
                "all_sample_count": raw_row.get("all_sample_count", 0),
                "all_20d_win_rate": raw_row.get("all_20d_win_rate", np.nan),
                "all_20d_avg_return": raw_row.get("all_20d_avg_return", np.nan),
                "high_probability_sample_count": raw_row.get("high_probability_sample_count", 0),
                "high_probability_20d_win_rate": raw_row.get(
                    "high_probability_20d_win_rate",
                    np.nan,
                ),
                "high_probability_20d_avg_return": raw_row.get(
                    "high_probability_20d_avg_return",
                    np.nan,
                ),
                "why_ranked_here": why,
                "why_ranked_here_zh": why_zh,
                "output_dir": raw_row.get("output_dir", ""),
            }
        )

    ranking = pd.DataFrame(rows)
    decision_rank = {
        "optimize_first": 0,
        "research_next": 1,
        "need_more_samples": 2,
        "deprioritize": 3,
        "data_issue": 4,
    }
    ranking["_decision_rank"] = ranking["decision"].map(decision_rank).fillna(9)
    ranking = (
        ranking.sort_values(
            ["_decision_rank", "ranking_score", "event_count"],
            ascending=[True, False, False],
            na_position="last",
        )
        .drop(columns=["_decision_rank"])
        .reset_index(drop=True)
    )
    ranking["rank"] = range(1, len(ranking) + 1)
    return ranking[columns]


def _candidate_ranking_score(row: dict[str, object]) -> float:
    if str(row.get("status", "")) != "ok":
        return 0.0

    priority = _safe_float(row.get("optimization_priority"))
    if not np.isfinite(priority):
        priority = 9.0
    event_count = _safe_int(row.get("event_count"))
    high_probability_sample = _safe_int(row.get("high_probability_sample_count"))
    all_win_rate = _safe_float(row.get("all_20d_win_rate"))
    all_avg_return = _safe_float(row.get("all_20d_avg_return"))
    high_probability_win_rate = _safe_float(row.get("high_probability_20d_win_rate"))
    high_probability_avg_return = _safe_float(row.get("high_probability_20d_avg_return"))
    avg_drawdown = _safe_float(row.get("all_avg_max_drawdown"))

    score = 45.0 + max(0.0, 10.0 - priority) * 4.0
    score += min(event_count / 100.0, 1.0) * 10.0

    if np.isfinite(all_win_rate):
        score += (all_win_rate - 0.50) * 50.0
    if np.isfinite(all_avg_return):
        score += all_avg_return * 150.0

    if high_probability_sample >= 10:
        score += min(high_probability_sample / 30.0, 1.0) * 8.0
        if np.isfinite(high_probability_win_rate):
            score += (high_probability_win_rate - 0.55) * 70.0
        if np.isfinite(high_probability_avg_return):
            score += high_probability_avg_return * 150.0
    elif high_probability_sample > 0:
        score += min(high_probability_sample / 10.0, 1.0) * 3.0 - 8.0
    else:
        score -= 12.0

    if np.isfinite(avg_drawdown):
        score -= min(abs(avg_drawdown), 0.30) * 40.0

    if event_count < 20:
        score = min(score, 45.0)

    return round(_clamp_number(score, 0.0, 100.0), 2)


def _candidate_ranking_decision(row: dict[str, object], ranking_score: float) -> tuple[str, str]:
    status = str(row.get("status", ""))
    event_count = _safe_int(row.get("event_count"))
    diagnostic_level = str(row.get("diagnostic_level", ""))
    high_probability_sample = _safe_int(row.get("high_probability_sample_count"))

    if status != "ok":
        return "data_issue", "数据问题"
    if event_count < 20:
        return "need_more_samples", "需要更多样本"
    if diagnostic_level == "optimize_first" or (
        ranking_score >= 75 and high_probability_sample >= 10
    ):
        return "optimize_first", "优先优化"
    if diagnostic_level in {
        "promising_but_needs_filtering",
        "thresholds_too_strict",
        "research_candidate",
        "gate_not_working_well",
    } or ranking_score >= 55:
        return "research_next", "继续研究"
    return "deprioritize", "降低优先级"


def _candidate_ranking_reason(
    row: dict[str, object],
    ranking_score: float,
    decision: str,
) -> tuple[str, str]:
    event_count = _safe_int(row.get("event_count"))
    high_probability_sample = _safe_int(row.get("high_probability_sample_count"))
    all_win_rate = _format_percent(row.get("all_20d_win_rate"))
    high_probability_win_rate = _format_percent(row.get("high_probability_20d_win_rate"))
    recommendation = str(row.get("recommendation", ""))
    recommendation_zh = str(row.get("recommendation_zh", ""))

    if decision == "data_issue":
        return "Validation failed; fix data coverage first.", "验证失败；先修复数据覆盖。"
    if decision == "need_more_samples":
        return (
            f"Only {event_count} validation events; score capped at {ranking_score:.2f}.",
            f"只有 {event_count} 个验证样本；排名分数被限制在 {ranking_score:.2f}。",
        )
    return (
        f"{recommendation} Events={event_count}, high-probability samples="
        f"{high_probability_sample}, all win rate={all_win_rate}, "
        f"strict win rate={high_probability_win_rate}, ranking score={ranking_score:.2f}.",
        f"{recommendation_zh} 样本={event_count}，高概率样本={high_probability_sample}，"
        f"整体胜率={all_win_rate}，严格筛选胜率={high_probability_win_rate}，"
        f"排名分数={ranking_score:.2f}。",
    )


def render_all_universe_validation_report(
    summary: pd.DataFrame,
    period: str,
    step_days: int,
    min_history_days: int,
    ranking: pd.DataFrame | None = None,
    coverage_plan: pd.DataFrame | None = None,
) -> str:
    ranking = build_best_universe_ranking(summary) if ranking is None else ranking
    coverage = (
        build_validation_coverage_plan(list(summary["universe"].dropna().astype(str)))
        if coverage_plan is None and not summary.empty and "universe" in summary.columns
        else coverage_plan
    )
    lines = [
        "# All Universe Walk-Forward Validation / 全部股票池滚动验证",
        "",
        "This report compares each built-in universe using the same validation settings.",
        "本报告用同一套验证参数比较每个内置股票池。",
        "",
        f"- Period / 数据周期: `{period}`",
        f"- Step days / 信号间隔天数: `{step_days}`",
        f"- Minimum history days / 最少历史天数: `{min_history_days}`",
        f"- Built-in universe count / 内置股票池数量: `{len(BUILT_IN_UNIVERSES)}`",
        "- Large sample universes / 大样本股票池: `growth-core`, `balanced-core`, `research-core`",
        "",
        "## Summary / 总结",
        "",
    ]
    lines.extend(_diagnostic_highlights(summary))
    lines.extend(["", "## Validation Coverage Plan / 验证覆盖计划", ""])
    coverage_frame = coverage if coverage is not None else pd.DataFrame()
    if coverage_frame.empty:
        lines.append("No coverage plan available. / 暂无覆盖计划。")
    else:
        lines.extend(_markdown_table(_format_coverage_for_report(coverage_frame)))
    lines.extend(["", "## Best Candidate Universe Ranking / 最佳股票池排序", ""])
    lines.extend(_markdown_table(_format_ranking_for_report(ranking)))
    lines.extend(["", "## Universe Diagnostics / 股票池诊断", ""])
    lines.extend(_markdown_table(_format_summary_for_report(summary)))
    lines.extend(
        [
            "",
            "## How To Read / 如何阅读",
            "",
            "- `all_20d_win_rate` shows the 20-trading-day win rate across all generated validation signals.",
            "- `all_20d_win_rate` 表示所有历史验证信号在未来20个交易日的胜率。",
            "- `high_probability_20d_win_rate` focuses only on signals that passed the strict high-probability gate.",
            "- `high_probability_20d_win_rate` 只统计通过高概率筛选器的信号，更适合判断严格筛选后的质量。",
            "- Higher sample count matters. A high win rate with very few samples should be treated carefully.",
            "- 样本数很重要。样本很少但胜率很高时，需要谨慎看待。",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def build_validation_coverage_plan(
    universe_names: Iterable[str] | None = None,
    standard_period: str = "5y",
    deep_period: str = "10y",
    standard_step_days: int = 20,
    deep_step_days: int = 10,
    min_history_days: int = 252,
) -> pd.DataFrame:
    selected_universes = _normalize_universe_names(universe_names)
    rows: list[dict[str, object]] = []
    for universe_name in selected_universes:
        tickers = load_universe_tickers(universe_name=universe_name)
        profile_name = UNIVERSE_PROFILE_MAP.get(universe_name, "default")
        target_events = max(60, len(tickers) * 8)
        target_high_probability_events = max(20, len(tickers) * 2)
        rows.append(
            {
                "universe": universe_name,
                "mapped_profile": profile_name,
                "ticker_count": len(tickers),
                "target_event_count": target_events,
                "target_high_probability_event_count": target_high_probability_events,
                "standard_command": (
                    f"python3 validate.py --universe {universe_name} --period {standard_period} "
                    f"--step-days {standard_step_days} --min-history-days 170"
                ),
                "deep_command": (
                    f"python3 validate.py --universe {universe_name} --period {deep_period} "
                    f"--step-days {deep_step_days} --min-history-days {min_history_days}"
                ),
                "config_apply_gate": "compare_runs_required",
                "config_apply_gate_zh": "必须先用 compare_runs.py 对比旧配置和候选配置",
                "coverage_note": (
                    "Use standard validation for iteration, then deep validation before changing config."
                ),
                "coverage_note_zh": "日常迭代用标准验证；真正改配置前必须跑深度验证。",
            }
        )
    return pd.DataFrame(rows)


def render_validation_coverage_plan(coverage_plan: pd.DataFrame) -> str:
    lines = [
        "# Validation Coverage Plan / 验证覆盖计划",
        "",
        "This plan defines the minimum validation workflow before changing screening rules.",
        "本计划定义修改筛选规则前的最低验证流程。",
        "",
        "## Coverage Matrix / 覆盖矩阵",
        "",
    ]
    lines.extend(_markdown_table(_format_coverage_for_report(coverage_plan)))
    lines.extend(
        [
            "",
            "## Safe Config Workflow / 安全配置流程",
            "",
            "1. Run the current config and save the output folder.",
            "1. 先用当前配置跑验证，并保留输出文件夹。",
            "2. Validate `suggested_screening.toml` into a separate output folder.",
            "2. 用 `suggested_screening.toml` 另跑一次验证到新的输出文件夹。",
            "3. Run `python3 compare_runs.py OLD_RUN NEW_RUN`.",
            "3. 运行 `python3 compare_runs.py 旧输出 新输出`。",
            "4. Apply the candidate config only if the final decision passes review.",
            "4. 只有最终结论通过审核时，才考虑采用候选配置。",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _normalize_universe_names(universe_names: Iterable[str] | None) -> list[str]:
    if universe_names is None:
        return list(BUILT_IN_UNIVERSES)
    normalized: list[str] = []
    for name in universe_names:
        key = str(name).lower().strip()
        if not key:
            continue
        if key not in BUILT_IN_UNIVERSES:
            choices = ", ".join(sorted(BUILT_IN_UNIVERSES))
            raise ValueError(f"Unknown built-in universe '{name}'. Choices: {choices}.")
        if key not in normalized:
            normalized.append(key)
    if not normalized:
        raise ValueError("At least one built-in universe is required.")
    return normalized


def _with_benchmark_tickers(tickers: list[str]) -> list[str]:
    combined = [*tickers, *DEFAULT_BENCHMARK_TICKERS]
    return list(dict.fromkeys(str(ticker).upper().strip() for ticker in combined if str(ticker).strip()))


def _universe_summary_row(
    universe_name: str,
    tickers: list[str],
    provider: str,
    output_dir: Path,
    result: WalkForwardResult,
    screening_config: ScreeningConfig,
    status: str,
    error: str,
) -> dict[str, object]:
    all_metrics = _bucket_metrics(result.events, bucket=None)
    high_probability_metrics = _bucket_metrics(result.events, bucket="high_probability")
    near_watchlist_metrics = _bucket_metrics(result.events, bucket="near_watchlist")
    early_watchlist_metrics = _bucket_metrics(result.events, bucket="early_watchlist")
    top_gate_failures = _top_semicolon_items(result.events, "quality_gate_fail_reasons")
    top_gate_failures_zh = _top_semicolon_items(result.events, "quality_gate_fail_reasons_zh")
    top_missing_items = _top_semicolon_items(result.events, "watchlist_missing_items")
    top_missing_items_zh = _top_semicolon_items(result.events, "watchlist_missing_items_zh")
    threshold_suggestions = _suggest_threshold_changes(
        universe_name=universe_name,
        events=result.events,
        screening_config=screening_config,
        all_metrics=all_metrics,
    )

    return {
        "universe": universe_name,
        "status": status,
        "ticker_count": len(tickers),
        "tickers": ", ".join(tickers),
        "price_provider": provider,
        "event_count": len(result.events),
        "all_sample_count": all_metrics["sample_count"],
        "all_20d_win_rate": all_metrics["win_rate_20d"],
        "all_20d_avg_return": all_metrics["avg_return_20d"],
        "all_avg_max_drawdown": all_metrics["avg_max_drawdown"],
        "high_probability_sample_count": high_probability_metrics["sample_count"],
        "high_probability_20d_win_rate": high_probability_metrics["win_rate_20d"],
        "high_probability_20d_avg_return": high_probability_metrics["avg_return_20d"],
        "near_watchlist_sample_count": near_watchlist_metrics["sample_count"],
        "near_watchlist_20d_win_rate": near_watchlist_metrics["win_rate_20d"],
        "near_watchlist_20d_avg_return": near_watchlist_metrics["avg_return_20d"],
        "early_watchlist_sample_count": early_watchlist_metrics["sample_count"],
        "early_watchlist_20d_win_rate": early_watchlist_metrics["win_rate_20d"],
        "early_watchlist_20d_avg_return": early_watchlist_metrics["avg_return_20d"],
        "top_quality_gate_failures": top_gate_failures,
        "top_quality_gate_failures_zh": top_gate_failures_zh,
        "top_watchlist_missing_items": top_missing_items,
        "top_watchlist_missing_items_zh": top_missing_items_zh,
        **threshold_suggestions,
        "output_dir": str(output_dir),
        "error": error,
    }


def _failed_universe_summary_row(
    universe_name: str,
    tickers: list[str],
    output_dir: Path,
    screening_config: ScreeningConfig,
    error: str,
) -> dict[str, object]:
    return _universe_summary_row(
        universe_name=universe_name,
        tickers=tickers,
        provider="",
        output_dir=output_dir,
        result=WalkForwardResult(
            events=pd.DataFrame(),
            summary=pd.DataFrame(),
            profile_summary=pd.DataFrame(),
            segment_summary=pd.DataFrame(),
            market_regime_summary=pd.DataFrame(),
            market_regime_policy=pd.DataFrame(),
            profile_calibration=pd.DataFrame(),
            calibration=pd.DataFrame(),
            report="",
        ),
        screening_config=screening_config,
        status="failed",
        error=error,
    )


def _with_universe_diagnosis(row: dict[str, object]) -> dict[str, object]:
    diagnosis = _diagnose_universe(row)
    return {**row, **diagnosis}


def _diagnose_universe(row: dict[str, object]) -> dict[str, object]:
    status = str(row.get("status", ""))
    event_count = _safe_int(row.get("event_count"))
    high_probability_sample = _safe_int(row.get("high_probability_sample_count"))
    early_watchlist_sample = _safe_int(row.get("early_watchlist_sample_count"))
    near_watchlist_sample = _safe_int(row.get("near_watchlist_sample_count"))
    all_win_rate = _safe_float(row.get("all_20d_win_rate"))
    all_avg_return = _safe_float(row.get("all_20d_avg_return"))
    high_probability_win_rate = _safe_float(row.get("high_probability_20d_win_rate"))
    high_probability_avg_return = _safe_float(row.get("high_probability_20d_avg_return"))

    if status != "ok":
        return {
            "optimization_priority": 99,
            "diagnostic_level": "data_issue",
            "diagnostic_level_zh": "数据问题",
            "recommendation": "Fix data download or ticker coverage before model work.",
            "recommendation_zh": "先修复数据下载或股票覆盖问题，再讨论模型优化。",
            "diagnosis_reasons": "Universe validation failed.",
            "diagnosis_reasons_zh": "该股票池验证失败。",
        }

    if event_count < 20:
        return {
            "optimization_priority": 50,
            "diagnostic_level": "insufficient_samples",
            "diagnostic_level_zh": "样本不足",
            "recommendation": "Increase period or lower validation frequency before judging this universe.",
            "recommendation_zh": "先增加历史周期或调整验证间隔，样本不足时不要急着判断优劣。",
            "diagnosis_reasons": f"Only {event_count} validation events.",
            "diagnosis_reasons_zh": f"只有 {event_count} 个验证样本。",
        }

    if (
        high_probability_sample >= 10
        and _is_number_at_least(high_probability_win_rate, 0.55)
        and _is_number_at_least(high_probability_avg_return, 0.0)
    ):
        return {
            "optimization_priority": 1,
            "diagnostic_level": "optimize_first",
            "diagnostic_level_zh": "优先优化",
            "recommendation": "Prioritize this universe for tighter thresholds and deeper feature work.",
            "recommendation_zh": "优先优化这个股票池，适合继续细化阈值和特征。",
            "diagnosis_reasons": (
                "High-probability samples have acceptable count, win rate, and average return."
            ),
            "diagnosis_reasons_zh": "高概率样本的数量、胜率、平均收益都基本合格。",
        }

    if high_probability_sample >= 5:
        if _is_number_at_least(high_probability_avg_return, 0.0):
            return {
                "optimization_priority": 2,
                "diagnostic_level": "promising_but_needs_filtering",
                "diagnostic_level_zh": "有潜力但需过滤",
                "recommendation": "Review failed trades and tighten the quality gate before expanding.",
                "recommendation_zh": "先复盘失败样本并收紧质量门槛，再扩大使用。",
                "diagnosis_reasons": "High-probability samples exist, but the evidence is not strong enough yet.",
                "diagnosis_reasons_zh": "已经有高概率样本，但证据强度还不够。",
            }
        return {
            "optimization_priority": 3,
            "diagnostic_level": "gate_not_working_well",
            "diagnostic_level_zh": "门槛效果偏弱",
            "recommendation": "Tighten or redesign high-probability gates for this universe.",
            "recommendation_zh": "需要收紧或重做这个股票池的高概率筛选门槛。",
            "diagnosis_reasons": "High-probability samples exist, but average return is not positive.",
            "diagnosis_reasons_zh": "有高概率样本，但平均收益不是正数。",
        }

    if near_watchlist_sample + early_watchlist_sample >= 5 and _is_number_at_least(
        all_avg_return,
        0.0,
    ):
        if high_probability_sample > 0:
            diagnosis_reasons = (
                f"Only {high_probability_sample} high-probability samples so far; "
                "watchlist samples have positive average return."
            )
            diagnosis_reasons_zh = (
                f"目前只有 {high_probability_sample} 个高概率样本，样本仍然太少；"
                "观察名单样本平均收益为正。"
            )
        else:
            diagnosis_reasons = (
                "No strict high-probability samples, but watchlist samples have positive average return."
            )
            diagnosis_reasons_zh = "没有严格高概率样本，但观察名单样本平均收益为正。"
        return {
            "optimization_priority": 4,
            "diagnostic_level": "thresholds_too_strict",
            "diagnostic_level_zh": "阈值可能过严",
            "recommendation": "Study near/early watchlist cases; current high-probability gate may be too strict.",
            "recommendation_zh": "重点看接近机会和早期观察样本，当前高概率门槛可能过严。",
            "diagnosis_reasons": diagnosis_reasons,
            "diagnosis_reasons_zh": diagnosis_reasons_zh,
        }

    if _is_number_at_least(all_win_rate, 0.52) and _is_number_at_least(all_avg_return, 0.0):
        if high_probability_sample > 0:
            diagnosis_reasons = (
                f"Only {high_probability_sample} high-probability samples so far; "
                "more evidence is needed before prioritizing this universe."
            )
            diagnosis_reasons_zh = (
                f"目前只有 {high_probability_sample} 个高概率样本，"
                "需要更多证据后再优先优化这个股票池。"
            )
        else:
            diagnosis_reasons = (
                "Overall validation is positive, but strict high-probability evidence is missing."
            )
            diagnosis_reasons_zh = "整体验证为正，但缺少严格高概率样本证据。"
        return {
            "optimization_priority": 5,
            "diagnostic_level": "research_candidate",
            "diagnostic_level_zh": "可研究候选",
            "recommendation": "Keep this universe in research, but do not prioritize it over stronger universes.",
            "recommendation_zh": "可以继续研究，但优先级低于证据更强的股票池。",
            "diagnosis_reasons": diagnosis_reasons,
            "diagnosis_reasons_zh": diagnosis_reasons_zh,
        }

    return {
        "optimization_priority": 9,
        "diagnostic_level": "deprioritize",
        "diagnostic_level_zh": "降低优先级",
        "recommendation": (
            "Do not optimize this universe first; improve data, features, "
            "or universe construction later."
        ),
        "recommendation_zh": "不要优先优化这个股票池，后续再改数据、特征或股票池构成。",
        "diagnosis_reasons": "Validation did not show enough positive evidence.",
        "diagnosis_reasons_zh": "验证结果没有显示足够正面证据。",
    }


def _bucket_metrics(events: pd.DataFrame, bucket: str | None) -> dict[str, object]:
    if events.empty:
        return {
            "sample_count": 0,
            "win_rate_20d": np.nan,
            "avg_return_20d": np.nan,
            "avg_max_drawdown": np.nan,
        }
    subset = events if bucket is None else events[events["validation_bucket"] == bucket]
    returns = (
        subset["forward_return_20d"].dropna()
        if "forward_return_20d" in subset
        else pd.Series(dtype=float)
    )
    drawdowns = (
        subset["max_drawdown_after_signal"].dropna()
        if "max_drawdown_after_signal" in subset
        else pd.Series(dtype=float)
    )
    return {
        "sample_count": int(len(returns)),
        "win_rate_20d": float((returns > 0).mean()) if len(returns) else np.nan,
        "avg_return_20d": float(returns.mean()) if len(returns) else np.nan,
        "avg_max_drawdown": float(drawdowns.mean()) if len(drawdowns) else np.nan,
    }


def _format_summary_for_report(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    columns = [
        "universe",
        "status",
        "optimization_priority",
        "diagnostic_level_zh",
        "recommendation_zh",
        "top_quality_gate_failures_zh",
        "top_watchlist_missing_items_zh",
        "suggested_threshold_changes_zh",
        "suggestion_rationale_zh",
        "ticker_count",
        "event_count",
        "all_sample_count",
        "all_20d_win_rate",
        "all_20d_avg_return",
        "high_probability_sample_count",
        "high_probability_20d_win_rate",
        "high_probability_20d_avg_return",
        "diagnosis_reasons_zh",
        "output_dir",
        "error",
    ]
    frame = summary[[column for column in columns if column in summary.columns]].copy()
    for column in frame.columns:
        if column.endswith(("win_rate", "avg_return", "drawdown")) or column.endswith(
            ("win_rate_20d", "avg_return_20d")
        ):
            frame[column] = frame[column].map(_format_percent)
    return frame


def _format_ranking_for_report(ranking: pd.DataFrame) -> pd.DataFrame:
    if ranking.empty:
        return ranking
    columns = [
        "rank",
        "universe",
        "decision_zh",
        "ranking_score",
        "event_count",
        "high_probability_sample_count",
        "all_20d_win_rate",
        "all_20d_avg_return",
        "high_probability_20d_win_rate",
        "high_probability_20d_avg_return",
        "why_ranked_here_zh",
    ]
    frame = ranking[[column for column in columns if column in ranking.columns]].copy()
    for column in [
        "all_20d_win_rate",
        "all_20d_avg_return",
        "high_probability_20d_win_rate",
        "high_probability_20d_avg_return",
    ]:
        if column in frame.columns:
            frame[column] = frame[column].map(_format_percent)
    return frame


def _format_coverage_for_report(coverage_plan: pd.DataFrame) -> pd.DataFrame:
    if coverage_plan.empty:
        return coverage_plan
    columns = [
        "universe",
        "mapped_profile",
        "ticker_count",
        "target_event_count",
        "target_high_probability_event_count",
        "standard_command",
        "deep_command",
        "config_apply_gate_zh",
        "coverage_note_zh",
    ]
    return coverage_plan[[column for column in columns if column in coverage_plan.columns]].copy()


def _diagnostic_highlights(summary: pd.DataFrame) -> list[str]:
    if summary.empty:
        return ["No universe diagnostics are available. / 暂无股票池诊断。"]
    ok = summary[summary["status"] == "ok"] if "status" in summary.columns else summary
    if ok.empty:
        return ["All universe validations failed. / 所有股票池验证都失败。"]

    first = ok.sort_values("optimization_priority", ascending=True).iloc[0]
    lines = [
        "## Primary Diagnostic / 主要诊断",
        "",
        f"- Best next research target / 下一步最值得研究: `{first['universe']}`",
        f"- Diagnostic level / 诊断等级: `{first['diagnostic_level']}` / `{first['diagnostic_level_zh']}`",
        f"- Recommendation / 建议: {first['recommendation']}",
        f"- 中文建议: {first['recommendation_zh']}",
        f"- Reason / 原因: {first['diagnosis_reasons']}",
        f"- 中文原因: {first['diagnosis_reasons_zh']}",
        f"- Top blockers / 主要卡点: {first.get('top_quality_gate_failures', 'N/A')}",
        f"- 主要卡点: {first.get('top_quality_gate_failures_zh', 'N/A')}",
        f"- Suggested threshold changes / 建议阈值调整: {first.get('suggested_threshold_changes', 'N/A')}",
        f"- 建议阈值调整: {first.get('suggested_threshold_changes_zh', 'N/A')}",
    ]
    prioritize = ok[ok["optimization_priority"] <= 3]
    if not prioritize.empty:
        names = ", ".join(str(value) for value in prioritize["universe"].to_list())
        lines.append(f"- Priority universes / 优先股票池: `{names}`")
    return lines


def _format_percent(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(number):
        return "N/A"
    return f"{number:.2%}"


def _top_semicolon_items(events: pd.DataFrame, column: str, limit: int = 3) -> str:
    if events.empty or column not in events.columns:
        return "N/A"
    counts: dict[str, int] = {}
    for value in events[column].dropna():
        normalized_value = str(value).replace("；", ";")
        for item in normalized_value.split(";"):
            clean = item.strip(" ；")
            if not clean or clean.lower() in {
                "none",
                "nan",
                "all strict quality gates passed",
            } or clean in {"无", "所有严格质量门槛通过"}:
                continue
            counts[clean] = counts.get(clean, 0) + 1
    if not counts:
        return "N/A"
    top_items = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]
    return "; ".join(f"{item} ({count})" for item, count in top_items)


def _suggest_threshold_changes(
    universe_name: str,
    events: pd.DataFrame,
    screening_config: ScreeningConfig,
    all_metrics: dict[str, object],
) -> dict[str, object]:
    if events.empty:
        return _empty_threshold_suggestion("No validation events available.", "没有可用验证样本。")

    all_win_rate = _safe_float(all_metrics.get("win_rate_20d"))
    all_avg_return = _safe_float(all_metrics.get("avg_return_20d"))
    if not (_is_number_at_least(all_win_rate, 0.52) and _is_number_at_least(all_avg_return, 0.0)):
        return _empty_threshold_suggestion(
            "No relaxation suggested because overall validation is not positive.",
            "整体验证结果不够正面，暂不建议放宽阈值。",
        )

    profile_name = UNIVERSE_PROFILE_MAP.get(universe_name)
    thresholds = _thresholds_for_profile(screening_config, profile_name)
    failure_counts = _failure_counts(events, "quality_gate_fail_reasons")
    changes: list[str] = []
    changes_zh: list[str] = []
    rationale: list[str] = []
    rationale_zh: list[str] = []

    def add_change(
        key: str,
        current: float | int,
        suggested: float | int,
        reason: str,
        reason_zh: str,
        formatter: Callable[[float | int], str] = _format_threshold_number,
    ) -> None:
        if len(changes) >= 3:
            return
        if suggested == current:
            return
        changes.append(f"{key}: {formatter(current)} -> {formatter(suggested)}")
        changes_zh.append(f"{key}: {formatter(current)} -> {formatter(suggested)}")
        rationale.append(reason)
        rationale_zh.append(reason_zh)

    if _count_contains(failure_counts, "signal score too low") >= 5:
        add_change(
            "signal_score_min",
            thresholds.signal_score_min,
            max(55.0, thresholds.signal_score_min - 3.0),
            "Signal score is the most common blocker while overall returns are positive.",
            "信号分数是最常见卡点，但整体验证收益为正。",
        )
    if _count_contains(failure_counts, "confidence too low") >= 5:
        add_change(
            "confidence_min",
            thresholds.confidence_min,
            max(58.0, thresholds.confidence_min - 3.0),
            "Confidence threshold blocks many otherwise positive validation events.",
            "置信度门槛挡住了较多整体表现为正的验证样本。",
        )
    if _count_contains(failure_counts, "backtest sample too small") >= 5:
        add_change(
            "backtest_sample_min",
            thresholds.backtest_sample_min,
            max(6, thresholds.backtest_sample_min - 2),
            "Entry backtest sample requirement is too strict for this universe and validation cadence.",
            "买点回测样本数要求对该股票池和验证频率偏严。",
        )
    if _count_contains(failure_counts, "backtest win rate too low") >= 5:
        add_change(
            "backtest_win_rate_min",
            thresholds.backtest_win_rate_min,
            max(0.50, thresholds.backtest_win_rate_min - 0.02),
            "Backtest win-rate gate blocks many candidates; reduce only slightly.",
            "买点回测胜率门槛挡住较多候选，只建议小幅放宽。",
            formatter=_format_threshold_decimal,
        )
    if _count_contains(failure_counts, "sector context weak") >= 5:
        add_change(
            "medium_long_sector_score_min",
            thresholds.medium_long_sector_score_min,
            max(40.0, thresholds.medium_long_sector_score_min - 3.0),
            "Sector context blocks medium/long signals despite positive universe validation.",
            "板块环境挡住中长期信号，但该股票池整体验证为正。",
        )
    if _count_contains(failure_counts, "relative strength weak") >= 5:
        add_change(
            "relative_strength_min",
            thresholds.relative_strength_min,
            max(40.0, thresholds.relative_strength_min - 3.0),
            "Relative strength threshold is frequently blocking candidates.",
            "相对强弱门槛频繁挡住候选。",
        )

    if not changes:
        return _empty_threshold_suggestion(
            "No threshold change met the minimum blocker count.",
            "没有任何阈值调整达到最低卡点次数要求。",
        )

    if _count_contains(failure_counts, "backtest average return not positive") >= 5:
        rationale.append("Do not lower average-return gate; improve entry logic instead.")
        rationale_zh.append("不建议降低平均收益门槛，应优先改进入场逻辑。")

    return {
        "suggested_threshold_changes": "; ".join(changes),
        "suggested_threshold_changes_zh": "；".join(changes_zh),
        "suggestion_rationale": "; ".join(rationale),
        "suggestion_rationale_zh": "；".join(rationale_zh),
    }


def _empty_threshold_suggestion(reason: str, reason_zh: str) -> dict[str, object]:
    return {
        "suggested_threshold_changes": "none",
        "suggested_threshold_changes_zh": "无",
        "suggestion_rationale": reason,
        "suggestion_rationale_zh": reason_zh,
    }


def _thresholds_for_profile(
    screening_config: ScreeningConfig,
    profile_name: str | None,
) -> ScreeningThresholds:
    if profile_name:
        for profile in screening_config.profiles:
            if profile.name == profile_name:
                return profile.thresholds
    return screening_config.default_thresholds


def _failure_counts(events: pd.DataFrame, column: str) -> dict[str, int]:
    if events.empty or column not in events.columns:
        return {}
    counts: dict[str, int] = {}
    for value in events[column].dropna():
        normalized_value = str(value).replace("；", ";")
        for item in normalized_value.split(";"):
            clean = item.strip(" ；")
            if not clean or clean.lower() in {
                "none",
                "nan",
                "all strict quality gates passed",
            }:
                continue
            counts[clean] = counts.get(clean, 0) + 1
    return counts


def _count_contains(counts: dict[str, int], needle: str) -> int:
    needle_lower = needle.lower()
    return sum(count for item, count in counts.items() if needle_lower in item.lower())


def _format_threshold_number(value: float | int) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.2f}"


def _format_threshold_decimal(value: float | int) -> str:
    return f"{float(value):.2f}"


def _safe_float(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def _safe_int(value: object) -> int:
    try:
        if pd.isna(value):
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def _is_number_at_least(value: float, threshold: float) -> bool:
    return bool(np.isfinite(value) and value >= threshold)


def _clamp_number(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["No rows / 暂无数据"]
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = [str(row[column]) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return lines
