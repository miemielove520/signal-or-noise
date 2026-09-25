"""Benchmark-aware policies, threshold tightening, sensitivity grids and the minimum-sample guard."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..screening_config import ScreeningConfig

from ._common import (
    _first_text,
    _is_finite,
)
from .config import (
    DEFAULT_HIGH_PROBABILITY_THRESHOLDS,
    DEFAULT_MARKET_THRESHOLDS,
    DEFAULT_RELATIVE_STRENGTH_THRESHOLDS,
    DEFAULT_SIGNAL_THRESHOLDS,
)


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
