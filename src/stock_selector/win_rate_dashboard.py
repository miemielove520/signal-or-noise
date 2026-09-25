from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .json_io import dataframe_records


DEFAULT_DASHBOARD_WINDOWS = (5, 20, 60)


def build_win_rate_dashboard(
    events: pd.DataFrame,
    forward_windows: Iterable[int] = DEFAULT_DASHBOARD_WINDOWS,
) -> dict[str, pd.DataFrame]:
    windows = tuple(int(window) for window in forward_windows)
    return {
        "overall": _summarize_groups(events, [], windows),
        "by_horizon": _summarize_groups(
            events,
            ["horizon", "horizon_zh_label"],
            windows,
        ),
        "by_entry_type": _summarize_groups(
            events,
            ["screening_backtest_entry_type"],
            windows,
        ),
        "by_profile": _summarize_groups(
            events,
            ["screening_profile", "screening_profile_zh"],
            windows,
        ),
        "by_quality_gate": _summarize_groups(
            events,
            [_quality_gate_column(events)],
            windows,
        ),
    }


def win_rate_dashboard_payload(dashboard: dict[str, pd.DataFrame]) -> dict[str, list[dict[str, object]]]:
    return {name: dataframe_records(frame) for name, frame in dashboard.items()}


def build_profile_health_dashboard(
    events: pd.DataFrame,
    target_window: int = 20,
    min_samples: int = 30,
    min_win_rate: float = 0.55,
    min_avg_return: float = 0.0,
    max_avg_drawdown: float = -0.12,
) -> pd.DataFrame:
    columns = [
        "screening_profile",
        "screening_profile_zh",
        "sample_count",
        "high_probability_sample_count",
        "quality_gate_pass_rate",
        "win_rate",
        "avg_return",
        "median_return",
        "worst_return",
        "avg_max_drawdown_after_signal",
        "sample_status",
        "sample_status_zh",
        "profile_health_score",
        "profile_health_level",
        "profile_health_level_zh",
        "profile_action",
        "profile_action_zh",
        "profile_reason",
        "profile_reason_zh",
    ]
    return_column = f"forward_return_{target_window}d"
    required = {"screening_profile", return_column}
    if events.empty or not required.issubset(events.columns):
        return pd.DataFrame(columns=columns)

    rows = []
    for profile_name, group in events.groupby("screening_profile", dropna=False):
        returns = pd.to_numeric(group[return_column], errors="coerce").dropna()
        sample_count = int(len(returns))
        win_rate = float((returns > 0).mean()) if sample_count else np.nan
        avg_return = float(returns.mean()) if sample_count else np.nan
        median_return = float(returns.median()) if sample_count else np.nan
        worst_return = float(returns.min()) if sample_count else np.nan
        high_probability_sample_count = (
            int((group["validation_bucket"].astype(str) == "high_probability").sum())
            if "validation_bucket" in group.columns
            else 0
        )
        quality_gate_pass_rate = (
            float(group["quality_gate_passed"].astype(bool).mean())
            if "quality_gate_passed" in group.columns and len(group)
            else np.nan
        )
        avg_drawdown = (
            float(pd.to_numeric(group["max_drawdown_after_signal"], errors="coerce").mean())
            if "max_drawdown_after_signal" in group.columns
            else np.nan
        )
        sample_status, sample_status_zh = _profile_sample_status(sample_count, min_samples)
        score = _profile_health_score(
            sample_count=sample_count,
            min_samples=min_samples,
            win_rate=win_rate,
            avg_return=avg_return,
            avg_drawdown=avg_drawdown,
            quality_gate_pass_rate=quality_gate_pass_rate,
            high_probability_sample_count=high_probability_sample_count,
        )
        level, level_zh = _profile_health_level(score)
        action, action_zh, reason, reason_zh = _profile_health_action(
            sample_count=sample_count,
            min_samples=min_samples,
            win_rate=win_rate,
            min_win_rate=min_win_rate,
            avg_return=avg_return,
            min_avg_return=min_avg_return,
            avg_drawdown=avg_drawdown,
            max_avg_drawdown=max_avg_drawdown,
            high_probability_sample_count=high_probability_sample_count,
        )
        rows.append(
            {
                "screening_profile": str(profile_name),
                "screening_profile_zh": _first_text(
                    group,
                    "screening_profile_zh",
                    str(profile_name),
                ),
                "sample_count": sample_count,
                "high_probability_sample_count": high_probability_sample_count,
                "quality_gate_pass_rate": quality_gate_pass_rate,
                "win_rate": win_rate,
                "avg_return": avg_return,
                "median_return": median_return,
                "worst_return": worst_return,
                "avg_max_drawdown_after_signal": avg_drawdown,
                "sample_status": sample_status,
                "sample_status_zh": sample_status_zh,
                "profile_health_score": score,
                "profile_health_level": level,
                "profile_health_level_zh": level_zh,
                "profile_action": action,
                "profile_action_zh": action_zh,
                "profile_reason": reason,
                "profile_reason_zh": reason_zh,
            }
        )
    frame = pd.DataFrame(rows, columns=columns)
    return frame.sort_values(
        ["profile_health_score", "sample_count"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)


def render_profile_health_dashboard(profile_health: pd.DataFrame) -> str:
    lines = [
        "# Profile Health Dashboard / 分类规则健康面板",
        "",
        (
            "This dashboard checks which screening profiles have enough evidence, "
            "acceptable win rate, positive return, and controlled drawdown."
        ),
        "这个面板检查不同分类规则是否有足够样本、可接受胜率、正收益和可控回撤。",
        "",
    ]
    if profile_health.empty:
        lines.append("No rows / 暂无数据。")
    else:
        lines.extend(_markdown_table(_format_dashboard_frame(profile_health)))
    return "\n".join(lines).rstrip() + "\n"


def build_profile_action_recommendations(profile_health: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "screening_profile",
        "screening_profile_zh",
        "profile_action",
        "recommendation_action",
        "recommendation_action_zh",
        "priority",
        "priority_zh",
        "threshold_attrs",
        "threshold_attrs_zh",
        "suggested_next_step",
        "suggested_next_step_zh",
        "suggested_command_hint",
        "suggested_command_hint_zh",
        "recommendation_reason",
        "recommendation_reason_zh",
    ]
    if profile_health.empty:
        return pd.DataFrame(columns=columns)

    rows = [_profile_recommendation_row(row) for row in profile_health.itertuples(index=False)]
    frame = pd.DataFrame(rows, columns=columns)
    priority_rank = {"high": 0, "medium": 1, "low": 2}
    frame["_priority_rank"] = frame["priority"].map(priority_rank).fillna(9)
    return (
        frame.sort_values(["_priority_rank", "screening_profile"])
        .drop(columns=["_priority_rank"])
        .reset_index(drop=True)
    )


def render_profile_action_recommendations(recommendations: pd.DataFrame) -> str:
    lines = [
        "# Profile Action Recommendations / 分类规则行动建议",
        "",
        (
            "This section translates profile health into concrete next steps for each "
            "screening profile."
        ),
        "本区块把分类规则健康状态转成每个规则组的具体下一步。",
        "",
    ]
    if recommendations.empty:
        lines.append("No rows / 暂无数据。")
    else:
        lines.extend(_markdown_table(_format_dashboard_frame(recommendations)))
    return "\n".join(lines).rstrip() + "\n"


def build_profile_blocker_dashboard(events: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    columns = [
        "screening_profile",
        "screening_profile_zh",
        "blocker_rank",
        "blocker",
        "blocker_zh",
        "occurrence_count",
        "profile_event_count",
        "blocker_rate",
        "blocker_category",
        "blocker_category_zh",
        "recommended_fix",
        "recommended_fix_zh",
    ]
    required = {"screening_profile", "quality_gate_fail_reasons"}
    if events.empty or not required.issubset(events.columns):
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for profile_name, group in events.groupby("screening_profile", dropna=False):
        profile_key = str(profile_name)
        profile_zh = _first_text(group, "screening_profile_zh", profile_key)
        profile_event_count = int(len(group))
        counts: dict[tuple[str, str], int] = {}
        for _, event in group.iterrows():
            reasons = _split_reasons(event.get("quality_gate_fail_reasons"))
            reasons_zh = _split_reasons(event.get("quality_gate_fail_reasons_zh"))
            for index, reason in enumerate(reasons):
                if _is_non_blocking_reason(reason):
                    continue
                reason_zh = reasons_zh[index] if index < len(reasons_zh) else reason
                key = (reason, reason_zh)
                counts[key] = counts.get(key, 0) + 1
        sorted_counts = sorted(counts.items(), key=lambda item: (-item[1], item[0][0]))
        for rank, ((reason, reason_zh), count) in enumerate(sorted_counts[:top_n], start=1):
            category, category_zh, fix, fix_zh = _blocker_fix(reason)
            rows.append(
                {
                    "screening_profile": profile_key,
                    "screening_profile_zh": profile_zh,
                    "blocker_rank": rank,
                    "blocker": reason,
                    "blocker_zh": reason_zh,
                    "occurrence_count": int(count),
                    "profile_event_count": profile_event_count,
                    "blocker_rate": float(count / profile_event_count)
                    if profile_event_count
                    else np.nan,
                    "blocker_category": category,
                    "blocker_category_zh": category_zh,
                    "recommended_fix": fix,
                    "recommended_fix_zh": fix_zh,
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["screening_profile", "blocker_rank"],
    ).reset_index(drop=True)


def render_profile_blocker_dashboard(blockers: pd.DataFrame) -> str:
    lines = [
        "# Profile Blocker Dashboard / 分类规则卡点面板",
        "",
        (
            "This dashboard summarizes the most common quality-gate blockers for each "
            "screening profile."
        ),
        "这个面板汇总每个分类规则最常见的质量门槛卡点。",
        "",
    ]
    if blockers.empty:
        lines.append("No blockers / 暂无卡点。")
    else:
        lines.extend(_markdown_table(_format_dashboard_frame(blockers)))
    return "\n".join(lines).rstrip() + "\n"


def build_historical_win_rate_gate(
    events: pd.DataFrame,
    target_window: int = 20,
    min_samples: int = 30,
    min_win_rate: float = 0.55,
    min_avg_return: float = 0.0,
    max_worst_return: float = -0.15,
) -> pd.DataFrame:
    columns = [
        "scope",
        "scope_zh",
        "group_value",
        "group_value_zh",
        "forward_window_days",
        "sample_count",
        "win_rate",
        "avg_return",
        "median_return",
        "worst_return",
        "best_return",
        "sample_quality",
        "sample_quality_zh",
        "deployment_gate_action",
        "deployment_gate_action_zh",
        "deployment_gate_reason",
        "deployment_gate_reason_zh",
    ]
    if events.empty:
        return pd.DataFrame(columns=columns)

    rows = [
        _gate_row(
            events,
            scope="overall",
            scope_zh="整体",
            group_value="all_events",
            group_value_zh="全部信号",
            target_window=target_window,
            min_samples=min_samples,
            min_win_rate=min_win_rate,
            min_avg_return=min_avg_return,
            max_worst_return=max_worst_return,
        )
    ]
    if "validation_bucket" in events.columns:
        high_probability = events[events["validation_bucket"].astype(str) == "high_probability"]
        rows.append(
            _gate_row(
                high_probability,
                scope="validation_bucket",
                scope_zh="验证分类",
                group_value="high_probability",
                group_value_zh="高概率信号",
                target_window=target_window,
                min_samples=min_samples,
                min_win_rate=min_win_rate,
                min_avg_return=min_avg_return,
                max_worst_return=max_worst_return,
            )
        )
    if "quality_gate_passed" in events.columns:
        passed = events[events["quality_gate_passed"].astype(bool)]
        rows.append(
            _gate_row(
                passed,
                scope="quality_gate",
                scope_zh="质量门控",
                group_value="passed",
                group_value_zh="通过质量门控",
                target_window=target_window,
                min_samples=min_samples,
                min_win_rate=min_win_rate,
                min_avg_return=min_avg_return,
                max_worst_return=max_worst_return,
            )
        )
    rows.extend(
        _gate_rows_for_group(
            events,
            group_column="screening_profile",
            group_zh_column="screening_profile_zh",
            scope="screening_profile",
            scope_zh="筛选规则",
            target_window=target_window,
            min_samples=min_samples,
            min_win_rate=min_win_rate,
            min_avg_return=min_avg_return,
            max_worst_return=max_worst_return,
        )
    )
    rows.extend(
        _gate_rows_for_group(
            events,
            group_column="horizon",
            group_zh_column="horizon_zh_label",
            scope="horizon",
            scope_zh="周期",
            target_window=target_window,
            min_samples=min_samples,
            min_win_rate=min_win_rate,
            min_avg_return=min_avg_return,
            max_worst_return=max_worst_return,
        )
    )
    rows.extend(
        _gate_rows_for_group(
            events,
            group_column="screening_backtest_entry_type",
            group_zh_column=None,
            scope="entry_type",
            scope_zh="买点类型",
            target_window=target_window,
            min_samples=min_samples,
            min_win_rate=min_win_rate,
            min_avg_return=min_avg_return,
            max_worst_return=max_worst_return,
        )
    )
    frame = pd.DataFrame(rows, columns=columns)
    return frame.sort_values(["scope", "group_value"]).reset_index(drop=True)


def render_historical_win_rate_gate(gate: pd.DataFrame) -> str:
    lines = [
        "# Historical Win-Rate Gate / 历史胜率部署门槛",
        "",
        (
            "This section decides whether the high-probability filter has enough "
            "historical evidence to be trusted, tightened, or paused."
        ),
        "本区块判断高概率筛选器是否有足够历史证据继续使用，还是应该收紧或暂停。",
        "",
    ]
    if gate.empty:
        lines.append("No rows / 暂无数据。")
    else:
        lines.extend(_markdown_table(_format_dashboard_frame(gate)))
    return "\n".join(lines).rstrip() + "\n"


def _profile_sample_status(sample_count: int, min_samples: int) -> tuple[str, str]:
    if sample_count <= 0:
        return "no_samples", "没有样本"
    if sample_count < min_samples:
        return "thin_samples", "样本偏少"
    return "enough_samples", "样本充足"


def _profile_recommendation_row(row: object) -> dict[str, object]:
    action = str(getattr(row, "profile_action", ""))
    profile = str(getattr(row, "screening_profile", ""))
    profile_zh = str(getattr(row, "screening_profile_zh", profile))
    base = {
        "screening_profile": profile,
        "screening_profile_zh": profile_zh,
        "profile_action": action,
        "recommendation_reason": str(getattr(row, "profile_reason", "")),
        "recommendation_reason_zh": str(getattr(row, "profile_reason_zh", "")),
    }

    if action == "keep_profile":
        return {
            **base,
            "recommendation_action": "keep_active",
            "recommendation_action_zh": "继续启用",
            "priority": "low",
            "priority_zh": "低",
            "threshold_attrs": "",
            "threshold_attrs_zh": "",
            "suggested_next_step": "Keep this profile active and continue monitoring.",
            "suggested_next_step_zh": "继续启用该规则组，并持续观察表现。",
            "suggested_command_hint": "",
            "suggested_command_hint_zh": "",
        }
    if action == "expand_sample":
        return {
            **base,
            "recommendation_action": "expand_validation_sample",
            "recommendation_action_zh": "扩大验证样本",
            "priority": "medium",
            "priority_zh": "中",
            "threshold_attrs": "",
            "threshold_attrs_zh": "",
            "suggested_next_step": "Add more tickers or run a deeper validation before tuning thresholds.",
            "suggested_next_step_zh": "先增加股票或运行更深度验证，再考虑调阈值。",
            "suggested_command_hint": "python3 validate.py --preset deep",
            "suggested_command_hint_zh": "建议先运行：python3 validate.py --preset deep",
        }
    if action == "tighten_profile":
        return {
            **base,
            "recommendation_action": "tighten_signal_quality",
            "recommendation_action_zh": "收紧信号质量",
            "priority": "high",
            "priority_zh": "高",
            "threshold_attrs": "signal_score_min; relative_strength_min; backtest_win_rate_min",
            "threshold_attrs_zh": "信号分下限；相对强弱下限；买点回测胜率下限",
            "suggested_next_step": "Raise signal, relative-strength, and entry-backtest thresholds for this profile.",
            "suggested_next_step_zh": "提高该规则组的信号分、相对强弱和买点回测胜率要求。",
            "suggested_command_hint": "Review suggested_screening.toml before replacing configs/screening.toml.",
            "suggested_command_hint_zh": "先审核 suggested_screening.toml，不要直接替换正式配置。",
        }
    if action == "tighten_risk":
        return {
            **base,
            "recommendation_action": "tighten_risk_controls",
            "recommendation_action_zh": "收紧风险控制",
            "priority": "high",
            "priority_zh": "高",
            "threshold_attrs": "max_backtest_stop_hit_rate; backtest_trust_score_min",
            "threshold_attrs_zh": "最大止损命中率；回测可信度下限",
            "suggested_next_step": "Lower stop-hit tolerance and require stronger backtest trust.",
            "suggested_next_step_zh": "降低止损命中率容忍度，并提高回测可信度要求。",
            "suggested_command_hint": "Review historical_threshold_recommendations.md.",
            "suggested_command_hint_zh": "查看 historical_threshold_recommendations.md。",
        }
    if action == "watch_only":
        return {
            **base,
            "recommendation_action": "keep_watch_only",
            "recommendation_action_zh": "仅观察",
            "priority": "medium",
            "priority_zh": "中",
            "threshold_attrs": "",
            "threshold_attrs_zh": "",
            "suggested_next_step": "Do not promote this profile to high-probability use until it produces enough high-probability samples.",
            "suggested_next_step_zh": "在产生足够高概率样本前，不把该规则组提升为高概率使用。",
            "suggested_command_hint": "Track profile_health_dashboard.md after each standard validation.",
            "suggested_command_hint_zh": "每次 standard 验证后查看 profile_health_dashboard.md。",
        }
    return {
        **base,
        "recommendation_action": "manual_review",
        "recommendation_action_zh": "人工复核",
        "priority": "medium",
        "priority_zh": "中",
        "threshold_attrs": "",
        "threshold_attrs_zh": "",
        "suggested_next_step": "Review this profile manually.",
        "suggested_next_step_zh": "人工复核该规则组。",
        "suggested_command_hint": "",
        "suggested_command_hint_zh": "",
    }


def _split_reasons(value: object) -> list[str]:
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    text = str(value).replace("；", ";")
    return [part.strip() for part in text.split(";") if part.strip()]


def _is_non_blocking_reason(reason: str) -> bool:
    text = reason.lower().strip()
    return (
        text == ""
        or "all strict quality gates passed" in text
        or text in {"none", "nan"}
    )


def _blocker_fix(reason: str) -> tuple[str, str, str, str]:
    text = reason.lower()
    if "sample too small" in text or "样本不足" in text:
        return (
            "sample",
            "样本",
            "expand_validation_sample",
            "扩大验证样本，不要根据薄样本调阈值。",
        )
    if "win rate too low" in text or "average return" in text or "stop-hit" in text:
        return (
            "entry_backtest",
            "买点回测",
            "tighten_entry_backtest",
            "收紧买点回测胜率、平均收益和止损命中率要求。",
        )
    if "signal score" in text:
        return (
            "signal",
            "信号",
            "tighten_signal_threshold",
            "提高信号分下限，减少弱信号进入候选。",
        )
    if "relative strength" in text:
        return (
            "relative_strength",
            "相对强弱",
            "tighten_relative_strength",
            "提高相对强弱要求，只保留强于基准的股票。",
        )
    if "market" in text:
        return (
            "market",
            "大盘",
            "tighten_market_filter",
            "提高大盘环境要求，大盘弱时减少新机会。",
        )
    if "data quality" in text or "liquidity" in text:
        return (
            "data_liquidity",
            "数据/流动性",
            "repair_data_or_liquidity",
            "修复数据质量或过滤流动性不足的股票。",
        )
    if "trust" in text or "recent backtest" in text or "decay" in text:
        return (
            "backtest_reliability",
            "回测可靠性",
            "improve_backtest_reliability",
            "提高回测可信度、近期稳定性和衰退检查要求。",
        )
    if (
        "event" in text
        or "sentiment" in text
        or "analyst" in text
        or "valuation" in text
        or "risk high" in text
    ):
        return (
            "risk_context",
            "风险环境",
            "keep_risk_blocker",
            "保留风险拦截，避免事件、情绪、估值或分析师风险过高时入场。",
        )
    if "entry trigger" in text or "risk-reward" in text:
        return (
            "trade_plan",
            "交易计划",
            "improve_trade_plan",
            "改善买点可执行性和盈亏比要求。",
        )
    return (
        "manual_review",
        "人工复核",
        "manual_review",
        "人工复核该卡点，再决定是否调整规则。",
    )


def _profile_health_score(
    sample_count: int,
    min_samples: int,
    win_rate: float,
    avg_return: float,
    avg_drawdown: float,
    quality_gate_pass_rate: float,
    high_probability_sample_count: int,
) -> float:
    sample_component = min(sample_count / max(min_samples, 1), 1.5) / 1.5 * 25.0
    win_component = _bounded_component(win_rate, low=0.45, high=0.65, weight=25.0)
    return_component = _bounded_component(avg_return, low=-0.02, high=0.05, weight=20.0)
    drawdown_component = _bounded_component(avg_drawdown, low=-0.25, high=-0.03, weight=15.0)
    gate_component = _bounded_component(quality_gate_pass_rate, low=0.0, high=0.25, weight=10.0)
    high_probability_component = min(high_probability_sample_count / 10.0, 1.0) * 5.0
    return float(
        max(
            0.0,
            min(
                100.0,
                sample_component
                + win_component
                + return_component
                + drawdown_component
                + gate_component
                + high_probability_component,
            ),
        )
    )


def _profile_health_level(score: float) -> tuple[str, str]:
    if score >= 75:
        return "healthy", "健康"
    if score >= 55:
        return "watch", "观察"
    if score >= 35:
        return "weak", "偏弱"
    return "poor", "较差"


def _profile_health_action(
    sample_count: int,
    min_samples: int,
    win_rate: float,
    min_win_rate: float,
    avg_return: float,
    min_avg_return: float,
    avg_drawdown: float,
    max_avg_drawdown: float,
    high_probability_sample_count: int,
) -> tuple[str, str, str, str]:
    if sample_count <= 0:
        return (
            "expand_sample",
            "扩大样本",
            "No validation samples were generated for this profile.",
            "该分类规则没有生成验证样本。",
        )
    if sample_count < min_samples:
        return (
            "expand_sample",
            "扩大样本",
            f"Sample count {sample_count} is below required {min_samples}.",
            f"样本数{sample_count}低于要求{min_samples}。",
        )
    if not _is_finite(win_rate) or win_rate < min_win_rate:
        return (
            "tighten_profile",
            "收紧规则",
            f"Win rate is below {min_win_rate:.0%}.",
            f"胜率低于{min_win_rate:.0%}。",
        )
    if not _is_finite(avg_return) or avg_return <= min_avg_return:
        return (
            "tighten_profile",
            "收紧规则",
            "Average return is not positive enough.",
            "平均收益不够理想。",
        )
    if _is_finite(avg_drawdown) and avg_drawdown < max_avg_drawdown:
        return (
            "tighten_risk",
            "收紧风险",
            f"Average post-signal drawdown is worse than {max_avg_drawdown:.0%}.",
            f"信号后平均回撤差于{max_avg_drawdown:.0%}。",
        )
    if high_probability_sample_count <= 0:
        return (
            "watch_only",
            "仅观察",
            "Profile has acceptable overall results but no high-probability samples.",
            "整体表现可接受，但没有高概率样本。",
        )
    return (
        "keep_profile",
        "保留规则",
        "Profile has acceptable evidence and can remain active.",
        "该分类规则证据可接受，可以继续保留。",
    )


def _bounded_component(value: float, low: float, high: float, weight: float) -> float:
    if not _is_finite(value) or high <= low:
        return 0.0
    ratio = (float(value) - low) / (high - low)
    return max(0.0, min(1.0, ratio)) * weight


def build_historical_threshold_recommendations(
    gate: pd.DataFrame,
    screening_config: object | None = None,
) -> pd.DataFrame:
    columns = [
        "scope",
        "scope_zh",
        "group_value",
        "group_value_zh",
        "deployment_gate_action",
        "recommendation_action",
        "recommendation_action_zh",
        "threshold_attr",
        "threshold_attr_zh",
        "current_threshold",
        "suggested_threshold",
        "change_direction",
        "priority",
        "priority_zh",
        "allow_auto_apply",
        "recommendation_reason",
        "recommendation_reason_zh",
        "suggested_command_hint",
        "suggested_command_hint_zh",
    ]
    if gate.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for gate_row in gate.itertuples(index=False):
        rows.extend(_threshold_recommendation_rows(gate_row, screening_config))
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows, columns=columns)
    priority_rank = {"high": 0, "medium": 1, "low": 2}
    frame["_priority_rank"] = frame["priority"].map(priority_rank).fillna(9)
    return (
        frame.sort_values(["_priority_rank", "scope", "group_value", "threshold_attr"])
        .drop(columns=["_priority_rank"])
        .reset_index(drop=True)
    )


def render_historical_threshold_recommendations(recommendations: pd.DataFrame) -> str:
    lines = [
        "# Historical Threshold Recommendations / 历史阈值建议",
        "",
        (
            "These recommendations translate the historical win-rate gate into concrete "
            "next steps. Thin samples lead to validation guidance, not automatic tightening."
        ),
        "这些建议把历史胜率门槛转成具体下一步；样本不足时只建议扩大验证，不自动乱收紧。",
        "",
    ]
    if recommendations.empty:
        lines.append("No rows / 暂无数据。")
    else:
        lines.extend(_markdown_table(_format_dashboard_frame(recommendations)))
    return "\n".join(lines).rstrip() + "\n"


def render_win_rate_dashboard(dashboard: dict[str, pd.DataFrame]) -> str:
    lines = [
        "# Historical Win-Rate Dashboard / 历史胜率统计面板",
        "",
        (
            "This dashboard summarizes past walk-forward validation outcomes. "
            "It is a model-quality diagnostic, not a promise that future signals will win."
        ),
        "这个面板统计过去滚动验证里的真实后验结果，用来检查模型质量，不代表未来一定成功。",
        "",
    ]
    sections = [
        (
            "Overall / 总体",
            "All historical validation events grouped only by forward window.",
            "所有历史验证信号按未来观察窗口统计。",
            dashboard.get("overall", pd.DataFrame()),
        ),
        (
            "By Horizon / 按周期",
            "Shows whether short, medium, or long signals have worked better historically.",
            "查看短期、中期、长期信号过去哪个更有效。",
            dashboard.get("by_horizon", pd.DataFrame()),
        ),
        (
            "By Entry Type / 按买点类型",
            "Compares breakout, pullback, and other entry types using realized forward returns.",
            "比较突破、回调等不同买点类型的历史真实收益。",
            dashboard.get("by_entry_type", pd.DataFrame()),
        ),
        (
            "By Screening Profile / 按筛选规则",
            "Compares the active screening profiles so weak rule sets can be tightened or removed.",
            "比较不同筛选规则，方便之后收紧或删除表现差的规则。",
            dashboard.get("by_profile", pd.DataFrame()),
        ),
        (
            "By Quality Gate / 按质量门控",
            "Checks whether signals that passed the quality gate actually performed better.",
            "检查通过质量门控的信号，历史表现是否真的更好。",
            dashboard.get("by_quality_gate", pd.DataFrame()),
        ),
    ]
    for title, note, note_zh, frame in sections:
        lines.extend(["", f"## {title}", "", note, note_zh, ""])
        lines.extend(_markdown_table(_format_dashboard_frame(frame)))
    return "\n".join(lines).rstrip() + "\n"


def _threshold_recommendation_rows(gate_row: object, screening_config: object | None) -> list[dict[str, object]]:
    action = str(getattr(gate_row, "deployment_gate_action", ""))
    if action == "allow_high_probability_filter":
        return [
            _threshold_recommendation_row(
                gate_row,
                screening_config,
                recommendation_action="keep_current_thresholds",
                recommendation_action_zh="保持当前阈值",
                threshold_attr="",
                threshold_attr_zh="",
                current_threshold=np.nan,
                suggested_threshold=np.nan,
                change_direction="hold",
                priority="low",
                priority_zh="低",
                allow_auto_apply=False,
                reason="historical evidence already meets the deployment gate",
                reason_zh="历史证据已经满足部署门槛",
            )
        ]
    if action in {"pause_until_samples_exist", "collect_more_samples"}:
        return [
            _threshold_recommendation_row(
                gate_row,
                screening_config,
                recommendation_action="expand_validation_coverage",
                recommendation_action_zh="扩大历史验证覆盖",
                threshold_attr="",
                threshold_attr_zh="",
                current_threshold=np.nan,
                suggested_threshold=np.nan,
                change_direction="none",
                priority="medium",
                priority_zh="中",
                allow_auto_apply=False,
                reason="sample size is too small; do not tune thresholds from thin evidence",
                reason_zh="样本太少，不能根据薄样本调阈值",
                command_hint="python3 validate.py <TICKERS> --preset deep --period 10y",
                command_hint_zh="先扩大股票池并使用更长周期验证",
            )
        ]
    if action == "tighten_or_block_high_probability":
        return _tightening_rows(
            gate_row,
            screening_config,
            [
                ("signal_score_min", 2.0, "up", "raise_signal_quality", "提高信号质量门槛"),
                ("relative_strength_min", 5.0, "up", "raise_relative_strength", "提高相对强弱门槛"),
                ("market_score_min", 5.0, "up", "raise_market_support", "提高大盘环境门槛"),
                ("backtest_win_rate_min", 0.02, "up", "raise_backtest_win_rate", "提高买点回测胜率要求"),
                ("high_probability_target_score", 3.0, "up", "raise_high_probability_score", "提高高概率分数要求"),
            ],
            reason="historical win rate is below the required level",
            reason_zh="历史胜率低于要求",
            priority="high",
        )
    if action == "tighten_return_quality":
        return _tightening_rows(
            gate_row,
            screening_config,
            [
                ("high_probability_target_score", 3.0, "up", "raise_high_probability_score", "提高高概率分数要求"),
                ("backtest_win_rate_min", 0.02, "up", "raise_backtest_win_rate", "提高买点回测胜率要求"),
                ("signal_score_min", 2.0, "up", "raise_signal_quality", "提高信号质量门槛"),
            ],
            reason="historical average return is not positive enough",
            reason_zh="历史平均收益不够理想",
            priority="high",
        )
    if action == "tighten_risk_controls":
        return _tightening_rows(
            gate_row,
            screening_config,
            [
                ("max_backtest_stop_hit_rate", 0.05, "down", "lower_stop_hit_tolerance", "降低止损命中率容忍度"),
                ("backtest_trust_score_min", 5.0, "up", "raise_backtest_trust", "提高回测可信度门槛"),
                ("high_probability_target_score", 3.0, "up", "raise_high_probability_score", "提高高概率分数要求"),
            ],
            reason="historical worst return is too large",
            reason_zh="历史最差亏损过大",
            priority="high",
        )
    return []


def _tightening_rows(
    gate_row: object,
    screening_config: object | None,
    specs: list[tuple[str, float, str, str, str]],
    reason: str,
    reason_zh: str,
    priority: str,
) -> list[dict[str, object]]:
    rows = []
    for threshold_attr, amount, direction, action, action_zh in specs:
        current = _current_threshold(screening_config, gate_row, threshold_attr)
        suggested = _suggest_threshold(current, amount, direction)
        rows.append(
            _threshold_recommendation_row(
                gate_row,
                screening_config,
                recommendation_action=action,
                recommendation_action_zh=action_zh,
                threshold_attr=threshold_attr,
                threshold_attr_zh=_threshold_attr_zh(threshold_attr),
                current_threshold=current,
                suggested_threshold=suggested,
                change_direction="tighten",
                priority=priority,
                priority_zh="高" if priority == "high" else "中",
                allow_auto_apply=_can_auto_apply_threshold(gate_row),
                reason=reason,
                reason_zh=reason_zh,
            )
        )
    return rows


def _threshold_recommendation_row(
    gate_row: object,
    screening_config: object | None,
    recommendation_action: str,
    recommendation_action_zh: str,
    threshold_attr: str,
    threshold_attr_zh: str,
    current_threshold: float,
    suggested_threshold: float,
    change_direction: str,
    priority: str,
    priority_zh: str,
    allow_auto_apply: bool,
    reason: str,
    reason_zh: str,
    command_hint: str = "",
    command_hint_zh: str = "",
) -> dict[str, object]:
    return {
        "scope": getattr(gate_row, "scope", ""),
        "scope_zh": getattr(gate_row, "scope_zh", ""),
        "group_value": getattr(gate_row, "group_value", ""),
        "group_value_zh": getattr(gate_row, "group_value_zh", ""),
        "deployment_gate_action": getattr(gate_row, "deployment_gate_action", ""),
        "recommendation_action": recommendation_action,
        "recommendation_action_zh": recommendation_action_zh,
        "threshold_attr": threshold_attr,
        "threshold_attr_zh": threshold_attr_zh,
        "current_threshold": current_threshold,
        "suggested_threshold": suggested_threshold,
        "change_direction": change_direction,
        "priority": priority,
        "priority_zh": priority_zh,
        "allow_auto_apply": bool(allow_auto_apply),
        "recommendation_reason": reason,
        "recommendation_reason_zh": reason_zh,
        "suggested_command_hint": command_hint,
        "suggested_command_hint_zh": command_hint_zh,
    }


def _current_threshold(screening_config: object | None, gate_row: object, attr: str) -> float:
    profile_name = "default"
    if str(getattr(gate_row, "scope", "")) == "screening_profile":
        profile_name = str(getattr(gate_row, "group_value", "default"))
    thresholds = getattr(screening_config, "default_thresholds", None)
    for profile in getattr(screening_config, "profiles", ()) if screening_config is not None else ():
        if getattr(profile, "name", "") == profile_name:
            thresholds = getattr(profile, "thresholds", thresholds)
            break
    if thresholds is None or not hasattr(thresholds, attr):
        defaults = {
            "signal_score_min": 65.0,
            "relative_strength_min": 45.0,
            "market_score_min": 55.0,
            "backtest_win_rate_min": 0.55,
            "high_probability_target_score": 65.0,
            "max_backtest_stop_hit_rate": 0.50,
            "backtest_trust_score_min": 60.0,
        }
        return float(defaults.get(attr, np.nan))
    return float(getattr(thresholds, attr))


def _suggest_threshold(current: float, amount: float, direction: str) -> float:
    if not _is_finite(current):
        return np.nan
    if direction == "down":
        return max(float(current) - float(amount), 0.20)
    if float(current) < 1:
        return min(float(current) + float(amount), 0.80)
    return min(float(current) + float(amount), 95.0)


def _can_auto_apply_threshold(gate_row: object) -> bool:
    return str(getattr(gate_row, "scope", "")) == "screening_profile"


def _threshold_attr_zh(attr: str) -> str:
    return {
        "signal_score_min": "信号分下限",
        "relative_strength_min": "相对强弱下限",
        "market_score_min": "大盘环境分下限",
        "backtest_win_rate_min": "买点回测胜率下限",
        "high_probability_target_score": "高概率分数下限",
        "max_backtest_stop_hit_rate": "最大止损命中率",
        "backtest_trust_score_min": "回测可信度下限",
    }.get(attr, attr)


def _gate_rows_for_group(
    events: pd.DataFrame,
    group_column: str,
    group_zh_column: str | None,
    scope: str,
    scope_zh: str,
    target_window: int,
    min_samples: int,
    min_win_rate: float,
    min_avg_return: float,
    max_worst_return: float,
) -> list[dict[str, object]]:
    if group_column not in events.columns:
        return []
    rows = []
    for value, group in events.groupby(group_column, dropna=False):
        value_zh = value
        if group_zh_column and group_zh_column in group.columns:
            value_zh = group[group_zh_column].dropna().astype(str).iloc[0]
        rows.append(
            _gate_row(
                group,
                scope=scope,
                scope_zh=scope_zh,
                group_value=str(value),
                group_value_zh=str(value_zh),
                target_window=target_window,
                min_samples=min_samples,
                min_win_rate=min_win_rate,
                min_avg_return=min_avg_return,
                max_worst_return=max_worst_return,
            )
        )
    return rows


def _gate_row(
    events: pd.DataFrame,
    scope: str,
    scope_zh: str,
    group_value: str,
    group_value_zh: str,
    target_window: int,
    min_samples: int,
    min_win_rate: float,
    min_avg_return: float,
    max_worst_return: float,
) -> dict[str, object]:
    return_column = f"forward_return_{target_window}d"
    returns = (
        pd.to_numeric(events[return_column], errors="coerce").dropna()
        if return_column in events.columns
        else pd.Series(dtype=float)
    )
    metrics = _metric_row(returns, target_window)
    action, action_zh, reason, reason_zh = _deployment_gate_decision(
        metrics=metrics,
        min_samples=min_samples,
        min_win_rate=min_win_rate,
        min_avg_return=min_avg_return,
        max_worst_return=max_worst_return,
    )
    return {
        "scope": scope,
        "scope_zh": scope_zh,
        "group_value": group_value,
        "group_value_zh": group_value_zh,
        "forward_window_days": target_window,
        "sample_count": metrics["sample_count"],
        "win_rate": metrics["win_rate"],
        "avg_return": metrics["avg_return"],
        "median_return": metrics["median_return"],
        "worst_return": metrics["worst_return"],
        "best_return": metrics["best_return"],
        "sample_quality": metrics["sample_quality"],
        "sample_quality_zh": metrics["sample_quality_zh"],
        "deployment_gate_action": action,
        "deployment_gate_action_zh": action_zh,
        "deployment_gate_reason": reason,
        "deployment_gate_reason_zh": reason_zh,
    }


def _deployment_gate_decision(
    metrics: dict[str, object],
    min_samples: int,
    min_win_rate: float,
    min_avg_return: float,
    max_worst_return: float,
) -> tuple[str, str, str, str]:
    sample_count = int(metrics["sample_count"])
    win_rate = float(metrics["win_rate"]) if _is_finite(metrics["win_rate"]) else np.nan
    avg_return = float(metrics["avg_return"]) if _is_finite(metrics["avg_return"]) else np.nan
    worst_return = float(metrics["worst_return"]) if _is_finite(metrics["worst_return"]) else np.nan
    if sample_count == 0:
        return (
            "pause_until_samples_exist",
            "暂停，等待样本",
            "no historical validation samples",
            "没有历史验证样本",
        )
    if sample_count < min_samples:
        return (
            "collect_more_samples",
            "继续收集样本",
            f"sample count {sample_count} is below required {min_samples}",
            f"样本数{sample_count}低于要求{min_samples}",
        )
    if not _is_finite(win_rate) or win_rate < min_win_rate:
        return (
            "tighten_or_block_high_probability",
            "收紧或阻止高概率通过",
            f"win rate below {min_win_rate:.0%}",
            f"胜率低于{min_win_rate:.0%}",
        )
    if not _is_finite(avg_return) or avg_return <= min_avg_return:
        return (
            "tighten_return_quality",
            "收紧收益质量",
            "average return is not positive enough",
            "平均收益不够理想",
        )
    if _is_finite(worst_return) and worst_return < max_worst_return:
        return (
            "tighten_risk_controls",
            "收紧风险控制",
            f"worst return below {max_worst_return:.0%}",
            f"最差收益低于{max_worst_return:.0%}",
        )
    return (
        "allow_high_probability_filter",
        "允许高概率筛选器继续使用",
        "historical evidence meets deployment gate",
        "历史证据满足部署门槛",
    )


def _summarize_groups(
    events: pd.DataFrame,
    group_columns: list[str],
    forward_windows: tuple[int, ...],
) -> pd.DataFrame:
    columns = _dashboard_columns(group_columns)
    if events.empty:
        return pd.DataFrame(columns=columns)

    usable_group_columns = [column for column in group_columns if column in events.columns]
    rows: list[dict[str, object]] = []
    if usable_group_columns:
        grouped = events.groupby(usable_group_columns, dropna=False)
        for keys, group in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)
            base = dict(zip(usable_group_columns, keys, strict=False))
            rows.extend(_window_rows(group, base, forward_windows))
    else:
        rows.extend(_window_rows(events, {}, forward_windows))

    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        return frame
    sort_columns = [column for column in group_columns if column in frame.columns] + ["forward_window_days"]
    return frame.sort_values(sort_columns).reset_index(drop=True)


def _window_rows(
    group: pd.DataFrame,
    base: dict[str, object],
    forward_windows: tuple[int, ...],
) -> list[dict[str, object]]:
    rows = []
    for window in forward_windows:
        return_column = f"forward_return_{window}d"
        if return_column not in group.columns:
            row = dict(base)
            row.update(_empty_metric_row(window))
            rows.append(row)
            continue
        returns = pd.to_numeric(group[return_column], errors="coerce").dropna()
        row = dict(base)
        row.update(_metric_row(returns, window))
        rows.append(row)
    return rows


def _metric_row(returns: pd.Series, window: int) -> dict[str, object]:
    sample_count = int(len(returns))
    if sample_count == 0:
        return _empty_metric_row(window)
    winners = returns[returns > 0]
    losers = returns[returns <= 0]
    positive_count = int(len(winners))
    negative_count = int(len(losers))
    avg_gain = float(winners.mean()) if positive_count else np.nan
    avg_loss = float(losers.mean()) if negative_count else np.nan
    payoff_ratio = avg_gain / abs(avg_loss) if _is_finite(avg_gain) and _is_finite(avg_loss) and avg_loss < 0 else np.nan
    win_rate = float(positive_count / sample_count)
    avg_return = float(returns.mean())
    sample_quality, sample_quality_zh = _sample_quality(sample_count)
    result_label, result_label_zh = _result_label(sample_count, win_rate, avg_return)
    return {
        "forward_window_days": window,
        "sample_count": sample_count,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "win_rate": win_rate,
        "avg_return": avg_return,
        "median_return": float(returns.median()),
        "avg_gain": avg_gain,
        "avg_loss": avg_loss,
        "payoff_ratio": float(payoff_ratio) if _is_finite(payoff_ratio) else np.nan,
        "expectancy": avg_return,
        "worst_return": float(returns.min()),
        "best_return": float(returns.max()),
        "sample_quality": sample_quality,
        "sample_quality_zh": sample_quality_zh,
        "result_label": result_label,
        "result_label_zh": result_label_zh,
    }


def _empty_metric_row(window: int) -> dict[str, object]:
    return {
        "forward_window_days": window,
        "sample_count": 0,
        "positive_count": 0,
        "negative_count": 0,
        "win_rate": np.nan,
        "avg_return": np.nan,
        "median_return": np.nan,
        "avg_gain": np.nan,
        "avg_loss": np.nan,
        "payoff_ratio": np.nan,
        "expectancy": np.nan,
        "worst_return": np.nan,
        "best_return": np.nan,
        "sample_quality": "none",
        "sample_quality_zh": "没有样本",
        "result_label": "no_data",
        "result_label_zh": "没有数据",
    }


def _sample_quality(sample_count: int) -> tuple[str, str]:
    if sample_count >= 50:
        return "strong", "样本较强"
    if sample_count >= 30:
        return "usable", "样本可用"
    if sample_count >= 10:
        return "limited", "样本偏少"
    if sample_count > 0:
        return "weak", "样本很少"
    return "none", "没有样本"


def _result_label(sample_count: int, win_rate: float, avg_return: float) -> tuple[str, str]:
    if sample_count < 10:
        return "too_few_samples", "样本不足"
    if win_rate >= 0.55 and avg_return > 0:
        if sample_count >= 30:
            return "historically_favorable", "历史表现较好"
        return "promising_but_thin", "表现不错但样本偏少"
    if avg_return > 0:
        return "positive_but_mixed", "收益为正但胜率一般"
    return "historically_weak", "历史表现偏弱"


def _dashboard_columns(group_columns: list[str]) -> list[str]:
    return group_columns + [
        "forward_window_days",
        "sample_count",
        "positive_count",
        "negative_count",
        "win_rate",
        "avg_return",
        "median_return",
        "avg_gain",
        "avg_loss",
        "payoff_ratio",
        "expectancy",
        "worst_return",
        "best_return",
        "sample_quality",
        "sample_quality_zh",
        "result_label",
        "result_label_zh",
    ]


def _quality_gate_column(events: pd.DataFrame) -> str:
    if "calibrated_quality_gate_passed" in events.columns:
        return "calibrated_quality_gate_passed"
    return "quality_gate_passed"


def _format_dashboard_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    formatted = frame.copy()
    percent_columns = [
        "win_rate",
        "avg_return",
        "median_return",
        "avg_gain",
        "avg_loss",
        "expectancy",
        "worst_return",
        "best_return",
        "quality_gate_pass_rate",
        "avg_max_drawdown_after_signal",
    ]
    for column in percent_columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(_format_percent)
    if "payoff_ratio" in formatted.columns:
        formatted["payoff_ratio"] = formatted["payoff_ratio"].map(_format_number)
    return formatted


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["No rows / 暂无数据。"]
    header = "| " + " | ".join(str(column) for column in frame.columns) + " |"
    separator = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = [header, separator]
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join(_cell(value) for value in row.tolist()) + " |")
    return rows


def _cell(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _format_percent(value: object) -> str:
    try:
        if value != value:
            return "N/A"
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "N/A"


def _format_number(value: object) -> str:
    try:
        if value != value:
            return "N/A"
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "N/A"


def _first_text(frame: pd.DataFrame, column: str, default: str) -> str:
    if column not in frame.columns:
        return default
    values = frame[column].dropna().astype(str)
    if values.empty:
        return default
    return values.iloc[0]


def _is_finite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False
