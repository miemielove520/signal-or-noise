"""Decision, blocker, confidence, risk and data-quality summaries built from horizon rows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import (
    _clamp_score,
    _coerce_bool,
    _format_number,
    _is_finite,
    _row_float,
)


def _build_decision_summary(analysis: pd.DataFrame) -> dict[str, object]:
    actionable_actions = {"entry_breakout_confirmed", "entry_pullback_zone"}
    watch_actions = {"watch_breakout_or_pullback", "wait_overextended"}
    risk_wait_actions = {
        "wait_market_weak",
        "wait_relative_weak",
        "wait_event_risk",
        "wait_sentiment_risk",
        "wait_fundamental_weak",
        "wait_sector_weak",
    }

    actionable = analysis[analysis["action"].isin(actionable_actions)]
    if not actionable.empty:
        focus = actionable.sort_values("signal_score", ascending=False).iloc[0]
        decision = "actionable_setup"
        decision_zh = "有可执行信号，但仍需按计划控制风险"
    else:
        risk_wait = analysis[analysis["action"].isin(risk_wait_actions)]
        if not risk_wait.empty:
            focus = risk_wait.sort_values("signal_score", ascending=False).iloc[0]
            decision = "wait_due_to_risk_filter"
            decision_zh = "等待，风险过滤条件未通过"
        else:
            watch = analysis[analysis["action"].isin(watch_actions)]
            if not watch.empty:
                focus = watch.sort_values("signal_score", ascending=False).iloc[0]
                decision = "watch_for_confirmation"
                decision_zh = "观察，等待突破或回调确认"
            else:
                focus = analysis.sort_values("signal_score", ascending=False).iloc[0]
                decision = "avoid_or_wait"
                decision_zh = "回避或等待，当前信号不足"

    reason, reason_zh = _decision_reason(focus)
    wait_for, wait_for_zh = _decision_wait_for(focus)
    invalidation, invalidation_zh = _decision_invalidation(focus)

    return {
        "primary_decision": decision,
        "primary_decision_zh": decision_zh,
        "decision_focus_horizon": focus["horizon"],
        "decision_reason": reason,
        "decision_reason_zh": reason_zh,
        "decision_wait_for": wait_for,
        "decision_wait_for_zh": wait_for_zh,
        "decision_invalidation": invalidation,
        "decision_invalidation_zh": invalidation_zh,
    }


def _build_final_decision_summary(analysis: pd.DataFrame) -> dict[str, object]:
    focus = analysis.sort_values(
        [
            "calibrated_quality_gate_passed",
            "calibrated_high_probability_score",
            "signal_score",
        ],
        ascending=[False, False, False],
    ).iloc[0]
    watchlist_status = str(focus["calibrated_watchlist_status"])
    if bool(focus["calibrated_quality_gate_passed"]):
        decision = "high_probability_candidate"
        decision_zh = "高概率候选，可以重点跟踪"
    elif watchlist_status == "close_but_not_ready":
        decision = "near_watchlist"
        decision_zh = "接近机会，但还未达标"
    elif watchlist_status == "early_watch":
        decision = "early_watch"
        decision_zh = "早期观察，暂不急"
    else:
        decision = "avoid_for_now"
        decision_zh = "暂时回避"

    reason, reason_zh = _final_decision_reason(focus)
    next_step, next_step_zh = _final_next_step(focus)
    return {
        "final_decision": decision,
        "final_decision_zh": decision_zh,
        "final_focus_horizon": focus["horizon"],
        "final_focus_horizon_zh": focus["horizon_zh_label"],
        "final_score": float(focus["calibrated_high_probability_score"]),
        "final_watchlist_status": focus["calibrated_watchlist_status"],
        "final_watchlist_status_zh": focus["calibrated_watchlist_status_zh"],
        "final_reason": reason,
        "final_reason_zh": reason_zh,
        "final_next_step": next_step,
        "final_next_step_zh": next_step_zh,
    }


def _build_priority_blocker_summary(analysis: pd.DataFrame) -> dict[str, object]:
    focus = analysis.sort_values(
        [
            "calibrated_quality_gate_passed",
            "calibrated_high_probability_score",
            "signal_score",
        ],
        ascending=[False, False, False],
    ).iloc[0]
    blockers = _priority_blockers_for_row(focus)
    if not blockers:
        return {
            "primary_blocker": "none",
            "primary_blocker_zh": "无",
            "priority_blockers": "none",
            "priority_blockers_zh": "无",
            "priority_blocker_count": 0,
            "priority_blocker_note": "No major blocker is detected for the selected focus horizon.",
            "priority_blocker_note_zh": "当前重点周期没有明显主要卡点。",
            "primary_blocker_resolution": "No blocker resolution is needed.",
            "primary_blocker_resolution_zh": "当前不需要解除主要卡点。",
            "blocker_resolution_steps": "none",
            "blocker_resolution_steps_zh": "无",
            "blocker_recheck_trigger": "Continue tracking the planned entry and risk controls.",
            "blocker_recheck_trigger_zh": "继续跟踪计划买点和风险控制。",
            "primary_blocker_progress": 100.0,
            "blocker_resolution_score": 100.0,
            "blocker_resolution_level": "ready",
            "blocker_resolution_level_zh": "已解除",
            "blocker_resolution_gap": "No major blocker remains.",
            "blocker_resolution_gap_zh": "当前没有剩余主要卡点。",
        }

    primary = blockers[0]
    resolution_steps = _blocker_resolution_steps_for_row(focus, blockers)
    resolution_progress = _blocker_resolution_progress_for_row(focus, blockers)
    return {
        "primary_blocker": primary["code"],
        "primary_blocker_zh": primary["code_zh"],
        "priority_blockers": "; ".join(str(item["code"]) for item in blockers),
        "priority_blockers_zh": "；".join(str(item["code_zh"]) for item in blockers),
        "priority_blocker_count": len(blockers),
        "priority_blocker_note": " ".join(str(item["note"]) for item in blockers),
        "priority_blocker_note_zh": " ".join(str(item["note_zh"]) for item in blockers),
        "primary_blocker_resolution": resolution_steps[0]["step"],
        "primary_blocker_resolution_zh": resolution_steps[0]["step_zh"],
        "blocker_resolution_steps": " | ".join(str(item["step"]) for item in resolution_steps),
        "blocker_resolution_steps_zh": "；".join(str(item["step_zh"]) for item in resolution_steps),
        "blocker_recheck_trigger": resolution_steps[0]["trigger"],
        "blocker_recheck_trigger_zh": resolution_steps[0]["trigger_zh"],
        "primary_blocker_progress": resolution_progress["primary_blocker_progress"],
        "blocker_resolution_score": resolution_progress["blocker_resolution_score"],
        "blocker_resolution_level": resolution_progress["blocker_resolution_level"],
        "blocker_resolution_level_zh": resolution_progress["blocker_resolution_level_zh"],
        "blocker_resolution_gap": resolution_progress["blocker_resolution_gap"],
        "blocker_resolution_gap_zh": resolution_progress["blocker_resolution_gap_zh"],
    }


def _blocker_resolution_progress_for_row(
    row: pd.Series,
    blockers: list[dict[str, object]],
) -> dict[str, object]:
    progress_items = [
        _blocker_progress_item(row, str(blocker["code"]), str(blocker["code_zh"]))
        for blocker in blockers
    ]
    if not progress_items:
        score = 100.0
        primary_progress = 100.0
    else:
        score = round(
            sum(float(item["progress"]) for item in progress_items) / len(progress_items),
            2,
        )
        primary_progress = round(float(progress_items[0]["progress"]), 2)
    level, level_zh = _blocker_resolution_level(score)
    gap = "; ".join(str(item["gap"]) for item in progress_items) or "No major blocker remains."
    gap_zh = "；".join(str(item["gap_zh"]) for item in progress_items) or "当前没有剩余主要卡点。"
    return {
        "primary_blocker_progress": primary_progress,
        "blocker_resolution_score": score,
        "blocker_resolution_level": level,
        "blocker_resolution_level_zh": level_zh,
        "blocker_resolution_gap": gap,
        "blocker_resolution_gap_zh": gap_zh,
    }


def _blocker_progress_item(
    row: pd.Series,
    blocker_code: str,
    blocker_code_zh: str,
) -> dict[str, object]:
    signal_threshold = _row_float(row, "recommended_signal_threshold", 65.0)
    confidence_threshold = _row_float(row, "recommended_confidence_threshold", 65.0)
    sample_threshold = max(1.0, _row_float(row, "recommended_backtest_sample_min", 10.0))
    win_rate_threshold = max(0.01, _row_float(row, "recommended_backtest_win_rate_min", 0.55))
    average_return_threshold = _row_float(row, "recommended_backtest_average_return_min", 0.0)
    current_map = {
        "entry_not_executable": _row_float(row, "entry_readiness_score", 0.0),
        "trade_plan_invalid": _row_float(row, "trade_plan_quality_score", 0.0),
        "overall_risk_high": 0.0 if str(row.get("overall_risk_level", "unknown")) == "high" else 100.0,
        "event_risk_blocks_entry": 0.0
        if str(row.get("event_risk_level", "unknown")) == "high"
        or _coerce_bool(row.get("event_block_new_entries", False))
        else 100.0,
        "sentiment_risk_high": 0.0
        if str(row.get("sentiment_risk_level", "unknown")) == "high"
        or _coerce_bool(row.get("sentiment_block_new_entries", False))
        else 100.0,
        "valuation_risk_high": 0.0
        if str(row.get("valuation_risk_level", "unknown")) == "high"
        or _coerce_bool(row.get("valuation_block_new_entries", False))
        else 100.0,
        "data_quality_insufficient": _ratio_progress(_row_float(row, "data_quality_score"), 70.0),
        "market_not_supportive": _ratio_progress(_row_float(row, "market_score", 50.0), 55.0),
        "relative_strength_weak": _ratio_progress(
            _row_float(row, "relative_strength_score", 50.0),
            45.0,
        ),
        "confidence_too_low": _ratio_progress(
            _row_float(row, "confidence_score"),
            confidence_threshold,
        ),
        "signal_score_too_low": _ratio_progress(
            _row_float(row, "signal_score"),
            signal_threshold,
        ),
        "backtest_sample_too_small": _ratio_progress(
            _row_float(row, "screening_backtest_trade_count"),
            sample_threshold,
        ),
        "backtest_win_rate_too_low": _ratio_progress(
            _row_float(row, "screening_backtest_win_rate", 0.0),
            win_rate_threshold,
        ),
        "backtest_average_return_not_positive": _average_return_progress(
            _row_float(row, "screening_backtest_average_return", np.nan),
            average_return_threshold,
        ),
    }
    progress = round(float(current_map.get(blocker_code, 0.0)), 2)
    return {
        "progress": progress,
        "gap": f"{blocker_code}: {progress:.1f}/100 resolved.",
        "gap_zh": f"{blocker_code_zh}: 已接近解除{progress:.1f}/100。",
    }


def _ratio_progress(current: float, target: float) -> float:
    if not _is_finite(current) or not _is_finite(target) or target <= 0:
        return 0.0
    return round(_clamp_score((current / target) * 100.0), 2)


def _average_return_progress(current: float, target: float) -> float:
    if not _is_finite(current):
        return 0.0
    if current > target:
        return 100.0
    return round(_clamp_score(50.0 + (current - target) * 1000.0), 2)


def _blocker_resolution_level(score: float) -> tuple[str, str]:
    if score >= 90:
        return "almost_ready", "接近解除"
    if score >= 70:
        return "near", "较接近"
    if score >= 40:
        return "moderate", "中等距离"
    return "far", "距离较远"


def _blocker_resolution_steps_for_row(
    row: pd.Series,
    blockers: list[dict[str, object]],
) -> list[dict[str, str]]:
    steps = [_blocker_resolution_step(row, str(blocker["code"])) for blocker in blockers]
    return steps or [
        {
            "step": "No blocker resolution is needed.",
            "step_zh": "当前不需要解除主要卡点。",
            "trigger": "Continue tracking the planned entry and risk controls.",
            "trigger_zh": "继续跟踪计划买点和风险控制。",
        }
    ]


def _blocker_resolution_step(row: pd.Series, blocker_code: str) -> dict[str, str]:
    signal_threshold = _row_float(row, "recommended_signal_threshold", 65.0)
    confidence_threshold = _row_float(row, "recommended_confidence_threshold", 65.0)
    sample_threshold = int(_row_float(row, "recommended_backtest_sample_min", 10.0))
    win_rate_threshold = _row_float(row, "recommended_backtest_win_rate_min", 0.55)
    average_return_threshold = _row_float(row, "recommended_backtest_average_return_min", 0.0)
    trigger_price = _row_float(
        row,
        "calibrated_watchlist_trigger_price",
        _row_float(row, "entry_price", np.nan),
    )
    signal_score = _row_float(row, "signal_score")
    confidence_score = _row_float(row, "confidence_score")
    data_quality_score = _row_float(row, "data_quality_score")
    market_score = _row_float(row, "market_score", 50.0)
    relative_strength_score = _row_float(row, "relative_strength_score", 50.0)
    trade_count = int(_row_float(row, "screening_backtest_trade_count"))
    win_rate = _row_float(row, "screening_backtest_win_rate", np.nan)
    average_return = _row_float(row, "screening_backtest_average_return", np.nan)
    risk_reward = _row_float(row, "risk_reward", np.nan)

    price_trigger = (
        f"price closes above {_format_number(trigger_price)}"
        if _is_finite(trigger_price)
        else "the planned entry trigger is available again"
    )
    price_trigger_zh = (
        f"价格收在{_format_number(trigger_price)}上方"
        if _is_finite(trigger_price)
        else "计划买点触发条件重新可用"
    )

    mapping = {
        "entry_not_executable": {
            "step": f"Wait until {price_trigger} and entry readiness turns true.",
            "step_zh": f"等待{price_trigger_zh}，并且买点可执行性变为通过。",
            "trigger": f"Re-check after {price_trigger} with entry_readiness_gate_passed=true.",
            "trigger_zh": f"{price_trigger_zh}且买点可执行性通过后重新检查。",
        },
        "trade_plan_invalid": {
            "step": (
                "Wait for a complete trade plan: executable entry type, stop below entry, "
                f"target above entry, and risk/reward at least 2.00. Current risk/reward={_format_number(risk_reward)}."
            ),
            "step_zh": (
                "等待完整交易计划：入场类型可执行、止损低于入场价、目标价高于入场价，"
                f"并且盈亏比至少2.00。当前盈亏比={_format_number(risk_reward)}。"
            ),
            "trigger": "Re-check when trade_plan_quality_gate_passed=true.",
            "trigger_zh": "交易计划质量通过后重新检查。",
        },
        "overall_risk_high": {
            "step": "Wait until overall_risk_level falls below high.",
            "step_zh": "等待综合风险等级降到高风险以下。",
            "trigger": "Re-check when overall_risk_level is medium or low.",
            "trigger_zh": "综合风险变为中或低后重新检查。",
        },
        "event_risk_blocks_entry": {
            "step": "Wait until the earnings or event window no longer blocks new entries.",
            "step_zh": "等待财报或事件窗口不再阻止新入场。",
            "trigger": "Re-check when event_block_new_entries=false and event risk is not high.",
            "trigger_zh": "事件不再阻止新入场且事件风险不是高风险后重新检查。",
        },
        "sentiment_risk_high": {
            "step": "Wait until news sentiment risk improves and no longer blocks new entries.",
            "step_zh": "等待新闻情绪风险改善，并且不再阻止新入场。",
            "trigger": "Re-check when sentiment risk is not high.",
            "trigger_zh": "新闻情绪风险不是高风险后重新检查。",
        },
        "valuation_risk_high": {
            "step": "Wait until valuation risk improves or price/earnings growth makes valuation less restrictive.",
            "step_zh": "等待估值风险改善，或价格、盈利增长让估值不再构成限制。",
            "trigger": "Re-check when valuation risk is not high and valuation does not block new entries.",
            "trigger_zh": "估值风险不是高风险且估值不阻止新入场后重新检查。",
        },
        "data_quality_insufficient": {
            "step": (
                f"Improve data coverage until data_quality_score reaches 70.0. "
                f"Current score={data_quality_score:.1f}; try a longer period such as 5y or refresh missing data layers."
            ),
            "step_zh": (
                f"提高数据覆盖率，直到数据质量分达到70.0。当前分数={data_quality_score:.1f}；"
                "可以尝试使用5y周期或刷新缺失的数据层。"
            ),
            "trigger": "Re-check when data_quality_score >= 70.0.",
            "trigger_zh": "数据质量分达到70.0以上后重新检查。",
        },
        "market_not_supportive": {
            "step": f"Wait until market_score reaches 55.0. Current score={market_score:.1f}.",
            "step_zh": f"等待大盘分数达到55.0。当前分数={market_score:.1f}。",
            "trigger": "Re-check when market_score >= 55.0.",
            "trigger_zh": "大盘分数达到55.0以上后重新检查。",
        },
        "relative_strength_weak": {
            "step": (
                f"Wait until relative_strength_score reaches 45.0 versus SPY/QQQ. "
                f"Current score={relative_strength_score:.1f}."
            ),
            "step_zh": (
                f"等待相对SPY/QQQ的相对强弱分数达到45.0。当前分数={relative_strength_score:.1f}。"
            ),
            "trigger": "Re-check when relative_strength_score >= 45.0.",
            "trigger_zh": "相对强弱分数达到45.0以上后重新检查。",
        },
        "confidence_too_low": {
            "step": (
                f"Wait until confidence_score reaches {confidence_threshold:.1f}. "
                f"Current score={confidence_score:.1f}."
            ),
            "step_zh": (
                f"等待置信度分数达到{confidence_threshold:.1f}。当前分数={confidence_score:.1f}。"
            ),
            "trigger": f"Re-check when confidence_score >= {confidence_threshold:.1f}.",
            "trigger_zh": f"置信度分数达到{confidence_threshold:.1f}以上后重新检查。",
        },
        "signal_score_too_low": {
            "step": (
                f"Wait until signal_score reaches {signal_threshold:.1f}. "
                f"Current score={signal_score:.1f}."
            ),
            "step_zh": f"等待信号分数达到{signal_threshold:.1f}。当前分数={signal_score:.1f}。",
            "trigger": f"Re-check when signal_score >= {signal_threshold:.1f}.",
            "trigger_zh": f"信号分数达到{signal_threshold:.1f}以上后重新检查。",
        },
        "backtest_sample_too_small": {
            "step": (
                f"Use more history or wait for more similar entries until sample count reaches {sample_threshold}. "
                f"Current sample={trade_count}."
            ),
            "step_zh": (
                f"使用更长历史数据或等待更多类似买点，直到样本数达到{sample_threshold}。当前样本={trade_count}。"
            ),
            "trigger": f"Re-check when screening_backtest_trade_count >= {sample_threshold}.",
            "trigger_zh": f"买点回测样本数达到{sample_threshold}以上后重新检查。",
        },
        "backtest_win_rate_too_low": {
            "step": (
                f"Wait until selected entry backtest win rate reaches {_format_number(win_rate_threshold)}. "
                f"Current win rate={_format_number(win_rate)}."
            ),
            "step_zh": (
                f"等待所选买点回测胜率达到{_format_number(win_rate_threshold)}。"
                f"当前胜率={_format_number(win_rate)}。"
            ),
            "trigger": f"Re-check when screening_backtest_win_rate >= {_format_number(win_rate_threshold)}.",
            "trigger_zh": f"买点回测胜率达到{_format_number(win_rate_threshold)}以上后重新检查。",
        },
        "backtest_average_return_not_positive": {
            "step": (
                f"Wait until selected entry backtest average return rises above {_format_number(average_return_threshold)}. "
                f"Current average return={_format_number(average_return)}."
            ),
            "step_zh": (
                f"等待所选买点回测平均收益高于{_format_number(average_return_threshold)}。"
                f"当前平均收益={_format_number(average_return)}。"
            ),
            "trigger": f"Re-check when screening_backtest_average_return > {_format_number(average_return_threshold)}.",
            "trigger_zh": f"买点回测平均收益高于{_format_number(average_return_threshold)}后重新检查。",
        },
    }
    return mapping.get(
        blocker_code,
        {
            "step": f"Resolve blocker: {blocker_code}.",
            "step_zh": f"解除卡点：{blocker_code}。",
            "trigger": f"Re-check when {blocker_code} is no longer present.",
            "trigger_zh": f"{blocker_code}不再出现后重新检查。",
        },
    )


def _priority_blockers_for_row(row: pd.Series) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    seen_codes: set[str] = set()

    def row_float(key: str, default: float = 0.0) -> float:
        value = row.get(key, default)
        return float(value) if _is_finite(value) else default

    def add(
        condition: bool,
        priority: int,
        code: str,
        code_zh: str,
        note: str,
        note_zh: str,
    ) -> None:
        if not condition or code in seen_codes:
            return
        seen_codes.add(code)
        blockers.append(
            {
                "priority": priority,
                "code": code,
                "code_zh": code_zh,
                "note": note,
                "note_zh": note_zh,
            }
        )

    signal_threshold = row_float("recommended_signal_threshold", 65.0)
    confidence_threshold = row_float("recommended_confidence_threshold", 65.0)
    sample_threshold = int(row_float("recommended_backtest_sample_min", 10.0))
    win_rate_threshold = row_float("recommended_backtest_win_rate_min", 0.55)
    average_return_threshold = row_float("recommended_backtest_average_return_min", 0.0)
    signal_score = row_float("signal_score")
    confidence_score = row_float("confidence_score")
    data_quality_score = row_float("data_quality_score")
    market_score = row_float("market_score", 50.0)
    relative_strength_score = row_float("relative_strength_score", 50.0)
    trade_count = int(row_float("screening_backtest_trade_count"))
    win_rate = row_float("screening_backtest_win_rate", np.nan)
    average_return = row_float("screening_backtest_average_return", np.nan)

    add(
        not _coerce_bool(row.get("entry_readiness_gate_passed", True), True),
        100,
        "entry_not_executable",
        "买点当前不可执行",
        f"Entry readiness failed: {row.get('entry_readiness_reason', 'unknown')}.",
        f"买点可执行性未通过：{row.get('entry_readiness_reason_zh', '未知原因')}。",
    )
    add(
        not _coerce_bool(row.get("trade_plan_quality_gate_passed", True), True),
        95,
        "trade_plan_invalid",
        "交易计划质量不足",
        f"Trade plan quality failed: {row.get('trade_plan_quality_reason', 'unknown')}.",
        f"交易计划质量未通过：{row.get('trade_plan_quality_reason_zh', '未知原因')}。",
    )
    add(
        str(row.get("overall_risk_level", "unknown")) == "high",
        90,
        "overall_risk_high",
        "综合风险过高",
        "Overall risk level is high, so the setup should be filtered before entry timing.",
        "综合风险等级为高，因此应先过滤，不急着看买点。",
    )
    add(
        str(row.get("event_risk_level", "unknown")) == "high"
        or _coerce_bool(row.get("event_block_new_entries", False)),
        88,
        "event_risk_blocks_entry",
        "事件风险阻止入场",
        "Event risk is high or the event window blocks new entries.",
        "事件风险偏高，或事件窗口阻止新入场。",
    )
    add(
        str(row.get("sentiment_risk_level", "unknown")) == "high"
        or _coerce_bool(row.get("sentiment_block_new_entries", False)),
        84,
        "sentiment_risk_high",
        "新闻情绪风险过高",
        "News sentiment is a high-risk input for this setup.",
        "新闻情绪对当前机会构成高风险。",
    )
    add(
        str(row.get("valuation_risk_level", "unknown")) == "high"
        or _coerce_bool(row.get("valuation_block_new_entries", False)),
        82,
        "valuation_risk_high",
        "估值风险过高",
        "Valuation risk is high or valuation blocks new entries.",
        "估值风险偏高，或估值层阻止新入场。",
    )
    add(
        data_quality_score < 70.0,
        80,
        "data_quality_insufficient",
        "数据质量不足",
        f"Data quality score is {data_quality_score:.1f}, below the strict 70.0 floor.",
        f"数据质量分数为{data_quality_score:.1f}，低于严格筛选的70.0下限。",
    )
    add(
        market_score < 55.0,
        76,
        "market_not_supportive",
        "大盘环境不支持",
        f"Market score is {market_score:.1f}, below the 55.0 support floor.",
        f"大盘环境分数为{market_score:.1f}，低于55.0支持下限。",
    )
    add(
        relative_strength_score < 45.0,
        74,
        "relative_strength_weak",
        "相对强弱不足",
        f"Relative strength score is {relative_strength_score:.1f}, below the 45.0 floor.",
        f"相对强弱分数为{relative_strength_score:.1f}，低于45.0下限。",
    )
    add(
        confidence_score < confidence_threshold,
        70,
        "confidence_too_low",
        "置信度不足",
        f"Confidence score is {confidence_score:.1f}, below the calibrated threshold {confidence_threshold:.1f}.",
        f"置信度分数为{confidence_score:.1f}，低于校准门槛{confidence_threshold:.1f}。",
    )
    add(
        signal_score < signal_threshold,
        66,
        "signal_score_too_low",
        "信号分数不足",
        f"Signal score is {signal_score:.1f}, below the calibrated threshold {signal_threshold:.1f}.",
        f"信号分数为{signal_score:.1f}，低于校准门槛{signal_threshold:.1f}。",
    )
    add(
        trade_count < sample_threshold,
        62,
        "backtest_sample_too_small",
        "回测样本不足",
        f"Entry backtest has {trade_count} samples, below the calibrated minimum {sample_threshold}.",
        f"买点回测样本数为{trade_count}，低于校准下限{sample_threshold}。",
    )
    add(
        not _is_finite(win_rate) or win_rate < win_rate_threshold,
        60,
        "backtest_win_rate_too_low",
        "回测胜率不足",
        f"Entry backtest win rate is {_format_number(win_rate)}, below the calibrated minimum {_format_number(win_rate_threshold)}.",
        f"买点回测胜率为{_format_number(win_rate)}，低于校准下限{_format_number(win_rate_threshold)}。",
    )
    add(
        not _is_finite(average_return) or average_return <= average_return_threshold,
        58,
        "backtest_average_return_not_positive",
        "回测平均收益不足",
        f"Entry backtest average return is {_format_number(average_return)}, not above the calibrated minimum {_format_number(average_return_threshold)}.",
        f"买点回测平均收益为{_format_number(average_return)}，未高于校准下限{_format_number(average_return_threshold)}。",
    )

    blockers.sort(key=lambda item: int(item["priority"]), reverse=True)
    return blockers[:3]


def _final_decision_reason(row: pd.Series) -> tuple[str, str]:
    score = float(row["calibrated_high_probability_score"])
    missing = str(row["calibrated_watchlist_missing_items"])
    missing_zh = str(row["calibrated_watchlist_missing_items_zh"])
    if bool(row["calibrated_quality_gate_passed"]):
        return (
            f"Calibrated filter passed on {row['horizon']} with score {score:.1f}.",
            f"{row['horizon_zh_label']}通过校准后筛选，分数{score:.1f}。",
        )
    return (
        f"Calibrated filter did not pass on {row['horizon']} with score {score:.1f}; missing items: {missing}.",
        f"{row['horizon_zh_label']}未通过校准后筛选，分数{score:.1f}；未达标条件：{missing_zh}。",
    )


def _final_next_step(row: pd.Series) -> tuple[str, str]:
    if bool(row["calibrated_quality_gate_passed"]):
        return (
            "Track the planned entry, stop, and invalidation levels; do not override risk controls.",
            "跟踪计划买点、止损和失效条件；不要绕过风险控制。",
        )
    return (
        str(row["calibrated_watchlist_recheck_reason"]),
        str(row["calibrated_watchlist_recheck_reason_zh"]),
    )


def _build_horizon_alignment_summary(analysis: pd.DataFrame) -> dict[str, object]:
    buckets = [_action_bucket(str(action)) for action in analysis["action"]]
    constructive_count = buckets.count("constructive")
    weak_count = buckets.count("weak")
    risk_wait_count = buckets.count("risk_wait")
    score_spread = float(analysis["signal_score"].max() - analysis["signal_score"].min())
    label, label_zh, score, note, note_zh = _horizon_alignment_label(
        constructive_count=constructive_count,
        weak_count=weak_count,
        risk_wait_count=risk_wait_count,
        total_count=len(buckets),
        score_spread=score_spread,
    )
    return {
        "horizon_alignment_score": round(score, 2),
        "horizon_alignment_label": label,
        "horizon_alignment_label_zh": label_zh,
        "constructive_horizon_count": constructive_count,
        "weak_horizon_count": weak_count,
        "risk_wait_horizon_count": risk_wait_count,
        "horizon_signal_score_spread": round(score_spread, 2),
        "horizon_alignment_note": note,
        "horizon_alignment_note_zh": note_zh,
    }


def _horizon_alignment_label(
    constructive_count: int,
    weak_count: int,
    risk_wait_count: int,
    total_count: int,
    score_spread: float,
) -> tuple[str, str, float, str, str]:
    if total_count <= 1:
        return (
            "single_horizon",
            "单周期",
            55.0,
            "Only one horizon was analyzed, so cross-horizon confirmation is unavailable.",
            "只分析了一个周期，因此没有短中长交叉确认。",
        )
    if constructive_count == total_count:
        return (
            "aligned_constructive",
            "多周期一致偏积极",
            90.0 if score_spread <= 20 else 82.0,
            "All analyzed horizons are constructive.",
            "所有已分析周期都偏积极。",
        )
    if weak_count == total_count:
        return (
            "aligned_weak",
            "多周期一致偏弱",
            35.0,
            "All analyzed horizons are weak, so there is no confirmation for a new entry.",
            "所有已分析周期都偏弱，因此没有新入场确认。",
        )
    if constructive_count > 0 and weak_count > 0:
        return (
            "conflicting",
            "周期信号冲突",
            25.0,
            "Constructive and weak horizons exist at the same time.",
            "偏积极和偏弱周期同时存在。",
        )
    if risk_wait_count > 0 and constructive_count > 0:
        return (
            "risk_blocked_mixed",
            "部分周期被风险阻挡",
            40.0,
            "Some horizons are constructive, but other horizons are blocked by risk filters.",
            "部分周期偏积极，但其他周期被风险过滤阻挡。",
        )
    if constructive_count >= 2:
        return (
            "mostly_constructive",
            "多数周期偏积极",
            72.0 if score_spread <= 25 else 64.0,
            "Most horizons are constructive, but confirmation is not complete.",
            "多数周期偏积极，但确认还不完整。",
        )
    return (
        "mixed",
        "周期信号混合",
        50.0,
        "Horizon signals are mixed without a clear cross-horizon confirmation.",
        "周期信号混合，暂时没有清晰的跨周期确认。",
    )


def _decision_reason(row: pd.Series) -> tuple[str, str]:
    score = float(row["signal_score"])
    reason = (
        f"The strongest current setup is {row['horizon']} with score {score:.1f}. "
        f"Action is {row['action']}. Technical score={float(row['technical_score']):.1f}, "
        f"market score={float(row['market_score']):.1f}, relative strength="
        f"{float(row['relative_strength_score']):.1f}, fundamental score="
        f"{float(row['fundamental_score']):.1f}, event risk={row['event_risk_level']}."
    )
    reason_zh = (
        f"当前最值得关注的是{row['horizon_zh_label']}，分数{score:.1f}。"
        f"动作是{row['action']}。技术分={float(row['technical_score']):.1f}，"
        f"大盘分={float(row['market_score']):.1f}，相对强弱分="
        f"{float(row['relative_strength_score']):.1f}，基本面分="
        f"{float(row['fundamental_score']):.1f}，事件风险={row['event_risk_level_zh']}。"
    )
    return reason, reason_zh


def _decision_wait_for(row: pd.Series) -> tuple[str, str]:
    action = str(row["action"])
    entry = _format_number(row["entry_price"])
    support = _format_number(row["support"])
    resistance = _format_number(row["resistance"])

    if action == "entry_breakout_confirmed":
        return (
            f"Breakout is already confirmed; only act if price holds above the entry area near {entry}.",
            f"突破已确认；只有价格能守住约{entry}的入场区域时才继续关注。",
        )
    if action == "entry_pullback_zone":
        return (
            f"Wait for the pullback area near {entry} to hold with improving momentum.",
            f"等待约{entry}附近的回调区域企稳，并且动量改善。",
        )
    if action == "watch_breakout_or_pullback":
        return (
            f"Wait for a breakout above resistance near {resistance}, or a controlled pullback near support around {support}.",
            f"等待突破约{resistance}附近压力位，或者回调到约{support}附近支撑后企稳。",
        )
    if action == "wait_overextended":
        return (
            f"Wait for price to cool off toward the planned entry area near {entry}; do not chase the current extension.",
            f"等待价格回落到约{entry}附近的计划区域；当前不适合追高。",
        )
    if action == "wait_market_weak":
        return (
            "Wait for the market score to recover above the weak zone while the stock setup remains intact.",
            "等待大盘分数脱离偏弱区间，同时个股结构仍保持有效。",
        )
    if action == "wait_relative_weak":
        return (
            "Wait for the stock to stop lagging SPY/QQQ and show improving relative strength.",
            "等待该股票不再弱于SPY/QQQ，并出现相对强弱改善。",
        )
    if action == "wait_event_risk":
        return (
            "Wait until the nearby earnings or event risk has passed, then reassess price reaction.",
            "等待临近财报或事件风险释放后，再重新评估价格反应。",
        )
    if action == "wait_fundamental_weak":
        return (
            "Wait for fundamental quality to improve before treating the long-term setup as investable.",
            "等待基本面质量改善后，再把长期结构视为可投资机会。",
        )
    if action == "wait_sector_weak":
        return (
            "Wait for the sector ETF to recover before trusting the medium or long-term setup.",
            "等待板块ETF恢复后，再相信中期或长期结构。",
        )
    if action == "avoid_or_wait_downtrend":
        return (
            "Wait for trend and momentum to turn positive before considering any fresh setup.",
            "等待趋势和动量转正后，再考虑新的机会。",
        )
    if action == "avoid_universe_filter":
        return (
            "Wait until the ticker passes the liquidity and history filters.",
            "等待该股票通过流动性和历史数据过滤。",
        )
    return (
        "Wait for a cleaner setup with better trend, market, and risk alignment.",
        "等待趋势、大盘和风险条件更一致的清晰结构。",
    )


def _decision_invalidation(row: pd.Series) -> tuple[str, str]:
    stop = _format_number(row["stop_loss"])
    support = _format_number(row["support"])
    return (
        f"The setup is invalid if price closes below the stop area near {stop}, loses support near {support}, "
        "or if market, event, relative-strength, or fundamental filters deteriorate materially.",
        f"如果价格收在约{stop}的止损区域下方、跌破约{support}附近支撑，"
        "或者大盘、事件风险、相对强弱、基本面过滤条件明显恶化，则判断失效。",
    )


def _build_confidence_summary(
    analysis: pd.DataFrame,
    focus_horizon: str,
) -> dict[str, object]:
    focus_matches = analysis[analysis["horizon"] == focus_horizon]
    focus = focus_matches.iloc[0] if not focus_matches.empty else analysis.iloc[0]

    score = 50.0
    notes: list[str] = []
    notes_zh: list[str] = []

    signal_score = float(focus["signal_score"])
    if signal_score >= 75:
        score += 15
        notes.append("primary signal is strong")
        notes_zh.append("主信号较强")
    elif signal_score >= 60:
        score += 8
        notes.append("primary signal is constructive")
        notes_zh.append("主信号尚可")
    elif signal_score < 45:
        score -= 10
        notes.append("primary signal is weak")
        notes_zh.append("主信号偏弱")
    else:
        notes.append("primary signal is mixed")
        notes_zh.append("主信号中性偏混合")

    total_samples = int(focus["breakout_trade_count"]) + int(focus["pullback_trade_count"])
    if total_samples >= 40:
        score += 15
        notes.append(f"entry backtest sample is strong ({total_samples} trades)")
        notes_zh.append(f"买点回测样本较强（{total_samples}笔）")
    elif total_samples >= 20:
        score += 8
        notes.append(f"entry backtest sample is moderate ({total_samples} trades)")
        notes_zh.append(f"买点回测样本一般（{total_samples}笔）")
    elif total_samples >= 10:
        score += 2
        notes.append(f"entry backtest sample is limited ({total_samples} trades)")
        notes_zh.append(f"买点回测样本偏少（{total_samples}笔）")
    else:
        score -= 10
        notes.append(f"entry backtest sample is too small ({total_samples} trades)")
        notes_zh.append(f"买点回测样本太少（{total_samples}笔）")

    warning_count = _warning_count(focus)
    if warning_count == 0:
        score += 8
        notes.append("core data layers are available")
        notes_zh.append("核心数据层可用")
    else:
        penalty = min(warning_count * 7, 21)
        score -= penalty
        notes.append(f"{warning_count} data layer warning(s)")
        notes_zh.append(f"存在{warning_count}个数据层提示")

    coverage = float(focus["fundamental_data_coverage"])
    if coverage >= 0.75:
        score += 5
        notes.append("fundamental coverage is good")
        notes_zh.append("基本面字段覆盖较好")
    elif coverage < 0.25:
        score -= 8
        notes.append("fundamental coverage is sparse")
        notes_zh.append("基本面字段覆盖不足")

    consistency_delta, consistency_note, consistency_note_zh = _signal_consistency_score(analysis)
    score += consistency_delta
    notes.append(consistency_note)
    notes_zh.append(consistency_note_zh)

    risk_delta, risk_note, risk_note_zh = _risk_filter_score(focus)
    score += risk_delta
    notes.append(risk_note)
    notes_zh.append(risk_note_zh)

    final_score = round(float(max(0.0, min(score, 100.0))), 2)
    level, level_zh = _confidence_level(final_score)
    return {
        "confidence_score": final_score,
        "confidence_level": level,
        "confidence_level_zh": level_zh,
        "confidence_note": "; ".join(notes) + ".",
        "confidence_note_zh": "；".join(notes_zh) + "。",
    }


def _warning_count(row: pd.Series) -> int:
    warning_fields = [
        "market_warning",
        "relative_strength_warning",
        "event_risk_warning",
        "sentiment_warning",
        "analyst_warning",
        "valuation_warning",
        "fundamental_warning",
        "sector_warning",
    ]
    count = 0
    for field in warning_fields:
        value = row.get(field, "")
        if isinstance(value, str) and value.strip():
            count += 1
    if str(row.get("event_risk_level", "")) == "unknown":
        count += 1
    if str(row.get("sentiment_risk_level", "")) == "unknown":
        count += 1
    if str(row.get("analyst_risk_level", "")) == "unknown":
        count += 1
    if str(row.get("valuation_risk_level", "")) == "unknown":
        count += 1
    if str(row.get("fundamental_quality", "")) == "unknown":
        count += 1
    if str(row.get("sector_status", "")) == "unknown":
        count += 1
    return count


def _signal_consistency_score(analysis: pd.DataFrame) -> tuple[float, str, str]:
    buckets = [_action_bucket(str(action)) for action in analysis["action"]]
    unique_buckets = set(buckets)
    if len(unique_buckets) == 1:
        return 10.0, "horizon signals are consistent", "短中长信号一致"
    if "constructive" in unique_buckets and "weak" in unique_buckets:
        return -8.0, "horizon signals are conflicting", "短中长信号存在冲突"
    if "risk_wait" in unique_buckets:
        return -4.0, "some horizons are blocked by risk filters", "部分周期受到风险过滤限制"
    return 0.0, "horizon signals are mixed", "短中长信号混合"


def _action_bucket(action: str) -> str:
    if action in {"entry_breakout_confirmed", "entry_pullback_zone", "watch_breakout_or_pullback"}:
        return "constructive"
    if action in {
        "wait_market_weak",
        "wait_relative_weak",
        "wait_event_risk",
        "wait_fundamental_weak",
        "wait_sector_weak",
    }:
        return "risk_wait"
    return "weak"


def _risk_filter_score(row: pd.Series) -> tuple[float, str, str]:
    penalties = 0.0
    notes: list[str] = []
    notes_zh: list[str] = []

    if float(row["market_score"]) <= 40:
        penalties -= 8
        notes.append("market is weak")
        notes_zh.append("大盘偏弱")
    if float(row["relative_strength_score"]) <= 40:
        penalties -= 7
        notes.append("relative strength is weak")
        notes_zh.append("相对强弱偏弱")
    if str(row["event_risk_level"]) == "high":
        penalties -= 10
        notes.append("event risk is high")
        notes_zh.append("事件风险高")
    if str(row.get("sentiment_risk_level", "unknown")) == "high":
        penalties -= 9
        notes.append("news sentiment risk is high")
        notes_zh.append("新闻情绪风险高")
    if str(row.get("analyst_risk_level", "unknown")) == "high":
        penalties -= 8
        notes.append("analyst expectations are weak")
        notes_zh.append("分析师预期偏弱")
    if str(row.get("valuation_risk_level", "unknown")) == "high":
        penalties -= 8
        notes.append("valuation risk is high")
        notes_zh.append("估值风险高")
    if str(row["fundamental_quality"]) == "weak":
        penalties -= 8
        notes.append("fundamentals are weak")
        notes_zh.append("基本面偏弱")
    if float(row.get("sector_score", 50.0)) <= 40:
        penalties -= 7
        notes.append("sector context is weak")
        notes_zh.append("板块环境偏弱")

    if not notes:
        return 5.0, "no major risk filter is blocking the setup", "没有主要风险过滤项阻挡当前结构"
    return penalties, ", ".join(notes), "，".join(notes_zh)


def _confidence_level(score: float) -> tuple[str, str]:
    if score >= 80:
        return "high", "高"
    if score >= 55:
        return "medium", "中等"
    return "low", "低"


def _build_risk_breakdown_summary(
    analysis: pd.DataFrame,
    focus_horizon: str,
) -> dict[str, object]:
    focus_matches = analysis[analysis["horizon"] == focus_horizon]
    focus = focus_matches.iloc[0] if not focus_matches.empty else analysis.iloc[0]

    technical_risk = _risk_from_positive_score(float(focus["technical_score"]))
    market_risk = _risk_from_positive_score(float(focus["market_score"]))
    relative_risk = _risk_from_positive_score(float(focus["relative_strength_score"]))
    sector_risk = _risk_from_positive_score(float(focus["sector_score"]))
    fundamental_risk = _fundamental_risk(str(focus["fundamental_quality"]))
    event_risk = _event_risk(str(focus["event_risk_level"]))
    sentiment_risk = _sentiment_risk(str(focus.get("sentiment_risk_level", "unknown")))
    analyst_risk = _analyst_risk(str(focus.get("analyst_risk_level", "unknown")))
    valuation_risk = _valuation_risk(str(focus.get("valuation_risk_level", "unknown")))
    data_risk = _data_risk(_warning_count(focus))

    weighted_score = (
        technical_risk[0] * 0.18
        + market_risk[0] * 0.15
        + relative_risk[0] * 0.11
        + sector_risk[0] * 0.09
        + fundamental_risk[0] * 0.12
        + event_risk[0] * 0.10
        + sentiment_risk[0] * 0.06
        + analyst_risk[0] * 0.05
        + valuation_risk[0] * 0.06
        + data_risk[0] * 0.08
    )
    overall_level, overall_level_zh = _risk_level_from_score(weighted_score)
    note, note_zh = _risk_breakdown_note(
        technical_risk=technical_risk,
        market_risk=market_risk,
        relative_risk=relative_risk,
        sector_risk=sector_risk,
        fundamental_risk=fundamental_risk,
        event_risk=event_risk,
        sentiment_risk=sentiment_risk,
        analyst_risk=analyst_risk,
        valuation_risk=valuation_risk,
        data_risk=data_risk,
    )

    return {
        "overall_risk_score": round(float(weighted_score), 2),
        "overall_risk_level": overall_level,
        "overall_risk_level_zh": overall_level_zh,
        "technical_risk_level": technical_risk[1],
        "technical_risk_level_zh": technical_risk[2],
        "market_risk_level": market_risk[1],
        "market_risk_level_zh": market_risk[2],
        "relative_strength_risk_level": relative_risk[1],
        "relative_strength_risk_level_zh": relative_risk[2],
        "sector_risk_level": sector_risk[1],
        "sector_risk_level_zh": sector_risk[2],
        "fundamental_risk_level": fundamental_risk[1],
        "fundamental_risk_level_zh": fundamental_risk[2],
        "event_risk_breakdown_level": event_risk[1],
        "event_risk_breakdown_level_zh": event_risk[2],
        "sentiment_risk_breakdown_level": sentiment_risk[1],
        "sentiment_risk_breakdown_level_zh": sentiment_risk[2],
        "analyst_risk_breakdown_level": analyst_risk[1],
        "analyst_risk_breakdown_level_zh": analyst_risk[2],
        "valuation_risk_breakdown_level": valuation_risk[1],
        "valuation_risk_breakdown_level_zh": valuation_risk[2],
        "data_risk_level": data_risk[1],
        "data_risk_level_zh": data_risk[2],
        "risk_breakdown_note": note,
        "risk_breakdown_note_zh": note_zh,
    }


def _risk_from_positive_score(score: float) -> tuple[float, str, str]:
    if score >= 65:
        return 20.0, "low", "低"
    if score >= 45:
        return 50.0, "medium", "中"
    return 80.0, "high", "高"


def _fundamental_risk(quality: str) -> tuple[float, str, str]:
    if quality in {"strong", "good"}:
        return 20.0, "low", "低"
    if quality in {"neutral", "unknown"}:
        return 50.0, "medium", "中"
    return 80.0, "high", "高"


def _event_risk(level: str) -> tuple[float, str, str]:
    if level == "low":
        return 20.0, "low", "低"
    if level in {"medium", "unknown"}:
        return 50.0, "medium", "中"
    return 85.0, "high", "高"


def _sentiment_risk(level: str) -> tuple[float, str, str]:
    if level == "low":
        return 20.0, "low", "低"
    if level in {"medium", "unknown"}:
        return 50.0, "medium", "中"
    return 85.0, "high", "高"


def _analyst_risk(level: str) -> tuple[float, str, str]:
    if level == "low":
        return 20.0, "low", "低"
    if level in {"medium", "unknown"}:
        return 50.0, "medium", "中"
    return 85.0, "high", "高"


def _valuation_risk(level: str) -> tuple[float, str, str]:
    if level == "low":
        return 20.0, "low", "低"
    if level in {"medium", "unknown"}:
        return 50.0, "medium", "中"
    return 85.0, "high", "高"


def _data_risk(warning_count: int) -> tuple[float, str, str]:
    if warning_count == 0:
        return 15.0, "low", "低"
    if warning_count <= 2:
        return 55.0, "medium", "中"
    return 80.0, "high", "高"


def _risk_level_from_score(score: float) -> tuple[str, str]:
    if score < 35:
        return "low", "低"
    if score < 65:
        return "medium", "中"
    return "high", "高"


def _risk_breakdown_note(
    technical_risk: tuple[float, str, str],
    market_risk: tuple[float, str, str],
    relative_risk: tuple[float, str, str],
    sector_risk: tuple[float, str, str],
    fundamental_risk: tuple[float, str, str],
    event_risk: tuple[float, str, str],
    sentiment_risk: tuple[float, str, str],
    analyst_risk: tuple[float, str, str],
    valuation_risk: tuple[float, str, str],
    data_risk: tuple[float, str, str],
) -> tuple[str, str]:
    items = [
        ("technical", "技术", technical_risk),
        ("market", "大盘", market_risk),
        ("relative strength", "相对强弱", relative_risk),
        ("sector", "板块", sector_risk),
        ("fundamental", "基本面", fundamental_risk),
        ("event", "事件", event_risk),
        ("news sentiment", "新闻情绪", sentiment_risk),
        ("analyst expectations", "分析师预期", analyst_risk),
        ("valuation", "估值", valuation_risk),
        ("data", "数据", data_risk),
    ]
    high_items = [(en, zh) for en, zh, risk in items if risk[1] == "high"]
    medium_items = [(en, zh) for en, zh, risk in items if risk[1] == "medium"]
    if high_items:
        en_text = ", ".join(en for en, _ in high_items)
        zh_text = "、".join(zh for _, zh in high_items)
        return (
            f"Main risk comes from {en_text}.",
            f"主要风险来自{zh_text}。",
        )
    if medium_items:
        en_text = ", ".join(en for en, _ in medium_items)
        zh_text = "、".join(zh for _, zh in medium_items)
        return (
            f"No high-risk layer, but {en_text} risk should be monitored.",
            f"没有高风险层，但需要关注{zh_text}风险。",
        )
    return (
        "No major risk layer is elevated.",
        "当前没有明显升高的主要风险层。",
    )


def _build_data_quality_summary(analysis: pd.DataFrame) -> dict[str, object]:
    first = analysis.iloc[0]
    score = 100.0
    fallback_count = 0
    notes: list[str] = []
    notes_zh: list[str] = []

    price_status = _price_data_quality_status(first)
    score -= price_status[2]
    fallback_count += price_status[3]
    notes.append(price_status[4])
    notes_zh.append(price_status[5])

    market_status = _warning_based_data_status(
        warning=str(first.get("market_warning", "")),
        warning_penalty=15.0,
        ok_note="market benchmark data is available",
        ok_note_zh="大盘基准数据可用",
        fallback_note="market data used neutral fallback",
        fallback_note_zh="大盘数据使用中性回退",
    )
    score -= market_status[2]
    fallback_count += market_status[3]
    notes.append(market_status[4])
    notes_zh.append(market_status[5])

    relative_status = _warning_based_data_status(
        warning=str(first.get("relative_strength_warning", "")),
        warning_penalty=10.0,
        ok_note="relative strength benchmark data is available",
        ok_note_zh="相对强弱基准数据可用",
        fallback_note="relative strength data used neutral fallback",
        fallback_note_zh="相对强弱数据使用中性回退",
    )
    score -= relative_status[2]
    fallback_count += relative_status[3]
    notes.append(relative_status[4])
    notes_zh.append(relative_status[5])

    sector_status = _sector_data_quality_status(first)
    score -= sector_status[2]
    fallback_count += sector_status[3]
    notes.append(sector_status[4])
    notes_zh.append(sector_status[5])

    fundamental_status = _fundamental_data_quality_status(first)
    score -= fundamental_status[2]
    fallback_count += fundamental_status[3]
    notes.append(fundamental_status[4])
    notes_zh.append(fundamental_status[5])

    event_status = _event_data_quality_status(first)
    score -= event_status[2]
    fallback_count += event_status[3]
    notes.append(event_status[4])
    notes_zh.append(event_status[5])

    sentiment_status = _sentiment_data_quality_status(first)
    score -= sentiment_status[2]
    fallback_count += sentiment_status[3]
    notes.append(sentiment_status[4])
    notes_zh.append(sentiment_status[5])

    analyst_status = _analyst_data_quality_status(first)
    score -= analyst_status[2]
    fallback_count += analyst_status[3]
    notes.append(analyst_status[4])
    notes_zh.append(analyst_status[5])

    valuation_status = _valuation_data_quality_status(first)
    score -= valuation_status[2]
    fallback_count += valuation_status[3]
    notes.append(valuation_status[4])
    notes_zh.append(valuation_status[5])

    final_score = round(float(max(0.0, min(score, 100.0))), 2)
    level, level_zh = _data_quality_level(final_score)
    repair_summary = _data_quality_repair_summary(
        [
            ("price", "价格", price_status),
            ("market", "大盘", market_status),
            ("relative_strength", "相对强弱", relative_status),
            ("sector", "板块", sector_status),
            ("fundamental", "基本面", fundamental_status),
            ("event", "事件", event_status),
            ("sentiment", "新闻情绪", sentiment_status),
            ("analyst", "分析师", analyst_status),
            ("valuation", "估值", valuation_status),
        ]
    )
    return {
        "data_quality_score": final_score,
        "data_quality_level": level,
        "data_quality_level_zh": level_zh,
        **repair_summary,
        "price_data_status": price_status[0],
        "price_data_status_zh": price_status[1],
        "market_data_status": market_status[0],
        "market_data_status_zh": market_status[1],
        "relative_strength_data_status": relative_status[0],
        "relative_strength_data_status_zh": relative_status[1],
        "sector_data_status": sector_status[0],
        "sector_data_status_zh": sector_status[1],
        "fundamental_data_status": fundamental_status[0],
        "fundamental_data_status_zh": fundamental_status[1],
        "event_data_status": event_status[0],
        "event_data_status_zh": event_status[1],
        "sentiment_data_status": sentiment_status[0],
        "sentiment_data_status_zh": sentiment_status[1],
        "analyst_data_status": analyst_status[0],
        "analyst_data_status_zh": analyst_status[1],
        "valuation_data_status": valuation_status[0],
        "valuation_data_status_zh": valuation_status[1],
        "missing_fallback_count": fallback_count,
        "data_quality_note": "; ".join(notes) + ".",
        "data_quality_note_zh": "；".join(notes_zh) + "。",
    }


def _data_quality_repair_summary(
    layers: list[tuple[str, str, tuple[str, str, float, int, str, str]]]
) -> dict[str, object]:
    weak_layers = [
        {
            "name": name,
            "name_zh": name_zh,
            "status": status[0],
            "status_zh": status[1],
            "penalty": float(status[2]),
            "fallback_count": int(status[3]),
            "repair": _data_layer_repair_action(name),
            "repair_zh": _data_layer_repair_action_zh(name),
        }
        for name, name_zh, status in layers
        if float(status[2]) > 0.0 or status[0] != "ok"
    ]
    weak_layers.sort(
        key=lambda item: (
            float(item["penalty"]),
            int(item["fallback_count"]),
            str(item["name"]),
        ),
        reverse=True,
    )
    if not weak_layers:
        return {
            "data_quality_weakest_layer": "none",
            "data_quality_weakest_layer_zh": "无",
            "data_quality_weak_layers": "none",
            "data_quality_weak_layers_zh": "无",
            "data_quality_repair_actions": "none",
            "data_quality_repair_actions_zh": "无",
            "data_quality_repair_priority": "none",
            "data_quality_repair_priority_zh": "无",
        }

    weakest = weak_layers[0]
    fallback_count = sum(int(item["fallback_count"]) for item in weak_layers)
    max_penalty = float(weakest["penalty"])
    if max_penalty >= 12.0 or fallback_count >= 2:
        priority, priority_zh = "high", "高"
    elif max_penalty >= 5.0:
        priority, priority_zh = "medium", "中"
    else:
        priority, priority_zh = "low", "低"
    top_layers = weak_layers[:3]
    return {
        "data_quality_weakest_layer": str(weakest["name"]),
        "data_quality_weakest_layer_zh": str(weakest["name_zh"]),
        "data_quality_weak_layers": "; ".join(
            f"{item['name']}({item['status']}, -{item['penalty']:.0f})"
            for item in top_layers
        ),
        "data_quality_weak_layers_zh": "；".join(
            f"{item['name_zh']}（{item['status_zh']}，扣{item['penalty']:.0f}分）"
            for item in top_layers
        ),
        "data_quality_repair_actions": "; ".join(str(item["repair"]) for item in top_layers),
        "data_quality_repair_actions_zh": "；".join(
            str(item["repair_zh"]) for item in top_layers
        ),
        "data_quality_repair_priority": priority,
        "data_quality_repair_priority_zh": priority_zh,
    }


def _data_layer_repair_action(layer: str) -> str:
    actions = {
        "price": "Extend the price period to 5y or refresh the OHLCV cache.",
        "market": "Refresh SPY, QQQ, and VIX benchmark data.",
        "relative_strength": "Refresh SPY and QQQ data so relative strength can be compared.",
        "sector": "Refresh the sector ETF data or improve sector/industry mapping.",
        "fundamental": "Refresh the company snapshot and fundamental fields.",
        "event": "Refresh earnings/event date data before trusting new-entry signals.",
        "sentiment": "Refresh recent news headlines or use a stronger news source.",
        "analyst": "Refresh analyst target and recommendation fields.",
        "valuation": "Refresh valuation fields such as forward PE, PEG, and market cap.",
    }
    return actions.get(layer, "Refresh this missing data layer.")


def _data_layer_repair_action_zh(layer: str) -> str:
    actions = {
        "price": "把价格周期延长到5年，或刷新OHLCV价格缓存。",
        "market": "刷新SPY、QQQ和VIX大盘基准数据。",
        "relative_strength": "刷新SPY和QQQ数据，用于比较相对强弱。",
        "sector": "刷新板块ETF数据，或完善板块/行业映射。",
        "fundamental": "刷新公司快照和基本面字段。",
        "event": "刷新财报/事件日期数据，再信任新买点信号。",
        "sentiment": "刷新近期新闻标题，或接入更强的新闻来源。",
        "analyst": "刷新分析师目标价和评级字段。",
        "valuation": "刷新预期市盈率、PEG、市值等估值字段。",
    }
    return actions.get(layer, "刷新这个缺失的数据层。")


def _price_data_quality_status(row: pd.Series) -> tuple[str, str, float, int, str, str]:
    history_days = int(row["history_days"])
    health_penalty = float(row.get("price_health_penalty", 0.0) or 0.0)
    health_issue_count = int(row.get("price_health_issue_count", 0) or 0)
    health_note = str(row.get("price_health_note", "") or "").strip()
    health_note_zh = str(row.get("price_health_note_zh", "") or "").strip()
    if history_days >= 756:
        status, status_zh, base_penalty, base_fallback, note, note_zh = (
            "ok",
            "可用",
            0.0,
            0,
            f"price history is strong ({history_days} trading rows)",
            f"价格历史充足（{history_days}个交易日）",
        )
    elif history_days >= 252:
        status, status_zh, base_penalty, base_fallback, note, note_zh = (
            "partial",
            "部分可用",
            8.0,
            0,
            f"price history is usable but shorter than preferred ({history_days} trading rows)",
            f"价格历史可用但不算很长（{history_days}个交易日）",
        )
    elif history_days >= 60:
        status, status_zh, base_penalty, base_fallback, note, note_zh = (
            "weak",
            "偏弱",
            18.0,
            1,
            f"price history is limited ({history_days} trading rows)",
            f"价格历史偏少（{history_days}个交易日）",
        )
    else:
        status, status_zh, base_penalty, base_fallback, note, note_zh = (
            "fallback",
            "中性回退",
            35.0,
            1,
            f"price history is too short ({history_days} trading rows)",
            f"价格历史太短（{history_days}个交易日）",
        )

    if health_penalty > 0:
        if health_penalty >= 18:
            status, status_zh = "fallback", "中性回退"
        elif health_penalty >= 10 and status == "ok":
            status, status_zh = "weak", "偏弱"
        elif health_penalty > 0 and status == "ok":
            status, status_zh = "partial", "部分可用"
        note = f"{note}; {health_note}" if health_note else note
        note_zh = f"{note_zh}；{health_note_zh}" if health_note_zh else note_zh

    return (
        status,
        status_zh,
        base_penalty + health_penalty,
        base_fallback + (1 if health_issue_count > 0 else 0),
        note,
        note_zh,
    )


def _warning_based_data_status(
    warning: str,
    warning_penalty: float,
    ok_note: str,
    ok_note_zh: str,
    fallback_note: str,
    fallback_note_zh: str,
) -> tuple[str, str, float, int, str, str]:
    if warning.strip():
        return (
            "fallback",
            "中性回退",
            warning_penalty,
            1,
            fallback_note,
            fallback_note_zh,
        )
    return "ok", "可用", 0.0, 0, ok_note, ok_note_zh


def _sector_data_quality_status(row: pd.Series) -> tuple[str, str, float, int, str, str]:
    warning = str(row.get("sector_warning", "")).strip()
    status = str(row.get("sector_status", "unknown"))
    etf = str(row.get("sector_etf", "")).strip()
    if warning or status == "unknown" or not etf:
        return (
            "fallback",
            "中性回退",
            12.0,
            1,
            "sector context used neutral fallback",
            "板块环境使用中性回退",
        )
    if float(row.get("sector_score", 50.0)) == 50.0:
        return (
            "partial",
            "部分可用",
            5.0,
            0,
            "sector ETF is available but signal is neutral",
            "板块ETF可用，但信号偏中性",
        )
    return (
        "ok",
        "可用",
        0.0,
        0,
        f"sector ETF data is available ({etf})",
        f"板块ETF数据可用（{etf}）",
    )


def _fundamental_data_quality_status(row: pd.Series) -> tuple[str, str, float, int, str, str]:
    warning = str(row.get("fundamental_warning", "")).strip()
    quality = str(row.get("fundamental_quality", "unknown"))
    coverage = float(row.get("fundamental_data_coverage", 0.0))
    if warning or quality == "unknown":
        return (
            "fallback",
            "中性回退",
            15.0,
            1,
            "fundamental data used neutral fallback",
            "基本面数据使用中性回退",
        )
    if coverage < 0.25:
        return (
            "weak",
            "偏弱",
            14.0,
            1,
            f"fundamental data coverage is sparse ({coverage:.0%})",
            f"基本面数据覆盖不足（{coverage:.0%}）",
        )
    if coverage < 0.75:
        return (
            "partial",
            "部分可用",
            7.0,
            0,
            f"fundamental data coverage is partial ({coverage:.0%})",
            f"基本面数据覆盖部分可用（{coverage:.0%}）",
        )
    return (
        "ok",
        "可用",
        0.0,
        0,
        f"fundamental data coverage is good ({coverage:.0%})",
        f"基本面数据覆盖较好（{coverage:.0%}）",
    )


def _event_data_quality_status(row: pd.Series) -> tuple[str, str, float, int, str, str]:
    warning = str(row.get("event_risk_warning", "")).strip()
    level = str(row.get("event_risk_level", "unknown"))
    next_earnings = str(row.get("next_earnings_date", "")).strip()
    if warning or level == "unknown":
        return (
            "fallback",
            "中性回退",
            10.0,
            1,
            "event date data used neutral fallback",
            "事件日期数据使用中性回退",
        )
    if not next_earnings:
        return (
            "partial",
            "部分可用",
            5.0,
            0,
            "event risk is available but next earnings date is missing",
            "事件风险可用，但下一次财报日期缺失",
        )
    return (
        "ok",
        "可用",
        0.0,
        0,
        f"event date data is available ({next_earnings})",
        f"事件日期数据可用（{next_earnings}）",
    )


def _sentiment_data_quality_status(row: pd.Series) -> tuple[str, str, float, int, str, str]:
    warning = str(row.get("sentiment_warning", "")).strip()
    level = str(row.get("sentiment_risk_level", "unknown"))
    titles_used = int(row.get("sentiment_titles_used", 0) or 0)
    if warning or level == "unknown":
        return (
            "fallback",
            "中性回退",
            8.0,
            1,
            "news sentiment data used neutral fallback",
            "新闻情绪数据使用中性回退",
        )
    if titles_used <= 0:
        return (
            "partial",
            "部分可用",
            5.0,
            0,
            "news sentiment context is available but no headlines were used",
            "新闻情绪环境可用，但没有使用新闻标题",
        )
    return (
        "ok",
        "可用",
        0.0,
        0,
        f"news sentiment data is available ({titles_used} headline(s))",
        f"新闻情绪数据可用（{titles_used}条标题）",
    )


def _analyst_data_quality_status(row: pd.Series) -> tuple[str, str, float, int, str, str]:
    warning = str(row.get("analyst_warning", "")).strip()
    level = str(row.get("analyst_risk_level", "unknown"))
    coverage = float(row.get("analyst_data_coverage", 0.0) or 0.0)
    if warning or level == "unknown":
        return (
            "fallback",
            "中性回退",
            8.0,
            1,
            "analyst expectation data used neutral fallback",
            "分析师预期数据使用中性回退",
        )
    if coverage < 0.50:
        return (
            "partial",
            "部分可用",
            5.0,
            0,
            f"analyst expectation data coverage is partial ({coverage:.0%})",
            f"分析师预期数据覆盖部分可用（{coverage:.0%}）",
        )
    return (
        "ok",
        "可用",
        0.0,
        0,
        f"analyst expectation data is available ({coverage:.0%} coverage)",
        f"分析师预期数据可用（覆盖率{coverage:.0%}）",
    )


def _valuation_data_quality_status(row: pd.Series) -> tuple[str, str, float, int, str, str]:
    warning = str(row.get("valuation_warning", "")).strip()
    level = str(row.get("valuation_risk_level", "unknown"))
    coverage = float(row.get("valuation_data_coverage", 0.0) or 0.0)
    if warning or level == "unknown":
        return (
            "fallback",
            "中性回退",
            8.0,
            1,
            "valuation data used neutral fallback",
            "估值数据使用中性回退",
        )
    if coverage < 0.50:
        return (
            "partial",
            "部分可用",
            5.0,
            0,
            f"valuation data coverage is partial ({coverage:.0%})",
            f"估值数据覆盖部分可用（{coverage:.0%}）",
        )
    return (
        "ok",
        "可用",
        0.0,
        0,
        f"valuation data is available ({coverage:.0%} coverage)",
        f"估值数据可用（覆盖率{coverage:.0%}）",
    )


def _data_quality_level(score: float) -> tuple[str, str]:
    if score >= 80:
        return "high", "高"
    if score >= 55:
        return "medium", "中等"
    return "low", "低"
