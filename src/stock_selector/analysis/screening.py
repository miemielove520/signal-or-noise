"""High-probability quality gates, probability/threshold calibration and watchlist plans."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..screening_config import ScreeningThresholds

from ._common import (
    _append_reason,
    _clamp_score,
    _coerce_bool,
    _format_number,
    _format_percent,
    _is_finite,
    _row_float,
    _safe_float_like,
    _safe_int_like,
)


def _add_high_probability_screening(
    analysis: pd.DataFrame,
    thresholds: ScreeningThresholds | None = None,
) -> pd.DataFrame:
    thresholds = thresholds or ScreeningThresholds()
    rows = [_high_probability_screening_for_row(row, thresholds) for _, row in analysis.iterrows()]
    result = analysis.copy()
    for key in rows[0]:
        result[key] = [row[key] for row in rows]
    return result


def _high_probability_screening_for_row(
    row: pd.Series,
    thresholds: ScreeningThresholds | None = None,
) -> dict[str, object]:
    thresholds = thresholds or ScreeningThresholds()
    (
        trade_count,
        win_rate,
        average_return,
        stop_hit_rate,
        entry_label,
        entry_label_zh,
    ) = _selected_entry_backtest(row)
    if int(trade_count) < thresholds.backtest_sample_min:
        # Fall back to pooled breakout+pullback evidence when the selected entry
        # type alone is under-sampled, so scarce history is judged on all similar
        # trades instead of hard-failing on a nearly-empty bucket.
        pooled = _pooled_entry_backtest(row)
        if pooled is not None and pooled[0] > int(trade_count):
            trade_count, win_rate, average_return, stop_hit_rate = pooled
            entry_label = "combined entry"
            entry_label_zh = "综合买点(样本合并)"
    liquidity_filter = _liquidity_filter_result(row, thresholds)
    entry_readiness = _entry_readiness_filter_result(row)
    trade_plan_quality = _trade_plan_quality_filter_result(row)
    fail_reasons: list[str] = []
    fail_reasons_zh: list[str] = []

    _add_gate_failure(
        fail_reasons,
        fail_reasons_zh,
        float(row["data_quality_score"]) < thresholds.data_quality_min,
        "data quality insufficient",
        "数据质量不足",
    )
    _add_gate_failure(
        fail_reasons,
        fail_reasons_zh,
        float(row["confidence_score"]) < thresholds.confidence_min,
        "confidence too low",
        "置信度不足",
    )
    _add_gate_failure(
        fail_reasons,
        fail_reasons_zh,
        str(row["overall_risk_level"]) == "high",
        "overall risk high",
        "综合风险过高",
    )
    _add_gate_failure(
        fail_reasons,
        fail_reasons_zh,
        float(row["market_score"]) < thresholds.market_score_min,
        "market not supportive",
        "大盘环境不够支持",
    )
    _add_gate_failure(
        fail_reasons,
        fail_reasons_zh,
        float(row["relative_strength_score"]) < thresholds.relative_strength_min,
        "relative strength weak",
        "相对强弱不足",
    )
    _add_gate_failure(
        fail_reasons,
        fail_reasons_zh,
        float(row["signal_score"]) < thresholds.signal_score_min,
        "signal score too low",
        "信号分数不足",
    )
    _add_gate_failure(
        fail_reasons,
        fail_reasons_zh,
        int(trade_count) < thresholds.backtest_sample_min,
        f"{entry_label} backtest sample too small",
        f"{entry_label_zh}回测样本不足",
    )
    _add_gate_failure(
        fail_reasons,
        fail_reasons_zh,
        not _is_finite(win_rate) or float(win_rate) < thresholds.backtest_win_rate_min,
        f"{entry_label} backtest win rate too low",
        f"{entry_label_zh}回测胜率不足",
    )
    _add_gate_failure(
        fail_reasons,
        fail_reasons_zh,
        not _is_finite(average_return)
        or float(average_return) <= thresholds.backtest_average_return_min,
        f"{entry_label} backtest average return not positive",
        f"{entry_label_zh}回测平均收益不是正数",
    )
    _add_gate_failure(
        fail_reasons,
        fail_reasons_zh,
        _is_finite(stop_hit_rate) and float(stop_hit_rate) > thresholds.max_backtest_stop_hit_rate,
        f"{entry_label} backtest stop-hit rate too high",
        f"{entry_label_zh}回测止损命中率过高",
    )
    _add_gate_failure(
        fail_reasons,
        fail_reasons_zh,
        float(row.get("backtest_trust_score", 0.0)) < thresholds.backtest_trust_score_min,
        "backtest trust score too low",
        "回测可信度分不足",
    )
    _add_gate_failure(
        fail_reasons,
        fail_reasons_zh,
        float(row.get("recent_backtest_score", 0.0)) < thresholds.recent_backtest_score_min,
        "recent backtest strength too weak",
        "近期回测强度不足",
    )
    _add_gate_failure(
        fail_reasons,
        fail_reasons_zh,
        float(row.get("backtest_decay_score", 0.0)) < thresholds.backtest_decay_score_min,
        "backtest decay check failed",
        "回测衰退检查未通过",
    )
    _add_gate_failure(
        fail_reasons,
        fail_reasons_zh,
        not liquidity_filter["passed"],
        str(liquidity_filter["reason"]),
        str(liquidity_filter["reason_zh"]),
    )
    _add_gate_failure(
        fail_reasons,
        fail_reasons_zh,
        not entry_readiness["passed"],
        str(entry_readiness["reason"]),
        str(entry_readiness["reason_zh"]),
    )
    _add_gate_failure(
        fail_reasons,
        fail_reasons_zh,
        not trade_plan_quality["passed"],
        str(trade_plan_quality["reason"]),
        str(trade_plan_quality["reason_zh"]),
    )
    _add_gate_failure(
        fail_reasons,
        fail_reasons_zh,
        str(row["horizon"]) == "long"
        and float(row["fundamental_score"]) < thresholds.long_fundamental_score_min,
        "long-term fundamental score too low",
        "长期基本面分数不足",
    )
    _add_gate_failure(
        fail_reasons,
        fail_reasons_zh,
        str(row["horizon"]) in {"medium", "long"}
        and float(row["sector_score"]) < thresholds.medium_long_sector_score_min,
        "sector context weak",
        "板块环境偏弱",
    )
    _add_gate_failure(
        fail_reasons,
        fail_reasons_zh,
        str(row["event_risk_level"]) == "high",
        "event risk high",
        "事件风险过高",
    )
    _add_gate_failure(
        fail_reasons,
        fail_reasons_zh,
        _coerce_bool(row.get("event_block_new_entries", False)),
        "event window blocks new entries",
        "事件窗口阻止新入场",
    )
    _add_gate_failure(
        fail_reasons,
        fail_reasons_zh,
        str(row.get("sentiment_risk_level", "unknown")) == "high",
        "news sentiment risk high",
        "新闻情绪风险过高",
    )
    _add_gate_failure(
        fail_reasons,
        fail_reasons_zh,
        _coerce_bool(row.get("sentiment_block_new_entries", False)),
        "news sentiment blocks new entries",
        "新闻情绪阻止新入场",
    )
    _add_gate_failure(
        fail_reasons,
        fail_reasons_zh,
        str(row.get("analyst_risk_level", "unknown")) == "high",
        "analyst expectation risk high",
        "分析师预期风险过高",
    )
    _add_gate_failure(
        fail_reasons,
        fail_reasons_zh,
        _coerce_bool(row.get("analyst_block_new_entries", False)),
        "analyst expectations block new entries",
        "分析师预期阻止新入场",
    )
    _add_gate_failure(
        fail_reasons,
        fail_reasons_zh,
        str(row.get("valuation_risk_level", "unknown")) == "high",
        "valuation risk high",
        "估值风险过高",
    )
    _add_gate_failure(
        fail_reasons,
        fail_reasons_zh,
        _coerce_bool(row.get("valuation_block_new_entries", False)),
        "valuation blocks new entries",
        "估值阻止新入场",
    )

    score_row = row.copy()
    score_row["entry_readiness_score"] = entry_readiness["score"]
    score_row["trade_plan_quality_score"] = trade_plan_quality["score"]
    score = _high_probability_score(
        row=score_row,
        trade_count=trade_count,
        win_rate=win_rate,
        average_return=average_return,
        fail_count=len(fail_reasons),
    )
    level, level_zh = _high_probability_level(score)
    probability_calibration = _calibrated_win_probability_profile(
        row=score_row,
        trade_count=trade_count,
        win_rate=win_rate,
        stop_hit_rate=stop_hit_rate,
        average_return=average_return,
        high_probability_score=score,
    )
    evidence = _entry_evidence_profile(
        trade_count=trade_count,
        win_rate=win_rate,
        average_return=average_return,
    )
    passed = len(fail_reasons) == 0
    if passed:
        screening_action = "high_probability_watchlist_candidate"
        screening_action_zh = "高概率观察候选"
    else:
        screening_action = "not_high_probability_setup_now"
        screening_action_zh = "当前不是高概率机会"

    reason_text = "; ".join(fail_reasons) if fail_reasons else "all strict quality gates passed"
    reason_text_zh = "；".join(fail_reasons_zh) if fail_reasons_zh else "所有严格质量门槛通过"
    return {
        "screening_action": screening_action,
        "screening_action_zh": screening_action_zh,
        "high_probability_score": score,
        "high_probability_level": level,
        "high_probability_level_zh": level_zh,
        "calibrated_win_probability": probability_calibration["calibrated_win_probability"],
        "calibrated_probability_level": probability_calibration[
            "calibrated_probability_level"
        ],
        "calibrated_probability_level_zh": probability_calibration[
            "calibrated_probability_level_zh"
        ],
        "calibrated_probability_confidence": probability_calibration[
            "calibrated_probability_confidence"
        ],
        "calibrated_probability_confidence_level": probability_calibration[
            "calibrated_probability_confidence_level"
        ],
        "calibrated_probability_confidence_level_zh": probability_calibration[
            "calibrated_probability_confidence_level_zh"
        ],
        "calibrated_probability_note": probability_calibration[
            "calibrated_probability_note"
        ],
        "calibrated_probability_note_zh": probability_calibration[
            "calibrated_probability_note_zh"
        ],
        "quality_gate_passed": bool(passed),
        "quality_gate_fail_reasons": reason_text,
        "quality_gate_fail_reasons_zh": reason_text_zh,
        "screening_backtest_entry_type": entry_label,
        "screening_backtest_trade_count": int(trade_count),
        "screening_backtest_win_rate": float(win_rate) if _is_finite(win_rate) else np.nan,
        "screening_backtest_stop_hit_rate": float(stop_hit_rate)
        if _is_finite(stop_hit_rate)
        else np.nan,
        "screening_backtest_average_return": float(average_return)
        if _is_finite(average_return)
        else np.nan,
        "sample_confidence_level": evidence["sample_confidence_level"],
        "sample_confidence_level_zh": evidence["sample_confidence_level_zh"],
        "evidence_strength": evidence["evidence_strength"],
        "evidence_strength_zh": evidence["evidence_strength_zh"],
        "evidence_note": evidence["evidence_note"],
        "evidence_note_zh": evidence["evidence_note_zh"],
        "liquidity_filter_passed": bool(liquidity_filter["passed"]),
        "liquidity_filter_reason": liquidity_filter["reason"],
        "liquidity_filter_reason_zh": liquidity_filter["reason_zh"],
        "entry_readiness_gate_passed": bool(entry_readiness["passed"]),
        "entry_readiness_status": entry_readiness["status"],
        "entry_readiness_status_zh": entry_readiness["status_zh"],
        "entry_readiness_score": entry_readiness["score"],
        "entry_readiness_reason": entry_readiness["reason"],
        "entry_readiness_reason_zh": entry_readiness["reason_zh"],
        "entry_readiness_note": entry_readiness["note"],
        "entry_readiness_note_zh": entry_readiness["note_zh"],
        "trade_plan_quality_gate_passed": bool(trade_plan_quality["passed"]),
        "trade_plan_quality_status": trade_plan_quality["status"],
        "trade_plan_quality_status_zh": trade_plan_quality["status_zh"],
        "trade_plan_quality_score": trade_plan_quality["score"],
        "trade_plan_quality_reason": trade_plan_quality["reason"],
        "trade_plan_quality_reason_zh": trade_plan_quality["reason_zh"],
        "trade_plan_quality_note": trade_plan_quality["note"],
        "trade_plan_quality_note_zh": trade_plan_quality["note_zh"],
    }


def _add_gate_failure(
    fail_reasons: list[str],
    fail_reasons_zh: list[str],
    condition: bool,
    reason: str,
    reason_zh: str,
) -> None:
    if condition:
        fail_reasons.append(reason)
        fail_reasons_zh.append(reason_zh)


def _liquidity_filter_result(
    row: pd.Series,
    thresholds: ScreeningThresholds,
) -> dict[str, object]:
    avg_dollar_volume = float(row.get("backtest_avg_dollar_volume", np.nan))
    slippage_pct = float(row.get("backtest_slippage_pct", np.nan))
    liquidity_label = str(row.get("backtest_liquidity_label", "unknown_liquidity"))
    liquidity_label_zh = str(row.get("backtest_liquidity_label_zh", "流动性未知"))
    reasons: list[str] = []
    reasons_zh: list[str] = []

    if not _is_finite(avg_dollar_volume):
        reasons.append("average dollar volume unavailable")
        reasons_zh.append("平均成交额不可用")
    elif avg_dollar_volume < thresholds.min_backtest_avg_dollar_volume:
        reasons.append(
            f"average dollar volume below ${thresholds.min_backtest_avg_dollar_volume:,.0f}"
        )
        reasons_zh.append(
            f"平均成交额低于${thresholds.min_backtest_avg_dollar_volume:,.0f}"
        )

    if not _is_finite(slippage_pct):
        reasons.append("dynamic slippage unavailable")
        reasons_zh.append("动态滑点不可用")
    elif slippage_pct > thresholds.max_backtest_slippage_pct:
        reasons.append(
            f"dynamic slippage above {thresholds.max_backtest_slippage_pct:.2%}"
        )
        reasons_zh.append(
            f"动态滑点高于{thresholds.max_backtest_slippage_pct:.2%}"
        )

    if liquidity_label in {"very_low_liquidity", "unknown_liquidity"}:
        reasons.append(f"liquidity label is {liquidity_label}")
        reasons_zh.append(f"流动性等级为{liquidity_label_zh}")

    if not reasons:
        return {
            "passed": True,
            "reason": "liquidity filter passed",
            "reason_zh": "流动性过滤通过",
        }
    return {
        "passed": False,
        "reason": "liquidity filter failed: " + "; ".join(reasons),
        "reason_zh": "流动性过滤未通过：" + "；".join(reasons_zh),
    }


def _entry_readiness_filter_result(row: pd.Series) -> dict[str, object]:
    chase_status = str(row.get("chase_status", "unknown"))
    distance = float(row.get("entry_distance_pct", np.nan))
    if chase_status in {"entry_allowed_if_price_holds", "pullback_entry_active"}:
        status = "executable_now"
        status_zh = "当前接近可执行买点"
        score = 90.0 if abs(distance) <= 0.03 else 82.0
        passed = True
        reason = "entry is executable"
        reason_zh = "买点当前可执行"
        note = "Price is close enough to an active entry zone."
        note_zh = "价格足够接近当前可执行买点区域。"
    elif chase_status == "do_not_chase_wait_for_breakout":
        status = "wait_for_breakout"
        status_zh = "等待突破触发"
        score = 55.0
        passed = False
        reason = "entry trigger not reached"
        reason_zh = "买点触发尚未到达"
        note = "Price has not reached the breakout trigger; avoid buying early."
        note_zh = "价格尚未到达突破触发位，避免提前买入。"
    elif chase_status == "do_not_chase_wait_for_pullback":
        status = "wait_for_pullback"
        status_zh = "等待回调触发"
        score = 45.0
        passed = False
        reason = "pullback entry not reached"
        reason_zh = "回调买点尚未到达"
        note = "Price is above the preferred pullback zone; avoid chasing."
        note_zh = "价格高于理想回调区域，避免追高。"
    elif chase_status == "blocked_by_risk_filter":
        status = "blocked_by_risk"
        status_zh = "被风险过滤阻挡"
        score = 10.0
        passed = False
        reason = "entry blocked by risk filter"
        reason_zh = "买点被风险过滤阻挡"
        note = "A risk filter blocks new entries even if a price level exists."
        note_zh = "即使存在价格位置，风险过滤仍阻止新入场。"
    elif chase_status == "no_entry_trend_not_ready":
        status = "trend_not_ready"
        status_zh = "趋势未准备好"
        score = 20.0
        passed = False
        reason = "trend not ready for entry"
        reason_zh = "趋势尚未准备好"
        note = "The trend setup is not ready for a fresh entry."
        note_zh = "趋势结构尚未准备好，不适合新入场。"
    else:
        status = "not_executable"
        status_zh = "当前不可执行"
        score = 25.0
        passed = False
        reason = "entry is not executable now"
        reason_zh = "当前买点不可执行"
        note = "No active executable entry is available at the current price."
        note_zh = "当前价格没有可执行的新入场买点。"

    if _is_finite(distance):
        note = f"{note} Distance to planned entry={distance:.2%}."
        note_zh = f"{note_zh} 距离计划买点={distance:.2%}。"

    return {
        "passed": passed,
        "status": status,
        "status_zh": status_zh,
        "score": round(float(score), 2),
        "reason": reason,
        "reason_zh": reason_zh,
        "note": note,
        "note_zh": note_zh,
    }


def _trade_plan_quality_filter_result(row: pd.Series) -> dict[str, object]:
    entry_type = str(row.get("entry_type", ""))
    horizon = str(row.get("horizon", "short"))
    entry_price = float(row.get("entry_price", np.nan))
    stop_loss = float(row.get("stop_loss", np.nan))
    take_profit = float(row.get("take_profit", np.nan))
    risk_reward = float(row.get("risk_reward", np.nan))
    risk_per_share = float(row.get("risk_per_share", np.nan))

    if entry_type == "none":
        return _trade_plan_quality_result(
            passed=False,
            status="no_trade_plan",
            status_zh="没有可执行交易计划",
            score=15.0,
            reason="no executable trade plan",
            reason_zh="没有可执行交易计划",
            note="The current row has no actionable entry type.",
            note_zh="当前周期没有可执行入场类型。",
        )

    required_values = [entry_price, stop_loss, take_profit, risk_reward, risk_per_share]
    if not all(_is_finite(value) for value in required_values):
        return _trade_plan_quality_result(
            passed=False,
            status="missing_trade_plan",
            status_zh="交易计划数据缺失",
            score=20.0,
            reason="trade plan data missing",
            reason_zh="交易计划数据缺失",
            note="Entry, stop, target, or risk-reward data is unavailable.",
            note_zh="入场、止损、目标或盈亏比数据不可用。",
        )

    if entry_price <= 0 or risk_per_share <= 0 or stop_loss >= entry_price or take_profit <= entry_price:
        return _trade_plan_quality_result(
            passed=False,
            status="invalid_trade_geometry",
            status_zh="交易结构无效",
            score=15.0,
            reason="trade plan geometry invalid",
            reason_zh="交易计划结构无效",
            note="The entry, stop, and target prices do not form a valid long setup.",
            note_zh="入场、止损和目标价格没有形成有效的多头交易结构。",
        )

    stop_distance_pct = risk_per_share / entry_price
    max_stop_distance = {
        "short": 0.10,
        "medium": 0.15,
        "long": 0.25,
    }.get(horizon, 0.15)

    if risk_reward < 1.8:
        return _trade_plan_quality_result(
            passed=False,
            status="risk_reward_too_low",
            status_zh="盈亏比不足",
            score=35.0,
            reason="risk-reward too low",
            reason_zh="盈亏比不足",
            note=f"Risk-reward is {risk_reward:.2f}, below the minimum 1.80.",
            note_zh=f"盈亏比为{risk_reward:.2f}，低于最低要求1.80。",
        )
    if stop_distance_pct < 0.005:
        return _trade_plan_quality_result(
            passed=False,
            status="stop_too_tight",
            status_zh="止损过窄",
            score=40.0,
            reason="stop distance too tight",
            reason_zh="止损距离过窄",
            note=f"Stop distance is {stop_distance_pct:.2%}, which may be too easy to trigger.",
            note_zh=f"止损距离为{stop_distance_pct:.2%}，可能过于容易被触发。",
        )
    if stop_distance_pct > max_stop_distance:
        return _trade_plan_quality_result(
            passed=False,
            status="stop_too_wide",
            status_zh="止损过宽",
            score=35.0,
            reason="stop distance too wide",
            reason_zh="止损距离过宽",
            note=(
                f"Stop distance is {stop_distance_pct:.2%}, above the "
                f"{horizon} limit of {max_stop_distance:.2%}."
            ),
            note_zh=(
                f"止损距离为{stop_distance_pct:.2%}，高于{horizon}"
                f"周期限制{max_stop_distance:.2%}。"
            ),
        )

    reward_score = min(max((risk_reward - 1.5) / 2.0, 0.0), 1.0) * 45.0
    stop_score = (1.0 - min(stop_distance_pct / max_stop_distance, 1.0)) * 35.0
    score = 20.0 + reward_score + stop_score
    return _trade_plan_quality_result(
        passed=True,
        status="valid_trade_plan",
        status_zh="交易计划有效",
        score=min(score, 100.0),
        reason="trade plan quality passed",
        reason_zh="交易计划质量通过",
        note=(
            f"Risk-reward={risk_reward:.2f}; stop distance={stop_distance_pct:.2%}; "
            f"entry={entry_price:.4f}, stop={stop_loss:.4f}, target={take_profit:.4f}."
        ),
        note_zh=(
            f"盈亏比={risk_reward:.2f}；止损距离={stop_distance_pct:.2%}；"
            f"入场={entry_price:.4f}，止损={stop_loss:.4f}，目标={take_profit:.4f}。"
        ),
    )


def _trade_plan_quality_result(
    passed: bool,
    status: str,
    status_zh: str,
    score: float,
    reason: str,
    reason_zh: str,
    note: str,
    note_zh: str,
) -> dict[str, object]:
    return {
        "passed": passed,
        "status": status,
        "status_zh": status_zh,
        "score": round(float(max(0.0, min(score, 100.0))), 2),
        "reason": reason,
        "reason_zh": reason_zh,
        "note": note,
        "note_zh": note_zh,
    }


def _selected_entry_backtest(row: pd.Series) -> tuple[int, float, float, float, str, str]:
    action = str(row.get("action", ""))
    entry_type = str(row.get("entry_type", ""))
    if action == "entry_pullback_zone" or entry_type == "limit_pullback":
        return (
            int(row["pullback_trade_count"]),
            float(row["pullback_win_rate"]),
            float(row["pullback_average_return"]),
            float(row["pullback_stop_hit_rate"]),
            "pullback",
            "回调买点",
        )
    if entry_type in {"market_or_limit", "stop_limit_breakout"} or action == "watch_breakout_or_pullback":
        return (
            int(row["breakout_trade_count"]),
            float(row["breakout_win_rate"]),
            float(row["breakout_average_return"]),
            float(row["breakout_stop_hit_rate"]),
            "breakout",
            "突破买点",
        )

    breakout_score = _backtest_quality_score(
        int(row["breakout_trade_count"]),
        float(row["breakout_win_rate"]),
        float(row["breakout_average_return"]),
    )
    pullback_score = _backtest_quality_score(
        int(row["pullback_trade_count"]),
        float(row["pullback_win_rate"]),
        float(row["pullback_average_return"]),
    )
    if pullback_score > breakout_score:
        return (
            int(row["pullback_trade_count"]),
            float(row["pullback_win_rate"]),
            float(row["pullback_average_return"]),
            float(row["pullback_stop_hit_rate"]),
            "pullback",
            "回调买点",
        )
    return (
        int(row["breakout_trade_count"]),
        float(row["breakout_win_rate"]),
        float(row["breakout_average_return"]),
        float(row["breakout_stop_hit_rate"]),
        "breakout",
        "突破买点",
    )


def _pooled_entry_backtest(row: pd.Series) -> tuple[int, float, float, float] | None:
    """Combine breakout and pullback backtest evidence, weighted by trade count.

    Used as a fallback when the selected entry type alone has too few samples:
    with ~2 years of history a single entry type on a single ticker rarely
    reaches the sample minimum, which made the sample gate a near-automatic
    veto instead of a quality check.
    """
    breakout_count = int(_row_float(row, "breakout_trade_count", 0.0) or 0)
    pullback_count = int(_row_float(row, "pullback_trade_count", 0.0) or 0)
    total = breakout_count + pullback_count
    if total <= 0:
        return None

    def _weighted(breakout_key: str, pullback_key: str) -> float:
        parts: list[tuple[int, float]] = []
        breakout_value = _row_float(row, breakout_key)
        pullback_value = _row_float(row, pullback_key)
        if breakout_count > 0 and _is_finite(breakout_value):
            parts.append((breakout_count, float(breakout_value)))
        if pullback_count > 0 and _is_finite(pullback_value):
            parts.append((pullback_count, float(pullback_value)))
        if not parts:
            return float("nan")
        weight_sum = sum(weight for weight, _ in parts)
        return sum(weight * value for weight, value in parts) / weight_sum

    return (
        total,
        _weighted("breakout_win_rate", "pullback_win_rate"),
        _weighted("breakout_average_return", "pullback_average_return"),
        _weighted("breakout_stop_hit_rate", "pullback_stop_hit_rate"),
    )


def _backtest_quality_score(trade_count: int, win_rate: float, average_return: float) -> float:
    if trade_count <= 0 or not _is_finite(win_rate) or not _is_finite(average_return):
        return 0.0
    sample_score = min(trade_count / 30.0, 1.0) * 25.0
    win_score = max(0.0, min(float(win_rate), 1.0)) * 50.0
    return_score = max(0.0, min(float(average_return) / 0.10, 1.0)) * 25.0
    return sample_score + win_score + return_score


def _high_probability_score(
    row: pd.Series,
    trade_count: int,
    win_rate: float,
    average_return: float,
    fail_count: int,
) -> float:
    backtest_score = _backtest_quality_score(trade_count, win_rate, average_return)
    backtest_trust_score = float(row.get("backtest_trust_score", 50.0))
    # signal_score is already a horizon-weighted composite of technical, market,
    # relative strength, fundamental, analyst, valuation and sector scores
    # (_combined_signal_score), so those components are NOT added again here --
    # re-adding them double-counted the same information and compressed ranks.
    value = (
        float(row["signal_score"]) * 0.32
        + float(row["confidence_score"]) * 0.14
        + float(row["data_quality_score"]) * 0.10
        + backtest_score * 0.09
        + backtest_trust_score * 0.09
        + float(row.get("horizon_alignment_score", 50.0)) * 0.08
        + float(row.get("entry_readiness_score", 50.0)) * 0.10
        + float(row.get("trade_plan_quality_score", 50.0)) * 0.08
    )
    if str(row["overall_risk_level"]) == "high":
        value -= 15.0
    if str(row["event_risk_level"]) == "high":
        value -= 20.0
    if str(row.get("sentiment_risk_level", "unknown")) == "high":
        value -= 16.0
    if str(row.get("analyst_risk_level", "unknown")) == "high":
        value -= 14.0
    if str(row.get("valuation_risk_level", "unknown")) == "high":
        value -= 14.0
    # No fail_count penalty here: gate failures already hard-veto the candidate
    # and the watchlist tiers count missing items explicitly, so subtracting
    # per-failure points from the score punished the same gap a third time.
    del fail_count
    return round(float(max(0.0, min(value, 100.0))), 2)


def _high_probability_level(score: float) -> tuple[str, str]:
    if score >= 80:
        return "high", "高"
    if score >= 60:
        return "medium", "中等"
    return "low", "低"


def _add_probability_calibration_feedback(
    analysis: pd.DataFrame,
    probability_calibration_context: object | None = None,
) -> pd.DataFrame:
    rows = [
        _probability_calibration_feedback_for_row(row, probability_calibration_context)
        for _, row in analysis.iterrows()
    ]
    result = analysis.copy()
    for key in rows[0]:
        result[key] = [row[key] for row in rows]
    return result


def _probability_calibration_feedback_for_row(
    row: pd.Series,
    probability_calibration_context: object | None,
) -> dict[str, object]:
    raw_probability = float(row.get("calibrated_win_probability", np.nan))
    if not _is_finite(raw_probability):
        raw_probability = np.nan
    selected = _select_probability_calibration_row(
        raw_probability,
        probability_calibration_context,
    )
    adjustment = float(selected.get("adjustment", 0.0))
    sample_count = int(selected.get("sample_count", 0))
    source = str(selected.get("source", "no_recent_walk_forward_calibration"))
    source_zh = str(selected.get("source_zh", "没有近期滚动验证校准"))
    action = str(selected.get("action", "none"))
    action_zh = str(selected.get("action_zh", "无"))
    note = str(selected.get("note", "No walk-forward probability calibration was applied."))
    note_zh = str(selected.get("note_zh", "未应用滚动验证概率校准。"))

    effective_adjustment = 0.0
    if _is_finite(raw_probability) and _is_finite(adjustment):
        confidence = float(row.get("calibrated_probability_confidence", 50.0))
        sample_weight = 1.0 if sample_count >= 15 else 0.5 if sample_count >= 5 else 0.0
        confidence_weight = 1.0 if confidence >= 55.0 else 0.5
        effective_adjustment = min(max(adjustment * sample_weight * confidence_weight, -0.08), 0.08)
        adjusted_probability = min(max(raw_probability + effective_adjustment, 0.05), 0.90)
    else:
        adjusted_probability = raw_probability

    level, level_zh = _calibrated_probability_level(adjusted_probability)
    return {
        "calibrated_win_probability_raw": raw_probability,
        "calibrated_win_probability": round(float(adjusted_probability), 4)
        if _is_finite(adjusted_probability)
        else np.nan,
        "calibrated_probability_level": level,
        "calibrated_probability_level_zh": level_zh,
        "probability_calibration_adjustment": round(float(effective_adjustment), 4),
        "probability_calibration_source": source,
        "probability_calibration_source_zh": source_zh,
        "probability_calibration_sample_count": sample_count,
        "probability_calibration_action": action,
        "probability_calibration_action_zh": action_zh,
        "probability_calibration_note": note,
        "probability_calibration_note_zh": note_zh,
    }


def _select_probability_calibration_row(
    probability: float,
    probability_calibration_context: object | None,
) -> dict[str, object]:
    records = _probability_calibration_records(probability_calibration_context)
    if not records:
        return {}
    bucket = _probability_bucket_for_value(probability)
    by_bucket = {str(record.get("probability_bucket", "")): record for record in records}
    candidates = [by_bucket.get(bucket), by_bucket.get("overall")]
    for record in candidates:
        if not record:
            continue
        sample_count = _safe_int_like(record.get("sample_count"))
        adjustment = _safe_float_like(record.get("recommended_probability_adjustment"))
        if sample_count < 5 or not _is_finite(adjustment):
            continue
        action = str(record.get("formula_action", "none"))
        action_zh = str(record.get("formula_action_zh", "无"))
        source = f"walk_forward_probability_bucket:{record.get('probability_bucket')}"
        source_zh = f"滚动验证概率分组：{record.get('probability_bucket_zh', record.get('probability_bucket'))}"
        return {
            "adjustment": min(max(adjustment, -0.10), 0.10),
            "sample_count": sample_count,
            "source": source,
            "source_zh": source_zh,
            "action": action,
            "action_zh": action_zh,
            "note": str(record.get("calibration_note", "")),
            "note_zh": str(record.get("calibration_note_zh", "")),
        }
    return {}


def _probability_calibration_records(context: object | None) -> list[dict[str, object]]:
    if context is None:
        return []
    if isinstance(context, pd.DataFrame):
        return context.to_dict(orient="records")
    if isinstance(context, list):
        return [item for item in context if isinstance(item, dict)]
    records = getattr(context, "records", None)
    if isinstance(records, list):
        return [item for item in records if isinstance(item, dict)]
    return []


def _probability_bucket_for_value(probability: float) -> str:
    if not _is_finite(probability):
        return "overall"
    if probability < 0.50:
        return "below_50"
    if probability < 0.55:
        return "50_to_55"
    if probability < 0.60:
        return "55_to_60"
    if probability < 0.65:
        return "60_to_65"
    return "65_plus"


def _calibrated_probability_level(probability: float) -> tuple[str, str]:
    if not _is_finite(probability):
        return "unknown", "未知"
    if probability >= 0.65:
        return "high", "高"
    if probability >= 0.58:
        return "constructive", "偏积极"
    if probability >= 0.52:
        return "neutral", "中性"
    return "low", "偏低"


def _calibrated_win_probability_profile(
    row: pd.Series,
    trade_count: int,
    win_rate: float,
    stop_hit_rate: float,
    average_return: float,
    high_probability_score: float,
) -> dict[str, object]:
    sample_weight = min(max(float(trade_count) / 30.0, 0.0), 1.0)
    observed_win_rate = float(win_rate) if _is_finite(win_rate) else 0.50
    score_probability = 0.35 + min(max(high_probability_score, 0.0), 100.0) / 100.0 * 0.35
    evidence_weight = 0.35 + sample_weight * 0.40

    trust_score = float(row.get("backtest_trust_score", 50.0))
    recent_score = float(row.get("recent_backtest_score", 50.0))
    decay_score = float(row.get("backtest_decay_score", 50.0))
    confidence_score = float(row.get("confidence_score", 50.0))
    market_score = float(row.get("market_score", 50.0))
    relative_strength_score = float(row.get("relative_strength_score", 50.0))

    trust_adjustment = (trust_score - 60.0) / 100.0 * 0.08
    recent_adjustment = (recent_score - 50.0) / 100.0 * 0.06
    decay_adjustment = (decay_score - 50.0) / 100.0 * 0.06
    confidence_adjustment = (confidence_score - 60.0) / 100.0 * 0.04
    market_adjustment = (market_score - 55.0) / 100.0 * 0.03
    relative_adjustment = (relative_strength_score - 50.0) / 100.0 * 0.03
    return_adjustment = (
        min(max(float(average_return), -0.05), 0.10) * 0.75
        if _is_finite(average_return)
        else 0.0
    )
    stop_penalty = (
        max(0.0, float(stop_hit_rate) - 0.45) * 0.45
        if _is_finite(stop_hit_rate)
        else 0.0
    )

    probability = (
        observed_win_rate * evidence_weight
        + score_probability * (1.0 - evidence_weight)
        + trust_adjustment
        + recent_adjustment
        + decay_adjustment
        + confidence_adjustment
        + market_adjustment
        + relative_adjustment
        + return_adjustment
        - stop_penalty
    )
    probability = min(max(probability, 0.05), 0.90)

    confidence = (
        sample_weight * 35.0
        + min(max(float(row.get("data_quality_score", 50.0)), 0.0), 100.0) * 0.15
        + min(max(trust_score, 0.0), 100.0) * 0.20
        + min(max(recent_score, 0.0), 100.0) * 0.10
        + min(max(decay_score, 0.0), 100.0) * 0.10
        + min(max(confidence_score, 0.0), 100.0) * 0.10
    )
    if _is_finite(stop_hit_rate):
        confidence += max(0.0, 1.0 - float(stop_hit_rate)) * 10.0
    confidence = min(max(confidence, 0.0), 100.0)

    if probability >= 0.65:
        level, level_zh = "high", "高"
    elif probability >= 0.58:
        level, level_zh = "constructive", "偏积极"
    elif probability >= 0.52:
        level, level_zh = "neutral", "中性"
    else:
        level, level_zh = "low", "偏低"

    if confidence >= 75.0:
        confidence_level, confidence_level_zh = "high", "较高"
    elif confidence >= 55.0:
        confidence_level, confidence_level_zh = "medium", "中等"
    else:
        confidence_level, confidence_level_zh = "low", "偏低"

    note = (
        "Research estimate blended selected-entry win rate, high-probability score, "
        "sample size, trust, recent strength, decay check, market context, relative strength, "
        "average return, and stop-hit risk."
    )
    note_zh = (
        "这是研究估计值，综合了所选买点胜率、高概率分数、样本数、回测可信度、"
        "近期强度、衰退检查、大盘环境、相对强弱、平均收益和止损命中风险。"
    )
    return {
        "calibrated_win_probability": round(float(probability), 4),
        "calibrated_probability_level": level,
        "calibrated_probability_level_zh": level_zh,
        "calibrated_probability_confidence": round(float(confidence), 2),
        "calibrated_probability_confidence_level": confidence_level,
        "calibrated_probability_confidence_level_zh": confidence_level_zh,
        "calibrated_probability_note": note,
        "calibrated_probability_note_zh": note_zh,
    }


def _entry_evidence_profile(
    trade_count: int,
    win_rate: float,
    average_return: float,
) -> dict[str, str]:
    sample_level, sample_level_zh = _sample_confidence_level(trade_count)
    if trade_count <= 0 or not _is_finite(win_rate) or not _is_finite(average_return):
        return {
            "sample_confidence_level": "unavailable",
            "sample_confidence_level_zh": "不可用",
            "evidence_strength": "unavailable",
            "evidence_strength_zh": "不可用",
            "evidence_note": "No valid entry backtest evidence is available.",
            "evidence_note_zh": "当前没有可用的买点回测证据。",
        }

    performance_positive = float(win_rate) >= 0.55 and float(average_return) > 0.0
    if trade_count >= 30 and performance_positive:
        evidence_strength, evidence_strength_zh = "strong", "较强"
        conclusion = "The sample is large enough to treat the backtest as meaningful supporting evidence."
        conclusion_zh = "样本数量较充足，可以把该回测作为有意义的辅助证据。"
    elif trade_count >= 15 and performance_positive:
        evidence_strength, evidence_strength_zh = "moderate", "中等"
        conclusion = "The sample is useful, but still needs more validation before being treated as stable."
        conclusion_zh = "样本有参考价值，但还需要更多验证，不能直接视为稳定优势。"
    elif trade_count >= 10 and performance_positive:
        evidence_strength, evidence_strength_zh = "limited", "有限"
        conclusion = "The sample barely meets a research threshold; use it cautiously."
        conclusion_zh = "样本刚达到研究参考门槛，需要谨慎使用。"
    elif performance_positive:
        evidence_strength, evidence_strength_zh = "weak", "偏弱"
        conclusion = "The win rate may look good, but the sample is too small to trust strongly."
        conclusion_zh = "胜率看起来可能不错，但样本太少，不能高度信任。"
    else:
        evidence_strength, evidence_strength_zh = "weak", "偏弱"
        conclusion = "The historical entry evidence is not supportive enough."
        conclusion_zh = "历史买点证据支持度不够。"

    return {
        "sample_confidence_level": sample_level,
        "sample_confidence_level_zh": sample_level_zh,
        "evidence_strength": evidence_strength,
        "evidence_strength_zh": evidence_strength_zh,
        "evidence_note": (
            f"sample={trade_count}, win rate={_format_percent(win_rate)}, "
            f"avg return={_format_percent(average_return)}. {conclusion}"
        ),
        "evidence_note_zh": (
            f"样本={trade_count}，胜率={_format_percent(win_rate)}，"
            f"平均收益={_format_percent(average_return)}。{conclusion_zh}"
        ),
    }


def _sample_confidence_level(sample_count: int) -> tuple[str, str]:
    if sample_count >= 50:
        return "very_strong", "很强"
    if sample_count >= 30:
        return "strong", "较强"
    if sample_count >= 15:
        return "moderate", "中等"
    if sample_count >= 10:
        return "limited", "有限"
    if sample_count >= 5:
        return "weak", "偏弱"
    if sample_count > 0:
        return "very_weak", "很弱"
    return "unavailable", "不可用"


def _add_threshold_calibration(
    analysis: pd.DataFrame,
    thresholds: ScreeningThresholds | None = None,
) -> pd.DataFrame:
    thresholds = thresholds or ScreeningThresholds()
    rows = [_threshold_calibration_for_row(row, thresholds) for _, row in analysis.iterrows()]
    result = analysis.copy()
    for key in rows[0]:
        result[key] = [row[key] for row in rows]
    return result


def _threshold_calibration_for_row(
    row: pd.Series,
    thresholds: ScreeningThresholds,
) -> dict[str, object]:
    trade_count = int(row.get("screening_backtest_trade_count", 0))
    win_rate = float(row.get("screening_backtest_win_rate", np.nan))
    average_return = float(row.get("screening_backtest_average_return", np.nan))
    entry_type = str(row.get("screening_backtest_entry_type", "entry"))

    signal_threshold = thresholds.signal_score_min
    confidence_threshold = thresholds.confidence_min
    sample_threshold = thresholds.backtest_sample_min
    win_threshold = thresholds.backtest_win_rate_min
    return_threshold = thresholds.backtest_average_return_min
    regime_adjustment = _market_regime_threshold_adjustment(row)
    signal_threshold = min(
        max(signal_threshold + regime_adjustment["signal_delta"], 55.0),
        82.0,
    )
    confidence_threshold = min(
        max(confidence_threshold + regime_adjustment["confidence_delta"], 55.0),
        82.0,
    )
    sample_threshold = max(
        5,
        int(sample_threshold + regime_adjustment["sample_delta"]),
    )
    win_threshold = min(
        max(win_threshold + regime_adjustment["win_rate_delta"], 0.50),
        0.72,
    )
    return_threshold = max(
        return_threshold + regime_adjustment["average_return_delta"],
        0.0,
    )

    if trade_count < sample_threshold:
        return _calibration_result(
            action="require_more_samples",
            action_zh="要求更多样本",
            level="insufficient_evidence",
            level_zh="证据不足",
            signal_threshold=signal_threshold,
            confidence_threshold=confidence_threshold,
            sample_threshold=sample_threshold,
            win_threshold=win_threshold,
            return_threshold=return_threshold,
            trade_count=trade_count,
            win_rate=win_rate,
            average_return=average_return,
            entry_type=entry_type,
            note=(
                "Entry backtest sample is below the regime-adjusted minimum, so thresholds should not be relaxed."
            ),
            note_zh="买点回测样本低于市场状态调整后的最低要求，所以不应放宽门槛。",
            regime_adjustment=regime_adjustment,
        )

    weak_win_rate = not _is_finite(win_rate) or win_rate < win_threshold
    weak_return = not _is_finite(average_return) or average_return <= return_threshold
    if weak_win_rate or weak_return:
        return _calibration_result(
            action="tighten_thresholds",
            action_zh="提高门槛",
            level="strict",
            level_zh="严格",
            signal_threshold=min(signal_threshold + 3.0, 78.0),
            confidence_threshold=min(confidence_threshold + 3.0, 78.0),
            sample_threshold=max(sample_threshold + 5, 15),
            win_threshold=min(win_threshold + 0.03, 0.68),
            return_threshold=max(return_threshold, 0.01),
            trade_count=trade_count,
            win_rate=win_rate,
            average_return=average_return,
            entry_type=entry_type,
            note=(
                "Historical entry evidence is weak, so the model should demand higher signal, confidence, sample, and win-rate thresholds."
            ),
            note_zh="历史买点证据偏弱，所以模型应要求更高的信号、置信度、样本和胜率门槛。",
            regime_adjustment=regime_adjustment,
        )

    if trade_count >= 30 and win_rate >= win_threshold + 0.05 and average_return >= 0.03:
        return _calibration_result(
            action="allow_slightly_more_flexible_thresholds",
            action_zh="允许略微灵活",
            level="supportive",
            level_zh="支持",
            signal_threshold=max(signal_threshold - 1.0, 60.0),
            confidence_threshold=max(confidence_threshold - 1.0, 60.0),
            sample_threshold=sample_threshold,
            win_threshold=max(win_threshold - 0.01, 0.52),
            return_threshold=return_threshold,
            trade_count=trade_count,
            win_rate=win_rate,
            average_return=average_return,
            entry_type=entry_type,
            note=(
                "Historical entry evidence is broad and positive; current thresholds can be kept, with only slight flexibility."
            ),
            note_zh="历史买点证据较充足且表现为正；当前门槛可以保留，并只允许轻微灵活。",
            regime_adjustment=regime_adjustment,
        )

    if trade_count >= 15:
        return _calibration_result(
            action="keep_current_thresholds",
            action_zh="保持当前门槛",
            level="balanced",
            level_zh="平衡",
            signal_threshold=signal_threshold,
            confidence_threshold=confidence_threshold,
            sample_threshold=sample_threshold,
            win_threshold=win_threshold,
            return_threshold=return_threshold,
            trade_count=trade_count,
            win_rate=win_rate,
            average_return=average_return,
            entry_type=entry_type,
            note="Historical entry evidence supports the current threshold set, but is not strong enough to relax it.",
            note_zh="历史买点证据支持当前门槛，但还不足以降低门槛。",
            regime_adjustment=regime_adjustment,
        )

    return _calibration_result(
        action="keep_current_with_caution",
        action_zh="谨慎保持当前门槛",
        level="limited",
        level_zh="有限",
        signal_threshold=signal_threshold,
        confidence_threshold=confidence_threshold,
        sample_threshold=max(sample_threshold, 15),
        win_threshold=min(win_threshold + 0.01, 0.65),
        return_threshold=return_threshold,
        trade_count=trade_count,
        win_rate=win_rate,
        average_return=average_return,
        entry_type=entry_type,
        note="Entry evidence is positive but still limited, so the model should avoid lowering thresholds.",
        note_zh="买点证据为正但样本仍有限，所以模型不应降低门槛。",
        regime_adjustment=regime_adjustment,
    )


def _market_regime_threshold_adjustment(row: pd.Series) -> dict[str, object]:
    market_score = float(row.get("market_score", 50.0))
    market_status = str(row.get("market_status", "neutral")).lower()
    if market_status == "supportive" or market_score >= 70.0:
        return {
            "market_regime": "supportive",
            "market_regime_zh": "强势或支持型大盘",
            "signal_delta": -1.0,
            "confidence_delta": 0.0,
            "sample_delta": 0,
            "win_rate_delta": -0.005,
            "average_return_delta": 0.0,
            "note": "Supportive market regime allows slightly more flexible signal and win-rate thresholds, but does not remove quality gates.",
            "note_zh": "强势或支持型大盘允许信号和胜率门槛略微灵活，但不会取消质量门槛。",
        }
    if market_status == "weak" or market_score <= 45.0:
        return {
            "market_regime": "weak",
            "market_regime_zh": "弱势大盘",
            "signal_delta": 4.0,
            "confidence_delta": 3.0,
            "sample_delta": 5,
            "win_rate_delta": 0.03,
            "average_return_delta": 0.01,
            "note": "Weak market regime requires stricter signal, confidence, sample, win-rate, and average-return thresholds.",
            "note_zh": "弱势大盘要求更高的信号、置信度、样本、胜率和平均收益门槛。",
        }
    return {
        "market_regime": "neutral",
        "market_regime_zh": "中性或震荡大盘",
        "signal_delta": 0.0,
        "confidence_delta": 0.0,
        "sample_delta": 0,
        "win_rate_delta": 0.0,
        "average_return_delta": 0.0,
        "note": "Neutral market regime keeps the base threshold set.",
        "note_zh": "中性或震荡大盘保持基础门槛。",
    }


def _calibration_result(
    action: str,
    action_zh: str,
    level: str,
    level_zh: str,
    signal_threshold: float,
    confidence_threshold: float,
    sample_threshold: int,
    win_threshold: float,
    return_threshold: float,
    trade_count: int,
    win_rate: float,
    average_return: float,
    entry_type: str,
    note: str,
    note_zh: str,
    regime_adjustment: dict[str, object],
) -> dict[str, object]:
    return {
        "calibration_action": action,
        "calibration_action_zh": action_zh,
        "calibration_level": level,
        "calibration_level_zh": level_zh,
        "recommended_signal_threshold": round(float(signal_threshold), 2),
        "recommended_confidence_threshold": round(float(confidence_threshold), 2),
        "recommended_backtest_sample_min": int(sample_threshold),
        "recommended_backtest_win_rate_min": round(float(win_threshold), 4),
        "recommended_backtest_average_return_min": round(float(return_threshold), 4),
        "market_regime": regime_adjustment["market_regime"],
        "market_regime_zh": regime_adjustment["market_regime_zh"],
        "market_regime_signal_delta": round(float(regime_adjustment["signal_delta"]), 2),
        "market_regime_confidence_delta": round(
            float(regime_adjustment["confidence_delta"]), 2
        ),
        "market_regime_sample_delta": int(regime_adjustment["sample_delta"]),
        "market_regime_win_rate_delta": round(
            float(regime_adjustment["win_rate_delta"]), 4
        ),
        "market_regime_average_return_delta": round(
            float(regime_adjustment["average_return_delta"]), 4
        ),
        "market_regime_note": regime_adjustment["note"],
        "market_regime_note_zh": regime_adjustment["note_zh"],
        "calibration_sample_count": int(trade_count),
        "calibration_entry_type": entry_type,
        "calibration_observed_win_rate": float(win_rate) if _is_finite(win_rate) else np.nan,
        "calibration_observed_average_return": float(average_return)
        if _is_finite(average_return)
        else np.nan,
        "calibration_note": (
            f"{note} Market regime: {regime_adjustment['note']} Observed {entry_type}: sample={trade_count}, "
            f"win rate={_format_percent(win_rate)}, avg return={_format_percent(average_return)}."
        ),
        "calibration_note_zh": (
            f"{note_zh} 市场状态：{regime_adjustment['note_zh']} 当前{entry_type}：样本={trade_count}，"
            f"胜率={_format_percent(win_rate)}，平均收益={_format_percent(average_return)}。"
        ),
    }


def _add_calibrated_screening(
    analysis: pd.DataFrame,
    base_thresholds: ScreeningThresholds | None = None,
) -> pd.DataFrame:
    base_thresholds = base_thresholds or ScreeningThresholds()
    rows = [
        _calibrated_screening_for_row(row, base_thresholds)
        for _, row in analysis.iterrows()
    ]
    result = analysis.copy()
    for key in rows[0]:
        result[key] = [row[key] for row in rows]
    return result


def _calibrated_screening_for_row(
    row: pd.Series,
    base_thresholds: ScreeningThresholds,
) -> dict[str, object]:
    calibrated_thresholds = _thresholds_from_calibration(row, base_thresholds)
    calibrated = _high_probability_screening_for_row(row, calibrated_thresholds)
    passed = bool(calibrated["quality_gate_passed"])
    action = (
        "calibrated_high_probability_candidate"
        if passed
        else "not_calibrated_high_probability_now"
    )
    action_zh = "校准后高概率候选" if passed else "校准后当前不是高概率机会"
    note = (
        "The calibrated gate re-runs the high-probability filter using recommended thresholds from the calibration layer."
    )
    note_zh = "校准后筛选会使用阈值校准层给出的建议门槛，重新运行高概率过滤。"
    return {
        "calibrated_screening_action": action,
        "calibrated_screening_action_zh": action_zh,
        "calibrated_quality_gate_passed": passed,
        "calibrated_high_probability_score": calibrated["high_probability_score"],
        "calibrated_high_probability_level": calibrated["high_probability_level"],
        "calibrated_high_probability_level_zh": calibrated["high_probability_level_zh"],
        "calibrated_quality_gate_fail_reasons": calibrated["quality_gate_fail_reasons"],
        "calibrated_quality_gate_fail_reasons_zh": calibrated["quality_gate_fail_reasons_zh"],
        "calibrated_screening_backtest_entry_type": calibrated["screening_backtest_entry_type"],
        "calibrated_screening_backtest_trade_count": calibrated["screening_backtest_trade_count"],
        "calibrated_screening_backtest_win_rate": calibrated["screening_backtest_win_rate"],
        "calibrated_screening_backtest_average_return": calibrated[
            "screening_backtest_average_return"
        ],
        "calibrated_screening_note": note,
        "calibrated_screening_note_zh": note_zh,
    }


def _add_signal_review_feedback(
    analysis: pd.DataFrame,
    feedback_context: object | None = None,
) -> pd.DataFrame:
    feedback = _signal_review_feedback_dict(feedback_context)
    result = analysis.copy()
    adjustment = float(feedback["adjustment"])
    result["signal_review_status"] = feedback["status"]
    result["signal_review_score"] = feedback["score"]
    result["signal_review_adjustment"] = adjustment
    result["signal_review_sample_count"] = feedback["sample_count"]
    result["signal_review_focus_window"] = feedback["focus_window"]
    result["signal_review_win_rate"] = feedback["win_rate"]
    result["signal_review_avg_return"] = feedback["avg_return"]
    result["signal_review_median_return"] = feedback["median_return"]
    result["signal_review_level"] = feedback["level"]
    result["signal_review_level_zh"] = feedback["level_zh"]
    result["signal_review_note"] = feedback["note"]
    result["signal_review_note_zh"] = feedback["note_zh"]

    for column in ["high_probability_score", "calibrated_high_probability_score"]:
        result[column] = pd.to_numeric(result[column], errors="coerce").apply(
            lambda value: round(_clamp_score(float(value) + adjustment), 2)
            if _is_finite(float(value))
            else value
        )
    probability_adjustment = adjustment / 100.0 * 0.10
    result["calibrated_win_probability"] = pd.to_numeric(
        result["calibrated_win_probability"],
        errors="coerce",
    ).apply(
        lambda value: round(float(min(max(value + probability_adjustment, 0.05), 0.90)), 6)
        if _is_finite(float(value))
        else value
    )
    result["high_probability_level"] = [
        _high_probability_level(float(value))[0]
        for value in result["high_probability_score"]
    ]
    result["high_probability_level_zh"] = [
        _high_probability_level(float(value))[1]
        for value in result["high_probability_score"]
    ]
    result["calibrated_high_probability_level"] = [
        _high_probability_level(float(value))[0]
        for value in result["calibrated_high_probability_score"]
    ]
    result["calibrated_high_probability_level_zh"] = [
        _high_probability_level(float(value))[1]
        for value in result["calibrated_high_probability_score"]
    ]

    if str(feedback["level"]) in {"weak", "poor"} and int(feedback["sample_count"]) >= 5:
        fail_reason = "signal review history weak"
        fail_reason_zh = "历史信号复盘偏弱"
        result["calibrated_quality_gate_passed"] = False
        result["calibrated_screening_action"] = "not_calibrated_high_probability_now"
        result["calibrated_screening_action_zh"] = "校准后当前不是高概率机会"
        result["calibrated_quality_gate_fail_reasons"] = result[
            "calibrated_quality_gate_fail_reasons"
        ].apply(lambda value: _append_reason(value, fail_reason))
        result["calibrated_quality_gate_fail_reasons_zh"] = result[
            "calibrated_quality_gate_fail_reasons_zh"
        ].apply(lambda value: _append_reason(value, fail_reason_zh, delimiter="；"))
    return result


def _signal_review_feedback_dict(feedback_context: object | None) -> dict[str, object]:
    if feedback_context is None:
        return _neutral_signal_review_feedback()
    if hasattr(feedback_context, "to_dict"):
        raw = feedback_context.to_dict()
    elif isinstance(feedback_context, dict):
        raw = feedback_context
    else:
        raw = {
            key: getattr(feedback_context, key)
            for key in [
                "status",
                "score",
                "adjustment",
                "sample_count",
                "focus_window",
                "win_rate",
                "avg_return",
                "median_return",
                "level",
                "level_zh",
                "note",
                "note_zh",
            ]
            if hasattr(feedback_context, key)
        }
    neutral = _neutral_signal_review_feedback()
    return {**neutral, **raw}


def _neutral_signal_review_feedback() -> dict[str, object]:
    return {
        "status": "insufficient_history",
        "score": 50.0,
        "adjustment": 0.0,
        "sample_count": 0,
        "focus_window": "none",
        "win_rate": np.nan,
        "avg_return": np.nan,
        "median_return": np.nan,
        "level": "insufficient_history",
        "level_zh": "历史样本不足",
        "note": "Not enough completed prior signals are available for feedback scoring.",
        "note_zh": "已完成的历史信号样本不足，暂不用于反向调分。",
    }


def _thresholds_from_calibration(
    row: pd.Series,
    base_thresholds: ScreeningThresholds,
) -> ScreeningThresholds:
    return ScreeningThresholds.from_mapping(
        {
            **base_thresholds.to_dict(),
            "signal_score_min": float(row["recommended_signal_threshold"]),
            "confidence_min": float(row["recommended_confidence_threshold"]),
            "backtest_sample_min": int(row["recommended_backtest_sample_min"]),
            "backtest_win_rate_min": float(row["recommended_backtest_win_rate_min"]),
            "backtest_average_return_min": float(
                row["recommended_backtest_average_return_min"]
            ),
        }
    )


def _add_calibrated_watchlist_plan(
    analysis: pd.DataFrame,
    base_thresholds: ScreeningThresholds | None = None,
) -> pd.DataFrame:
    base_thresholds = base_thresholds or ScreeningThresholds()
    rows = [
        _calibrated_watchlist_plan_for_row(row, base_thresholds)
        for _, row in analysis.iterrows()
    ]
    result = analysis.copy()
    for key in rows[0]:
        result[key] = [row[key] for row in rows]
    return result


def _calibrated_watchlist_plan_for_row(
    row: pd.Series,
    base_thresholds: ScreeningThresholds,
) -> dict[str, object]:
    thresholds = _thresholds_from_calibration(row, base_thresholds)
    plan_row = row.copy()
    plan_row["quality_gate_passed"] = bool(row["calibrated_quality_gate_passed"])
    plan_row["high_probability_score"] = float(row["calibrated_high_probability_score"])
    plan = _watchlist_plan_for_row(plan_row, thresholds)
    return {
        "calibrated_watchlist_status": plan["watchlist_status"],
        "calibrated_watchlist_status_zh": plan["watchlist_status_zh"],
        "calibrated_watchlist_gap_score": plan["watchlist_gap_score"],
        "calibrated_watchlist_ready_items": plan["watchlist_ready_items"],
        "calibrated_watchlist_ready_items_zh": plan["watchlist_ready_items_zh"],
        "calibrated_watchlist_missing_items": plan["watchlist_missing_items"],
        "calibrated_watchlist_missing_items_zh": plan["watchlist_missing_items_zh"],
        "calibrated_watchlist_missing_count": plan["watchlist_missing_count"],
        "calibrated_watchlist_trigger_price": plan["watchlist_trigger_price"],
        "calibrated_watchlist_recheck_reason": plan["watchlist_recheck_reason"],
        "calibrated_watchlist_recheck_reason_zh": plan["watchlist_recheck_reason_zh"],
    }


def _add_watchlist_plan(
    analysis: pd.DataFrame,
    thresholds: ScreeningThresholds | None = None,
) -> pd.DataFrame:
    thresholds = thresholds or ScreeningThresholds()
    rows = [_watchlist_plan_for_row(row, thresholds) for _, row in analysis.iterrows()]
    result = analysis.copy()
    for key in rows[0]:
        result[key] = [row[key] for row in rows]
    return result


def _watchlist_plan_for_row(
    row: pd.Series,
    thresholds: ScreeningThresholds | None = None,
) -> dict[str, object]:
    thresholds = thresholds or ScreeningThresholds()
    checks = _watchlist_checks(row, thresholds)
    ready_items = [item[1] for item in checks if item[0]]
    ready_items_zh = [item[2] for item in checks if item[0]]
    missing_items = [item[1] for item in checks if not item[0]]
    missing_items_zh = [item[2] for item in checks if not item[0]]
    missing_count = len(missing_items)
    score = float(row["high_probability_score"])
    gap_score = round(float(max(0.0, thresholds.high_probability_target_score - score)), 2)

    if bool(row["quality_gate_passed"]):
        status, status_zh = "ready_high_probability", "已通过高概率筛选"
    elif (
        missing_count <= thresholds.near_watchlist_max_missing
        and score >= thresholds.near_watchlist_min_score
    ):
        status, status_zh = "close_but_not_ready", "接近但还没准备好"
    elif (
        missing_count <= thresholds.early_watchlist_max_missing
        or score >= thresholds.early_watchlist_min_score
    ):
        status, status_zh = "early_watch", "早期观察"
    else:
        status, status_zh = "not_ready", "暂不值得重点观察"

    trigger_price = _watchlist_trigger_price(row)
    reason, reason_zh = _watchlist_recheck_reason(
        row=row,
        status=status,
        trigger_price=trigger_price,
        missing_items=missing_items,
        missing_items_zh=missing_items_zh,
    )
    return {
        "watchlist_status": status,
        "watchlist_status_zh": status_zh,
        "watchlist_gap_score": gap_score,
        "watchlist_ready_items": "; ".join(ready_items) if ready_items else "none",
        "watchlist_ready_items_zh": "；".join(ready_items_zh) if ready_items_zh else "无",
        "watchlist_missing_items": "; ".join(missing_items) if missing_items else "none",
        "watchlist_missing_items_zh": "；".join(missing_items_zh) if missing_items_zh else "无",
        "watchlist_missing_count": missing_count,
        "watchlist_trigger_price": trigger_price,
        "watchlist_recheck_reason": reason,
        "watchlist_recheck_reason_zh": reason_zh,
    }


def _watchlist_checks(
    row: pd.Series,
    thresholds: ScreeningThresholds | None = None,
) -> list[tuple[bool, str, str]]:
    thresholds = thresholds or ScreeningThresholds()
    trade_count = int(row["screening_backtest_trade_count"])
    win_rate = float(row["screening_backtest_win_rate"])
    stop_hit_rate = float(row.get("screening_backtest_stop_hit_rate", np.nan))
    average_return = float(row["screening_backtest_average_return"])
    horizon = str(row["horizon"])
    checks = [
        (
            float(row["data_quality_score"]) >= thresholds.data_quality_min,
            f"data quality >= {thresholds.data_quality_min:g}",
            f"数据质量达到{thresholds.data_quality_min:g}分",
        ),
        (
            float(row["confidence_score"]) >= thresholds.confidence_min,
            f"confidence >= {thresholds.confidence_min:g}",
            f"置信度达到{thresholds.confidence_min:g}分",
        ),
        (
            str(row["overall_risk_level"]) != "high",
            "overall risk is not high",
            "综合风险不是高风险",
        ),
        (
            float(row["market_score"]) >= thresholds.market_score_min,
            f"market score >= {thresholds.market_score_min:g}",
            f"大盘分数达到{thresholds.market_score_min:g}分",
        ),
        (
            float(row["relative_strength_score"]) >= thresholds.relative_strength_min,
            f"relative strength >= {thresholds.relative_strength_min:g}",
            f"相对强弱达到{thresholds.relative_strength_min:g}分",
        ),
        (
            float(row["signal_score"]) >= thresholds.signal_score_min,
            f"signal score >= {thresholds.signal_score_min:g}",
            f"信号分数达到{thresholds.signal_score_min:g}分",
        ),
        (
            trade_count >= thresholds.backtest_sample_min,
            f"entry backtest sample >= {thresholds.backtest_sample_min}",
            f"买点回测样本至少{thresholds.backtest_sample_min}笔",
        ),
        (
            _is_finite(win_rate) and win_rate >= thresholds.backtest_win_rate_min,
            f"entry backtest win rate >= {thresholds.backtest_win_rate_min:.0%}",
            f"买点回测胜率至少{thresholds.backtest_win_rate_min:.0%}",
        ),
        (
            _is_finite(average_return)
            and average_return > thresholds.backtest_average_return_min,
            f"entry backtest average return > {thresholds.backtest_average_return_min:.0%}",
            f"买点回测平均收益高于{thresholds.backtest_average_return_min:.0%}",
        ),
        (
            not _is_finite(stop_hit_rate)
            or stop_hit_rate <= thresholds.max_backtest_stop_hit_rate,
            f"entry backtest stop-hit rate <= {thresholds.max_backtest_stop_hit_rate:.0%}",
            f"买点回测止损命中率不高于{thresholds.max_backtest_stop_hit_rate:.0%}",
        ),
        (
            float(row.get("backtest_trust_score", 0.0)) >= thresholds.backtest_trust_score_min,
            f"backtest trust score >= {thresholds.backtest_trust_score_min:g}",
            f"回测可信度分达到{thresholds.backtest_trust_score_min:g}分",
        ),
        (
            float(row.get("recent_backtest_score", 0.0)) >= thresholds.recent_backtest_score_min,
            f"recent backtest score >= {thresholds.recent_backtest_score_min:g}",
            f"近期回测强度分达到{thresholds.recent_backtest_score_min:g}分",
        ),
        (
            float(row.get("backtest_decay_score", 0.0)) >= thresholds.backtest_decay_score_min,
            f"backtest decay score >= {thresholds.backtest_decay_score_min:g}",
            f"回测衰退检查分达到{thresholds.backtest_decay_score_min:g}分",
        ),
        (
            bool(row["liquidity_filter_passed"]),
            "liquidity filter passed",
            "流动性过滤通过",
        ),
        (
            bool(row["entry_readiness_gate_passed"]),
            "entry is executable now",
            "买点当前可执行",
        ),
        (
            bool(row["trade_plan_quality_gate_passed"]),
            "trade plan quality passed",
            "交易计划质量通过",
        ),
        (
            str(row["event_risk_level"]) != "high"
            and not _coerce_bool(row.get("event_block_new_entries", False)),
            "event risk does not block new entries",
            "事件风险不阻止新入场",
        ),
        (
            str(row.get("sentiment_risk_level", "unknown")) != "high"
            and not _coerce_bool(row.get("sentiment_block_new_entries", False)),
            "news sentiment does not block new entries",
            "新闻情绪不阻止新入场",
        ),
        (
            str(row.get("analyst_risk_level", "unknown")) != "high"
            and not _coerce_bool(row.get("analyst_block_new_entries", False)),
            "analyst expectations do not block new entries",
            "分析师预期不阻止新入场",
        ),
        (
            str(row.get("valuation_risk_level", "unknown")) != "high"
            and not _coerce_bool(row.get("valuation_block_new_entries", False)),
            "valuation does not block new entries",
            "估值不阻止新入场",
        ),
    ]
    if horizon == "long":
        checks.append(
            (
                float(row["fundamental_score"]) >= thresholds.long_fundamental_score_min,
                f"long-term fundamental score >= {thresholds.long_fundamental_score_min:g}",
                f"长期基本面分数达到{thresholds.long_fundamental_score_min:g}分",
            )
        )
    if horizon in {"medium", "long"}:
        checks.append(
            (
                float(row["sector_score"]) >= thresholds.medium_long_sector_score_min,
                f"sector score >= {thresholds.medium_long_sector_score_min:g}",
                f"板块分数达到{thresholds.medium_long_sector_score_min:g}分",
            )
        )
    return checks


def _watchlist_trigger_price(row: pd.Series) -> float:
    if str(row.get("entry_type", "")) != "none" and _is_finite(row.get("entry_price")):
        return float(row["entry_price"])
    if _is_finite(row.get("resistance")):
        return float(row["resistance"])
    if _is_finite(row.get("entry_price")):
        return float(row["entry_price"])
    return np.nan


def _watchlist_recheck_reason(
    row: pd.Series,
    status: str,
    trigger_price: float,
    missing_items: list[str],
    missing_items_zh: list[str],
) -> tuple[str, str]:
    trigger_text = _format_number(trigger_price) if _is_finite(trigger_price) else "a valid trigger price"
    trigger_text_zh = _format_number(trigger_price) if _is_finite(trigger_price) else "有效触发价"
    if status == "ready_high_probability":
        return (
            "This setup already passed the strict high-probability filter; monitor the planned entry and risk levels.",
            "该结构已经通过严格高概率筛选；继续跟踪计划买点和风险位置。",
        )
    if not missing_items:
        return (
            f"Re-check if price closes above {trigger_text} and the setup remains stable.",
            f"如果价格收在{trigger_text_zh}上方且结构保持稳定，可以重新检查。",
        )
    top_missing = ", ".join(missing_items[:2])
    top_missing_zh = "、".join(missing_items_zh[:2])
    return (
        f"Re-check if price closes above {trigger_text}, high-probability score improves toward 65, "
        f"and these missing items improve: {top_missing}.",
        f"如果价格收在{trigger_text_zh}上方、高概率分数接近65，并且这些缺口改善：{top_missing_zh}，可以重新检查。",
    )
