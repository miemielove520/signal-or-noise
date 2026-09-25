from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Iterable

import numpy as np
import pandas as pd

from .screening_config import ScreeningThresholds, TradingRules


MIN_BACKTEST_SLIPPAGE_PCT = 0.0005
MAX_BACKTEST_SLIPPAGE_PCT = 0.01
SLIPPAGE_LIQUIDITY_WINDOW = 20


@dataclass(frozen=True)
class HorizonSpec:
    name: str
    label: str
    zh_label: str
    time_range: str
    time_range_zh: str
    lookback_days: int
    trend_window: int
    momentum_window: int
    volume_window: int
    atr_window: int
    atr_stop_multiple: float
    target_r_multiple: float
    max_chase_pct: float
    time_stop_days: int
    trailing_stop_trigger_r: float
    trailing_stop_lock_r: float
    min_history_days: int


# Under the "breakout" entry style, a confirmed breakout stays executable even when the
# stock is extended above its baseline, up to this multiple of the horizon's max_chase_pct.
# Beyond it the move is treated as parabolic and we wait for a pullback instead.
BREAKOUT_EXTENSION_MULTIPLE = 2.5


HORIZON_SPECS: dict[str, HorizonSpec] = {
    "short": HorizonSpec(
        name="short",
        label="short term",
        zh_label="短期",
        time_range="short-term trading setup",
        time_range_zh="短线交易节奏",
        lookback_days=20,
        trend_window=20,
        momentum_window=5,
        volume_window=10,
        atr_window=14,
        atr_stop_multiple=1.5,
        target_r_multiple=2.0,
        max_chase_pct=0.04,
        time_stop_days=20,
        trailing_stop_trigger_r=1.5,
        trailing_stop_lock_r=0.2,
        min_history_days=30,
    ),
    "medium": HorizonSpec(
        name="medium",
        label="medium term",
        zh_label="中期",
        time_range="trend continuation setup",
        time_range_zh="趋势延续判断",
        lookback_days=60,
        trend_window=50,
        momentum_window=20,
        volume_window=20,
        atr_window=14,
        atr_stop_multiple=2.0,
        target_r_multiple=2.5,
        max_chase_pct=0.08,
        time_stop_days=60,
        trailing_stop_trigger_r=1.5,
        trailing_stop_lock_r=0.2,
        min_history_days=75,
    ),
    "long": HorizonSpec(
        name="long",
        label="long term",
        zh_label="长期",
        time_range="long-term quality and trend setup",
        time_range_zh="长期质量与趋势判断",
        lookback_days=120,
        trend_window=100,
        momentum_window=60,
        volume_window=30,
        atr_window=20,
        atr_stop_multiple=3.0,
        target_r_multiple=3.0,
        max_chase_pct=0.15,
        time_stop_days=120,
        trailing_stop_trigger_r=1.5,
        trailing_stop_lock_r=0.2,
        min_history_days=140,
    ),
}


def _effective_horizon_spec(spec: HorizonSpec, trading_rules: TradingRules) -> HorizonSpec:
    return replace(
        spec,
        atr_stop_multiple=trading_rules.atr_stop_multiple_for(spec.name),
        target_r_multiple=trading_rules.target_r_multiple_for(spec.name),
        max_chase_pct=trading_rules.max_chase_pct_for(spec.name),
        time_stop_days=trading_rules.time_stop_days_for(spec.name),
        trailing_stop_trigger_r=trading_rules.trailing_stop_trigger_r,
        trailing_stop_lock_r=trading_rules.trailing_stop_lock_r,
    )


def normalize_horizons(horizons: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(horizons, str):
        raw = [horizons]
    else:
        raw = list(horizons)

    normalized: list[str] = []
    for horizon in raw:
        value = str(horizon).lower().strip()
        if value == "all":
            for name in HORIZON_SPECS:
                if name not in normalized:
                    normalized.append(name)
            continue
        if value not in HORIZON_SPECS:
            choices = ", ".join([*HORIZON_SPECS.keys(), "all"])
            raise ValueError(f"Unsupported horizon '{horizon}'. Choices: {choices}.")
        if value not in normalized:
            normalized.append(value)
    if not normalized:
        raise ValueError("At least one horizon is required.")
    return tuple(normalized)


def analyze_ticker(
    scored: pd.DataFrame,
    prices: pd.DataFrame,
    ticker: str,
    horizons: str | Iterable[str] = "all",
    as_of_date: str | pd.Timestamp | None = None,
    entry_buffer_pct: float = 0.003,
    market_context: object | None = None,
    relative_strength_contexts: dict[str, object] | None = None,
    event_risk_context: object | None = None,
    fundamental_context: object | None = None,
    sentiment_context: object | None = None,
    analyst_context: object | None = None,
    valuation_context: object | None = None,
    sector_context: object | None = None,
    screening_thresholds: ScreeningThresholds | None = None,
    trading_rules: TradingRules | None = None,
    probability_calibration_context: object | None = None,
    signal_review_feedback_context: object | None = None,
) -> pd.DataFrame:
    if entry_buffer_pct < 0:
        raise ValueError("entry_buffer_pct cannot be negative.")

    ticker = ticker.upper().strip()
    if not ticker:
        raise ValueError("ticker cannot be empty.")

    selected_horizons = normalize_horizons(horizons)
    price_frame = _ticker_prices(prices, ticker, as_of_date)
    scored_frame = _ticker_scores(scored, ticker, as_of_date)
    score_snapshot = _latest_score_snapshot(scored, scored_frame, price_frame["date"].max())

    effective_trading_rules = trading_rules or TradingRules()
    rows = [
        _analyze_horizon(
            price_frame=price_frame,
            ticker=ticker,
            spec=_effective_horizon_spec(HORIZON_SPECS[horizon], effective_trading_rules),
            trading_rules=effective_trading_rules,
            score_snapshot=score_snapshot,
            entry_buffer_pct=entry_buffer_pct,
            market_context=market_context,
            relative_strength_context=(relative_strength_contexts or {}).get(horizon),
            event_risk_context=event_risk_context,
            fundamental_context=fundamental_context,
            sentiment_context=sentiment_context,
            analyst_context=analyst_context,
            valuation_context=valuation_context,
            sector_context=sector_context,
        )
        for horizon in selected_horizons
    ]
    analysis = pd.DataFrame(rows)
    decision_summary = _build_decision_summary(analysis)
    for key, value in decision_summary.items():
        analysis[key] = value
    horizon_alignment_summary = _build_horizon_alignment_summary(analysis)
    for key, value in horizon_alignment_summary.items():
        analysis[key] = value
    confidence_summary = _build_confidence_summary(
        analysis,
        focus_horizon=str(decision_summary["decision_focus_horizon"]),
    )
    for key, value in confidence_summary.items():
        analysis[key] = value
    risk_summary = _build_risk_breakdown_summary(
        analysis,
        focus_horizon=str(decision_summary["decision_focus_horizon"]),
    )
    for key, value in risk_summary.items():
        analysis[key] = value
    data_quality_summary = _build_data_quality_summary(analysis)
    for key, value in data_quality_summary.items():
        analysis[key] = value
    thresholds = screening_thresholds or ScreeningThresholds()
    analysis = _add_high_probability_screening(analysis, thresholds)
    analysis = _add_probability_calibration_feedback(
        analysis,
        probability_calibration_context,
    )
    analysis = _add_threshold_calibration(analysis, thresholds)
    analysis = _add_calibrated_screening(analysis, thresholds)
    analysis = _add_signal_review_feedback(analysis, signal_review_feedback_context)
    analysis = _add_calibrated_watchlist_plan(analysis, thresholds)
    analysis = _add_watchlist_plan(analysis, thresholds)
    final_summary = _build_final_decision_summary(analysis)
    for key, value in final_summary.items():
        analysis[key] = value
    priority_blocker_summary = _build_priority_blocker_summary(analysis)
    for key, value in priority_blocker_summary.items():
        analysis[key] = value
    return analysis


def _final_next_step_line(first) -> str:
    """For a holding, the entry-trigger next step is meaningless — use the
    holder-appropriate one instead."""
    if bool(first.get("position_is_held", False)) and str(first.get("position_next_step_zh", "")):
        return f"{first.get('position_next_step', '')} / {first.get('position_next_step_zh', '')}"
    return f"{first['final_next_step']} / {first['final_next_step_zh']}"


def _position_section_lines(first) -> list[str]:
    """Position-awareness section, shown only when the ticker is a real holding."""
    if not bool(first.get("position_is_held", False)):
        return []
    concentration = str(first.get("position_concentration_level", ""))
    warn = " ⚠️ 单一持仓集中度偏高，注意风险。" if concentration in {"high", "extreme"} else ""
    return [
        "## Position Awareness / 持仓感知",
        "",
        "You hold this stock — the section below frames the analysis for an existing position, not a fresh entry.",
        "你持有这只股票——下面是针对现有持仓的参考，不是买入点。",
        "",
        f"- Holding / 持仓: `{_format_number(first.get('position_shares'))}` 股 @ "
        f"均价 `{_format_number(first.get('position_avg_cost'))}`",
        f"- Unrealized P/L / 浮动盈亏: `{_format_percent(first.get('position_unrealized_pl_pct'))}` "
        f"(`{_format_number(first.get('position_unrealized_pl'))}`)",
        f"- Portfolio weight / 组合占比: `{_format_number(first.get('position_weight_pct'))}%` "
        f"(集中度 `{first.get('position_concentration_level_zh', '')}`){warn}",
        f"- What to watch / 持有观察: {first.get('position_next_step', '')} / "
        f"{first.get('position_next_step_zh', '')}",
        f"- {first.get('position_note', '')} / {first.get('position_note_zh', '')}",
        "",
    ]


def render_ticker_analysis(analysis: pd.DataFrame) -> str:
    if analysis.empty:
        return "# Ticker Analysis\n\nNo analysis rows produced.\n"

    first = analysis.iloc[0]
    screening_focus = analysis.sort_values("high_probability_score", ascending=False).iloc[0]
    lines = [
        f"# Ticker Analysis: {first['ticker']}",
        "",
        "This report is generated from local research data and is not investment advice.",
        "本报告由本地研究系统生成，不构成投资建议。",
        "",
        f"- Analysis date: `{_format_date(first['date'])}`",
        f"- Latest price: `{_format_number(first['latest_price'])}`",
        f"- Requested period / 请求周期: `{first.get('requested_period', 'unknown')}`",
        f"- Effective period / 实际使用周期: `{first.get('analysis_period', 'unknown')}`",
        (
            "- Auto period upgrade / 自动周期升级: "
            f"`{first.get('auto_period_upgraded', False)}`"
        ),
        (
            "- Period note / 周期说明: "
            f"{first.get('auto_period_upgrade_reason', '')} / "
            f"{first.get('auto_period_upgrade_reason_zh', '')}"
        ),
        "- Signal timing: `generated_after_latest_close`",
        "- Earliest execution: `next_trading_session`",
        "",
        "## Final Decision / 最终执行结论",
        "",
        f"- Final decision / 最终结论: `{first['final_decision']}` / {first['final_decision_zh']}",
        f"- Final focus horizon / 最终重点周期: `{first['final_focus_horizon']}` / `{first['final_focus_horizon_zh']}`",
        f"- Final score / 最终分数: `{_format_number(first['final_score'])}`",
        f"- Final watchlist status / 最终观察状态: `{first['final_watchlist_status']}` / `{first['final_watchlist_status_zh']}`",
        f"- Final reason / 最终原因: {first['final_reason']} / {first['final_reason_zh']}",
        f"- Final next step / 下一步: {_final_next_step_line(first)}",
        "",
        *_position_section_lines(first),
        "## Priority Blockers / 主要卡点排序",
        "",
        f"- Primary blocker / 第一卡点: `{first['primary_blocker']}` / {first['primary_blocker_zh']}",
        f"- Priority blockers / 主要卡点: {first['priority_blockers']} / {first['priority_blockers_zh']}",
        f"- Blocker count / 卡点数量: `{first['priority_blocker_count']}`",
        f"- Blocker note / 卡点说明: {first['priority_blocker_note']} / {first['priority_blocker_note_zh']}",
        f"- Primary blocker resolution / 第一卡点解除条件: {first['primary_blocker_resolution']} / {first['primary_blocker_resolution_zh']}",
        f"- Resolution steps / 解除步骤: {first['blocker_resolution_steps']} / {first['blocker_resolution_steps_zh']}",
        f"- Re-check trigger / 重新检查触发条件: {first['blocker_recheck_trigger']} / {first['blocker_recheck_trigger_zh']}",
        f"- Primary blocker progress / 第一卡点解除进度: `{_format_number(first['primary_blocker_progress'])}/100`",
        f"- Blocker resolution score / 卡点解除分数: `{_format_number(first['blocker_resolution_score'])}/100`",
        f"- Blocker resolution level / 卡点解除等级: `{first['blocker_resolution_level']}` / {first['blocker_resolution_level_zh']}",
        f"- Resolution gap / 解除差距: {first['blocker_resolution_gap']} / {first['blocker_resolution_gap_zh']}",
        "",
        "## Horizon Alignment / 周期一致性",
        "",
        f"- Alignment label / 一致性标签: `{first['horizon_alignment_label']}` / `{first['horizon_alignment_label_zh']}`",
        f"- Alignment score / 一致性分数: `{_format_number(first['horizon_alignment_score'])}`",
        f"- Constructive horizons / 偏积极周期数: `{first['constructive_horizon_count']}`",
        f"- Weak horizons / 偏弱周期数: `{first['weak_horizon_count']}`",
        f"- Risk-wait horizons / 风险等待周期数: `{first['risk_wait_horizon_count']}`",
        f"- Signal score spread / 信号分差: `{_format_number(first['horizon_signal_score_spread'])}`",
        f"- Alignment note / 一致性说明: {first['horizon_alignment_note']} / {first['horizon_alignment_note_zh']}",
        "",
        "## Primary Decision / 主要结论",
        "",
        f"- Decision / 结论: `{first['primary_decision']}` / {first['primary_decision_zh']}",
        f"- Focus horizon / 重点周期: `{first['decision_focus_horizon']}`",
        f"- Why / 为什么: {first['decision_reason']} / {first['decision_reason_zh']}",
        f"- What to wait for / 等什么: {first['decision_wait_for']} / {first['decision_wait_for_zh']}",
        f"- Invalidation / 判断失效条件: {first['decision_invalidation']} / {first['decision_invalidation_zh']}",
        "",
        "## High Probability Filter / 高概率筛选器",
        "",
        (
            "- Screening profile / 筛选规则: "
            f"`{screening_focus.get('screening_profile', 'default')}` / "
            f"{screening_focus.get('screening_profile_zh', '默认规则')}"
        ),
        (
            "- Final screening / 最终筛选结论: "
            f"`{screening_focus['screening_action']}` / {screening_focus['screening_action_zh']}"
        ),
        f"- Focus horizon / 重点周期: `{screening_focus['horizon']}` / `{screening_focus['horizon_zh_label']}`",
        (
            "- High probability score / 高概率分数: "
            f"`{_format_number(screening_focus['high_probability_score'])}`"
        ),
        (
            "- High probability level / 高概率等级: "
            f"`{screening_focus['high_probability_level']}` / `{screening_focus['high_probability_level_zh']}`"
        ),
        (
            "- Calibrated win probability / 校准后胜率估计: "
            f"`{_format_percent(screening_focus['calibrated_win_probability'])}`，"
            f"level `{screening_focus['calibrated_probability_level']}` / "
            f"`{screening_focus['calibrated_probability_level_zh']}`"
        ),
        (
            "- Probability calibration feedback / 概率校准反馈: "
            f"raw `{_format_percent(screening_focus['calibrated_win_probability_raw'])}`，"
            f"adjustment `{_format_percent(screening_focus['probability_calibration_adjustment'])}`，"
            f"source `{screening_focus['probability_calibration_source']}` / "
            f"`{screening_focus['probability_calibration_source_zh']}`"
        ),
        (
            "- Probability calibration action / 概率校准动作: "
            f"`{screening_focus['probability_calibration_action']}` / "
            f"`{screening_focus['probability_calibration_action_zh']}`，"
            f"sample `{screening_focus['probability_calibration_sample_count']}`"
        ),
        (
            "- Probability calibration note / 概率校准说明: "
            f"{screening_focus['probability_calibration_note']} / "
            f"{screening_focus['probability_calibration_note_zh']}"
        ),
        (
            "- Probability confidence / 概率置信度: "
            f"`{_format_number(screening_focus['calibrated_probability_confidence'])}`，"
            f"level `{screening_focus['calibrated_probability_confidence_level']}` / "
            f"`{screening_focus['calibrated_probability_confidence_level_zh']}`"
        ),
        (
            "- Probability note / 概率说明: "
            f"{screening_focus['calibrated_probability_note']} / "
            f"{screening_focus['calibrated_probability_note_zh']}"
        ),
        (
            "- Quality gate passed / 是否通过质量门槛: "
            f"`{screening_focus['quality_gate_passed']}`"
        ),
        (
            "- Backtest used / 使用的买点回测: "
            f"`{screening_focus['screening_backtest_entry_type']}`，"
            f"sample `{screening_focus['screening_backtest_trade_count']}`，"
            f"win rate `{_format_percent(screening_focus['screening_backtest_win_rate'])}`，"
            f"stop hit `{_format_percent(screening_focus['screening_backtest_stop_hit_rate'])}`，"
            f"avg return `{_format_percent(screening_focus['screening_backtest_average_return'])}`"
        ),
        (
            "- Backtest trust gate / 回测可信度门槛: "
            f"score `{_format_number(screening_focus['backtest_trust_score'])}`，"
            f"level `{screening_focus['backtest_trust_level']}` / "
            f"`{screening_focus['backtest_trust_level_zh']}`"
        ),
        (
            "- Stability gates / 稳定性门槛: "
            f"recent `{_format_number(screening_focus['recent_backtest_score'])}` / "
            f"`{screening_focus['recent_backtest_level_zh']}`，"
            f"decay `{_format_number(screening_focus['backtest_decay_score'])}` / "
            f"`{screening_focus['backtest_decay_level_zh']}`"
        ),
        (
            "- Sample confidence / 样本置信度: "
            f"`{screening_focus['sample_confidence_level']}` / "
            f"`{screening_focus['sample_confidence_level_zh']}`"
        ),
        (
            "- Evidence strength / 证据强度: "
            f"`{screening_focus['evidence_strength']}` / "
            f"`{screening_focus['evidence_strength_zh']}`"
        ),
        (
            "- Evidence note / 证据说明: "
            f"{screening_focus['evidence_note']} / "
            f"{screening_focus['evidence_note_zh']}"
        ),
        (
            "- Liquidity filter / 流动性过滤: "
            f"`{screening_focus['liquidity_filter_passed']}`，"
            f"{screening_focus['liquidity_filter_reason']} / "
            f"{screening_focus['liquidity_filter_reason_zh']}"
        ),
        (
            "- Entry readiness / 买点可执行性: "
            f"`{screening_focus['entry_readiness_gate_passed']}`，"
            f"`{screening_focus['entry_readiness_status']}` / "
            f"`{screening_focus['entry_readiness_status_zh']}`，"
            f"score `{_format_number(screening_focus['entry_readiness_score'])}`"
        ),
        (
            "- Entry readiness note / 买点可执行性说明: "
            f"{screening_focus['entry_readiness_note']} / "
            f"{screening_focus['entry_readiness_note_zh']}"
        ),
        (
            "- Trade plan quality / 交易计划质量: "
            f"`{screening_focus['trade_plan_quality_gate_passed']}`，"
            f"`{screening_focus['trade_plan_quality_status']}` / "
            f"`{screening_focus['trade_plan_quality_status_zh']}`，"
            f"score `{_format_number(screening_focus['trade_plan_quality_score'])}`"
        ),
        (
            "- Trade plan quality note / 交易计划质量说明: "
            f"{screening_focus['trade_plan_quality_note']} / "
            f"{screening_focus['trade_plan_quality_note_zh']}"
        ),
        (
            "- Gate result / 门槛结果: "
            f"{screening_focus['quality_gate_fail_reasons']} / "
            f"{screening_focus['quality_gate_fail_reasons_zh']}"
        ),
        "",
        "## Profile Trading Rules / 大类交易规则",
        "",
        (
            "- Entry style / 买点风格: "
            f"`{screening_focus['trading_rule_entry_style']}` / "
            f"`{screening_focus['trading_rule_entry_style_zh']}`"
        ),
        (
            "- Stop and target / 止损与目标: "
            f"ATR stop `{_format_number(screening_focus['trading_rule_atr_stop_multiple'])}x`, "
            f"target `{_format_number(screening_focus['trading_rule_target_r_multiple'])}R`, "
            f"max chase `{_format_percent(screening_focus['trading_rule_max_chase_pct'])}`"
        ),
        (
            "- Time and trailing exit / 时间止损与移动止损: "
            f"time stop `{screening_focus['trading_rule_time_stop_days']}` days, "
            f"trailing trigger `{_format_number(screening_focus['trading_rule_trailing_stop_trigger_r'])}R`, "
            f"lock `{_format_number(screening_focus['trading_rule_trailing_stop_lock_r'])}R`"
        ),
        (
            "- Sell rule / 卖出规则: "
            f"`{screening_focus['trading_rule_sell_rule']}` / "
            f"{screening_focus['trading_rule_sell_rule_zh']}"
        ),
        (
            "- Rule note / 规则说明: "
            f"{screening_focus['trading_rule_note']} / {screening_focus['trading_rule_note_zh']}"
        ),
        "",
        "## Threshold Calibration / 阈值校准",
        "",
        (
            "- Calibration action / 校准动作: "
            f"`{screening_focus['calibration_action']}` / "
            f"{screening_focus['calibration_action_zh']}"
        ),
        (
            "- Calibration level / 校准等级: "
            f"`{screening_focus['calibration_level']}` / "
            f"`{screening_focus['calibration_level_zh']}`"
        ),
        (
            "- Market regime / 市场状态分层: "
            f"`{screening_focus['market_regime']}` / "
            f"`{screening_focus['market_regime_zh']}`"
        ),
        (
            "- Market regime adjustment / 市场状态门槛调整: "
            f"signal `{_format_number(screening_focus['market_regime_signal_delta'])}`, "
            f"confidence `{_format_number(screening_focus['market_regime_confidence_delta'])}`, "
            f"sample `{screening_focus['market_regime_sample_delta']}`, "
            f"win rate `{_format_percent(screening_focus['market_regime_win_rate_delta'])}`, "
            f"avg return `{_format_percent(screening_focus['market_regime_average_return_delta'])}`"
        ),
        (
            "- Market regime note / 市场状态说明: "
            f"{screening_focus['market_regime_note']} / "
            f"{screening_focus['market_regime_note_zh']}"
        ),
        (
            "- Recommended signal threshold / 建议信号分门槛: "
            f"`{_format_number(screening_focus['recommended_signal_threshold'])}`"
        ),
        (
            "- Recommended confidence threshold / 建议置信度门槛: "
            f"`{_format_number(screening_focus['recommended_confidence_threshold'])}`"
        ),
        (
            "- Recommended backtest sample min / 建议回测样本下限: "
            f"`{screening_focus['recommended_backtest_sample_min']}`"
        ),
        (
            "- Recommended backtest win rate min / 建议回测胜率下限: "
            f"`{_format_percent(screening_focus['recommended_backtest_win_rate_min'])}`"
        ),
        (
            "- Recommended backtest avg return min / 建议回测平均收益下限: "
            f"`{_format_percent(screening_focus['recommended_backtest_average_return_min'])}`"
        ),
        (
            "- Calibration note / 校准说明: "
            f"{screening_focus['calibration_note']} / "
            f"{screening_focus['calibration_note_zh']}"
        ),
        "",
        "## Calibrated Screening / 校准后筛选",
        "",
        (
            "- Calibrated final screening / 校准后最终结论: "
            f"`{screening_focus['calibrated_screening_action']}` / "
            f"{screening_focus['calibrated_screening_action_zh']}"
        ),
        (
            "- Calibrated quality gate passed / 校准后是否通过质量门槛: "
            f"`{screening_focus['calibrated_quality_gate_passed']}`"
        ),
        (
            "- Calibrated high probability score / 校准后高概率分数: "
            f"`{_format_number(screening_focus['calibrated_high_probability_score'])}`"
        ),
        (
            "- Calibrated gate result / 校准后门槛结果: "
            f"{screening_focus['calibrated_quality_gate_fail_reasons']} / "
            f"{screening_focus['calibrated_quality_gate_fail_reasons_zh']}"
        ),
        (
            "- Calibrated note / 校准后说明: "
            f"{screening_focus['calibrated_screening_note']} / "
            f"{screening_focus['calibrated_screening_note_zh']}"
        ),
        "",
        "## Signal Review Feedback / 信号复盘反馈",
        "",
        (
            "- Signal review score / 复盘反馈分: "
            f"`{_format_number(screening_focus['signal_review_score'])}`"
        ),
        (
            "- Score adjustment / 分数调整: "
            f"`{_format_signed_number(screening_focus['signal_review_adjustment'])}`"
        ),
        (
            "- Feedback level / 反馈等级: "
            f"`{screening_focus['signal_review_level']}` / "
            f"`{screening_focus['signal_review_level_zh']}`"
        ),
        (
            "- Completed samples / 已完成样本: "
            f"`{screening_focus['signal_review_sample_count']}`"
        ),
        (
            "- Focus window / 重点复盘窗口: "
            f"`{screening_focus['signal_review_focus_window']}`"
        ),
        (
            "- Review win rate / 复盘胜率: "
            f"`{_format_percent(screening_focus['signal_review_win_rate'])}`"
        ),
        (
            "- Review average return / 复盘平均收益: "
            f"`{_format_percent(screening_focus['signal_review_avg_return'])}`"
        ),
        (
            "- Review note / 复盘说明: "
            f"{screening_focus['signal_review_note']} / "
            f"{screening_focus['signal_review_note_zh']}"
        ),
        (
            "- Calibrated watchlist status / 校准后观察状态: "
            f"`{screening_focus['calibrated_watchlist_status']}` / "
            f"`{screening_focus['calibrated_watchlist_status_zh']}`"
        ),
        (
            "- Calibrated missing items / 校准后未达标条件: "
            f"{screening_focus['calibrated_watchlist_missing_items']} / "
            f"{screening_focus['calibrated_watchlist_missing_items_zh']}"
        ),
        (
            "- Calibrated re-check reason / 校准后重新检查原因: "
            f"{screening_focus['calibrated_watchlist_recheck_reason']} / "
            f"{screening_focus['calibrated_watchlist_recheck_reason_zh']}"
        ),
        "",
        "## Watchlist Plan / 观察计划",
        "",
        (
            "- Watchlist status / 观察状态: "
            f"`{screening_focus['watchlist_status']}` / `{screening_focus['watchlist_status_zh']}`"
        ),
        (
            "- Watchlist gap score / 观察差距分: "
            f"`{_format_number(screening_focus['watchlist_gap_score'])}`"
        ),
        (
            "- Trigger price / 重新检查触发价: "
            f"`{_format_number(screening_focus['watchlist_trigger_price'])}`"
        ),
        (
            "- Ready items / 已达标条件: "
            f"{screening_focus['watchlist_ready_items']} / "
            f"{screening_focus['watchlist_ready_items_zh']}"
        ),
        (
            "- Missing items / 未达标条件: "
            f"{screening_focus['watchlist_missing_items']} / "
            f"{screening_focus['watchlist_missing_items_zh']}"
        ),
        (
            "- Re-check reason / 重新检查原因: "
            f"{screening_focus['watchlist_recheck_reason']} / "
            f"{screening_focus['watchlist_recheck_reason_zh']}"
        ),
        "",
        "## Confidence / 置信度",
        "",
        f"- Confidence score / 置信度分数: `{_format_number(first['confidence_score'])}`",
        f"- Confidence level / 置信度等级: `{first['confidence_level']}` / `{first['confidence_level_zh']}`",
        f"- Confidence note / 置信度说明: {first['confidence_note']} / {first['confidence_note_zh']}",
        "",
        "## Risk Breakdown / 风险拆解",
        "",
        f"- Overall risk / 综合风险: `{first['overall_risk_level']}` / `{first['overall_risk_level_zh']}`",
        f"- Overall risk score / 综合风险分数: `{_format_number(first['overall_risk_score'])}`",
        f"- Technical risk / 技术风险: `{first['technical_risk_level']}` / `{first['technical_risk_level_zh']}`",
        f"- Market risk / 大盘风险: `{first['market_risk_level']}` / `{first['market_risk_level_zh']}`",
        f"- Relative strength risk / 相对强弱风险: `{first['relative_strength_risk_level']}` / `{first['relative_strength_risk_level_zh']}`",
        f"- Sector risk / 板块风险: `{first['sector_risk_level']}` / `{first['sector_risk_level_zh']}`",
        f"- Fundamental risk / 基本面风险: `{first['fundamental_risk_level']}` / `{first['fundamental_risk_level_zh']}`",
        f"- Event risk / 事件风险: `{first['event_risk_breakdown_level']}` / `{first['event_risk_breakdown_level_zh']}`",
        f"- News sentiment risk / 新闻情绪风险: `{first['sentiment_risk_breakdown_level']}` / `{first['sentiment_risk_breakdown_level_zh']}`",
        f"- Analyst expectation risk / 分析师预期风险: `{first['analyst_risk_breakdown_level']}` / `{first['analyst_risk_breakdown_level_zh']}`",
        f"- Valuation risk / 估值风险: `{first['valuation_risk_breakdown_level']}` / `{first['valuation_risk_breakdown_level_zh']}`",
        f"- Data risk / 数据风险: `{first['data_risk_level']}` / `{first['data_risk_level_zh']}`",
        f"- Risk note / 风险说明: {first['risk_breakdown_note']} / {first['risk_breakdown_note_zh']}",
        "",
        "## Data Quality / 数据质量",
        "",
        (
            "- Data quality score / 数据质量分数: "
            f"`{_format_number(first['data_quality_score'])}`"
        ),
        (
            "- Data quality level / 数据质量等级: "
            f"`{first['data_quality_level']}` / `{first['data_quality_level_zh']}`"
        ),
        (
            "- Weakest data layer / 最弱数据层: "
            f"`{first['data_quality_weakest_layer']}` / "
            f"`{first['data_quality_weakest_layer_zh']}`"
        ),
        (
            "- Weak data layers / 薄弱数据层: "
            f"{first['data_quality_weak_layers']} / {first['data_quality_weak_layers_zh']}"
        ),
        (
            "- Repair priority / 修复优先级: "
            f"`{first['data_quality_repair_priority']}` / "
            f"`{first['data_quality_repair_priority_zh']}`"
        ),
        (
            "- Repair actions / 修复动作: "
            f"{first['data_quality_repair_actions']} / "
            f"{first['data_quality_repair_actions_zh']}"
        ),
        (
            "- Data repair actions applied / 已自动修复: "
            f"{first.get('data_repair_actions_applied', 'none')} / "
            f"{first.get('data_repair_actions_applied_zh', '无')}"
        ),
        f"- Price data / 价格数据: `{first['price_data_status']}` / `{first['price_data_status_zh']}`",
        f"- Price health score / 价格健康分: `{_format_number(first['price_health_score'])}`",
        f"- Price health level / 价格健康等级: `{first['price_health_level']}` / `{first['price_health_level_zh']}`",
        f"- Max calendar gap / 最大日期缺口: `{first['price_max_calendar_gap_days']}` day(s)",
        f"- Large gap count / 较大缺口数量: `{first['price_large_gap_count']}`",
        f"- Zero-volume days / 零成交量天数: `{first['price_zero_volume_days']}`",
        f"- Missing OHLCV rows / OHLCV缺失记录: `{first['price_missing_ohlcv_rows']}`",
        f"- Extreme return count / 极端跳动次数: `{first['price_extreme_return_count']}`",
        (
            "- Price health note / 价格健康说明: "
            f"{first['price_health_note']} / {first['price_health_note_zh']}"
        ),
        f"- Market data / 大盘数据: `{first['market_data_status']}` / `{first['market_data_status_zh']}`",
        (
            "- Relative strength data / 相对强弱数据: "
            f"`{first['relative_strength_data_status']}` / `{first['relative_strength_data_status_zh']}`"
        ),
        f"- Sector data / 板块数据: `{first['sector_data_status']}` / `{first['sector_data_status_zh']}`",
        (
            "- Fundamental data / 基本面数据: "
            f"`{first['fundamental_data_status']}` / `{first['fundamental_data_status_zh']}`"
        ),
        f"- Event data / 事件数据: `{first['event_data_status']}` / `{first['event_data_status_zh']}`",
        f"- News sentiment data / 新闻情绪数据: `{first['sentiment_data_status']}` / `{first['sentiment_data_status_zh']}`",
        f"- Analyst data / 分析师数据: `{first['analyst_data_status']}` / `{first['analyst_data_status_zh']}`",
        f"- Valuation data / 估值数据: `{first['valuation_data_status']}` / `{first['valuation_data_status_zh']}`",
        f"- Missing fallback count / 缺失回退数量: `{first['missing_fallback_count']}`",
        (
            "- Data quality note / 数据质量说明: "
            f"{first['data_quality_note']} / {first['data_quality_note_zh']}"
        ),
        "",
        "## Plain Summary / 简明结论",
        "",
    ]
    summary_columns = [
        "horizon",
        "horizon_time_range",
        "action",
        "signal_score",
        "plain_summary",
        "plain_summary_zh",
    ]
    lines.extend(_markdown_table(analysis[summary_columns]))
    lines.extend(
        [
            "",
            "## Entry Plan / 买点计划",
            "",
            (
                "This section explains whether the current price is near the planned entry, "
                "whether chasing is acceptable, and what trigger should be watched."
            ),
            "本区块说明当前价离计划买点有多远、是否适合追高、应该等待什么触发条件。",
            "",
        ]
    )
    entry_plan_columns = [
        "horizon",
        "entry_type",
        "entry_price",
        "entry_distance_pct",
        "entry_distance_note",
        "entry_distance_note_zh",
        "chase_status",
        "chase_status_zh",
        "entry_plan",
        "entry_plan_zh",
    ]
    lines.extend(_markdown_table(analysis[entry_plan_columns]))
    lines.extend(
        [
            "",
            "## Backtest Reliability / 回测可靠性",
            "",
            (
                "This section translates historical sample size and backtest results into "
                "a reliability note. It should support, not replace, the current setup."
            ),
            "本区块把历史样本数量和回测结果翻译成可靠性说明。它只能辅助当前结构判断，不能替代当前信号。",
            "",
        ]
    )
    reliability_columns = [
        "horizon",
        "backtest_reliability_level",
        "backtest_reliability_level_zh",
        "backtest_reliability_note",
        "backtest_reliability_note_zh",
    ]
    lines.extend(_markdown_table(analysis[reliability_columns]))
    lines.extend(
        [
            "",
            "## Horizon Definition / 周期定义",
            "",
            "These labels describe analysis style, not fixed holding periods.",
            "这些分类描述的是分析风格，不是固定持有时间。",
            "",
        ]
    )
    lines.extend(_markdown_table(_horizon_definition_frame()))
    lines.extend(
        [
            "",
            "## Market And Relative Strength / 大盘环境与相对强弱",
            "",
            f"- Market score / 大盘分数: `{_format_number(first['market_score'])}`",
            f"- Market status / 大盘状态: `{first['market_status']}`",
            f"- Market note / 大盘说明: {first['market_note']} / {first['market_note_zh']}",
            "",
            "## Sector Context / 板块环境",
            "",
            f"- Sector / 板块: `{first['sector']}`",
            f"- Industry / 行业: `{first['industry']}`",
            f"- Sector ETF / 板块ETF: `{first['sector_etf']}`",
            f"- Sector score / 板块分数: `{_format_number(first['sector_score'])}`",
            f"- Sector status / 板块状态: `{first['sector_status']}`",
            f"- Sector relative strength / 板块相对强弱: `{_format_percent(first['sector_relative_strength'])}`",
            f"- Sector note / 板块说明: {first['sector_note']} / {first['sector_note_zh']}",
            "",
            "## Fundamental Quality / 基本面质量",
            "",
            f"- Fundamental score / 基本面分数: `{_format_number(first['fundamental_score'])}`",
            f"- Fundamental quality / 基本面质量: `{first['fundamental_quality']}` / `{first['fundamental_quality_zh']}`",
            f"- Data coverage / 数据覆盖率: `{_format_number(first['fundamental_data_coverage'])}`",
            f"- Revenue growth / 营收增长: `{_format_percent(first['revenue_growth'])}`",
            f"- Earnings growth / 盈利增长: `{_format_percent(first['earnings_growth'])}`",
            f"- Profit margin / 净利率: `{_format_percent(first['profit_margin'])}`",
            f"- Gross / operating margin / 毛利率·营业利润率: "
            f"`{_format_percent(first['fundamental_gross_margin'])}` / "
            f"`{_format_percent(first['fundamental_operating_margin'])}`",
            f"- Return on equity / ROE: `{_format_percent(first['return_on_equity'])}`",
            f"- Cash-flow quality / 现金流质量: `{_format_number(first['fundamental_cash_flow_quality'])}` "
            f"(cash conversion / 现金转化率 `{_format_number(first['fundamental_cash_conversion'])}`, "
            f"FCF margin / 自由现金流率 `{_format_percent(first['fundamental_fcf_margin'])}`)",
            f"- Forward PE / 预期市盈率: `{_format_number(first['forward_pe'])}`",
            f"- PEG ratio / PEG: `{_format_number(first['peg_ratio'])}`",
            f"- Debt to equity / 负债权益比: `{_format_number(first['debt_to_equity'])}`",
            f"- Net debt / equity / 净负债权益比: `{_format_number(first['fundamental_net_debt_to_equity'])}`",
            f"- Fundamental trend / 基本面趋势: `{first['fundamental_trend_direction']}` / "
            f"`{first['fundamental_trend_direction_zh']}` "
            f"(score / 趋势分 `{_format_number(first['fundamental_trend_score'])}`)",
            f"- Margin trend (gross/op/net) / 利润率趋势(毛利·营业·净利): "
            f"`{first['fundamental_gross_margin_trend']}` / "
            f"`{first['fundamental_operating_margin_trend']}` / "
            f"`{first['fundamental_net_margin_trend']}`",
            f"- FCF-margin / revenue-growth trend / 现金流率·营收增速趋势: "
            f"`{first['fundamental_fcf_margin_trend']}` / `{first['fundamental_revenue_growth_trend']}`",
            f"- Fundamental note / 基本面说明: {first['fundamental_note']} / {first['fundamental_note_zh']}",
            "",
            "## Analyst Expectations / 分析师预期",
            "",
            f"- Analyst score / 分析师分数: `{_format_number(first['analyst_score'])}`",
            f"- Analyst label / 分析师标签: `{first['analyst_label']}` / `{first['analyst_label_zh']}`",
            f"- Analyst risk / 分析师风险: `{first['analyst_risk_level']}` / `{first['analyst_risk_level_zh']}`",
            f"- Block new entries / 是否阻止新入场: `{bool(first['analyst_block_new_entries'])}`",
            f"- Analyst upside / 目标价上行空间: `{_format_percent(first['analyst_upside'])}`",
            f"- Recommendation mean / 推荐均值: `{_format_number(first['recommendation_mean'])}`",
            f"- Recommendation key / 推荐关键词: `{first['recommendation_key']}`",
            f"- Number of analysts / 覆盖分析师数: `{_format_number(first['number_of_analysts'])}`",
            f"- Target mean price / 平均目标价: `{_format_number(first['target_mean_price'])}`",
            f"- Analyst note / 分析师说明: {first['analyst_note']} / {first['analyst_note_zh']}",
            "",
            "## Valuation Risk / 估值风险",
            "",
            f"- Valuation score / 估值分数: `{_format_number(first['valuation_score'])}`",
            f"- Valuation label / 估值标签: `{first['valuation_label']}` / `{first['valuation_label_zh']}`",
            f"- Valuation risk / 估值风险: `{first['valuation_risk_level']}` / `{first['valuation_risk_level_zh']}`",
            f"- Block new entries / 是否阻止新入场: `{bool(first['valuation_block_new_entries'])}`",
            f"- Forward PE / 预期市盈率: `{_format_number(first['valuation_forward_pe'])}`",
            f"- Trailing PE / 静态市盈率: `{_format_number(first['valuation_trailing_pe'])}`",
            f"- PEG ratio / PEG: `{_format_number(first['valuation_peg_ratio'])}`",
            f"- Free cash flow yield / 自由现金流收益率: `{_format_percent(first['valuation_free_cash_flow_yield'])}`",
            f"- Growth reference / 增长参考: `{_format_percent(first['valuation_growth_reference'])}`",
            f"- Profit margin / 净利率: `{_format_percent(first['valuation_profit_margin'])}`",
            f"- Valuation note / 估值说明: {first['valuation_note']} / {first['valuation_note_zh']}",
            "",
            "## News Sentiment / 新闻情绪",
            "",
            f"- Sentiment score / 情绪分数: `{_format_number(first['sentiment_score'])}`",
            f"- Sentiment label / 情绪标签: `{first['sentiment_label']}` / `{first['sentiment_label_zh']}`",
            f"- Sentiment risk / 情绪风险: `{first['sentiment_risk_level']}` / `{first['sentiment_risk_level_zh']}`",
            f"- Block new entries / 是否阻止新入场: `{bool(first['sentiment_block_new_entries'])}`",
            f"- Positive news level / 利好等级: `{first['positive_news_level']}` / `{first['positive_news_level_zh']}`",
            f"- Positive news score / 利好加分: `{_format_number(first['positive_news_score'])}`",
            f"- Positive news drivers / 利好触发项: {first['positive_news_drivers']} / {first['positive_news_drivers_zh']}",
            f"- Risk news level / 风险新闻等级: `{first['risk_news_level']}` / `{first['risk_news_level_zh']}`",
            f"- Fake-catalyst / dilution hits / 假利好·稀释命中: `{first['fake_catalyst_count']}` / `{first['dilution_count']}`",
            f"- Risk news drivers / 风险触发项: {first['risk_news_drivers']} / {first['risk_news_drivers_zh']}",
            f"- Headlines used / 使用新闻标题数: `{first['sentiment_titles_used']}`",
            f"- Positive keywords / 正面关键词数: `{first['sentiment_positive_count']}`",
            f"- Negative keywords / 负面关键词数: `{first['sentiment_negative_count']}`",
            f"- High-risk keywords / 高风险关键词数: `{first['sentiment_high_risk_count']}`",
            f"- Sentiment note / 情绪说明: {first['sentiment_note']} / {first['sentiment_note_zh']}",
            "",
            "## Event Risk / 事件风险",
            "",
            (
                "- Event risk score / 事件风险分数: "
                f"`{_format_number(first['event_risk_score'])}` "
                "(higher means higher event risk / 分数越高代表事件风险越高)"
            ),
            f"- Event risk level / 事件风险等级: `{first['event_risk_level']}` / `{first['event_risk_level_zh']}`",
            f"- Event window / 事件窗口: `{first['event_window']}` / `{first['event_window_zh']}`",
            f"- Block new entries / 是否阻止新入场: `{bool(first['event_block_new_entries'])}`",
            f"- Cooldown active / 财报后冷却是否生效: `{bool(first['event_cooldown_active'])}`",
            f"- Next earnings date / 下一次财报日期: `{_format_number(first['next_earnings_date'])}`",
            f"- Days until earnings / 距离财报天数: `{_format_number(first['days_until_earnings'])}`",
            f"- Last earnings date / 上一次财报日期: `{_format_number(first['last_earnings_date'])}`",
            f"- Days since earnings / 距离上次财报天数: `{_format_number(first['days_since_earnings'])}`",
            f"- Event risk note / 事件风险说明: {first['event_risk_note']} / {first['event_risk_note_zh']}",
        ]
    )
    lines.extend(
        [
            "",
            "## Signal Plan / 信号计划",
            "",
        ]
    )
    display_columns = [
        "horizon",
        "horizon_time_range",
        "action",
        "signal_score",
        "trading_rule_entry_style",
        "trading_rule_atr_stop_multiple",
        "trading_rule_target_r_multiple",
        "trading_rule_max_chase_pct",
        "trading_rule_time_stop_days",
        "technical_score",
        "market_score",
        "relative_strength_score",
        "sector_score",
        "sector_status",
        "fundamental_score",
        "fundamental_quality",
        "analyst_score",
        "analyst_risk_level",
        "valuation_score",
        "valuation_risk_level",
        "sentiment_score",
        "sentiment_risk_level",
        "screening_action",
        "high_probability_score",
        "high_probability_level",
        "calibrated_win_probability",
        "calibrated_win_probability_raw",
        "probability_calibration_adjustment",
        "probability_calibration_source",
        "calibrated_probability_level",
        "calibrated_probability_confidence",
        "calibration_action",
        "calibrated_screening_action",
        "calibrated_quality_gate_passed",
        "calibrated_high_probability_score",
        "signal_review_score",
        "signal_review_adjustment",
        "signal_review_level",
        "calibrated_watchlist_status",
        "calibrated_watchlist_missing_count",
        "recommended_signal_threshold",
        "recommended_confidence_threshold",
        "recommended_backtest_sample_min",
        "recommended_backtest_win_rate_min",
        "sample_confidence_level",
        "evidence_strength",
        "liquidity_filter_passed",
        "quality_gate_passed",
        "quality_gate_fail_reasons",
        "watchlist_status",
        "watchlist_gap_score",
        "watchlist_missing_count",
        "event_risk_level",
        "event_risk_score",
        "signal_timing",
        "entry_type",
        "entry_price",
        "stop_loss",
        "take_profit",
        "risk_reward",
    ]
    lines.extend(_markdown_table(analysis[display_columns]))
    lines.extend(
        [
            "",
            "## Entry Backtest / 买点回测",
            "",
            (
                "This section reviews similar historical breakout and pullback entries "
                "for the same ticker and horizon. A win means the target was reached "
                "before the stop, or the holding window ended above entry if neither "
                "level was reached."
            ),
            (
                "本区块回测同一股票、同一周期下过去类似的突破买点和回调买点。"
                "胜利代表先到目标价再到止损；如果目标价和止损都没触发，则持有窗口结束时高于入场价算胜利。"
            ),
            (
                "- Execution model / 执行模型: "
                f"`{first['backtest_execution_model']}` / `{first['backtest_execution_model_zh']}`"
            ),
            (
                "- Time stop / 时间止损: "
                f"`{first['backtest_time_stop_days']}` trading sessions"
            ),
            (
                "- Trailing stop / 移动止损: "
                f"trigger `{_format_number(first['backtest_trailing_stop_trigger_r'])}R`, "
                f"lock `{_format_number(first['backtest_trailing_stop_lock_r'])}R`"
            ),
            (
                "- Slippage / 滑点: "
                f"`{_format_percent(first['backtest_slippage_pct'])}`"
            ),
            (
                "- Slippage drivers / 滑点依据: "
                f"avg dollar volume `{_format_number(first['backtest_avg_dollar_volume'])}`, "
                f"ATR ratio `{_format_percent(first['backtest_atr_ratio'])}`, "
                f"liquidity `{first['backtest_liquidity_label']}` / `{first['backtest_liquidity_label_zh']}`, "
                f"volatility `{first['backtest_volatility_label']}` / `{first['backtest_volatility_label_zh']}`"
            ),
            (
                "- Execution note / 执行说明: "
                f"{first['backtest_execution_note']} / {first['backtest_execution_note_zh']}"
            ),
            (
                "- Backtest trust score / 回测可信度分: "
                f"`{_format_number(first['backtest_trust_score'])}`"
            ),
            (
                "- Backtest trust level / 回测可信度等级: "
                f"`{first['backtest_trust_level']}` / `{first['backtest_trust_level_zh']}`"
            ),
            (
                "- Backtest trust note / 回测可信度说明: "
                f"{first['backtest_trust_note']} / {first['backtest_trust_note_zh']}"
            ),
            (
                "- Regime coverage score / 行情覆盖分: "
                f"`{_format_number(first['regime_coverage_score'])}`，"
                f"level `{first['regime_coverage_level']}` / "
                f"`{first['regime_coverage_level_zh']}`"
            ),
            (
                "- Regime coverage note / 行情覆盖说明: "
                f"{first['regime_coverage_note']} / {first['regime_coverage_note_zh']}"
            ),
            (
                "- Recent backtest strength / 近期回测强度: "
                f"`{_format_number(first['recent_backtest_score'])}`，"
                f"level `{first['recent_backtest_level']}` / "
                f"`{first['recent_backtest_level_zh']}`"
            ),
            (
                "- Recent backtest note / 近期回测说明: "
                f"{first['recent_backtest_note']} / {first['recent_backtest_note_zh']}"
            ),
            (
                "- Backtest decay check / 回测衰退检查: "
                f"`{_format_number(first['backtest_decay_score'])}`，"
                f"level `{first['backtest_decay_level']}` / "
                f"`{first['backtest_decay_level_zh']}`"
            ),
            (
                "- Backtest decay note / 回测衰退说明: "
                f"{first['backtest_decay_note']} / {first['backtest_decay_note_zh']}"
            ),
            "",
        ]
    )
    backtest_columns = [
        "horizon",
        "horizon_time_range",
        "backtest_trust_score",
        "backtest_trust_level",
        "backtest_sample_score",
        "backtest_liquidity_score",
        "backtest_slippage_score",
        "backtest_return_evidence_score",
        "regime_coverage_score",
        "regime_coverage_level",
        "regime_coverage_regime_count",
        "regime_coverage_dominant_regime",
        "regime_coverage_dominant_share",
        "recent_backtest_score",
        "recent_backtest_level",
        "recent_backtest_trade_count",
        "recent_backtest_win_rate",
        "recent_backtest_average_return",
        "recent_backtest_return_delta",
        "backtest_decay_score",
        "backtest_decay_level",
        "backtest_decay_early_trade_count",
        "backtest_decay_late_trade_count",
        "backtest_decay_early_win_rate",
        "backtest_decay_late_win_rate",
        "backtest_decay_early_average_return",
        "backtest_decay_late_average_return",
        "backtest_decay_average_return_delta",
        "backtest_execution_model",
        "backtest_time_stop_days",
        "backtest_trailing_stop_trigger_r",
        "backtest_trailing_stop_lock_r",
        "backtest_slippage_pct",
        "backtest_avg_dollar_volume",
        "backtest_atr_ratio",
        "backtest_liquidity_label",
        "backtest_volatility_label",
        "breakout_trade_count",
        "breakout_sample_quality",
        "breakout_sample_quality_zh",
        "breakout_win_rate",
        "breakout_target_hit_rate",
        "breakout_stop_hit_rate",
        "breakout_trailing_stop_hit_rate",
        "breakout_average_gain",
        "breakout_average_loss",
        "breakout_average_return",
        "pullback_trade_count",
        "pullback_sample_quality",
        "pullback_sample_quality_zh",
        "pullback_win_rate",
        "pullback_target_hit_rate",
        "pullback_stop_hit_rate",
        "pullback_trailing_stop_hit_rate",
        "pullback_average_gain",
        "pullback_average_loss",
        "pullback_average_return",
        "entry_backtest_note",
        "entry_backtest_note_zh",
    ]
    lines.extend(_markdown_table(_entry_backtest_frame(analysis[backtest_columns])))
    lines.append("")
    lines.append("## Rationale / 理由")
    lines.append("")
    for row in analysis.itertuples(index=False):
        lines.append(f"- `{row.horizon}`: {row.rationale}")
        lines.append(f"  中文: {row.rationale_zh}")
        lines.append(
            f"  Action explanation / 动作解释: "
            f"{row.action_explanation} / {row.action_explanation_zh}"
        )
        lines.append(
            f"  Relative strength / 相对强弱: "
            f"{row.relative_strength_note} / {row.relative_strength_note_zh}"
        )
        lines.append(
            f"  Event risk / 事件风险: "
            f"{row.event_risk_note} / {row.event_risk_note_zh}"
        )
        lines.append(
            f"  Fundamental quality / 基本面质量: "
            f"{row.fundamental_note} / {row.fundamental_note_zh}"
        )
    return "\n".join(lines).rstrip() + "\n"


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


def _clamp_score(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


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


def _row_float(row: pd.Series, key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    return float(value) if _is_finite(value) else default


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


def _safe_float_like(value: object) -> float:
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _safe_int_like(value: object) -> int:
    try:
        if pd.isna(value):
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


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


def _append_reason(value: object, reason: str, delimiter: str = "; ") -> str:
    text = str(value or "").strip()
    if not text or text in {"nan", "None"}:
        return reason
    parts = [part.strip() for part in text.split(delimiter) if part.strip()]
    if reason in parts:
        return text
    return text + delimiter + reason


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


def _ticker_prices(
    prices: pd.DataFrame,
    ticker: str,
    as_of_date: str | pd.Timestamp | None,
) -> pd.DataFrame:
    required = {"date", "ticker", "high", "low", "adj_close", "volume"}
    missing = required.difference(prices.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Missing price columns for ticker analysis: {missing_text}")

    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=False)
    frame["ticker"] = frame["ticker"].astype("string").str.upper().str.strip()
    frame = frame[frame["ticker"] == ticker].copy()
    if as_of_date is not None:
        frame = frame[frame["date"] <= pd.Timestamp(as_of_date)]
    if frame.empty:
        raise ValueError(f"No price data found for ticker {ticker}.")

    numeric_columns = ["high", "low", "adj_close", "volume"]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=["date", "ticker", "adj_close"])
    frame = frame.sort_values("date").reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No usable price data found for ticker {ticker}.")
    return frame


def _price_data_health_profile(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {
            "score": 0.0,
            "level": "poor",
            "level_zh": "差",
            "penalty": 25.0,
            "issue_count": 1,
            "max_calendar_gap_days": 0,
            "large_gap_count": 0,
            "zero_volume_days": 0,
            "missing_ohlcv_rows": 0,
            "extreme_return_count": 0,
            "high_low_inversion_count": 0,
            "adjustment_anomaly_count": 0,
            "note": "price frame is empty",
            "note_zh": "价格数据为空",
        }

    dates = pd.to_datetime(frame["date"], errors="coerce")
    date_gaps = dates.sort_values().diff().dt.days.dropna()
    max_gap = int(date_gaps.max()) if not date_gaps.empty and pd.notna(date_gaps.max()) else 0
    large_gap_count = int((date_gaps > 7).sum()) if not date_gaps.empty else 0

    adj_close = pd.to_numeric(frame.get("adj_close"), errors="coerce")
    high = pd.to_numeric(frame.get("high"), errors="coerce")
    low = pd.to_numeric(frame.get("low"), errors="coerce")
    volume = pd.to_numeric(frame.get("volume"), errors="coerce")
    close = pd.to_numeric(frame.get("close"), errors="coerce") if "close" in frame else pd.Series(dtype=float)

    missing_ohlcv_rows = int(
        pd.concat([high, low, adj_close, volume], axis=1)
        .isna()
        .any(axis=1)
        .sum()
    )
    zero_volume_days = int((volume.fillna(0.0) <= 0).sum())
    high_low_inversion_count = int(((high < low) & high.notna() & low.notna()).sum())

    returns = adj_close.pct_change().replace([np.inf, -np.inf], np.nan)
    extreme_return_count = int((returns.abs() > 0.35).sum())

    adjustment_anomaly_count = 0
    if not close.empty:
        ratio = (adj_close / close.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        adjustment_anomaly_count = int(((ratio <= 0.01) | (ratio >= 20.0)).sum())
    adjustment_anomaly_count += int((adj_close <= 0).sum())

    penalty = 0.0
    if max_gap > 14:
        penalty += 8.0
    elif max_gap > 7:
        penalty += 4.0
    penalty += min(large_gap_count * 2.0, 8.0)
    penalty += min(zero_volume_days * 1.5, 10.0)
    penalty += min(missing_ohlcv_rows * 2.0, 10.0)
    penalty += min(extreme_return_count * 4.0, 12.0)
    penalty += min(high_low_inversion_count * 4.0, 12.0)
    penalty += min(adjustment_anomaly_count * 4.0, 12.0)
    penalty = round(float(min(penalty, 30.0)), 2)

    issue_count = sum(
        count > 0
        for count in [
            large_gap_count,
            zero_volume_days,
            missing_ohlcv_rows,
            extreme_return_count,
            high_low_inversion_count,
            adjustment_anomaly_count,
        ]
    )
    score = round(float(max(0.0, 100.0 - penalty * 3.0)), 2)
    if score >= 90:
        level, level_zh = "clean", "干净"
    elif score >= 75:
        level, level_zh = "usable", "可用"
    elif score >= 55:
        level, level_zh = "watch", "需关注"
    else:
        level, level_zh = "poor", "差"

    notes: list[str] = []
    notes_zh: list[str] = []
    if large_gap_count:
        notes.append(f"{large_gap_count} large calendar gap(s), max gap {max_gap} day(s)")
        notes_zh.append(f"{large_gap_count}个较大日期缺口，最大缺口{max_gap}天")
    if zero_volume_days:
        notes.append(f"{zero_volume_days} zero-volume row(s)")
        notes_zh.append(f"{zero_volume_days}条零成交量记录")
    if missing_ohlcv_rows:
        notes.append(f"{missing_ohlcv_rows} row(s) with missing OHLCV fields")
        notes_zh.append(f"{missing_ohlcv_rows}条OHLCV字段缺失记录")
    if extreme_return_count:
        notes.append(f"{extreme_return_count} extreme adjusted return move(s)")
        notes_zh.append(f"{extreme_return_count}次复权价格极端跳动")
    if high_low_inversion_count:
        notes.append(f"{high_low_inversion_count} high/low inversion row(s)")
        notes_zh.append(f"{high_low_inversion_count}条最高价低于最低价记录")
    if adjustment_anomaly_count:
        notes.append(f"{adjustment_anomaly_count} adjusted-price anomaly row(s)")
        notes_zh.append(f"{adjustment_anomaly_count}条复权价格异常记录")
    if not notes:
        notes.append("price health checks found no major anomalies")
        notes_zh.append("价格健康检查未发现明显异常")

    return {
        "score": score,
        "level": level,
        "level_zh": level_zh,
        "penalty": penalty,
        "issue_count": int(issue_count),
        "max_calendar_gap_days": max_gap,
        "large_gap_count": large_gap_count,
        "zero_volume_days": zero_volume_days,
        "missing_ohlcv_rows": missing_ohlcv_rows,
        "extreme_return_count": extreme_return_count,
        "high_low_inversion_count": high_low_inversion_count,
        "adjustment_anomaly_count": adjustment_anomaly_count,
        "note": "; ".join(notes),
        "note_zh": "；".join(notes_zh),
    }


def _ticker_scores(
    scored: pd.DataFrame,
    ticker: str,
    as_of_date: str | pd.Timestamp | None,
) -> pd.DataFrame:
    if scored.empty or "ticker" not in scored.columns or "date" not in scored.columns:
        return pd.DataFrame()
    frame = scored.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=False)
    frame["ticker"] = frame["ticker"].astype("string").str.upper().str.strip()
    frame = frame[frame["ticker"] == ticker].copy()
    if as_of_date is not None:
        frame = frame[frame["date"] <= pd.Timestamp(as_of_date)]
    return frame.sort_values("date").reset_index(drop=True)


def _latest_score_snapshot(
    scored: pd.DataFrame,
    ticker_scores: pd.DataFrame,
    price_date: pd.Timestamp,
) -> dict[str, object]:
    if ticker_scores.empty:
        return {
            "score_date": pd.NaT,
            "score": np.nan,
            "score_percentile": np.nan,
            "passes_universe": True,
        }

    latest_date = ticker_scores[ticker_scores["date"] <= price_date]["date"].max()
    latest = ticker_scores[ticker_scores["date"] == latest_date].iloc[-1]
    score = float(latest["score"]) if "score" in latest and pd.notna(latest["score"]) else np.nan

    percentile = np.nan
    if "score" in scored.columns:
        scored_dates = scored.copy()
        scored_dates["date"] = pd.to_datetime(scored_dates["date"], utc=False)
        cross_section = scored_dates[
            (scored_dates["date"] == latest_date) & scored_dates["score"].notna()
        ].copy()
        if len(cross_section) > 1 and math.isfinite(score):
            ranks = cross_section["score"].rank(method="average", pct=True)
            ticker_value = str(latest["ticker"]).upper()
            ticker_mask = cross_section["ticker"].astype("string").str.upper() == ticker_value
            if ticker_mask.any():
                percentile = float(ranks.loc[ticker_mask].iloc[-1])

    passes_universe = True
    if "passes_universe" in latest:
        passes_universe = bool(latest["passes_universe"])

    return {
        "score_date": latest_date,
        "score": score,
        "score_percentile": percentile,
        "passes_universe": passes_universe,
    }


def _analyze_horizon(
    price_frame: pd.DataFrame,
    ticker: str,
    spec: HorizonSpec,
    trading_rules: TradingRules,
    score_snapshot: dict[str, object],
    entry_buffer_pct: float,
    market_context: object | None,
    relative_strength_context: object | None,
    event_risk_context: object | None,
    fundamental_context: object | None,
    sentiment_context: object | None,
    analyst_context: object | None,
    valuation_context: object | None,
    sector_context: object | None,
) -> dict[str, object]:
    frame = price_frame.copy()
    close = frame["adj_close"].astype(float)
    high = frame["high"].astype(float).fillna(close)
    low = frame["low"].astype(float).fillna(close)
    volume = frame["volume"].astype(float)

    latest = frame.iloc[-1]
    latest_price = float(latest["adj_close"])
    history_days = int(len(frame))
    trend_ma = _last_valid(close.rolling(spec.trend_window).mean())
    trend_distance = _safe_ratio(latest_price, trend_ma) - 1.0
    trend_slope = _moving_average_slope(close, spec.trend_window)
    momentum = _window_return(close, spec.momentum_window)
    atr = _last_valid(_average_true_range(high, low, close, spec.atr_window))
    avg_volume = _last_valid(volume.rolling(spec.volume_window).mean())
    volume_ratio = _safe_ratio(float(latest["volume"]), avg_volume)

    prior = frame.iloc[:-1].tail(spec.lookback_days)
    if prior.empty:
        prior = frame.tail(spec.lookback_days)
    support = float(prior["low"].min()) if not prior.empty else np.nan
    resistance = float(prior["high"].max()) if not prior.empty else np.nan

    breakout_entry = _finite_or(resistance * (1.0 + entry_buffer_pct), latest_price)
    pullback_entry = _pullback_entry(latest_price, trend_ma, support, entry_buffer_pct)
    action, entry_type, entry_price = _classify_action(
        latest_price=latest_price,
        breakout_entry=breakout_entry,
        pullback_entry=pullback_entry,
        trend_distance=trend_distance,
        trend_slope=trend_slope,
        momentum=momentum,
        volume_ratio=volume_ratio,
        history_days=history_days,
        spec=spec,
        preferred_entry_style=trading_rules.preferred_entry_style,
        passes_universe=bool(score_snapshot["passes_universe"]),
    )
    stop_loss = _stop_loss(
        entry_price=entry_price,
        support=support,
        atr=atr,
        atr_multiple=spec.atr_stop_multiple,
        entry_buffer_pct=entry_buffer_pct,
    )
    risk_per_share = max(entry_price - stop_loss, 0.0)
    take_profit = entry_price + risk_per_share * spec.target_r_multiple
    risk_reward = _safe_ratio(take_profit - entry_price, risk_per_share)
    entry_backtest = _entry_backtest_summary(
        frame=frame,
        spec=spec,
        entry_buffer_pct=entry_buffer_pct,
    )
    backtest_reliability = _backtest_reliability(entry_backtest)
    price_health = _price_data_health_profile(frame)
    backtest_trust = _backtest_trust_profile(
        entry_backtest=entry_backtest,
        price_health=price_health,
    )

    technical_score = _signal_score(
        trend_distance=trend_distance,
        trend_slope=trend_slope,
        momentum=momentum,
        volume_ratio=volume_ratio,
        score_percentile=float(score_snapshot["score_percentile"])
        if pd.notna(score_snapshot["score_percentile"])
        else np.nan,
        history_days=history_days,
        min_history_days=spec.min_history_days,
        passes_universe=bool(score_snapshot["passes_universe"]),
    )
    market_score = _context_float(market_context, "market_score", 50.0)
    relative_strength_score = _context_float(relative_strength_context, "score", 50.0)
    event_risk_level = _context_text(event_risk_context, "event_risk_level", "unknown")
    event_risk_score = _context_float(event_risk_context, "event_risk_score", 50.0)
    event_block_new_entries = _context_bool(event_risk_context, "event_block_new_entries", False)
    event_cooldown_active = _context_bool(event_risk_context, "event_cooldown_active", False)
    fundamental_score = _context_float(fundamental_context, "fundamental_score", 50.0)
    fundamental_quality = _context_text(fundamental_context, "fundamental_quality", "unknown")
    sentiment_score = _context_float(sentiment_context, "sentiment_score", 50.0)
    sentiment_risk_level = _context_text(sentiment_context, "sentiment_risk_level", "unknown")
    sentiment_block_new_entries = _context_bool(
        sentiment_context,
        "sentiment_block_new_entries",
        False,
    )
    analyst_score = _context_float(analyst_context, "analyst_score", 50.0)
    analyst_risk_level = _context_text(analyst_context, "analyst_risk_level", "unknown")
    analyst_block_new_entries = _context_bool(
        analyst_context,
        "analyst_block_new_entries",
        False,
    )
    valuation_score = _context_float(valuation_context, "valuation_score", 50.0)
    valuation_risk_level = _context_text(valuation_context, "valuation_risk_level", "unknown")
    valuation_block_new_entries = _context_bool(
        valuation_context,
        "valuation_block_new_entries",
        False,
    )
    sector_score = _context_float(sector_context, "sector_score", 50.0)
    signal_score = _combined_signal_score(
        spec=spec,
        technical_score=technical_score,
        market_score=market_score,
        relative_strength_score=relative_strength_score,
        fundamental_score=fundamental_score,
        analyst_score=analyst_score,
        valuation_score=valuation_score,
        sector_score=sector_score,
    )
    original_action = action
    action, entry_type = _apply_context_downgrade(
        action=action,
        entry_type=entry_type,
        market_score=market_score,
        relative_strength_score=relative_strength_score,
        event_risk_level=event_risk_level,
        event_block_new_entries=event_block_new_entries,
        sentiment_risk_level=sentiment_risk_level,
        sentiment_block_new_entries=sentiment_block_new_entries,
        analyst_risk_level=analyst_risk_level,
        analyst_block_new_entries=analyst_block_new_entries,
        valuation_risk_level=valuation_risk_level,
        valuation_block_new_entries=valuation_block_new_entries,
        fundamental_score=fundamental_score,
        fundamental_quality=fundamental_quality,
        sector_score=sector_score,
        horizon=spec.name,
    )
    rationale = _rationale(
        spec=spec,
        action=action,
        trend_distance=trend_distance,
        trend_slope=trend_slope,
        momentum=momentum,
        volume_ratio=volume_ratio,
        support=support,
        resistance=resistance,
        history_days=history_days,
        passes_universe=bool(score_snapshot["passes_universe"]),
    )
    rationale_zh = _rationale_zh(
        spec=spec,
        action=action,
        trend_distance=trend_distance,
        trend_slope=trend_slope,
        momentum=momentum,
        volume_ratio=volume_ratio,
        support=support,
        resistance=resistance,
        history_days=history_days,
        passes_universe=bool(score_snapshot["passes_universe"]),
    )
    action_explanation, action_explanation_zh = _action_explanation(
        action=action,
        original_action=original_action,
        market_score=market_score,
        relative_strength_score=relative_strength_score,
        event_risk_level=event_risk_level,
        days_until_earnings=_context_float(event_risk_context, "days_until_earnings", np.nan),
        event_window=_context_text(event_risk_context, "event_window", "unknown"),
        sentiment_risk_level=sentiment_risk_level,
        sentiment_score=sentiment_score,
        analyst_risk_level=analyst_risk_level,
        analyst_score=analyst_score,
        valuation_risk_level=valuation_risk_level,
        valuation_score=valuation_score,
        fundamental_score=fundamental_score,
        fundamental_quality=fundamental_quality,
    )
    plain_summary, plain_summary_zh = _plain_summary(
        spec=spec,
        action=action,
        signal_score=signal_score,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
    )
    entry_distance_pct = _safe_ratio(entry_price, latest_price) - 1.0
    entry_plan = _entry_plan(
        action=action,
        entry_type=entry_type,
        latest_price=latest_price,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        support=support,
        resistance=resistance,
        entry_distance_pct=entry_distance_pct,
    )

    return {
        "date": latest["date"],
        "ticker": ticker,
        "horizon": spec.name,
        "horizon_label": spec.label,
        "horizon_zh_label": spec.zh_label,
        "horizon_time_range": spec.time_range,
        "horizon_time_range_zh": spec.time_range_zh,
        "trading_rule_entry_style": trading_rules.preferred_entry_style,
        "trading_rule_entry_style_zh": trading_rules.preferred_entry_style_zh,
        "trading_rule_atr_stop_multiple": spec.atr_stop_multiple,
        "trading_rule_target_r_multiple": spec.target_r_multiple,
        "trading_rule_max_chase_pct": spec.max_chase_pct,
        "trading_rule_time_stop_days": spec.time_stop_days,
        "trading_rule_trailing_stop_trigger_r": spec.trailing_stop_trigger_r,
        "trading_rule_trailing_stop_lock_r": spec.trailing_stop_lock_r,
        "trading_rule_sell_rule": trading_rules.sell_rule,
        "trading_rule_sell_rule_zh": trading_rules.sell_rule_zh,
        "trading_rule_note": _trading_rule_note(spec, trading_rules),
        "trading_rule_note_zh": _trading_rule_note_zh(spec, trading_rules),
        "action": action,
        "original_action": original_action,
        "signal_score": signal_score,
        "technical_score": technical_score,
        "market_score": market_score,
        "relative_strength_score": relative_strength_score,
        "event_risk_score": event_risk_score,
        "event_window": _context_text(event_risk_context, "event_window", "unknown"),
        "event_window_zh": _context_text(event_risk_context, "event_window_zh", "未知"),
        "event_block_new_entries": event_block_new_entries,
        "event_cooldown_active": event_cooldown_active,
        "fundamental_score": fundamental_score,
        "sentiment_score": sentiment_score,
        "sentiment_label": _context_text(sentiment_context, "sentiment_label", "unknown"),
        "sentiment_label_zh": _context_text(sentiment_context, "sentiment_label_zh", "未知"),
        "sentiment_risk_level": sentiment_risk_level,
        "sentiment_risk_level_zh": _context_text(
            sentiment_context,
            "sentiment_risk_level_zh",
            "未知",
        ),
        "sentiment_block_new_entries": sentiment_block_new_entries,
        "positive_news_level": _context_text(sentiment_context, "positive_news_level", "unknown"),
        "positive_news_level_zh": _context_text(
            sentiment_context,
            "positive_news_level_zh",
            "未知",
        ),
        "positive_news_score": _context_float(sentiment_context, "positive_news_score", 0.0),
        "positive_news_major_count": int(
            _context_float(sentiment_context, "positive_news_major_count", 0.0)
        ),
        "positive_news_strong_count": int(
            _context_float(sentiment_context, "positive_news_strong_count", 0.0)
        ),
        "positive_news_moderate_count": int(
            _context_float(sentiment_context, "positive_news_moderate_count", 0.0)
        ),
        "positive_news_drivers": _context_text(
            sentiment_context,
            "positive_news_drivers",
            "none",
        ),
        "positive_news_drivers_zh": _context_text(
            sentiment_context,
            "positive_news_drivers_zh",
            "无",
        ),
        "risk_news_level": _context_text(sentiment_context, "risk_news_level", "none"),
        "risk_news_level_zh": _context_text(sentiment_context, "risk_news_level_zh", "无明显风险"),
        "risk_news_score": _context_float(sentiment_context, "risk_news_score", 0.0),
        "fake_catalyst_count": int(_context_float(sentiment_context, "fake_catalyst_count", 0.0)),
        "dilution_count": int(_context_float(sentiment_context, "dilution_count", 0.0)),
        "risk_news_drivers": _context_text(sentiment_context, "risk_news_drivers", "none"),
        "risk_news_drivers_zh": _context_text(sentiment_context, "risk_news_drivers_zh", "无"),
        "sentiment_positive_count": int(
            _context_float(sentiment_context, "sentiment_positive_count", 0.0)
        ),
        "sentiment_negative_count": int(
            _context_float(sentiment_context, "sentiment_negative_count", 0.0)
        ),
        "sentiment_high_risk_count": int(
            _context_float(sentiment_context, "sentiment_high_risk_count", 0.0)
        ),
        "sentiment_titles_used": int(
            _context_float(sentiment_context, "sentiment_titles_used", 0.0)
        ),
        "sentiment_note": _context_text(
            sentiment_context,
            "sentiment_note",
            "Recent news title data is unavailable; neutral score 50 is used.",
        ),
        "sentiment_note_zh": _context_text(
            sentiment_context,
            "sentiment_note_zh",
            "近期新闻标题不可用；使用中性分数50。",
        ),
        "sentiment_warning": _context_text(sentiment_context, "sentiment_warning", ""),
        "analyst_score": analyst_score,
        "analyst_label": _context_text(analyst_context, "analyst_label", "unknown"),
        "analyst_label_zh": _context_text(analyst_context, "analyst_label_zh", "未知"),
        "analyst_risk_level": analyst_risk_level,
        "analyst_risk_level_zh": _context_text(
            analyst_context,
            "analyst_risk_level_zh",
            "未知",
        ),
        "analyst_block_new_entries": analyst_block_new_entries,
        "analyst_upside": _context_float(analyst_context, "analyst_upside", np.nan),
        "recommendation_mean": _context_float(analyst_context, "recommendation_mean", np.nan),
        "recommendation_key": _context_text(analyst_context, "recommendation_key", ""),
        "number_of_analysts": _context_float(analyst_context, "number_of_analysts", np.nan),
        "target_mean_price": _context_float(analyst_context, "target_mean_price", np.nan),
        "analyst_note": _context_text(
            analyst_context,
            "analyst_note",
            "Analyst expectation data is unavailable; neutral score 50 is used.",
        ),
        "analyst_note_zh": _context_text(
            analyst_context,
            "analyst_note_zh",
            "分析师预期数据不可用；使用中性分数50。",
        ),
        "analyst_warning": _context_text(analyst_context, "analyst_warning", ""),
        "analyst_data_coverage": _context_float(analyst_context, "data_coverage", 0.0),
        "valuation_score": valuation_score,
        "valuation_label": _context_text(valuation_context, "valuation_label", "unknown"),
        "valuation_label_zh": _context_text(valuation_context, "valuation_label_zh", "未知"),
        "valuation_risk_level": valuation_risk_level,
        "valuation_risk_level_zh": _context_text(
            valuation_context,
            "valuation_risk_level_zh",
            "未知",
        ),
        "valuation_block_new_entries": valuation_block_new_entries,
        "valuation_forward_pe": _context_float(valuation_context, "valuation_forward_pe", np.nan),
        "valuation_trailing_pe": _context_float(valuation_context, "valuation_trailing_pe", np.nan),
        "valuation_peg_ratio": _context_float(valuation_context, "valuation_peg_ratio", np.nan),
        "valuation_free_cash_flow_yield": _context_float(
            valuation_context,
            "valuation_free_cash_flow_yield",
            np.nan,
        ),
        "valuation_market_cap": _context_float(
            valuation_context,
            "valuation_market_cap",
            np.nan,
        ),
        "valuation_growth_reference": _context_float(
            valuation_context,
            "valuation_growth_reference",
            np.nan,
        ),
        "valuation_profit_margin": _context_float(
            valuation_context,
            "valuation_profit_margin",
            np.nan,
        ),
        "valuation_note": _context_text(
            valuation_context,
            "valuation_note",
            "Valuation data is unavailable; neutral score 50 is used.",
        ),
        "valuation_note_zh": _context_text(
            valuation_context,
            "valuation_note_zh",
            "估值数据不可用；使用中性分数50。",
        ),
        "valuation_warning": _context_text(valuation_context, "valuation_warning", ""),
        "valuation_data_coverage": _context_float(valuation_context, "data_coverage", 0.0),
        "sector_score": sector_score,
        "sector_status": _context_text(sector_context, "sector_status", "unknown"),
        "sector": _context_text(sector_context, "sector", ""),
        "industry": _context_text(sector_context, "industry", ""),
        "sector_etf": _context_text(sector_context, "sector_etf", ""),
        "sector_note": _context_text(
            sector_context,
            "sector_note",
            "Sector ETF data is unavailable; neutral sector score 50 is used.",
        ),
        "sector_note_zh": _context_text(
            sector_context,
            "sector_note_zh",
            "板块ETF数据不可用，使用中性板块分数50。",
        ),
        "sector_warning": _context_text(sector_context, "sector_warning", ""),
        "sector_trend_score": _context_float(sector_context, "sector_trend_score", 50.0),
        "sector_relative_strength": _context_float(
            sector_context,
            "sector_relative_strength",
            np.nan,
        ),
        "fundamental_quality": fundamental_quality,
        "fundamental_quality_zh": _context_text(
            fundamental_context,
            "fundamental_quality_zh",
            "未知",
        ),
        "fundamental_note": _context_text(
            fundamental_context,
            "fundamental_note",
            "Fundamental data is unavailable or too sparse; neutral score 50 is used.",
        ),
        "fundamental_note_zh": _context_text(
            fundamental_context,
            "fundamental_note_zh",
            "基本面数据不可用或字段太少；使用中性分数50。",
        ),
        "fundamental_warning": _context_text(fundamental_context, "fundamental_warning", ""),
        "fundamental_data_coverage": _context_float(fundamental_context, "data_coverage", 0.0),
        "revenue_growth": _context_float(fundamental_context, "revenue_growth", np.nan),
        "earnings_growth": _context_float(fundamental_context, "earnings_growth", np.nan),
        "profit_margin": _context_float(fundamental_context, "profit_margin", np.nan),
        "return_on_equity": _context_float(fundamental_context, "return_on_equity", np.nan),
        "free_cash_flow": _context_float(fundamental_context, "free_cash_flow", np.nan),
        "forward_pe": _context_float(fundamental_context, "forward_pe", np.nan),
        "peg_ratio": _context_float(fundamental_context, "peg_ratio", np.nan),
        "debt_to_equity": _context_float(fundamental_context, "debt_to_equity", np.nan),
        "fundamental_cash_flow_quality": _context_float(
            fundamental_context, "cash_flow_quality_score", np.nan
        ),
        "fundamental_gross_margin": _context_float(fundamental_context, "gross_margin", np.nan),
        "fundamental_operating_margin": _context_float(
            fundamental_context, "operating_margin", np.nan
        ),
        "fundamental_fcf_margin": _context_float(fundamental_context, "fcf_margin", np.nan),
        "fundamental_cash_conversion": _context_float(
            fundamental_context, "cash_conversion", np.nan
        ),
        "fundamental_net_debt_to_equity": _context_float(
            fundamental_context, "net_debt_to_equity", np.nan
        ),
        "fundamental_trend_status": _context_text(
            fundamental_context, "trend_status", "insufficient_history"
        ),
        "fundamental_trend_direction": _context_text(
            fundamental_context, "trend_direction", "unknown"
        ),
        "fundamental_trend_direction_zh": _context_text(
            fundamental_context, "trend_direction_zh", "未知"
        ),
        "fundamental_trend_score": _context_float(fundamental_context, "trend_score", np.nan),
        "fundamental_gross_margin_trend": _context_text(
            fundamental_context, "gross_margin_trend", "unknown"
        ),
        "fundamental_operating_margin_trend": _context_text(
            fundamental_context, "operating_margin_trend", "unknown"
        ),
        "fundamental_net_margin_trend": _context_text(
            fundamental_context, "net_margin_trend", "unknown"
        ),
        "fundamental_fcf_margin_trend": _context_text(
            fundamental_context, "fcf_margin_trend", "unknown"
        ),
        "fundamental_revenue_growth_trend": _context_text(
            fundamental_context, "revenue_growth_trend", "unknown"
        ),
        "signal_timing": "generated_after_latest_close",
        "earliest_execution": "next_trading_session",
        "latest_price": latest_price,
        "entry_type": entry_type,
        "entry_price": entry_price,
        "current_distance_to_entry_pct": entry_distance_pct,
        "entry_distance_pct": entry_distance_pct,
        "entry_distance_note": entry_plan["entry_distance_note"],
        "entry_distance_note_zh": entry_plan["entry_distance_note_zh"],
        "chase_status": entry_plan["chase_status"],
        "chase_status_zh": entry_plan["chase_status_zh"],
        "entry_plan": entry_plan["entry_plan"],
        "entry_plan_zh": entry_plan["entry_plan_zh"],
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "risk_reward": risk_reward,
        "risk_per_share": risk_per_share,
        "backtest_execution_model": entry_backtest["backtest_execution_model"],
        "backtest_execution_model_zh": entry_backtest["backtest_execution_model_zh"],
        "backtest_time_stop_days": entry_backtest["backtest_time_stop_days"],
        "backtest_trailing_stop_trigger_r": entry_backtest["backtest_trailing_stop_trigger_r"],
        "backtest_trailing_stop_lock_r": entry_backtest["backtest_trailing_stop_lock_r"],
        "backtest_slippage_pct": entry_backtest["backtest_slippage_pct"],
        "backtest_avg_dollar_volume": entry_backtest["backtest_avg_dollar_volume"],
        "backtest_atr_ratio": entry_backtest["backtest_atr_ratio"],
        "backtest_liquidity_label": entry_backtest["backtest_liquidity_label"],
        "backtest_liquidity_label_zh": entry_backtest["backtest_liquidity_label_zh"],
        "backtest_volatility_label": entry_backtest["backtest_volatility_label"],
        "backtest_volatility_label_zh": entry_backtest["backtest_volatility_label_zh"],
        "backtest_execution_note": entry_backtest["backtest_execution_note"],
        "backtest_execution_note_zh": entry_backtest["backtest_execution_note_zh"],
        "regime_coverage_score": entry_backtest["regime_coverage_score"],
        "regime_coverage_level": entry_backtest["regime_coverage_level"],
        "regime_coverage_level_zh": entry_backtest["regime_coverage_level_zh"],
        "regime_coverage_regime_count": entry_backtest["regime_coverage_regime_count"],
        "regime_coverage_dominant_regime": entry_backtest["regime_coverage_dominant_regime"],
        "regime_coverage_dominant_regime_zh": entry_backtest[
            "regime_coverage_dominant_regime_zh"
        ],
        "regime_coverage_dominant_share": entry_backtest["regime_coverage_dominant_share"],
        "regime_coverage_note": entry_backtest["regime_coverage_note"],
        "regime_coverage_note_zh": entry_backtest["regime_coverage_note_zh"],
        "recent_backtest_score": entry_backtest["recent_backtest_score"],
        "recent_backtest_level": entry_backtest["recent_backtest_level"],
        "recent_backtest_level_zh": entry_backtest["recent_backtest_level_zh"],
        "recent_backtest_trade_count": entry_backtest["recent_backtest_trade_count"],
        "recent_backtest_win_rate": entry_backtest["recent_backtest_win_rate"],
        "recent_backtest_average_return": entry_backtest["recent_backtest_average_return"],
        "recent_backtest_return_delta": entry_backtest["recent_backtest_return_delta"],
        "recent_backtest_note": entry_backtest["recent_backtest_note"],
        "recent_backtest_note_zh": entry_backtest["recent_backtest_note_zh"],
        "backtest_decay_score": entry_backtest["backtest_decay_score"],
        "backtest_decay_level": entry_backtest["backtest_decay_level"],
        "backtest_decay_level_zh": entry_backtest["backtest_decay_level_zh"],
        "backtest_decay_early_trade_count": entry_backtest["backtest_decay_early_trade_count"],
        "backtest_decay_late_trade_count": entry_backtest["backtest_decay_late_trade_count"],
        "backtest_decay_early_win_rate": entry_backtest["backtest_decay_early_win_rate"],
        "backtest_decay_late_win_rate": entry_backtest["backtest_decay_late_win_rate"],
        "backtest_decay_early_average_return": entry_backtest[
            "backtest_decay_early_average_return"
        ],
        "backtest_decay_late_average_return": entry_backtest[
            "backtest_decay_late_average_return"
        ],
        "backtest_decay_win_rate_delta": entry_backtest["backtest_decay_win_rate_delta"],
        "backtest_decay_average_return_delta": entry_backtest[
            "backtest_decay_average_return_delta"
        ],
        "backtest_decay_note": entry_backtest["backtest_decay_note"],
        "backtest_decay_note_zh": entry_backtest["backtest_decay_note_zh"],
        "breakout_trade_count": entry_backtest["breakout_trade_count"],
        "breakout_sample_quality": entry_backtest["breakout_sample_quality"],
        "breakout_sample_quality_zh": entry_backtest["breakout_sample_quality_zh"],
        "breakout_win_rate": entry_backtest["breakout_win_rate"],
        "breakout_target_hit_rate": entry_backtest["breakout_target_hit_rate"],
        "breakout_stop_hit_rate": entry_backtest["breakout_stop_hit_rate"],
        "breakout_trailing_stop_hit_rate": entry_backtest["breakout_trailing_stop_hit_rate"],
        "breakout_average_gain": entry_backtest["breakout_average_gain"],
        "breakout_average_loss": entry_backtest["breakout_average_loss"],
        "breakout_average_return": entry_backtest["breakout_average_return"],
        "pullback_trade_count": entry_backtest["pullback_trade_count"],
        "pullback_sample_quality": entry_backtest["pullback_sample_quality"],
        "pullback_sample_quality_zh": entry_backtest["pullback_sample_quality_zh"],
        "pullback_win_rate": entry_backtest["pullback_win_rate"],
        "pullback_target_hit_rate": entry_backtest["pullback_target_hit_rate"],
        "pullback_stop_hit_rate": entry_backtest["pullback_stop_hit_rate"],
        "pullback_trailing_stop_hit_rate": entry_backtest["pullback_trailing_stop_hit_rate"],
        "pullback_average_gain": entry_backtest["pullback_average_gain"],
        "pullback_average_loss": entry_backtest["pullback_average_loss"],
        "pullback_average_return": entry_backtest["pullback_average_return"],
        "entry_backtest_note": entry_backtest["entry_backtest_note"],
        "entry_backtest_note_zh": entry_backtest["entry_backtest_note_zh"],
        "backtest_reliability_level": backtest_reliability["level"],
        "backtest_reliability_level_zh": backtest_reliability["level_zh"],
        "backtest_reliability_note": backtest_reliability["note"],
        "backtest_reliability_note_zh": backtest_reliability["note_zh"],
        "backtest_trust_score": backtest_trust["score"],
        "backtest_trust_level": backtest_trust["level"],
        "backtest_trust_level_zh": backtest_trust["level_zh"],
        "backtest_sample_score": backtest_trust["sample_score"],
        "backtest_liquidity_score": backtest_trust["liquidity_score"],
        "backtest_slippage_score": backtest_trust["slippage_score"],
        "backtest_return_evidence_score": backtest_trust["return_evidence_score"],
        "backtest_trust_note": backtest_trust["note"],
        "backtest_trust_note_zh": backtest_trust["note_zh"],
        "price_health_score": price_health["score"],
        "price_health_level": price_health["level"],
        "price_health_level_zh": price_health["level_zh"],
        "price_health_penalty": price_health["penalty"],
        "price_health_issue_count": price_health["issue_count"],
        "price_max_calendar_gap_days": price_health["max_calendar_gap_days"],
        "price_large_gap_count": price_health["large_gap_count"],
        "price_zero_volume_days": price_health["zero_volume_days"],
        "price_missing_ohlcv_rows": price_health["missing_ohlcv_rows"],
        "price_extreme_return_count": price_health["extreme_return_count"],
        "price_high_low_inversion_count": price_health["high_low_inversion_count"],
        "price_adjustment_anomaly_count": price_health["adjustment_anomaly_count"],
        "price_health_note": price_health["note"],
        "price_health_note_zh": price_health["note_zh"],
        "support": support,
        "resistance": resistance,
        "trend_ma": trend_ma,
        "trend_distance": trend_distance,
        "trend_slope": trend_slope,
        "momentum": momentum,
        "atr": atr,
        "volume_ratio": volume_ratio,
        "history_days": history_days,
        "passes_universe": bool(score_snapshot["passes_universe"]),
        "score_date": score_snapshot["score_date"],
        "score": score_snapshot["score"],
        "score_percentile": score_snapshot["score_percentile"],
        "market_status": _context_text(market_context, "market_status", "neutral"),
        "market_note": _context_text(
            market_context,
            "note",
            "Market data unavailable; neutral score 50 is used.",
        ),
        "market_note_zh": _context_text(
            market_context,
            "note_zh",
            "大盘数据不可用，使用中性分数50。",
        ),
        "market_warning": _context_text(market_context, "warning", ""),
        "event_risk_level": event_risk_level,
        "event_risk_level_zh": _context_text(event_risk_context, "event_risk_level_zh", "未知"),
        "next_earnings_date": _context_text(event_risk_context, "next_earnings_date", ""),
        "days_until_earnings": _context_float(event_risk_context, "days_until_earnings", np.nan),
        "last_earnings_date": _context_text(event_risk_context, "last_earnings_date", ""),
        "days_since_earnings": _context_float(event_risk_context, "days_since_earnings", np.nan),
        "event_risk_note": _context_text(
            event_risk_context,
            "event_risk_note",
            "Upcoming earnings date is unavailable; no event-risk downgrade is applied.",
        ),
        "event_risk_note_zh": _context_text(
            event_risk_context,
            "event_risk_note_zh",
            "暂时无法取得下一次财报日期；不进行事件风险降级。",
        ),
        "event_risk_warning": _context_text(event_risk_context, "event_risk_warning", ""),
        "relative_strength_note": _context_text(
            relative_strength_context,
            "note",
            "Relative strength data unavailable; neutral score 50 is used.",
        ),
        "relative_strength_note_zh": _context_text(
            relative_strength_context,
            "note_zh",
            "相对强弱数据不可用，使用中性分数50。",
        ),
        "relative_strength_warning": _context_text(relative_strength_context, "warning", ""),
        "vs_spy_return": _context_float(relative_strength_context, "vs_spy_return", np.nan),
        "vs_qqq_return": _context_float(relative_strength_context, "vs_qqq_return", np.nan),
        "rationale": rationale,
        "rationale_zh": rationale_zh,
        "action_explanation": action_explanation,
        "action_explanation_zh": action_explanation_zh,
        "plain_summary": plain_summary,
        "plain_summary_zh": plain_summary_zh,
    }


def _average_true_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int,
) -> pd.Series:
    previous_close = close.shift(1)
    ranges = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    )
    true_range = ranges.max(axis=1)
    return true_range.rolling(window).mean()


def _moving_average_slope(close: pd.Series, window: int) -> float:
    moving_average = close.rolling(window).mean()
    lag = max(5, min(20, window // 2))
    current = _last_valid(moving_average)
    previous = _last_valid(moving_average.shift(lag))
    return _safe_ratio(current, previous) - 1.0


def _window_return(close: pd.Series, window: int) -> float:
    if len(close) <= window:
        return np.nan
    previous = float(close.iloc[-window - 1])
    current = float(close.iloc[-1])
    return _safe_ratio(current, previous) - 1.0


def _pullback_entry(
    latest_price: float,
    trend_ma: float,
    support: float,
    entry_buffer_pct: float,
) -> float:
    candidates = [value for value in [trend_ma, support * (1.0 + entry_buffer_pct)] if _is_finite(value)]
    if not candidates:
        return latest_price
    below_or_near = [value for value in candidates if value <= latest_price * 1.02]
    if below_or_near:
        return max(below_or_near)
    return min(candidates)


def _trading_rule_note(spec: HorizonSpec, trading_rules: TradingRules) -> str:
    return (
        f"{spec.zh_label}/{spec.name} uses {trading_rules.preferred_entry_style} entries, "
        f"ATR stop {spec.atr_stop_multiple:.2f}x, target {spec.target_r_multiple:.2f}R, "
        f"max chase {spec.max_chase_pct:.2%}, time stop {spec.time_stop_days} days, "
        f"trailing trigger {spec.trailing_stop_trigger_r:.2f}R, "
        f"and sell rule: {trading_rules.sell_rule}."
    )


def _trading_rule_note_zh(spec: HorizonSpec, trading_rules: TradingRules) -> str:
    return (
        f"{spec.zh_label}使用{trading_rules.preferred_entry_style_zh}，"
        f"ATR止损{spec.atr_stop_multiple:.2f}倍，目标{spec.target_r_multiple:.2f}R，"
        f"最大追高{spec.max_chase_pct:.2%}，时间止损{spec.time_stop_days}天，"
        f"移动止损触发{spec.trailing_stop_trigger_r:.2f}R，"
        f"卖出规则：{trading_rules.sell_rule_zh}。"
    )


def _classify_action(
    latest_price: float,
    breakout_entry: float,
    pullback_entry: float,
    trend_distance: float,
    trend_slope: float,
    momentum: float,
    volume_ratio: float,
    history_days: int,
    spec: HorizonSpec,
    preferred_entry_style: str,
    passes_universe: bool,
) -> tuple[str, str, float]:
    if history_days < spec.min_history_days:
        return "wait_insufficient_history", "none", latest_price
    if not passes_universe:
        return "avoid_universe_filter", "none", latest_price

    uptrend = trend_distance > 0 and trend_slope >= 0
    positive_momentum = momentum > 0
    breakout_confirmed = latest_price >= breakout_entry and volume_ratio >= 1.0
    near_pullback = abs(_safe_ratio(latest_price, pullback_entry) - 1.0) <= 0.02
    extended = trend_distance > spec.max_chase_pct
    entry_style = str(preferred_entry_style or "balanced").lower().strip()

    if entry_style in {"pullback", "conservative"}:
        if uptrend and positive_momentum and near_pullback:
            return "entry_pullback_zone", "limit_pullback", min(latest_price, pullback_entry)
        if uptrend and positive_momentum and (extended or breakout_confirmed):
            return "wait_overextended", "limit_pullback", pullback_entry
        if uptrend and positive_momentum:
            return "watch_breakout_or_pullback", "limit_pullback", pullback_entry
        return "avoid_or_wait_downtrend", "none", latest_price

    if entry_style == "breakout":
        # Breakout/momentum style: a confirmed breakout (new high on volume) is executable
        # even when the stock is extended above its baseline -- that is the whole point of
        # chasing strength. Only refuse when the move is parabolic (far beyond the normal
        # chase cap), to avoid buying a blown-out spike.
        parabolic = trend_distance > spec.max_chase_pct * BREAKOUT_EXTENSION_MULTIPLE
        if breakout_confirmed and uptrend and positive_momentum and not parabolic:
            return "entry_breakout_confirmed", "market_or_limit", latest_price
        if uptrend and positive_momentum and not extended:
            return "watch_breakout_or_pullback", "stop_limit_breakout", breakout_entry
        if uptrend and positive_momentum and near_pullback:
            return "entry_pullback_zone", "limit_pullback", min(latest_price, pullback_entry)
        if uptrend and positive_momentum and extended:
            return "wait_overextended", "limit_pullback", pullback_entry
        return "avoid_or_wait_downtrend", "none", latest_price

    if breakout_confirmed and uptrend and positive_momentum:
        return "entry_breakout_confirmed", "market_or_limit", latest_price
    if uptrend and positive_momentum and near_pullback:
        return "entry_pullback_zone", "limit_pullback", min(latest_price, pullback_entry)
    if uptrend and positive_momentum and not extended:
        return "watch_breakout_or_pullback", "stop_limit_breakout", breakout_entry
    if uptrend and positive_momentum and extended:
        return "wait_overextended", "limit_pullback", pullback_entry
    return "avoid_or_wait_downtrend", "none", latest_price


def _stop_loss(
    entry_price: float,
    support: float,
    atr: float,
    atr_multiple: float,
    entry_buffer_pct: float,
) -> float:
    fallback_risk = max(entry_price * 0.05, 0.01)
    atr_stop = entry_price - atr * atr_multiple if _is_finite(atr) and atr > 0 else entry_price - fallback_risk
    structure_stop = support * (1.0 - entry_buffer_pct) if _is_finite(support) else np.nan
    candidates = [value for value in [atr_stop, structure_stop] if _is_finite(value) and value < entry_price]
    if not candidates:
        return max(entry_price - fallback_risk, 0.01)
    return max(candidates)


def _dynamic_slippage_profile(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    spec: HorizonSpec,
) -> dict[str, object]:
    avg_dollar_volume = _last_valid(
        (close.astype(float) * volume.astype(float)).rolling(SLIPPAGE_LIQUIDITY_WINDOW).mean()
    )
    atr = _last_valid(_average_true_range(high, low, close, spec.atr_window))
    latest_close = _last_valid(close)
    atr_ratio = _safe_ratio(atr, latest_close)

    if not _is_finite(avg_dollar_volume):
        liquidity_slippage = 0.003
        liquidity_label = "unknown_liquidity"
        liquidity_label_zh = "流动性未知"
    elif avg_dollar_volume >= 1_000_000_000:
        liquidity_slippage = 0.0005
        liquidity_label = "very_high_liquidity"
        liquidity_label_zh = "流动性很高"
    elif avg_dollar_volume >= 250_000_000:
        liquidity_slippage = 0.0008
        liquidity_label = "high_liquidity"
        liquidity_label_zh = "流动性较高"
    elif avg_dollar_volume >= 50_000_000:
        liquidity_slippage = 0.0015
        liquidity_label = "moderate_liquidity"
        liquidity_label_zh = "流动性中等"
    elif avg_dollar_volume >= 10_000_000:
        liquidity_slippage = 0.0030
        liquidity_label = "low_liquidity"
        liquidity_label_zh = "流动性偏低"
    else:
        liquidity_slippage = 0.0060
        liquidity_label = "very_low_liquidity"
        liquidity_label_zh = "流动性很低"

    if not _is_finite(atr_ratio):
        volatility_addon = 0.0010
        volatility_label = "unknown_volatility"
        volatility_label_zh = "波动率未知"
    elif atr_ratio >= 0.06:
        volatility_addon = 0.0030
        volatility_label = "very_high_volatility"
        volatility_label_zh = "波动率很高"
    elif atr_ratio >= 0.04:
        volatility_addon = 0.0020
        volatility_label = "high_volatility"
        volatility_label_zh = "波动率较高"
    elif atr_ratio >= 0.025:
        volatility_addon = 0.0010
        volatility_label = "moderate_volatility"
        volatility_label_zh = "波动率中等"
    else:
        volatility_addon = 0.0
        volatility_label = "normal_volatility"
        volatility_label_zh = "波动率正常"

    slippage_pct = min(
        max(liquidity_slippage + volatility_addon, MIN_BACKTEST_SLIPPAGE_PCT),
        MAX_BACKTEST_SLIPPAGE_PCT,
    )
    note = (
        f"Dynamic slippage uses {SLIPPAGE_LIQUIDITY_WINDOW}-day average dollar volume "
        f"and ATR ratio. Liquidity={liquidity_label}, volatility={volatility_label}."
    )
    note_zh = (
        f"动态滑点使用近{SLIPPAGE_LIQUIDITY_WINDOW}日平均成交额和ATR波动率。"
        f"流动性={liquidity_label_zh}，波动率={volatility_label_zh}。"
    )
    return {
        "slippage_pct": round(float(slippage_pct), 6),
        "avg_dollar_volume": avg_dollar_volume,
        "atr_ratio": atr_ratio,
        "liquidity_label": liquidity_label,
        "liquidity_label_zh": liquidity_label_zh,
        "volatility_label": volatility_label,
        "volatility_label_zh": volatility_label_zh,
        "note": note,
        "note_zh": note_zh,
    }


def _entry_backtest_summary(
    frame: pd.DataFrame,
    spec: HorizonSpec,
    entry_buffer_pct: float,
) -> dict[str, object]:
    close = frame["adj_close"].astype(float).reset_index(drop=True)
    open_price = (
        frame["open"].astype(float).fillna(close).reset_index(drop=True)
        if "open" in frame
        else close.copy()
    )
    high = frame["high"].astype(float).fillna(close).reset_index(drop=True)
    low = frame["low"].astype(float).fillna(close).reset_index(drop=True)
    volume = frame["volume"].astype(float).reset_index(drop=True)
    slippage_profile = _dynamic_slippage_profile(
        close=close,
        high=high,
        low=low,
        volume=volume,
        spec=spec,
    )
    slippage_pct = float(slippage_profile["slippage_pct"])

    min_index = max(spec.min_history_days, spec.lookback_days, spec.trend_window, spec.atr_window)
    breakout_outcomes: list[str] = []
    breakout_returns: list[float] = []
    breakout_regimes: list[str] = []
    pullback_outcomes: list[str] = []
    pullback_returns: list[float] = []
    pullback_regimes: list[str] = []
    all_trade_returns: list[tuple[int, float]] = []

    for index in range(min_index, len(frame) - 2):
        history_close = close.iloc[: index + 1]
        latest_price = float(close.iloc[index])
        trend_ma = _last_valid(history_close.rolling(spec.trend_window).mean())
        trend_distance = _safe_ratio(latest_price, trend_ma) - 1.0
        trend_slope = _moving_average_slope(history_close, spec.trend_window)
        momentum = _window_return(history_close, spec.momentum_window)
        avg_volume = _last_valid(volume.iloc[: index + 1].rolling(spec.volume_window).mean())
        volume_ratio = _safe_ratio(float(volume.iloc[index]), avg_volume)
        atr = _last_valid(
            _average_true_range(
                high.iloc[: index + 1],
                low.iloc[: index + 1],
                history_close,
                spec.atr_window,
            )
        )

        prior_start = max(0, index - spec.lookback_days)
        prior_high = high.iloc[prior_start:index]
        prior_low = low.iloc[prior_start:index]
        if prior_high.empty or prior_low.empty:
            continue

        support = float(prior_low.min())
        resistance = float(prior_high.max())
        breakout_entry = resistance * (1.0 + entry_buffer_pct)
        pullback_entry = _pullback_entry(latest_price, trend_ma, support, entry_buffer_pct)
        uptrend = trend_distance > 0 and trend_slope >= 0
        positive_momentum = momentum > 0
        atr_ratio = _safe_ratio(atr, latest_price)
        regime_label, _ = _backtest_regime_label(
            trend_distance=trend_distance,
            trend_slope=trend_slope,
            momentum=momentum,
            atr_ratio=atr_ratio,
        )
        future_end = index + 1 + spec.time_stop_days
        future_high = high.iloc[index + 1 : future_end]
        future_low = low.iloc[index + 1 : future_end]
        future_open = open_price.iloc[index + 1 : future_end]
        future_close = close.iloc[index + 1 : future_end]
        if future_close.empty:
            continue

        if (
            uptrend
            and positive_momentum
            and volume_ratio >= 1.0
            and latest_price >= breakout_entry
        ):
            executed_entry = _realistic_long_entry_price(
                planned_entry=latest_price,
                next_open=float(future_open.iloc[0]),
                slippage_pct=slippage_pct,
            )
            stop_loss = _stop_loss(
                entry_price=executed_entry,
                support=support,
                atr=atr,
                atr_multiple=spec.atr_stop_multiple,
                entry_buffer_pct=entry_buffer_pct,
            )
            outcome, trade_return = _trade_result(
                entry_price=executed_entry,
                stop_loss=stop_loss,
                target_r_multiple=spec.target_r_multiple,
                future_open=future_open,
                future_high=future_high,
                future_low=future_low,
                future_close=future_close,
                slippage_pct=slippage_pct,
                trailing_stop_trigger_r=spec.trailing_stop_trigger_r,
                trailing_stop_lock_r=spec.trailing_stop_lock_r,
            )
            breakout_outcomes.append(outcome)
            breakout_returns.append(trade_return)
            breakout_regimes.append(regime_label)
            all_trade_returns.append((index, trade_return))

        if (
            uptrend
            and positive_momentum
            and _is_finite(pullback_entry)
            and float(low.iloc[index]) <= pullback_entry <= float(high.iloc[index])
        ):
            executed_entry = _realistic_long_entry_price(
                planned_entry=pullback_entry,
                next_open=float(future_open.iloc[0]),
                slippage_pct=slippage_pct,
            )
            stop_loss = _stop_loss(
                entry_price=executed_entry,
                support=support,
                atr=atr,
                atr_multiple=spec.atr_stop_multiple,
                entry_buffer_pct=entry_buffer_pct,
            )
            outcome, trade_return = _trade_result(
                entry_price=executed_entry,
                stop_loss=stop_loss,
                target_r_multiple=spec.target_r_multiple,
                future_open=future_open,
                future_high=future_high,
                future_low=future_low,
                future_close=future_close,
                slippage_pct=slippage_pct,
                trailing_stop_trigger_r=spec.trailing_stop_trigger_r,
                trailing_stop_lock_r=spec.trailing_stop_lock_r,
            )
            pullback_outcomes.append(outcome)
            pullback_returns.append(trade_return)
            pullback_regimes.append(regime_label)
            all_trade_returns.append((index, trade_return))

    breakout_win_rate = _win_rate(breakout_outcomes)
    breakout_target_hit_rate = _outcome_rate(breakout_outcomes, "target_hit")
    breakout_stop_hit_rate = _outcome_rate(breakout_outcomes, "stop_hit")
    breakout_trailing_stop_hit_rate = _outcome_rate(breakout_outcomes, "trailing_stop")
    breakout_average_gain = _average_gain(breakout_returns)
    breakout_average_loss = _average_loss(breakout_returns)
    breakout_average_return = _average_return(breakout_returns)
    breakout_sample_quality, breakout_sample_quality_zh = _sample_quality(len(breakout_outcomes))
    pullback_win_rate = _win_rate(pullback_outcomes)
    pullback_target_hit_rate = _outcome_rate(pullback_outcomes, "target_hit")
    pullback_stop_hit_rate = _outcome_rate(pullback_outcomes, "stop_hit")
    pullback_trailing_stop_hit_rate = _outcome_rate(pullback_outcomes, "trailing_stop")
    pullback_average_gain = _average_gain(pullback_returns)
    pullback_average_loss = _average_loss(pullback_returns)
    pullback_average_return = _average_return(pullback_returns)
    pullback_sample_quality, pullback_sample_quality_zh = _sample_quality(len(pullback_outcomes))
    regime_coverage = _regime_coverage_profile(breakout_regimes + pullback_regimes)
    recent_backtest = _recent_backtest_profile(all_trade_returns)
    backtest_decay = _backtest_decay_profile(all_trade_returns)
    execution_model = "next_open_dynamic_slippage"
    execution_model_zh = "下一交易日开盘并计入动态滑点"
    execution_note = (
        "Backtest assumes signals are generated after the close, entries execute at the next "
        f"session open with {_format_percent(slippage_pct)} dynamic slippage, and gap-down stops "
        f"exit at the open instead of the ideal stop price. Trades use a maximum "
        f"{spec.time_stop_days}-session time stop before timeout exit. Trailing stop activates "
        f"after {spec.trailing_stop_trigger_r:.2f}R and locks {spec.trailing_stop_lock_r:.2f}R. "
        f"{slippage_profile['note']}"
    )
    execution_note_zh = (
        "回测假设信号在收盘后生成，买入按下一交易日开盘价并计入"
        f"{_format_percent(slippage_pct)}动态滑点；如果止损日跳空低开，"
        f"按开盘价退出，而不是按理想止损价退出。每笔交易最多持有"
        f"{spec.time_stop_days}个交易日，超过后按超时退出处理。移动止损在"
        f"{spec.trailing_stop_trigger_r:.2f}R后触发，并锁定{spec.trailing_stop_lock_r:.2f}R。"
        f"{slippage_profile['note_zh']}"
    )
    note = (
        f"Breakout sample={len(breakout_outcomes)}, quality={breakout_sample_quality}, "
        f"win_rate={_format_percent(breakout_win_rate)}, "
        f"target_hit={_format_percent(breakout_target_hit_rate)}, "
        f"stop_hit={_format_percent(breakout_stop_hit_rate)}, "
        f"trailing_stop={_format_percent(breakout_trailing_stop_hit_rate)}, "
        f"avg_gain={_format_percent(breakout_average_gain)}, "
        f"avg_loss={_format_percent(breakout_average_loss)}, "
        f"avg_return={_format_percent(breakout_average_return)}; "
        f"pullback sample={len(pullback_outcomes)}, quality={pullback_sample_quality}, "
        f"win_rate={_format_percent(pullback_win_rate)}, "
        f"target_hit={_format_percent(pullback_target_hit_rate)}, "
        f"stop_hit={_format_percent(pullback_stop_hit_rate)}, "
        f"trailing_stop={_format_percent(pullback_trailing_stop_hit_rate)}, "
        f"avg_gain={_format_percent(pullback_average_gain)}, "
        f"avg_loss={_format_percent(pullback_average_loss)}, "
        f"avg_return={_format_percent(pullback_average_return)}; "
        f"regime coverage score={_format_number(regime_coverage['score'])}, "
        f"dominant regime={regime_coverage['dominant_regime']}; "
        f"recent strength score={_format_number(recent_backtest['score'])}, "
        f"recent avg return={_format_percent(recent_backtest['recent_average_return'])}; "
        f"decay score={_format_number(backtest_decay['score'])}, "
        f"late avg return={_format_percent(backtest_decay['late_average_return'])}. "
        f"{execution_note}"
    )
    note_zh = (
        f"突破样本={len(breakout_outcomes)}，可靠性={breakout_sample_quality_zh}，"
        f"胜率={_format_percent(breakout_win_rate)}，"
        f"目标价命中率={_format_percent(breakout_target_hit_rate)}，"
        f"止损命中率={_format_percent(breakout_stop_hit_rate)}，"
        f"移动止损命中率={_format_percent(breakout_trailing_stop_hit_rate)}，"
        f"平均盈利={_format_percent(breakout_average_gain)}，"
        f"平均亏损={_format_percent(breakout_average_loss)}，"
        f"平均收益={_format_percent(breakout_average_return)}；"
        f"回调样本={len(pullback_outcomes)}，可靠性={pullback_sample_quality_zh}，"
        f"胜率={_format_percent(pullback_win_rate)}，"
        f"目标价命中率={_format_percent(pullback_target_hit_rate)}，"
        f"止损命中率={_format_percent(pullback_stop_hit_rate)}，"
        f"移动止损命中率={_format_percent(pullback_trailing_stop_hit_rate)}，"
        f"平均盈利={_format_percent(pullback_average_gain)}，"
        f"平均亏损={_format_percent(pullback_average_loss)}，"
        f"平均收益={_format_percent(pullback_average_return)}；"
        f"行情覆盖分={_format_number(regime_coverage['score'])}，"
        f"主导行情={regime_coverage['dominant_regime_zh']}；"
        f"近期强度分={_format_number(recent_backtest['score'])}，"
        f"近期平均收益={_format_percent(recent_backtest['recent_average_return'])}；"
        f"衰退检查分={_format_number(backtest_decay['score'])}，"
        f"后半段平均收益={_format_percent(backtest_decay['late_average_return'])}。"
        f"{execution_note_zh}"
    )
    return {
        "backtest_execution_model": execution_model,
        "backtest_execution_model_zh": execution_model_zh,
        "backtest_time_stop_days": spec.time_stop_days,
        "backtest_trailing_stop_trigger_r": spec.trailing_stop_trigger_r,
        "backtest_trailing_stop_lock_r": spec.trailing_stop_lock_r,
        "backtest_slippage_pct": slippage_pct,
        "backtest_avg_dollar_volume": slippage_profile["avg_dollar_volume"],
        "backtest_atr_ratio": slippage_profile["atr_ratio"],
        "backtest_liquidity_label": slippage_profile["liquidity_label"],
        "backtest_liquidity_label_zh": slippage_profile["liquidity_label_zh"],
        "backtest_volatility_label": slippage_profile["volatility_label"],
        "backtest_volatility_label_zh": slippage_profile["volatility_label_zh"],
        "backtest_execution_note": execution_note,
        "backtest_execution_note_zh": execution_note_zh,
        "regime_coverage_score": regime_coverage["score"],
        "regime_coverage_level": regime_coverage["level"],
        "regime_coverage_level_zh": regime_coverage["level_zh"],
        "regime_coverage_regime_count": regime_coverage["regime_count"],
        "regime_coverage_dominant_regime": regime_coverage["dominant_regime"],
        "regime_coverage_dominant_regime_zh": regime_coverage["dominant_regime_zh"],
        "regime_coverage_dominant_share": regime_coverage["dominant_share"],
        "regime_coverage_note": regime_coverage["note"],
        "regime_coverage_note_zh": regime_coverage["note_zh"],
        "recent_backtest_score": recent_backtest["score"],
        "recent_backtest_level": recent_backtest["level"],
        "recent_backtest_level_zh": recent_backtest["level_zh"],
        "recent_backtest_trade_count": recent_backtest["recent_trade_count"],
        "recent_backtest_win_rate": recent_backtest["recent_win_rate"],
        "recent_backtest_average_return": recent_backtest["recent_average_return"],
        "recent_backtest_return_delta": recent_backtest["return_delta"],
        "recent_backtest_note": recent_backtest["note"],
        "recent_backtest_note_zh": recent_backtest["note_zh"],
        "backtest_decay_score": backtest_decay["score"],
        "backtest_decay_level": backtest_decay["level"],
        "backtest_decay_level_zh": backtest_decay["level_zh"],
        "backtest_decay_early_trade_count": backtest_decay["early_trade_count"],
        "backtest_decay_late_trade_count": backtest_decay["late_trade_count"],
        "backtest_decay_early_win_rate": backtest_decay["early_win_rate"],
        "backtest_decay_late_win_rate": backtest_decay["late_win_rate"],
        "backtest_decay_early_average_return": backtest_decay["early_average_return"],
        "backtest_decay_late_average_return": backtest_decay["late_average_return"],
        "backtest_decay_win_rate_delta": backtest_decay["win_rate_delta"],
        "backtest_decay_average_return_delta": backtest_decay["average_return_delta"],
        "backtest_decay_note": backtest_decay["note"],
        "backtest_decay_note_zh": backtest_decay["note_zh"],
        "breakout_trade_count": len(breakout_outcomes),
        "breakout_sample_quality": breakout_sample_quality,
        "breakout_sample_quality_zh": breakout_sample_quality_zh,
        "breakout_win_rate": breakout_win_rate,
        "breakout_target_hit_rate": breakout_target_hit_rate,
        "breakout_stop_hit_rate": breakout_stop_hit_rate,
        "breakout_trailing_stop_hit_rate": breakout_trailing_stop_hit_rate,
        "breakout_average_gain": breakout_average_gain,
        "breakout_average_loss": breakout_average_loss,
        "breakout_average_return": breakout_average_return,
        "pullback_trade_count": len(pullback_outcomes),
        "pullback_sample_quality": pullback_sample_quality,
        "pullback_sample_quality_zh": pullback_sample_quality_zh,
        "pullback_win_rate": pullback_win_rate,
        "pullback_target_hit_rate": pullback_target_hit_rate,
        "pullback_stop_hit_rate": pullback_stop_hit_rate,
        "pullback_trailing_stop_hit_rate": pullback_trailing_stop_hit_rate,
        "pullback_average_gain": pullback_average_gain,
        "pullback_average_loss": pullback_average_loss,
        "pullback_average_return": pullback_average_return,
        "entry_backtest_note": note,
        "entry_backtest_note_zh": note_zh,
    }


def _backtest_regime_label(
    trend_distance: float,
    trend_slope: float,
    momentum: float,
    atr_ratio: float,
) -> tuple[str, str]:
    if _is_finite(atr_ratio) and atr_ratio >= 0.07:
        return "volatile", "高波动"
    if trend_distance >= 0.08 and trend_slope > 0 and momentum > 0:
        return "strong_uptrend", "强趋势"
    if trend_distance > 0 and trend_slope >= 0 and momentum > 0:
        return "steady_uptrend", "稳定上升"
    if trend_distance < 0 or trend_slope < 0:
        return "weak_or_downtrend", "弱势或下跌"
    return "mixed", "震荡混合"


def _regime_label_zh(label: str) -> str:
    return {
        "strong_uptrend": "强趋势",
        "steady_uptrend": "稳定上升",
        "volatile": "高波动",
        "weak_or_downtrend": "弱势或下跌",
        "mixed": "震荡混合",
        "none": "无样本",
    }.get(label, "未知")


def _regime_coverage_profile(regime_labels: list[str]) -> dict[str, object]:
    total = len(regime_labels)
    if total == 0:
        return {
            "score": 20.0,
            "level": "no_sample",
            "level_zh": "无样本",
            "regime_count": 0,
            "dominant_regime": "none",
            "dominant_regime_zh": "无样本",
            "dominant_share": np.nan,
            "note": "No entry backtest samples are available, so regime coverage cannot be trusted.",
            "note_zh": "没有买点回测样本，因此无法判断行情覆盖度。",
        }

    counts = pd.Series(regime_labels).value_counts()
    regime_count = int(len(counts))
    dominant_regime = str(counts.index[0])
    dominant_share = float(counts.iloc[0] / total)

    sample_score = min(total / 40.0, 1.0) * 35.0
    diversity_score = min(regime_count / 3.0, 1.0) * 35.0
    concentration_score = max(0.0, (1.0 - dominant_share) / 0.55) * 30.0
    score = round(float(max(0.0, min(sample_score + diversity_score + concentration_score, 100.0))), 2)

    if score >= 75:
        level, level_zh = "broad", "覆盖较广"
    elif score >= 55:
        level, level_zh = "moderate", "覆盖一般"
    elif score >= 35:
        level, level_zh = "narrow", "覆盖偏窄"
    else:
        level, level_zh = "thin", "覆盖很弱"

    counts_text = ", ".join(f"{label}={int(count)}" for label, count in counts.items())
    counts_text_zh = "，".join(
        f"{_regime_label_zh(str(label))}={int(count)}" for label, count in counts.items()
    )
    note = (
        f"Regime coverage uses {total} entry samples across {regime_count} regime types; "
        f"dominant regime is {dominant_regime} at {_format_percent(dominant_share)}. "
        f"Counts: {counts_text}."
    )
    note_zh = (
        f"行情覆盖度使用{total}个买点样本，覆盖{regime_count}类行情；"
        f"主导行情是{_regime_label_zh(dominant_regime)}，占比{_format_percent(dominant_share)}。"
        f"分布：{counts_text_zh}。"
    )
    return {
        "score": score,
        "level": level,
        "level_zh": level_zh,
        "regime_count": regime_count,
        "dominant_regime": dominant_regime,
        "dominant_regime_zh": _regime_label_zh(dominant_regime),
        "dominant_share": round(float(dominant_share), 4),
        "note": note,
        "note_zh": note_zh,
    }


def _recent_backtest_profile(trade_returns: list[tuple[int, float]]) -> dict[str, object]:
    ordered_returns = [
        float(value)
        for _, value in sorted(trade_returns, key=lambda item: item[0])
        if _is_finite(value)
    ]
    total_count = len(ordered_returns)
    if total_count == 0:
        return {
            "score": 20.0,
            "level": "no_sample",
            "level_zh": "无样本",
            "recent_trade_count": 0,
            "recent_win_rate": np.nan,
            "recent_average_return": np.nan,
            "return_delta": np.nan,
            "note": "No recent entry samples are available, so recent backtest strength cannot be measured.",
            "note_zh": "没有近期买点样本，因此无法衡量近期回测强度。",
        }

    recent_count = min(total_count, max(5, int(math.ceil(total_count * 0.35))))
    recent_returns = ordered_returns[-recent_count:]
    overall_average = float(np.mean(ordered_returns))
    recent_average = float(np.mean(recent_returns))
    recent_win_rate = float(np.mean([value > 0 for value in recent_returns]))
    return_delta = recent_average - overall_average

    sample_component = min(recent_count / 10.0, 1.0) * 20.0
    win_component = recent_win_rate * 35.0
    return_component = max(0.0, min(35.0, 17.5 + recent_average * 500.0))
    delta_component = max(0.0, min(10.0, 5.0 + return_delta * 250.0))
    score = round(
        float(max(0.0, min(sample_component + win_component + return_component + delta_component, 100.0))),
        2,
    )

    if score >= 75:
        level, level_zh = "improving", "近期改善"
    elif score >= 55:
        level, level_zh = "stable", "近期稳定"
    elif score >= 35:
        level, level_zh = "weakening", "近期转弱"
    else:
        level, level_zh = "poor_recent_evidence", "近期证据较差"

    note = (
        f"Recent backtest strength uses the latest {recent_count} of {total_count} entry samples; "
        f"recent win rate={_format_percent(recent_win_rate)}, recent average return="
        f"{_format_percent(recent_average)}, return delta versus all samples="
        f"{_format_percent(return_delta)}."
    )
    note_zh = (
        f"近期回测强度使用最近{recent_count}笔、总共{total_count}笔买点样本；"
        f"近期胜率={_format_percent(recent_win_rate)}，近期平均收益="
        f"{_format_percent(recent_average)}，相对全部样本的收益差="
        f"{_format_percent(return_delta)}。"
    )
    return {
        "score": score,
        "level": level,
        "level_zh": level_zh,
        "recent_trade_count": recent_count,
        "recent_win_rate": round(float(recent_win_rate), 4),
        "recent_average_return": round(float(recent_average), 4),
        "return_delta": round(float(return_delta), 4),
        "note": note,
        "note_zh": note_zh,
    }


def _backtest_decay_profile(trade_returns: list[tuple[int, float]]) -> dict[str, object]:
    ordered_returns = [
        float(value)
        for _, value in sorted(trade_returns, key=lambda item: item[0])
        if _is_finite(value)
    ]
    total_count = len(ordered_returns)
    if total_count < 6:
        return {
            "score": 35.0 if total_count > 0 else 20.0,
            "level": "too_few_samples",
            "level_zh": "样本太少",
            "early_trade_count": total_count // 2,
            "late_trade_count": total_count - total_count // 2,
            "early_win_rate": np.nan,
            "late_win_rate": np.nan,
            "early_average_return": np.nan,
            "late_average_return": np.nan,
            "win_rate_delta": np.nan,
            "average_return_delta": np.nan,
            "note": "Too few entry samples are available to measure backtest decay reliably.",
            "note_zh": "买点样本太少，无法可靠衡量回测是否衰退。",
        }

    split_index = total_count // 2
    early_returns = ordered_returns[:split_index]
    late_returns = ordered_returns[split_index:]

    early_win_rate = float(np.mean([value > 0 for value in early_returns]))
    late_win_rate = float(np.mean([value > 0 for value in late_returns]))
    early_average = float(np.mean(early_returns))
    late_average = float(np.mean(late_returns))
    win_rate_delta = late_win_rate - early_win_rate
    average_return_delta = late_average - early_average

    sample_component = min(total_count / 30.0, 1.0) * 20.0
    late_win_component = late_win_rate * 25.0
    late_return_component = max(0.0, min(30.0, 15.0 + late_average * 500.0))
    win_delta_component = max(0.0, min(10.0, 5.0 + win_rate_delta * 20.0))
    return_delta_component = max(0.0, min(15.0, 7.5 + average_return_delta * 300.0))
    score = round(
        float(
            max(
                0.0,
                min(
                    sample_component
                    + late_win_component
                    + late_return_component
                    + win_delta_component
                    + return_delta_component,
                    100.0,
                ),
            )
        ),
        2,
    )

    if score >= 75:
        level, level_zh = "durable", "较稳定"
    elif score >= 55:
        level, level_zh = "acceptable", "可接受"
    elif score >= 35:
        level, level_zh = "decaying", "出现衰退"
    else:
        level, level_zh = "severe_decay", "明显衰退"

    note = (
        f"Backtest decay compares the first {len(early_returns)} samples with the latest "
        f"{len(late_returns)} samples; early win rate={_format_percent(early_win_rate)}, "
        f"late win rate={_format_percent(late_win_rate)}, early average return="
        f"{_format_percent(early_average)}, late average return={_format_percent(late_average)}, "
        f"late-minus-early return delta={_format_percent(average_return_delta)}."
    )
    note_zh = (
        f"回测衰退检查比较前半段{len(early_returns)}笔和后半段{len(late_returns)}笔样本；"
        f"前半段胜率={_format_percent(early_win_rate)}，后半段胜率={_format_percent(late_win_rate)}，"
        f"前半段平均收益={_format_percent(early_average)}，后半段平均收益={_format_percent(late_average)}，"
        f"后半段相对前半段收益差={_format_percent(average_return_delta)}。"
    )
    return {
        "score": score,
        "level": level,
        "level_zh": level_zh,
        "early_trade_count": len(early_returns),
        "late_trade_count": len(late_returns),
        "early_win_rate": round(float(early_win_rate), 4),
        "late_win_rate": round(float(late_win_rate), 4),
        "early_average_return": round(float(early_average), 4),
        "late_average_return": round(float(late_average), 4),
        "win_rate_delta": round(float(win_rate_delta), 4),
        "average_return_delta": round(float(average_return_delta), 4),
        "note": note,
        "note_zh": note_zh,
    }


def _trade_result(
    entry_price: float,
    stop_loss: float,
    target_r_multiple: float,
    future_open: pd.Series,
    future_high: pd.Series,
    future_low: pd.Series,
    future_close: pd.Series,
    slippage_pct: float = MIN_BACKTEST_SLIPPAGE_PCT,
    trailing_stop_trigger_r: float | None = None,
    trailing_stop_lock_r: float | None = None,
) -> tuple[str, float]:
    risk_per_share = entry_price - stop_loss
    if not _is_finite(risk_per_share) or risk_per_share <= 0:
        return "invalid", np.nan
    take_profit = entry_price + risk_per_share * target_r_multiple
    active_stop = stop_loss
    trailing_active = False
    trigger_r = float(trailing_stop_trigger_r) if _is_finite(trailing_stop_trigger_r) else np.nan
    lock_r = float(trailing_stop_lock_r) if _is_finite(trailing_stop_lock_r) else np.nan
    trailing_enabled = _is_finite(trigger_r) and trigger_r > 0 and _is_finite(lock_r)

    for open_value, high_value, low_value in zip(future_open, future_high, future_low):
        open_float = float(open_value)
        high_float = float(high_value)
        low_float = float(low_value)
        if open_float <= active_stop:
            exit_price = max(open_float * (1.0 - slippage_pct), 0.01)
            outcome = "gap_trailing_stop_hit" if trailing_active else "gap_stop_hit"
            return outcome, exit_price / entry_price - 1.0
        if low_float <= active_stop:
            exit_price = max(active_stop * (1.0 - slippage_pct), 0.01)
            if trailing_active:
                outcome = "trailing_stop_win" if exit_price > entry_price else "trailing_stop_loss"
            else:
                outcome = "stop_hit"
            return outcome, exit_price / entry_price - 1.0
        if high_float >= take_profit:
            exit_price = take_profit * (1.0 - slippage_pct)
            return "target_hit", exit_price / entry_price - 1.0
        if trailing_enabled and high_float >= entry_price + risk_per_share * trigger_r:
            active_stop = max(active_stop, entry_price + risk_per_share * lock_r)
            trailing_active = active_stop > stop_loss
    final_exit = float(future_close.iloc[-1]) * (1.0 - slippage_pct)
    final_return = final_exit / entry_price - 1.0
    return ("timeout_win" if final_return > 0 else "timeout_loss"), final_return


def _realistic_long_entry_price(
    planned_entry: float,
    next_open: float,
    slippage_pct: float = MIN_BACKTEST_SLIPPAGE_PCT,
) -> float:
    if _is_finite(next_open) and next_open > 0:
        raw_entry = float(next_open)
    else:
        raw_entry = float(planned_entry)
    return raw_entry * (1.0 + slippage_pct)


def _win_rate(outcomes: list[str]) -> float:
    if not outcomes:
        return np.nan
    wins = sum(
        outcome in {"target_hit", "timeout_win", "trailing_stop_win"}
        for outcome in outcomes
    )
    return float(wins / len(outcomes))


def _outcome_rate(outcomes: list[str], target_outcome: str) -> float:
    if not outcomes:
        return np.nan
    if target_outcome == "trailing_stop":
        return float(
            sum(
                outcome
                in {"trailing_stop_win", "trailing_stop_loss", "gap_trailing_stop_hit"}
                for outcome in outcomes
            )
            / len(outcomes)
        )
    if target_outcome == "stop_hit":
        return float(
            sum(
                outcome
                in {
                    "stop_hit",
                    "gap_stop_hit",
                    "trailing_stop_loss",
                    "trailing_stop_win",
                    "gap_trailing_stop_hit",
                }
                for outcome in outcomes
            )
            / len(outcomes)
        )
    return float(sum(outcome == target_outcome for outcome in outcomes) / len(outcomes))


def _average_gain(returns: list[float]) -> float:
    gains = [value for value in returns if _is_finite(value) and value > 0]
    if not gains:
        return np.nan
    return float(np.mean(gains))


def _average_loss(returns: list[float]) -> float:
    losses = [value for value in returns if _is_finite(value) and value < 0]
    if not losses:
        return np.nan
    return float(np.mean(losses))


def _average_return(returns: list[float]) -> float:
    values = [value for value in returns if _is_finite(value)]
    if not values:
        return np.nan
    return float(np.mean(values))


def _sample_quality(sample_count: int) -> tuple[str, str]:
    if sample_count >= 30:
        return "strong", "较可靠"
    if sample_count >= 15:
        return "moderate", "一般"
    if sample_count >= 5:
        return "weak", "偏弱"
    return "insufficient", "样本不足"


def _backtest_reliability(entry_backtest: dict[str, object]) -> dict[str, str]:
    breakout_count = int(entry_backtest["breakout_trade_count"])
    pullback_count = int(entry_backtest["pullback_trade_count"])
    total_count = breakout_count + pullback_count
    breakout_return = float(entry_backtest["breakout_average_return"])
    pullback_return = float(entry_backtest["pullback_average_return"])
    breakout_win = float(entry_backtest["breakout_win_rate"])
    pullback_win = float(entry_backtest["pullback_win_rate"])

    if total_count >= 50:
        level, level_zh = "strong", "较可靠"
    elif total_count >= 25:
        level, level_zh = "moderate", "一般"
    elif total_count >= 10:
        level, level_zh = "weak", "偏弱"
    else:
        level, level_zh = "insufficient", "样本不足"

    notes: list[str] = [f"total sample={total_count}"]
    notes_zh: list[str] = [f"总样本={total_count}"]
    if breakout_count > 0:
        notes.append(
            f"breakout sample={breakout_count}, win rate={_format_percent(breakout_win)}, avg return={_format_percent(breakout_return)}"
        )
        notes_zh.append(
            f"突破样本={breakout_count}，胜率={_format_percent(breakout_win)}，平均收益={_format_percent(breakout_return)}"
        )
    if pullback_count > 0:
        notes.append(
            f"pullback sample={pullback_count}, win rate={_format_percent(pullback_win)}, avg return={_format_percent(pullback_return)}"
        )
        notes_zh.append(
            f"回调样本={pullback_count}，胜率={_format_percent(pullback_win)}，平均收益={_format_percent(pullback_return)}"
        )

    if level == "insufficient":
        conclusion = "Do not rely on this backtest alone."
        conclusion_zh = "不能单独依赖该回测。"
    elif level == "weak":
        conclusion = "Use this backtest only as a weak reference."
        conclusion_zh = "该回测只能作为偏弱参考。"
    elif level == "moderate":
        conclusion = "This backtest is usable as supporting evidence, not as a standalone signal."
        conclusion_zh = "该回测可作为辅助证据，但不能单独作为信号。"
    else:
        conclusion = "This backtest has better sample support, but still needs current setup confirmation."
        conclusion_zh = "该回测样本支持较好，但仍需要当前结构确认。"

    return {
        "level": level,
        "level_zh": level_zh,
        "note": "; ".join(notes) + f". {conclusion}",
        "note_zh": "；".join(notes_zh) + f"。{conclusion_zh}",
    }


def _backtest_trust_profile(
    entry_backtest: dict[str, object],
    price_health: dict[str, object],
) -> dict[str, object]:
    breakout_count = int(entry_backtest["breakout_trade_count"])
    pullback_count = int(entry_backtest["pullback_trade_count"])
    total_count = breakout_count + pullback_count
    sample_score = min(total_count / 50.0, 1.0) * 100.0

    avg_dollar_volume = float(entry_backtest.get("backtest_avg_dollar_volume", np.nan))
    if not _is_finite(avg_dollar_volume):
        liquidity_score = 45.0
    elif avg_dollar_volume >= 1_000_000_000:
        liquidity_score = 100.0
    elif avg_dollar_volume >= 250_000_000:
        liquidity_score = 90.0
    elif avg_dollar_volume >= 50_000_000:
        liquidity_score = 75.0
    elif avg_dollar_volume >= 10_000_000:
        liquidity_score = 55.0
    else:
        liquidity_score = 30.0

    slippage_pct = float(entry_backtest.get("backtest_slippage_pct", np.nan))
    if not _is_finite(slippage_pct):
        slippage_score = 45.0
    else:
        slippage_score = 100.0 - (slippage_pct / MAX_BACKTEST_SLIPPAGE_PCT) * 70.0
        slippage_score = max(20.0, min(100.0, slippage_score))

    breakout_return = float(entry_backtest.get("breakout_average_return", np.nan))
    pullback_return = float(entry_backtest.get("pullback_average_return", np.nan))
    return_values = [value for value in [breakout_return, pullback_return] if _is_finite(value)]
    if return_values:
        average_return = float(np.mean(return_values))
        return_evidence_score = 50.0 + average_return * 500.0
        return_evidence_score = max(0.0, min(100.0, return_evidence_score))
    else:
        average_return = np.nan
        return_evidence_score = 35.0

    price_health_score = float(price_health.get("score", 50.0))
    regime_coverage_score = float(entry_backtest.get("regime_coverage_score", 50.0))
    recent_backtest_score = float(entry_backtest.get("recent_backtest_score", 50.0))
    backtest_decay_score = float(entry_backtest.get("backtest_decay_score", 50.0))
    score = (
        sample_score * 0.22
        + price_health_score * 0.18
        + liquidity_score * 0.14
        + slippage_score * 0.10
        + return_evidence_score * 0.14
        + regime_coverage_score * 0.07
        + recent_backtest_score * 0.07
        + backtest_decay_score * 0.08
    )
    score = round(float(max(0.0, min(score, 100.0))), 2)
    if score >= 80:
        level, level_zh = "high_trust", "可信度较高"
    elif score >= 65:
        level, level_zh = "usable", "可作为辅助"
    elif score >= 45:
        level, level_zh = "low_trust", "可信度偏低"
    else:
        level, level_zh = "not_enough_evidence", "证据不足"

    note = (
        f"Trust score blends sample count={total_count}, price health={price_health_score:.1f}, "
        f"avg dollar volume={_format_number(avg_dollar_volume)}, slippage={_format_percent(slippage_pct)}, "
        f"average entry return={_format_percent(average_return)}, and regime coverage="
        f"{regime_coverage_score:.1f}, recent strength={recent_backtest_score:.1f}, "
        f"decay check={backtest_decay_score:.1f}."
    )
    note_zh = (
        f"可信度分综合样本数={total_count}、价格健康分={price_health_score:.1f}、"
        f"平均成交额={_format_number(avg_dollar_volume)}、滑点={_format_percent(slippage_pct)}、"
        f"平均买点收益={_format_percent(average_return)}、行情覆盖分={regime_coverage_score:.1f}、"
        f"近期强度分={recent_backtest_score:.1f}、衰退检查分={backtest_decay_score:.1f}。"
    )
    return {
        "score": score,
        "level": level,
        "level_zh": level_zh,
        "sample_score": round(float(sample_score), 2),
        "liquidity_score": round(float(liquidity_score), 2),
        "slippage_score": round(float(slippage_score), 2),
        "return_evidence_score": round(float(return_evidence_score), 2),
        "regime_coverage_score": round(float(regime_coverage_score), 2),
        "recent_backtest_score": round(float(recent_backtest_score), 2),
        "backtest_decay_score": round(float(backtest_decay_score), 2),
        "note": note,
        "note_zh": note_zh,
    }


def _signal_score(
    trend_distance: float,
    trend_slope: float,
    momentum: float,
    volume_ratio: float,
    score_percentile: float,
    history_days: int,
    min_history_days: int,
    passes_universe: bool,
) -> float:
    if history_days < min_history_days or not passes_universe:
        return 0.0

    score = 0.0
    if trend_distance > 0:
        score += min(trend_distance / 0.10, 1.0) * 25.0
    if trend_slope > 0:
        score += min(trend_slope / 0.05, 1.0) * 15.0
    if momentum > 0:
        score += min(momentum / 0.12, 1.0) * 25.0
    if volume_ratio > 1.0:
        score += min((volume_ratio - 1.0) / 1.0, 1.0) * 10.0
    if _is_finite(score_percentile):
        score += score_percentile * 25.0
    else:
        score += 12.5
    return round(float(max(0.0, min(score, 100.0))), 2)


def _combined_signal_score(
    spec: HorizonSpec,
    technical_score: float,
    market_score: float,
    relative_strength_score: float,
    fundamental_score: float,
    analyst_score: float,
    valuation_score: float,
    sector_score: float,
) -> float:
    weights = {
        "short": (0.48, 0.18, 0.13, 0.07, 0.06, 0.06, 0.02),
        "medium": (0.36, 0.16, 0.13, 0.14, 0.06, 0.07, 0.08),
        "long": (0.25, 0.12, 0.09, 0.33, 0.07, 0.07, 0.07),
    }
    (
        technical_weight,
        market_weight,
        relative_weight,
        fundamental_weight,
        analyst_weight,
        valuation_weight,
        sector_weight,
    ) = weights[spec.name]
    value = (
        technical_weight * technical_score
        + market_weight * market_score
        + relative_weight * relative_strength_score
        + fundamental_weight * fundamental_score
        + analyst_weight * analyst_score
        + valuation_weight * valuation_score
        + sector_weight * sector_score
    )
    return round(float(max(0.0, min(value, 100.0))), 2)


def _apply_context_downgrade(
    action: str,
    entry_type: str,
    market_score: float,
    relative_strength_score: float,
    event_risk_level: str,
    event_block_new_entries: bool,
    sentiment_risk_level: str,
    sentiment_block_new_entries: bool,
    analyst_risk_level: str,
    analyst_block_new_entries: bool,
    valuation_risk_level: str,
    valuation_block_new_entries: bool,
    fundamental_score: float,
    fundamental_quality: str,
    sector_score: float,
    horizon: str,
) -> tuple[str, str]:
    if action in {"wait_insufficient_history", "avoid_universe_filter", "avoid_or_wait_downtrend"}:
        return action, entry_type
    if event_risk_level == "high" or event_block_new_entries:
        return "wait_event_risk", "deferred_event_risk_filter"
    if sentiment_risk_level == "high" or sentiment_block_new_entries:
        return "wait_sentiment_risk", "deferred_sentiment_filter"
    if analyst_risk_level == "high" or analyst_block_new_entries:
        return "wait_analyst_weak", "deferred_analyst_filter"
    if valuation_risk_level == "high" or valuation_block_new_entries:
        return "wait_valuation_rich", "deferred_valuation_filter"
    if horizon == "long" and fundamental_quality == "weak" and fundamental_score <= 40:
        return "wait_fundamental_weak", "deferred_fundamental_filter"
    if horizon in {"medium", "long"} and sector_score <= 40:
        return "wait_sector_weak", "deferred_sector_filter"
    if market_score <= 40:
        return "wait_market_weak", "deferred_market_filter"
    if relative_strength_score <= 40:
        return "wait_relative_weak", "deferred_relative_strength_filter"
    return action, entry_type


def _rationale(
    spec: HorizonSpec,
    action: str,
    trend_distance: float,
    trend_slope: float,
    momentum: float,
    volume_ratio: float,
    support: float,
    resistance: float,
    history_days: int,
    passes_universe: bool,
) -> str:
    if history_days < spec.min_history_days:
        return (
            f"Only {history_days} price rows are available; "
            f"{spec.name} horizon needs at least {spec.min_history_days}."
        )
    if not passes_universe:
        return "Ticker does not pass the configured universe filter."

    trend_text = "above" if trend_distance > 0 else "below"
    slope_text = "rising" if trend_slope >= 0 else "falling"
    momentum_text = "positive" if momentum > 0 else "negative"
    volume_text = "above average" if volume_ratio >= 1 else "below average"
    return (
        f"{action}; price is {trend_text} the {spec.trend_window}-day trend, "
        f"trend is {slope_text}, {spec.momentum_window}-day momentum is {momentum_text}, "
        f"volume is {volume_text}, support is {_format_number(support)}, "
        f"resistance is {_format_number(resistance)}."
    )


def _rationale_zh(
    spec: HorizonSpec,
    action: str,
    trend_distance: float,
    trend_slope: float,
    momentum: float,
    volume_ratio: float,
    support: float,
    resistance: float,
    history_days: int,
    passes_universe: bool,
) -> str:
    if history_days < spec.min_history_days:
        return f"当前只有{history_days}条价格数据，{spec.zh_label}至少需要{spec.min_history_days}条。"
    if not passes_universe:
        return "该股票没有通过当前流动性或历史数据过滤。"

    trend_text = "高于" if trend_distance > 0 else "低于"
    slope_text = "上行" if trend_slope >= 0 else "下行"
    momentum_text = "为正" if momentum > 0 else "为负"
    volume_text = "高于均量" if volume_ratio >= 1 else "低于均量"
    return (
        f"{action}; 价格{trend_text}{spec.trend_window}日趋势线，趋势斜率{slope_text}，"
        f"{spec.momentum_window}日动量{momentum_text}，成交量{volume_text}，"
        f"支撑位{_format_number(support)}，压力位{_format_number(resistance)}。"
    )


def _action_explanation(
    action: str,
    original_action: str,
    market_score: float,
    relative_strength_score: float,
    event_risk_level: str,
    days_until_earnings: float,
    event_window: str,
    sentiment_risk_level: str,
    sentiment_score: float,
    analyst_risk_level: str,
    analyst_score: float,
    valuation_risk_level: str,
    valuation_score: float,
    fundamental_score: float,
    fundamental_quality: str,
) -> tuple[str, str]:
    event_days_text = (
        f"{int(days_until_earnings)} days"
        if _is_finite(days_until_earnings)
        else "an unknown number of days"
    )
    event_days_text_zh = (
        f"{int(days_until_earnings)}天"
        if _is_finite(days_until_earnings)
        else "未知天数"
    )
    explanations = {
        "entry_breakout_confirmed": (
            "Breakout is confirmed by trend, momentum, and volume.",
            "趋势、动量和成交量确认突破，可按计划入场。",
        ),
        "entry_pullback_zone": (
            "Price is near a pullback zone inside an uptrend.",
            "价格处于上升趋势中的回调区域，可小心分批。",
        ),
        "watch_breakout_or_pullback": (
            "Setup is constructive, but entry needs breakout or pullback confirmation.",
            "结构尚可，但需要突破或回调确认后再入场。",
        ),
        "wait_overextended": (
            "Price is extended above trend; wait for a better entry.",
            "价格相对趋势偏高，等待更好的回调位置。",
        ),
        "wait_market_weak": (
            f"Market score is weak at {market_score:.1f}; original action was {original_action}.",
            f"大盘分数偏弱({market_score:.1f})，原始动作为{original_action}，因此降级等待。",
        ),
        "wait_relative_weak": (
            f"Relative strength score is weak at {relative_strength_score:.1f}; original action was {original_action}.",
            f"相对强弱分数偏弱({relative_strength_score:.1f})，原始动作为{original_action}，因此降级等待。",
        ),
        "wait_event_risk": (
            (
                f"Event risk is {event_risk_level}; next known earnings are in "
                f"{event_days_text}; event window is {event_window}; original action was {original_action}."
            ),
            (
                f"事件风险为{event_risk_level}，距离已知下一次财报还有{event_days_text_zh}，"
                f"事件窗口为{event_window}，原始动作为{original_action}，因此降级等待。"
            ),
        ),
        "wait_sentiment_risk": (
            (
                f"News sentiment risk is {sentiment_risk_level} with score "
                f"{sentiment_score:.1f}; original action was {original_action}."
            ),
            (
                f"新闻情绪风险为{sentiment_risk_level}，分数{sentiment_score:.1f}，"
                f"原始动作为{original_action}，因此降级等待。"
            ),
        ),
        "wait_analyst_weak": (
            (
                f"Analyst expectation risk is {analyst_risk_level} with score "
                f"{analyst_score:.1f}; original action was {original_action}."
            ),
            (
                f"分析师预期风险为{analyst_risk_level}，分数{analyst_score:.1f}，"
                f"原始动作为{original_action}，因此降级等待。"
            ),
        ),
        "wait_valuation_rich": (
            (
                f"Valuation risk is {valuation_risk_level} with score "
                f"{valuation_score:.1f}; original action was {original_action}."
            ),
            (
                f"估值风险为{valuation_risk_level}，分数{valuation_score:.1f}，"
                f"原始动作为{original_action}，因此等待更合理估值。"
            ),
        ),
        "wait_fundamental_weak": (
            (
                f"Fundamental quality is {fundamental_quality} with score "
                f"{fundamental_score:.1f}; original action was {original_action}."
            ),
            (
                f"基本面质量为{fundamental_quality}，分数{fundamental_score:.1f}，"
                f"原始动作为{original_action}，因此长期信号降级等待。"
            ),
        ),
        "wait_sector_weak": (
            (
                f"Sector score is weak; original action was {original_action}, "
                "so medium or long-term entry is deferred."
            ),
            f"板块分数偏弱，原始动作为{original_action}，因此中期或长期信号降级等待。",
        ),
        "wait_insufficient_history": (
            "There is not enough price history for this horizon.",
            "当前历史数据不足，暂不生成有效入场信号。",
        ),
        "avoid_universe_filter": (
            "The ticker does not pass history or liquidity filters.",
            "该股票没有通过历史数据或流动性过滤。",
        ),
        "avoid_or_wait_downtrend": (
            "Trend or momentum is not strong enough.",
            "趋势或动量不足，暂时回避或等待。",
        ),
    }
    return explanations.get(
        action,
        ("No detailed action explanation available.", "暂无详细动作解释。"),
    )


def _plain_summary(
    spec: HorizonSpec,
    action: str,
    signal_score: float,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
) -> tuple[str, str]:
    price_plan = (
        f"entry around {_format_number(entry_price)}, stop around {_format_number(stop_loss)}, "
        f"target around {_format_number(take_profit)}"
    )
    price_plan_zh = (
        f"参考入场价约{_format_number(entry_price)}，止损约{_format_number(stop_loss)}，"
        f"目标价约{_format_number(take_profit)}"
    )
    prefix = f"{spec.label} ({spec.time_range}), score {signal_score:.1f}: "
    prefix_zh = f"{spec.zh_label}（{spec.time_range_zh}），分数{signal_score:.1f}："

    if action == "entry_breakout_confirmed":
        return (
            prefix + f"breakout is confirmed; {price_plan}.",
            prefix_zh + f"突破已经确认；{price_plan_zh}。",
        )
    if action == "entry_pullback_zone":
        return (
            prefix + f"price is in a pullback entry zone; {price_plan}.",
            prefix_zh + f"价格处在回调买点区域；{price_plan_zh}。",
        )
    if action == "watch_breakout_or_pullback":
        return (
            prefix + f"watch only; wait for breakout or pullback confirmation; {price_plan}.",
            prefix_zh + f"先观察，等待突破或回调确认；{price_plan_zh}。",
        )
    if action == "wait_overextended":
        return (
            prefix + f"wait; price is extended, so chasing is not preferred; {price_plan}.",
            prefix_zh + f"等待，价格偏高，不适合追高；{price_plan_zh}。",
        )
    if action == "wait_market_weak":
        return (
            prefix + "wait; market background is weak, so new entries are downgraded.",
            prefix_zh + "等待，大盘环境偏弱，新的入场信号被降级。",
        )
    if action == "wait_relative_weak":
        return (
            prefix + "wait; the stock is weak versus SPY/QQQ.",
            prefix_zh + "等待，该股票相对SPY/QQQ偏弱。",
        )
    if action == "wait_event_risk":
        return (
            prefix + "wait; earnings or event risk is too close for a fresh entry.",
            prefix_zh + "等待，财报或事件风险太近，不适合新开仓。",
        )
    if action == "wait_sentiment_risk":
        return (
            prefix + "wait; recent news or sentiment risk is elevated.",
            prefix_zh + "等待，近期新闻或情绪风险偏高，不适合新开仓。",
        )
    if action == "wait_analyst_weak":
        return (
            prefix + "wait; analyst expectations are weak or imply meaningful downside.",
            prefix_zh + "等待，分析师预期偏弱或暗示明显下行空间。",
        )
    if action == "wait_valuation_rich":
        return (
            prefix + "wait; valuation risk is elevated, so entry needs a better price.",
            prefix_zh + "等待，估值风险偏高，需要更合理的价格再考虑。",
        )
    if action == "wait_fundamental_weak":
        return (
            prefix + "wait; long-term company quality is too weak for a fresh entry.",
            prefix_zh + "等待，长期公司质量偏弱，不适合新开仓。",
        )
    if action == "wait_sector_weak":
        return (
            prefix + "wait; sector context is weak, so confirmation is not reliable enough.",
            prefix_zh + "等待，板块环境偏弱，当前确认度不够。",
        )
    if action == "wait_insufficient_history":
        return (
            prefix + "wait; there is not enough price history for this horizon.",
            prefix_zh + "等待，当前历史数据不足以支持该周期判断。",
        )
    if action == "avoid_universe_filter":
        return (
            prefix + "avoid; the ticker does not pass the current liquidity/history filter.",
            prefix_zh + "回避，该股票没有通过当前流动性或历史数据过滤。",
        )
    if action == "avoid_or_wait_downtrend":
        return (
            prefix + "avoid or wait; trend or momentum is not strong enough.",
            prefix_zh + "回避或等待，趋势或动量还不够强。",
        )
    return (
        prefix + f"watchlist only; {price_plan}.",
        prefix_zh + f"仅加入观察；{price_plan_zh}。",
    )


def _entry_plan(
    action: str,
    entry_type: str,
    latest_price: float,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    support: float,
    resistance: float,
    entry_distance_pct: float,
) -> dict[str, str]:
    distance_note, distance_note_zh = _entry_distance_note(entry_distance_pct)
    stop_text = _format_number(stop_loss)
    target_text = _format_number(take_profit)
    entry_text = _format_number(entry_price)
    support_text = _format_number(support)
    resistance_text = _format_number(resistance)
    latest_text = _format_number(latest_price)

    if action == "entry_breakout_confirmed":
        chase_status = "entry_allowed_if_price_holds"
        chase_status_zh = "价格守住突破区才可考虑"
        plan = (
            f"Breakout is confirmed. Entry reference is {entry_text}; stop is {stop_text}; "
            f"target is {target_text}. Avoid adding if price quickly falls back below the breakout area."
        )
        plan_zh = (
            f"突破已经确认。参考入场价{entry_text}，止损{stop_text}，目标{target_text}。"
            "如果价格很快跌回突破区域下方，不宜继续加仓。"
        )
    elif action == "entry_pullback_zone":
        chase_status = "pullback_entry_active"
        chase_status_zh = "回调买点已接近"
        plan = (
            f"Price is near the pullback entry zone around {entry_text}. Wait for price to hold "
            f"above support near {support_text}; stop is {stop_text}; target is {target_text}."
        )
        plan_zh = (
            f"价格接近约{entry_text}的回调买点。等待价格守住约{support_text}附近支撑；"
            f"止损{stop_text}，目标{target_text}。"
        )
    elif action == "watch_breakout_or_pullback" and entry_type == "stop_limit_breakout":
        chase_status = "do_not_chase_wait_for_breakout"
        chase_status_zh = "不要追高，等待突破触发"
        plan = (
            f"Current price is {latest_text}. Use a breakout trigger near {entry_text}, above "
            f"resistance around {resistance_text}. If triggered, stop is {stop_text} and target is {target_text}."
        )
        plan_zh = (
            f"当前价{latest_text}。等待价格突破，触发价约{entry_text}，压力位约{resistance_text}。"
            f"触发后止损{stop_text}，目标{target_text}。"
        )
    elif action == "wait_overextended":
        chase_status = "do_not_chase_wait_for_pullback"
        chase_status_zh = "不要追高，等待回调"
        plan = (
            f"Price is extended. Wait for a pullback toward {entry_text}; stop would be {stop_text} "
            f"and target would be {target_text} if the pullback stabilizes."
        )
        plan_zh = (
            f"价格偏高，不适合追。等待回调到约{entry_text}附近；如果企稳，"
            f"止损参考{stop_text}，目标参考{target_text}。"
        )
    elif action in {
        "wait_market_weak",
        "wait_relative_weak",
        "wait_event_risk",
        "wait_fundamental_weak",
        "wait_sector_weak",
    }:
        chase_status = "blocked_by_risk_filter"
        chase_status_zh = "被风险过滤阻挡"
        plan = (
            f"No fresh entry until the risk filter clears. If conditions improve, reassess entry near {entry_text}, "
            f"with stop near {stop_text} and target near {target_text}."
        )
        plan_zh = (
            f"风险过滤解除前不做新入场。条件改善后，再评估约{entry_text}附近的入场，"
            f"止损约{stop_text}，目标约{target_text}。"
        )
    elif action == "avoid_or_wait_downtrend":
        chase_status = "no_entry_trend_not_ready"
        chase_status_zh = "无入场，趋势未准备好"
        plan = (
            f"No entry plan yet. Wait for trend and momentum to improve before using resistance near "
            f"{resistance_text} or support near {support_text} as actionable levels."
        )
        plan_zh = (
            f"暂时没有可执行买点。等待趋势和动量改善后，再把约{resistance_text}压力位或"
            f"约{support_text}支撑位作为可操作位置。"
        )
    else:
        chase_status = "no_fresh_entry"
        chase_status_zh = "暂不新开仓"
        plan = (
            f"No clean entry trigger. Reassess if price structure improves around entry {entry_text}, "
            f"support {support_text}, or resistance {resistance_text}."
        )
        plan_zh = (
            f"暂时没有清晰入场触发。若价格结构在入场价{entry_text}、支撑{support_text}"
            f"或压力{resistance_text}附近改善，再重新评估。"
        )

    return {
        "entry_distance_note": distance_note,
        "entry_distance_note_zh": distance_note_zh,
        "chase_status": chase_status,
        "chase_status_zh": chase_status_zh,
        "entry_plan": plan,
        "entry_plan_zh": plan_zh,
    }


def _entry_distance_note(entry_distance_pct: float) -> tuple[str, str]:
    if not _is_finite(entry_distance_pct):
        return "Entry distance is unavailable.", "买点距离不可用。"
    distance = abs(entry_distance_pct)
    if distance <= 0.005:
        return (
            "Current price is very close to the planned entry.",
            "当前价非常接近计划买点。",
        )
    if entry_distance_pct > 0:
        return (
            f"Planned entry is {entry_distance_pct:.2%} above current price.",
            f"计划买点比当前价高{entry_distance_pct:.2%}。",
        )
    return (
        f"Current price is {distance:.2%} above the planned entry.",
        f"当前价比计划买点高{distance:.2%}。",
    )


def _context_float(context: object | None, name: str, default: float) -> float:
    if context is None:
        return default
    if isinstance(context, dict):
        value = context.get(name, default)
    else:
        value = getattr(context, name, default)
    return float(value) if _is_finite(value) else default


def _context_bool(context: object | None, name: str, default: bool) -> bool:
    if context is None:
        return default
    if isinstance(context, dict):
        value = context.get(name, default)
    else:
        value = getattr(context, name, default)
    return _coerce_bool(value, default)


def _context_text(context: object | None, name: str, default: str) -> str:
    if context is None:
        return default
    if isinstance(context, dict):
        value = context.get(name, default)
    else:
        value = getattr(context, name, default)
    return str(value) if value is not None else default


def _coerce_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n", ""}:
            return False
        return default
    if isinstance(value, (int, float, np.integer, np.floating)):
        if not _is_finite(value):
            return default
        return bool(value)
    return bool(value)


def _horizon_definition_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "horizon": spec.name,
                "label": spec.label,
                "zh_label": spec.zh_label,
                "time_range": spec.time_range,
                "time_range_zh": spec.time_range_zh,
            }
            for spec in HORIZON_SPECS.values()
        ]
    )


def _entry_backtest_frame(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    for column in [
        "backtest_slippage_pct",
        "backtest_atr_ratio",
        "regime_coverage_dominant_share",
        "recent_backtest_win_rate",
        "recent_backtest_average_return",
        "recent_backtest_return_delta",
        "backtest_decay_early_win_rate",
        "backtest_decay_late_win_rate",
        "backtest_decay_early_average_return",
        "backtest_decay_late_average_return",
        "backtest_decay_average_return_delta",
        "breakout_win_rate",
        "breakout_target_hit_rate",
        "breakout_stop_hit_rate",
        "breakout_average_gain",
        "breakout_average_loss",
        "breakout_average_return",
        "pullback_win_rate",
        "pullback_target_hit_rate",
        "pullback_stop_hit_rate",
        "pullback_average_gain",
        "pullback_average_loss",
        "pullback_average_return",
    ]:
        display[column] = display[column].map(_format_percent)
    return display


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not _is_finite(numerator) or not _is_finite(denominator) or denominator == 0:
        return np.nan
    return float(numerator / denominator)


def _last_valid(series: pd.Series) -> float:
    valid = series.replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return np.nan
    return float(valid.iloc[-1])


def _finite_or(value: float, fallback: float) -> float:
    return float(value) if _is_finite(value) else float(fallback)


def _is_finite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


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
    if isinstance(value, pd.Timestamp):
        return _format_date(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _format_signed_number(value: object) -> str:
    if pd.isna(value):
        return ""
    try:
        return f"{float(value):+.4f}"
    except (TypeError, ValueError):
        return str(value)


def _format_percent(value: object) -> str:
    if pd.isna(value):
        return "N/A"
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return str(value)


def _format_date(value: object) -> str:
    return pd.Timestamp(value).date().isoformat()
