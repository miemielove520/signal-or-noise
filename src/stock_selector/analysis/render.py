"""Markdown rendering of a single-ticker analysis."""

from __future__ import annotations

import pandas as pd

from ._common import (
    _format_date,
    _format_number,
    _format_percent,
    _format_signed_number,
    _markdown_table,
)
from .backtest import (
    _entry_backtest_frame,
)
from .horizons import (
    _horizon_definition_frame,
)


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
    lines: list[str] = []
    lines.extend(
        [
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
        ]
    )
    lines.extend(_final_decision_section(first))
    lines.extend(_priority_blockers_section(first))
    lines.extend(_horizon_alignment_section(first))
    lines.extend(_primary_decision_section(first))
    lines.extend(_high_probability_filter_section(screening_focus))
    lines.extend(_profile_trading_rules_section(screening_focus))
    lines.extend(_threshold_calibration_section(screening_focus))
    lines.extend(_calibrated_screening_section(screening_focus))
    lines.extend(_signal_review_feedback_section(screening_focus))
    lines.extend(_watchlist_plan_section(screening_focus))
    lines.extend(_confidence_section(first))
    lines.extend(_risk_breakdown_section(first))
    lines.extend(_data_quality_section(first))
    lines.extend(_plain_summary_section(analysis))
    lines.extend(_entry_plan_section(analysis))
    lines.extend(_backtest_reliability_section(analysis))
    lines.extend(_horizon_definition_section())
    lines.extend(_market_and_relative_strength_section(first))
    lines.extend(_sector_context_section(first))
    lines.extend(_fundamental_quality_section(first))
    lines.extend(_analyst_expectations_section(first))
    lines.extend(_valuation_risk_section(first))
    lines.extend(_news_sentiment_section(first))
    lines.extend(_event_risk_section(first))
    lines.extend(_signal_plan_section(analysis))
    lines.extend(_entry_backtest_section(analysis, first))
    lines.extend(_rationale_section(analysis))
    return "\n".join(lines).rstrip() + "\n"


def _final_decision_section(first: pd.Series) -> list[str]:
    lines: list[str] = []
    lines.extend(
        [
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
        ]
    )
    return lines


def _priority_blockers_section(first: pd.Series) -> list[str]:
    lines: list[str] = []
    lines.extend(
        [
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
        ]
    )
    return lines


def _horizon_alignment_section(first: pd.Series) -> list[str]:
    lines: list[str] = []
    lines.extend(
        [
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
        ]
    )
    return lines


def _primary_decision_section(first: pd.Series) -> list[str]:
    lines: list[str] = []
    lines.extend(
        [
            "## Primary Decision / 主要结论",
            "",
            f"- Decision / 结论: `{first['primary_decision']}` / {first['primary_decision_zh']}",
            f"- Focus horizon / 重点周期: `{first['decision_focus_horizon']}`",
            f"- Why / 为什么: {first['decision_reason']} / {first['decision_reason_zh']}",
            f"- What to wait for / 等什么: {first['decision_wait_for']} / {first['decision_wait_for_zh']}",
            f"- Invalidation / 判断失效条件: {first['decision_invalidation']} / {first['decision_invalidation_zh']}",
            "",
        ]
    )
    return lines


def _high_probability_filter_section(screening_focus: pd.Series) -> list[str]:
    lines: list[str] = []
    lines.extend(
        [
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
        ]
    )
    return lines


def _profile_trading_rules_section(screening_focus: pd.Series) -> list[str]:
    lines: list[str] = []
    lines.extend(
        [
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
        ]
    )
    return lines


def _threshold_calibration_section(screening_focus: pd.Series) -> list[str]:
    lines: list[str] = []
    lines.extend(
        [
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
        ]
    )
    return lines


def _calibrated_screening_section(screening_focus: pd.Series) -> list[str]:
    lines: list[str] = []
    lines.extend(
        [
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
        ]
    )
    return lines


def _signal_review_feedback_section(screening_focus: pd.Series) -> list[str]:
    lines: list[str] = []
    lines.extend(
        [
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
        ]
    )
    return lines


def _watchlist_plan_section(screening_focus: pd.Series) -> list[str]:
    lines: list[str] = []
    lines.extend(
        [
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
        ]
    )
    return lines


def _confidence_section(first: pd.Series) -> list[str]:
    lines: list[str] = []
    lines.extend(
        [
            "## Confidence / 置信度",
            "",
            f"- Confidence score / 置信度分数: `{_format_number(first['confidence_score'])}`",
            f"- Confidence level / 置信度等级: `{first['confidence_level']}` / `{first['confidence_level_zh']}`",
            f"- Confidence note / 置信度说明: {first['confidence_note']} / {first['confidence_note_zh']}",
            "",
        ]
    )
    return lines


def _risk_breakdown_section(first: pd.Series) -> list[str]:
    lines: list[str] = []
    lines.extend(
        [
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
        ]
    )
    return lines


def _data_quality_section(first: pd.Series) -> list[str]:
    lines: list[str] = []
    lines.extend(
        [
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
        ]
    )
    return lines


def _plain_summary_section(analysis: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    lines.extend(
        [
            "## Plain Summary / 简明结论",
            "",
        ]
    )
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
        ]
    )
    return lines


def _entry_plan_section(analysis: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    lines.extend(
        [
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
        ]
    )
    return lines


def _backtest_reliability_section(analysis: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    lines.extend(
        [
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
        ]
    )
    return lines


def _horizon_definition_section() -> list[str]:
    lines: list[str] = []
    lines.extend(
        [
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
        ]
    )
    return lines


def _market_and_relative_strength_section(first: pd.Series) -> list[str]:
    lines: list[str] = []
    lines.extend(
        [
            "## Market And Relative Strength / 大盘环境与相对强弱",
            "",
            f"- Market score / 大盘分数: `{_format_number(first['market_score'])}`",
            f"- Market status / 大盘状态: `{first['market_status']}`",
            f"- Market note / 大盘说明: {first['market_note']} / {first['market_note_zh']}",
            "",
        ]
    )
    return lines


def _sector_context_section(first: pd.Series) -> list[str]:
    lines: list[str] = []
    lines.extend(
        [
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
        ]
    )
    return lines


def _fundamental_quality_section(first: pd.Series) -> list[str]:
    lines: list[str] = []
    lines.extend(
        [
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
        ]
    )
    return lines


def _analyst_expectations_section(first: pd.Series) -> list[str]:
    lines: list[str] = []
    lines.extend(
        [
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
        ]
    )
    return lines


def _valuation_risk_section(first: pd.Series) -> list[str]:
    lines: list[str] = []
    lines.extend(
        [
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
        ]
    )
    return lines


def _news_sentiment_section(first: pd.Series) -> list[str]:
    lines: list[str] = []
    lines.extend(
        [
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
        ]
    )
    return lines


def _event_risk_section(first: pd.Series) -> list[str]:
    lines: list[str] = []
    lines.extend(
        [
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
        ]
    )
    return lines


def _signal_plan_section(analysis: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    lines.extend(
        [
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
        ]
    )
    return lines


def _entry_backtest_section(analysis: pd.DataFrame, first: pd.Series) -> list[str]:
    lines: list[str] = []
    lines.extend(
        [
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
    return lines


def _rationale_section(analysis: pd.DataFrame) -> list[str]:
    lines: list[str] = []
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
    return lines
