"""Console summary of a single-ticker analysis (used by ``run.py`` and ``stock-selector real``)."""

from __future__ import annotations

import pandas as pd

from .data_sources import DataReadinessReport
from .real_data import RealTickerAnalysisResult


def print_ticker_report(result: RealTickerAnalysisResult) -> None:
    """Print every section of the ticker analysis to stdout, in report order."""
    readiness = result.data_readiness
    first = result.analysis.iloc[0]
    screening_focus = result.analysis.sort_values("high_probability_score", ascending=False).iloc[0]
    _print_real_ticker_analysis_completed(result)
    _print_data_sources(result)
    _print_data_source_readiness(readiness)
    _print_horizon_definition()
    _print_final_decision(first)
    _print_priority_blockers(first)
    _print_horizon_alignment(first)
    _print_primary_decision(first)
    _print_high_probability_filter(screening_focus)
    _print_threshold_calibration(screening_focus)
    _print_calibrated_screening(screening_focus)
    _print_signal_review_feedback(screening_focus)
    _print_calibrated_watchlist_plan(screening_focus)
    _print_watchlist_plan(screening_focus)
    _print_confidence(first)
    _print_risk_breakdown(first)
    _print_data_quality(first)
    _print_market(first)
    _print_sector_context(first, result)
    _print_analyst_expectations(first)
    _print_valuation_risk(first)
    _print_news_sentiment(first)
    _print_event_risk(first)
    _print_plain_summary(result)
    _print_entry_plan(result)
    _print_backtest_reliability(result)
    _print_entry_backtest(first, result)
    _print_signal_summary(result)


def _print_real_ticker_analysis_completed(result: RealTickerAnalysisResult) -> None:
    print()
    print("Real ticker analysis completed.")
    print(f"Ticker: {result.ticker}")
    print(f"Requested period / 请求周期: {result.requested_period}")
    print(f"Effective period / 实际使用周期: {result.effective_period}")
    print(f"Auto period upgraded / 自动周期升级: {result.auto_period_upgraded}")
    print(f"Period note / 周期说明: {result.auto_period_upgrade_reason_zh}")
    print(f"Price rows: {len(result.prices)}")
    print(f"Report folder: {result.output_dir}")


def _print_data_sources(result: RealTickerAnalysisResult) -> None:
    print()
    print("Data sources / 数据源")
    for symbol, provider in result.data_sources.items():
        print(f"- {symbol}: {provider}")
    warnings = [
        f"{symbol}: {'; '.join(items)}"
        for symbol, items in result.data_source_warnings.items()
        if items
    ]
    if warnings:
        print("Data source warnings / 数据源提示")
        for warning in warnings:
            print(f"- {warning}")


def _print_data_source_readiness(readiness: DataReadinessReport) -> None:
    print()
    print("Data source readiness / 数据源准备度")
    print(
        "Overall / 总体: "
        f"{readiness.overall_status} / {readiness.overall_status_zh}, "
        f"score={readiness.overall_score:.2f}"
    )
    print(
        "Repair priority / 修复优先级: "
        f"{readiness.repair_priority} / {readiness.repair_priority_zh}"
    )
    print(
        "Price validation / 价格源验证: "
        f"{readiness.source_validation.get('status', 'unknown')} / "
        f"{readiness.source_validation.get('status_zh', '未知')}"
    )
    blockers_zh = "；".join(readiness.primary_blockers_zh) or "无"
    print(f"Primary data blockers / 主要数据卡点: {blockers_zh}")
    if readiness.overall_status in {"insufficient", "conflict_warning"}:
        print("Data needs repair / 数据需要修复")


def _print_horizon_definition() -> None:
    print()
    print("Horizon definition / 周期定义")
    print("These labels describe analysis style, not fixed holding periods.")
    print("这些分类描述的是分析风格，不是固定持有时间。")
    print("- short / 短期: short-term trading setup / 短线交易节奏")
    print("- medium / 中期: trend continuation setup / 趋势延续判断")
    print("- long / 长期: long-term quality and trend setup / 长期质量与趋势判断")


def _print_final_decision(first: pd.Series) -> None:
    print()
    print("Final decision / 最终执行结论")
    print(f"Final decision / 最终结论: {first.final_decision} / {first.final_decision_zh}")
    print(f"Final focus horizon / 最终重点周期: {first.final_focus_horizon} / {first.final_focus_horizon_zh}")
    print(f"Final score / 最终分数: {first.final_score:.2f}")
    print(
        "Final watchlist status / 最终观察状态: "
        f"{first.final_watchlist_status} / {first.final_watchlist_status_zh}"
    )
    print(f"Final reason / 最终原因: {first.final_reason_zh}")
    print(f"Final next step / 下一步: {first.final_next_step_zh}")


def _print_priority_blockers(first: pd.Series) -> None:
    print()
    print("Priority blockers / 主要卡点排序")
    print(f"Primary blocker / 第一卡点: {first.primary_blocker} / {first.primary_blocker_zh}")
    print(f"Priority blockers / 主要卡点: {first.priority_blockers_zh}")
    print(f"Blocker count / 卡点数量: {int(first.priority_blocker_count)}")
    print(f"Blocker note / 卡点说明: {first.priority_blocker_note_zh}")
    print(f"Primary blocker resolution / 第一卡点解除条件: {first.primary_blocker_resolution_zh}")
    print(f"Re-check trigger / 重新检查触发条件: {first.blocker_recheck_trigger_zh}")
    print(
        "Blocker resolution score / 卡点解除分数: "
        f"{first.blocker_resolution_score:.2f}/100, "
        f"{first.blocker_resolution_level} / {first.blocker_resolution_level_zh}"
    )
    print(f"Resolution gap / 解除差距: {first.blocker_resolution_gap_zh}")


def _print_horizon_alignment(first: pd.Series) -> None:
    print()
    print("Horizon alignment / 周期一致性")
    print(
        "Alignment / 一致性: "
        f"{first.horizon_alignment_label} / {first.horizon_alignment_label_zh}"
    )
    print(f"Alignment score / 一致性分数: {first.horizon_alignment_score:.2f}")
    print(
        "Horizon counts / 周期数量: "
        f"constructive={int(first.constructive_horizon_count)}, "
        f"weak={int(first.weak_horizon_count)}, "
        f"risk_wait={int(first.risk_wait_horizon_count)}"
    )
    print(f"Score spread / 分数差距: {first.horizon_signal_score_spread:.2f}")
    print(f"Alignment note / 一致性说明: {first.horizon_alignment_note_zh}")


def _print_primary_decision(first: pd.Series) -> None:
    print()
    print("Primary decision / 主要结论")
    print(f"Decision / 结论: {first.primary_decision} / {first.primary_decision_zh}")
    print(f"Focus horizon / 重点周期: {first.decision_focus_horizon}")
    print(f"Why / 为什么: {first.decision_reason_zh}")
    print(f"Wait for / 等什么: {first.decision_wait_for_zh}")
    print(f"Invalidation / 判断失效: {first.decision_invalidation_zh}")


def _print_high_probability_filter(screening_focus: pd.Series) -> None:
    print()
    print("High probability filter / 高概率筛选器")
    print(
        "Screening profile / 筛选规则: "
        f"{screening_focus.screening_profile} / {screening_focus.screening_profile_zh}"
    )
    print(
        "Final screening / 最终筛选结论: "
        f"{screening_focus.screening_action} / {screening_focus.screening_action_zh}"
    )
    print(f"Focus horizon / 重点周期: {screening_focus.horizon} / {screening_focus.horizon_zh_label}")
    print(f"High probability score / 高概率分数: {screening_focus.high_probability_score:.2f}")
    print(
        "High probability level / 高概率等级: "
        f"{screening_focus.high_probability_level} / {screening_focus.high_probability_level_zh}"
    )
    print(
        "Calibrated win probability / 校准后胜率估计: "
        f"{_format_optional_percent(screening_focus.calibrated_win_probability)}, "
        f"{screening_focus.calibrated_probability_level} / "
        f"{screening_focus.calibrated_probability_level_zh}"
    )
    print(
        "Probability calibration feedback / 概率校准反馈: "
        f"raw={_format_optional_percent(screening_focus.calibrated_win_probability_raw)}, "
        f"adjustment={_format_optional_percent(screening_focus.probability_calibration_adjustment)}, "
        f"source={screening_focus.probability_calibration_source_zh}, "
        f"sample={int(screening_focus.probability_calibration_sample_count)}"
    )
    print(
        "Probability calibration action / 概率校准动作: "
        f"{screening_focus.probability_calibration_action} / "
        f"{screening_focus.probability_calibration_action_zh}"
    )
    print(
        "Probability confidence / 概率置信度: "
        f"{screening_focus.calibrated_probability_confidence:.2f}, "
        f"{screening_focus.calibrated_probability_confidence_level} / "
        f"{screening_focus.calibrated_probability_confidence_level_zh}"
    )
    print(f"Probability note / 概率说明: {screening_focus.calibrated_probability_note_zh}")
    print(f"Quality gate passed / 是否通过质量门槛: {screening_focus.quality_gate_passed}")
    print(
        "Backtest used / 使用的买点回测: "
        f"{screening_focus.screening_backtest_entry_type}, "
        f"sample={screening_focus.screening_backtest_trade_count}, "
        f"win_rate={_format_optional_percent(screening_focus.screening_backtest_win_rate)}, "
        f"stop_hit={_format_optional_percent(screening_focus.screening_backtest_stop_hit_rate)}, "
        f"avg_return={_format_optional_percent(screening_focus.screening_backtest_average_return)}"
    )
    print(
        "Backtest trust gate / 回测可信度门槛: "
        f"{screening_focus.backtest_trust_score:.2f}, "
        f"{screening_focus.backtest_trust_level} / {screening_focus.backtest_trust_level_zh}"
    )
    print(
        "Stability gates / 稳定性门槛: "
        f"recent={screening_focus.recent_backtest_score:.2f} / "
        f"{screening_focus.recent_backtest_level_zh}, "
        f"decay={screening_focus.backtest_decay_score:.2f} / "
        f"{screening_focus.backtest_decay_level_zh}"
    )
    print(
        "Sample confidence / 样本置信度: "
        f"{screening_focus.sample_confidence_level} / {screening_focus.sample_confidence_level_zh}"
    )
    print(
        "Evidence strength / 证据强度: "
        f"{screening_focus.evidence_strength} / {screening_focus.evidence_strength_zh}"
    )
    print(f"Evidence note / 证据说明: {screening_focus.evidence_note_zh}")
    print(
        "Liquidity filter / 流动性过滤: "
        f"{screening_focus.liquidity_filter_passed}, "
        f"{screening_focus.liquidity_filter_reason_zh}"
    )
    print(
        "Entry readiness / 买点可执行性: "
        f"{screening_focus.entry_readiness_gate_passed}, "
        f"{screening_focus.entry_readiness_status} / "
        f"{screening_focus.entry_readiness_status_zh}, "
        f"score={screening_focus.entry_readiness_score:.2f}"
    )
    print(f"Entry readiness note / 买点可执行性说明: {screening_focus.entry_readiness_note_zh}")
    print(
        "Trade plan quality / 交易计划质量: "
        f"{screening_focus.trade_plan_quality_gate_passed}, "
        f"{screening_focus.trade_plan_quality_status} / "
        f"{screening_focus.trade_plan_quality_status_zh}, "
        f"score={screening_focus.trade_plan_quality_score:.2f}"
    )
    print(f"Trade plan quality note / 交易计划质量说明: {screening_focus.trade_plan_quality_note_zh}")
    print(f"Gate result / 门槛结果: {screening_focus.quality_gate_fail_reasons_zh}")


def _print_threshold_calibration(screening_focus: pd.Series) -> None:
    print()
    print("Threshold calibration / 阈值校准")
    print(
        "Calibration action / 校准动作: "
        f"{screening_focus.calibration_action} / {screening_focus.calibration_action_zh}"
    )
    print(
        "Market regime / 市场状态分层: "
        f"{screening_focus.market_regime} / {screening_focus.market_regime_zh}"
    )
    print(
        "Market regime adjustment / 市场状态门槛调整: "
        f"signal={screening_focus.market_regime_signal_delta:.2f}, "
        f"confidence={screening_focus.market_regime_confidence_delta:.2f}, "
        f"sample={int(screening_focus.market_regime_sample_delta)}, "
        f"win_rate={_format_optional_percent(screening_focus.market_regime_win_rate_delta)}, "
        f"avg_return={_format_optional_percent(screening_focus.market_regime_average_return_delta)}"
    )
    print(f"Market regime note / 市场状态说明: {screening_focus.market_regime_note_zh}")
    print(
        "Recommended signal threshold / 建议信号分门槛: "
        f"{screening_focus.recommended_signal_threshold:.2f}"
    )
    print(
        "Recommended confidence threshold / 建议置信度门槛: "
        f"{screening_focus.recommended_confidence_threshold:.2f}"
    )
    print(
        "Recommended backtest sample min / 建议回测样本下限: "
        f"{int(screening_focus.recommended_backtest_sample_min)}"
    )
    print(
        "Recommended win rate min / 建议胜率下限: "
        f"{_format_optional_percent(screening_focus.recommended_backtest_win_rate_min)}"
    )
    print(f"Calibration note / 校准说明: {screening_focus.calibration_note_zh}")


def _print_calibrated_screening(screening_focus: pd.Series) -> None:
    print()
    print("Calibrated screening / 校准后筛选")
    print(
        "Calibrated final screening / 校准后最终结论: "
        f"{screening_focus.calibrated_screening_action} / "
        f"{screening_focus.calibrated_screening_action_zh}"
    )
    print(
        "Calibrated quality gate passed / 校准后是否通过质量门槛: "
        f"{screening_focus.calibrated_quality_gate_passed}"
    )
    print(
        "Calibrated high probability score / 校准后高概率分数: "
        f"{screening_focus.calibrated_high_probability_score:.2f}"
    )
    print(
        "Calibrated gate result / 校准后门槛结果: "
        f"{screening_focus.calibrated_quality_gate_fail_reasons_zh}"
    )


def _print_signal_review_feedback(screening_focus: pd.Series) -> None:
    print()
    print("Signal review feedback / 信号复盘反馈")
    print(
        "Signal review score / 复盘反馈分: "
        f"{screening_focus.signal_review_score:.2f}, "
        f"adjustment={_format_signed_number(screening_focus.signal_review_adjustment)}"
    )
    print(
        "Signal review level / 复盘反馈等级: "
        f"{screening_focus.signal_review_level} / {screening_focus.signal_review_level_zh}"
    )
    print(
        "Signal review stats / 复盘统计: "
        f"samples={_format_optional_count(screening_focus.signal_review_sample_count)}, "
        f"window={screening_focus.signal_review_focus_window}, "
        f"win_rate={_format_optional_percent(screening_focus.signal_review_win_rate)}, "
        f"avg_return={_format_optional_percent(screening_focus.signal_review_avg_return)}"
    )
    print(f"Signal review note / 复盘反馈说明: {screening_focus.signal_review_note_zh}")


def _print_calibrated_watchlist_plan(screening_focus: pd.Series) -> None:
    print()
    print("Calibrated watchlist plan / 校准后观察计划")
    print(
        "Calibrated watchlist status / 校准后观察状态: "
        f"{screening_focus.calibrated_watchlist_status} / "
        f"{screening_focus.calibrated_watchlist_status_zh}"
    )
    print(
        "Calibrated watchlist gap score / 校准后观察差距分: "
        f"{screening_focus.calibrated_watchlist_gap_score:.2f}"
    )
    print(
        "Calibrated trigger price / 校准后重新检查触发价: "
        f"{screening_focus.calibrated_watchlist_trigger_price:.2f}"
    )
    print(f"Calibrated missing items / 校准后未达标条件: {screening_focus.calibrated_watchlist_missing_items_zh}")
    print(f"Calibrated re-check reason / 校准后重新检查原因: {screening_focus.calibrated_watchlist_recheck_reason_zh}")


def _print_watchlist_plan(screening_focus: pd.Series) -> None:
    print()
    print("Watchlist plan / 观察计划")
    print(
        "Watchlist status / 观察状态: "
        f"{screening_focus.watchlist_status} / {screening_focus.watchlist_status_zh}"
    )
    print(f"Watchlist gap score / 观察差距分: {screening_focus.watchlist_gap_score:.2f}")
    print(f"Trigger price / 重新检查触发价: {screening_focus.watchlist_trigger_price:.2f}")
    print(f"Ready items / 已达标条件: {screening_focus.watchlist_ready_items_zh}")
    print(f"Missing items / 未达标条件: {screening_focus.watchlist_missing_items_zh}")
    print(f"Re-check reason / 重新检查原因: {screening_focus.watchlist_recheck_reason_zh}")


def _print_confidence(first: pd.Series) -> None:
    print()
    print("Confidence / 置信度")
    print(f"Confidence score / 置信度分数: {first.confidence_score:.2f}")
    print(f"Confidence level / 置信度等级: {first.confidence_level} / {first.confidence_level_zh}")
    print(f"Confidence note / 置信度说明: {first.confidence_note_zh}")


def _print_risk_breakdown(first: pd.Series) -> None:
    print()
    print("Risk breakdown / 风险拆解")
    print(f"Overall risk / 综合风险: {first.overall_risk_level} / {first.overall_risk_level_zh}")
    print(f"Technical risk / 技术风险: {first.technical_risk_level} / {first.technical_risk_level_zh}")
    print(f"Market risk / 大盘风险: {first.market_risk_level} / {first.market_risk_level_zh}")
    print(
        "Relative strength risk / 相对强弱风险: "
        f"{first.relative_strength_risk_level} / {first.relative_strength_risk_level_zh}"
    )
    print(f"Sector risk / 板块风险: {first.sector_risk_level} / {first.sector_risk_level_zh}")
    print(f"Fundamental risk / 基本面风险: {first.fundamental_risk_level} / {first.fundamental_risk_level_zh}")
    print(f"Event risk / 事件风险: {first.event_risk_breakdown_level} / {first.event_risk_breakdown_level_zh}")
    print(
        "News sentiment risk / 新闻情绪风险: "
        f"{first.sentiment_risk_breakdown_level} / {first.sentiment_risk_breakdown_level_zh}"
    )
    print(
        "Analyst expectation risk / 分析师预期风险: "
        f"{first.analyst_risk_breakdown_level} / {first.analyst_risk_breakdown_level_zh}"
    )
    print(f"Valuation risk / 估值风险: {first.valuation_risk_breakdown_level} / {first.valuation_risk_breakdown_level_zh}")
    print(f"Data risk / 数据风险: {first.data_risk_level} / {first.data_risk_level_zh}")
    print(f"Risk note / 风险说明: {first.risk_breakdown_note_zh}")


def _print_data_quality(first: pd.Series) -> None:
    print()
    print("Data quality / 数据质量")
    print(f"Data quality score / 数据质量分数: {first.data_quality_score:.2f}")
    print(f"Data quality level / 数据质量等级: {first.data_quality_level} / {first.data_quality_level_zh}")
    print(
        "Weakest data layer / 最弱数据层: "
        f"{first.data_quality_weakest_layer} / {first.data_quality_weakest_layer_zh}"
    )
    print(f"Weak data layers / 薄弱数据层: {first.data_quality_weak_layers_zh}")
    print(
        "Repair priority / 修复优先级: "
        f"{first.data_quality_repair_priority} / {first.data_quality_repair_priority_zh}"
    )
    print(f"Repair actions / 修复动作: {first.data_quality_repair_actions_zh}")
    print(f"Data repair actions applied / 已自动修复: {first.data_repair_actions_applied_zh}")
    print(f"Price data / 价格数据: {first.price_data_status} / {first.price_data_status_zh}")
    print(
        "Price health / 价格健康: "
        f"{first.price_health_score:.2f}, "
        f"{first.price_health_level} / {first.price_health_level_zh}"
    )
    print(
        "Price anomalies / 价格异常: "
        f"max_gap_days={int(first.price_max_calendar_gap_days)}, "
        f"large_gaps={int(first.price_large_gap_count)}, "
        f"zero_volume={int(first.price_zero_volume_days)}, "
        f"missing_ohlcv={int(first.price_missing_ohlcv_rows)}, "
        f"extreme_returns={int(first.price_extreme_return_count)}"
    )
    print(f"Price health note / 价格健康说明: {first.price_health_note_zh}")
    print(f"Market data / 大盘数据: {first.market_data_status} / {first.market_data_status_zh}")
    print(
        "Relative strength data / 相对强弱数据: "
        f"{first.relative_strength_data_status} / {first.relative_strength_data_status_zh}"
    )
    print(f"Sector data / 板块数据: {first.sector_data_status} / {first.sector_data_status_zh}")
    print(
        "Fundamental data / 基本面数据: "
        f"{first.fundamental_data_status} / {first.fundamental_data_status_zh}"
    )
    print(f"Event data / 事件数据: {first.event_data_status} / {first.event_data_status_zh}")
    print(
        "News sentiment data / 新闻情绪数据: "
        f"{first.sentiment_data_status} / {first.sentiment_data_status_zh}"
    )
    print(f"Analyst data / 分析师数据: {first.analyst_data_status} / {first.analyst_data_status_zh}")
    print(f"Valuation data / 估值数据: {first.valuation_data_status} / {first.valuation_data_status_zh}")
    print(f"Missing fallback count / 缺失回退数量: {first.missing_fallback_count}")
    print(f"Data quality note / 数据质量说明: {first.data_quality_note_zh}")


def _print_market(first: pd.Series) -> None:
    print()
    print("Market / 大盘")
    print(f"Market score / 大盘分数: {first.market_score:.2f}")
    print(f"Market status / 大盘状态: {first.market_status}")


def _print_sector_context(first: pd.Series, result: RealTickerAnalysisResult) -> None:
    print()
    print("Sector context / 板块环境")
    print(f"Sector / 板块: {first.sector or 'N/A'}")
    print(f"Industry / 行业: {first.industry or 'N/A'}")
    print(f"Sector ETF / 板块ETF: {first.sector_etf or 'N/A'}")
    print(f"Sector score / 板块分数: {first.sector_score:.2f}")
    print(f"Sector status / 板块状态: {first.sector_status}")
    print(f"Note / 说明: {first.sector_note_zh}")
    print()
    if not result.peer_comparison.empty:
        target_peer_row = result.peer_comparison[result.peer_comparison["ticker"] == result.ticker]
        if not target_peer_row.empty:
            target_peer = target_peer_row.iloc[0]
            print("Peer comparison / 同业比较")
            print(
                f"Peer rank / 同业排名: {int(target_peer.peer_rank)} "
                f"of {len(result.peer_comparison)}"
            )
            print(f"Peer score / 同业高概率分数: {target_peer.high_probability_score:.2f}")
            print(
                "Peer screening / 同业筛选结论: "
                f"{target_peer.screening_action} / {target_peer.screening_action_zh}"
            )
            print(
                "Peer valuation / 同业估值: "
                f"{target_peer.peer_valuation_label} / {target_peer.peer_valuation_label_zh}"
            )
            print(
                f"Valuation rank / 估值排名: {int(target_peer.valuation_peer_rank)} "
                f"of {len(result.peer_comparison)}"
            )
            print(
                "Forward PE vs peer median / 预期市盈率相对同业中位数: "
                f"{_format_optional_percent(target_peer.forward_pe_vs_peer_median_pct)}"
            )
            print(
                "PEG vs peer median / PEG相对同业中位数: "
                f"{_format_optional_percent(target_peer.peg_vs_peer_median_pct)}"
            )
            print("Top peers / 同业前几名:")
            for row in result.peer_comparison.head(5).itertuples(index=False):
                print(
                    f"- {row.peer_rank}. {row.ticker}: "
                    f"score={row.high_probability_score:.2f}, "
                    f"passed={row.quality_gate_passed}, "
                    f"calibrated_passed={row.calibrated_quality_gate_passed}, "
                    f"horizon={row.focus_horizon}"
                )
            print(f"Peer report / 同业报告: {result.output_dir / 'peer_comparison.md'}")
            print()
    print("Fundamental quality / 基本面质量")
    print(f"Fundamental score / 基本面分数: {first.fundamental_score:.2f}")
    print(
        f"Quality / 质量: {first.fundamental_quality} / "
        f"{first.fundamental_quality_zh}"
    )
    print(f"Note / 说明: {first.fundamental_note_zh}")


def _print_analyst_expectations(first: pd.Series) -> None:
    print()
    print("Analyst expectations / 分析师预期")
    print(f"Analyst score / 分析师分数: {first.analyst_score:.2f}")
    print(f"Analyst label / 分析师标签: {first.analyst_label} / {first.analyst_label_zh}")
    print(f"Analyst risk / 分析师风险: {first.analyst_risk_level} / {first.analyst_risk_level_zh}")
    print(f"Block new entries / 是否阻止新入场: {bool(first.analyst_block_new_entries)}")
    print(f"Analyst upside / 目标价上行空间: {_format_optional_percent(first.analyst_upside)}")
    print(f"Recommendation mean / 推荐均值: {_format_optional_number(first.recommendation_mean)}")
    print(f"Number of analysts / 覆盖分析师数: {_format_optional_number(first.number_of_analysts)}")
    print(f"Target mean price / 平均目标价: {_format_optional_number(first.target_mean_price)}")
    print(f"Note / 说明: {first.analyst_note_zh}")


def _print_valuation_risk(first: pd.Series) -> None:
    print()
    print("Valuation risk / 估值风险")
    print(f"Valuation score / 估值分数: {first.valuation_score:.2f}")
    print(f"Valuation label / 估值标签: {first.valuation_label} / {first.valuation_label_zh}")
    print(f"Valuation risk / 估值风险: {first.valuation_risk_level} / {first.valuation_risk_level_zh}")
    print(f"Block new entries / 是否阻止新入场: {bool(first.valuation_block_new_entries)}")
    print(f"Forward PE / 预期市盈率: {_format_optional_number(first.valuation_forward_pe)}")
    print(f"Trailing PE / 静态市盈率: {_format_optional_number(first.valuation_trailing_pe)}")
    print(f"PEG ratio / PEG: {_format_optional_number(first.valuation_peg_ratio)}")
    print(
        "Free cash flow yield / 自由现金流收益率: "
        f"{_format_optional_percent(first.valuation_free_cash_flow_yield)}"
    )
    print(f"Growth reference / 增长参考: {_format_optional_percent(first.valuation_growth_reference)}")
    print(f"Profit margin / 净利率: {_format_optional_percent(first.valuation_profit_margin)}")
    print(f"Note / 说明: {first.valuation_note_zh}")


def _print_news_sentiment(first: pd.Series) -> None:
    print()
    print("News sentiment / 新闻情绪")
    print(f"Sentiment score / 情绪分数: {first.sentiment_score:.2f}")
    print(f"Sentiment label / 情绪标签: {first.sentiment_label} / {first.sentiment_label_zh}")
    print(
        "Sentiment risk / 情绪风险: "
        f"{first.sentiment_risk_level} / {first.sentiment_risk_level_zh}"
    )
    print(f"Block new entries / 是否阻止新入场: {bool(first.sentiment_block_new_entries)}")
    print(
        "Positive news level / 利好等级: "
        f"{first.positive_news_level} / {first.positive_news_level_zh}"
    )
    print(f"Positive news score / 利好加分: {first.positive_news_score:.2f}")
    print(
        "Positive news drivers / 利好触发项: "
        f"{first.positive_news_drivers} / {first.positive_news_drivers_zh}"
    )
    print(
        "Keywords / 关键词: "
        f"positive={int(first.sentiment_positive_count)}, "
        f"negative={int(first.sentiment_negative_count)}, "
        f"high_risk={int(first.sentiment_high_risk_count)}"
    )
    print(f"Note / 说明: {first.sentiment_note_zh}")


def _print_event_risk(first: pd.Series) -> None:
    print()
    print("Event risk / 事件风险")
    print(
        f"Risk level / 风险等级: {first.event_risk_level} / "
        f"{first.event_risk_level_zh}"
    )
    print(f"Event window / 事件窗口: {first.event_window} / {first.event_window_zh}")
    print(f"Block new entries / 是否阻止新入场: {bool(first.event_block_new_entries)}")
    print(f"Cooldown active / 财报后冷却是否生效: {bool(first.event_cooldown_active)}")
    print(f"Next earnings / 下一次财报: {first.next_earnings_date or 'N/A'}")
    days_until = "N/A" if first.days_until_earnings != first.days_until_earnings else int(first.days_until_earnings)
    print(f"Days until earnings / 距离财报天数: {days_until}")
    print(f"Last earnings / 上一次财报: {first.last_earnings_date or 'N/A'}")
    days_since = "N/A" if first.days_since_earnings != first.days_since_earnings else int(first.days_since_earnings)
    print(f"Days since earnings / 距离上次财报天数: {days_since}")
    print(f"Note / 说明: {first.event_risk_note_zh}")


def _print_plain_summary(result: RealTickerAnalysisResult) -> None:
    print()
    print("Plain summary / 简明结论")
    for row in result.analysis.itertuples(index=False):
        print(f"- {row.horizon} / {row.horizon_zh_label}: {row.plain_summary_zh}")


def _print_entry_plan(result: RealTickerAnalysisResult) -> None:
    print()
    print("Entry plan / 买点计划")
    for row in result.analysis.itertuples(index=False):
        print(
            f"- {row.horizon} / {row.horizon_zh_label}: "
            f"{row.entry_distance_note_zh} {row.chase_status_zh} {row.entry_plan_zh}"
        )


def _print_backtest_reliability(result: RealTickerAnalysisResult) -> None:
    print()
    print("Backtest reliability / 回测可靠性")
    for row in result.analysis.itertuples(index=False):
        print(
            f"- {row.horizon} / {row.horizon_zh_label}: "
            f"trust={row.backtest_trust_score:.2f} / {row.backtest_trust_level_zh}；"
            f"{row.backtest_reliability_note_zh}"
        )


def _print_entry_backtest(first: pd.Series, result: RealTickerAnalysisResult) -> None:
    print()
    print("Entry backtest / 买点回测")
    print("How to read / 怎么看:")
    print(
        "execution model / 执行模型: "
        f"{first.backtest_execution_model} / {first.backtest_execution_model_zh}"
    )
    print(f"slippage / 滑点: {_format_optional_percent(first.backtest_slippage_pct)}")
    print(
        "slippage drivers / 滑点依据: "
        f"avg_dollar_volume={_format_optional_number(first.backtest_avg_dollar_volume)}, "
        f"atr_ratio={_format_optional_percent(first.backtest_atr_ratio)}, "
        f"liquidity={first.backtest_liquidity_label} / {first.backtest_liquidity_label_zh}, "
        f"volatility={first.backtest_volatility_label} / {first.backtest_volatility_label_zh}"
    )
    print(f"execution note / 执行说明: {first.backtest_execution_note_zh}")
    print(
        "backtest trust / 回测可信度: "
        f"{first.backtest_trust_score:.2f}, "
        f"{first.backtest_trust_level} / {first.backtest_trust_level_zh}"
    )
    print(f"trust note / 可信度说明: {first.backtest_trust_note_zh}")
    print(
        "regime coverage / 行情覆盖: "
        f"{first.regime_coverage_score:.2f}, "
        f"{first.regime_coverage_level} / {first.regime_coverage_level_zh}, "
        f"dominant={first.regime_coverage_dominant_regime} / "
        f"{first.regime_coverage_dominant_regime_zh}, "
        f"share={_format_optional_percent(first.regime_coverage_dominant_share)}"
    )
    print(f"regime coverage note / 行情覆盖说明: {first.regime_coverage_note_zh}")
    print(
        "recent backtest strength / 近期回测强度: "
        f"{first.recent_backtest_score:.2f}, "
        f"{first.recent_backtest_level} / {first.recent_backtest_level_zh}, "
        f"recent_win={_format_optional_percent(first.recent_backtest_win_rate)}, "
        f"recent_avg={_format_optional_percent(first.recent_backtest_average_return)}"
    )
    print(f"recent backtest note / 近期回测说明: {first.recent_backtest_note_zh}")
    print(
        "backtest decay / 回测衰退: "
        f"{first.backtest_decay_score:.2f}, "
        f"{first.backtest_decay_level} / {first.backtest_decay_level_zh}, "
        f"late_win={_format_optional_percent(first.backtest_decay_late_win_rate)}, "
        f"late_avg={_format_optional_percent(first.backtest_decay_late_average_return)}"
    )
    print(f"backtest decay note / 回测衰退说明: {first.backtest_decay_note_zh}")
    print("- win rate: historical similar entries that ended profitable / 过去类似买点最终赚钱的比例")
    print("- target: reached target before stop / 先到目标价的比例")
    print("- stop: reached stop before target / 先到止损价的比例")
    print("- avg return: average historical return for that entry type / 该类买点的历史平均收益")
    print("- quality: sample reliability based on sample count / 根据样本数判断统计可靠性")
    for row in result.analysis.itertuples(index=False):
        _print_entry_backtest_row(row)


def _print_signal_summary(result: RealTickerAnalysisResult) -> None:
    print()
    print("Signal summary / 信号摘要")
    for row in result.analysis.itertuples(index=False):
        print(
            f"- {row.horizon} ({row.horizon_time_range}): {row.action}, "
            f"score={row.signal_score:.2f}, "
            f"entry={row.entry_price:.2f}, "
            f"stop={row.stop_loss:.2f}, "
            f"target={row.take_profit:.2f}, "
            f"risk_reward={row.risk_reward:.2f}"
        )
    signal_review = result.cache_metadata.get("signal_review", {})
    if isinstance(signal_review, dict) and signal_review.get("status") == "ok":
        print()
        print("Signal review / 信号复盘")
        print(f"Signal history / 信号历史: {signal_review.get('history_path')}")
        print(f"Review report / 复盘报告: {signal_review.get('report_path')}")
        print(f"Ticker signal count / 该股票信号记录数: {signal_review.get('ticker_signal_count')}")
        print(
            "Review learning state / 复盘学习状态: "
            f"score={_format_optional_number(signal_review.get('review_learning_score'))}, "
            f"level={signal_review.get('review_learning_level') or 'N/A'} / "
            f"{signal_review.get('review_learning_level_zh') or 'N/A'}, "
            f"focus={signal_review.get('review_learning_focus_window') or 'N/A'}, "
            f"samples={_format_optional_count(signal_review.get('review_learning_sample_count'))}, "
            f"adjustment={_format_signed_number(signal_review.get('review_learning_adjustment'))}"
        )
        print(
            "Review learning note / 复盘学习说明: "
            f"{signal_review.get('review_learning_note_zh') or 'N/A'}"
        )
        for window in (5, 20, 60):
            print(
                f"{window}d review / {window}日复盘: "
                f"completed={_format_optional_count(signal_review.get(f'completed_{window}d_count'))}, "
                f"pending={_format_optional_count(signal_review.get(f'pending_{window}d_count'))}, "
                f"next_review={signal_review.get(f'next_estimated_review_date_{window}d') or 'N/A'}, "
                f"remaining_trading_days="
                f"{_format_optional_count(signal_review.get(f'min_trading_days_remaining_{window}d'))}, "
                f"win_rate={_format_optional_percent(signal_review.get(f'win_rate_{window}d'))}, "
                f"avg_return={_format_optional_percent(signal_review.get(f'avg_return_{window}d'))}, "
                f"median_return={_format_optional_percent(signal_review.get(f'median_return_{window}d'))}"
            )
    elif isinstance(signal_review, dict) and signal_review.get("status") == "failed":
        print()
        print("Signal review / 信号复盘")
        print(f"Review warning / 复盘提示: {signal_review.get('warning')}")
    print()
    print(f"Open report: {result.output_dir / 'ticker_analysis.md'}")


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
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if abs(number) >= 1_000_000_000:
        return f"{number / 1_000_000_000:.2f}B"
    if abs(number) >= 1_000_000:
        return f"{number / 1_000_000:.2f}M"
    return f"{number:.2f}"


def _format_optional_count(value: object) -> str:
    try:
        if value is None or value != value:
            return "0"
        return str(int(float(value)))
    except (TypeError, ValueError):
        return "0"


def _format_signed_number(value: object) -> str:
    try:
        if value is None or value != value:
            return "N/A"
        return f"{float(value):+.2f}"
    except (TypeError, ValueError):
        return "N/A"


def _print_entry_backtest_row(row: object) -> None:
    print(f"\n{row.horizon} / {row.horizon_zh_label} ({row.horizon_time_range})")
    print(
        "  regime coverage / 行情覆盖: "
        f"{row.regime_coverage_score:.2f}, "
        f"{row.regime_coverage_level} / {row.regime_coverage_level_zh}, "
        f"dominant={row.regime_coverage_dominant_regime} / "
        f"{row.regime_coverage_dominant_regime_zh}, "
        f"share={_format_optional_percent(row.regime_coverage_dominant_share)}"
    )
    print(
        "  recent strength / 近期强度: "
        f"{row.recent_backtest_score:.2f}, "
        f"{row.recent_backtest_level} / {row.recent_backtest_level_zh}, "
        f"recent_win={_format_optional_percent(row.recent_backtest_win_rate)}, "
        f"recent_avg={_format_optional_percent(row.recent_backtest_average_return)}"
    )
    print(
        "  decay check / 衰退检查: "
        f"{row.backtest_decay_score:.2f}, "
        f"{row.backtest_decay_level} / {row.backtest_decay_level_zh}, "
        f"late_win={_format_optional_percent(row.backtest_decay_late_win_rate)}, "
        f"late_avg={_format_optional_percent(row.backtest_decay_late_average_return)}"
    )
    _print_entry_type_backtest(
        label="Breakout / 突破买点",
        trade_count=row.breakout_trade_count,
        sample_quality=row.breakout_sample_quality,
        sample_quality_zh=row.breakout_sample_quality_zh,
        win_rate=row.breakout_win_rate,
        target_hit_rate=row.breakout_target_hit_rate,
        stop_hit_rate=row.breakout_stop_hit_rate,
        average_gain=row.breakout_average_gain,
        average_loss=row.breakout_average_loss,
        average_return=row.breakout_average_return,
    )
    _print_entry_type_backtest(
        label="Pullback / 回调买点",
        trade_count=row.pullback_trade_count,
        sample_quality=row.pullback_sample_quality,
        sample_quality_zh=row.pullback_sample_quality_zh,
        win_rate=row.pullback_win_rate,
        target_hit_rate=row.pullback_target_hit_rate,
        stop_hit_rate=row.pullback_stop_hit_rate,
        average_gain=row.pullback_average_gain,
        average_loss=row.pullback_average_loss,
        average_return=row.pullback_average_return,
    )


def _print_entry_type_backtest(
    label: str,
    trade_count: int,
    sample_quality: str,
    sample_quality_zh: str,
    win_rate: object,
    target_hit_rate: object,
    stop_hit_rate: object,
    average_gain: object,
    average_loss: object,
    average_return: object,
) -> None:
    print(f"  {label}")
    print(f"    sample / 样本数: {trade_count}")
    print(f"    quality / 可靠性: {sample_quality} / {sample_quality_zh}")
    print(f"    win rate / 胜率: {_format_optional_percent(win_rate)}")
    print(f"    target hit / 目标价命中: {_format_optional_percent(target_hit_rate)}")
    print(f"    stop hit / 止损命中: {_format_optional_percent(stop_hit_rate)}")
    print(f"    avg gain / 平均盈利: {_format_optional_percent(average_gain)}")
    print(f"    avg loss / 平均亏损: {_format_optional_percent(average_loss)}")
    print(f"    avg return / 平均收益: {_format_optional_percent(average_return)}")
