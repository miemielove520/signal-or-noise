from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from stock_selector.scanner import (
    build_top_candidates,
    render_scan_report,
    render_top_candidates_report,
    run_high_probability_scan,
)


class ScannerTest(unittest.TestCase):
    def test_render_scan_report_groups_candidates(self) -> None:
        summary = pd.DataFrame(
            [
                make_scan_row("AAA", True, 85.0),
                make_scan_row("BBB", False, 65.0),
                make_scan_row("CCC", False, 45.0),
            ]
        )
        summary["scan_rank"] = [1, 2, 3]

        report = render_scan_report(summary, [])

        self.assertIn("## Calibrated High Probability Candidates / 校准后高概率候选", report)
        self.assertIn("## Near Watchlist / 接近机会", report)
        self.assertIn("## Early Watchlist / 早期观察", report)
        self.assertIn("## Filtered Out / 被过滤", report)
        self.assertIn("## Re-check Queue / 重新检查队列", report)
        self.assertIn("## Re-check Action Breakdown / 重新检查动作分类", report)
        self.assertIn("recheck_priority_score", report)
        self.assertIn("recheck_trigger_distance_pct", report)
        self.assertIn("recheck_action_type", report)
        self.assertIn("data_remediation_action", report)
        self.assertIn("data_readiness_level", report)
        self.assertIn("data_source_validation_status", report)
        self.assertIn("data_quality_weakest_layer", report)
        self.assertIn("AAA", report)
        self.assertIn("BBB", report)
        self.assertIn("CCC", report)

    def test_build_top_candidates_creates_first_read_summary(self) -> None:
        summary = pd.DataFrame(
            [
                make_scan_row("AAA", True, 85.0),
                make_scan_row("BBB", False, 65.0),
                make_scan_row("CCC", False, 45.0),
            ]
        )
        summary["scan_rank"] = [1, 2, 3]

        top_candidates = build_top_candidates(summary)
        report = render_top_candidates_report(top_candidates, summary, [])

        self.assertEqual(top_candidates.iloc[0]["ticker"], "AAA")
        self.assertEqual(top_candidates.iloc[0]["candidate_category"], "high_probability")
        self.assertIn("candidate_category_zh", top_candidates.columns)
        self.assertIn("attention_score", top_candidates.columns)
        self.assertIn("missing_items_zh", top_candidates.columns)
        self.assertIn("data_readiness_level", top_candidates.columns)
        self.assertIn("data_repair_note_zh", top_candidates.columns)
        self.assertIn("Top Candidates / 最佳候选总结", report)
        self.assertIn("Priority List / 优先列表", report)
        self.assertIn("How To Use / 怎么用", report)
        self.assertIn("AAA", report)

    def test_run_high_probability_scan_writes_outputs_and_ranks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "scan"
            journal_root = Path(directory) / "journal"
            with patch(
                "stock_selector.scanner.run_real_ticker_analysis",
                side_effect=[
                    StubRealResult("LOW", 55.0, False, output_dir / "tickers" / "LOW"),
                    StubRealResult("HIGH", 88.0, True, output_dir / "tickers" / "HIGH"),
                ],
            ):
                result = run_high_probability_scan(
                    tickers=["LOW", "HIGH"],
                    output_dir=output_dir,
                    include_snapshot=False,
                    write_journal=True,
                    journal_root=journal_root,
                )

            self.assertEqual(result.summary.iloc[0]["ticker"], "HIGH")
            self.assertEqual(result.top_candidates.iloc[0]["ticker"], "HIGH")
            self.assertIn("requested_period", result.summary.columns)
            self.assertIn("analysis_period", result.summary.columns)
            self.assertIn("auto_period_upgraded", result.summary.columns)
            self.assertIn("recheck_rank", result.summary.columns)
            self.assertIn("recheck_priority_score", result.summary.columns)
            self.assertIn("recheck_priority_level", result.summary.columns)
            self.assertIn("recheck_trigger_distance_pct", result.summary.columns)
            self.assertIn("recheck_trigger_distance_label", result.summary.columns)
            self.assertIn("recheck_action_type", result.summary.columns)
            self.assertIn("recheck_action_type_zh", result.summary.columns)
            self.assertIn("recheck_non_price_blocker_count", result.summary.columns)
            self.assertIn("data_remediation_action", result.summary.columns)
            self.assertIn("data_remediation_note_zh", result.summary.columns)
            self.assertIn("data_readiness_level", result.summary.columns)
            self.assertIn("data_readiness_score", result.summary.columns)
            self.assertIn("data_source_validation_status", result.summary.columns)
            self.assertIn("data_quality_weakest_layer", result.summary.columns)
            self.assertIn("data_quality_repair_actions_zh", result.summary.columns)
            self.assertIn("data_repair_actions_applied", result.summary.columns)
            self.assertIn("data_repair_actions_applied_zh", result.summary.columns)
            self.assertIn("price_health_score", result.summary.columns)
            self.assertIn("price_health_level", result.summary.columns)
            self.assertIn("backtest_trust_score", result.summary.columns)
            self.assertIn("backtest_trust_level", result.summary.columns)
            self.assertIn("calibrated_win_probability", result.summary.columns)
            self.assertIn("calibrated_probability_confidence", result.summary.columns)
            self.assertGreater(
                float(result.summary.loc[result.summary["ticker"] == "LOW", "recheck_priority_score"].iloc[0]),
                0.0,
            )
            self.assertEqual(
                result.summary.loc[result.summary["ticker"] == "LOW", "recheck_action_type"].iloc[0],
                "wait_for_price_trigger",
            )
            self.assertTrue((output_dir / "high_probability_scan.csv").exists())
            self.assertTrue((output_dir / "high_probability_scan.md").exists())
            self.assertTrue((output_dir / "top_candidates.csv").exists())
            self.assertTrue((output_dir / "top_candidates.md").exists())
            self.assertTrue((output_dir / "top_candidates.json").exists())
            self.assertTrue((output_dir / "data_readiness_summary.csv").exists())
            self.assertTrue((output_dir / "scan_result.json").exists())
            self.assertTrue((output_dir / "cache_metadata.json").exists())
            self.assertFalse(result.data_readiness_summary.empty)
            scan_payload = json.loads((output_dir / "scan_result.json").read_text(encoding="utf-8"))
            self.assertEqual(scan_payload["tickers_analyzed"], ["LOW", "HIGH"])
            self.assertIn("summary", scan_payload)
            self.assertIn("top_candidates", scan_payload)
            self.assertIn("data_readiness_summary", scan_payload)
            self.assertIn("top_candidates_report", scan_payload["output_files"])
            self.assertIn("data_readiness_summary_csv", scan_payload["output_files"])
            self.assertIn("recheck_action_summary", scan_payload)
            self.assertGreaterEqual(len(scan_payload["recheck_action_summary"]), 1)
            self.assertIsNotNone(result.journal)
            self.assertTrue((journal_root / "daily_journal.md").exists())


class StubRealResult:
    def __init__(self, ticker: str, score: float, passed: bool, output_dir: Path) -> None:
        self.ticker = ticker
        self.requested_period = "5y"
        self.effective_period = "5y"
        self.auto_period_upgraded = False
        self.auto_period_upgrade_reason = "Requested period was used without automatic extension."
        self.auto_period_upgrade_reason_zh = "系统使用了你请求的数据周期，没有自动延长。"
        self.snapshot = {"company_name": ticker}
        self.output_dir = output_dir
        self.analysis = pd.DataFrame([make_scan_row(ticker, passed, score)])


def make_scan_row(ticker: str, passed: bool, score: float) -> dict[str, object]:
    return {
        "ticker": ticker,
        "analysis_period": "5y",
        "company_name": ticker,
        "sector": "Technology",
        "industry": "Software",
        "final_decision": "high_probability_candidate" if passed else "near_watchlist",
        "final_decision_zh": "高概率候选，可以重点跟踪" if passed else "接近机会，但还未达标",
        "final_focus_horizon": "short",
        "final_focus_horizon_zh": "短期",
        "final_score": score,
        "latest_price": 95.0,
        "final_watchlist_status": "ready_high_probability"
        if passed
        else ("close_but_not_ready" if score >= 60 else "not_ready"),
        "final_watchlist_status_zh": "已通过高概率筛选"
        if passed
        else ("接近但还没准备好" if score >= 60 else "暂不值得重点观察"),
        "final_reason": "Calibrated filter result summary.",
        "final_reason_zh": "校准后筛选结果摘要。",
        "final_next_step": "Re-check if calibrated signal improves.",
        "final_next_step_zh": "如果校准后信号改善，可以重新检查。",
        "data_readiness_level": "ready",
        "data_readiness_level_zh": "准备充分",
        "data_readiness_score": 91.0,
        "data_readiness_repair_priority": "low",
        "data_readiness_repair_priority_zh": "低",
        "data_readiness_primary_blockers": "none",
        "data_readiness_primary_blockers_zh": "无",
        "data_source_validation_status": "single_source_available",
        "data_source_validation_status_zh": "只有单一价格源可用",
        "data_needs_repair": False,
        "primary_blocker": "none" if passed else "entry_not_executable",
        "primary_blocker_zh": "无" if passed else "买点当前不可执行",
        "priority_blockers": "none" if passed else "entry_not_executable",
        "priority_blockers_zh": "无" if passed else "买点当前不可执行",
        "priority_blocker_count": 0 if passed else 1,
        "priority_blocker_note": "No major blocker." if passed else "Entry readiness failed.",
        "priority_blocker_note_zh": "没有主要卡点。" if passed else "买点可执行性未通过。",
        "primary_blocker_resolution": "No blocker resolution is needed."
        if passed
        else "Wait until price closes above the trigger.",
        "primary_blocker_resolution_zh": "当前不需要解除主要卡点。"
        if passed
        else "等待价格收在触发价上方。",
        "blocker_resolution_steps": "none"
        if passed
        else "Wait until price closes above the trigger.",
        "blocker_resolution_steps_zh": "无" if passed else "等待价格收在触发价上方。",
        "blocker_recheck_trigger": "Continue tracking the plan."
        if passed
        else "Re-check after price trigger.",
        "blocker_recheck_trigger_zh": "继续跟踪计划。"
        if passed
        else "价格触发后重新检查。",
        "primary_blocker_progress": 100.0 if passed else 25.0,
        "blocker_resolution_score": 100.0 if passed else 25.0,
        "blocker_resolution_level": "ready" if passed else "far",
        "blocker_resolution_level_zh": "已解除" if passed else "距离较远",
        "blocker_resolution_gap": "No major blocker remains."
        if passed
        else "entry_not_executable: 25.0/100 resolved.",
        "blocker_resolution_gap_zh": "当前没有剩余主要卡点。"
        if passed
        else "买点当前不可执行: 已接近解除25.0/100。",
        "horizon_alignment_score": 72.0,
        "horizon_alignment_label": "mostly_constructive",
        "horizon_alignment_label_zh": "多数周期偏积极",
        "constructive_horizon_count": 2,
        "weak_horizon_count": 1,
        "risk_wait_horizon_count": 0,
        "horizon_signal_score_spread": 12.0,
        "horizon_alignment_note": "Most horizons are constructive, but confirmation is not complete.",
        "horizon_alignment_note_zh": "多数周期偏积极，但确认还不完整。",
        "horizon": "short",
        "focus_horizon": "short",
        "screening_action": "high_probability_watchlist_candidate"
        if passed
        else "not_high_probability_setup_now",
        "screening_action_zh": "高概率观察候选" if passed else "当前不是高概率机会",
        "screening_profile": "software",
        "screening_profile_zh": "软件股规则",
        "quality_gate_passed": passed,
        "high_probability_score": score,
        "high_probability_level": "high" if passed else "low",
        "calibrated_win_probability": 0.64 if passed else 0.48,
        "calibrated_win_probability_raw": 0.61 if passed else 0.46,
        "probability_calibration_adjustment": 0.03 if passed else 0.02,
        "probability_calibration_source": "walk_forward_probability_bucket:60_to_65"
        if passed
        else "walk_forward_probability_bucket:below_50",
        "probability_calibration_source_zh": "滚动验证概率分组：60%-65%"
        if passed
        else "滚动验证概率分组：低于50%",
        "probability_calibration_sample_count": 15 if passed else 8,
        "probability_calibration_action": "raise_probability_estimate",
        "probability_calibration_action_zh": "上调胜率估计",
        "calibrated_probability_level": "constructive" if passed else "low",
        "calibrated_probability_level_zh": "偏积极" if passed else "偏低",
        "calibrated_probability_confidence": 78.0 if passed else 52.0,
        "calibrated_probability_confidence_level": "high" if passed else "low",
        "calibrated_probability_confidence_level_zh": "较高" if passed else "偏低",
        "calibrated_probability_note": "Research estimate blended evidence.",
        "calibrated_probability_note_zh": "这是综合证据后的研究估计值。",
        "watchlist_status": "ready_high_probability"
        if passed
        else ("close_but_not_ready" if score >= 60 else "not_ready"),
        "watchlist_status_zh": "已通过高概率筛选"
        if passed
        else ("接近但还没准备好" if score >= 60 else "暂不值得重点观察"),
        "watchlist_gap_score": max(0.0, 65.0 - score),
        "watchlist_ready_items": "data quality >= 70",
        "watchlist_ready_items_zh": "数据质量达到70分",
        "watchlist_missing_items": "signal score >= 65" if not passed else "none",
        "watchlist_missing_items_zh": "信号分数达到65分" if not passed else "无",
        "watchlist_missing_count": 0 if passed else 1,
        "watchlist_trigger_price": 100.0,
        "watchlist_recheck_reason": "Re-check if signal improves.",
        "watchlist_recheck_reason_zh": "如果信号改善，可以重新检查。",
        "signal_score": score,
        "confidence_score": score,
        "data_quality_score": 90.0,
        "data_quality_weakest_layer": "none",
        "data_quality_weakest_layer_zh": "无",
        "data_quality_weak_layers": "none",
        "data_quality_weak_layers_zh": "无",
        "data_quality_repair_actions": "none",
        "data_quality_repair_actions_zh": "无",
        "data_quality_repair_priority": "none",
        "data_quality_repair_priority_zh": "无",
        "data_repair_actions_applied": "none",
        "data_repair_actions_applied_zh": "无",
        "price_health_score": 98.0,
        "price_health_level": "clean",
        "price_health_level_zh": "干净",
        "price_health_issue_count": 0,
        "price_max_calendar_gap_days": 3,
        "price_large_gap_count": 0,
        "price_zero_volume_days": 0,
        "price_missing_ohlcv_rows": 0,
        "price_extreme_return_count": 0,
        "price_health_note": "price health checks found no major anomalies",
        "price_health_note_zh": "价格健康检查未发现明显异常",
        "backtest_trust_score": 82.0,
        "backtest_trust_level": "high_trust",
        "backtest_trust_level_zh": "可信度较高",
        "backtest_sample_score": 80.0,
        "backtest_liquidity_score": 90.0,
        "backtest_slippage_score": 93.0,
        "backtest_return_evidence_score": 70.0,
        "regime_coverage_score": 78.0,
        "regime_coverage_level": "broad",
        "regime_coverage_level_zh": "覆盖较广",
        "regime_coverage_regime_count": 3,
        "regime_coverage_dominant_regime": "steady_uptrend",
        "regime_coverage_dominant_regime_zh": "稳定上升",
        "regime_coverage_dominant_share": 0.45,
        "regime_coverage_note": "Regime coverage uses broad entry samples.",
        "regime_coverage_note_zh": "行情覆盖度使用较广的买点样本。",
        "recent_backtest_score": 76.0,
        "recent_backtest_level": "stable",
        "recent_backtest_level_zh": "近期稳定",
        "recent_backtest_trade_count": 8,
        "recent_backtest_win_rate": 0.625,
        "recent_backtest_average_return": 0.025,
        "recent_backtest_return_delta": 0.005,
        "recent_backtest_note": "Recent backtest strength is stable.",
        "recent_backtest_note_zh": "近期回测强度稳定。",
        "backtest_decay_score": 74.0,
        "backtest_decay_level": "acceptable",
        "backtest_decay_level_zh": "可接受",
        "backtest_decay_early_trade_count": 10,
        "backtest_decay_late_trade_count": 10,
        "backtest_decay_early_win_rate": 0.60,
        "backtest_decay_late_win_rate": 0.62,
        "backtest_decay_early_average_return": 0.020,
        "backtest_decay_late_average_return": 0.026,
        "backtest_decay_win_rate_delta": 0.02,
        "backtest_decay_average_return_delta": 0.006,
        "backtest_decay_note": "Backtest decay is acceptable.",
        "backtest_decay_note_zh": "回测衰退检查可接受。",
        "backtest_trust_note": "Trust score blends sample count, price health, liquidity, slippage, and return evidence.",
        "backtest_trust_note_zh": "可信度分综合样本数、价格健康、流动性、滑点和收益证据。",
        "overall_risk_level": "low",
        "market_score": 70.0,
        "relative_strength_score": 60.0,
        "fundamental_score": 70.0,
        "sector_score": 60.0,
        "event_risk_level": "low",
        "sentiment_score": 65.0,
        "sentiment_risk_level": "low",
        "analyst_score": 70.0,
        "analyst_risk_level": "low",
        "valuation_score": 65.0,
        "valuation_risk_level": "low",
        "screening_backtest_entry_type": "breakout",
        "screening_backtest_trade_count": 20,
        "screening_backtest_win_rate": 0.60,
        "screening_backtest_stop_hit_rate": 0.35,
        "screening_backtest_average_return": 0.04,
        "entry_readiness_gate_passed": True,
        "entry_readiness_status": "executable_now",
        "entry_readiness_status_zh": "当前接近可执行买点",
        "entry_readiness_score": 90.0,
        "entry_readiness_reason": "entry is executable",
        "entry_readiness_reason_zh": "买点当前可执行",
        "entry_readiness_note": "Price is close enough to an active entry zone.",
        "entry_readiness_note_zh": "价格足够接近当前可执行买点区域。",
        "trade_plan_quality_gate_passed": True,
        "trade_plan_quality_status": "valid_trade_plan",
        "trade_plan_quality_status_zh": "交易计划有效",
        "trade_plan_quality_score": 85.0,
        "trade_plan_quality_reason": "trade plan quality passed",
        "trade_plan_quality_reason_zh": "交易计划质量通过",
        "trade_plan_quality_note": "Risk-reward=2.00; stop distance=6.00%.",
        "trade_plan_quality_note_zh": "盈亏比=2.00；止损距离=6.00%。",
        "calibration_action": "keep_current_thresholds",
        "calibration_action_zh": "保持当前门槛",
        "calibration_level": "balanced",
        "calibration_level_zh": "平衡",
        "market_regime": "supportive",
        "market_regime_zh": "强势或支持型大盘",
        "market_regime_note": "Supportive market regime allows slightly more flexible signal and win-rate thresholds, but does not remove quality gates.",
        "market_regime_note_zh": "强势或支持型大盘允许信号和胜率门槛略微灵活，但不会取消质量门槛。",
        "market_regime_signal_delta": -1.0,
        "market_regime_confidence_delta": 0.0,
        "market_regime_sample_delta": 0,
        "market_regime_win_rate_delta": -0.005,
        "market_regime_average_return_delta": 0.0,
        "recommended_signal_threshold": 65.0,
        "recommended_confidence_threshold": 65.0,
        "recommended_backtest_sample_min": 10,
        "recommended_backtest_win_rate_min": 0.55,
        "recommended_backtest_average_return_min": 0.0,
        "calibration_note": "Historical entry evidence supports the current threshold set.",
        "calibration_note_zh": "历史买点证据支持当前门槛。",
        "calibrated_screening_action": "calibrated_high_probability_candidate"
        if passed
        else "not_calibrated_high_probability_now",
        "calibrated_screening_action_zh": "校准后高概率候选"
        if passed
        else "校准后当前不是高概率机会",
        "calibrated_quality_gate_passed": passed,
        "calibrated_high_probability_score": score,
        "calibrated_high_probability_level": "high" if passed else "low",
        "calibrated_quality_gate_fail_reasons": "all strict quality gates passed"
        if passed
        else "signal score too low",
        "calibrated_quality_gate_fail_reasons_zh": "所有严格质量门槛通过"
        if passed
        else "信号分数不足",
        "calibrated_watchlist_status": "ready_high_probability"
        if passed
        else ("close_but_not_ready" if score >= 60 else "not_ready"),
        "calibrated_watchlist_status_zh": "已通过高概率筛选"
        if passed
        else ("接近但还没准备好" if score >= 60 else "暂不值得重点观察"),
        "calibrated_watchlist_gap_score": max(0.0, 65.0 - score),
        "calibrated_watchlist_missing_items": "signal score >= 65"
        if not passed
        else "none",
        "calibrated_watchlist_missing_items_zh": "信号分数达到65分"
        if not passed
        else "无",
        "calibrated_watchlist_missing_count": 0 if passed else 1,
        "calibrated_watchlist_trigger_price": 100.0,
        "calibrated_watchlist_recheck_reason": "Re-check if calibrated signal improves.",
        "calibrated_watchlist_recheck_reason_zh": "如果校准后信号改善，可以重新检查。",
        "quality_gate_fail_reasons": "all strict quality gates passed"
        if passed
        else "signal score too low",
        "quality_gate_fail_reasons_zh": "所有严格质量门槛通过" if passed else "信号分数不足",
    }


if __name__ == "__main__":
    unittest.main()
