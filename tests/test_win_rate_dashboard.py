from __future__ import annotations

import unittest

import pandas as pd

from stock_selector.win_rate_dashboard import (
    build_historical_win_rate_gate,
    build_historical_threshold_recommendations,
    build_profile_action_recommendations,
    build_profile_blocker_dashboard,
    build_profile_health_dashboard,
    build_win_rate_dashboard,
    render_historical_threshold_recommendations,
    render_historical_win_rate_gate,
    render_profile_action_recommendations,
    render_profile_blocker_dashboard,
    render_profile_health_dashboard,
    render_win_rate_dashboard,
    win_rate_dashboard_payload,
)


class WinRateDashboardTest(unittest.TestCase):
    def test_build_dashboard_summarizes_overall_and_groups(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "horizon": "short",
                    "horizon_zh_label": "短期",
                    "screening_profile": "default",
                    "screening_profile_zh": "默认规则",
                    "screening_backtest_entry_type": "breakout",
                    "quality_gate_passed": True,
                    "forward_return_5d": 0.04,
                    "forward_return_20d": 0.08,
                },
                {
                    "ticker": "BBB",
                    "horizon": "short",
                    "horizon_zh_label": "短期",
                    "screening_profile": "default",
                    "screening_profile_zh": "默认规则",
                    "screening_backtest_entry_type": "breakout",
                    "quality_gate_passed": True,
                    "forward_return_5d": -0.02,
                    "forward_return_20d": -0.04,
                },
                {
                    "ticker": "CCC",
                    "horizon": "long",
                    "horizon_zh_label": "长期",
                    "screening_profile": "growth",
                    "screening_profile_zh": "成长规则",
                    "screening_backtest_entry_type": "pullback",
                    "quality_gate_passed": False,
                    "forward_return_5d": 0.01,
                    "forward_return_20d": 0.06,
                },
            ]
        )

        dashboard = build_win_rate_dashboard(events, forward_windows=(5, 20))

        overall_20d = dashboard["overall"][
            dashboard["overall"]["forward_window_days"] == 20
        ].iloc[0]
        self.assertEqual(overall_20d["sample_count"], 3)
        self.assertAlmostEqual(float(overall_20d["win_rate"]), 2 / 3)
        self.assertAlmostEqual(float(overall_20d["avg_return"]), (0.08 - 0.04 + 0.06) / 3)
        self.assertEqual(overall_20d["sample_quality"], "weak")
        self.assertIn("horizon_zh_label", dashboard["by_horizon"].columns)
        self.assertIn("screening_backtest_entry_type", dashboard["by_entry_type"].columns)

    def test_profile_health_dashboard_marks_keep_and_expand_sample(self) -> None:
        strong_rows = [
            {
                "screening_profile": "strong_profile",
                "screening_profile_zh": "强规则",
                "validation_bucket": "high_probability",
                "quality_gate_passed": True,
                "forward_return_20d": 0.04,
                "max_drawdown_after_signal": -0.04,
            }
            for _ in range(32)
        ]
        thin_rows = [
            {
                "screening_profile": "thin_profile",
                "screening_profile_zh": "薄样本规则",
                "validation_bucket": "filtered_out",
                "quality_gate_passed": False,
                "forward_return_20d": 0.03,
                "max_drawdown_after_signal": -0.03,
            }
            for _ in range(5)
        ]

        health = build_profile_health_dashboard(
            pd.DataFrame(strong_rows + thin_rows),
            target_window=20,
            min_samples=30,
        )
        markdown = render_profile_health_dashboard(health)

        strong = health[health["screening_profile"] == "strong_profile"].iloc[0]
        thin = health[health["screening_profile"] == "thin_profile"].iloc[0]
        self.assertEqual(strong["profile_action"], "keep_profile")
        self.assertEqual(thin["profile_action"], "expand_sample")
        self.assertIn("Profile Health Dashboard / 分类规则健康面板", markdown)

    def test_profile_action_recommendations_translate_health_actions(self) -> None:
        profile_health = pd.DataFrame(
            [
                {
                    "screening_profile": "strong_profile",
                    "screening_profile_zh": "强规则",
                    "profile_action": "keep_profile",
                    "profile_reason": "Profile is acceptable.",
                    "profile_reason_zh": "规则可接受。",
                },
                {
                    "screening_profile": "weak_profile",
                    "screening_profile_zh": "弱规则",
                    "profile_action": "tighten_profile",
                    "profile_reason": "Win rate is too low.",
                    "profile_reason_zh": "胜率太低。",
                },
                {
                    "screening_profile": "thin_profile",
                    "screening_profile_zh": "薄样本规则",
                    "profile_action": "expand_sample",
                    "profile_reason": "Samples are thin.",
                    "profile_reason_zh": "样本偏少。",
                },
            ]
        )

        recommendations = build_profile_action_recommendations(profile_health)
        markdown = render_profile_action_recommendations(recommendations)

        weak = recommendations[
            recommendations["screening_profile"] == "weak_profile"
        ].iloc[0]
        thin = recommendations[
            recommendations["screening_profile"] == "thin_profile"
        ].iloc[0]
        self.assertEqual(weak["recommendation_action"], "tighten_signal_quality")
        self.assertIn("signal_score_min", weak["threshold_attrs"])
        self.assertEqual(thin["recommendation_action"], "expand_validation_sample")
        self.assertIn("Profile Action Recommendations / 分类规则行动建议", markdown)

    def test_profile_blocker_dashboard_groups_fail_reasons_by_profile(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "screening_profile": "saas_software",
                    "screening_profile_zh": "SaaS软件规则",
                    "quality_gate_fail_reasons": (
                        "signal score too low; breakout backtest sample too small"
                    ),
                    "quality_gate_fail_reasons_zh": (
                        "信号分数不足；突破买点回测样本不足"
                    ),
                },
                {
                    "screening_profile": "saas_software",
                    "screening_profile_zh": "SaaS软件规则",
                    "quality_gate_fail_reasons": "signal score too low",
                    "quality_gate_fail_reasons_zh": "信号分数不足",
                },
                {
                    "screening_profile": "semiconductor",
                    "screening_profile_zh": "半导体规则",
                    "quality_gate_fail_reasons": "market not supportive",
                    "quality_gate_fail_reasons_zh": "大盘环境不够支持",
                },
            ]
        )

        blockers = build_profile_blocker_dashboard(events)
        markdown = render_profile_blocker_dashboard(blockers)

        saas_top = blockers[
            (blockers["screening_profile"] == "saas_software")
            & (blockers["blocker_rank"] == 1)
        ].iloc[0]
        self.assertEqual(saas_top["blocker"], "signal score too low")
        self.assertEqual(saas_top["occurrence_count"], 2)
        self.assertEqual(saas_top["recommended_fix"], "tighten_signal_threshold")
        self.assertIn("Profile Blocker Dashboard / 分类规则卡点面板", markdown)

    def test_render_and_payload_include_bilingual_dashboard(self) -> None:
        dashboard = build_win_rate_dashboard(
            pd.DataFrame(
                [
                    {
                        "horizon": "short",
                        "screening_backtest_entry_type": "breakout",
                        "screening_profile": "default",
                        "screening_profile_zh": "默认规则",
                        "quality_gate_passed": True,
                        "forward_return_5d": 0.03,
                    }
                    for _ in range(12)
                ]
            ),
            forward_windows=(5,),
        )

        markdown = render_win_rate_dashboard(dashboard)
        payload = win_rate_dashboard_payload(dashboard)

        self.assertIn("Historical Win-Rate Dashboard / 历史胜率统计面板", markdown)
        self.assertIn("By Entry Type / 按买点类型", markdown)
        self.assertIn("overall", payload)
        self.assertEqual(payload["overall"][0]["result_label"], "promising_but_thin")

    def test_historical_win_rate_gate_allows_strong_sample_and_blocks_weak_group(self) -> None:
        strong_rows = [
            {
                "validation_bucket": "high_probability",
                "quality_gate_passed": True,
                "screening_profile": "default",
                "screening_profile_zh": "默认规则",
                "horizon": "short",
                "horizon_zh_label": "短期",
                "screening_backtest_entry_type": "breakout",
                "forward_return_20d": 0.04,
            }
            for _ in range(24)
        ]
        weak_rows = [
            {
                "validation_bucket": "filtered_out",
                "quality_gate_passed": False,
                "screening_profile": "weak_profile",
                "screening_profile_zh": "弱规则",
                "horizon": "medium",
                "horizon_zh_label": "中期",
                "screening_backtest_entry_type": "pullback",
                "forward_return_20d": -0.03,
            }
            for _ in range(6)
        ]
        events = pd.DataFrame(strong_rows + weak_rows)

        gate = build_historical_win_rate_gate(
            events,
            target_window=20,
            min_samples=20,
            min_win_rate=0.55,
        )
        markdown = render_historical_win_rate_gate(gate)

        quality_row = gate[
            (gate["scope"] == "quality_gate") & (gate["group_value"] == "passed")
        ].iloc[0]
        weak_profile = gate[
            (gate["scope"] == "screening_profile")
            & (gate["group_value"] == "weak_profile")
        ].iloc[0]
        self.assertEqual(
            quality_row["deployment_gate_action"],
            "allow_high_probability_filter",
        )
        self.assertEqual(
            weak_profile["deployment_gate_action"],
            "collect_more_samples",
        )
        self.assertIn("Historical Win-Rate Gate / 历史胜率部署门槛", markdown)

    def test_historical_threshold_recommendations_tighten_weak_win_rate(self) -> None:
        weak_rows = [
            {
                "validation_bucket": "high_probability" if index < 12 else "filtered_out",
                "quality_gate_passed": index < 12,
                "screening_profile": "default",
                "screening_profile_zh": "默认规则",
                "horizon": "short",
                "horizon_zh_label": "短期",
                "screening_backtest_entry_type": "breakout",
                "forward_return_20d": 0.04 if index % 3 == 0 else -0.02,
            }
            for index in range(36)
        ]
        gate = build_historical_win_rate_gate(
            pd.DataFrame(weak_rows),
            target_window=20,
            min_samples=20,
            min_win_rate=0.55,
        )

        recommendations = build_historical_threshold_recommendations(gate)
        markdown = render_historical_threshold_recommendations(recommendations)

        self.assertIn("Historical Threshold Recommendations / 历史阈值建议", markdown)
        attrs = set(recommendations["threshold_attr"].astype(str))
        self.assertIn("signal_score_min", attrs)
        self.assertIn("relative_strength_min", attrs)
        self.assertIn("backtest_win_rate_min", attrs)
        signal_row = recommendations[
            recommendations["threshold_attr"] == "signal_score_min"
        ].iloc[0]
        self.assertGreater(
            float(signal_row["suggested_threshold"]),
            float(signal_row["current_threshold"]),
        )

    def test_historical_threshold_recommendations_do_not_tune_thin_samples(self) -> None:
        gate = build_historical_win_rate_gate(
            pd.DataFrame(
                [
                    {
                        "validation_bucket": "filtered_out",
                        "quality_gate_passed": False,
                        "screening_profile": "default",
                        "screening_profile_zh": "默认规则",
                        "horizon": "short",
                        "horizon_zh_label": "短期",
                        "screening_backtest_entry_type": "breakout",
                        "forward_return_20d": 0.05,
                    }
                    for _ in range(3)
                ]
            ),
            target_window=20,
            min_samples=20,
        )

        recommendations = build_historical_threshold_recommendations(gate)

        self.assertIn(
            "expand_validation_coverage",
            recommendations["recommendation_action"].to_list(),
        )
        self.assertNotIn("signal_score_min", set(recommendations["threshold_attr"].astype(str)))


if __name__ == "__main__":
    unittest.main()
