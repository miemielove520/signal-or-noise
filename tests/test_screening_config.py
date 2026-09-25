from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from stock_selector.screening_config import (
    ScreeningThresholds,
    TradingRules,
    default_screening_config,
    load_screening_config,
    load_screening_thresholds,
)


class ScreeningConfigTest(unittest.TestCase):
    def test_load_screening_thresholds_from_toml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "screening.toml"
            path.write_text(
                "\n".join(
                    [
                        "[screening_thresholds]",
                        "signal_score_min = 70",
                        "backtest_sample_min = 15",
                        "backtest_win_rate_min = 0.6",
                        "max_backtest_stop_hit_rate = 0.44",
                        "backtest_trust_score_min = 66",
                        "recent_backtest_score_min = 52",
                        "backtest_decay_score_min = 53",
                        "min_backtest_avg_dollar_volume = 25000000",
                        "max_backtest_slippage_pct = 0.004",
                    ]
                ),
                encoding="utf-8",
            )

            thresholds = load_screening_thresholds(path)

        self.assertEqual(thresholds.signal_score_min, 70.0)
        self.assertEqual(thresholds.backtest_sample_min, 15)
        self.assertEqual(thresholds.backtest_win_rate_min, 0.60)
        self.assertEqual(thresholds.max_backtest_stop_hit_rate, 0.44)
        self.assertEqual(thresholds.backtest_trust_score_min, 66.0)
        self.assertEqual(thresholds.recent_backtest_score_min, 52.0)
        self.assertEqual(thresholds.backtest_decay_score_min, 53.0)
        self.assertEqual(thresholds.min_backtest_avg_dollar_volume, 25_000_000.0)
        self.assertEqual(thresholds.max_backtest_slippage_pct, 0.004)
        self.assertEqual(thresholds.market_score_min, ScreeningThresholds().market_score_min)

    def test_unknown_threshold_key_raises_clear_error(self) -> None:
        with self.assertRaises(ValueError):
            ScreeningThresholds.from_mapping({"unknown_rule": 1})

    def test_default_config_resolves_stock_type_profiles(self) -> None:
        config = default_screening_config()

        saas = config.resolve_profile("NOW", "Technology", "Software - Application")
        ai_infrastructure = config.resolve_profile("NVDA", "Technology", "Semiconductors")
        semiconductor = config.resolve_profile("INTC", "Technology", "Semiconductors")
        cybersecurity = config.resolve_profile("PANW", "Technology", "Software - Infrastructure")
        cybersecurity_replacement = config.resolve_profile("QLYS", "Technology", "Software - Infrastructure")
        fintech = config.resolve_profile("PYPL", "Financial Services", "Credit Services")
        block = config.resolve_profile("XYZ", "Financial Services", "Credit Services")
        defensive = config.resolve_profile("COST", "Consumer Defensive", "Discount Stores")
        industrial = config.resolve_profile("AAON", "Industrials", "Building Products")
        mega_cap = config.resolve_profile("AAPL", "Technology", "Consumer Electronics")

        self.assertEqual(saas.name, "saas_software")
        self.assertEqual(saas.thresholds.relative_strength_min, 50.0)
        self.assertEqual(ai_infrastructure.name, "ai_infrastructure")
        self.assertEqual(ai_infrastructure.thresholds.medium_long_sector_score_min, 52.0)
        self.assertEqual(ai_infrastructure.trading_rules.preferred_entry_style, "breakout")
        self.assertEqual(ai_infrastructure.trading_rules.short_target_r_multiple, 2.2)
        self.assertEqual(semiconductor.name, "semiconductor")
        self.assertEqual(semiconductor.thresholds.backtest_sample_min, 12)
        self.assertEqual(semiconductor.trading_rules.medium_atr_stop_multiple, 2.2)
        self.assertEqual(cybersecurity.name, "cybersecurity")
        self.assertEqual(cybersecurity_replacement.name, "cybersecurity")
        self.assertEqual(cybersecurity.thresholds.relative_strength_min, 52.0)
        self.assertEqual(fintech.name, "fintech_high_beta")
        self.assertEqual(block.name, "fintech_high_beta")
        self.assertEqual(fintech.thresholds.backtest_sample_min, 14)
        self.assertEqual(fintech.trading_rules.preferred_entry_style, "pullback")
        self.assertEqual(fintech.thresholds.max_backtest_stop_hit_rate, 0.52)
        self.assertEqual(fintech.thresholds.backtest_trust_score_min, 65.0)
        self.assertEqual(fintech.thresholds.recent_backtest_score_min, 50.0)
        self.assertEqual(fintech.thresholds.backtest_decay_score_min, 50.0)
        self.assertEqual(defensive.name, "defensive_quality")
        self.assertEqual(defensive.thresholds.signal_score_min, 62.0)
        self.assertEqual(defensive.trading_rules.long_time_stop_days, 160)
        self.assertEqual(industrial.name, "industrial_quality")
        self.assertEqual(industrial.thresholds.signal_score_min, 64.0)
        self.assertEqual(industrial.thresholds.long_fundamental_score_min, 62.0)
        self.assertEqual(mega_cap.name, "mega_cap_tech")
        self.assertEqual(mega_cap.thresholds.data_quality_min, 72.0)
        self.assertEqual(mega_cap.thresholds.max_backtest_stop_hit_rate, 0.50)
        self.assertEqual(mega_cap.thresholds.backtest_trust_score_min, 60.0)
        self.assertEqual(mega_cap.thresholds.recent_backtest_score_min, 45.0)
        self.assertEqual(mega_cap.thresholds.backtest_decay_score_min, 45.0)
        self.assertEqual(mega_cap.trading_rules.sell_rule, "mega_cap_trend_or_quality_break")

    def test_trading_rules_reject_unknown_keys(self) -> None:
        with self.assertRaises(ValueError):
            TradingRules.from_mapping({"unknown_rule": 1})

    def test_load_screening_config_keeps_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "screening.toml"
            path.write_text(
                "\n".join(
                    [
                        "[screening_thresholds]",
                        "signal_score_min = 65",
                        "",
                        "[profiles.custom_software]",
                        'name_zh = "自定义软件规则"',
                        'industry_keywords = ["software"]',
                        "signal_score_min = 71",
                        "",
                        "[profiles.custom_software.trading]",
                        'preferred_entry_style = "pullback"',
                        'preferred_entry_style_zh = "回调优先"',
                        "short_target_r_multiple = 2.7",
                        "short_atr_stop_multiple = 1.8",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_screening_config(path)

        profile = config.resolve_profile("CRM", "Technology", "Software")
        self.assertEqual(profile.name, "custom_software")
        self.assertEqual(profile.thresholds.signal_score_min, 71.0)
        self.assertEqual(profile.trading_rules.preferred_entry_style, "pullback")
        self.assertEqual(profile.trading_rules.short_target_r_multiple, 2.7)
        self.assertEqual(profile.trading_rules.short_atr_stop_multiple, 1.8)


if __name__ == "__main__":
    unittest.main()
