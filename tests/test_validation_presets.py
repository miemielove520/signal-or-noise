from __future__ import annotations

import unittest
from argparse import Namespace

from stock_selector.validation_presets import apply_validation_preset


class ValidationPresetTest(unittest.TestCase):
    def test_quick_preset_uses_research_core_when_no_manual_universe(self) -> None:
        args = Namespace(
            preset="quick",
            tickers=[],
            universe=None,
            universe_file=None,
            all_universes=False,
            period=None,
            step_days=None,
            min_history_days=None,
            output_dir=None,
        )

        result = apply_validation_preset(args)

        self.assertEqual(result.universe, "research-core")
        self.assertFalse(result.all_universes)
        self.assertEqual(result.period, "1y")
        self.assertEqual(result.step_days, 120)
        self.assertEqual(result.min_history_days, 80)

    def test_preset_does_not_override_manual_tickers(self) -> None:
        args = Namespace(
            preset="quick",
            tickers=["AAPL"],
            universe=None,
            universe_file=None,
            all_universes=False,
            period=None,
            step_days=None,
            min_history_days=None,
            output_dir=None,
        )

        result = apply_validation_preset(args)

        self.assertIsNone(result.universe)
        self.assertFalse(result.all_universes)
        self.assertEqual(result.period, "1y")

    def test_standard_preset_uses_research_grade_sample_settings(self) -> None:
        args = Namespace(
            preset="standard",
            tickers=[],
            universe=None,
            universe_file=None,
            all_universes=False,
            period=None,
            step_days=None,
            min_history_days=None,
            output_dir=None,
        )

        result = apply_validation_preset(args)

        self.assertEqual(result.universe, "research-core")
        self.assertEqual(result.period, "5y")
        self.assertEqual(result.step_days, 20)
        self.assertEqual(result.min_history_days, 170)
        self.assertIn("研究级", result.preset_description_zh)

    def test_deep_preset_uses_all_universes_without_manual_selection(self) -> None:
        args = Namespace(
            preset="deep",
            tickers=[],
            universe=None,
            universe_file=None,
            all_universes=False,
            period=None,
            step_days=None,
            min_history_days=None,
            output_dir=None,
        )

        result = apply_validation_preset(args)

        self.assertTrue(result.all_universes)
        self.assertIsNone(result.universe)
        self.assertEqual(result.period, "10y")
        self.assertEqual(result.min_history_days, 252)

    def test_extreme_preset_uses_dense_all_universe_validation(self) -> None:
        args = Namespace(
            preset="extreme",
            tickers=[],
            universe=None,
            universe_file=None,
            all_universes=False,
            period=None,
            step_days=None,
            min_history_days=None,
            output_dir=None,
        )

        result = apply_validation_preset(args)

        self.assertTrue(result.all_universes)
        self.assertEqual(result.period, "10y")
        self.assertEqual(result.step_days, 10)
        self.assertEqual(result.min_history_days, 252)
        self.assertIn("最大覆盖", result.preset_description_zh)


if __name__ == "__main__":
    unittest.main()
