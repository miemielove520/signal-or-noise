import unittest

import numpy as np
import pandas as pd

from stock_selector.stats_tests import (
    bonferroni_t_threshold,
    bootstrap_statistic,
    brier_skill,
    compounded_return,
    duplication_factor,
    information_coefficients,
    one_sample_mean_test,
    reliability_table,
    simulate_window_returns,
    stationary_bootstrap_indices,
    wilson_interval,
)


class BootstrapTests(unittest.TestCase):
    def test_indices_in_range_and_reproducible(self):
        a = stationary_bootstrap_indices(50, 5.0, np.random.default_rng(1))
        b = stationary_bootstrap_indices(50, 5.0, np.random.default_rng(1))
        self.assertTrue(np.array_equal(a, b))
        self.assertTrue(((a >= 0) & (a < 50)).all())

    def test_block_length_one_behaves_like_iid(self):
        idx = stationary_bootstrap_indices(1000, 1.0, np.random.default_rng(0))
        consecutive = np.mean(np.diff(idx) == 1)
        self.assertLess(consecutive, 0.02)

    def test_long_blocks_keep_neighbours_together(self):
        idx = stationary_bootstrap_indices(1000, 50.0, np.random.default_rng(0))
        consecutive = np.mean((np.diff(idx) % 1000) == 1)
        self.assertGreater(consecutive, 0.9)

    def test_ci_covers_true_mean(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0.001, 0.01, 500)
        res = bootstrap_statistic(x, np.mean, np.random.default_rng(0), n_resamples=2000)
        self.assertLess(res.ci_low, 0.001)
        self.assertGreater(res.ci_high, 0.001)

    def test_clearly_negative_series_has_small_p(self):
        x = np.full(100, -0.01) + np.random.default_rng(0).normal(0, 0.001, 100)
        res = bootstrap_statistic(x, np.mean, np.random.default_rng(0), n_resamples=1000)
        self.assertEqual(res.p_value_le_zero, 1.0)
        self.assertLess(res.ci_high, 0)

    def test_simulated_windows_match_constant_return(self):
        out = simulate_window_returns(np.full(30, 0.01), 10, np.random.default_rng(0), n_resamples=50)
        self.assertTrue(np.allclose(out, 1.01 ** 10 - 1))

    def test_compounded_return(self):
        self.assertAlmostEqual(compounded_return([0.1, -0.1]), -0.01)


class RankingTests(unittest.TestCase):
    def test_perfect_ranking_gives_ic_one(self):
        frame = pd.DataFrame({
            "date": ["d1"] * 5 + ["d2"] * 5,
            "score": list(range(5)) * 2,
            "ret": [0.01 * i for i in range(5)] * 2,
        })
        ic = information_coefficients(frame, "score", "ret")
        self.assertTrue(np.allclose(ic.values, 1.0))

    def test_constant_score_is_nan_not_error(self):
        frame = pd.DataFrame({"date": ["d"] * 4, "score": [1] * 4, "ret": [0.1, 0.2, 0.3, 0.4]})
        self.assertTrue(np.isnan(information_coefficients(frame, "score", "ret").iloc[0]))

    def test_mean_test_and_sign_test(self):
        res = one_sample_mean_test([0.02, 0.03, 0.01, 0.04, 0.02, -0.01])
        self.assertEqual(res.n, 6)
        self.assertEqual(res.positive_count, 5)
        self.assertGreater(res.t_stat, 0)
        self.assertAlmostEqual(res.sign_test_p_one_sided, 7 / 64)

    def test_bonferroni_grows_with_trials(self):
        one = bonferroni_t_threshold(1, 13)
        many = bonferroni_t_threshold(35, 13)
        self.assertAlmostEqual(one, 2.160, places=2)
        self.assertGreater(many, one)


class CalibrationTests(unittest.TestCase):
    def test_base_rate_forecast_has_zero_skill(self):
        y = np.array([1, 0, 1, 0])
        res = brier_skill(np.full(4, 0.5), y)
        self.assertAlmostEqual(res.skill_score, 0.0)

    def test_overconfident_wrong_forecast_has_negative_skill(self):
        y = np.array([1, 0, 1, 0])
        res = brier_skill(np.array([0.1, 0.9, 0.1, 0.9]), y)
        self.assertLess(res.skill_score, 0)

    def test_wilson_interval_contains_phat(self):
        lo, hi = wilson_interval(30, 100)
        self.assertLess(lo, 0.3)
        self.assertGreater(hi, 0.3)

    def test_reliability_table_bins(self):
        p = [0.1, 0.2, 0.7, 0.8]
        y = [0, 0, 1, 1]
        table = reliability_table(p, y, [0, 0.5, 1])
        self.assertEqual(list(table["n"]), [2, 2])
        self.assertEqual(list(table["observed"]), [0.0, 1.0])


class SampleHygieneTests(unittest.TestCase):
    def test_duplication_factor(self):
        frame = pd.DataFrame({"date": ["a"] * 3 + ["b"] * 3, "ticker": ["X"] * 6, "h": [1, 2, 3] * 2})
        self.assertEqual(duplication_factor(frame, ["date", "ticker"]), 3.0)


if __name__ == "__main__":
    unittest.main()
