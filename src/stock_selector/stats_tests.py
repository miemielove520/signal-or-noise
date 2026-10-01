"""Statistical tests for judging a stock model: skill or noise?

Every function here is pure (arrays / frames in, numbers out) so the research
scripts stay thin and each test can be unit-tested on synthetic data.

Conventions
-----------
* Returns are simple (not log) returns, as fractions (0.01 = 1 %).
* Bootstrap routines take an explicit ``rng`` (``numpy.random.Generator``) so
  every reported interval is reproducible from a seed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------
# Bootstrap for time series
# --------------------------------------------------------------------------


def stationary_bootstrap_indices(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """One resample of ``range(n)`` using the Politis–Romano stationary bootstrap.

    Blocks start at random positions and have geometric length with mean
    ``mean_block``; indices wrap around. Keeping consecutive days together
    preserves the short-range autocorrelation (volatility clustering) that an
    i.i.d. bootstrap would destroy.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if mean_block < 1:
        raise ValueError("mean_block must be >= 1")
    p = 1.0 / mean_block
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(n)
    new_block = rng.random(n) < p
    starts = rng.integers(n, size=n)
    for t in range(1, n):
        idx[t] = starts[t] if new_block[t] else (idx[t - 1] + 1) % n
    return idx


@dataclass(frozen=True)
class BootstrapResult:
    """Point estimate, percentile confidence interval and the share of resamples at or below zero."""

    estimate: float
    ci_low: float
    ci_high: float
    p_value_le_zero: float  # share of resamples with statistic <= 0
    n_resamples: int


def bootstrap_statistic(
    values: np.ndarray | pd.Series,
    statistic,
    rng: np.random.Generator,
    n_resamples: int = 10_000,
    mean_block: float = 5.0,
    ci: float = 0.95,
) -> BootstrapResult:
    """Stationary-bootstrap percentile CI for ``statistic(values)``."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if x.size < 2:
        raise ValueError("need at least two observations")
    draws = np.empty(n_resamples)
    for i in range(n_resamples):
        draws[i] = statistic(x[stationary_bootstrap_indices(x.size, mean_block, rng)])
    alpha = (1.0 - ci) / 2.0
    return BootstrapResult(
        estimate=float(statistic(x)),
        ci_low=float(np.quantile(draws, alpha)),
        ci_high=float(np.quantile(draws, 1.0 - alpha)),
        p_value_le_zero=float(np.mean(draws <= 0.0)),
        n_resamples=n_resamples,
    )


def compounded_return(daily_returns: np.ndarray) -> float:
    """Total return from a sequence of daily simple returns."""
    return float(np.prod(1.0 + np.asarray(daily_returns, dtype=float)) - 1.0)


def simulate_window_returns(
    daily_returns: np.ndarray | pd.Series,
    window: int,
    rng: np.random.Generator,
    n_resamples: int = 10_000,
    mean_block: float = 5.0,
) -> np.ndarray:
    """Distribution of ``window``-day compounded returns implied by a history.

    Answers: *if the future looks like this history, what range of
    ``window``-day outcomes should we expect?* Each draw stitches together
    stationary-bootstrap blocks from the history until it is ``window`` days long.
    """
    x = np.asarray(daily_returns, dtype=float)
    x = x[~np.isnan(x)]
    if x.size < 2:
        raise ValueError("need at least two observations")
    out = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = stationary_bootstrap_indices(max(window, x.size), mean_block, rng)[:window]
        out[i] = compounded_return(x[idx])
    return out


# --------------------------------------------------------------------------
# Cross-sectional ranking skill
# --------------------------------------------------------------------------


def information_coefficients(
    frame: pd.DataFrame, score_col: str, return_col: str, date_col: str = "date"
) -> pd.Series:
    """Spearman rank correlation between score and forward return, per date.

    The mean IC over dates measures whether higher scores really rank stocks
    better. Each date is one independent observation; stocks within a date are not.
    """

    def _ic(group: pd.DataFrame) -> float:
        g = group[[score_col, return_col]].dropna()
        if len(g) < 3 or g[score_col].nunique() < 2 or g[return_col].nunique() < 2:
            return float("nan")
        return float(stats.spearmanr(g[score_col], g[return_col])[0])

    return frame.groupby(date_col)[[score_col, return_col]].apply(_ic).rename("ic")


@dataclass(frozen=True)
class MeanTest:
    """One-sample t-test of mean = 0 (with its 95 % interval) and sign / Wilcoxon checks."""

    mean: float
    std: float
    n: int
    t_stat: float
    p_value_two_sided: float
    positive_count: int
    sign_test_p_one_sided: float
    ci_low: float = float("nan")
    ci_high: float = float("nan")
    sign_test_p_two_sided: float = float("nan")
    wilcoxon_p_two_sided: float = float("nan")


def one_sample_mean_test(values: np.ndarray | pd.Series) -> MeanTest:
    """t-test of mean = 0, plus a distribution-free sign test (H1: median > 0)."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n < 2:
        raise ValueError("need at least two observations")
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    t = mean / (sd / math.sqrt(n)) if sd > 0 else float("inf") * np.sign(mean)
    p_t = float(2 * stats.t.sf(abs(t), df=n - 1)) if np.isfinite(t) else 0.0
    positives = int((x > 0).sum())
    nonzero = int((x != 0).sum())
    p_sign = float(stats.binomtest(positives, nonzero, 0.5, alternative="greater").pvalue) if nonzero else 1.0
    p_sign_two = float(stats.binomtest(positives, nonzero, 0.5).pvalue) if nonzero else 1.0
    half = float(stats.t.ppf(0.975, df=n - 1)) * sd / math.sqrt(n)
    p_wilcoxon = float(stats.wilcoxon(x[x != 0]).pvalue) if nonzero >= 2 else float("nan")
    return MeanTest(
        mean, sd, n, float(t), p_t, positives, p_sign, mean - half, mean + half, p_sign_two, p_wilcoxon
    )


def newey_west_t(values: np.ndarray | pd.Series, lags: int = 5) -> float:
    """t-statistic of the mean with a Newey–West (Bartlett) long-run variance."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n <= lags + 1:
        raise ValueError("need more observations than lags")
    e = x - x.mean()
    long_run = float(e @ e) / n
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1)
        long_run += 2.0 * weight * float(e[lag:] @ e[:-lag]) / n
    return float(x.mean() / math.sqrt(long_run / n))


def required_sample_size(effect: float, sd: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Observations needed for a two-sided one-sample test to detect ``effect`` (normal approximation)."""
    if effect == 0 or sd <= 0:
        raise ValueError("effect must be non-zero and sd positive")
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return int(math.ceil((z * sd / abs(effect)) ** 2))


def bonferroni_t_threshold(n_trials: int, df: int, alpha: float = 0.05) -> float:
    """Two-sided |t| a single result must exceed after ``n_trials`` looks.

    Trying many configurations and keeping the best one inflates false
    positives; Bonferroni is the simplest (conservative) correction.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    return float(stats.t.isf(alpha / (2 * n_trials), df))


# --------------------------------------------------------------------------
# Probability calibration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BrierResult:
    """Brier score of the forecasts, of the constant base-rate forecast, and the skill score between them."""

    brier: float
    reference_brier: float  # always predicting the base rate
    skill_score: float  # 1 - brier / reference; < 0 = worse than the base rate
    base_rate: float
    n: int


def brier_skill(probabilities: np.ndarray, outcomes: np.ndarray) -> BrierResult:
    """Brier score of probability forecasts and its skill relative to always predicting the observed base rate."""
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    mask = ~(np.isnan(p) | np.isnan(y))
    p, y = p[mask], y[mask]
    if p.size == 0:
        raise ValueError("no observations")
    base = float(y.mean())
    brier = float(np.mean((p - y) ** 2))
    ref = float(np.mean((base - y) ** 2))
    skill = 1.0 - brier / ref if ref > 0 else float("nan")
    return BrierResult(brier, ref, skill, base, int(p.size))


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    phat = successes / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def reliability_table(
    probabilities: np.ndarray | pd.Series,
    outcomes: np.ndarray | pd.Series,
    bins: list[float],
) -> pd.DataFrame:
    """Predicted vs. observed frequency per probability bin, with Wilson 95 % CIs."""
    frame = pd.DataFrame({"p": np.asarray(probabilities, float), "y": np.asarray(outcomes, float)}).dropna()
    frame["bin"] = pd.cut(frame["p"], bins=bins, include_lowest=True)
    rows = []
    for interval, g in frame.groupby("bin", observed=True):
        k, n = int(g["y"].sum()), len(g)
        lo, hi = wilson_interval(k, n)
        rows.append(
            {
                "bin": str(interval),
                "n": n,
                "mean_predicted": float(g["p"].mean()),
                "observed": k / n,
                "ci_low": lo,
                "ci_high": hi,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Sample-size hygiene
# --------------------------------------------------------------------------


def cluster_bootstrap(
    frame: pd.DataFrame,
    cluster_col: str,
    statistic,
    rng: np.random.Generator,
    n_resamples: int = 4_000,
    ci: float = 0.95,
) -> BootstrapResult:
    """Percentile CI for ``statistic(frame)`` resampling whole clusters (e.g. dates) with replacement.

    Rows in the same cluster stay together, so within-cluster correlation is
    reflected in the interval instead of being treated as independent evidence.
    """
    groups = [g for _, g in frame.groupby(cluster_col)]
    if len(groups) < 2:
        raise ValueError("need at least two clusters")
    draws = np.empty(n_resamples)
    for i in range(n_resamples):
        picks = rng.integers(len(groups), size=len(groups))
        draws[i] = statistic(pd.concat([groups[j] for j in picks], ignore_index=True))
    alpha = (1.0 - ci) / 2.0
    return BootstrapResult(
        estimate=float(statistic(frame)),
        ci_low=float(np.quantile(draws, alpha)),
        ci_high=float(np.quantile(draws, 1.0 - alpha)),
        p_value_le_zero=float(np.mean(draws <= 0.0)),
        n_resamples=n_resamples,
    )


@dataclass(frozen=True)
class BrierDecomposition:
    """Murphy (1973) decomposition: Brier ≈ reliability − resolution + uncertainty (binned)."""

    reliability: float
    resolution: float
    uncertainty: float
    mean_forecast: float
    base_rate: float


def brier_decomposition(probabilities, outcomes, bins: list[float]) -> BrierDecomposition:
    """Reliability (miscalibration), resolution (separation) and uncertainty of binned forecasts."""
    frame = pd.DataFrame({"p": np.asarray(probabilities, float), "y": np.asarray(outcomes, float)}).dropna()
    frame["bin"] = pd.cut(frame["p"], bins=bins, include_lowest=True)
    base = float(frame["y"].mean())
    n = len(frame)
    reliability = resolution = 0.0
    for _, g in frame.groupby("bin", observed=True):
        weight = len(g) / n
        reliability += weight * (g["p"].mean() - g["y"].mean()) ** 2
        resolution += weight * (g["y"].mean() - base) ** 2
    return BrierDecomposition(
        float(reliability), float(resolution), base * (1 - base), float(frame["p"].mean()), base
    )


def intraclass_correlation(frame: pd.DataFrame, cluster_col: str, value_col: str) -> float:
    """One-way ANOVA estimate of the share of variance shared within clusters."""
    g = frame.groupby(cluster_col)[value_col]
    k = g.ngroups
    sizes = g.size()
    n = int(sizes.sum())
    m0 = (n - float((sizes**2).sum()) / n) / (k - 1)
    grand = frame[value_col].mean()
    msb = float((sizes * (g.mean() - grand) ** 2).sum()) / (k - 1)
    msw = float(((frame[value_col] - g.transform("mean")) ** 2).sum()) / (n - k)
    return (msb - msw) / (msb + (m0 - 1) * msw)


def duplication_factor(frame: pd.DataFrame, key_cols: list[str]) -> float:
    """Rows per unique key. > 1 means observations are counted more than once."""
    unique = len(frame.drop_duplicates(key_cols))
    return len(frame) / unique if unique else float("nan")
