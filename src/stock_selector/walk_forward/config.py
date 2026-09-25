"""Default grids, benchmark tickers and the minimum calibration sample size."""

from __future__ import annotations

from typing import Callable



DEFAULT_FORWARD_WINDOWS = (5, 20, 60)
DEFAULT_SIGNAL_THRESHOLDS = (55, 60, 65, 70, 75)
DEFAULT_HIGH_PROBABILITY_THRESHOLDS = (50, 55, 60, 65, 70)
DEFAULT_PROBABILITY_THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70)
DEFAULT_RELATIVE_STRENGTH_THRESHOLDS = (40, 45, 50, 55, 60)
DEFAULT_MARKET_THRESHOLDS = (50, 55, 60, 65, 70)
DEFAULT_BENCHMARK_TICKERS = ("SPY", "QQQ")

# Minimum completed walk-forward samples before a calibrated threshold is allowed to
# be WRITTEN into an adoptable config. Calibrating a trading threshold on 5-10 samples
# is statistically meaningless (huge confidence interval) and produces parameters that
# look smart but are overfit. 30 is a conservative floor for a system headed to live money.
MIN_CALIBRATION_SAMPLE_COUNT = 30

WalkForwardProgressCallback = Callable[[str, str, int, int], None]
