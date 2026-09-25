"""Walk-forward validation: replay past signals, measure what happened next, calibrate thresholds.

The public API is re-exported here; implementation lives in the submodules."""

from __future__ import annotations

from .calibration import (
    build_overfitting_risk_report,
    calibrate_walk_forward_profiles,
    calibrate_walk_forward_rules,
    render_suggested_screening_config,
)
from .config import (
    DEFAULT_BENCHMARK_TICKERS,
    DEFAULT_FORWARD_WINDOWS,
    DEFAULT_HIGH_PROBABILITY_THRESHOLDS,
    DEFAULT_MARKET_THRESHOLDS,
    DEFAULT_PROBABILITY_THRESHOLDS,
    DEFAULT_RELATIVE_STRENGTH_THRESHOLDS,
    DEFAULT_SIGNAL_THRESHOLDS,
    MIN_CALIBRATION_SAMPLE_COUNT,
)
from .core import (
    WalkForwardResult,
    run_walk_forward_validation,
)
from .events import (
    build_validation_market_regime_lookup,
)
from .policy import (
    build_benchmark_aware_policy,
    build_benchmark_tightening_recommendations,
    build_minimum_sample_guard,
    build_threshold_sensitivity_grid,
    build_tightening_impact_validation,
)
from .portfolio import (
    build_walk_forward_benchmark_comparison,
    build_walk_forward_portfolio_equity,
    summarize_walk_forward_portfolios,
)
from .report import (
    render_walk_forward_report,
)
from .summaries import (
    build_market_regime_protection_policy,
    build_sample_sufficiency_guidance,
    build_ticker_validation_ranking,
    summarize_market_regime_validation,
    summarize_probability_calibration,
    summarize_walk_forward_events,
    summarize_walk_forward_profiles,
    summarize_walk_forward_segments,
)

__all__ = [
    "DEFAULT_FORWARD_WINDOWS",
    "DEFAULT_SIGNAL_THRESHOLDS",
    "DEFAULT_HIGH_PROBABILITY_THRESHOLDS",
    "DEFAULT_PROBABILITY_THRESHOLDS",
    "DEFAULT_RELATIVE_STRENGTH_THRESHOLDS",
    "DEFAULT_MARKET_THRESHOLDS",
    "DEFAULT_BENCHMARK_TICKERS",
    "MIN_CALIBRATION_SAMPLE_COUNT",
    "WalkForwardResult",
    "run_walk_forward_validation",
    "summarize_walk_forward_events",
    "build_ticker_validation_ranking",
    "build_sample_sufficiency_guidance",
    "summarize_walk_forward_profiles",
    "summarize_walk_forward_segments",
    "summarize_market_regime_validation",
    "build_market_regime_protection_policy",
    "summarize_probability_calibration",
    "summarize_walk_forward_portfolios",
    "build_walk_forward_portfolio_equity",
    "build_walk_forward_benchmark_comparison",
    "build_benchmark_aware_policy",
    "build_benchmark_tightening_recommendations",
    "build_tightening_impact_validation",
    "build_threshold_sensitivity_grid",
    "build_minimum_sample_guard",
    "calibrate_walk_forward_rules",
    "calibrate_walk_forward_profiles",
    "build_overfitting_risk_report",
    "render_suggested_screening_config",
    "render_walk_forward_report",
    "build_validation_market_regime_lookup",
]
