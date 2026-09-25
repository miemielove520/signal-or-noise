"""Single-ticker analysis: horizons, signals, backtests, gates and reports.

The public API is re-exported here; implementation lives in the submodules."""

from __future__ import annotations

from .core import (
    analyze_ticker,
)
from .horizons import (
    BREAKOUT_EXTENSION_MULTIPLE,
    HORIZON_SPECS,
    HorizonSpec,
    normalize_horizons,
)
from .render import (
    render_ticker_analysis,
)

__all__ = [
    "HORIZON_SPECS",
    "HorizonSpec",
    "BREAKOUT_EXTENSION_MULTIPLE",
    "analyze_ticker",
    "normalize_horizons",
    "render_ticker_analysis",
]
