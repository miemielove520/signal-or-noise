from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FundamentalContext:
    ticker: str
    fundamental_score: float
    fundamental_quality: str
    fundamental_quality_zh: str
    fundamental_note: str
    fundamental_note_zh: str
    fundamental_warning: str
    data_coverage: float
    revenue_growth: float | None
    earnings_growth: float | None
    profit_margin: float | None
    return_on_equity: float | None
    free_cash_flow: float | None
    forward_pe: float | None
    peg_ratio: float | None
    debt_to_equity: float | None
    source: str
    # Deeper structure derived from fields already present in the snapshot.
    cash_flow_quality_score: float | None = None
    gross_margin: float | None = None
    operating_margin: float | None = None
    fcf_margin: float | None = None
    cash_conversion: float | None = None
    net_debt: float | None = None
    net_debt_to_equity: float | None = None
    # Multi-period trend direction from point-in-time SEC history (optional).
    trend_status: str = "insufficient_history"
    trend_direction: str = "unknown"
    trend_direction_zh: str = "未知"
    trend_score: float | None = None
    gross_margin_trend: str = "unknown"
    operating_margin_trend: str = "unknown"
    net_margin_trend: str = "unknown"
    fcf_margin_trend: str = "unknown"
    revenue_growth_trend: str = "unknown"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_fundamental_context(
    ticker: str,
    snapshot: dict[str, Any] | None,
    trend_context: object | None = None,
) -> FundamentalContext:
    ticker = ticker.upper().strip()
    if not snapshot:
        return _unknown_context(ticker, "snapshot unavailable")

    metrics = {
        "revenue_growth": _safe_float(snapshot.get("revenue_growth")),
        "earnings_growth": _safe_float(snapshot.get("earnings_growth")),
        "profit_margin": _safe_float(snapshot.get("profit_margin")),
        "return_on_equity": _safe_float(snapshot.get("return_on_equity")),
        "free_cash_flow": _safe_float(snapshot.get("free_cash_flow")),
        "forward_pe": _safe_float(snapshot.get("forward_pe")),
        "peg_ratio": _safe_float(snapshot.get("peg_ratio")),
        "debt_to_equity": _safe_float(snapshot.get("debt_to_equity")),
    }
    available = [value for value in metrics.values() if value is not None]
    coverage = len(available) / len(metrics)
    if coverage < 0.25:
        return _unknown_context(ticker, "too few fundamental fields available")

    # Deeper structure derived from fields already present in the snapshot but
    # previously unused: margin structure, cash-flow quality, and net debt.
    depth = _derive_depth_metrics(snapshot, metrics["free_cash_flow"])

    profitability = _average_available(
        [
            _score_margin(metrics["profit_margin"]),
            _score_gross_margin(depth["gross_margin"]),
            _score_operating_margin(depth["operating_margin"]),
            _score_roe(metrics["return_on_equity"]),
        ]
    )
    # Earnings quality: is reported profit backed by real cash?
    cash_flow_quality = _average_available(
        [
            _score_cash_flow(metrics["free_cash_flow"]),
            _score_cash_conversion(depth["cash_conversion"]),
            _score_fcf_margin(depth["fcf_margin"]),
        ]
    )
    growth = _average_available(
        [
            _score_growth(metrics["revenue_growth"]),
            _score_growth(metrics["earnings_growth"]),
        ]
    )
    valuation = _average_available(
        [
            _score_forward_pe(metrics["forward_pe"]),
            _score_peg(metrics["peg_ratio"]),
        ]
    )
    balance_sheet = _average_available(
        [
            _score_debt(metrics["debt_to_equity"]),
            _score_net_debt_to_equity(depth["net_debt_to_equity"]),
        ]
    )

    score = _average_available(
        [
            (profitability, 0.28),
            (cash_flow_quality, 0.20),
            (growth, 0.22),
            (valuation, 0.15),
            (balance_sheet, 0.15),
        ],
        weighted=True,
    )
    if score is None:
        return _unknown_context(ticker, "fundamental score could not be calculated")
    score = round(float(max(0.0, min(score, 100.0))), 2)
    quality, quality_zh = _quality_from_score(score)
    note, note_zh = _build_note(metrics, depth, score, quality, quality_zh, coverage)

    return FundamentalContext(
        ticker=ticker,
        fundamental_score=score,
        fundamental_quality=quality,
        fundamental_quality_zh=quality_zh,
        fundamental_note=note,
        fundamental_note_zh=note_zh,
        fundamental_warning="",
        data_coverage=round(coverage, 2),
        revenue_growth=metrics["revenue_growth"],
        earnings_growth=metrics["earnings_growth"],
        profit_margin=metrics["profit_margin"],
        return_on_equity=metrics["return_on_equity"],
        free_cash_flow=metrics["free_cash_flow"],
        forward_pe=metrics["forward_pe"],
        peg_ratio=metrics["peg_ratio"],
        debt_to_equity=metrics["debt_to_equity"],
        source=str(
            snapshot.get("fundamental_data_source")
            or snapshot.get("data_source")
            or "yfinance_restated"
        ),
        cash_flow_quality_score=(
            round(float(cash_flow_quality), 2) if cash_flow_quality is not None else None
        ),
        gross_margin=depth["gross_margin"],
        operating_margin=depth["operating_margin"],
        fcf_margin=depth["fcf_margin"],
        cash_conversion=depth["cash_conversion"],
        net_debt=depth["net_debt"],
        net_debt_to_equity=depth["net_debt_to_equity"],
        **_trend_fields(trend_context),
    )


def _unknown_context(ticker: str, warning: str) -> FundamentalContext:
    return FundamentalContext(
        ticker=ticker,
        fundamental_score=50.0,
        fundamental_quality="unknown",
        fundamental_quality_zh="未知",
        fundamental_note=(
            "Fundamental data is unavailable or too sparse; neutral score 50 is used."
        ),
        fundamental_note_zh="基本面数据不可用或字段太少；使用中性分数50。",
        fundamental_warning=warning,
        data_coverage=0.0,
        revenue_growth=None,
        earnings_growth=None,
        profit_margin=None,
        return_on_equity=None,
        free_cash_flow=None,
        forward_pe=None,
        peg_ratio=None,
        debt_to_equity=None,
        source="unavailable",
    )


def _trend_fields(trend_context: object | None) -> dict[str, object]:
    """Extract trend fields from a FundamentalTrendContext (duck-typed). Empty when
    absent so the dataclass defaults ("unknown") apply."""
    if trend_context is None:
        return {}

    def get(name: str, default: object) -> object:
        return getattr(trend_context, name, default)

    return {
        "trend_status": str(get("status", "insufficient_history")),
        "trend_direction": str(get("trend_direction", "unknown")),
        "trend_direction_zh": str(get("trend_direction_zh", "未知")),
        "trend_score": _safe_float(get("trend_score", None)),
        "gross_margin_trend": str(get("gross_margin_trend", "unknown")),
        "operating_margin_trend": str(get("operating_margin_trend", "unknown")),
        "net_margin_trend": str(get("net_margin_trend", "unknown")),
        "fcf_margin_trend": str(get("fcf_margin_trend", "unknown")),
        "revenue_growth_trend": str(get("revenue_growth_trend", "unknown")),
    }


def _derive_depth_metrics(
    snapshot: dict[str, Any], free_cash_flow: float | None
) -> dict[str, float | None]:
    gross_margin = _safe_float(snapshot.get("gross_margin"))
    operating_margin = _safe_float(snapshot.get("operating_margin"))
    revenue = _safe_float(snapshot.get("revenue"))
    net_income = _safe_float(snapshot.get("net_income"))
    operating_cash_flow = _safe_float(snapshot.get("operating_cash_flow"))
    total_cash = _safe_float(snapshot.get("total_cash"))
    total_debt = _safe_float(snapshot.get("total_debt"))
    equity = _safe_float(snapshot.get("shareholders_equity"))

    fcf_margin = (
        free_cash_flow / revenue
        if free_cash_flow is not None and revenue is not None and revenue > 0
        else None
    )
    # Cash conversion only makes sense against positive net income; a negative
    # denominator would flip the sign and mislead.
    cash_conversion = (
        operating_cash_flow / net_income
        if operating_cash_flow is not None and net_income is not None and net_income > 0
        else None
    )
    net_debt = (
        total_debt - total_cash
        if total_debt is not None and total_cash is not None
        else None
    )
    net_debt_to_equity = (
        net_debt / equity if net_debt is not None and equity is not None and equity > 0 else None
    )
    return {
        "gross_margin": gross_margin,
        "operating_margin": operating_margin,
        "fcf_margin": _finite(fcf_margin),
        "cash_conversion": _finite(cash_conversion),
        "net_debt": net_debt,
        "net_debt_to_equity": _finite(net_debt_to_equity),
    }


def _score_gross_margin(value: float | None) -> float | None:
    if value is None:
        return None
    if value >= 0.60:
        return 90.0
    if value >= 0.40:
        return 75.0
    if value >= 0.25:
        return 60.0
    if value >= 0.10:
        return 45.0
    return 25.0


def _score_operating_margin(value: float | None) -> float | None:
    if value is None:
        return None
    if value >= 0.25:
        return 90.0
    if value >= 0.15:
        return 75.0
    if value >= 0.08:
        return 60.0
    if value >= 0.0:
        return 45.0
    return 15.0


def _score_cash_conversion(value: float | None) -> float | None:
    # Operating cash flow / net income. >=1 means earnings are backed by cash.
    if value is None:
        return None
    if value >= 1.1:
        return 90.0
    if value >= 0.9:
        return 78.0
    if value >= 0.7:
        return 62.0
    if value >= 0.5:
        return 45.0
    if value > 0.0:
        return 30.0
    return 10.0


def _score_fcf_margin(value: float | None) -> float | None:
    if value is None:
        return None
    if value >= 0.20:
        return 90.0
    if value >= 0.10:
        return 75.0
    if value >= 0.05:
        return 60.0
    if value >= 0.0:
        return 45.0
    return 15.0


def _score_net_debt_to_equity(value: float | None) -> float | None:
    if value is None:
        return None
    if value <= 0.0:  # net cash
        return 90.0
    if value <= 0.5:
        return 72.0
    if value <= 1.0:
        return 58.0
    if value <= 2.0:
        return 42.0
    return 25.0


def _score_growth(value: float | None) -> float | None:
    if value is None:
        return None
    if value >= 0.25:
        return 95.0
    if value >= 0.15:
        return 80.0
    if value >= 0.05:
        return 65.0
    if value >= 0:
        return 50.0
    return 20.0


def _score_margin(value: float | None) -> float | None:
    if value is None:
        return None
    if value >= 0.25:
        return 90.0
    if value >= 0.15:
        return 75.0
    if value >= 0.08:
        return 60.0
    if value >= 0:
        return 45.0
    return 15.0


def _score_roe(value: float | None) -> float | None:
    if value is None:
        return None
    if value >= 0.25:
        return 90.0
    if value >= 0.15:
        return 75.0
    if value >= 0.08:
        return 60.0
    if value >= 0:
        return 40.0
    return 15.0


def _score_cash_flow(value: float | None) -> float | None:
    if value is None:
        return None
    return 70.0 if value > 0 else 20.0


def _score_forward_pe(value: float | None) -> float | None:
    if value is None or value <= 0:
        return None
    if value <= 15:
        return 80.0
    if value <= 25:
        return 70.0
    if value <= 40:
        return 55.0
    if value <= 60:
        return 35.0
    return 20.0


def _score_peg(value: float | None) -> float | None:
    if value is None or value <= 0:
        return None
    if value <= 1:
        return 85.0
    if value <= 2:
        return 70.0
    if value <= 3:
        return 50.0
    return 25.0


def _score_debt(value: float | None) -> float | None:
    if value is None:
        return None
    if value <= 50:
        return 80.0
    if value <= 150:
        return 60.0
    if value <= 300:
        return 40.0
    return 20.0


def _average_available(
    values: list[float | None] | list[tuple[float | None, float]],
    weighted: bool = False,
) -> float | None:
    if weighted:
        total = 0.0
        weight_sum = 0.0
        for value, weight in values:  # type: ignore[misc]
            if value is not None:
                total += float(value) * float(weight)
                weight_sum += float(weight)
        return total / weight_sum if weight_sum else None

    available = [float(value) for value in values if value is not None]  # type: ignore[arg-type]
    if not available:
        return None
    return float(np.mean(available))


def _quality_from_score(score: float) -> tuple[str, str]:
    if score >= 75:
        return "strong", "强"
    if score >= 60:
        return "good", "良好"
    if score >= 45:
        return "neutral", "中性"
    return "weak", "弱"


def _build_note(
    metrics: dict[str, float | None],
    depth: dict[str, float | None],
    score: float,
    quality: str,
    quality_zh: str,
    coverage: float,
) -> tuple[str, str]:
    pieces = [
        f"quality={quality}",
        f"coverage={coverage:.0%}",
    ]
    pieces_zh = [
        f"质量={quality_zh}",
        f"字段覆盖率={coverage:.0%}",
    ]
    if metrics["revenue_growth"] is not None:
        pieces.append(f"revenue growth={metrics['revenue_growth']:.2%}")
        pieces_zh.append(f"营收增长={metrics['revenue_growth']:.2%}")
    if metrics["profit_margin"] is not None:
        pieces.append(f"profit margin={metrics['profit_margin']:.2%}")
        pieces_zh.append(f"净利率={metrics['profit_margin']:.2%}")
    if metrics["return_on_equity"] is not None:
        pieces.append(f"ROE={metrics['return_on_equity']:.2%}")
        pieces_zh.append(f"ROE={metrics['return_on_equity']:.2%}")
    if depth["cash_conversion"] is not None:
        pieces.append(f"cash conversion={depth['cash_conversion']:.2f}")
        pieces_zh.append(f"现金转化率={depth['cash_conversion']:.2f}")
    if depth["net_debt_to_equity"] is not None:
        pieces.append(f"net debt/equity={depth['net_debt_to_equity']:.2f}")
        pieces_zh.append(f"净负债权益比={depth['net_debt_to_equity']:.2f}")
    if metrics["forward_pe"] is not None:
        pieces.append(f"forward PE={metrics['forward_pe']:.2f}")
        pieces_zh.append(f"forward PE={metrics['forward_pe']:.2f}")
    return (
        f"Fundamental score is {score:.1f}; " + ", ".join(pieces) + ".",
        f"基本面分数为{score:.1f}；" + "，".join(pieces_zh) + "。",
    )


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        result = float(value)
    except Exception:
        return None
    return result if np.isfinite(result) else None


def _finite(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value) if np.isfinite(value) else None
