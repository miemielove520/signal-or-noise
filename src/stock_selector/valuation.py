from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ValuationContext:
    ticker: str
    valuation_score: float
    valuation_label: str
    valuation_label_zh: str
    valuation_risk_level: str
    valuation_risk_level_zh: str
    valuation_block_new_entries: bool
    valuation_forward_pe: float | None
    valuation_trailing_pe: float | None
    valuation_peg_ratio: float | None
    valuation_free_cash_flow_yield: float | None
    valuation_market_cap: float | None
    valuation_growth_reference: float | None
    valuation_profit_margin: float | None
    valuation_note: str
    valuation_note_zh: str
    valuation_warning: str
    data_coverage: float
    source: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_valuation_context(
    ticker: str,
    snapshot: dict[str, Any] | None,
) -> ValuationContext:
    ticker = ticker.upper().strip()
    if not snapshot:
        return _unknown_context(ticker, "snapshot unavailable")

    forward_pe = _safe_float(snapshot.get("forward_pe"))
    trailing_pe = _safe_float(snapshot.get("trailing_pe"))
    peg_ratio = _safe_float(snapshot.get("peg_ratio"))
    market_cap = _safe_float(snapshot.get("market_cap"))
    free_cash_flow = _safe_float(snapshot.get("free_cash_flow"))
    free_cash_flow_yield = _safe_float(snapshot.get("free_cash_flow_yield"))
    if free_cash_flow_yield is None and market_cap is not None and market_cap > 0 and free_cash_flow is not None:
        free_cash_flow_yield = free_cash_flow / market_cap
    revenue_growth = _safe_float(snapshot.get("revenue_growth"))
    earnings_growth = _safe_float(snapshot.get("earnings_growth"))
    profit_margin = _safe_float(snapshot.get("profit_margin"))
    growth_reference = _growth_reference(revenue_growth, earnings_growth)

    fields = [
        forward_pe,
        trailing_pe,
        peg_ratio,
        free_cash_flow_yield,
        growth_reference,
        profit_margin,
    ]
    coverage = sum(value is not None for value in fields) / len(fields)
    if coverage < 0.25:
        return _unknown_context(ticker, "too few valuation fields available")

    score = _weighted_average(
        [
            (_score_forward_pe(forward_pe, growth_reference), 0.45),
            (_score_trailing_pe(trailing_pe, growth_reference), 0.20),
            (_score_peg(peg_ratio), 0.25),
            (_score_free_cash_flow_yield(free_cash_flow_yield), 0.15),
            (_score_growth_support(growth_reference, profit_margin), 0.20),
        ]
    )
    if score is None:
        return _unknown_context(ticker, "valuation score could not be calculated")
    score = round(float(max(0.0, min(score, 100.0))), 2)
    label, label_zh = _label_from_score(score)
    risk_level, risk_level_zh = _risk_level(
        score,
        forward_pe,
        trailing_pe,
        peg_ratio,
        growth_reference,
    )
    block_new_entries = risk_level == "high"
    note, note_zh = _build_note(
        score=score,
        label=label,
        label_zh=label_zh,
        forward_pe=forward_pe,
        trailing_pe=trailing_pe,
        peg_ratio=peg_ratio,
        free_cash_flow_yield=free_cash_flow_yield,
        growth_reference=growth_reference,
        profit_margin=profit_margin,
        coverage=coverage,
    )

    return ValuationContext(
        ticker=ticker,
        valuation_score=score,
        valuation_label=label,
        valuation_label_zh=label_zh,
        valuation_risk_level=risk_level,
        valuation_risk_level_zh=risk_level_zh,
        valuation_block_new_entries=block_new_entries,
        valuation_forward_pe=forward_pe,
        valuation_trailing_pe=trailing_pe,
        valuation_peg_ratio=peg_ratio,
        valuation_free_cash_flow_yield=free_cash_flow_yield,
        valuation_market_cap=market_cap,
        valuation_growth_reference=growth_reference,
        valuation_profit_margin=profit_margin,
        valuation_note=note,
        valuation_note_zh=note_zh,
        valuation_warning="",
        data_coverage=round(coverage, 2),
        source=str(
            snapshot.get("valuation_data_source")
            or snapshot.get("data_source")
            or "yfinance_current_snapshot"
        ),
    )


def _unknown_context(ticker: str, warning: str) -> ValuationContext:
    return ValuationContext(
        ticker=ticker,
        valuation_score=50.0,
        valuation_label="unknown",
        valuation_label_zh="未知",
        valuation_risk_level="unknown",
        valuation_risk_level_zh="未知",
        valuation_block_new_entries=False,
        valuation_forward_pe=None,
        valuation_trailing_pe=None,
        valuation_peg_ratio=None,
        valuation_free_cash_flow_yield=None,
        valuation_market_cap=None,
        valuation_growth_reference=None,
        valuation_profit_margin=None,
        valuation_note="Valuation data is unavailable; neutral score 50 is used.",
        valuation_note_zh="估值数据不可用；使用中性分数50。",
        valuation_warning=warning,
        data_coverage=0.0,
        source="unavailable",
    )


def _growth_reference(revenue_growth: float | None, earnings_growth: float | None) -> float | None:
    values = [value for value in [revenue_growth, earnings_growth] if value is not None]
    if not values:
        return None
    return max(values)


def _score_forward_pe(value: float | None, growth_reference: float | None) -> float | None:
    if value is None or value <= 0:
        return None
    growth = growth_reference if growth_reference is not None else 0.0
    if value <= 15:
        return 90.0
    if value <= 25:
        return 75.0
    if value <= 40:
        return 65.0 if growth >= 0.15 else 45.0
    if value <= 60:
        return 50.0 if growth >= 0.25 else 25.0
    return 35.0 if growth >= 0.30 else 10.0


def _score_trailing_pe(value: float | None, growth_reference: float | None) -> float | None:
    if value is None or value <= 0:
        return None
    growth = growth_reference if growth_reference is not None else 0.0
    if value <= 18:
        return 80.0
    if value <= 30:
        return 68.0
    if value <= 45:
        return 58.0 if growth >= 0.12 else 40.0
    if value <= 70:
        return 45.0 if growth >= 0.20 else 25.0
    return 30.0 if growth >= 0.30 else 10.0


def _score_peg(value: float | None) -> float | None:
    if value is None or value <= 0:
        return None
    if value <= 1.0:
        return 90.0
    if value <= 1.8:
        return 75.0
    if value <= 2.5:
        return 55.0
    if value <= 3.5:
        return 35.0
    return 15.0


def _score_free_cash_flow_yield(value: float | None) -> float | None:
    if value is None:
        return None
    if value >= 0.06:
        return 90.0
    if value >= 0.035:
        return 75.0
    if value >= 0.015:
        return 55.0
    if value >= 0:
        return 35.0
    return 10.0


def _score_growth_support(
    growth_reference: float | None,
    profit_margin: float | None,
) -> float | None:
    scores: list[float] = []
    if growth_reference is not None:
        if growth_reference >= 0.25:
            scores.append(90.0)
        elif growth_reference >= 0.15:
            scores.append(75.0)
        elif growth_reference >= 0.05:
            scores.append(55.0)
        elif growth_reference >= 0:
            scores.append(40.0)
        else:
            scores.append(15.0)
    if profit_margin is not None:
        if profit_margin >= 0.25:
            scores.append(85.0)
        elif profit_margin >= 0.15:
            scores.append(70.0)
        elif profit_margin >= 0.05:
            scores.append(50.0)
        elif profit_margin >= 0:
            scores.append(35.0)
        else:
            scores.append(10.0)
    if not scores:
        return None
    return sum(scores) / len(scores)


def _risk_level(
    score: float,
    forward_pe: float | None,
    trailing_pe: float | None,
    peg_ratio: float | None,
    growth_reference: float | None,
) -> tuple[str, str]:
    growth = growth_reference if growth_reference is not None else 0.0
    if score <= 30 or (forward_pe is not None and forward_pe >= 60 and growth < 0.25):
        return "high", "高"
    if trailing_pe is not None and trailing_pe >= 80 and growth < 0.25:
        return "high", "高"
    if peg_ratio is not None and peg_ratio >= 4.0:
        return "high", "高"
    if score <= 45 or (forward_pe is not None and forward_pe >= 45 and growth < 0.15):
        return "medium", "中"
    if trailing_pe is not None and trailing_pe >= 55 and growth < 0.15:
        return "medium", "中"
    return "low", "低"


def _label_from_score(score: float) -> tuple[str, str]:
    if score >= 70:
        return "attractive", "有吸引力"
    if score <= 40:
        return "expensive", "偏贵"
    return "reasonable", "合理"


def _build_note(
    score: float,
    label: str,
    label_zh: str,
    forward_pe: float | None,
    trailing_pe: float | None,
    peg_ratio: float | None,
    free_cash_flow_yield: float | None,
    growth_reference: float | None,
    profit_margin: float | None,
    coverage: float,
) -> tuple[str, str]:
    return (
        f"Valuation score={score:.1f}; label={label}; forward PE={_format_number(forward_pe)}; "
        f"trailing PE={_format_number(trailing_pe)}; PEG={_format_number(peg_ratio)}; "
        f"FCF yield={_format_percent(free_cash_flow_yield)}; growth reference={_format_percent(growth_reference)}; "
        f"profit margin={_format_percent(profit_margin)}; field coverage={coverage:.0%}.",
        f"估值分数={score:.1f}；标签={label_zh}；forward PE={_format_number(forward_pe)}；"
        f"trailing PE={_format_number(trailing_pe)}；PEG={_format_number(peg_ratio)}；"
        f"自由现金流收益率={_format_percent(free_cash_flow_yield)}；增长参考={_format_percent(growth_reference)}；"
        f"净利率={_format_percent(profit_margin)}；字段覆盖率={coverage:.0%}。",
    )


def _weighted_average(parts: list[tuple[float | None, float]]) -> float | None:
    valid = [(value, weight) for value, weight in parts if value is not None]
    if not valid:
        return None
    total_weight = sum(weight for _, weight in valid)
    if total_weight <= 0:
        return None
    return sum(float(value) * weight for value, weight in valid) / total_weight


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.replace(",", "").replace("%", "").strip()
        if pd.isna(value):
            return None
        result = float(value)
    except Exception:
        return None
    return result if np.isfinite(result) else None


def _format_number(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f}"


def _format_percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2%}"
