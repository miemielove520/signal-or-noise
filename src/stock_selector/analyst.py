from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AnalystContext:
    ticker: str
    analyst_score: float
    analyst_label: str
    analyst_label_zh: str
    analyst_risk_level: str
    analyst_risk_level_zh: str
    analyst_block_new_entries: bool
    analyst_upside: float | None
    recommendation_mean: float | None
    recommendation_key: str | None
    number_of_analysts: float | None
    target_mean_price: float | None
    analyst_note: str
    analyst_note_zh: str
    analyst_warning: str
    data_coverage: float
    source: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_analyst_context(
    ticker: str,
    snapshot: dict[str, Any] | None,
) -> AnalystContext:
    ticker = ticker.upper().strip()
    if not snapshot:
        return _unknown_context(ticker, "snapshot unavailable")

    analyst_upside = _safe_float(snapshot.get("analyst_upside"))
    recommendation_mean = _safe_float(snapshot.get("recommendation_mean"))
    number_of_analysts = _safe_float(snapshot.get("number_of_analysts"))
    target_mean_price = _safe_float(snapshot.get("target_mean_price"))
    recommendation_key = snapshot.get("recommendation_key")
    recommendation_key = str(recommendation_key) if recommendation_key else None

    fields = [analyst_upside, recommendation_mean, number_of_analysts, target_mean_price]
    coverage = sum(value is not None for value in fields) / len(fields)
    if coverage < 0.25:
        return _unknown_context(ticker, "too few analyst fields available")

    score_parts = [
        (_score_upside(analyst_upside), 0.45),
        (_score_recommendation(recommendation_mean), 0.35),
        (_score_coverage(number_of_analysts), 0.20),
    ]
    score = _weighted_average(score_parts)
    if score is None:
        return _unknown_context(ticker, "analyst score could not be calculated")
    score = round(float(max(0.0, min(score, 100.0))), 2)
    label, label_zh = _label_from_score(score)
    risk_level, risk_level_zh = _risk_level(
        score=score,
        analyst_upside=analyst_upside,
        recommendation_mean=recommendation_mean,
    )
    block_new_entries = risk_level == "high"
    note, note_zh = _build_note(
        score=score,
        label=label,
        label_zh=label_zh,
        analyst_upside=analyst_upside,
        recommendation_mean=recommendation_mean,
        number_of_analysts=number_of_analysts,
        target_mean_price=target_mean_price,
        coverage=coverage,
    )

    return AnalystContext(
        ticker=ticker,
        analyst_score=score,
        analyst_label=label,
        analyst_label_zh=label_zh,
        analyst_risk_level=risk_level,
        analyst_risk_level_zh=risk_level_zh,
        analyst_block_new_entries=block_new_entries,
        analyst_upside=analyst_upside,
        recommendation_mean=recommendation_mean,
        recommendation_key=recommendation_key,
        number_of_analysts=number_of_analysts,
        target_mean_price=target_mean_price,
        analyst_note=note,
        analyst_note_zh=note_zh,
        analyst_warning="",
        data_coverage=round(coverage, 2),
        source=str(snapshot.get("data_source") or "yfinance_current_snapshot"),
    )


def _unknown_context(ticker: str, warning: str) -> AnalystContext:
    return AnalystContext(
        ticker=ticker,
        analyst_score=50.0,
        analyst_label="unknown",
        analyst_label_zh="未知",
        analyst_risk_level="unknown",
        analyst_risk_level_zh="未知",
        analyst_block_new_entries=False,
        analyst_upside=None,
        recommendation_mean=None,
        recommendation_key=None,
        number_of_analysts=None,
        target_mean_price=None,
        analyst_note="Analyst expectation data is unavailable; neutral score 50 is used.",
        analyst_note_zh="分析师预期数据不可用；使用中性分数50。",
        analyst_warning=warning,
        data_coverage=0.0,
        source="unavailable",
    )


def _score_upside(value: float | None) -> float | None:
    if value is None:
        return None
    if value >= 0.25:
        return 90.0
    if value >= 0.10:
        return 75.0
    if value >= 0:
        return 55.0
    if value >= -0.10:
        return 35.0
    return 15.0


def _score_recommendation(value: float | None) -> float | None:
    if value is None:
        return None
    if value <= 1.8:
        return 90.0
    if value <= 2.4:
        return 75.0
    if value <= 3.0:
        return 55.0
    if value <= 3.5:
        return 35.0
    return 15.0


def _score_coverage(value: float | None) -> float | None:
    if value is None:
        return None
    if value >= 25:
        return 90.0
    if value >= 12:
        return 75.0
    if value >= 5:
        return 55.0
    return 35.0


def _risk_level(
    score: float,
    analyst_upside: float | None,
    recommendation_mean: float | None,
) -> tuple[str, str]:
    if (
        score <= 30
        or (analyst_upside is not None and analyst_upside <= -0.10)
        or (recommendation_mean is not None and recommendation_mean >= 3.6)
    ):
        return "high", "高"
    if score <= 45:
        return "medium", "中"
    return "low", "低"


def _label_from_score(score: float) -> tuple[str, str]:
    if score >= 70:
        return "positive", "正面"
    if score <= 40:
        return "negative", "负面"
    return "neutral", "中性"


def _build_note(
    score: float,
    label: str,
    label_zh: str,
    analyst_upside: float | None,
    recommendation_mean: float | None,
    number_of_analysts: float | None,
    target_mean_price: float | None,
    coverage: float,
) -> tuple[str, str]:
    upside_text = _format_percent(analyst_upside)
    recommendation_text = _format_number(recommendation_mean)
    coverage_text = _format_number(number_of_analysts)
    target_text = _format_number(target_mean_price)
    return (
        f"Analyst score={score:.1f}; label={label}; upside={upside_text}; "
        f"recommendation mean={recommendation_text}; analysts={coverage_text}; "
        f"target mean price={target_text}; field coverage={coverage:.0%}.",
        f"分析师分数={score:.1f}；标签={label_zh}；目标价上行空间={upside_text}；"
        f"推荐均值={recommendation_text}；覆盖分析师={coverage_text}；"
        f"平均目标价={target_text}；字段覆盖率={coverage:.0%}。",
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
