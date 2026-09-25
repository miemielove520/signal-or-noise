from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .json_io import dataframe_records


@dataclass(frozen=True)
class DataLayerReadiness:
    layer: str
    layer_zh: str
    source: str
    status: str
    status_zh: str
    freshness: str
    freshness_zh: str
    coverage: float
    warning_count: int
    warnings: tuple[str, ...]
    usable_for_scoring: bool
    note: str
    note_zh: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DataReadinessReport:
    overall_status: str
    overall_status_zh: str
    overall_score: float
    repair_priority: str
    repair_priority_zh: str
    primary_blockers: tuple[str, ...]
    primary_blockers_zh: tuple[str, ...]
    layers: tuple[DataLayerReadiness, ...]
    source_validation: dict[str, object]
    sec_status: str
    sec_status_zh: str
    fred_status: str
    fred_status_zh: str
    fundamentals_source: str = "unknown"
    fundamentals_source_zh: str = "未知"

    def to_dict(self) -> dict[str, object]:
        return {
            "overall_status": self.overall_status,
            "overall_status_zh": self.overall_status_zh,
            "overall_score": self.overall_score,
            "repair_priority": self.repair_priority,
            "repair_priority_zh": self.repair_priority_zh,
            "primary_blockers": list(self.primary_blockers),
            "primary_blockers_zh": list(self.primary_blockers_zh),
            "layers": [layer.to_dict() for layer in self.layers],
            "source_validation": self.source_validation,
            "sec_status": self.sec_status,
            "sec_status_zh": self.sec_status_zh,
            "fred_status": self.fred_status,
            "fred_status_zh": self.fred_status_zh,
            "fundamentals_source": self.fundamentals_source,
            "fundamentals_source_zh": self.fundamentals_source_zh,
        }


def build_data_readiness_report(
    ticker: str,
    prices: pd.DataFrame,
    price_provider: str,
    price_attempts: tuple[str, ...],
    price_warnings: tuple[str, ...],
    price_missing_tickers: tuple[str, ...],
    price_source_validation: dict[str, object] | None,
    benchmark_prices: dict[str, pd.DataFrame],
    benchmark_sources: dict[str, str],
    benchmark_warnings: dict[str, tuple[str, ...]],
    sector_etf: str,
    sector_prices: pd.DataFrame,
    sector_source: str,
    sector_warnings: tuple[str, ...],
    snapshot: dict[str, object] | None,
    contexts: dict[str, object],
) -> DataReadinessReport:
    validation = price_source_validation or _single_source_validation()
    layers = [
        _price_layer(
            ticker=ticker,
            prices=prices,
            provider=price_provider,
            attempts=price_attempts,
            warnings=price_warnings,
            missing_tickers=price_missing_tickers,
            validation=validation,
        ),
        _benchmark_layer(benchmark_prices, benchmark_sources, benchmark_warnings),
        _sector_layer(sector_etf, sector_prices, sector_source, sector_warnings),
        _snapshot_layer(snapshot),
        _context_layer("fundamental", "基本面", contexts.get("fundamental")),
        _context_layer("valuation", "估值", contexts.get("valuation")),
        _context_layer("news_sentiment", "新闻情绪", contexts.get("sentiment")),
        _context_layer("event", "事件", contexts.get("event")),
        _context_layer("analyst", "分析师", contexts.get("analyst")),
    ]
    score = _overall_score(layers)
    status, status_zh = _overall_status(score, layers)
    priority, priority_zh = _repair_priority(score, layers)
    blockers = tuple(layer.layer for layer in layers if not layer.usable_for_scoring)
    blockers_zh = tuple(layer.layer_zh for layer in layers if not layer.usable_for_scoring)
    sec_status, sec_status_zh = _sec_status(snapshot)
    fundamentals_source, fundamentals_source_zh = _fundamentals_source_status(
        contexts.get("fundamental")
    )
    return DataReadinessReport(
        overall_status=status,
        overall_status_zh=status_zh,
        overall_score=score,
        repair_priority=priority,
        repair_priority_zh=priority_zh,
        primary_blockers=blockers,
        primary_blockers_zh=blockers_zh,
        layers=tuple(layers),
        source_validation=validation,
        sec_status=sec_status,
        sec_status_zh=sec_status_zh,
        fred_status="planned_not_enabled",
        fred_status_zh="已规划但尚未启用",
        fundamentals_source=fundamentals_source,
        fundamentals_source_zh=fundamentals_source_zh,
    )


def data_readiness_frame(report: DataReadinessReport) -> pd.DataFrame:
    rows = []
    for layer in report.layers:
        row = layer.to_dict()
        row["warnings"] = "; ".join(layer.warnings)
        row["overall_status"] = report.overall_status
        row["overall_status_zh"] = report.overall_status_zh
        row["overall_score"] = report.overall_score
        row["repair_priority"] = report.repair_priority
        row["repair_priority_zh"] = report.repair_priority_zh
        rows.append(row)
    return pd.DataFrame(rows)


def render_data_readiness_report(report: DataReadinessReport) -> str:
    lines = [
        "# Data Source Readiness / 数据源准备度",
        "",
        f"- Overall status / 总体状态: `{report.overall_status}` / `{report.overall_status_zh}`",
        f"- Overall score / 总体分数: `{report.overall_score:.2f}`",
        f"- Repair priority / 修复优先级: `{report.repair_priority}` / `{report.repair_priority_zh}`",
        f"- SEC status / SEC状态: `{report.sec_status}` / `{report.sec_status_zh}`",
        f"- Fundamentals source / 基本面来源: `{report.fundamentals_source}` / `{report.fundamentals_source_zh}`",
        f"- FRED status / FRED状态: `{report.fred_status}` / `{report.fred_status_zh}`",
        "",
        "## Layer Status / 数据层状态",
        "",
    ]
    lines.extend(_markdown_table(data_readiness_frame(report)[_display_columns()]))
    lines.extend(["", "## Price Source Validation / 价格源交叉验证", ""])
    lines.extend(_source_validation_lines(report.source_validation))
    lines.extend(
        [
            "",
            "How to read / 怎么看:",
            "",
            "- `validated`: at least one additional provider broadly agrees with the selected price source.",
            "- `validated`: 至少一个额外数据源与当前价格源大体一致。",
            "- `single_source_available`: only one provider is available, so the result can be used but is less independently verified.",
            "- `single_source_available`: 当前只有单一数据源可用，可以使用，但缺少独立验证。",
            "- `conflict_warning`: providers disagree materially; treat scoring confidence as lower.",
            "- `conflict_warning`: 多个数据源差异较大，评分置信度应降低。",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def scan_data_readiness_summary(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "ticker",
        "data_readiness_level",
        "data_readiness_level_zh",
        "data_readiness_score",
        "data_readiness_repair_priority",
        "data_readiness_primary_blockers_zh",
    ]
    if summary.empty:
        return pd.DataFrame(columns=columns)
    frame = summary.copy()
    for column in columns:
        if column not in frame.columns:
            frame[column] = "" if column != "data_readiness_score" else np.nan
    return frame[columns].sort_values(
        ["data_readiness_score", "ticker"],
        ascending=[True, True],
        na_position="last",
    ).reset_index(drop=True)


def data_readiness_payload(report: DataReadinessReport) -> dict[str, object]:
    payload = report.to_dict()
    payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["layer_records"] = dataframe_records(data_readiness_frame(report))
    return payload


def _price_layer(
    ticker: str,
    prices: pd.DataFrame,
    provider: str,
    attempts: tuple[str, ...],
    warnings: tuple[str, ...],
    missing_tickers: tuple[str, ...],
    validation: dict[str, object],
) -> DataLayerReadiness:
    row_count = len(prices)
    coverage = 1.0 if row_count >= 252 else max(0.0, min(row_count / 252.0, 1.0))
    freshness, freshness_zh = _freshness(prices)
    validation_status = str(validation.get("status") or "single_source_available")
    all_warnings = list(warnings) + list(validation.get("warnings") or [])
    if missing_tickers:
        all_warnings.append(f"missing tickers: {', '.join(missing_tickers)}")
    if validation_status == "conflict_warning":
        status, status_zh = "conflict_warning", "价格源冲突"
        usable = False
    elif prices.empty or ticker.upper() in missing_tickers:
        status, status_zh = "missing", "缺失"
        usable = False
    elif validation_status == "validated":
        status, status_zh = "validated", "已验证"
        usable = True
    else:
        status, status_zh = "single_source_available", "单一来源可用"
        usable = True
    note = f"Provider={provider}; attempts={', '.join(attempts)}; rows={row_count}."
    note_zh = f"价格源={provider}；尝试顺序={', '.join(attempts)}；行数={row_count}。"
    return DataLayerReadiness(
        layer="price",
        layer_zh="价格",
        source=provider,
        status=status,
        status_zh=status_zh,
        freshness=freshness,
        freshness_zh=freshness_zh,
        coverage=round(float(coverage), 2),
        warning_count=len(all_warnings),
        warnings=tuple(all_warnings),
        usable_for_scoring=usable,
        note=note,
        note_zh=note_zh,
    )


def _benchmark_layer(
    benchmark_prices: dict[str, pd.DataFrame],
    benchmark_sources: dict[str, str],
    benchmark_warnings: dict[str, tuple[str, ...]],
) -> DataLayerReadiness:
    expected = ("SPY", "QQQ", "^VIX")
    available = [symbol for symbol in expected if not benchmark_prices.get(symbol, pd.DataFrame()).empty]
    warnings = tuple(
        warning
        for symbol in expected
        for warning in benchmark_warnings.get(symbol, ())
    )
    coverage = len(available) / len(expected)
    if coverage == 1.0:
        status, status_zh, usable = "available", "可用", True
    elif coverage > 0:
        status, status_zh, usable = "partial", "部分可用", True
    else:
        status, status_zh, usable = "missing", "缺失", False
    source = ", ".join(f"{symbol}:{benchmark_sources.get(symbol, 'unavailable')}" for symbol in expected)
    return DataLayerReadiness(
        layer="benchmark",
        layer_zh="大盘基准",
        source=source,
        status=status,
        status_zh=status_zh,
        freshness="mixed",
        freshness_zh="混合",
        coverage=round(float(coverage), 2),
        warning_count=len(warnings),
        warnings=warnings,
        usable_for_scoring=usable,
        note=f"Available benchmarks: {', '.join(available) or 'none'}.",
        note_zh=f"可用基准：{', '.join(available) or '无'}。",
    )


def _sector_layer(
    sector_etf: str,
    sector_prices: pd.DataFrame,
    sector_source: str,
    sector_warnings: tuple[str, ...],
) -> DataLayerReadiness:
    if not sector_etf:
        return _unavailable_layer("sector", "板块", "unavailable", "sector ETF unavailable")
    if sector_prices.empty:
        return _unavailable_layer("sector", "板块", sector_source, "sector ETF price data unavailable")
    freshness, freshness_zh = _freshness(sector_prices)
    return DataLayerReadiness(
        layer="sector",
        layer_zh="板块",
        source=f"{sector_etf}:{sector_source}",
        status="available",
        status_zh="可用",
        freshness=freshness,
        freshness_zh=freshness_zh,
        coverage=1.0,
        warning_count=len(sector_warnings),
        warnings=tuple(sector_warnings),
        usable_for_scoring=True,
        note=f"Sector ETF {sector_etf} is available.",
        note_zh=f"板块ETF {sector_etf} 可用。",
    )


def _snapshot_layer(snapshot: dict[str, object] | None) -> DataLayerReadiness:
    if not snapshot:
        return _unavailable_layer("snapshot", "公司快照", "unavailable", "snapshot unavailable")
    useful_fields = [
        "company_name",
        "sector",
        "industry",
        "forward_pe",
        "peg_ratio",
        "target_mean_price",
        "news_titles",
    ]
    coverage = sum(_has_value(snapshot.get(field)) for field in useful_fields) / len(useful_fields)
    status = "available" if coverage >= 0.5 else "partial"
    status_zh = "可用" if status == "available" else "部分可用"
    return DataLayerReadiness(
        layer="snapshot",
        layer_zh="公司快照",
        source=str(snapshot.get("data_source") or "current_snapshot"),
        status=status,
        status_zh=status_zh,
        freshness="current_snapshot",
        freshness_zh="当前快照",
        coverage=round(float(coverage), 2),
        warning_count=0,
        warnings=(),
        usable_for_scoring=coverage > 0,
        note="Current snapshot fields are used only for current analysis, not historical backtests.",
        note_zh="当前快照字段只用于当前分析，不用于历史回测。",
    )


def _context_layer(layer: str, layer_zh: str, context: object | None) -> DataLayerReadiness:
    if context is None:
        return _unavailable_layer(layer, layer_zh, "unavailable", f"{layer} context unavailable")
    source = str(getattr(context, "source", "unknown"))
    coverage = getattr(context, "data_coverage", None)
    if coverage is None:
        if layer == "news_sentiment":
            titles_used = int(getattr(context, "sentiment_titles_used", 0) or 0)
            coverage = 1.0 if titles_used > 0 else 0.0
        elif layer == "event":
            coverage = 0.0 if getattr(context, "event_risk_level", "unknown") == "unknown" else 1.0
        else:
            coverage = 0.5
    warning = _context_warning(context, layer)
    coverage_float = float(coverage)
    if coverage_float <= 0:
        status, status_zh, usable = "fallback", "中性回退", False
    elif coverage_float < 0.5:
        status, status_zh, usable = "partial", "部分可用", True
    else:
        status, status_zh, usable = "available", "可用", True
    freshness, freshness_zh = _context_freshness(layer, source)
    return DataLayerReadiness(
        layer=layer,
        layer_zh=layer_zh,
        source=source,
        status=status,
        status_zh=status_zh,
        freshness=freshness,
        freshness_zh=freshness_zh,
        coverage=round(coverage_float, 2),
        warning_count=1 if warning else 0,
        warnings=(warning,) if warning else (),
        usable_for_scoring=usable,
        note=f"{layer} source={source}; coverage={coverage_float:.0%}.",
        note_zh=f"{layer_zh}来源={source}；覆盖率={coverage_float:.0%}。",
    )


def _context_freshness(layer: str, source: str) -> tuple[str, str]:
    if layer == "fundamental" and "sec_pit" in source:
        return "point_in_time_filing_date", "SEC申报日时点数据"
    if layer == "fundamental" and "yfinance_restated" in source:
        return "restated_current_snapshot", "当前重述快照"
    if any(token in source for token in ["snapshot", "yfinance", "sec_companyfacts"]):
        return "current_snapshot", "当前快照"
    return "unknown", "未知"


def _fundamentals_source_status(context: object | None) -> tuple[str, str]:
    source = str(getattr(context, "source", "") or "").strip()
    if not source or source == "unavailable":
        return "unavailable", "不可用"
    if "sec_pit" in source:
        return "sec_pit", "SEC申报日时点数据"
    if "sec_companyfacts_current_snapshot" in source or "sec_companyfacts" in source:
        return "sec_companyfacts_current_snapshot", "SEC当前companyfacts快照"
    if "yfinance_restated" in source or "yfinance" in source or "current_snapshot" in source:
        return "yfinance_restated", "yfinance当前重述快照"
    return source, "未知来源"


def _unavailable_layer(layer: str, layer_zh: str, source: str, warning: str) -> DataLayerReadiness:
    return DataLayerReadiness(
        layer=layer,
        layer_zh=layer_zh,
        source=source,
        status="missing",
        status_zh="缺失",
        freshness="unavailable",
        freshness_zh="不可用",
        coverage=0.0,
        warning_count=1,
        warnings=(warning,),
        usable_for_scoring=False,
        note=warning,
        note_zh=f"{layer_zh}数据不可用。",
    )


def _overall_score(layers: list[DataLayerReadiness]) -> float:
    weights = {
        "price": 0.25,
        "benchmark": 0.15,
        "sector": 0.10,
        "snapshot": 0.10,
        "fundamental": 0.10,
        "valuation": 0.08,
        "news_sentiment": 0.07,
        "event": 0.08,
        "analyst": 0.07,
    }
    score = 0.0
    weight_sum = 0.0
    for layer in layers:
        weight = weights.get(layer.layer, 0.05)
        layer_score = layer.coverage * 100.0
        if not layer.usable_for_scoring:
            layer_score = min(layer_score, 35.0)
        if layer.status == "conflict_warning":
            layer_score = min(layer_score, 30.0)
        score += layer_score * weight
        weight_sum += weight
    return round(float(score / weight_sum if weight_sum else 0.0), 2)


def _overall_status(score: float, layers: list[DataLayerReadiness]) -> tuple[str, str]:
    if any(layer.status == "conflict_warning" for layer in layers):
        return "conflict_warning", "存在数据冲突"
    if score >= 80:
        return "ready", "准备充分"
    if score >= 60:
        return "usable_with_warnings", "可用但有警告"
    return "insufficient", "数据不足"


def _repair_priority(score: float, layers: list[DataLayerReadiness]) -> tuple[str, str]:
    if score < 60 or any(layer.layer == "price" and not layer.usable_for_scoring for layer in layers):
        return "high", "高"
    if score < 80 or any(layer.warning_count > 0 for layer in layers):
        return "medium", "中"
    return "low", "低"


def _freshness(frame: pd.DataFrame) -> tuple[str, str]:
    if frame.empty or "date" not in frame.columns:
        return "unavailable", "不可用"
    latest = pd.to_datetime(frame["date"], errors="coerce").max()
    if pd.isna(latest):
        return "unavailable", "不可用"
    age_days = (pd.Timestamp.now().normalize() - latest.normalize()).days
    if age_days <= 5:
        return "fresh", "新鲜"
    if age_days <= 14:
        return "stale_warning", "略旧"
    return "stale", "过旧"


def _context_warning(context: object, layer: str) -> str:
    warning_fields = {
        "fundamental": "fundamental_warning",
        "valuation": "valuation_warning",
        "news_sentiment": "sentiment_warning",
        "event": "event_risk_warning",
        "analyst": "analyst_warning",
    }
    return str(getattr(context, warning_fields.get(layer, "warning"), "") or "")


def _has_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip() != ""


def _single_source_validation() -> dict[str, object]:
    return {
        "status": "single_source_available",
        "status_zh": "只有单一价格源可用",
        "comparison_provider_count": 0,
        "max_close_diff_pct": None,
        "max_volume_diff_pct": None,
        "max_missing_date_count": 0,
        "comparisons": [],
        "warnings": [],
    }


def _sec_status(snapshot: dict[str, object] | None) -> tuple[str, str]:
    if snapshot and _has_value(snapshot.get("sec_cik")):
        coverage = snapshot.get("sec_fundamental_coverage")
        try:
            coverage_float = float(coverage)
        except (TypeError, ValueError):
            coverage_float = 0.0
        if coverage_float >= 0.5:
            return "available", "可用"
        return "partial", "部分可用"
    return "planned_not_enabled", "已规划但本次未启用或不可用"


def _source_validation_lines(validation: dict[str, object]) -> list[str]:
    lines = [
        f"- Status / 状态: `{validation.get('status')}` / `{validation.get('status_zh')}`",
        f"- Comparison providers / 对比源数量: `{validation.get('comparison_provider_count', 0)}`",
        f"- Max close difference / 最大收盘价差异: `{_format_optional_percent(validation.get('max_close_diff_pct'))}`",
        f"- Max volume difference / 最大成交量差异: `{_format_optional_percent(validation.get('max_volume_diff_pct'))}`",
        f"- Max missing date count / 最大缺失日期数: `{validation.get('max_missing_date_count', 0)}`",
        "",
    ]
    comparisons = validation.get("comparisons") or []
    if comparisons:
        lines.extend(_markdown_table(pd.DataFrame(comparisons)))
    else:
        lines.append("No comparison rows. / 暂无对比行。")
    warnings = validation.get("warnings") or []
    if warnings:
        lines.extend(["", "Warnings / 警告:"])
        lines.extend(f"- {warning}" for warning in warnings)
    return lines


def _display_columns() -> list[str]:
    return [
        "layer",
        "layer_zh",
        "source",
        "status",
        "status_zh",
        "freshness",
        "coverage",
        "warning_count",
        "usable_for_scoring",
        "note_zh",
    ]


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["No rows. / 暂无。"]
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(_format_value(value) for value in row) + " |")
    return lines


def _format_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.2%}" if -1.0 <= value <= 1.0 else f"{value:.2f}"
    return str(value)


def _format_optional_percent(value: object) -> str:
    try:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "N/A"
