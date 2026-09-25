"""Small coercion and formatting helpers shared by the analysis modules."""

from __future__ import annotations

import numpy as np
import pandas as pd



def _clamp_score(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def _row_float(row: pd.Series, key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    return float(value) if _is_finite(value) else default


def _safe_float_like(value: object) -> float:
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _safe_int_like(value: object) -> int:
    try:
        if pd.isna(value):
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def _append_reason(value: object, reason: str, delimiter: str = "; ") -> str:
    text = str(value or "").strip()
    if not text or text in {"nan", "None"}:
        return reason
    parts = [part.strip() for part in text.split(delimiter) if part.strip()]
    if reason in parts:
        return text
    return text + delimiter + reason


def _context_float(context: object | None, name: str, default: float) -> float:
    if context is None:
        return default
    if isinstance(context, dict):
        value = context.get(name, default)
    else:
        value = getattr(context, name, default)
    return float(value) if _is_finite(value) else default


def _context_bool(context: object | None, name: str, default: bool) -> bool:
    if context is None:
        return default
    if isinstance(context, dict):
        value = context.get(name, default)
    else:
        value = getattr(context, name, default)
    return _coerce_bool(value, default)


def _context_text(context: object | None, name: str, default: str) -> str:
    if context is None:
        return default
    if isinstance(context, dict):
        value = context.get(name, default)
    else:
        value = getattr(context, name, default)
    return str(value) if value is not None else default


def _coerce_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n", ""}:
            return False
        return default
    if isinstance(value, (int, float, np.integer, np.floating)):
        if not _is_finite(value):
            return default
        return bool(value)
    return bool(value)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not _is_finite(numerator) or not _is_finite(denominator) or denominator == 0:
        return np.nan
    return float(numerator / denominator)


def _last_valid(series: pd.Series) -> float:
    valid = series.replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return np.nan
    return float(valid.iloc[-1])


def _finite_or(value: float, fallback: float) -> float:
    return float(value) if _is_finite(value) else float(fallback)


def _is_finite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["No rows."]
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False):
        values = [_format_number(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _format_number(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return _format_date(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _format_signed_number(value: object) -> str:
    if pd.isna(value):
        return ""
    try:
        return f"{float(value):+.4f}"
    except (TypeError, ValueError):
        return str(value)


def _format_percent(value: object) -> str:
    if pd.isna(value):
        return "N/A"
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return str(value)


def _format_date(value: object) -> str:
    return pd.Timestamp(value).date().isoformat()
