"""Small coercion and formatting helpers shared by the walk-forward modules."""

from __future__ import annotations

import numpy as np
import pandas as pd



def _coerce_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", ""}:
        return False
    return default


def _split_reason_text(value: object) -> list[str]:
    if value is None:
        return []
    text = str(value).replace("；", ";")
    return [
        item.strip()
        for item in text.split(";")
        if item.strip()
        and item.strip().lower() not in {"none", "nan", "all strict quality gates passed"}
        and item.strip() not in {"无", "所有严格质量门槛通过"}
    ]


def _first_text(frame: pd.DataFrame, column: str, default: str) -> str:
    if column not in frame.columns:
        return default
    values = frame[column].dropna()
    if values.empty:
        return default
    return str(values.iloc[0])


def _safe_int(value: object) -> int:
    try:
        if pd.isna(value):
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def _safe_optional_float(value: object) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _optional_timestamp(value: object) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(timestamp):
        return None
    return timestamp.normalize()


def _format_optional_date(value: object) -> str:
    timestamp = _optional_timestamp(value)
    if timestamp is None:
        return ""
    return timestamp.date().isoformat()


def _numeric_mean(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if len(values) else np.nan


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _format_metric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    for column in display.columns:
        if column in {"probability_bucket", "probability_bucket_zh"}:
            continue
        if any(
            token in column
            for token in ["return", "win_rate", "drawdown", "probability", "calibration_error"]
        ):
            display[column] = display[column].map(_format_percent)
    return display


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
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _format_percent(value: object) -> str:
    if pd.isna(value):
        return ""
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return str(value)


def _is_finite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False
