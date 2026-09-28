"""JSON helpers that make pandas/numpy values (NaN, timestamps, numpy scalars) serialisable."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def write_json(path: str | Path, payload: Any) -> None:
    """Write a payload as UTF-8 JSON after converting pandas/numpy values."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(json_safe(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """A DataFrame as a list of JSON-safe row dicts."""
    return [json_safe(row) for row in frame.to_dict(orient="records")]


def json_safe(value: Any) -> Any:
    """Recursively convert NaN, timestamps, paths and numpy scalars into JSON-safe values."""
    if isinstance(value, dict):
        return {str(key): json_safe(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(nested) for nested in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return dataframe_records(value)
    if isinstance(value, pd.Series):
        return json_safe(value.to_dict())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        float_value = float(value)
        return None if np.isnan(float_value) or np.isinf(float_value) else float_value
    if isinstance(value, np.ndarray):
        return [json_safe(nested) for nested in value.tolist()]
    if isinstance(value, float):
        return None if np.isnan(value) or np.isinf(value) else value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value
