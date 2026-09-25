"""Persist executed paper orders for the human-vs-model scoreboard."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .human_log import CORE_COLUMNS, DEFAULT_BENCHMARK


MODEL_TRADES_FILENAME = "model_trades.csv"


def load_model_trades(review_root: str | Path = "outputs/signal_review") -> pd.DataFrame:
    path = Path(review_root) / MODEL_TRADES_FILENAME
    if not path.exists():
        return pd.DataFrame(columns=list(CORE_COLUMNS))
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=list(CORE_COLUMNS))


def update_model_trades_from_paper_result(
    paper_result_path: str | Path,
    review_root: str | Path = "outputs/signal_review",
    benchmark: str = DEFAULT_BENCHMARK,
) -> pd.DataFrame:
    """Append actual state-updating paper orders to the model decision history."""
    review_dir = Path(review_root)
    existing = load_model_trades(review_dir)
    path = Path(paper_result_path)
    if not path.exists():
        return existing
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return existing
    if not bool(payload.get("state_updated")):
        return existing

    orders = payload.get("orders") or []
    if not isinstance(orders, list) or not orders:
        return existing
    signal_date = _date_text(payload.get("as_of_date"))
    pre_trade_value = _finite_float((payload.get("summary") or {}).get("pre_trade_value"))
    rows: list[dict[str, object]] = []
    for order in orders:
        if not isinstance(order, dict):
            continue
        ticker = str(order.get("ticker") or "").upper().strip()
        action = str(order.get("action") or "").upper().strip()
        price = _finite_float(order.get("price")) or _finite_float(order.get("close_price"))
        if not ticker or action not in {"BUY", "SELL"} or price is None or price <= 0:
            continue
        notional = _finite_float(order.get("notional"))
        target_weight = _finite_float(order.get("target_weight"))
        weight_pct = (
            abs(notional) / pre_trade_value * 100.0
            if notional is not None and pre_trade_value is not None and pre_trade_value > 0
            else abs(target_weight or 0.0) * 100.0
        )
        rows.append(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "signal_date": signal_date,
                "ticker": ticker,
                "actor": "model",
                "action": action,
                "weight_pct": round(weight_pct, 4),
                "signal_price": round(price, 6),
                "benchmark": str(benchmark).upper().strip(),
                "reason": "Executed paper rebalance order.",
                "report_path": str(path.with_name("paper_trade_report.md")),
                "quantity": _finite_float(order.get("quantity")),
                "notional": notional,
                "estimated_cost": _finite_float(order.get("estimated_cost")),
            }
        )
    if not rows:
        return existing

    combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    combined = combined.drop_duplicates(
        ["signal_date", "ticker", "action"], keep="last"
    ).sort_values(["signal_date", "ticker", "action"])
    review_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(review_dir / MODEL_TRADES_FILENAME, index=False)
    return combined.reset_index(drop=True)


def _date_text(value: object) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return pd.Timestamp.today().normalize().date().isoformat()
    return pd.Timestamp(parsed).normalize().date().isoformat()


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None
