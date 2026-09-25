from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd


def fetch_yfinance_snapshot(ticker: str) -> dict[str, Any]:
    ticker = ticker.upper().strip()
    if not ticker:
        raise ValueError("ticker cannot be empty.")

    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install the optional 'data' dependency to use yfinance.") from exc

    stock = yf.Ticker(ticker)
    info = _safe_dict_call(lambda: stock.info)
    targets = _safe_dict_call(lambda: stock.analyst_price_targets)
    news_items = _safe_list_call(lambda: stock.news)
    news_titles = extract_news_titles(news_items)

    current_price = _safe_float(
        info.get("currentPrice")
        or info.get("regularMarketPrice")
        or info.get("previousClose")
    )
    target_mean = _first_float(
        info,
        targets,
        ["targetMeanPrice", "mean", "average"],
    )
    analyst_upside = (
        target_mean / current_price - 1.0
        if target_mean is not None and current_price and current_price > 0
        else None
    )

    snapshot = {
        "ticker": ticker,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": "yfinance_current_snapshot",
        "historical_backtest_safe": False,
        "usage_note": (
            "Current snapshot only. Do not merge these fields into historical "
            "backtests unless you have point-in-time history for the same fields."
        ),
        "company_name": info.get("longName") or info.get("shortName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "current_price": current_price,
        "market_cap": _safe_float(info.get("marketCap")),
        "trailing_pe": _safe_float(info.get("trailingPE")),
        "forward_pe": _safe_float(info.get("forwardPE")),
        "peg_ratio": _safe_float(info.get("pegRatio")),
        "revenue_growth": _safe_float(info.get("revenueGrowth")),
        "earnings_growth": _safe_float(info.get("earningsGrowth")),
        "gross_margin": _safe_float(info.get("grossMargins")),
        "operating_margin": _safe_float(info.get("operatingMargins")),
        "profit_margin": _safe_float(info.get("profitMargins")),
        "return_on_equity": _safe_float(info.get("returnOnEquity")),
        "debt_to_equity": _safe_float(info.get("debtToEquity")),
        "free_cash_flow": _safe_float(info.get("freeCashflow")),
        "total_cash": _safe_float(info.get("totalCash")),
        "total_debt": _safe_float(info.get("totalDebt")),
        "target_mean_price": target_mean,
        "target_high_price": _first_float(info, targets, ["targetHighPrice", "high"]),
        "target_low_price": _first_float(info, targets, ["targetLowPrice", "low"]),
        "analyst_upside": analyst_upside,
        "recommendation_mean": _safe_float(info.get("recommendationMean")),
        "recommendation_key": info.get("recommendationKey"),
        "number_of_analysts": _safe_float(info.get("numberOfAnalystOpinions")),
        "news_titles": news_titles,
    }
    return _json_safe(snapshot)


def extract_news_titles(news_items: list[Any]) -> list[str]:
    titles: list[str] = []

    def find_titles(value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                if key.lower() == "title" and isinstance(nested, str):
                    titles.append(nested)
                else:
                    find_titles(nested)
        elif isinstance(value, list):
            for nested in value:
                find_titles(nested)

    find_titles(news_items)
    unique_titles: list[str] = []
    for title in titles:
        cleaned = title.strip()
        if cleaned and cleaned not in unique_titles:
            unique_titles.append(cleaned)
    return unique_titles[:10]


def _safe_dict_call(callback) -> dict[str, Any]:
    try:
        value = callback()
    except Exception:
        return {}
    if isinstance(value, pd.DataFrame):
        return value.to_dict()
    return value if isinstance(value, dict) else {}


def _safe_list_call(callback) -> list[Any]:
    try:
        value = callback()
    except Exception:
        return []
    return value if isinstance(value, list) else []


def _first_float(
    info: dict[str, Any],
    targets: dict[str, Any],
    keys: list[str],
) -> float | None:
    for key in keys:
        value = _safe_float(info.get(key))
        if value is not None:
            return value
    for key in keys:
        value = _safe_float(targets.get(key))
        if value is not None:
            return value
    return None


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


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return [_json_safe(nested) for nested in value]
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
