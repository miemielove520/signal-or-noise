from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketContext:
    market_score: float
    market_status: str
    note: str
    note_zh: str
    warning: str
    spy_trend_score: float
    qqq_trend_score: float
    vix_risk_score: float


@dataclass(frozen=True)
class RelativeStrengthContext:
    score: float
    note: str
    note_zh: str
    warning: str
    vs_spy_return: float
    vs_qqq_return: float


@dataclass(frozen=True)
class SectorContext:
    sector: str
    industry: str
    sector_etf: str
    sector_score: float
    sector_status: str
    sector_note: str
    sector_note_zh: str
    sector_warning: str
    sector_trend_score: float
    sector_relative_strength: float


SECTOR_ETF_MAP = {
    "technology": "XLK",
    "communication services": "XLC",
    "consumer cyclical": "XLY",
    "consumer defensive": "XLP",
    "financial services": "XLF",
    "healthcare": "XLV",
    "industrials": "XLI",
    "energy": "XLE",
    "basic materials": "XLB",
    "real estate": "XLRE",
    "utilities": "XLU",
}

INDUSTRY_ETF_KEYWORDS = {
    "semiconductor": "SMH",
    "software": "IGV",
    "bank": "KBE",
    "biotechnology": "XBI",
    "oil": "XLE",
    "gas": "XLE",
}


def choose_sector_etf(sector: str | None, industry: str | None = None) -> str:
    sector_text = str(sector or "").lower().strip()
    industry_text = str(industry or "").lower().strip()
    for keyword, etf in INDUSTRY_ETF_KEYWORDS.items():
        if keyword in industry_text:
            return etf
    return SECTOR_ETF_MAP.get(sector_text, "")


def build_sector_context(
    target_prices: pd.DataFrame,
    sector_prices: pd.DataFrame | None,
    sector: str | None,
    industry: str | None,
    sector_etf: str | None,
    lookback_days: int = 60,
) -> SectorContext:
    sector_text = str(sector or "").strip()
    industry_text = str(industry or "").strip()
    etf = str(sector_etf or "").strip().upper()

    if not etf or sector_prices is None or sector_prices.empty:
        return SectorContext(
            sector=sector_text,
            industry=industry_text,
            sector_etf=etf,
            sector_score=50.0,
            sector_status="unknown",
            sector_note="Sector ETF data is unavailable; neutral sector score 50 is used.",
            sector_note_zh="板块ETF数据不可用，使用中性板块分数50。",
            sector_warning="sector data unavailable",
            sector_trend_score=50.0,
            sector_relative_strength=np.nan,
        )

    sector_trend_score, sector_trend_note, sector_warning = _trend_score(sector_prices, etf)
    target_return = _window_return(target_prices, lookback_days)
    sector_return = _window_return(sector_prices, lookback_days)
    if _is_finite(target_return) and _is_finite(sector_return):
        relative_strength = float(target_return - sector_return)
        relative_score = max(0.0, min(100.0, 50.0 + relative_strength * 250.0))
        warning = sector_warning
    else:
        relative_strength = np.nan
        relative_score = 50.0
        warning = "; ".join(filter(None, [sector_warning, "sector relative strength unavailable"]))

    sector_score = 0.65 * sector_trend_score + 0.35 * relative_score
    sector_score = round(float(max(0.0, min(sector_score, 100.0))), 2)
    if sector_score >= 65:
        status = "supportive"
        note = f"Sector context is supportive. {sector_trend_note}"
        note_zh = "板块环境偏支持。"
    elif sector_score <= 40:
        status = "weak"
        note = f"Sector context is weak. {sector_trend_note}"
        note_zh = "板块环境偏弱。"
    else:
        status = "neutral"
        note = f"Sector context is neutral or mixed. {sector_trend_note}"
        note_zh = "板块环境中性或分化。"

    if _is_finite(relative_strength):
        note = f"{note} Stock vs sector return={relative_strength:.2%}."
        if relative_strength > 0:
            note_zh = f"{note_zh} 个股强于板块{relative_strength:.2%}。"
        else:
            note_zh = f"{note_zh} 个股弱于板块{abs(relative_strength):.2%}。"
    if warning:
        note = f"{note} Warning: {warning}."
        note_zh = f"{note_zh} 部分板块数据缺失或不足。"

    return SectorContext(
        sector=sector_text,
        industry=industry_text,
        sector_etf=etf,
        sector_score=sector_score,
        sector_status=status,
        sector_note=note,
        sector_note_zh=note_zh,
        sector_warning=warning,
        sector_trend_score=round(float(sector_trend_score), 2),
        sector_relative_strength=relative_strength,
    )


def build_market_context(benchmark_prices: dict[str, pd.DataFrame]) -> MarketContext:
    spy_score, spy_note, spy_warning = _trend_score(benchmark_prices.get("SPY"), "SPY")
    qqq_score, qqq_note, qqq_warning = _trend_score(benchmark_prices.get("QQQ"), "QQQ")
    vix_prices = benchmark_prices.get("^VIX")
    if vix_prices is None:
        vix_prices = benchmark_prices.get("VIX")
    vix_score, vix_note, vix_warning = _vix_score(vix_prices)

    warnings = [warning for warning in [spy_warning, qqq_warning, vix_warning] if warning]
    market_score = 0.45 * spy_score + 0.35 * qqq_score + 0.20 * vix_score
    if market_score >= 65:
        status = "supportive"
        note = "Market background is supportive."
        note_zh = "大盘环境偏支持。"
    elif market_score <= 40:
        status = "weak"
        note = "Market background is weak; new entries should be delayed or reduced."
        note_zh = "大盘环境偏弱，新的入场应延后或降低仓位。"
    else:
        status = "neutral"
        note = "Market background is mixed or neutral."
        note_zh = "大盘环境中性或分化。"

    detail = "; ".join([spy_note, qqq_note, vix_note])
    if detail:
        note = f"{note} {detail}"
    if warnings:
        note = f"{note} Missing data fallback: {', '.join(warnings)}."
        note_zh = f"{note_zh} 部分大盘数据缺失，缺失项使用中性分数50。"

    return MarketContext(
        market_score=round(float(max(0.0, min(market_score, 100.0))), 2),
        market_status=status,
        note=note,
        note_zh=note_zh,
        warning="; ".join(warnings),
        spy_trend_score=round(spy_score, 2),
        qqq_trend_score=round(qqq_score, 2),
        vix_risk_score=round(vix_score, 2),
    )


def build_relative_strength_contexts(
    target_prices: pd.DataFrame,
    benchmark_prices: dict[str, pd.DataFrame],
    horizon_windows: dict[str, int],
) -> dict[str, RelativeStrengthContext]:
    return {
        horizon: build_relative_strength_context(
            target_prices=target_prices,
            spy_prices=benchmark_prices.get("SPY"),
            qqq_prices=benchmark_prices.get("QQQ"),
            lookback_days=lookback_days,
        )
        for horizon, lookback_days in horizon_windows.items()
    }


def build_relative_strength_context(
    target_prices: pd.DataFrame,
    spy_prices: pd.DataFrame | None,
    qqq_prices: pd.DataFrame | None,
    lookback_days: int,
) -> RelativeStrengthContext:
    target_return = _window_return(target_prices, lookback_days)
    spy_return = _window_return(spy_prices, lookback_days)
    qqq_return = _window_return(qqq_prices, lookback_days)
    warnings: list[str] = []
    excess_returns: list[float] = []

    if _is_finite(spy_return) and _is_finite(target_return):
        excess_returns.append(float(target_return - spy_return))
    else:
        warnings.append("SPY")
    if _is_finite(qqq_return) and _is_finite(target_return):
        excess_returns.append(float(target_return - qqq_return))
    else:
        warnings.append("QQQ")

    if not _is_finite(target_return) or not excess_returns:
        return RelativeStrengthContext(
            score=50.0,
            note="Relative strength data is incomplete; using neutral score 50.",
            note_zh="相对强弱数据不完整，使用中性分数50。",
            warning=", ".join(warnings or ["target"]),
            vs_spy_return=np.nan,
            vs_qqq_return=np.nan,
        )

    average_excess = float(np.mean(excess_returns))
    score = max(0.0, min(100.0, 50.0 + average_excess * 250.0))
    if score >= 65:
        note = "The stock is outperforming benchmark ETFs."
        note_zh = "该股票相对基准ETF表现更强。"
    elif score <= 40:
        note = "The stock is underperforming benchmark ETFs."
        note_zh = "该股票相对基准ETF表现偏弱。"
    else:
        note = "The stock is roughly in line with benchmark ETFs."
        note_zh = "该股票相对基准ETF表现接近中性。"
    if warnings:
        note = f"{note} Missing benchmark fallback: {', '.join(warnings)}."
        note_zh = f"{note_zh} 部分基准数据缺失。"

    return RelativeStrengthContext(
        score=round(float(score), 2),
        note=note,
        note_zh=note_zh,
        warning=", ".join(warnings),
        vs_spy_return=float(target_return - spy_return) if _is_finite(spy_return) else np.nan,
        vs_qqq_return=float(target_return - qqq_return) if _is_finite(qqq_return) else np.nan,
    )


def _trend_score(prices: pd.DataFrame | None, label: str) -> tuple[float, str, str]:
    if prices is None or prices.empty:
        return 50.0, f"{label} unavailable", label
    close = _close_series(prices)
    if len(close.dropna()) < 100:
        return 50.0, f"{label} has insufficient history", label
    latest = float(close.iloc[-1])
    ma_100 = float(close.rolling(100).mean().iloc[-1])
    ret_20 = _series_window_return(close, 20)
    trend_component = max(-25.0, min(25.0, (_safe_ratio(latest, ma_100) - 1.0) * 300.0))
    momentum_component = max(-15.0, min(15.0, ret_20 * 200.0))
    score = max(0.0, min(100.0, 50.0 + trend_component + momentum_component))
    note = f"{label} score={score:.1f}"
    return float(score), note, ""


def _vix_score(prices: pd.DataFrame | None) -> tuple[float, str, str]:
    if prices is None or prices.empty:
        return 50.0, "VIX unavailable", "VIX"
    close = _close_series(prices)
    if close.dropna().empty:
        return 50.0, "VIX unavailable", "VIX"
    latest = float(close.iloc[-1])
    if latest <= 15:
        score = 80.0
    elif latest <= 20:
        score = 70.0
    elif latest <= 25:
        score = 55.0
    elif latest <= 30:
        score = 40.0
    elif latest <= 40:
        score = 25.0
    else:
        score = 10.0
    return score, f"VIX={latest:.2f}, risk score={score:.1f}", ""


def _window_return(prices: pd.DataFrame | None, lookback_days: int) -> float:
    if prices is None or prices.empty:
        return np.nan
    close = _close_series(prices)
    return _series_window_return(close, lookback_days)


def _close_series(prices: pd.DataFrame) -> pd.Series:
    frame = prices.copy().sort_values("date") if "date" in prices.columns else prices.copy()
    column = "adj_close" if "adj_close" in frame.columns else "close"
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _series_window_return(close: pd.Series, lookback_days: int) -> float:
    close = close.dropna()
    if len(close) <= lookback_days:
        return np.nan
    return _safe_ratio(float(close.iloc[-1]), float(close.iloc[-lookback_days - 1])) - 1.0


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not _is_finite(numerator) or not _is_finite(denominator) or denominator == 0:
        return np.nan
    return float(numerator / denominator)


def _is_finite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False
