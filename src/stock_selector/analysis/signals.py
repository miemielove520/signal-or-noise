"""Per-horizon technical analysis: prices, scores, action classification and entry plans."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..screening_config import TradingRules

from ._common import (
    _context_bool,
    _context_float,
    _context_text,
    _finite_or,
    _format_number,
    _is_finite,
    _last_valid,
    _safe_ratio,
)
from .backtest import (
    _backtest_reliability,
    _backtest_trust_profile,
    _entry_backtest_summary,
)
from .horizons import (
    BREAKOUT_EXTENSION_MULTIPLE,
    HorizonSpec,
    _trading_rule_note,
    _trading_rule_note_zh,
)
from .indicators import (
    _average_true_range,
    _moving_average_slope,
    _pullback_entry,
    _stop_loss,
    _window_return,
)


def _ticker_prices(
    prices: pd.DataFrame,
    ticker: str,
    as_of_date: str | pd.Timestamp | None,
) -> pd.DataFrame:
    required = {"date", "ticker", "high", "low", "adj_close", "volume"}
    missing = required.difference(prices.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Missing price columns for ticker analysis: {missing_text}")

    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=False)
    frame["ticker"] = frame["ticker"].astype("string").str.upper().str.strip()
    frame = frame[frame["ticker"] == ticker].copy()
    if as_of_date is not None:
        frame = frame[frame["date"] <= pd.Timestamp(as_of_date)]
    if frame.empty:
        raise ValueError(f"No price data found for ticker {ticker}.")

    numeric_columns = ["high", "low", "adj_close", "volume"]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=["date", "ticker", "adj_close"])
    frame = frame.sort_values("date").reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No usable price data found for ticker {ticker}.")
    return frame


def _price_data_health_profile(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {
            "score": 0.0,
            "level": "poor",
            "level_zh": "差",
            "penalty": 25.0,
            "issue_count": 1,
            "max_calendar_gap_days": 0,
            "large_gap_count": 0,
            "zero_volume_days": 0,
            "missing_ohlcv_rows": 0,
            "extreme_return_count": 0,
            "high_low_inversion_count": 0,
            "adjustment_anomaly_count": 0,
            "note": "price frame is empty",
            "note_zh": "价格数据为空",
        }

    dates = pd.to_datetime(frame["date"], errors="coerce")
    date_gaps = dates.sort_values().diff().dt.days.dropna()
    max_gap = int(date_gaps.max()) if not date_gaps.empty and pd.notna(date_gaps.max()) else 0
    large_gap_count = int((date_gaps > 7).sum()) if not date_gaps.empty else 0

    adj_close = pd.to_numeric(frame.get("adj_close"), errors="coerce")
    high = pd.to_numeric(frame.get("high"), errors="coerce")
    low = pd.to_numeric(frame.get("low"), errors="coerce")
    volume = pd.to_numeric(frame.get("volume"), errors="coerce")
    close = pd.to_numeric(frame.get("close"), errors="coerce") if "close" in frame else pd.Series(dtype=float)

    missing_ohlcv_rows = int(
        pd.concat([high, low, adj_close, volume], axis=1)
        .isna()
        .any(axis=1)
        .sum()
    )
    zero_volume_days = int((volume.fillna(0.0) <= 0).sum())
    high_low_inversion_count = int(((high < low) & high.notna() & low.notna()).sum())

    returns = adj_close.pct_change().replace([np.inf, -np.inf], np.nan)
    extreme_return_count = int((returns.abs() > 0.35).sum())

    adjustment_anomaly_count = 0
    if not close.empty:
        ratio = (adj_close / close.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        adjustment_anomaly_count = int(((ratio <= 0.01) | (ratio >= 20.0)).sum())
    adjustment_anomaly_count += int((adj_close <= 0).sum())

    penalty = 0.0
    if max_gap > 14:
        penalty += 8.0
    elif max_gap > 7:
        penalty += 4.0
    penalty += min(large_gap_count * 2.0, 8.0)
    penalty += min(zero_volume_days * 1.5, 10.0)
    penalty += min(missing_ohlcv_rows * 2.0, 10.0)
    penalty += min(extreme_return_count * 4.0, 12.0)
    penalty += min(high_low_inversion_count * 4.0, 12.0)
    penalty += min(adjustment_anomaly_count * 4.0, 12.0)
    penalty = round(float(min(penalty, 30.0)), 2)

    issue_count = sum(
        count > 0
        for count in [
            large_gap_count,
            zero_volume_days,
            missing_ohlcv_rows,
            extreme_return_count,
            high_low_inversion_count,
            adjustment_anomaly_count,
        ]
    )
    score = round(float(max(0.0, 100.0 - penalty * 3.0)), 2)
    if score >= 90:
        level, level_zh = "clean", "干净"
    elif score >= 75:
        level, level_zh = "usable", "可用"
    elif score >= 55:
        level, level_zh = "watch", "需关注"
    else:
        level, level_zh = "poor", "差"

    notes: list[str] = []
    notes_zh: list[str] = []
    if large_gap_count:
        notes.append(f"{large_gap_count} large calendar gap(s), max gap {max_gap} day(s)")
        notes_zh.append(f"{large_gap_count}个较大日期缺口，最大缺口{max_gap}天")
    if zero_volume_days:
        notes.append(f"{zero_volume_days} zero-volume row(s)")
        notes_zh.append(f"{zero_volume_days}条零成交量记录")
    if missing_ohlcv_rows:
        notes.append(f"{missing_ohlcv_rows} row(s) with missing OHLCV fields")
        notes_zh.append(f"{missing_ohlcv_rows}条OHLCV字段缺失记录")
    if extreme_return_count:
        notes.append(f"{extreme_return_count} extreme adjusted return move(s)")
        notes_zh.append(f"{extreme_return_count}次复权价格极端跳动")
    if high_low_inversion_count:
        notes.append(f"{high_low_inversion_count} high/low inversion row(s)")
        notes_zh.append(f"{high_low_inversion_count}条最高价低于最低价记录")
    if adjustment_anomaly_count:
        notes.append(f"{adjustment_anomaly_count} adjusted-price anomaly row(s)")
        notes_zh.append(f"{adjustment_anomaly_count}条复权价格异常记录")
    if not notes:
        notes.append("price health checks found no major anomalies")
        notes_zh.append("价格健康检查未发现明显异常")

    return {
        "score": score,
        "level": level,
        "level_zh": level_zh,
        "penalty": penalty,
        "issue_count": int(issue_count),
        "max_calendar_gap_days": max_gap,
        "large_gap_count": large_gap_count,
        "zero_volume_days": zero_volume_days,
        "missing_ohlcv_rows": missing_ohlcv_rows,
        "extreme_return_count": extreme_return_count,
        "high_low_inversion_count": high_low_inversion_count,
        "adjustment_anomaly_count": adjustment_anomaly_count,
        "note": "; ".join(notes),
        "note_zh": "；".join(notes_zh),
    }


def _ticker_scores(
    scored: pd.DataFrame,
    ticker: str,
    as_of_date: str | pd.Timestamp | None,
) -> pd.DataFrame:
    if scored.empty or "ticker" not in scored.columns or "date" not in scored.columns:
        return pd.DataFrame()
    frame = scored.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=False)
    frame["ticker"] = frame["ticker"].astype("string").str.upper().str.strip()
    frame = frame[frame["ticker"] == ticker].copy()
    if as_of_date is not None:
        frame = frame[frame["date"] <= pd.Timestamp(as_of_date)]
    return frame.sort_values("date").reset_index(drop=True)


def _latest_score_snapshot(
    scored: pd.DataFrame,
    ticker_scores: pd.DataFrame,
    price_date: pd.Timestamp,
) -> dict[str, object]:
    if ticker_scores.empty:
        return {
            "score_date": pd.NaT,
            "score": np.nan,
            "score_percentile": np.nan,
            "passes_universe": True,
        }

    latest_date = ticker_scores[ticker_scores["date"] <= price_date]["date"].max()
    latest = ticker_scores[ticker_scores["date"] == latest_date].iloc[-1]
    score = float(latest["score"]) if "score" in latest and pd.notna(latest["score"]) else np.nan

    percentile = np.nan
    if "score" in scored.columns:
        scored_dates = scored.copy()
        scored_dates["date"] = pd.to_datetime(scored_dates["date"], utc=False)
        cross_section = scored_dates[
            (scored_dates["date"] == latest_date) & scored_dates["score"].notna()
        ].copy()
        if len(cross_section) > 1 and math.isfinite(score):
            ranks = cross_section["score"].rank(method="average", pct=True)
            ticker_value = str(latest["ticker"]).upper()
            ticker_mask = cross_section["ticker"].astype("string").str.upper() == ticker_value
            if ticker_mask.any():
                percentile = float(ranks.loc[ticker_mask].iloc[-1])

    passes_universe = True
    if "passes_universe" in latest:
        passes_universe = bool(latest["passes_universe"])

    return {
        "score_date": latest_date,
        "score": score,
        "score_percentile": percentile,
        "passes_universe": passes_universe,
    }


def _analyze_horizon(
    price_frame: pd.DataFrame,
    ticker: str,
    spec: HorizonSpec,
    trading_rules: TradingRules,
    score_snapshot: dict[str, object],
    entry_buffer_pct: float,
    market_context: object | None,
    relative_strength_context: object | None,
    event_risk_context: object | None,
    fundamental_context: object | None,
    sentiment_context: object | None,
    analyst_context: object | None,
    valuation_context: object | None,
    sector_context: object | None,
) -> dict[str, object]:
    frame = price_frame.copy()
    close = frame["adj_close"].astype(float)
    high = frame["high"].astype(float).fillna(close)
    low = frame["low"].astype(float).fillna(close)
    volume = frame["volume"].astype(float)

    latest = frame.iloc[-1]
    latest_price = float(latest["adj_close"])
    history_days = int(len(frame))
    trend_ma = _last_valid(close.rolling(spec.trend_window).mean())
    trend_distance = _safe_ratio(latest_price, trend_ma) - 1.0
    trend_slope = _moving_average_slope(close, spec.trend_window)
    momentum = _window_return(close, spec.momentum_window)
    atr = _last_valid(_average_true_range(high, low, close, spec.atr_window))
    avg_volume = _last_valid(volume.rolling(spec.volume_window).mean())
    volume_ratio = _safe_ratio(float(latest["volume"]), avg_volume)

    prior = frame.iloc[:-1].tail(spec.lookback_days)
    if prior.empty:
        prior = frame.tail(spec.lookback_days)
    support = float(prior["low"].min()) if not prior.empty else np.nan
    resistance = float(prior["high"].max()) if not prior.empty else np.nan

    breakout_entry = _finite_or(resistance * (1.0 + entry_buffer_pct), latest_price)
    pullback_entry = _pullback_entry(latest_price, trend_ma, support, entry_buffer_pct)
    action, entry_type, entry_price = _classify_action(
        latest_price=latest_price,
        breakout_entry=breakout_entry,
        pullback_entry=pullback_entry,
        trend_distance=trend_distance,
        trend_slope=trend_slope,
        momentum=momentum,
        volume_ratio=volume_ratio,
        history_days=history_days,
        spec=spec,
        preferred_entry_style=trading_rules.preferred_entry_style,
        passes_universe=bool(score_snapshot["passes_universe"]),
    )
    stop_loss = _stop_loss(
        entry_price=entry_price,
        support=support,
        atr=atr,
        atr_multiple=spec.atr_stop_multiple,
        entry_buffer_pct=entry_buffer_pct,
    )
    risk_per_share = max(entry_price - stop_loss, 0.0)
    take_profit = entry_price + risk_per_share * spec.target_r_multiple
    risk_reward = _safe_ratio(take_profit - entry_price, risk_per_share)
    entry_backtest = _entry_backtest_summary(
        frame=frame,
        spec=spec,
        entry_buffer_pct=entry_buffer_pct,
    )
    backtest_reliability = _backtest_reliability(entry_backtest)
    price_health = _price_data_health_profile(frame)
    backtest_trust = _backtest_trust_profile(
        entry_backtest=entry_backtest,
        price_health=price_health,
    )

    technical_score = _signal_score(
        trend_distance=trend_distance,
        trend_slope=trend_slope,
        momentum=momentum,
        volume_ratio=volume_ratio,
        score_percentile=float(score_snapshot["score_percentile"])
        if pd.notna(score_snapshot["score_percentile"])
        else np.nan,
        history_days=history_days,
        min_history_days=spec.min_history_days,
        passes_universe=bool(score_snapshot["passes_universe"]),
    )
    market_score = _context_float(market_context, "market_score", 50.0)
    relative_strength_score = _context_float(relative_strength_context, "score", 50.0)
    event_risk_level = _context_text(event_risk_context, "event_risk_level", "unknown")
    event_risk_score = _context_float(event_risk_context, "event_risk_score", 50.0)
    event_block_new_entries = _context_bool(event_risk_context, "event_block_new_entries", False)
    event_cooldown_active = _context_bool(event_risk_context, "event_cooldown_active", False)
    fundamental_score = _context_float(fundamental_context, "fundamental_score", 50.0)
    fundamental_quality = _context_text(fundamental_context, "fundamental_quality", "unknown")
    sentiment_score = _context_float(sentiment_context, "sentiment_score", 50.0)
    sentiment_risk_level = _context_text(sentiment_context, "sentiment_risk_level", "unknown")
    sentiment_block_new_entries = _context_bool(
        sentiment_context,
        "sentiment_block_new_entries",
        False,
    )
    analyst_score = _context_float(analyst_context, "analyst_score", 50.0)
    analyst_risk_level = _context_text(analyst_context, "analyst_risk_level", "unknown")
    analyst_block_new_entries = _context_bool(
        analyst_context,
        "analyst_block_new_entries",
        False,
    )
    valuation_score = _context_float(valuation_context, "valuation_score", 50.0)
    valuation_risk_level = _context_text(valuation_context, "valuation_risk_level", "unknown")
    valuation_block_new_entries = _context_bool(
        valuation_context,
        "valuation_block_new_entries",
        False,
    )
    sector_score = _context_float(sector_context, "sector_score", 50.0)
    signal_score = _combined_signal_score(
        spec=spec,
        technical_score=technical_score,
        market_score=market_score,
        relative_strength_score=relative_strength_score,
        fundamental_score=fundamental_score,
        analyst_score=analyst_score,
        valuation_score=valuation_score,
        sector_score=sector_score,
    )
    original_action = action
    action, entry_type = _apply_context_downgrade(
        action=action,
        entry_type=entry_type,
        market_score=market_score,
        relative_strength_score=relative_strength_score,
        event_risk_level=event_risk_level,
        event_block_new_entries=event_block_new_entries,
        sentiment_risk_level=sentiment_risk_level,
        sentiment_block_new_entries=sentiment_block_new_entries,
        analyst_risk_level=analyst_risk_level,
        analyst_block_new_entries=analyst_block_new_entries,
        valuation_risk_level=valuation_risk_level,
        valuation_block_new_entries=valuation_block_new_entries,
        fundamental_score=fundamental_score,
        fundamental_quality=fundamental_quality,
        sector_score=sector_score,
        horizon=spec.name,
    )
    rationale = _rationale(
        spec=spec,
        action=action,
        trend_distance=trend_distance,
        trend_slope=trend_slope,
        momentum=momentum,
        volume_ratio=volume_ratio,
        support=support,
        resistance=resistance,
        history_days=history_days,
        passes_universe=bool(score_snapshot["passes_universe"]),
    )
    rationale_zh = _rationale_zh(
        spec=spec,
        action=action,
        trend_distance=trend_distance,
        trend_slope=trend_slope,
        momentum=momentum,
        volume_ratio=volume_ratio,
        support=support,
        resistance=resistance,
        history_days=history_days,
        passes_universe=bool(score_snapshot["passes_universe"]),
    )
    action_explanation, action_explanation_zh = _action_explanation(
        action=action,
        original_action=original_action,
        market_score=market_score,
        relative_strength_score=relative_strength_score,
        event_risk_level=event_risk_level,
        days_until_earnings=_context_float(event_risk_context, "days_until_earnings", np.nan),
        event_window=_context_text(event_risk_context, "event_window", "unknown"),
        sentiment_risk_level=sentiment_risk_level,
        sentiment_score=sentiment_score,
        analyst_risk_level=analyst_risk_level,
        analyst_score=analyst_score,
        valuation_risk_level=valuation_risk_level,
        valuation_score=valuation_score,
        fundamental_score=fundamental_score,
        fundamental_quality=fundamental_quality,
    )
    plain_summary, plain_summary_zh = _plain_summary(
        spec=spec,
        action=action,
        signal_score=signal_score,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
    )
    entry_distance_pct = _safe_ratio(entry_price, latest_price) - 1.0
    entry_plan = _entry_plan(
        action=action,
        entry_type=entry_type,
        latest_price=latest_price,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        support=support,
        resistance=resistance,
        entry_distance_pct=entry_distance_pct,
    )

    return {
        "date": latest["date"],
        "ticker": ticker,
        "horizon": spec.name,
        "horizon_label": spec.label,
        "horizon_zh_label": spec.zh_label,
        "horizon_time_range": spec.time_range,
        "horizon_time_range_zh": spec.time_range_zh,
        "trading_rule_entry_style": trading_rules.preferred_entry_style,
        "trading_rule_entry_style_zh": trading_rules.preferred_entry_style_zh,
        "trading_rule_atr_stop_multiple": spec.atr_stop_multiple,
        "trading_rule_target_r_multiple": spec.target_r_multiple,
        "trading_rule_max_chase_pct": spec.max_chase_pct,
        "trading_rule_time_stop_days": spec.time_stop_days,
        "trading_rule_trailing_stop_trigger_r": spec.trailing_stop_trigger_r,
        "trading_rule_trailing_stop_lock_r": spec.trailing_stop_lock_r,
        "trading_rule_sell_rule": trading_rules.sell_rule,
        "trading_rule_sell_rule_zh": trading_rules.sell_rule_zh,
        "trading_rule_note": _trading_rule_note(spec, trading_rules),
        "trading_rule_note_zh": _trading_rule_note_zh(spec, trading_rules),
        "action": action,
        "original_action": original_action,
        "signal_score": signal_score,
        "technical_score": technical_score,
        "market_score": market_score,
        "relative_strength_score": relative_strength_score,
        "event_risk_score": event_risk_score,
        "event_window": _context_text(event_risk_context, "event_window", "unknown"),
        "event_window_zh": _context_text(event_risk_context, "event_window_zh", "未知"),
        "event_block_new_entries": event_block_new_entries,
        "event_cooldown_active": event_cooldown_active,
        "fundamental_score": fundamental_score,
        "sentiment_score": sentiment_score,
        "sentiment_label": _context_text(sentiment_context, "sentiment_label", "unknown"),
        "sentiment_label_zh": _context_text(sentiment_context, "sentiment_label_zh", "未知"),
        "sentiment_risk_level": sentiment_risk_level,
        "sentiment_risk_level_zh": _context_text(
            sentiment_context,
            "sentiment_risk_level_zh",
            "未知",
        ),
        "sentiment_block_new_entries": sentiment_block_new_entries,
        "positive_news_level": _context_text(sentiment_context, "positive_news_level", "unknown"),
        "positive_news_level_zh": _context_text(
            sentiment_context,
            "positive_news_level_zh",
            "未知",
        ),
        "positive_news_score": _context_float(sentiment_context, "positive_news_score", 0.0),
        "positive_news_major_count": int(
            _context_float(sentiment_context, "positive_news_major_count", 0.0)
        ),
        "positive_news_strong_count": int(
            _context_float(sentiment_context, "positive_news_strong_count", 0.0)
        ),
        "positive_news_moderate_count": int(
            _context_float(sentiment_context, "positive_news_moderate_count", 0.0)
        ),
        "positive_news_drivers": _context_text(
            sentiment_context,
            "positive_news_drivers",
            "none",
        ),
        "positive_news_drivers_zh": _context_text(
            sentiment_context,
            "positive_news_drivers_zh",
            "无",
        ),
        "risk_news_level": _context_text(sentiment_context, "risk_news_level", "none"),
        "risk_news_level_zh": _context_text(sentiment_context, "risk_news_level_zh", "无明显风险"),
        "risk_news_score": _context_float(sentiment_context, "risk_news_score", 0.0),
        "fake_catalyst_count": int(_context_float(sentiment_context, "fake_catalyst_count", 0.0)),
        "dilution_count": int(_context_float(sentiment_context, "dilution_count", 0.0)),
        "risk_news_drivers": _context_text(sentiment_context, "risk_news_drivers", "none"),
        "risk_news_drivers_zh": _context_text(sentiment_context, "risk_news_drivers_zh", "无"),
        "sentiment_positive_count": int(
            _context_float(sentiment_context, "sentiment_positive_count", 0.0)
        ),
        "sentiment_negative_count": int(
            _context_float(sentiment_context, "sentiment_negative_count", 0.0)
        ),
        "sentiment_high_risk_count": int(
            _context_float(sentiment_context, "sentiment_high_risk_count", 0.0)
        ),
        "sentiment_titles_used": int(
            _context_float(sentiment_context, "sentiment_titles_used", 0.0)
        ),
        "sentiment_note": _context_text(
            sentiment_context,
            "sentiment_note",
            "Recent news title data is unavailable; neutral score 50 is used.",
        ),
        "sentiment_note_zh": _context_text(
            sentiment_context,
            "sentiment_note_zh",
            "近期新闻标题不可用；使用中性分数50。",
        ),
        "sentiment_warning": _context_text(sentiment_context, "sentiment_warning", ""),
        "analyst_score": analyst_score,
        "analyst_label": _context_text(analyst_context, "analyst_label", "unknown"),
        "analyst_label_zh": _context_text(analyst_context, "analyst_label_zh", "未知"),
        "analyst_risk_level": analyst_risk_level,
        "analyst_risk_level_zh": _context_text(
            analyst_context,
            "analyst_risk_level_zh",
            "未知",
        ),
        "analyst_block_new_entries": analyst_block_new_entries,
        "analyst_upside": _context_float(analyst_context, "analyst_upside", np.nan),
        "recommendation_mean": _context_float(analyst_context, "recommendation_mean", np.nan),
        "recommendation_key": _context_text(analyst_context, "recommendation_key", ""),
        "number_of_analysts": _context_float(analyst_context, "number_of_analysts", np.nan),
        "target_mean_price": _context_float(analyst_context, "target_mean_price", np.nan),
        "analyst_note": _context_text(
            analyst_context,
            "analyst_note",
            "Analyst expectation data is unavailable; neutral score 50 is used.",
        ),
        "analyst_note_zh": _context_text(
            analyst_context,
            "analyst_note_zh",
            "分析师预期数据不可用；使用中性分数50。",
        ),
        "analyst_warning": _context_text(analyst_context, "analyst_warning", ""),
        "analyst_data_coverage": _context_float(analyst_context, "data_coverage", 0.0),
        "valuation_score": valuation_score,
        "valuation_label": _context_text(valuation_context, "valuation_label", "unknown"),
        "valuation_label_zh": _context_text(valuation_context, "valuation_label_zh", "未知"),
        "valuation_risk_level": valuation_risk_level,
        "valuation_risk_level_zh": _context_text(
            valuation_context,
            "valuation_risk_level_zh",
            "未知",
        ),
        "valuation_block_new_entries": valuation_block_new_entries,
        "valuation_forward_pe": _context_float(valuation_context, "valuation_forward_pe", np.nan),
        "valuation_trailing_pe": _context_float(valuation_context, "valuation_trailing_pe", np.nan),
        "valuation_peg_ratio": _context_float(valuation_context, "valuation_peg_ratio", np.nan),
        "valuation_free_cash_flow_yield": _context_float(
            valuation_context,
            "valuation_free_cash_flow_yield",
            np.nan,
        ),
        "valuation_market_cap": _context_float(
            valuation_context,
            "valuation_market_cap",
            np.nan,
        ),
        "valuation_growth_reference": _context_float(
            valuation_context,
            "valuation_growth_reference",
            np.nan,
        ),
        "valuation_profit_margin": _context_float(
            valuation_context,
            "valuation_profit_margin",
            np.nan,
        ),
        "valuation_note": _context_text(
            valuation_context,
            "valuation_note",
            "Valuation data is unavailable; neutral score 50 is used.",
        ),
        "valuation_note_zh": _context_text(
            valuation_context,
            "valuation_note_zh",
            "估值数据不可用；使用中性分数50。",
        ),
        "valuation_warning": _context_text(valuation_context, "valuation_warning", ""),
        "valuation_data_coverage": _context_float(valuation_context, "data_coverage", 0.0),
        "sector_score": sector_score,
        "sector_status": _context_text(sector_context, "sector_status", "unknown"),
        "sector": _context_text(sector_context, "sector", ""),
        "industry": _context_text(sector_context, "industry", ""),
        "sector_etf": _context_text(sector_context, "sector_etf", ""),
        "sector_note": _context_text(
            sector_context,
            "sector_note",
            "Sector ETF data is unavailable; neutral sector score 50 is used.",
        ),
        "sector_note_zh": _context_text(
            sector_context,
            "sector_note_zh",
            "板块ETF数据不可用，使用中性板块分数50。",
        ),
        "sector_warning": _context_text(sector_context, "sector_warning", ""),
        "sector_trend_score": _context_float(sector_context, "sector_trend_score", 50.0),
        "sector_relative_strength": _context_float(
            sector_context,
            "sector_relative_strength",
            np.nan,
        ),
        "fundamental_quality": fundamental_quality,
        "fundamental_quality_zh": _context_text(
            fundamental_context,
            "fundamental_quality_zh",
            "未知",
        ),
        "fundamental_note": _context_text(
            fundamental_context,
            "fundamental_note",
            "Fundamental data is unavailable or too sparse; neutral score 50 is used.",
        ),
        "fundamental_note_zh": _context_text(
            fundamental_context,
            "fundamental_note_zh",
            "基本面数据不可用或字段太少；使用中性分数50。",
        ),
        "fundamental_warning": _context_text(fundamental_context, "fundamental_warning", ""),
        "fundamental_data_coverage": _context_float(fundamental_context, "data_coverage", 0.0),
        "revenue_growth": _context_float(fundamental_context, "revenue_growth", np.nan),
        "earnings_growth": _context_float(fundamental_context, "earnings_growth", np.nan),
        "profit_margin": _context_float(fundamental_context, "profit_margin", np.nan),
        "return_on_equity": _context_float(fundamental_context, "return_on_equity", np.nan),
        "free_cash_flow": _context_float(fundamental_context, "free_cash_flow", np.nan),
        "forward_pe": _context_float(fundamental_context, "forward_pe", np.nan),
        "peg_ratio": _context_float(fundamental_context, "peg_ratio", np.nan),
        "debt_to_equity": _context_float(fundamental_context, "debt_to_equity", np.nan),
        "fundamental_cash_flow_quality": _context_float(
            fundamental_context, "cash_flow_quality_score", np.nan
        ),
        "fundamental_gross_margin": _context_float(fundamental_context, "gross_margin", np.nan),
        "fundamental_operating_margin": _context_float(
            fundamental_context, "operating_margin", np.nan
        ),
        "fundamental_fcf_margin": _context_float(fundamental_context, "fcf_margin", np.nan),
        "fundamental_cash_conversion": _context_float(
            fundamental_context, "cash_conversion", np.nan
        ),
        "fundamental_net_debt_to_equity": _context_float(
            fundamental_context, "net_debt_to_equity", np.nan
        ),
        "fundamental_trend_status": _context_text(
            fundamental_context, "trend_status", "insufficient_history"
        ),
        "fundamental_trend_direction": _context_text(
            fundamental_context, "trend_direction", "unknown"
        ),
        "fundamental_trend_direction_zh": _context_text(
            fundamental_context, "trend_direction_zh", "未知"
        ),
        "fundamental_trend_score": _context_float(fundamental_context, "trend_score", np.nan),
        "fundamental_gross_margin_trend": _context_text(
            fundamental_context, "gross_margin_trend", "unknown"
        ),
        "fundamental_operating_margin_trend": _context_text(
            fundamental_context, "operating_margin_trend", "unknown"
        ),
        "fundamental_net_margin_trend": _context_text(
            fundamental_context, "net_margin_trend", "unknown"
        ),
        "fundamental_fcf_margin_trend": _context_text(
            fundamental_context, "fcf_margin_trend", "unknown"
        ),
        "fundamental_revenue_growth_trend": _context_text(
            fundamental_context, "revenue_growth_trend", "unknown"
        ),
        "signal_timing": "generated_after_latest_close",
        "earliest_execution": "next_trading_session",
        "latest_price": latest_price,
        "entry_type": entry_type,
        "entry_price": entry_price,
        "current_distance_to_entry_pct": entry_distance_pct,
        "entry_distance_pct": entry_distance_pct,
        "entry_distance_note": entry_plan["entry_distance_note"],
        "entry_distance_note_zh": entry_plan["entry_distance_note_zh"],
        "chase_status": entry_plan["chase_status"],
        "chase_status_zh": entry_plan["chase_status_zh"],
        "entry_plan": entry_plan["entry_plan"],
        "entry_plan_zh": entry_plan["entry_plan_zh"],
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "risk_reward": risk_reward,
        "risk_per_share": risk_per_share,
        "backtest_execution_model": entry_backtest["backtest_execution_model"],
        "backtest_execution_model_zh": entry_backtest["backtest_execution_model_zh"],
        "backtest_time_stop_days": entry_backtest["backtest_time_stop_days"],
        "backtest_trailing_stop_trigger_r": entry_backtest["backtest_trailing_stop_trigger_r"],
        "backtest_trailing_stop_lock_r": entry_backtest["backtest_trailing_stop_lock_r"],
        "backtest_slippage_pct": entry_backtest["backtest_slippage_pct"],
        "backtest_avg_dollar_volume": entry_backtest["backtest_avg_dollar_volume"],
        "backtest_atr_ratio": entry_backtest["backtest_atr_ratio"],
        "backtest_liquidity_label": entry_backtest["backtest_liquidity_label"],
        "backtest_liquidity_label_zh": entry_backtest["backtest_liquidity_label_zh"],
        "backtest_volatility_label": entry_backtest["backtest_volatility_label"],
        "backtest_volatility_label_zh": entry_backtest["backtest_volatility_label_zh"],
        "backtest_execution_note": entry_backtest["backtest_execution_note"],
        "backtest_execution_note_zh": entry_backtest["backtest_execution_note_zh"],
        "regime_coverage_score": entry_backtest["regime_coverage_score"],
        "regime_coverage_level": entry_backtest["regime_coverage_level"],
        "regime_coverage_level_zh": entry_backtest["regime_coverage_level_zh"],
        "regime_coverage_regime_count": entry_backtest["regime_coverage_regime_count"],
        "regime_coverage_dominant_regime": entry_backtest["regime_coverage_dominant_regime"],
        "regime_coverage_dominant_regime_zh": entry_backtest[
            "regime_coverage_dominant_regime_zh"
        ],
        "regime_coverage_dominant_share": entry_backtest["regime_coverage_dominant_share"],
        "regime_coverage_note": entry_backtest["regime_coverage_note"],
        "regime_coverage_note_zh": entry_backtest["regime_coverage_note_zh"],
        "recent_backtest_score": entry_backtest["recent_backtest_score"],
        "recent_backtest_level": entry_backtest["recent_backtest_level"],
        "recent_backtest_level_zh": entry_backtest["recent_backtest_level_zh"],
        "recent_backtest_trade_count": entry_backtest["recent_backtest_trade_count"],
        "recent_backtest_win_rate": entry_backtest["recent_backtest_win_rate"],
        "recent_backtest_average_return": entry_backtest["recent_backtest_average_return"],
        "recent_backtest_return_delta": entry_backtest["recent_backtest_return_delta"],
        "recent_backtest_note": entry_backtest["recent_backtest_note"],
        "recent_backtest_note_zh": entry_backtest["recent_backtest_note_zh"],
        "backtest_decay_score": entry_backtest["backtest_decay_score"],
        "backtest_decay_level": entry_backtest["backtest_decay_level"],
        "backtest_decay_level_zh": entry_backtest["backtest_decay_level_zh"],
        "backtest_decay_early_trade_count": entry_backtest["backtest_decay_early_trade_count"],
        "backtest_decay_late_trade_count": entry_backtest["backtest_decay_late_trade_count"],
        "backtest_decay_early_win_rate": entry_backtest["backtest_decay_early_win_rate"],
        "backtest_decay_late_win_rate": entry_backtest["backtest_decay_late_win_rate"],
        "backtest_decay_early_average_return": entry_backtest[
            "backtest_decay_early_average_return"
        ],
        "backtest_decay_late_average_return": entry_backtest[
            "backtest_decay_late_average_return"
        ],
        "backtest_decay_win_rate_delta": entry_backtest["backtest_decay_win_rate_delta"],
        "backtest_decay_average_return_delta": entry_backtest[
            "backtest_decay_average_return_delta"
        ],
        "backtest_decay_note": entry_backtest["backtest_decay_note"],
        "backtest_decay_note_zh": entry_backtest["backtest_decay_note_zh"],
        "breakout_trade_count": entry_backtest["breakout_trade_count"],
        "breakout_sample_quality": entry_backtest["breakout_sample_quality"],
        "breakout_sample_quality_zh": entry_backtest["breakout_sample_quality_zh"],
        "breakout_win_rate": entry_backtest["breakout_win_rate"],
        "breakout_target_hit_rate": entry_backtest["breakout_target_hit_rate"],
        "breakout_stop_hit_rate": entry_backtest["breakout_stop_hit_rate"],
        "breakout_trailing_stop_hit_rate": entry_backtest["breakout_trailing_stop_hit_rate"],
        "breakout_average_gain": entry_backtest["breakout_average_gain"],
        "breakout_average_loss": entry_backtest["breakout_average_loss"],
        "breakout_average_return": entry_backtest["breakout_average_return"],
        "pullback_trade_count": entry_backtest["pullback_trade_count"],
        "pullback_sample_quality": entry_backtest["pullback_sample_quality"],
        "pullback_sample_quality_zh": entry_backtest["pullback_sample_quality_zh"],
        "pullback_win_rate": entry_backtest["pullback_win_rate"],
        "pullback_target_hit_rate": entry_backtest["pullback_target_hit_rate"],
        "pullback_stop_hit_rate": entry_backtest["pullback_stop_hit_rate"],
        "pullback_trailing_stop_hit_rate": entry_backtest["pullback_trailing_stop_hit_rate"],
        "pullback_average_gain": entry_backtest["pullback_average_gain"],
        "pullback_average_loss": entry_backtest["pullback_average_loss"],
        "pullback_average_return": entry_backtest["pullback_average_return"],
        "entry_backtest_note": entry_backtest["entry_backtest_note"],
        "entry_backtest_note_zh": entry_backtest["entry_backtest_note_zh"],
        "backtest_reliability_level": backtest_reliability["level"],
        "backtest_reliability_level_zh": backtest_reliability["level_zh"],
        "backtest_reliability_note": backtest_reliability["note"],
        "backtest_reliability_note_zh": backtest_reliability["note_zh"],
        "backtest_trust_score": backtest_trust["score"],
        "backtest_trust_level": backtest_trust["level"],
        "backtest_trust_level_zh": backtest_trust["level_zh"],
        "backtest_sample_score": backtest_trust["sample_score"],
        "backtest_liquidity_score": backtest_trust["liquidity_score"],
        "backtest_slippage_score": backtest_trust["slippage_score"],
        "backtest_return_evidence_score": backtest_trust["return_evidence_score"],
        "backtest_trust_note": backtest_trust["note"],
        "backtest_trust_note_zh": backtest_trust["note_zh"],
        "price_health_score": price_health["score"],
        "price_health_level": price_health["level"],
        "price_health_level_zh": price_health["level_zh"],
        "price_health_penalty": price_health["penalty"],
        "price_health_issue_count": price_health["issue_count"],
        "price_max_calendar_gap_days": price_health["max_calendar_gap_days"],
        "price_large_gap_count": price_health["large_gap_count"],
        "price_zero_volume_days": price_health["zero_volume_days"],
        "price_missing_ohlcv_rows": price_health["missing_ohlcv_rows"],
        "price_extreme_return_count": price_health["extreme_return_count"],
        "price_high_low_inversion_count": price_health["high_low_inversion_count"],
        "price_adjustment_anomaly_count": price_health["adjustment_anomaly_count"],
        "price_health_note": price_health["note"],
        "price_health_note_zh": price_health["note_zh"],
        "support": support,
        "resistance": resistance,
        "trend_ma": trend_ma,
        "trend_distance": trend_distance,
        "trend_slope": trend_slope,
        "momentum": momentum,
        "atr": atr,
        "volume_ratio": volume_ratio,
        "history_days": history_days,
        "passes_universe": bool(score_snapshot["passes_universe"]),
        "score_date": score_snapshot["score_date"],
        "score": score_snapshot["score"],
        "score_percentile": score_snapshot["score_percentile"],
        "market_status": _context_text(market_context, "market_status", "neutral"),
        "market_note": _context_text(
            market_context,
            "note",
            "Market data unavailable; neutral score 50 is used.",
        ),
        "market_note_zh": _context_text(
            market_context,
            "note_zh",
            "大盘数据不可用，使用中性分数50。",
        ),
        "market_warning": _context_text(market_context, "warning", ""),
        "event_risk_level": event_risk_level,
        "event_risk_level_zh": _context_text(event_risk_context, "event_risk_level_zh", "未知"),
        "next_earnings_date": _context_text(event_risk_context, "next_earnings_date", ""),
        "days_until_earnings": _context_float(event_risk_context, "days_until_earnings", np.nan),
        "last_earnings_date": _context_text(event_risk_context, "last_earnings_date", ""),
        "days_since_earnings": _context_float(event_risk_context, "days_since_earnings", np.nan),
        "event_risk_note": _context_text(
            event_risk_context,
            "event_risk_note",
            "Upcoming earnings date is unavailable; no event-risk downgrade is applied.",
        ),
        "event_risk_note_zh": _context_text(
            event_risk_context,
            "event_risk_note_zh",
            "暂时无法取得下一次财报日期；不进行事件风险降级。",
        ),
        "event_risk_warning": _context_text(event_risk_context, "event_risk_warning", ""),
        "relative_strength_note": _context_text(
            relative_strength_context,
            "note",
            "Relative strength data unavailable; neutral score 50 is used.",
        ),
        "relative_strength_note_zh": _context_text(
            relative_strength_context,
            "note_zh",
            "相对强弱数据不可用，使用中性分数50。",
        ),
        "relative_strength_warning": _context_text(relative_strength_context, "warning", ""),
        "vs_spy_return": _context_float(relative_strength_context, "vs_spy_return", np.nan),
        "vs_qqq_return": _context_float(relative_strength_context, "vs_qqq_return", np.nan),
        "rationale": rationale,
        "rationale_zh": rationale_zh,
        "action_explanation": action_explanation,
        "action_explanation_zh": action_explanation_zh,
        "plain_summary": plain_summary,
        "plain_summary_zh": plain_summary_zh,
    }


def _classify_action(
    latest_price: float,
    breakout_entry: float,
    pullback_entry: float,
    trend_distance: float,
    trend_slope: float,
    momentum: float,
    volume_ratio: float,
    history_days: int,
    spec: HorizonSpec,
    preferred_entry_style: str,
    passes_universe: bool,
) -> tuple[str, str, float]:
    if history_days < spec.min_history_days:
        return "wait_insufficient_history", "none", latest_price
    if not passes_universe:
        return "avoid_universe_filter", "none", latest_price

    uptrend = trend_distance > 0 and trend_slope >= 0
    positive_momentum = momentum > 0
    breakout_confirmed = latest_price >= breakout_entry and volume_ratio >= 1.0
    near_pullback = abs(_safe_ratio(latest_price, pullback_entry) - 1.0) <= 0.02
    extended = trend_distance > spec.max_chase_pct
    entry_style = str(preferred_entry_style or "balanced").lower().strip()

    if entry_style in {"pullback", "conservative"}:
        if uptrend and positive_momentum and near_pullback:
            return "entry_pullback_zone", "limit_pullback", min(latest_price, pullback_entry)
        if uptrend and positive_momentum and (extended or breakout_confirmed):
            return "wait_overextended", "limit_pullback", pullback_entry
        if uptrend and positive_momentum:
            return "watch_breakout_or_pullback", "limit_pullback", pullback_entry
        return "avoid_or_wait_downtrend", "none", latest_price

    if entry_style == "breakout":
        # Breakout/momentum style: a confirmed breakout (new high on volume) is executable
        # even when the stock is extended above its baseline -- that is the whole point of
        # chasing strength. Only refuse when the move is parabolic (far beyond the normal
        # chase cap), to avoid buying a blown-out spike.
        parabolic = trend_distance > spec.max_chase_pct * BREAKOUT_EXTENSION_MULTIPLE
        if breakout_confirmed and uptrend and positive_momentum and not parabolic:
            return "entry_breakout_confirmed", "market_or_limit", latest_price
        if uptrend and positive_momentum and not extended:
            return "watch_breakout_or_pullback", "stop_limit_breakout", breakout_entry
        if uptrend and positive_momentum and near_pullback:
            return "entry_pullback_zone", "limit_pullback", min(latest_price, pullback_entry)
        if uptrend and positive_momentum and extended:
            return "wait_overextended", "limit_pullback", pullback_entry
        return "avoid_or_wait_downtrend", "none", latest_price

    if breakout_confirmed and uptrend and positive_momentum:
        return "entry_breakout_confirmed", "market_or_limit", latest_price
    if uptrend and positive_momentum and near_pullback:
        return "entry_pullback_zone", "limit_pullback", min(latest_price, pullback_entry)
    if uptrend and positive_momentum and not extended:
        return "watch_breakout_or_pullback", "stop_limit_breakout", breakout_entry
    if uptrend and positive_momentum and extended:
        return "wait_overextended", "limit_pullback", pullback_entry
    return "avoid_or_wait_downtrend", "none", latest_price


def _signal_score(
    trend_distance: float,
    trend_slope: float,
    momentum: float,
    volume_ratio: float,
    score_percentile: float,
    history_days: int,
    min_history_days: int,
    passes_universe: bool,
) -> float:
    if history_days < min_history_days or not passes_universe:
        return 0.0

    score = 0.0
    if trend_distance > 0:
        score += min(trend_distance / 0.10, 1.0) * 25.0
    if trend_slope > 0:
        score += min(trend_slope / 0.05, 1.0) * 15.0
    if momentum > 0:
        score += min(momentum / 0.12, 1.0) * 25.0
    if volume_ratio > 1.0:
        score += min((volume_ratio - 1.0) / 1.0, 1.0) * 10.0
    if _is_finite(score_percentile):
        score += score_percentile * 25.0
    else:
        score += 12.5
    return round(float(max(0.0, min(score, 100.0))), 2)


def _combined_signal_score(
    spec: HorizonSpec,
    technical_score: float,
    market_score: float,
    relative_strength_score: float,
    fundamental_score: float,
    analyst_score: float,
    valuation_score: float,
    sector_score: float,
) -> float:
    weights = {
        "short": (0.48, 0.18, 0.13, 0.07, 0.06, 0.06, 0.02),
        "medium": (0.36, 0.16, 0.13, 0.14, 0.06, 0.07, 0.08),
        "long": (0.25, 0.12, 0.09, 0.33, 0.07, 0.07, 0.07),
    }
    (
        technical_weight,
        market_weight,
        relative_weight,
        fundamental_weight,
        analyst_weight,
        valuation_weight,
        sector_weight,
    ) = weights[spec.name]
    value = (
        technical_weight * technical_score
        + market_weight * market_score
        + relative_weight * relative_strength_score
        + fundamental_weight * fundamental_score
        + analyst_weight * analyst_score
        + valuation_weight * valuation_score
        + sector_weight * sector_score
    )
    return round(float(max(0.0, min(value, 100.0))), 2)


def _apply_context_downgrade(
    action: str,
    entry_type: str,
    market_score: float,
    relative_strength_score: float,
    event_risk_level: str,
    event_block_new_entries: bool,
    sentiment_risk_level: str,
    sentiment_block_new_entries: bool,
    analyst_risk_level: str,
    analyst_block_new_entries: bool,
    valuation_risk_level: str,
    valuation_block_new_entries: bool,
    fundamental_score: float,
    fundamental_quality: str,
    sector_score: float,
    horizon: str,
) -> tuple[str, str]:
    if action in {"wait_insufficient_history", "avoid_universe_filter", "avoid_or_wait_downtrend"}:
        return action, entry_type
    if event_risk_level == "high" or event_block_new_entries:
        return "wait_event_risk", "deferred_event_risk_filter"
    if sentiment_risk_level == "high" or sentiment_block_new_entries:
        return "wait_sentiment_risk", "deferred_sentiment_filter"
    if analyst_risk_level == "high" or analyst_block_new_entries:
        return "wait_analyst_weak", "deferred_analyst_filter"
    if valuation_risk_level == "high" or valuation_block_new_entries:
        return "wait_valuation_rich", "deferred_valuation_filter"
    if horizon == "long" and fundamental_quality == "weak" and fundamental_score <= 40:
        return "wait_fundamental_weak", "deferred_fundamental_filter"
    if horizon in {"medium", "long"} and sector_score <= 40:
        return "wait_sector_weak", "deferred_sector_filter"
    if market_score <= 40:
        return "wait_market_weak", "deferred_market_filter"
    if relative_strength_score <= 40:
        return "wait_relative_weak", "deferred_relative_strength_filter"
    return action, entry_type


def _rationale(
    spec: HorizonSpec,
    action: str,
    trend_distance: float,
    trend_slope: float,
    momentum: float,
    volume_ratio: float,
    support: float,
    resistance: float,
    history_days: int,
    passes_universe: bool,
) -> str:
    if history_days < spec.min_history_days:
        return (
            f"Only {history_days} price rows are available; "
            f"{spec.name} horizon needs at least {spec.min_history_days}."
        )
    if not passes_universe:
        return "Ticker does not pass the configured universe filter."

    trend_text = "above" if trend_distance > 0 else "below"
    slope_text = "rising" if trend_slope >= 0 else "falling"
    momentum_text = "positive" if momentum > 0 else "negative"
    volume_text = "above average" if volume_ratio >= 1 else "below average"
    return (
        f"{action}; price is {trend_text} the {spec.trend_window}-day trend, "
        f"trend is {slope_text}, {spec.momentum_window}-day momentum is {momentum_text}, "
        f"volume is {volume_text}, support is {_format_number(support)}, "
        f"resistance is {_format_number(resistance)}."
    )


def _rationale_zh(
    spec: HorizonSpec,
    action: str,
    trend_distance: float,
    trend_slope: float,
    momentum: float,
    volume_ratio: float,
    support: float,
    resistance: float,
    history_days: int,
    passes_universe: bool,
) -> str:
    if history_days < spec.min_history_days:
        return f"当前只有{history_days}条价格数据，{spec.zh_label}至少需要{spec.min_history_days}条。"
    if not passes_universe:
        return "该股票没有通过当前流动性或历史数据过滤。"

    trend_text = "高于" if trend_distance > 0 else "低于"
    slope_text = "上行" if trend_slope >= 0 else "下行"
    momentum_text = "为正" if momentum > 0 else "为负"
    volume_text = "高于均量" if volume_ratio >= 1 else "低于均量"
    return (
        f"{action}; 价格{trend_text}{spec.trend_window}日趋势线，趋势斜率{slope_text}，"
        f"{spec.momentum_window}日动量{momentum_text}，成交量{volume_text}，"
        f"支撑位{_format_number(support)}，压力位{_format_number(resistance)}。"
    )


def _action_explanation(
    action: str,
    original_action: str,
    market_score: float,
    relative_strength_score: float,
    event_risk_level: str,
    days_until_earnings: float,
    event_window: str,
    sentiment_risk_level: str,
    sentiment_score: float,
    analyst_risk_level: str,
    analyst_score: float,
    valuation_risk_level: str,
    valuation_score: float,
    fundamental_score: float,
    fundamental_quality: str,
) -> tuple[str, str]:
    event_days_text = (
        f"{int(days_until_earnings)} days"
        if _is_finite(days_until_earnings)
        else "an unknown number of days"
    )
    event_days_text_zh = (
        f"{int(days_until_earnings)}天"
        if _is_finite(days_until_earnings)
        else "未知天数"
    )
    explanations = {
        "entry_breakout_confirmed": (
            "Breakout is confirmed by trend, momentum, and volume.",
            "趋势、动量和成交量确认突破，可按计划入场。",
        ),
        "entry_pullback_zone": (
            "Price is near a pullback zone inside an uptrend.",
            "价格处于上升趋势中的回调区域，可小心分批。",
        ),
        "watch_breakout_or_pullback": (
            "Setup is constructive, but entry needs breakout or pullback confirmation.",
            "结构尚可，但需要突破或回调确认后再入场。",
        ),
        "wait_overextended": (
            "Price is extended above trend; wait for a better entry.",
            "价格相对趋势偏高，等待更好的回调位置。",
        ),
        "wait_market_weak": (
            f"Market score is weak at {market_score:.1f}; original action was {original_action}.",
            f"大盘分数偏弱({market_score:.1f})，原始动作为{original_action}，因此降级等待。",
        ),
        "wait_relative_weak": (
            f"Relative strength score is weak at {relative_strength_score:.1f}; original action was {original_action}.",
            f"相对强弱分数偏弱({relative_strength_score:.1f})，原始动作为{original_action}，因此降级等待。",
        ),
        "wait_event_risk": (
            (
                f"Event risk is {event_risk_level}; next known earnings are in "
                f"{event_days_text}; event window is {event_window}; original action was {original_action}."
            ),
            (
                f"事件风险为{event_risk_level}，距离已知下一次财报还有{event_days_text_zh}，"
                f"事件窗口为{event_window}，原始动作为{original_action}，因此降级等待。"
            ),
        ),
        "wait_sentiment_risk": (
            (
                f"News sentiment risk is {sentiment_risk_level} with score "
                f"{sentiment_score:.1f}; original action was {original_action}."
            ),
            (
                f"新闻情绪风险为{sentiment_risk_level}，分数{sentiment_score:.1f}，"
                f"原始动作为{original_action}，因此降级等待。"
            ),
        ),
        "wait_analyst_weak": (
            (
                f"Analyst expectation risk is {analyst_risk_level} with score "
                f"{analyst_score:.1f}; original action was {original_action}."
            ),
            (
                f"分析师预期风险为{analyst_risk_level}，分数{analyst_score:.1f}，"
                f"原始动作为{original_action}，因此降级等待。"
            ),
        ),
        "wait_valuation_rich": (
            (
                f"Valuation risk is {valuation_risk_level} with score "
                f"{valuation_score:.1f}; original action was {original_action}."
            ),
            (
                f"估值风险为{valuation_risk_level}，分数{valuation_score:.1f}，"
                f"原始动作为{original_action}，因此等待更合理估值。"
            ),
        ),
        "wait_fundamental_weak": (
            (
                f"Fundamental quality is {fundamental_quality} with score "
                f"{fundamental_score:.1f}; original action was {original_action}."
            ),
            (
                f"基本面质量为{fundamental_quality}，分数{fundamental_score:.1f}，"
                f"原始动作为{original_action}，因此长期信号降级等待。"
            ),
        ),
        "wait_sector_weak": (
            (
                f"Sector score is weak; original action was {original_action}, "
                "so medium or long-term entry is deferred."
            ),
            f"板块分数偏弱，原始动作为{original_action}，因此中期或长期信号降级等待。",
        ),
        "wait_insufficient_history": (
            "There is not enough price history for this horizon.",
            "当前历史数据不足，暂不生成有效入场信号。",
        ),
        "avoid_universe_filter": (
            "The ticker does not pass history or liquidity filters.",
            "该股票没有通过历史数据或流动性过滤。",
        ),
        "avoid_or_wait_downtrend": (
            "Trend or momentum is not strong enough.",
            "趋势或动量不足，暂时回避或等待。",
        ),
    }
    return explanations.get(
        action,
        ("No detailed action explanation available.", "暂无详细动作解释。"),
    )


def _plain_summary(
    spec: HorizonSpec,
    action: str,
    signal_score: float,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
) -> tuple[str, str]:
    price_plan = (
        f"entry around {_format_number(entry_price)}, stop around {_format_number(stop_loss)}, "
        f"target around {_format_number(take_profit)}"
    )
    price_plan_zh = (
        f"参考入场价约{_format_number(entry_price)}，止损约{_format_number(stop_loss)}，"
        f"目标价约{_format_number(take_profit)}"
    )
    prefix = f"{spec.label} ({spec.time_range}), score {signal_score:.1f}: "
    prefix_zh = f"{spec.zh_label}（{spec.time_range_zh}），分数{signal_score:.1f}："

    if action == "entry_breakout_confirmed":
        return (
            prefix + f"breakout is confirmed; {price_plan}.",
            prefix_zh + f"突破已经确认；{price_plan_zh}。",
        )
    if action == "entry_pullback_zone":
        return (
            prefix + f"price is in a pullback entry zone; {price_plan}.",
            prefix_zh + f"价格处在回调买点区域；{price_plan_zh}。",
        )
    if action == "watch_breakout_or_pullback":
        return (
            prefix + f"watch only; wait for breakout or pullback confirmation; {price_plan}.",
            prefix_zh + f"先观察，等待突破或回调确认；{price_plan_zh}。",
        )
    if action == "wait_overextended":
        return (
            prefix + f"wait; price is extended, so chasing is not preferred; {price_plan}.",
            prefix_zh + f"等待，价格偏高，不适合追高；{price_plan_zh}。",
        )
    if action == "wait_market_weak":
        return (
            prefix + "wait; market background is weak, so new entries are downgraded.",
            prefix_zh + "等待，大盘环境偏弱，新的入场信号被降级。",
        )
    if action == "wait_relative_weak":
        return (
            prefix + "wait; the stock is weak versus SPY/QQQ.",
            prefix_zh + "等待，该股票相对SPY/QQQ偏弱。",
        )
    if action == "wait_event_risk":
        return (
            prefix + "wait; earnings or event risk is too close for a fresh entry.",
            prefix_zh + "等待，财报或事件风险太近，不适合新开仓。",
        )
    if action == "wait_sentiment_risk":
        return (
            prefix + "wait; recent news or sentiment risk is elevated.",
            prefix_zh + "等待，近期新闻或情绪风险偏高，不适合新开仓。",
        )
    if action == "wait_analyst_weak":
        return (
            prefix + "wait; analyst expectations are weak or imply meaningful downside.",
            prefix_zh + "等待，分析师预期偏弱或暗示明显下行空间。",
        )
    if action == "wait_valuation_rich":
        return (
            prefix + "wait; valuation risk is elevated, so entry needs a better price.",
            prefix_zh + "等待，估值风险偏高，需要更合理的价格再考虑。",
        )
    if action == "wait_fundamental_weak":
        return (
            prefix + "wait; long-term company quality is too weak for a fresh entry.",
            prefix_zh + "等待，长期公司质量偏弱，不适合新开仓。",
        )
    if action == "wait_sector_weak":
        return (
            prefix + "wait; sector context is weak, so confirmation is not reliable enough.",
            prefix_zh + "等待，板块环境偏弱，当前确认度不够。",
        )
    if action == "wait_insufficient_history":
        return (
            prefix + "wait; there is not enough price history for this horizon.",
            prefix_zh + "等待，当前历史数据不足以支持该周期判断。",
        )
    if action == "avoid_universe_filter":
        return (
            prefix + "avoid; the ticker does not pass the current liquidity/history filter.",
            prefix_zh + "回避，该股票没有通过当前流动性或历史数据过滤。",
        )
    if action == "avoid_or_wait_downtrend":
        return (
            prefix + "avoid or wait; trend or momentum is not strong enough.",
            prefix_zh + "回避或等待，趋势或动量还不够强。",
        )
    return (
        prefix + f"watchlist only; {price_plan}.",
        prefix_zh + f"仅加入观察；{price_plan_zh}。",
    )


def _entry_plan(
    action: str,
    entry_type: str,
    latest_price: float,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    support: float,
    resistance: float,
    entry_distance_pct: float,
) -> dict[str, str]:
    distance_note, distance_note_zh = _entry_distance_note(entry_distance_pct)
    stop_text = _format_number(stop_loss)
    target_text = _format_number(take_profit)
    entry_text = _format_number(entry_price)
    support_text = _format_number(support)
    resistance_text = _format_number(resistance)
    latest_text = _format_number(latest_price)

    if action == "entry_breakout_confirmed":
        chase_status = "entry_allowed_if_price_holds"
        chase_status_zh = "价格守住突破区才可考虑"
        plan = (
            f"Breakout is confirmed. Entry reference is {entry_text}; stop is {stop_text}; "
            f"target is {target_text}. Avoid adding if price quickly falls back below the breakout area."
        )
        plan_zh = (
            f"突破已经确认。参考入场价{entry_text}，止损{stop_text}，目标{target_text}。"
            "如果价格很快跌回突破区域下方，不宜继续加仓。"
        )
    elif action == "entry_pullback_zone":
        chase_status = "pullback_entry_active"
        chase_status_zh = "回调买点已接近"
        plan = (
            f"Price is near the pullback entry zone around {entry_text}. Wait for price to hold "
            f"above support near {support_text}; stop is {stop_text}; target is {target_text}."
        )
        plan_zh = (
            f"价格接近约{entry_text}的回调买点。等待价格守住约{support_text}附近支撑；"
            f"止损{stop_text}，目标{target_text}。"
        )
    elif action == "watch_breakout_or_pullback" and entry_type == "stop_limit_breakout":
        chase_status = "do_not_chase_wait_for_breakout"
        chase_status_zh = "不要追高，等待突破触发"
        plan = (
            f"Current price is {latest_text}. Use a breakout trigger near {entry_text}, above "
            f"resistance around {resistance_text}. If triggered, stop is {stop_text} and target is {target_text}."
        )
        plan_zh = (
            f"当前价{latest_text}。等待价格突破，触发价约{entry_text}，压力位约{resistance_text}。"
            f"触发后止损{stop_text}，目标{target_text}。"
        )
    elif action == "wait_overextended":
        chase_status = "do_not_chase_wait_for_pullback"
        chase_status_zh = "不要追高，等待回调"
        plan = (
            f"Price is extended. Wait for a pullback toward {entry_text}; stop would be {stop_text} "
            f"and target would be {target_text} if the pullback stabilizes."
        )
        plan_zh = (
            f"价格偏高，不适合追。等待回调到约{entry_text}附近；如果企稳，"
            f"止损参考{stop_text}，目标参考{target_text}。"
        )
    elif action in {
        "wait_market_weak",
        "wait_relative_weak",
        "wait_event_risk",
        "wait_fundamental_weak",
        "wait_sector_weak",
    }:
        chase_status = "blocked_by_risk_filter"
        chase_status_zh = "被风险过滤阻挡"
        plan = (
            f"No fresh entry until the risk filter clears. If conditions improve, reassess entry near {entry_text}, "
            f"with stop near {stop_text} and target near {target_text}."
        )
        plan_zh = (
            f"风险过滤解除前不做新入场。条件改善后，再评估约{entry_text}附近的入场，"
            f"止损约{stop_text}，目标约{target_text}。"
        )
    elif action == "avoid_or_wait_downtrend":
        chase_status = "no_entry_trend_not_ready"
        chase_status_zh = "无入场，趋势未准备好"
        plan = (
            f"No entry plan yet. Wait for trend and momentum to improve before using resistance near "
            f"{resistance_text} or support near {support_text} as actionable levels."
        )
        plan_zh = (
            f"暂时没有可执行买点。等待趋势和动量改善后，再把约{resistance_text}压力位或"
            f"约{support_text}支撑位作为可操作位置。"
        )
    else:
        chase_status = "no_fresh_entry"
        chase_status_zh = "暂不新开仓"
        plan = (
            f"No clean entry trigger. Reassess if price structure improves around entry {entry_text}, "
            f"support {support_text}, or resistance {resistance_text}."
        )
        plan_zh = (
            f"暂时没有清晰入场触发。若价格结构在入场价{entry_text}、支撑{support_text}"
            f"或压力{resistance_text}附近改善，再重新评估。"
        )

    return {
        "entry_distance_note": distance_note,
        "entry_distance_note_zh": distance_note_zh,
        "chase_status": chase_status,
        "chase_status_zh": chase_status_zh,
        "entry_plan": plan,
        "entry_plan_zh": plan_zh,
    }


def _entry_distance_note(entry_distance_pct: float) -> tuple[str, str]:
    if not _is_finite(entry_distance_pct):
        return "Entry distance is unavailable.", "买点距离不可用。"
    distance = abs(entry_distance_pct)
    if distance <= 0.005:
        return (
            "Current price is very close to the planned entry.",
            "当前价非常接近计划买点。",
        )
    if entry_distance_pct > 0:
        return (
            f"Planned entry is {entry_distance_pct:.2%} above current price.",
            f"计划买点比当前价高{entry_distance_pct:.2%}。",
        )
    return (
        f"Current price is {distance:.2%} above the planned entry.",
        f"当前价比计划买点高{distance:.2%}。",
    )
