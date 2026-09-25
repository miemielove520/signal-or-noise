"""Building one validation event per (signal date, ticker, horizon), plus the neutral contexts and market-regime lookup used while replaying history."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from ..analysis import HORIZON_SPECS
from ..analysis.screening import _entry_evidence_profile
from ..real_data import normalize_ticker
from ..screening_config import ScreeningConfig
from ..universe import HistoricalUniverseMembership

from ._common import (
    _clamp,
    _format_optional_date,
    _is_finite,
    _optional_timestamp,
    _safe_float,
    _safe_int,
    _safe_optional_float,
    _split_reason_text,
)


def _event_row(
    ticker_prices: pd.DataFrame,
    signal_index: int,
    analysis_row: object,
    forward_windows: tuple[int, ...],
    max_forward_window: int,
    market_regime_lookup: dict[pd.Timestamp, dict[str, object]] | None = None,
    membership_record: dict[str, object] | None = None,
) -> dict[str, object]:
    signal_price = float(ticker_prices.loc[signal_index, "adj_close"])
    signal_date = pd.Timestamp(ticker_prices.loc[signal_index, "date"]).normalize()
    market_regime = _lookup_market_regime(signal_date, market_regime_lookup or {})
    delisted_date = _optional_timestamp((membership_record or {}).get("delisted_date"))
    delisting_return = _safe_optional_float((membership_record or {}).get("delisting_return"))
    row = {
        "date": ticker_prices.loc[signal_index, "date"],
        "ticker": analysis_row.ticker,
        "horizon": analysis_row.horizon,
        "horizon_zh_label": getattr(
            analysis_row,
            "horizon_zh_label",
            HORIZON_SPECS[analysis_row.horizon].zh_label,
        ),
        "action": analysis_row.action,
        "screening_action": analysis_row.screening_action,
        "screening_profile": getattr(analysis_row, "screening_profile", "default"),
        "screening_profile_zh": getattr(analysis_row, "screening_profile_zh", "默认规则"),
        "quality_gate_passed": bool(analysis_row.quality_gate_passed),
        "watchlist_status": analysis_row.watchlist_status,
        "validation_bucket": _validation_bucket(analysis_row),
        "high_probability_score": float(analysis_row.high_probability_score),
        "calibrated_win_probability": float(analysis_row.calibrated_win_probability)
        if _is_finite(analysis_row.calibrated_win_probability)
        else np.nan,
        "calibrated_probability_level": analysis_row.calibrated_probability_level,
        "calibrated_probability_level_zh": analysis_row.calibrated_probability_level_zh,
        "calibrated_probability_confidence": float(
            analysis_row.calibrated_probability_confidence
        )
        if _is_finite(analysis_row.calibrated_probability_confidence)
        else np.nan,
        "calibrated_probability_confidence_level": (
            analysis_row.calibrated_probability_confidence_level
        ),
        "calibrated_probability_confidence_level_zh": (
            analysis_row.calibrated_probability_confidence_level_zh
        ),
        "signal_score": float(analysis_row.signal_score),
        "confidence_score": float(analysis_row.confidence_score),
        "market_score": float(analysis_row.market_score),
        "relative_strength_score": float(analysis_row.relative_strength_score),
        "event_risk_level": analysis_row.event_risk_level,
        "event_window": analysis_row.event_window,
        "event_block_new_entries": bool(analysis_row.event_block_new_entries),
        "event_cooldown_active": bool(analysis_row.event_cooldown_active),
        "sentiment_score": float(analysis_row.sentiment_score),
        "sentiment_risk_level": analysis_row.sentiment_risk_level,
        "sentiment_block_new_entries": bool(analysis_row.sentiment_block_new_entries),
        "analyst_score": float(analysis_row.analyst_score),
        "analyst_risk_level": analysis_row.analyst_risk_level,
        "analyst_block_new_entries": bool(analysis_row.analyst_block_new_entries),
        "valuation_score": float(analysis_row.valuation_score),
        "valuation_risk_level": analysis_row.valuation_risk_level,
        "valuation_block_new_entries": bool(analysis_row.valuation_block_new_entries),
        "screening_backtest_entry_type": analysis_row.screening_backtest_entry_type,
        "screening_backtest_win_rate": float(analysis_row.screening_backtest_win_rate)
        if _is_finite(analysis_row.screening_backtest_win_rate)
        else np.nan,
        "screening_backtest_average_return": float(analysis_row.screening_backtest_average_return)
        if _is_finite(analysis_row.screening_backtest_average_return)
        else np.nan,
        "screening_backtest_trade_count": int(analysis_row.screening_backtest_trade_count),
        "sample_confidence_level": analysis_row.sample_confidence_level,
        "sample_confidence_level_zh": analysis_row.sample_confidence_level_zh,
        "evidence_strength": analysis_row.evidence_strength,
        "evidence_strength_zh": analysis_row.evidence_strength_zh,
        "evidence_note": analysis_row.evidence_note,
        "evidence_note_zh": analysis_row.evidence_note_zh,
        "liquidity_filter_passed": bool(analysis_row.liquidity_filter_passed),
        "liquidity_filter_reason": analysis_row.liquidity_filter_reason,
        "liquidity_filter_reason_zh": analysis_row.liquidity_filter_reason_zh,
        "quality_gate_fail_reasons": analysis_row.quality_gate_fail_reasons,
        "quality_gate_fail_reasons_zh": analysis_row.quality_gate_fail_reasons_zh,
        "watchlist_missing_items": analysis_row.watchlist_missing_items,
        "watchlist_missing_items_zh": analysis_row.watchlist_missing_items_zh,
        "validation_market_regime": market_regime["validation_market_regime"],
        "validation_market_regime_zh": market_regime["validation_market_regime_zh"],
        "validation_spy_trend_score": market_regime["validation_spy_trend_score"],
        "validation_qqq_trend_score": market_regime["validation_qqq_trend_score"],
        "validation_benchmark_60d_return": market_regime[
            "validation_benchmark_60d_return"
        ],
        "validation_benchmark_20d_volatility": market_regime[
            "validation_benchmark_20d_volatility"
        ],
        "point_in_time_universe_member": membership_record is not None,
        "universe_start_date": _format_optional_date((membership_record or {}).get("start_date")),
        "universe_end_date": _format_optional_date((membership_record or {}).get("end_date")),
        "delisted_date": _format_optional_date(delisted_date),
        "delisting_return_used": np.nan,
        "delisted_during_forward_window": False,
    }
    for window in forward_windows:
        future_index = signal_index + window
        delisting_adjusted = False
        if future_index < len(ticker_prices):
            future_price = float(ticker_prices.loc[future_index, "adj_close"])
        else:
            last_price = float(ticker_prices["adj_close"].iloc[-1])
            if delisting_return is not None:
                future_price = last_price * (1.0 + delisting_return)
                row["delisting_return_used"] = delisting_return
            else:
                future_price = last_price
            delisting_adjusted = True
        row[f"forward_return_{window}d_delisting_adjusted"] = delisting_adjusted
        row["delisted_during_forward_window"] = (
            bool(row["delisted_during_forward_window"]) or delisting_adjusted
        )
        row[f"forward_return_{window}d"] = future_price / signal_price - 1.0
    future_prices = ticker_prices.loc[
        signal_index : min(signal_index + max_forward_window, len(ticker_prices) - 1),
        "adj_close",
    ].astype(float)
    row["max_drawdown_after_signal"] = float((future_prices / signal_price - 1.0).min())
    return row


def _membership_record_for_signal(
    ticker: str,
    signal_date: object,
    universe_membership: HistoricalUniverseMembership | None,
) -> dict[str, object] | None:
    if universe_membership is None:
        return None
    return universe_membership.active_record(ticker, signal_date)


def _ticker_allows_truncated_forward(
    ticker: str,
    universe_membership: HistoricalUniverseMembership | None,
) -> bool:
    if universe_membership is None or universe_membership.records.empty:
        return False
    clean = normalize_ticker(str(ticker))
    records = universe_membership.records[universe_membership.records["ticker"] == clean]
    if records.empty:
        return False
    if "delisted_date" in records.columns and records["delisted_date"].notna().any():
        return True
    if "status" in records.columns:
        return bool(records["status"].fillna("").astype(str).str.lower().str.contains("delist").any())
    return False


def _profile_by_name(config: ScreeningConfig, profile_name: str | None):
    if not profile_name:
        return None
    for profile in config.profiles:
        if profile.name == profile_name:
            return profile
    raise ValueError(f"Unknown screening profile: {profile_name}")


def _apply_pooled_entry_backtest(events: pd.DataFrame, config: ScreeningConfig) -> pd.DataFrame:
    if events.empty:
        return events
    required = {
        "screening_profile",
        "horizon",
        "screening_backtest_entry_type",
        "screening_backtest_trade_count",
        "screening_backtest_win_rate",
        "screening_backtest_average_return",
        "quality_gate_fail_reasons",
        "quality_gate_fail_reasons_zh",
        "validation_bucket",
    }
    if not required.issubset(events.columns):
        return events

    result = events.copy()
    result["original_quality_gate_passed"] = result["quality_gate_passed"]
    result["original_validation_bucket"] = result["validation_bucket"]
    result["pooled_entry_backtest_used"] = False
    result["pooled_screening_backtest_trade_count"] = result["screening_backtest_trade_count"]
    result["pooled_screening_backtest_win_rate"] = result["screening_backtest_win_rate"]
    result["pooled_screening_backtest_average_return"] = result["screening_backtest_average_return"]
    result["pooled_quality_gate_fail_reasons"] = result["quality_gate_fail_reasons"]
    result["pooled_quality_gate_fail_reasons_zh"] = result["quality_gate_fail_reasons_zh"]

    thresholds_by_profile = {
        "default": config.default_thresholds,
        **{profile.name: profile.thresholds for profile in config.profiles},
    }
    group_columns = ["screening_profile", "horizon", "screening_backtest_entry_type"]
    pooled_metrics = {
        keys: _pooled_backtest_metrics(group)
        for keys, group in result.groupby(group_columns, dropna=False)
    }

    for index, row in result.iterrows():
        profile_name = str(row["screening_profile"])
        thresholds = thresholds_by_profile.get(profile_name, config.default_thresholds)
        original_bucket = str(row["validation_bucket"])
        own_count = _safe_int(row["screening_backtest_trade_count"])
        keys = (
            row["screening_profile"],
            row["horizon"],
            row["screening_backtest_entry_type"],
        )
        pooled = pooled_metrics.get(keys, _empty_pooled_metrics())
        if own_count >= thresholds.backtest_sample_min:
            continue
        if int(pooled["trade_count"]) < thresholds.backtest_sample_min:
            continue
        if original_bucket != "near_watchlist":
            continue

        fail_reasons = _split_reason_text(row["quality_gate_fail_reasons"])
        fail_reasons_zh = _split_reason_text(row["quality_gate_fail_reasons_zh"])
        fail_reasons = _remove_entry_backtest_reasons(fail_reasons)
        fail_reasons_zh = _remove_entry_backtest_reasons_zh(fail_reasons_zh)

        if not _is_finite(pooled["win_rate"]) or float(pooled["win_rate"]) < thresholds.backtest_win_rate_min:
            fail_reasons.append("pooled entry backtest win rate too low")
            fail_reasons_zh.append("同类聚合买点回测胜率不足")
        if (
            not _is_finite(pooled["average_return"])
            or float(pooled["average_return"]) <= thresholds.backtest_average_return_min
        ):
            fail_reasons.append("pooled entry backtest average return not positive")
            fail_reasons_zh.append("同类聚合买点回测平均收益不是正数")

        result.at[index, "pooled_entry_backtest_used"] = True
        result.at[index, "pooled_screening_backtest_trade_count"] = int(pooled["trade_count"])
        result.at[index, "pooled_screening_backtest_win_rate"] = float(pooled["win_rate"])
        result.at[index, "pooled_screening_backtest_average_return"] = float(pooled["average_return"])

        if fail_reasons:
            result.at[index, "pooled_quality_gate_fail_reasons"] = "; ".join(fail_reasons)
            result.at[index, "pooled_quality_gate_fail_reasons_zh"] = "；".join(fail_reasons_zh)
        else:
            evidence = _entry_evidence_profile(
                trade_count=int(pooled["trade_count"]),
                win_rate=float(pooled["win_rate"]),
                average_return=float(pooled["average_return"]),
            )
            result.at[index, "pooled_quality_gate_fail_reasons"] = "all strict quality gates passed with pooled entry backtest"
            result.at[index, "pooled_quality_gate_fail_reasons_zh"] = "使用同类聚合买点回测后，所有严格质量门槛通过"
            result.at[index, "quality_gate_passed"] = True
            result.at[index, "validation_bucket"] = "high_probability"
            result.at[index, "quality_gate_fail_reasons"] = result.at[index, "pooled_quality_gate_fail_reasons"]
            result.at[index, "quality_gate_fail_reasons_zh"] = result.at[index, "pooled_quality_gate_fail_reasons_zh"]
            result.at[index, "screening_backtest_trade_count"] = int(pooled["trade_count"])
            result.at[index, "screening_backtest_win_rate"] = float(pooled["win_rate"])
            result.at[index, "screening_backtest_average_return"] = float(pooled["average_return"])
            for key, value in evidence.items():
                result.at[index, key] = value

    return result


def _pooled_backtest_metrics(group: pd.DataFrame) -> dict[str, float | int]:
    valid = group[
        (pd.to_numeric(group["screening_backtest_trade_count"], errors="coerce") > 0)
        & pd.to_numeric(group["screening_backtest_win_rate"], errors="coerce").notna()
        & pd.to_numeric(group["screening_backtest_average_return"], errors="coerce").notna()
    ].copy()
    if valid.empty:
        return _empty_pooled_metrics()
    counts = pd.to_numeric(valid["screening_backtest_trade_count"], errors="coerce").astype(float)
    win_rates = pd.to_numeric(valid["screening_backtest_win_rate"], errors="coerce").astype(float)
    average_returns = pd.to_numeric(valid["screening_backtest_average_return"], errors="coerce").astype(float)
    total_count = int(counts.sum())
    if total_count <= 0:
        return _empty_pooled_metrics()
    return {
        "trade_count": total_count,
        "win_rate": float((win_rates * counts).sum() / total_count),
        "average_return": float((average_returns * counts).sum() / total_count),
    }


def _empty_pooled_metrics() -> dict[str, float | int]:
    return {"trade_count": 0, "win_rate": np.nan, "average_return": np.nan}


def _remove_entry_backtest_reasons(reasons: list[str]) -> list[str]:
    blocked = (
        "backtest sample too small",
        "backtest win rate too low",
        "backtest average return not positive",
    )
    return [reason for reason in reasons if not any(text in reason for text in blocked)]


def _remove_entry_backtest_reasons_zh(reasons: list[str]) -> list[str]:
    blocked = (
        "回测样本不足",
        "回测胜率不足",
        "回测平均收益不是正数",
    )
    return [reason for reason in reasons if not any(text in reason for text in blocked)]


def _validation_bucket(row: object) -> str:
    if bool(row.quality_gate_passed):
        return "high_probability"
    status = str(row.watchlist_status)
    if status == "close_but_not_ready":
        return "near_watchlist"
    if status == "early_watch":
        return "early_watchlist"
    return "filtered_out"


def _target_window(forward_windows: tuple[int, ...]) -> int:
    return 20 if 20 in forward_windows else sorted(forward_windows)[len(forward_windows) // 2]


def build_validation_market_regime_lookup(prices: pd.DataFrame) -> dict[pd.Timestamp, dict[str, object]]:
    if prices.empty:
        return {}
    frame = _normalize_prices(prices)
    spy = frame[frame["ticker"] == "SPY"][["date", "adj_close"]].copy()
    qqq = frame[frame["ticker"] == "QQQ"][["date", "adj_close"]].copy()
    if spy.empty and qqq.empty:
        return {}

    benchmark = spy if not spy.empty else qqq
    benchmark = benchmark.sort_values("date").drop_duplicates("date", keep="last")
    benchmark["benchmark_return"] = benchmark["adj_close"].pct_change()
    benchmark["benchmark_60d_return"] = benchmark["adj_close"].pct_change(60)
    benchmark["benchmark_20d_volatility"] = (
        benchmark["benchmark_return"].rolling(20).std() * np.sqrt(252)
    )
    benchmark["benchmark_ma_200"] = benchmark["adj_close"].rolling(200).mean()

    if qqq.empty:
        qqq_trend = pd.DataFrame(columns=["date", "qqq_trend_score"])
    else:
        qqq_trend = _benchmark_trend_frame(qqq, "qqq_trend_score")
    spy_trend = _benchmark_trend_frame(spy, "spy_trend_score") if not spy.empty else pd.DataFrame(
        columns=["date", "spy_trend_score"]
    )
    regime_frame = benchmark.merge(spy_trend, on="date", how="left").merge(
        qqq_trend,
        on="date",
        how="left",
    )

    lookup: dict[pd.Timestamp, dict[str, object]] = {}
    for row in regime_frame.itertuples(index=False):
        date = pd.Timestamp(row.date).normalize()
        regime, regime_zh = _validation_market_regime_label(
            close=float(row.adj_close),
            ma_200=_safe_float(getattr(row, "benchmark_ma_200", np.nan)),
            return_60d=_safe_float(getattr(row, "benchmark_60d_return", np.nan)),
            volatility_20d=_safe_float(getattr(row, "benchmark_20d_volatility", np.nan)),
        )
        lookup[date] = {
            "validation_market_regime": regime,
            "validation_market_regime_zh": regime_zh,
            "validation_spy_trend_score": _safe_float(
                getattr(row, "spy_trend_score", np.nan)
            ),
            "validation_qqq_trend_score": _safe_float(
                getattr(row, "qqq_trend_score", np.nan)
            ),
            "validation_benchmark_60d_return": _safe_float(
                getattr(row, "benchmark_60d_return", np.nan)
            ),
            "validation_benchmark_20d_volatility": _safe_float(
                getattr(row, "benchmark_20d_volatility", np.nan)
            ),
        }
    return lookup


def _benchmark_trend_frame(prices: pd.DataFrame, output_column: str) -> pd.DataFrame:
    frame = prices[["date", "adj_close"]].copy().sort_values("date")
    frame = frame.drop_duplicates("date", keep="last")
    frame["return_60d"] = frame["adj_close"].pct_change(60)
    frame["ma_200"] = frame["adj_close"].rolling(200).mean()
    scores: list[float] = []
    for row in frame.itertuples(index=False):
        score = 50.0
        close = _safe_float(row.adj_close)
        ma_200 = _safe_float(row.ma_200)
        return_60d = _safe_float(row.return_60d)
        if _is_finite(close) and _is_finite(ma_200):
            score += 20.0 if close >= ma_200 else -20.0
        if _is_finite(return_60d):
            score += max(-20.0, min(20.0, return_60d * 200.0))
        scores.append(round(float(_clamp(score, 0.0, 100.0)), 2))
    return frame[["date"]].assign(**{output_column: scores})


def _validation_market_regime_label(
    close: float,
    ma_200: float,
    return_60d: float,
    volatility_20d: float,
) -> tuple[str, str]:
    if not _is_finite(close) or not _is_finite(ma_200):
        return "unknown", "未知市场状态"
    if _is_finite(volatility_20d) and volatility_20d >= 0.28:
        return "high_volatility", "高波动市场"
    if close >= ma_200 and _is_finite(return_60d) and return_60d > 0:
        return "bull_uptrend", "牛市上升"
    if close < ma_200 and _is_finite(return_60d) and return_60d < 0:
        return "bear_downtrend", "熊市下跌"
    return "sideways_mixed", "震荡分化"


def _lookup_market_regime(
    signal_date: pd.Timestamp,
    lookup: dict[pd.Timestamp, dict[str, object]],
) -> dict[str, object]:
    default = {
        "validation_market_regime": "unknown",
        "validation_market_regime_zh": "未知市场状态",
        "validation_spy_trend_score": np.nan,
        "validation_qqq_trend_score": np.nan,
        "validation_benchmark_60d_return": np.nan,
        "validation_benchmark_20d_volatility": np.nan,
    }
    if not lookup:
        return default
    date = pd.Timestamp(signal_date).normalize()
    if date in lookup:
        return lookup[date]
    available_dates = [candidate for candidate in lookup if candidate <= date]
    if not available_dates:
        return default
    return lookup[max(available_dates)]


def _normalize_prices(prices: pd.DataFrame) -> pd.DataFrame:
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=False)
    frame["ticker"] = frame["ticker"].astype("string").str.upper().str.strip()
    numeric_columns = ["open", "high", "low", "close", "adj_close", "volume"]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["date", "ticker", "adj_close"]).sort_values(["ticker", "date"]).reset_index(drop=True)


def _validation_market_context() -> SimpleNamespace:
    return SimpleNamespace(
        market_score=60.0,
        market_status="neutral_validation",
        note="Validation uses neutral market context.",
        note_zh="验证使用中性大盘环境。",
        warning="",
    )


def _validation_relative_strength_contexts(horizons: tuple[str, ...]) -> dict[str, SimpleNamespace]:
    return {
        horizon: SimpleNamespace(
            score=50.0,
            note="Validation uses neutral benchmark-relative context.",
            note_zh="验证使用中性相对强弱环境。",
            warning="",
            vs_spy_return=np.nan,
            vs_qqq_return=np.nan,
        )
        for horizon in horizons
    }


def _validation_event_context() -> SimpleNamespace:
    return SimpleNamespace(
        event_risk_level="low",
        event_risk_level_zh="低",
        event_risk_score=20.0,
        next_earnings_date="validation_neutral",
        days_until_earnings=999,
        last_earnings_date="validation_neutral",
        days_since_earnings=999,
        event_window="normal",
        event_window_zh="正常",
        event_block_new_entries=False,
        event_cooldown_active=False,
        event_risk_note="Validation uses neutral event-risk context.",
        event_risk_note_zh="验证使用中性事件风险。",
        event_risk_warning="",
    )


def _validation_fundamental_context(ticker: str) -> SimpleNamespace:
    return SimpleNamespace(
        ticker=ticker,
        fundamental_score=60.0,
        fundamental_quality="neutral",
        fundamental_quality_zh="中性",
        fundamental_note="Validation uses neutral fundamental context.",
        fundamental_note_zh="验证使用中性基本面环境。",
        fundamental_warning="",
        data_coverage=1.0,
        revenue_growth=np.nan,
        earnings_growth=np.nan,
        profit_margin=np.nan,
        return_on_equity=np.nan,
        free_cash_flow=np.nan,
        forward_pe=np.nan,
        peg_ratio=np.nan,
        debt_to_equity=np.nan,
    )


def _validation_sentiment_context(ticker: str) -> SimpleNamespace:
    return SimpleNamespace(
        ticker=ticker,
        sentiment_score=50.0,
        sentiment_label="neutral",
        sentiment_label_zh="中性",
        sentiment_risk_level="low",
        sentiment_risk_level_zh="低",
        sentiment_block_new_entries=False,
        sentiment_positive_count=0,
        sentiment_negative_count=0,
        sentiment_high_risk_count=0,
        sentiment_titles_used=0,
        sentiment_note="Validation uses neutral news-sentiment context.",
        sentiment_note_zh="验证使用中性新闻情绪环境。",
        sentiment_warning="",
    )


def _validation_analyst_context(ticker: str) -> SimpleNamespace:
    return SimpleNamespace(
        ticker=ticker,
        analyst_score=50.0,
        analyst_label="neutral",
        analyst_label_zh="中性",
        analyst_risk_level="low",
        analyst_risk_level_zh="低",
        analyst_block_new_entries=False,
        analyst_upside=np.nan,
        recommendation_mean=np.nan,
        recommendation_key="validation_neutral",
        number_of_analysts=np.nan,
        target_mean_price=np.nan,
        analyst_note="Validation uses neutral analyst-expectation context.",
        analyst_note_zh="验证使用中性分析师预期环境。",
        analyst_warning="",
        data_coverage=1.0,
    )


def _validation_valuation_context(ticker: str) -> SimpleNamespace:
    return SimpleNamespace(
        ticker=ticker,
        valuation_score=50.0,
        valuation_label="reasonable",
        valuation_label_zh="合理",
        valuation_risk_level="low",
        valuation_risk_level_zh="低",
        valuation_block_new_entries=False,
        valuation_forward_pe=np.nan,
        valuation_peg_ratio=np.nan,
        valuation_growth_reference=np.nan,
        valuation_profit_margin=np.nan,
        valuation_note="Validation uses neutral valuation context.",
        valuation_note_zh="验证使用中性估值环境。",
        valuation_warning="",
        data_coverage=1.0,
    )


def _validation_sector_context() -> SimpleNamespace:
    return SimpleNamespace(
        sector="validation_neutral",
        industry="validation_neutral",
        sector_etf="VALIDATION",
        sector_score=50.0,
        sector_status="neutral",
        sector_note="Validation uses neutral sector context.",
        sector_note_zh="验证使用中性板块环境。",
        sector_warning="",
        sector_trend_score=50.0,
        sector_relative_strength=np.nan,
    )
