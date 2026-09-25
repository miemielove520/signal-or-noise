"""Aggregate statistics over validation events: buckets, tickers, profiles, segments, regimes and probability calibration."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from ..real_data import normalize_ticker

from ._common import (
    _clamp,
    _format_percent,
    _is_finite,
    _numeric_mean,
    _safe_float,
    _safe_int,
)
from .config import (
    DEFAULT_FORWARD_WINDOWS,
)
from .events import (
    _target_window,
)


def summarize_walk_forward_events(
    events: pd.DataFrame,
    forward_windows: tuple[int, ...] = DEFAULT_FORWARD_WINDOWS,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["bucket", "sample_count"])

    rows: list[dict[str, object]] = []
    for bucket, group in events.groupby("validation_bucket", dropna=False):
        row: dict[str, object] = {"bucket": bucket, "sample_count": len(group)}
        for window in forward_windows:
            returns = group[f"forward_return_{window}d"].dropna()
            row[f"win_rate_{window}d"] = float((returns > 0).mean()) if len(returns) else np.nan
            row[f"avg_return_{window}d"] = float(returns.mean()) if len(returns) else np.nan
            row[f"median_return_{window}d"] = float(returns.median()) if len(returns) else np.nan
            row[f"worst_return_{window}d"] = float(returns.min()) if len(returns) else np.nan
        drawdowns = group["max_drawdown_after_signal"].dropna()
        row["avg_max_drawdown_after_signal"] = float(drawdowns.mean()) if len(drawdowns) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("bucket").reset_index(drop=True)


def build_ticker_validation_ranking(
    events: pd.DataFrame,
    target_window: int = 20,
) -> pd.DataFrame:
    columns = [
        "rank",
        "ticker",
        "ticker_decision",
        "ticker_decision_zh",
        "ticker_ranking_score",
        "sample_count",
        "high_probability_sample_count",
        "watchlist_or_better_sample_count",
        f"win_rate_{target_window}d",
        f"avg_return_{target_window}d",
        f"median_return_{target_window}d",
        f"worst_return_{target_window}d",
        "avg_max_drawdown_after_signal",
        "quality_gate_pass_rate",
        "avg_signal_score",
        "avg_confidence_score",
        "avg_high_probability_score",
        "avg_calibrated_win_probability",
        "ticker_reason",
        "ticker_reason_zh",
    ]
    return_column = f"forward_return_{target_window}d"
    required = {"ticker", "validation_bucket", return_column}
    if events.empty or not required.issubset(events.columns):
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for ticker, group in events.groupby("ticker", dropna=False):
        returns = pd.to_numeric(group[return_column], errors="coerce").dropna()
        sample_count = len(returns)
        high_probability_sample_count = int(
            (group["validation_bucket"] == "high_probability").sum()
        )
        watchlist_or_better_sample_count = int(
            group["validation_bucket"].isin(["high_probability", "near_watchlist"]).sum()
        )
        avg_drawdown = _numeric_mean(group, "max_drawdown_after_signal")
        win_rate = float((returns > 0).mean()) if sample_count else np.nan
        avg_return = float(returns.mean()) if sample_count else np.nan
        row = {
            "rank": 0,
            "ticker": str(ticker),
            "_target_window": target_window,
            "ticker_decision": "",
            "ticker_decision_zh": "",
            "ticker_ranking_score": 0.0,
            "sample_count": sample_count,
            "high_probability_sample_count": high_probability_sample_count,
            "watchlist_or_better_sample_count": watchlist_or_better_sample_count,
            f"win_rate_{target_window}d": win_rate,
            f"avg_return_{target_window}d": avg_return,
            f"median_return_{target_window}d": float(returns.median()) if sample_count else np.nan,
            f"worst_return_{target_window}d": float(returns.min()) if sample_count else np.nan,
            "avg_max_drawdown_after_signal": avg_drawdown,
            "quality_gate_pass_rate": _numeric_mean(group, "quality_gate_passed"),
            "avg_signal_score": _numeric_mean(group, "signal_score"),
            "avg_confidence_score": _numeric_mean(group, "confidence_score"),
            "avg_high_probability_score": _numeric_mean(group, "high_probability_score"),
            "avg_calibrated_win_probability": _numeric_mean(
                group,
                "calibrated_win_probability",
            ),
        }
        score = _ticker_ranking_score(row)
        decision, decision_zh = _ticker_ranking_decision(row, score)
        reason, reason_zh = _ticker_ranking_reason(row, score, target_window)
        row["ticker_ranking_score"] = score
        row["ticker_decision"] = decision
        row["ticker_decision_zh"] = decision_zh
        row["ticker_reason"] = reason
        row["ticker_reason_zh"] = reason_zh
        rows.append(row)

    ranking = pd.DataFrame(rows)
    decision_rank = {
        "priority_candidate": 0,
        "watchlist_candidate": 1,
        "research_only": 2,
        "need_more_samples": 3,
        "deprioritize": 4,
    }
    ranking["_decision_rank"] = ranking["ticker_decision"].map(decision_rank).fillna(9)
    ranking = (
        ranking.sort_values(
            ["_decision_rank", "ticker_ranking_score", "sample_count"],
            ascending=[True, False, False],
            na_position="last",
        )
        .drop(columns=["_decision_rank"])
        .reset_index(drop=True)
    )
    ranking["rank"] = range(1, len(ranking) + 1)
    return ranking[columns]


def _ticker_ranking_score(row: dict[str, object]) -> float:
    target_window = _safe_int(row.get("_target_window")) or 20
    sample_count = _safe_int(row.get("sample_count"))
    high_probability_sample_count = _safe_int(row.get("high_probability_sample_count"))
    watchlist_or_better_sample_count = _safe_int(row.get("watchlist_or_better_sample_count"))
    win_rate = _safe_float(row.get(f"win_rate_{target_window}d"))
    avg_return = _safe_float(row.get(f"avg_return_{target_window}d"))
    avg_drawdown = _safe_float(row.get("avg_max_drawdown_after_signal"))
    quality_gate_pass_rate = _safe_float(row.get("quality_gate_pass_rate"))
    avg_signal_score = _safe_float(row.get("avg_signal_score"))
    avg_confidence_score = _safe_float(row.get("avg_confidence_score"))
    avg_high_probability_score = _safe_float(row.get("avg_high_probability_score"))
    avg_calibrated_win_probability = _safe_float(row.get("avg_calibrated_win_probability"))

    if sample_count <= 0:
        return 0.0

    score = 35.0
    score += min(sample_count / 20.0, 1.0) * 10.0
    score += min(high_probability_sample_count / 5.0, 1.0) * 10.0
    score += min(watchlist_or_better_sample_count / 10.0, 1.0) * 5.0

    if _is_finite(win_rate):
        score += (win_rate - 0.50) * 55.0
    if _is_finite(avg_return):
        score += avg_return * 180.0
    if _is_finite(avg_drawdown):
        score -= min(abs(avg_drawdown), 0.30) * 35.0
    if _is_finite(quality_gate_pass_rate):
        score += quality_gate_pass_rate * 8.0
    if _is_finite(avg_signal_score):
        score += (avg_signal_score - 60.0) * 0.15
    if _is_finite(avg_confidence_score):
        score += (avg_confidence_score - 60.0) * 0.12
    if _is_finite(avg_high_probability_score):
        score += (avg_high_probability_score - 60.0) * 0.12
    if _is_finite(avg_calibrated_win_probability):
        score += (avg_calibrated_win_probability - 0.50) * 25.0

    if sample_count < 5:
        score = min(score, 50.0)

    return round(_clamp(score, 0.0, 100.0), 2)


def _ticker_ranking_decision(
    row: dict[str, object],
    score: float,
) -> tuple[str, str]:
    target_window = _safe_int(row.get("_target_window")) or 20
    sample_count = _safe_int(row.get("sample_count"))
    high_probability_sample_count = _safe_int(row.get("high_probability_sample_count"))
    win_rate = _safe_float(row.get(f"win_rate_{target_window}d"))
    avg_return = _safe_float(row.get(f"avg_return_{target_window}d"))

    if sample_count < 5:
        return "need_more_samples", "样本不足"
    if (
        score >= 72
        and high_probability_sample_count >= 3
        and _is_finite(win_rate)
        and win_rate >= 0.55
        and _is_finite(avg_return)
        and avg_return > 0
    ):
        return "priority_candidate", "优先候选"
    if score >= 58 and _is_finite(avg_return) and avg_return > 0:
        return "watchlist_candidate", "观察候选"
    if score >= 45:
        return "research_only", "仅研究"
    return "deprioritize", "降低优先级"


def _ticker_ranking_reason(
    row: dict[str, object],
    score: float,
    target_window: int,
) -> tuple[str, str]:
    sample_count = _safe_int(row.get("sample_count"))
    high_probability_sample_count = _safe_int(row.get("high_probability_sample_count"))
    win_rate_text = _format_percent(row.get(f"win_rate_{target_window}d"))
    avg_return_text = _format_percent(row.get(f"avg_return_{target_window}d"))
    drawdown_text = _format_percent(row.get("avg_max_drawdown_after_signal"))
    reason = (
        f"Samples={sample_count}, high-probability samples={high_probability_sample_count}, "
        f"{target_window}d win rate={win_rate_text}, avg return={avg_return_text}, "
        f"avg drawdown={drawdown_text}, ranking score={score:.2f}."
    )
    reason_zh = (
        f"样本={sample_count}，高概率样本={high_probability_sample_count}，"
        f"{target_window}日胜率={win_rate_text}，平均收益={avg_return_text}，"
        f"平均回撤={drawdown_text}，排名分数={score:.2f}。"
    )
    return reason, reason_zh


def build_sample_sufficiency_guidance(
    events: pd.DataFrame,
    tickers: Iterable[str],
    ticker_ranking: pd.DataFrame | None = None,
    step_days: int = 20,
    min_history_days: int = 170,
    min_total_samples: int | None = None,
    min_ticker_samples: int = 10,
) -> pd.DataFrame:
    columns = [
        "scope",
        "ticker",
        "sample_count",
        "target_sample_count",
        "sample_status",
        "sample_status_zh",
        "guidance_action",
        "guidance_action_zh",
        "current_step_days",
        "suggested_step_days",
        "current_min_history_days",
        "suggested_min_history_days",
        "suggested_period",
        "suggested_command_hint",
        "suggested_command_hint_zh",
        "reason",
        "reason_zh",
    ]
    normalized_tickers = [normalize_ticker(ticker) for ticker in tickers if str(ticker).strip()]
    target_total = min_total_samples or max(30, len(normalized_tickers) * 5)
    ranking = ticker_ranking if ticker_ranking is not None else build_ticker_validation_ranking(events)

    rows: list[dict[str, object]] = []
    total_sample_count = int(len(events))
    insufficient_ticker_count = 0
    if not ranking.empty and "sample_count" in ranking.columns:
        insufficient_ticker_count = int((ranking["sample_count"] < min_ticker_samples).sum())
    missing_tickers = set(normalized_tickers)
    if not ranking.empty and "ticker" in ranking.columns:
        missing_tickers -= set(ranking["ticker"].astype(str).str.upper())
    insufficient_ticker_count += len(missing_tickers)

    rows.append(
        _sample_guidance_row(
            scope="overall",
            ticker="ALL",
            sample_count=total_sample_count,
            target_sample_count=target_total,
            step_days=step_days,
            min_history_days=min_history_days,
            reason_suffix=(
                f"{insufficient_ticker_count} tickers are below the ticker-level sample target."
                if insufficient_ticker_count
                else "Ticker-level sample coverage is acceptable."
            ),
            reason_suffix_zh=(
                f"{insufficient_ticker_count} 只股票低于个股样本目标。"
                if insufficient_ticker_count
                else "个股层面的样本覆盖可以接受。"
            ),
        )
    )

    if not ranking.empty:
        for row in ranking.itertuples(index=False):
            sample_count = _safe_int(getattr(row, "sample_count", 0))
            if sample_count >= min_ticker_samples:
                continue
            rows.append(
                _sample_guidance_row(
                    scope="ticker",
                    ticker=str(getattr(row, "ticker", "")),
                    sample_count=sample_count,
                    target_sample_count=min_ticker_samples,
                    step_days=step_days,
                    min_history_days=min_history_days,
                    reason_suffix="Ticker ranking should not be trusted until more signals are collected.",
                    reason_suffix_zh="收集更多信号前，不应过度信任这只股票的排名。",
                )
            )

    for ticker in sorted(missing_tickers):
        rows.append(
            _sample_guidance_row(
                scope="ticker",
                ticker=ticker,
                sample_count=0,
                target_sample_count=min_ticker_samples,
                step_days=step_days,
                min_history_days=min_history_days,
                reason_suffix="No validation signals were generated for this ticker.",
                reason_suffix_zh="这只股票没有生成验证信号。",
            )
        )

    return pd.DataFrame(rows, columns=columns)


def _sample_guidance_row(
    scope: str,
    ticker: str,
    sample_count: int,
    target_sample_count: int,
    step_days: int,
    min_history_days: int,
    reason_suffix: str,
    reason_suffix_zh: str,
) -> dict[str, object]:
    status, status_zh = _sample_status(sample_count, target_sample_count)
    suggested_step_days = _suggested_step_days(step_days, sample_count, target_sample_count)
    suggested_min_history_days = _suggested_min_history_days(min_history_days, sample_count, target_sample_count)
    suggested_period = _suggested_period(sample_count, target_sample_count)
    if sample_count >= target_sample_count:
        action = "keep_settings"
        action_zh = "保持当前设置"
    elif sample_count == 0:
        action = "expand_history_and_lower_step_days"
        action_zh = "扩大历史数据并降低信号间隔"
    else:
        action = "increase_history_or_frequency"
        action_zh = "增加历史周期或提高取样频率"

    command_hint = (
        f"--period {suggested_period} --step-days {suggested_step_days} "
        f"--min-history-days {suggested_min_history_days}"
    )
    reason = (
        f"{scope} sample count is {sample_count}, target is {target_sample_count}. "
        f"{reason_suffix}"
    )
    reason_zh = (
        f"{scope} 当前样本数为 {sample_count}，目标样本数为 {target_sample_count}。"
        f"{reason_suffix_zh}"
    )
    return {
        "scope": scope,
        "ticker": ticker,
        "sample_count": sample_count,
        "target_sample_count": target_sample_count,
        "sample_status": status,
        "sample_status_zh": status_zh,
        "guidance_action": action,
        "guidance_action_zh": action_zh,
        "current_step_days": step_days,
        "suggested_step_days": suggested_step_days,
        "current_min_history_days": min_history_days,
        "suggested_min_history_days": suggested_min_history_days,
        "suggested_period": suggested_period,
        "suggested_command_hint": command_hint,
        "suggested_command_hint_zh": f"建议参数：{command_hint}",
        "reason": reason,
        "reason_zh": reason_zh,
    }


def _sample_status(sample_count: int, target_sample_count: int) -> tuple[str, str]:
    if sample_count <= 0:
        return "no_samples", "没有样本"
    if sample_count < target_sample_count:
        return "thin_samples", "样本偏少"
    return "enough_samples", "样本充足"


def _suggested_step_days(step_days: int, sample_count: int, target_sample_count: int) -> int:
    if sample_count >= target_sample_count:
        return step_days
    if step_days > 60:
        return 20
    if step_days > 20:
        return max(20, step_days // 2)
    return step_days


def _suggested_min_history_days(
    min_history_days: int,
    sample_count: int,
    target_sample_count: int,
) -> int:
    if sample_count >= target_sample_count:
        return min_history_days
    if min_history_days > 170:
        return 170
    if min_history_days > 120:
        return 120
    return min_history_days


def _suggested_period(sample_count: int, target_sample_count: int) -> str:
    if sample_count >= target_sample_count:
        return "current"
    return "5y"


def summarize_walk_forward_profiles(
    events: pd.DataFrame,
    forward_windows: tuple[int, ...] = DEFAULT_FORWARD_WINDOWS,
) -> pd.DataFrame:
    columns = [
        "screening_profile",
        "screening_profile_zh",
        "validation_bucket",
        "sample_count",
    ]
    if events.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    group_columns = ["screening_profile", "screening_profile_zh", "validation_bucket"]
    for keys, group in events.groupby(group_columns, dropna=False):
        profile, profile_zh, bucket = keys
        row: dict[str, object] = {
            "screening_profile": profile,
            "screening_profile_zh": profile_zh,
            "validation_bucket": bucket,
            "sample_count": len(group),
            "quality_gate_pass_rate": float(group["quality_gate_passed"].mean())
            if "quality_gate_passed" in group
            else np.nan,
            "avg_high_probability_score": float(group["high_probability_score"].mean())
            if "high_probability_score" in group
            else np.nan,
            "avg_calibrated_win_probability": float(
                group["calibrated_win_probability"].mean()
            )
            if "calibrated_win_probability" in group
            else np.nan,
        }
        for window in forward_windows:
            returns = group[f"forward_return_{window}d"].dropna()
            row[f"win_rate_{window}d"] = float((returns > 0).mean()) if len(returns) else np.nan
            row[f"avg_return_{window}d"] = float(returns.mean()) if len(returns) else np.nan
            row[f"median_return_{window}d"] = float(returns.median()) if len(returns) else np.nan
            row[f"worst_return_{window}d"] = float(returns.min()) if len(returns) else np.nan
        drawdowns = group["max_drawdown_after_signal"].dropna()
        row["avg_max_drawdown_after_signal"] = float(drawdowns.mean()) if len(drawdowns) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["screening_profile", "validation_bucket"]
    ).reset_index(drop=True)


def summarize_walk_forward_segments(
    events: pd.DataFrame,
    forward_windows: tuple[int, ...] = DEFAULT_FORWARD_WINDOWS,
) -> pd.DataFrame:
    columns = [
        "validation_segment",
        "validation_segment_zh",
        "screening_profile",
        "screening_profile_zh",
        "horizon",
        "horizon_zh_label",
        "entry_type",
        "validation_bucket",
        "sample_count",
        "quality_gate_pass_rate",
        "avg_signal_score",
        "avg_confidence_score",
        "avg_high_probability_score",
        "avg_calibrated_win_probability",
        "avg_max_drawdown_after_signal",
        "segment_decision",
        "segment_decision_zh",
        "segment_note",
        "segment_note_zh",
    ]
    for window in forward_windows:
        columns.extend(
            [
                f"win_rate_{window}d",
                f"avg_return_{window}d",
                f"median_return_{window}d",
                f"worst_return_{window}d",
            ]
        )
    if events.empty:
        return pd.DataFrame(columns=columns)

    frame = events.copy()
    for column, default in [
        ("screening_profile", "default"),
        ("screening_profile_zh", "默认规则"),
        ("horizon", "unknown"),
        ("horizon_zh_label", "未知周期"),
        ("screening_backtest_entry_type", "unknown"),
        ("validation_bucket", "unknown"),
    ]:
        if column not in frame.columns:
            frame[column] = default
        frame[column] = frame[column].fillna(default).astype(str)

    group_columns = [
        "screening_profile",
        "screening_profile_zh",
        "horizon",
        "horizon_zh_label",
        "screening_backtest_entry_type",
        "validation_bucket",
    ]
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        profile, profile_zh, horizon, horizon_zh, entry_type, bucket = keys
        row: dict[str, object] = {
            "validation_segment": f"{profile}/{horizon}/{entry_type}/{bucket}",
            "validation_segment_zh": f"{profile_zh}/{horizon_zh}/{entry_type}/{bucket}",
            "screening_profile": profile,
            "screening_profile_zh": profile_zh,
            "horizon": horizon,
            "horizon_zh_label": horizon_zh,
            "entry_type": entry_type,
            "validation_bucket": bucket,
            "sample_count": int(len(group)),
            "quality_gate_pass_rate": _numeric_mean(group, "quality_gate_passed"),
            "avg_signal_score": _numeric_mean(group, "signal_score"),
            "avg_confidence_score": _numeric_mean(group, "confidence_score"),
            "avg_high_probability_score": _numeric_mean(group, "high_probability_score"),
            "avg_calibrated_win_probability": _numeric_mean(
                group,
                "calibrated_win_probability",
            ),
            "avg_max_drawdown_after_signal": _numeric_mean(
                group,
                "max_drawdown_after_signal",
            ),
        }
        for window in forward_windows:
            returns = pd.to_numeric(
                group.get(f"forward_return_{window}d", pd.Series(dtype=float)),
                errors="coerce",
            ).dropna()
            row[f"win_rate_{window}d"] = float((returns > 0).mean()) if len(returns) else np.nan
            row[f"avg_return_{window}d"] = float(returns.mean()) if len(returns) else np.nan
            row[f"median_return_{window}d"] = float(returns.median()) if len(returns) else np.nan
            row[f"worst_return_{window}d"] = float(returns.min()) if len(returns) else np.nan
        decision, decision_zh, note, note_zh = _segment_decision(row, _target_window(forward_windows))
        row["segment_decision"] = decision
        row["segment_decision_zh"] = decision_zh
        row["segment_note"] = note
        row["segment_note_zh"] = note_zh
        rows.append(row)

    result = pd.DataFrame(rows)
    result["_decision_rank"] = result["segment_decision"].map(
        {
            "strong_segment": 0,
            "watch_segment": 1,
            "thin_sample": 2,
            "weak_segment": 3,
        }
    ).fillna(9)
    result = result.sort_values(
        ["_decision_rank", "sample_count", f"win_rate_{_target_window(forward_windows)}d"],
        ascending=[True, False, False],
        na_position="last",
    ).drop(columns=["_decision_rank"])
    return result[columns].reset_index(drop=True)


def summarize_market_regime_validation(
    events: pd.DataFrame,
    forward_windows: tuple[int, ...] = DEFAULT_FORWARD_WINDOWS,
) -> pd.DataFrame:
    columns = [
        "validation_market_regime",
        "validation_market_regime_zh",
        "sample_count",
        "high_probability_sample_count",
        "quality_gate_pass_rate",
        "avg_signal_score",
        "avg_confidence_score",
        "avg_high_probability_score",
        "avg_calibrated_win_probability",
        "avg_max_drawdown_after_signal",
        "regime_decision",
        "regime_decision_zh",
        "regime_note",
        "regime_note_zh",
    ]
    for window in forward_windows:
        columns.extend(
            [
                f"win_rate_{window}d",
                f"avg_return_{window}d",
                f"median_return_{window}d",
                f"worst_return_{window}d",
            ]
        )
    if events.empty:
        return pd.DataFrame(columns=columns)

    frame = events.copy()
    if "validation_market_regime" not in frame.columns:
        frame["validation_market_regime"] = "unknown"
    if "validation_market_regime_zh" not in frame.columns:
        frame["validation_market_regime_zh"] = "未知市场状态"
    frame["validation_market_regime"] = frame["validation_market_regime"].fillna("unknown").astype(str)
    frame["validation_market_regime_zh"] = (
        frame["validation_market_regime_zh"].fillna("未知市场状态").astype(str)
    )

    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(
        ["validation_market_regime", "validation_market_regime_zh"],
        dropna=False,
    ):
        regime, regime_zh = keys
        row: dict[str, object] = {
            "validation_market_regime": regime,
            "validation_market_regime_zh": regime_zh,
            "sample_count": int(len(group)),
            "high_probability_sample_count": int(
                (group.get("validation_bucket", pd.Series(dtype=str)) == "high_probability").sum()
            ),
            "quality_gate_pass_rate": _numeric_mean(group, "quality_gate_passed"),
            "avg_signal_score": _numeric_mean(group, "signal_score"),
            "avg_confidence_score": _numeric_mean(group, "confidence_score"),
            "avg_high_probability_score": _numeric_mean(group, "high_probability_score"),
            "avg_calibrated_win_probability": _numeric_mean(
                group,
                "calibrated_win_probability",
            ),
            "avg_max_drawdown_after_signal": _numeric_mean(
                group,
                "max_drawdown_after_signal",
            ),
        }
        for window in forward_windows:
            returns = pd.to_numeric(
                group.get(f"forward_return_{window}d", pd.Series(dtype=float)),
                errors="coerce",
            ).dropna()
            row[f"win_rate_{window}d"] = float((returns > 0).mean()) if len(returns) else np.nan
            row[f"avg_return_{window}d"] = float(returns.mean()) if len(returns) else np.nan
            row[f"median_return_{window}d"] = float(returns.median()) if len(returns) else np.nan
            row[f"worst_return_{window}d"] = float(returns.min()) if len(returns) else np.nan
        decision, decision_zh, note, note_zh = _market_regime_validation_decision(
            row,
            _target_window(forward_windows),
        )
        row["regime_decision"] = decision
        row["regime_decision_zh"] = decision_zh
        row["regime_note"] = note
        row["regime_note_zh"] = note_zh
        rows.append(row)

    result = pd.DataFrame(rows)
    result["_decision_rank"] = result["regime_decision"].map(
        {
            "robust_regime": 0,
            "usable_regime": 1,
            "thin_sample": 2,
            "weak_regime": 3,
        }
    ).fillna(9)
    result = result.sort_values(
        ["_decision_rank", "sample_count", f"win_rate_{_target_window(forward_windows)}d"],
        ascending=[True, False, False],
        na_position="last",
    ).drop(columns=["_decision_rank"])
    return result[columns].reset_index(drop=True)


def _market_regime_validation_decision(
    row: dict[str, object],
    target_window: int,
) -> tuple[str, str, str, str]:
    sample_count = _safe_int(row.get("sample_count"))
    win_rate = _safe_float(row.get(f"win_rate_{target_window}d"))
    avg_return = _safe_float(row.get(f"avg_return_{target_window}d"))
    drawdown = _safe_float(row.get("avg_max_drawdown_after_signal"))
    if sample_count < 10:
        return (
            "thin_sample",
            "样本不足",
            f"Only {sample_count} samples in this market regime.",
            f"这个市场状态下只有 {sample_count} 个样本。",
        )
    if (
        _is_finite(win_rate)
        and win_rate >= 0.55
        and _is_finite(avg_return)
        and avg_return > 0
        and (not _is_finite(drawdown) or drawdown > -0.12)
    ):
        return (
            "robust_regime",
            "稳健市场状态",
            "Signals are historically robust in this market regime.",
            "该市场状态下信号历史表现较稳健。",
        )
    if _is_finite(avg_return) and avg_return > 0:
        return (
            "usable_regime",
            "可用但需谨慎",
            "Average return is positive, but regime evidence is not strong enough.",
            "平均收益为正，但该市场状态证据还不够强。",
        )
    return (
        "weak_regime",
        "弱市场状态",
        "Signals do not show positive evidence in this market regime.",
        "该市场状态下信号没有显示正向证据。",
    )


def build_market_regime_protection_policy(
    market_regime_summary: pd.DataFrame,
    target_window: int = 20,
) -> pd.DataFrame:
    columns = [
        "validation_market_regime",
        "validation_market_regime_zh",
        "sample_count",
        "win_rate",
        "avg_return",
        "avg_max_drawdown_after_signal",
        "regime_decision",
        "protection_action",
        "protection_action_zh",
        "allow_new_entries",
        "allowed_signal_bucket",
        "signal_score_delta",
        "confidence_score_delta",
        "high_probability_score_delta",
        "position_scale",
        "policy_severity",
        "policy_note",
        "policy_note_zh",
    ]
    if market_regime_summary.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for row in market_regime_summary.to_dict(orient="records"):
        regime = str(row.get("validation_market_regime", "unknown"))
        sample_count = _safe_int(row.get("sample_count"))
        win_rate = _safe_float(row.get(f"win_rate_{target_window}d"))
        avg_return = _safe_float(row.get(f"avg_return_{target_window}d"))
        drawdown = _safe_float(row.get("avg_max_drawdown_after_signal"))
        decision = str(row.get("regime_decision", ""))
        policy = _market_regime_policy_decision(
            regime=regime,
            sample_count=sample_count,
            win_rate=win_rate,
            avg_return=avg_return,
            drawdown=drawdown,
            regime_decision=decision,
        )
        rows.append(
            {
                "validation_market_regime": regime,
                "validation_market_regime_zh": row.get(
                    "validation_market_regime_zh",
                    "",
                ),
                "sample_count": sample_count,
                "win_rate": win_rate,
                "avg_return": avg_return,
                "avg_max_drawdown_after_signal": drawdown,
                "regime_decision": decision,
                **policy,
            }
        )
    result = pd.DataFrame(rows, columns=columns)
    result["_severity_rank"] = result["policy_severity"].map(
        {"critical": 0, "high": 1, "medium": 2, "info": 3}
    ).fillna(9)
    return result.sort_values(
        ["_severity_rank", "sample_count"],
        ascending=[True, False],
    ).drop(columns=["_severity_rank"]).reset_index(drop=True)


def _market_regime_policy_decision(
    regime: str,
    sample_count: int,
    win_rate: float,
    avg_return: float,
    drawdown: float,
    regime_decision: str,
) -> dict[str, object]:
    weak_evidence = (
        (_is_finite(win_rate) and win_rate < 0.50)
        or (_is_finite(avg_return) and avg_return <= 0)
        or (_is_finite(drawdown) and drawdown <= -0.15)
        or regime_decision == "weak_regime"
    )
    risk_regime = regime in {"bear_downtrend", "high_volatility"}
    if sample_count < 10:
        return {
            "protection_action": "collect_more_samples",
            "protection_action_zh": "继续收集样本",
            "allow_new_entries": False if risk_regime else True,
            "allowed_signal_bucket": "strict_high_probability_only" if risk_regime else "normal",
            "signal_score_delta": 3.0 if risk_regime else 0.0,
            "confidence_score_delta": 3.0 if risk_regime else 0.0,
            "high_probability_score_delta": 5.0 if risk_regime else 0.0,
            "position_scale": 0.50 if risk_regime else 1.0,
            "policy_severity": "medium" if risk_regime else "info",
            "policy_note": "Market-regime sample is thin; use conservative defaults until more evidence exists.",
            "policy_note_zh": "该市场状态样本不足；证据更多前使用保守默认规则。",
        }
    if risk_regime and weak_evidence:
        return {
            "protection_action": "block_new_entries",
            "protection_action_zh": "暂停新开仓",
            "allow_new_entries": False,
            "allowed_signal_bucket": "none",
            "signal_score_delta": 8.0,
            "confidence_score_delta": 8.0,
            "high_probability_score_delta": 10.0,
            "position_scale": 0.0,
            "policy_severity": "critical",
            "policy_note": "Bear or high-volatility regime shows weak evidence; new entries should be blocked until conditions improve.",
            "policy_note_zh": "熊市或高波动环境下历史证据偏弱；条件改善前应暂停新开仓。",
        }
    if risk_regime:
        return {
            "protection_action": "strict_only",
            "protection_action_zh": "只允许严格高概率",
            "allow_new_entries": True,
            "allowed_signal_bucket": "strict_high_probability_only",
            "signal_score_delta": 5.0,
            "confidence_score_delta": 5.0,
            "high_probability_score_delta": 7.0,
            "position_scale": 0.50,
            "policy_severity": "high",
            "policy_note": "Risk regime is usable but requires stricter thresholds and smaller position size.",
            "policy_note_zh": "风险市场状态仍可用，但需要更严格门槛和更小仓位。",
        }
    if weak_evidence:
        return {
            "protection_action": "tighten_entries",
            "protection_action_zh": "收紧入场",
            "allow_new_entries": True,
            "allowed_signal_bucket": "high_probability_or_near_only",
            "signal_score_delta": 3.0,
            "confidence_score_delta": 3.0,
            "high_probability_score_delta": 5.0,
            "position_scale": 0.75,
            "policy_severity": "medium",
            "policy_note": "Regime evidence is weak; allow only stronger setups.",
            "policy_note_zh": "该市场状态证据偏弱；只允许更强的机会。",
        }
    return {
        "protection_action": "normal_rules",
        "protection_action_zh": "使用正常规则",
        "allow_new_entries": True,
        "allowed_signal_bucket": "normal",
        "signal_score_delta": 0.0,
        "confidence_score_delta": 0.0,
        "high_probability_score_delta": 0.0,
        "position_scale": 1.0,
        "policy_severity": "info",
        "policy_note": "Market-regime validation is acceptable; use base rules.",
        "policy_note_zh": "该市场状态验证可接受；使用基础规则。",
    }


def _segment_decision(
    row: dict[str, object],
    target_window: int,
) -> tuple[str, str, str, str]:
    sample_count = _safe_int(row.get("sample_count"))
    win_rate = _safe_float(row.get(f"win_rate_{target_window}d"))
    avg_return = _safe_float(row.get(f"avg_return_{target_window}d"))
    drawdown = _safe_float(row.get("avg_max_drawdown_after_signal"))
    if sample_count < 10:
        return (
            "thin_sample",
            "样本不足",
            f"Only {sample_count} samples; do not tune this segment yet.",
            f"只有 {sample_count} 个样本；暂时不要根据该分层单独调参。",
        )
    if (
        _is_finite(win_rate)
        and win_rate >= 0.55
        and _is_finite(avg_return)
        and avg_return > 0
        and (not _is_finite(drawdown) or drawdown > -0.12)
    ):
        return (
            "strong_segment",
            "强分层",
            "Segment has enough samples, positive win rate, positive average return, and controlled drawdown.",
            "该分层样本足够，胜率、平均收益为正，回撤可控。",
        )
    if _is_finite(avg_return) and avg_return > 0:
        return (
            "watch_segment",
            "观察分层",
            "Segment average return is positive, but at least one quality metric is not strong enough.",
            "该分层平均收益为正，但至少一个质量指标还不够强。",
        )
    return (
        "weak_segment",
        "弱分层",
        "Segment does not yet show positive historical evidence.",
        "该分层还没有显示正向历史证据。",
    )


def summarize_probability_calibration(
    events: pd.DataFrame,
    target_window: int = 20,
) -> pd.DataFrame:
    columns = [
        "probability_bucket",
        "probability_bucket_zh",
        "sample_count",
        "avg_estimated_probability",
        f"actual_win_rate_{target_window}d",
        f"avg_return_{target_window}d",
        "calibration_error",
        "calibration_error_abs",
        "recommended_probability_adjustment",
        "adjusted_estimated_probability",
        "adjusted_calibration_error_abs",
        "brier_score",
        "expected_calibration_error_component",
        "formula_action",
        "formula_action_zh",
        "calibration_quality",
        "calibration_quality_zh",
        "calibration_note",
        "calibration_note_zh",
    ]
    required = {"calibrated_win_probability", f"forward_return_{target_window}d"}
    if events.empty or not required.issubset(events.columns):
        return pd.DataFrame(columns=columns)

    frame = events.copy()
    frame["calibrated_win_probability"] = pd.to_numeric(
        frame["calibrated_win_probability"],
        errors="coerce",
    )
    frame[f"forward_return_{target_window}d"] = pd.to_numeric(
        frame[f"forward_return_{target_window}d"],
        errors="coerce",
    )
    frame = frame.dropna(subset=["calibrated_win_probability", f"forward_return_{target_window}d"])
    if frame.empty:
        return pd.DataFrame(columns=columns)

    bins = [-np.inf, 0.50, 0.55, 0.60, 0.65, np.inf]
    labels = ["below_50", "50_to_55", "55_to_60", "60_to_65", "65_plus"]
    labels_zh = {
        "below_50": "低于50%",
        "50_to_55": "50%-55%",
        "55_to_60": "55%-60%",
        "60_to_65": "60%-65%",
        "65_plus": "65%以上",
    }
    frame["probability_bucket"] = pd.cut(
        frame["calibrated_win_probability"],
        bins=bins,
        labels=labels,
        right=False,
    ).astype("string")

    rows: list[dict[str, object]] = []
    return_column = f"forward_return_{target_window}d"
    total_samples = len(frame)
    for bucket in labels:
        group = frame[frame["probability_bucket"] == bucket]
        returns = group[return_column].dropna()
        if returns.empty:
            rows.append(
                {
                    "probability_bucket": bucket,
                    "probability_bucket_zh": labels_zh[bucket],
                    "sample_count": 0,
                    "avg_estimated_probability": np.nan,
                    f"actual_win_rate_{target_window}d": np.nan,
                    f"avg_return_{target_window}d": np.nan,
                    "calibration_error": np.nan,
                    "calibration_error_abs": np.nan,
                    "recommended_probability_adjustment": np.nan,
                    "adjusted_estimated_probability": np.nan,
                    "adjusted_calibration_error_abs": np.nan,
                    "brier_score": np.nan,
                    "expected_calibration_error_component": np.nan,
                    "formula_action": "collect_more_samples",
                    "formula_action_zh": "继续收集样本",
                    "calibration_quality": "insufficient",
                    "calibration_quality_zh": "样本不足",
                    "calibration_note": "No samples in this probability bucket.",
                    "calibration_note_zh": "这个概率分组没有样本。",
                }
            )
            continue
        estimated = float(group["calibrated_win_probability"].mean())
        actual = float((returns > 0).mean())
        avg_return = float(returns.mean())
        error = actual - estimated
        adjustment = min(max(error, -0.10), 0.10)
        adjusted_estimated = min(max(estimated + adjustment, 0.05), 0.90)
        adjusted_error_abs = abs(actual - adjusted_estimated)
        outcomes = (returns > 0).astype(float)
        brier_score = float(((outcomes - group.loc[returns.index, "calibrated_win_probability"]) ** 2).mean())
        ece_component = abs(error) * len(returns) / total_samples if total_samples else np.nan
        quality, quality_zh = _probability_calibration_quality(
            sample_count=len(returns),
            error_abs=abs(error),
        )
        formula_action, formula_action_zh = _probability_formula_action(
            sample_count=len(returns),
            error=error,
            error_abs=abs(error),
        )
        rows.append(
            {
                "probability_bucket": bucket,
                "probability_bucket_zh": labels_zh[bucket],
                "sample_count": int(len(returns)),
                "avg_estimated_probability": estimated,
                f"actual_win_rate_{target_window}d": actual,
                f"avg_return_{target_window}d": avg_return,
                "calibration_error": error,
                "calibration_error_abs": abs(error),
                "recommended_probability_adjustment": adjustment,
                "adjusted_estimated_probability": adjusted_estimated,
                "adjusted_calibration_error_abs": adjusted_error_abs,
                "brier_score": brier_score,
                "expected_calibration_error_component": ece_component,
                "formula_action": formula_action,
                "formula_action_zh": formula_action_zh,
                "calibration_quality": quality,
                "calibration_quality_zh": quality_zh,
                "calibration_note": (
                    f"Estimated win probability averaged {estimated:.2%}; "
                    f"actual {target_window}d win rate was {actual:.2%}. "
                    f"Suggested probability adjustment is {adjustment:+.2%}."
                ),
                "calibration_note_zh": (
                    f"估计胜率平均为{estimated:.2%}；"
                    f"未来{target_window}日实际胜率为{actual:.2%}。"
                    f"建议概率修正为{adjustment:+.2%}。"
                ),
            }
        )
    rows.append(_overall_probability_calibration_row(frame, target_window, total_samples))
    return pd.DataFrame(rows, columns=columns)


def _probability_calibration_quality(sample_count: int, error_abs: float) -> tuple[str, str]:
    if sample_count < 5 or not _is_finite(error_abs):
        return "insufficient", "样本不足"
    if error_abs <= 0.05:
        return "well_calibrated", "校准较好"
    if error_abs <= 0.10:
        return "acceptable", "可接受"
    return "miscalibrated", "偏差较大"


def _probability_formula_action(
    sample_count: int,
    error: float,
    error_abs: float,
) -> tuple[str, str]:
    if sample_count < 5 or not _is_finite(error) or not _is_finite(error_abs):
        return "collect_more_samples", "继续收集样本"
    if error > 0.08:
        return "raise_probability_estimate", "上调胜率估计"
    if error < -0.08:
        return "lower_probability_estimate", "下调胜率估计"
    if error_abs <= 0.05:
        return "keep_probability_formula", "保持概率公式"
    return "monitor_probability_formula", "继续观察概率公式"


def _overall_probability_calibration_row(
    frame: pd.DataFrame,
    target_window: int,
    total_samples: int,
) -> dict[str, object]:
    return_column = f"forward_return_{target_window}d"
    returns = frame[return_column].dropna()
    if returns.empty:
        return {
            "probability_bucket": "overall",
            "probability_bucket_zh": "整体",
            "sample_count": 0,
            "avg_estimated_probability": np.nan,
            f"actual_win_rate_{target_window}d": np.nan,
            f"avg_return_{target_window}d": np.nan,
            "calibration_error": np.nan,
            "calibration_error_abs": np.nan,
            "recommended_probability_adjustment": np.nan,
            "adjusted_estimated_probability": np.nan,
            "adjusted_calibration_error_abs": np.nan,
            "brier_score": np.nan,
            "expected_calibration_error_component": np.nan,
            "formula_action": "collect_more_samples",
            "formula_action_zh": "继续收集样本",
            "calibration_quality": "insufficient",
            "calibration_quality_zh": "样本不足",
            "calibration_note": "No validation samples are available.",
            "calibration_note_zh": "没有可用验证样本。",
        }

    estimated = float(frame.loc[returns.index, "calibrated_win_probability"].mean())
    actual = float((returns > 0).mean())
    avg_return = float(returns.mean())
    error = actual - estimated
    adjustment = min(max(error, -0.10), 0.10)
    adjusted_estimated = min(max(estimated + adjustment, 0.05), 0.90)
    adjusted_error_abs = abs(actual - adjusted_estimated)
    outcomes = (returns > 0).astype(float)
    brier_score = float(
        ((outcomes - frame.loc[returns.index, "calibrated_win_probability"]) ** 2).mean()
    )
    quality, quality_zh = _probability_calibration_quality(
        sample_count=len(returns),
        error_abs=abs(error),
    )
    formula_action, formula_action_zh = _probability_formula_action(
        sample_count=len(returns),
        error=error,
        error_abs=abs(error),
    )
    return {
        "probability_bucket": "overall",
        "probability_bucket_zh": "整体",
        "sample_count": int(len(returns)),
        "avg_estimated_probability": estimated,
        f"actual_win_rate_{target_window}d": actual,
        f"avg_return_{target_window}d": avg_return,
        "calibration_error": error,
        "calibration_error_abs": abs(error),
        "recommended_probability_adjustment": adjustment,
        "adjusted_estimated_probability": adjusted_estimated,
        "adjusted_calibration_error_abs": adjusted_error_abs,
        "brier_score": brier_score,
        "expected_calibration_error_component": abs(error)
        * len(returns)
        / total_samples
        if total_samples
        else np.nan,
        "formula_action": formula_action,
        "formula_action_zh": formula_action_zh,
        "calibration_quality": quality,
        "calibration_quality_zh": quality_zh,
        "calibration_note": (
            f"Overall estimated win probability averaged {estimated:.2%}; "
            f"actual {target_window}d win rate was {actual:.2%}. "
            f"Suggested global probability adjustment is {adjustment:+.2%}."
        ),
        "calibration_note_zh": (
            f"整体估计胜率平均为{estimated:.2%}；"
            f"未来{target_window}日实际胜率为{actual:.2%}。"
            f"建议整体概率修正为{adjustment:+.2%}。"
        ),
    }
