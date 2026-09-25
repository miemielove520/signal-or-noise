"""Per-ticker entry backtests: slippage, trade outcomes, regime coverage, decay and trust."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ._common import (
    _format_number,
    _format_percent,
    _is_finite,
    _last_valid,
    _safe_ratio,
)
from .horizons import (
    HorizonSpec,
)
from .indicators import (
    _average_true_range,
    _moving_average_slope,
    _pullback_entry,
    _stop_loss,
    _window_return,
)


MIN_BACKTEST_SLIPPAGE_PCT = 0.0005
MAX_BACKTEST_SLIPPAGE_PCT = 0.01
SLIPPAGE_LIQUIDITY_WINDOW = 20


def _dynamic_slippage_profile(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    spec: HorizonSpec,
) -> dict[str, object]:
    avg_dollar_volume = _last_valid(
        (close.astype(float) * volume.astype(float)).rolling(SLIPPAGE_LIQUIDITY_WINDOW).mean()
    )
    atr = _last_valid(_average_true_range(high, low, close, spec.atr_window))
    latest_close = _last_valid(close)
    atr_ratio = _safe_ratio(atr, latest_close)

    if not _is_finite(avg_dollar_volume):
        liquidity_slippage = 0.003
        liquidity_label = "unknown_liquidity"
        liquidity_label_zh = "流动性未知"
    elif avg_dollar_volume >= 1_000_000_000:
        liquidity_slippage = 0.0005
        liquidity_label = "very_high_liquidity"
        liquidity_label_zh = "流动性很高"
    elif avg_dollar_volume >= 250_000_000:
        liquidity_slippage = 0.0008
        liquidity_label = "high_liquidity"
        liquidity_label_zh = "流动性较高"
    elif avg_dollar_volume >= 50_000_000:
        liquidity_slippage = 0.0015
        liquidity_label = "moderate_liquidity"
        liquidity_label_zh = "流动性中等"
    elif avg_dollar_volume >= 10_000_000:
        liquidity_slippage = 0.0030
        liquidity_label = "low_liquidity"
        liquidity_label_zh = "流动性偏低"
    else:
        liquidity_slippage = 0.0060
        liquidity_label = "very_low_liquidity"
        liquidity_label_zh = "流动性很低"

    if not _is_finite(atr_ratio):
        volatility_addon = 0.0010
        volatility_label = "unknown_volatility"
        volatility_label_zh = "波动率未知"
    elif atr_ratio >= 0.06:
        volatility_addon = 0.0030
        volatility_label = "very_high_volatility"
        volatility_label_zh = "波动率很高"
    elif atr_ratio >= 0.04:
        volatility_addon = 0.0020
        volatility_label = "high_volatility"
        volatility_label_zh = "波动率较高"
    elif atr_ratio >= 0.025:
        volatility_addon = 0.0010
        volatility_label = "moderate_volatility"
        volatility_label_zh = "波动率中等"
    else:
        volatility_addon = 0.0
        volatility_label = "normal_volatility"
        volatility_label_zh = "波动率正常"

    slippage_pct = min(
        max(liquidity_slippage + volatility_addon, MIN_BACKTEST_SLIPPAGE_PCT),
        MAX_BACKTEST_SLIPPAGE_PCT,
    )
    note = (
        f"Dynamic slippage uses {SLIPPAGE_LIQUIDITY_WINDOW}-day average dollar volume "
        f"and ATR ratio. Liquidity={liquidity_label}, volatility={volatility_label}."
    )
    note_zh = (
        f"动态滑点使用近{SLIPPAGE_LIQUIDITY_WINDOW}日平均成交额和ATR波动率。"
        f"流动性={liquidity_label_zh}，波动率={volatility_label_zh}。"
    )
    return {
        "slippage_pct": round(float(slippage_pct), 6),
        "avg_dollar_volume": avg_dollar_volume,
        "atr_ratio": atr_ratio,
        "liquidity_label": liquidity_label,
        "liquidity_label_zh": liquidity_label_zh,
        "volatility_label": volatility_label,
        "volatility_label_zh": volatility_label_zh,
        "note": note,
        "note_zh": note_zh,
    }


def _entry_backtest_summary(
    frame: pd.DataFrame,
    spec: HorizonSpec,
    entry_buffer_pct: float,
) -> dict[str, object]:
    close = frame["adj_close"].astype(float).reset_index(drop=True)
    open_price = (
        frame["open"].astype(float).fillna(close).reset_index(drop=True)
        if "open" in frame
        else close.copy()
    )
    high = frame["high"].astype(float).fillna(close).reset_index(drop=True)
    low = frame["low"].astype(float).fillna(close).reset_index(drop=True)
    volume = frame["volume"].astype(float).reset_index(drop=True)
    slippage_profile = _dynamic_slippage_profile(
        close=close,
        high=high,
        low=low,
        volume=volume,
        spec=spec,
    )
    slippage_pct = float(slippage_profile["slippage_pct"])

    min_index = max(spec.min_history_days, spec.lookback_days, spec.trend_window, spec.atr_window)
    breakout_outcomes: list[str] = []
    breakout_returns: list[float] = []
    breakout_regimes: list[str] = []
    pullback_outcomes: list[str] = []
    pullback_returns: list[float] = []
    pullback_regimes: list[str] = []
    all_trade_returns: list[tuple[int, float]] = []

    for index in range(min_index, len(frame) - 2):
        history_close = close.iloc[: index + 1]
        latest_price = float(close.iloc[index])
        trend_ma = _last_valid(history_close.rolling(spec.trend_window).mean())
        trend_distance = _safe_ratio(latest_price, trend_ma) - 1.0
        trend_slope = _moving_average_slope(history_close, spec.trend_window)
        momentum = _window_return(history_close, spec.momentum_window)
        avg_volume = _last_valid(volume.iloc[: index + 1].rolling(spec.volume_window).mean())
        volume_ratio = _safe_ratio(float(volume.iloc[index]), avg_volume)
        atr = _last_valid(
            _average_true_range(
                high.iloc[: index + 1],
                low.iloc[: index + 1],
                history_close,
                spec.atr_window,
            )
        )

        prior_start = max(0, index - spec.lookback_days)
        prior_high = high.iloc[prior_start:index]
        prior_low = low.iloc[prior_start:index]
        if prior_high.empty or prior_low.empty:
            continue

        support = float(prior_low.min())
        resistance = float(prior_high.max())
        breakout_entry = resistance * (1.0 + entry_buffer_pct)
        pullback_entry = _pullback_entry(latest_price, trend_ma, support, entry_buffer_pct)
        uptrend = trend_distance > 0 and trend_slope >= 0
        positive_momentum = momentum > 0
        atr_ratio = _safe_ratio(atr, latest_price)
        regime_label, _ = _backtest_regime_label(
            trend_distance=trend_distance,
            trend_slope=trend_slope,
            momentum=momentum,
            atr_ratio=atr_ratio,
        )
        future_end = index + 1 + spec.time_stop_days
        future_high = high.iloc[index + 1 : future_end]
        future_low = low.iloc[index + 1 : future_end]
        future_open = open_price.iloc[index + 1 : future_end]
        future_close = close.iloc[index + 1 : future_end]
        if future_close.empty:
            continue

        if (
            uptrend
            and positive_momentum
            and volume_ratio >= 1.0
            and latest_price >= breakout_entry
        ):
            executed_entry = _realistic_long_entry_price(
                planned_entry=latest_price,
                next_open=float(future_open.iloc[0]),
                slippage_pct=slippage_pct,
            )
            stop_loss = _stop_loss(
                entry_price=executed_entry,
                support=support,
                atr=atr,
                atr_multiple=spec.atr_stop_multiple,
                entry_buffer_pct=entry_buffer_pct,
            )
            outcome, trade_return = _trade_result(
                entry_price=executed_entry,
                stop_loss=stop_loss,
                target_r_multiple=spec.target_r_multiple,
                future_open=future_open,
                future_high=future_high,
                future_low=future_low,
                future_close=future_close,
                slippage_pct=slippage_pct,
                trailing_stop_trigger_r=spec.trailing_stop_trigger_r,
                trailing_stop_lock_r=spec.trailing_stop_lock_r,
            )
            breakout_outcomes.append(outcome)
            breakout_returns.append(trade_return)
            breakout_regimes.append(regime_label)
            all_trade_returns.append((index, trade_return))

        if (
            uptrend
            and positive_momentum
            and _is_finite(pullback_entry)
            and float(low.iloc[index]) <= pullback_entry <= float(high.iloc[index])
        ):
            executed_entry = _realistic_long_entry_price(
                planned_entry=pullback_entry,
                next_open=float(future_open.iloc[0]),
                slippage_pct=slippage_pct,
            )
            stop_loss = _stop_loss(
                entry_price=executed_entry,
                support=support,
                atr=atr,
                atr_multiple=spec.atr_stop_multiple,
                entry_buffer_pct=entry_buffer_pct,
            )
            outcome, trade_return = _trade_result(
                entry_price=executed_entry,
                stop_loss=stop_loss,
                target_r_multiple=spec.target_r_multiple,
                future_open=future_open,
                future_high=future_high,
                future_low=future_low,
                future_close=future_close,
                slippage_pct=slippage_pct,
                trailing_stop_trigger_r=spec.trailing_stop_trigger_r,
                trailing_stop_lock_r=spec.trailing_stop_lock_r,
            )
            pullback_outcomes.append(outcome)
            pullback_returns.append(trade_return)
            pullback_regimes.append(regime_label)
            all_trade_returns.append((index, trade_return))

    breakout_win_rate = _win_rate(breakout_outcomes)
    breakout_target_hit_rate = _outcome_rate(breakout_outcomes, "target_hit")
    breakout_stop_hit_rate = _outcome_rate(breakout_outcomes, "stop_hit")
    breakout_trailing_stop_hit_rate = _outcome_rate(breakout_outcomes, "trailing_stop")
    breakout_average_gain = _average_gain(breakout_returns)
    breakout_average_loss = _average_loss(breakout_returns)
    breakout_average_return = _average_return(breakout_returns)
    breakout_sample_quality, breakout_sample_quality_zh = _sample_quality(len(breakout_outcomes))
    pullback_win_rate = _win_rate(pullback_outcomes)
    pullback_target_hit_rate = _outcome_rate(pullback_outcomes, "target_hit")
    pullback_stop_hit_rate = _outcome_rate(pullback_outcomes, "stop_hit")
    pullback_trailing_stop_hit_rate = _outcome_rate(pullback_outcomes, "trailing_stop")
    pullback_average_gain = _average_gain(pullback_returns)
    pullback_average_loss = _average_loss(pullback_returns)
    pullback_average_return = _average_return(pullback_returns)
    pullback_sample_quality, pullback_sample_quality_zh = _sample_quality(len(pullback_outcomes))
    regime_coverage = _regime_coverage_profile(breakout_regimes + pullback_regimes)
    recent_backtest = _recent_backtest_profile(all_trade_returns)
    backtest_decay = _backtest_decay_profile(all_trade_returns)
    execution_model = "next_open_dynamic_slippage"
    execution_model_zh = "下一交易日开盘并计入动态滑点"
    execution_note = (
        "Backtest assumes signals are generated after the close, entries execute at the next "
        f"session open with {_format_percent(slippage_pct)} dynamic slippage, and gap-down stops "
        f"exit at the open instead of the ideal stop price. Trades use a maximum "
        f"{spec.time_stop_days}-session time stop before timeout exit. Trailing stop activates "
        f"after {spec.trailing_stop_trigger_r:.2f}R and locks {spec.trailing_stop_lock_r:.2f}R. "
        f"{slippage_profile['note']}"
    )
    execution_note_zh = (
        "回测假设信号在收盘后生成，买入按下一交易日开盘价并计入"
        f"{_format_percent(slippage_pct)}动态滑点；如果止损日跳空低开，"
        f"按开盘价退出，而不是按理想止损价退出。每笔交易最多持有"
        f"{spec.time_stop_days}个交易日，超过后按超时退出处理。移动止损在"
        f"{spec.trailing_stop_trigger_r:.2f}R后触发，并锁定{spec.trailing_stop_lock_r:.2f}R。"
        f"{slippage_profile['note_zh']}"
    )
    note = (
        f"Breakout sample={len(breakout_outcomes)}, quality={breakout_sample_quality}, "
        f"win_rate={_format_percent(breakout_win_rate)}, "
        f"target_hit={_format_percent(breakout_target_hit_rate)}, "
        f"stop_hit={_format_percent(breakout_stop_hit_rate)}, "
        f"trailing_stop={_format_percent(breakout_trailing_stop_hit_rate)}, "
        f"avg_gain={_format_percent(breakout_average_gain)}, "
        f"avg_loss={_format_percent(breakout_average_loss)}, "
        f"avg_return={_format_percent(breakout_average_return)}; "
        f"pullback sample={len(pullback_outcomes)}, quality={pullback_sample_quality}, "
        f"win_rate={_format_percent(pullback_win_rate)}, "
        f"target_hit={_format_percent(pullback_target_hit_rate)}, "
        f"stop_hit={_format_percent(pullback_stop_hit_rate)}, "
        f"trailing_stop={_format_percent(pullback_trailing_stop_hit_rate)}, "
        f"avg_gain={_format_percent(pullback_average_gain)}, "
        f"avg_loss={_format_percent(pullback_average_loss)}, "
        f"avg_return={_format_percent(pullback_average_return)}; "
        f"regime coverage score={_format_number(regime_coverage['score'])}, "
        f"dominant regime={regime_coverage['dominant_regime']}; "
        f"recent strength score={_format_number(recent_backtest['score'])}, "
        f"recent avg return={_format_percent(recent_backtest['recent_average_return'])}; "
        f"decay score={_format_number(backtest_decay['score'])}, "
        f"late avg return={_format_percent(backtest_decay['late_average_return'])}. "
        f"{execution_note}"
    )
    note_zh = (
        f"突破样本={len(breakout_outcomes)}，可靠性={breakout_sample_quality_zh}，"
        f"胜率={_format_percent(breakout_win_rate)}，"
        f"目标价命中率={_format_percent(breakout_target_hit_rate)}，"
        f"止损命中率={_format_percent(breakout_stop_hit_rate)}，"
        f"移动止损命中率={_format_percent(breakout_trailing_stop_hit_rate)}，"
        f"平均盈利={_format_percent(breakout_average_gain)}，"
        f"平均亏损={_format_percent(breakout_average_loss)}，"
        f"平均收益={_format_percent(breakout_average_return)}；"
        f"回调样本={len(pullback_outcomes)}，可靠性={pullback_sample_quality_zh}，"
        f"胜率={_format_percent(pullback_win_rate)}，"
        f"目标价命中率={_format_percent(pullback_target_hit_rate)}，"
        f"止损命中率={_format_percent(pullback_stop_hit_rate)}，"
        f"移动止损命中率={_format_percent(pullback_trailing_stop_hit_rate)}，"
        f"平均盈利={_format_percent(pullback_average_gain)}，"
        f"平均亏损={_format_percent(pullback_average_loss)}，"
        f"平均收益={_format_percent(pullback_average_return)}；"
        f"行情覆盖分={_format_number(regime_coverage['score'])}，"
        f"主导行情={regime_coverage['dominant_regime_zh']}；"
        f"近期强度分={_format_number(recent_backtest['score'])}，"
        f"近期平均收益={_format_percent(recent_backtest['recent_average_return'])}；"
        f"衰退检查分={_format_number(backtest_decay['score'])}，"
        f"后半段平均收益={_format_percent(backtest_decay['late_average_return'])}。"
        f"{execution_note_zh}"
    )
    return {
        "backtest_execution_model": execution_model,
        "backtest_execution_model_zh": execution_model_zh,
        "backtest_time_stop_days": spec.time_stop_days,
        "backtest_trailing_stop_trigger_r": spec.trailing_stop_trigger_r,
        "backtest_trailing_stop_lock_r": spec.trailing_stop_lock_r,
        "backtest_slippage_pct": slippage_pct,
        "backtest_avg_dollar_volume": slippage_profile["avg_dollar_volume"],
        "backtest_atr_ratio": slippage_profile["atr_ratio"],
        "backtest_liquidity_label": slippage_profile["liquidity_label"],
        "backtest_liquidity_label_zh": slippage_profile["liquidity_label_zh"],
        "backtest_volatility_label": slippage_profile["volatility_label"],
        "backtest_volatility_label_zh": slippage_profile["volatility_label_zh"],
        "backtest_execution_note": execution_note,
        "backtest_execution_note_zh": execution_note_zh,
        "regime_coverage_score": regime_coverage["score"],
        "regime_coverage_level": regime_coverage["level"],
        "regime_coverage_level_zh": regime_coverage["level_zh"],
        "regime_coverage_regime_count": regime_coverage["regime_count"],
        "regime_coverage_dominant_regime": regime_coverage["dominant_regime"],
        "regime_coverage_dominant_regime_zh": regime_coverage["dominant_regime_zh"],
        "regime_coverage_dominant_share": regime_coverage["dominant_share"],
        "regime_coverage_note": regime_coverage["note"],
        "regime_coverage_note_zh": regime_coverage["note_zh"],
        "recent_backtest_score": recent_backtest["score"],
        "recent_backtest_level": recent_backtest["level"],
        "recent_backtest_level_zh": recent_backtest["level_zh"],
        "recent_backtest_trade_count": recent_backtest["recent_trade_count"],
        "recent_backtest_win_rate": recent_backtest["recent_win_rate"],
        "recent_backtest_average_return": recent_backtest["recent_average_return"],
        "recent_backtest_return_delta": recent_backtest["return_delta"],
        "recent_backtest_note": recent_backtest["note"],
        "recent_backtest_note_zh": recent_backtest["note_zh"],
        "backtest_decay_score": backtest_decay["score"],
        "backtest_decay_level": backtest_decay["level"],
        "backtest_decay_level_zh": backtest_decay["level_zh"],
        "backtest_decay_early_trade_count": backtest_decay["early_trade_count"],
        "backtest_decay_late_trade_count": backtest_decay["late_trade_count"],
        "backtest_decay_early_win_rate": backtest_decay["early_win_rate"],
        "backtest_decay_late_win_rate": backtest_decay["late_win_rate"],
        "backtest_decay_early_average_return": backtest_decay["early_average_return"],
        "backtest_decay_late_average_return": backtest_decay["late_average_return"],
        "backtest_decay_win_rate_delta": backtest_decay["win_rate_delta"],
        "backtest_decay_average_return_delta": backtest_decay["average_return_delta"],
        "backtest_decay_note": backtest_decay["note"],
        "backtest_decay_note_zh": backtest_decay["note_zh"],
        "breakout_trade_count": len(breakout_outcomes),
        "breakout_sample_quality": breakout_sample_quality,
        "breakout_sample_quality_zh": breakout_sample_quality_zh,
        "breakout_win_rate": breakout_win_rate,
        "breakout_target_hit_rate": breakout_target_hit_rate,
        "breakout_stop_hit_rate": breakout_stop_hit_rate,
        "breakout_trailing_stop_hit_rate": breakout_trailing_stop_hit_rate,
        "breakout_average_gain": breakout_average_gain,
        "breakout_average_loss": breakout_average_loss,
        "breakout_average_return": breakout_average_return,
        "pullback_trade_count": len(pullback_outcomes),
        "pullback_sample_quality": pullback_sample_quality,
        "pullback_sample_quality_zh": pullback_sample_quality_zh,
        "pullback_win_rate": pullback_win_rate,
        "pullback_target_hit_rate": pullback_target_hit_rate,
        "pullback_stop_hit_rate": pullback_stop_hit_rate,
        "pullback_trailing_stop_hit_rate": pullback_trailing_stop_hit_rate,
        "pullback_average_gain": pullback_average_gain,
        "pullback_average_loss": pullback_average_loss,
        "pullback_average_return": pullback_average_return,
        "entry_backtest_note": note,
        "entry_backtest_note_zh": note_zh,
    }


def _backtest_regime_label(
    trend_distance: float,
    trend_slope: float,
    momentum: float,
    atr_ratio: float,
) -> tuple[str, str]:
    if _is_finite(atr_ratio) and atr_ratio >= 0.07:
        return "volatile", "高波动"
    if trend_distance >= 0.08 and trend_slope > 0 and momentum > 0:
        return "strong_uptrend", "强趋势"
    if trend_distance > 0 and trend_slope >= 0 and momentum > 0:
        return "steady_uptrend", "稳定上升"
    if trend_distance < 0 or trend_slope < 0:
        return "weak_or_downtrend", "弱势或下跌"
    return "mixed", "震荡混合"


def _regime_label_zh(label: str) -> str:
    return {
        "strong_uptrend": "强趋势",
        "steady_uptrend": "稳定上升",
        "volatile": "高波动",
        "weak_or_downtrend": "弱势或下跌",
        "mixed": "震荡混合",
        "none": "无样本",
    }.get(label, "未知")


def _regime_coverage_profile(regime_labels: list[str]) -> dict[str, object]:
    total = len(regime_labels)
    if total == 0:
        return {
            "score": 20.0,
            "level": "no_sample",
            "level_zh": "无样本",
            "regime_count": 0,
            "dominant_regime": "none",
            "dominant_regime_zh": "无样本",
            "dominant_share": np.nan,
            "note": "No entry backtest samples are available, so regime coverage cannot be trusted.",
            "note_zh": "没有买点回测样本，因此无法判断行情覆盖度。",
        }

    counts = pd.Series(regime_labels).value_counts()
    regime_count = int(len(counts))
    dominant_regime = str(counts.index[0])
    dominant_share = float(counts.iloc[0] / total)

    sample_score = min(total / 40.0, 1.0) * 35.0
    diversity_score = min(regime_count / 3.0, 1.0) * 35.0
    concentration_score = max(0.0, (1.0 - dominant_share) / 0.55) * 30.0
    score = round(float(max(0.0, min(sample_score + diversity_score + concentration_score, 100.0))), 2)

    if score >= 75:
        level, level_zh = "broad", "覆盖较广"
    elif score >= 55:
        level, level_zh = "moderate", "覆盖一般"
    elif score >= 35:
        level, level_zh = "narrow", "覆盖偏窄"
    else:
        level, level_zh = "thin", "覆盖很弱"

    counts_text = ", ".join(f"{label}={int(count)}" for label, count in counts.items())
    counts_text_zh = "，".join(
        f"{_regime_label_zh(str(label))}={int(count)}" for label, count in counts.items()
    )
    note = (
        f"Regime coverage uses {total} entry samples across {regime_count} regime types; "
        f"dominant regime is {dominant_regime} at {_format_percent(dominant_share)}. "
        f"Counts: {counts_text}."
    )
    note_zh = (
        f"行情覆盖度使用{total}个买点样本，覆盖{regime_count}类行情；"
        f"主导行情是{_regime_label_zh(dominant_regime)}，占比{_format_percent(dominant_share)}。"
        f"分布：{counts_text_zh}。"
    )
    return {
        "score": score,
        "level": level,
        "level_zh": level_zh,
        "regime_count": regime_count,
        "dominant_regime": dominant_regime,
        "dominant_regime_zh": _regime_label_zh(dominant_regime),
        "dominant_share": round(float(dominant_share), 4),
        "note": note,
        "note_zh": note_zh,
    }


def _recent_backtest_profile(trade_returns: list[tuple[int, float]]) -> dict[str, object]:
    ordered_returns = [
        float(value)
        for _, value in sorted(trade_returns, key=lambda item: item[0])
        if _is_finite(value)
    ]
    total_count = len(ordered_returns)
    if total_count == 0:
        return {
            "score": 20.0,
            "level": "no_sample",
            "level_zh": "无样本",
            "recent_trade_count": 0,
            "recent_win_rate": np.nan,
            "recent_average_return": np.nan,
            "return_delta": np.nan,
            "note": "No recent entry samples are available, so recent backtest strength cannot be measured.",
            "note_zh": "没有近期买点样本，因此无法衡量近期回测强度。",
        }

    recent_count = min(total_count, max(5, int(math.ceil(total_count * 0.35))))
    recent_returns = ordered_returns[-recent_count:]
    overall_average = float(np.mean(ordered_returns))
    recent_average = float(np.mean(recent_returns))
    recent_win_rate = float(np.mean([value > 0 for value in recent_returns]))
    return_delta = recent_average - overall_average

    sample_component = min(recent_count / 10.0, 1.0) * 20.0
    win_component = recent_win_rate * 35.0
    return_component = max(0.0, min(35.0, 17.5 + recent_average * 500.0))
    delta_component = max(0.0, min(10.0, 5.0 + return_delta * 250.0))
    score = round(
        float(max(0.0, min(sample_component + win_component + return_component + delta_component, 100.0))),
        2,
    )

    if score >= 75:
        level, level_zh = "improving", "近期改善"
    elif score >= 55:
        level, level_zh = "stable", "近期稳定"
    elif score >= 35:
        level, level_zh = "weakening", "近期转弱"
    else:
        level, level_zh = "poor_recent_evidence", "近期证据较差"

    note = (
        f"Recent backtest strength uses the latest {recent_count} of {total_count} entry samples; "
        f"recent win rate={_format_percent(recent_win_rate)}, recent average return="
        f"{_format_percent(recent_average)}, return delta versus all samples="
        f"{_format_percent(return_delta)}."
    )
    note_zh = (
        f"近期回测强度使用最近{recent_count}笔、总共{total_count}笔买点样本；"
        f"近期胜率={_format_percent(recent_win_rate)}，近期平均收益="
        f"{_format_percent(recent_average)}，相对全部样本的收益差="
        f"{_format_percent(return_delta)}。"
    )
    return {
        "score": score,
        "level": level,
        "level_zh": level_zh,
        "recent_trade_count": recent_count,
        "recent_win_rate": round(float(recent_win_rate), 4),
        "recent_average_return": round(float(recent_average), 4),
        "return_delta": round(float(return_delta), 4),
        "note": note,
        "note_zh": note_zh,
    }


def _backtest_decay_profile(trade_returns: list[tuple[int, float]]) -> dict[str, object]:
    ordered_returns = [
        float(value)
        for _, value in sorted(trade_returns, key=lambda item: item[0])
        if _is_finite(value)
    ]
    total_count = len(ordered_returns)
    if total_count < 6:
        return {
            "score": 35.0 if total_count > 0 else 20.0,
            "level": "too_few_samples",
            "level_zh": "样本太少",
            "early_trade_count": total_count // 2,
            "late_trade_count": total_count - total_count // 2,
            "early_win_rate": np.nan,
            "late_win_rate": np.nan,
            "early_average_return": np.nan,
            "late_average_return": np.nan,
            "win_rate_delta": np.nan,
            "average_return_delta": np.nan,
            "note": "Too few entry samples are available to measure backtest decay reliably.",
            "note_zh": "买点样本太少，无法可靠衡量回测是否衰退。",
        }

    split_index = total_count // 2
    early_returns = ordered_returns[:split_index]
    late_returns = ordered_returns[split_index:]

    early_win_rate = float(np.mean([value > 0 for value in early_returns]))
    late_win_rate = float(np.mean([value > 0 for value in late_returns]))
    early_average = float(np.mean(early_returns))
    late_average = float(np.mean(late_returns))
    win_rate_delta = late_win_rate - early_win_rate
    average_return_delta = late_average - early_average

    sample_component = min(total_count / 30.0, 1.0) * 20.0
    late_win_component = late_win_rate * 25.0
    late_return_component = max(0.0, min(30.0, 15.0 + late_average * 500.0))
    win_delta_component = max(0.0, min(10.0, 5.0 + win_rate_delta * 20.0))
    return_delta_component = max(0.0, min(15.0, 7.5 + average_return_delta * 300.0))
    score = round(
        float(
            max(
                0.0,
                min(
                    sample_component
                    + late_win_component
                    + late_return_component
                    + win_delta_component
                    + return_delta_component,
                    100.0,
                ),
            )
        ),
        2,
    )

    if score >= 75:
        level, level_zh = "durable", "较稳定"
    elif score >= 55:
        level, level_zh = "acceptable", "可接受"
    elif score >= 35:
        level, level_zh = "decaying", "出现衰退"
    else:
        level, level_zh = "severe_decay", "明显衰退"

    note = (
        f"Backtest decay compares the first {len(early_returns)} samples with the latest "
        f"{len(late_returns)} samples; early win rate={_format_percent(early_win_rate)}, "
        f"late win rate={_format_percent(late_win_rate)}, early average return="
        f"{_format_percent(early_average)}, late average return={_format_percent(late_average)}, "
        f"late-minus-early return delta={_format_percent(average_return_delta)}."
    )
    note_zh = (
        f"回测衰退检查比较前半段{len(early_returns)}笔和后半段{len(late_returns)}笔样本；"
        f"前半段胜率={_format_percent(early_win_rate)}，后半段胜率={_format_percent(late_win_rate)}，"
        f"前半段平均收益={_format_percent(early_average)}，后半段平均收益={_format_percent(late_average)}，"
        f"后半段相对前半段收益差={_format_percent(average_return_delta)}。"
    )
    return {
        "score": score,
        "level": level,
        "level_zh": level_zh,
        "early_trade_count": len(early_returns),
        "late_trade_count": len(late_returns),
        "early_win_rate": round(float(early_win_rate), 4),
        "late_win_rate": round(float(late_win_rate), 4),
        "early_average_return": round(float(early_average), 4),
        "late_average_return": round(float(late_average), 4),
        "win_rate_delta": round(float(win_rate_delta), 4),
        "average_return_delta": round(float(average_return_delta), 4),
        "note": note,
        "note_zh": note_zh,
    }


def _trade_result(
    entry_price: float,
    stop_loss: float,
    target_r_multiple: float,
    future_open: pd.Series,
    future_high: pd.Series,
    future_low: pd.Series,
    future_close: pd.Series,
    slippage_pct: float = MIN_BACKTEST_SLIPPAGE_PCT,
    trailing_stop_trigger_r: float | None = None,
    trailing_stop_lock_r: float | None = None,
) -> tuple[str, float]:
    risk_per_share = entry_price - stop_loss
    if not _is_finite(risk_per_share) or risk_per_share <= 0:
        return "invalid", np.nan
    take_profit = entry_price + risk_per_share * target_r_multiple
    active_stop = stop_loss
    trailing_active = False
    trigger_r = float(trailing_stop_trigger_r) if _is_finite(trailing_stop_trigger_r) else np.nan
    lock_r = float(trailing_stop_lock_r) if _is_finite(trailing_stop_lock_r) else np.nan
    trailing_enabled = _is_finite(trigger_r) and trigger_r > 0 and _is_finite(lock_r)

    for open_value, high_value, low_value in zip(future_open, future_high, future_low):
        open_float = float(open_value)
        high_float = float(high_value)
        low_float = float(low_value)
        if open_float <= active_stop:
            exit_price = max(open_float * (1.0 - slippage_pct), 0.01)
            outcome = "gap_trailing_stop_hit" if trailing_active else "gap_stop_hit"
            return outcome, exit_price / entry_price - 1.0
        if low_float <= active_stop:
            exit_price = max(active_stop * (1.0 - slippage_pct), 0.01)
            if trailing_active:
                outcome = "trailing_stop_win" if exit_price > entry_price else "trailing_stop_loss"
            else:
                outcome = "stop_hit"
            return outcome, exit_price / entry_price - 1.0
        if high_float >= take_profit:
            exit_price = take_profit * (1.0 - slippage_pct)
            return "target_hit", exit_price / entry_price - 1.0
        if trailing_enabled and high_float >= entry_price + risk_per_share * trigger_r:
            active_stop = max(active_stop, entry_price + risk_per_share * lock_r)
            trailing_active = active_stop > stop_loss
    final_exit = float(future_close.iloc[-1]) * (1.0 - slippage_pct)
    final_return = final_exit / entry_price - 1.0
    return ("timeout_win" if final_return > 0 else "timeout_loss"), final_return


def _realistic_long_entry_price(
    planned_entry: float,
    next_open: float,
    slippage_pct: float = MIN_BACKTEST_SLIPPAGE_PCT,
) -> float:
    if _is_finite(next_open) and next_open > 0:
        raw_entry = float(next_open)
    else:
        raw_entry = float(planned_entry)
    return raw_entry * (1.0 + slippage_pct)


def _win_rate(outcomes: list[str]) -> float:
    if not outcomes:
        return np.nan
    wins = sum(
        outcome in {"target_hit", "timeout_win", "trailing_stop_win"}
        for outcome in outcomes
    )
    return float(wins / len(outcomes))


def _outcome_rate(outcomes: list[str], target_outcome: str) -> float:
    if not outcomes:
        return np.nan
    if target_outcome == "trailing_stop":
        return float(
            sum(
                outcome
                in {"trailing_stop_win", "trailing_stop_loss", "gap_trailing_stop_hit"}
                for outcome in outcomes
            )
            / len(outcomes)
        )
    if target_outcome == "stop_hit":
        return float(
            sum(
                outcome
                in {
                    "stop_hit",
                    "gap_stop_hit",
                    "trailing_stop_loss",
                    "trailing_stop_win",
                    "gap_trailing_stop_hit",
                }
                for outcome in outcomes
            )
            / len(outcomes)
        )
    return float(sum(outcome == target_outcome for outcome in outcomes) / len(outcomes))


def _average_gain(returns: list[float]) -> float:
    gains = [value for value in returns if _is_finite(value) and value > 0]
    if not gains:
        return np.nan
    return float(np.mean(gains))


def _average_loss(returns: list[float]) -> float:
    losses = [value for value in returns if _is_finite(value) and value < 0]
    if not losses:
        return np.nan
    return float(np.mean(losses))


def _average_return(returns: list[float]) -> float:
    values = [value for value in returns if _is_finite(value)]
    if not values:
        return np.nan
    return float(np.mean(values))


def _sample_quality(sample_count: int) -> tuple[str, str]:
    if sample_count >= 30:
        return "strong", "较可靠"
    if sample_count >= 15:
        return "moderate", "一般"
    if sample_count >= 5:
        return "weak", "偏弱"
    return "insufficient", "样本不足"


def _backtest_reliability(entry_backtest: dict[str, object]) -> dict[str, str]:
    breakout_count = int(entry_backtest["breakout_trade_count"])
    pullback_count = int(entry_backtest["pullback_trade_count"])
    total_count = breakout_count + pullback_count
    breakout_return = float(entry_backtest["breakout_average_return"])
    pullback_return = float(entry_backtest["pullback_average_return"])
    breakout_win = float(entry_backtest["breakout_win_rate"])
    pullback_win = float(entry_backtest["pullback_win_rate"])

    if total_count >= 50:
        level, level_zh = "strong", "较可靠"
    elif total_count >= 25:
        level, level_zh = "moderate", "一般"
    elif total_count >= 10:
        level, level_zh = "weak", "偏弱"
    else:
        level, level_zh = "insufficient", "样本不足"

    notes: list[str] = [f"total sample={total_count}"]
    notes_zh: list[str] = [f"总样本={total_count}"]
    if breakout_count > 0:
        notes.append(
            f"breakout sample={breakout_count}, win rate={_format_percent(breakout_win)}, avg return={_format_percent(breakout_return)}"
        )
        notes_zh.append(
            f"突破样本={breakout_count}，胜率={_format_percent(breakout_win)}，平均收益={_format_percent(breakout_return)}"
        )
    if pullback_count > 0:
        notes.append(
            f"pullback sample={pullback_count}, win rate={_format_percent(pullback_win)}, avg return={_format_percent(pullback_return)}"
        )
        notes_zh.append(
            f"回调样本={pullback_count}，胜率={_format_percent(pullback_win)}，平均收益={_format_percent(pullback_return)}"
        )

    if level == "insufficient":
        conclusion = "Do not rely on this backtest alone."
        conclusion_zh = "不能单独依赖该回测。"
    elif level == "weak":
        conclusion = "Use this backtest only as a weak reference."
        conclusion_zh = "该回测只能作为偏弱参考。"
    elif level == "moderate":
        conclusion = "This backtest is usable as supporting evidence, not as a standalone signal."
        conclusion_zh = "该回测可作为辅助证据，但不能单独作为信号。"
    else:
        conclusion = "This backtest has better sample support, but still needs current setup confirmation."
        conclusion_zh = "该回测样本支持较好，但仍需要当前结构确认。"

    return {
        "level": level,
        "level_zh": level_zh,
        "note": "; ".join(notes) + f". {conclusion}",
        "note_zh": "；".join(notes_zh) + f"。{conclusion_zh}",
    }


def _backtest_trust_profile(
    entry_backtest: dict[str, object],
    price_health: dict[str, object],
) -> dict[str, object]:
    breakout_count = int(entry_backtest["breakout_trade_count"])
    pullback_count = int(entry_backtest["pullback_trade_count"])
    total_count = breakout_count + pullback_count
    sample_score = min(total_count / 50.0, 1.0) * 100.0

    avg_dollar_volume = float(entry_backtest.get("backtest_avg_dollar_volume", np.nan))
    if not _is_finite(avg_dollar_volume):
        liquidity_score = 45.0
    elif avg_dollar_volume >= 1_000_000_000:
        liquidity_score = 100.0
    elif avg_dollar_volume >= 250_000_000:
        liquidity_score = 90.0
    elif avg_dollar_volume >= 50_000_000:
        liquidity_score = 75.0
    elif avg_dollar_volume >= 10_000_000:
        liquidity_score = 55.0
    else:
        liquidity_score = 30.0

    slippage_pct = float(entry_backtest.get("backtest_slippage_pct", np.nan))
    if not _is_finite(slippage_pct):
        slippage_score = 45.0
    else:
        slippage_score = 100.0 - (slippage_pct / MAX_BACKTEST_SLIPPAGE_PCT) * 70.0
        slippage_score = max(20.0, min(100.0, slippage_score))

    breakout_return = float(entry_backtest.get("breakout_average_return", np.nan))
    pullback_return = float(entry_backtest.get("pullback_average_return", np.nan))
    return_values = [value for value in [breakout_return, pullback_return] if _is_finite(value)]
    if return_values:
        average_return = float(np.mean(return_values))
        return_evidence_score = 50.0 + average_return * 500.0
        return_evidence_score = max(0.0, min(100.0, return_evidence_score))
    else:
        average_return = np.nan
        return_evidence_score = 35.0

    price_health_score = float(price_health.get("score", 50.0))
    regime_coverage_score = float(entry_backtest.get("regime_coverage_score", 50.0))
    recent_backtest_score = float(entry_backtest.get("recent_backtest_score", 50.0))
    backtest_decay_score = float(entry_backtest.get("backtest_decay_score", 50.0))
    score = (
        sample_score * 0.22
        + price_health_score * 0.18
        + liquidity_score * 0.14
        + slippage_score * 0.10
        + return_evidence_score * 0.14
        + regime_coverage_score * 0.07
        + recent_backtest_score * 0.07
        + backtest_decay_score * 0.08
    )
    score = round(float(max(0.0, min(score, 100.0))), 2)
    if score >= 80:
        level, level_zh = "high_trust", "可信度较高"
    elif score >= 65:
        level, level_zh = "usable", "可作为辅助"
    elif score >= 45:
        level, level_zh = "low_trust", "可信度偏低"
    else:
        level, level_zh = "not_enough_evidence", "证据不足"

    note = (
        f"Trust score blends sample count={total_count}, price health={price_health_score:.1f}, "
        f"avg dollar volume={_format_number(avg_dollar_volume)}, slippage={_format_percent(slippage_pct)}, "
        f"average entry return={_format_percent(average_return)}, and regime coverage="
        f"{regime_coverage_score:.1f}, recent strength={recent_backtest_score:.1f}, "
        f"decay check={backtest_decay_score:.1f}."
    )
    note_zh = (
        f"可信度分综合样本数={total_count}、价格健康分={price_health_score:.1f}、"
        f"平均成交额={_format_number(avg_dollar_volume)}、滑点={_format_percent(slippage_pct)}、"
        f"平均买点收益={_format_percent(average_return)}、行情覆盖分={regime_coverage_score:.1f}、"
        f"近期强度分={recent_backtest_score:.1f}、衰退检查分={backtest_decay_score:.1f}。"
    )
    return {
        "score": score,
        "level": level,
        "level_zh": level_zh,
        "sample_score": round(float(sample_score), 2),
        "liquidity_score": round(float(liquidity_score), 2),
        "slippage_score": round(float(slippage_score), 2),
        "return_evidence_score": round(float(return_evidence_score), 2),
        "regime_coverage_score": round(float(regime_coverage_score), 2),
        "recent_backtest_score": round(float(recent_backtest_score), 2),
        "backtest_decay_score": round(float(backtest_decay_score), 2),
        "note": note,
        "note_zh": note_zh,
    }


def _entry_backtest_frame(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    for column in [
        "backtest_slippage_pct",
        "backtest_atr_ratio",
        "regime_coverage_dominant_share",
        "recent_backtest_win_rate",
        "recent_backtest_average_return",
        "recent_backtest_return_delta",
        "backtest_decay_early_win_rate",
        "backtest_decay_late_win_rate",
        "backtest_decay_early_average_return",
        "backtest_decay_late_average_return",
        "backtest_decay_average_return_delta",
        "breakout_win_rate",
        "breakout_target_hit_rate",
        "breakout_stop_hit_rate",
        "breakout_average_gain",
        "breakout_average_loss",
        "breakout_average_return",
        "pullback_win_rate",
        "pullback_target_hit_rate",
        "pullback_stop_hit_rate",
        "pullback_average_gain",
        "pullback_average_loss",
        "pullback_average_return",
    ]:
        display[column] = display[column].map(_format_percent)
    return display
