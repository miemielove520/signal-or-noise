"""Holding-horizon specifications (short / medium / long) and their trading rules."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import Iterable

import pandas as pd

from ..screening_config import TradingRules



@dataclass(frozen=True)
class HorizonSpec:
    name: str
    label: str
    zh_label: str
    time_range: str
    time_range_zh: str
    lookback_days: int
    trend_window: int
    momentum_window: int
    volume_window: int
    atr_window: int
    atr_stop_multiple: float
    target_r_multiple: float
    max_chase_pct: float
    time_stop_days: int
    trailing_stop_trigger_r: float
    trailing_stop_lock_r: float
    min_history_days: int


# Under the "breakout" entry style, a confirmed breakout stays executable even when the
# stock is extended above its baseline, up to this multiple of the horizon's max_chase_pct.
# Beyond it the move is treated as parabolic and we wait for a pullback instead.
BREAKOUT_EXTENSION_MULTIPLE = 2.5


HORIZON_SPECS: dict[str, HorizonSpec] = {
    "short": HorizonSpec(
        name="short",
        label="short term",
        zh_label="短期",
        time_range="short-term trading setup",
        time_range_zh="短线交易节奏",
        lookback_days=20,
        trend_window=20,
        momentum_window=5,
        volume_window=10,
        atr_window=14,
        atr_stop_multiple=1.5,
        target_r_multiple=2.0,
        max_chase_pct=0.04,
        time_stop_days=20,
        trailing_stop_trigger_r=1.5,
        trailing_stop_lock_r=0.2,
        min_history_days=30,
    ),
    "medium": HorizonSpec(
        name="medium",
        label="medium term",
        zh_label="中期",
        time_range="trend continuation setup",
        time_range_zh="趋势延续判断",
        lookback_days=60,
        trend_window=50,
        momentum_window=20,
        volume_window=20,
        atr_window=14,
        atr_stop_multiple=2.0,
        target_r_multiple=2.5,
        max_chase_pct=0.08,
        time_stop_days=60,
        trailing_stop_trigger_r=1.5,
        trailing_stop_lock_r=0.2,
        min_history_days=75,
    ),
    "long": HorizonSpec(
        name="long",
        label="long term",
        zh_label="长期",
        time_range="long-term quality and trend setup",
        time_range_zh="长期质量与趋势判断",
        lookback_days=120,
        trend_window=100,
        momentum_window=60,
        volume_window=30,
        atr_window=20,
        atr_stop_multiple=3.0,
        target_r_multiple=3.0,
        max_chase_pct=0.15,
        time_stop_days=120,
        trailing_stop_trigger_r=1.5,
        trailing_stop_lock_r=0.2,
        min_history_days=140,
    ),
}


def _effective_horizon_spec(spec: HorizonSpec, trading_rules: TradingRules) -> HorizonSpec:
    return replace(
        spec,
        atr_stop_multiple=trading_rules.atr_stop_multiple_for(spec.name),
        target_r_multiple=trading_rules.target_r_multiple_for(spec.name),
        max_chase_pct=trading_rules.max_chase_pct_for(spec.name),
        time_stop_days=trading_rules.time_stop_days_for(spec.name),
        trailing_stop_trigger_r=trading_rules.trailing_stop_trigger_r,
        trailing_stop_lock_r=trading_rules.trailing_stop_lock_r,
    )


def normalize_horizons(horizons: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(horizons, str):
        raw = [horizons]
    else:
        raw = list(horizons)

    normalized: list[str] = []
    for horizon in raw:
        value = str(horizon).lower().strip()
        if value == "all":
            for name in HORIZON_SPECS:
                if name not in normalized:
                    normalized.append(name)
            continue
        if value not in HORIZON_SPECS:
            choices = ", ".join([*HORIZON_SPECS.keys(), "all"])
            raise ValueError(f"Unsupported horizon '{horizon}'. Choices: {choices}.")
        if value not in normalized:
            normalized.append(value)
    if not normalized:
        raise ValueError("At least one horizon is required.")
    return tuple(normalized)


def _trading_rule_note(spec: HorizonSpec, trading_rules: TradingRules) -> str:
    return (
        f"{spec.zh_label}/{spec.name} uses {trading_rules.preferred_entry_style} entries, "
        f"ATR stop {spec.atr_stop_multiple:.2f}x, target {spec.target_r_multiple:.2f}R, "
        f"max chase {spec.max_chase_pct:.2%}, time stop {spec.time_stop_days} days, "
        f"trailing trigger {spec.trailing_stop_trigger_r:.2f}R, "
        f"and sell rule: {trading_rules.sell_rule}."
    )


def _trading_rule_note_zh(spec: HorizonSpec, trading_rules: TradingRules) -> str:
    return (
        f"{spec.zh_label}使用{trading_rules.preferred_entry_style_zh}，"
        f"ATR止损{spec.atr_stop_multiple:.2f}倍，目标{spec.target_r_multiple:.2f}R，"
        f"最大追高{spec.max_chase_pct:.2%}，时间止损{spec.time_stop_days}天，"
        f"移动止损触发{spec.trailing_stop_trigger_r:.2f}R，"
        f"卖出规则：{trading_rules.sell_rule_zh}。"
    )


def _horizon_definition_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "horizon": spec.name,
                "label": spec.label,
                "zh_label": spec.zh_label,
                "time_range": spec.time_range,
                "time_range_zh": spec.time_range_zh,
            }
            for spec in HORIZON_SPECS.values()
        ]
    )
