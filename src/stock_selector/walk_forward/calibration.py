"""Rule and profile threshold calibration, the overfitting-risk report and suggested screening config."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..screening_config import ScreeningConfig
from ..screening_config import ScreeningThresholds
from ..screening_config import default_screening_config
from ..screening_config import render_screening_config_toml

from ._common import (
    _coerce_bool,
    _first_text,
    _is_finite,
)
from .config import (
    DEFAULT_HIGH_PROBABILITY_THRESHOLDS,
    DEFAULT_MARKET_THRESHOLDS,
    DEFAULT_PROBABILITY_THRESHOLDS,
    DEFAULT_RELATIVE_STRENGTH_THRESHOLDS,
    DEFAULT_SIGNAL_THRESHOLDS,
    MIN_CALIBRATION_SAMPLE_COUNT,
)


def calibrate_walk_forward_rules(
    events: pd.DataFrame,
    target_window: int = 20,
    min_sample_count: int = 5,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["rule", "current_threshold", "suggested_threshold"])

    specs = [
        ("signal_score", 65.0, DEFAULT_SIGNAL_THRESHOLDS, "higher"),
        ("high_probability_score", 65.0, DEFAULT_HIGH_PROBABILITY_THRESHOLDS, "higher"),
        ("calibrated_win_probability", 0.58, DEFAULT_PROBABILITY_THRESHOLDS, "higher"),
        ("relative_strength_score", 45.0, DEFAULT_RELATIVE_STRENGTH_THRESHOLDS, "higher"),
        ("market_score", 55.0, DEFAULT_MARKET_THRESHOLDS, "higher"),
    ]
    rows: list[dict[str, object]] = []
    for column, current_threshold, candidates, direction in specs:
        best = _best_threshold(
            events=events,
            column=column,
            current_threshold=current_threshold,
            candidates=candidates,
            target_window=target_window,
            min_sample_count=min_sample_count,
            direction=direction,
        )
        rows.append(
            {
                "rule": column,
                "current_threshold": current_threshold,
                "suggested_threshold": best["threshold"],
                "sample_count": best["sample_count"],
                f"win_rate_{target_window}d": best["win_rate"],
                f"avg_return_{target_window}d": best["avg_return"],
                "recommendation": best["recommendation"],
                "recommendation_zh": best["recommendation_zh"],
            }
        )
    return pd.DataFrame(rows)


def calibrate_walk_forward_profiles(
    events: pd.DataFrame,
    screening_config: ScreeningConfig | None = None,
    target_window: int = 20,
    min_sample_count: int = 5,
) -> pd.DataFrame:
    columns = [
        "screening_profile",
        "screening_profile_zh",
        "rule",
        "current_threshold",
        "suggested_threshold",
        "sample_count",
        f"win_rate_{target_window}d",
        f"avg_return_{target_window}d",
        "recommendation",
        "recommendation_zh",
        "suggestion_confidence",
        "suggestion_confidence_zh",
        "suggestion_note",
        "suggestion_note_zh",
    ]
    if events.empty or "screening_profile" not in events.columns:
        return pd.DataFrame(columns=columns)

    config = screening_config or default_screening_config()
    profile_thresholds = {
        "default": config.default_thresholds,
        **{profile.name: profile.thresholds for profile in config.profiles},
    }
    profile_names_zh = {
        "default": "默认规则",
        **{profile.name: profile.name_zh for profile in config.profiles},
    }
    specs = [
        (
            "signal_score",
            "signal_score_min",
            DEFAULT_SIGNAL_THRESHOLDS,
            "higher",
        ),
        (
            "high_probability_score",
            "high_probability_target_score",
            DEFAULT_HIGH_PROBABILITY_THRESHOLDS,
            "higher",
        ),
        (
            "calibrated_win_probability",
            None,
            DEFAULT_PROBABILITY_THRESHOLDS,
            "higher",
        ),
        (
            "relative_strength_score",
            "relative_strength_min",
            DEFAULT_RELATIVE_STRENGTH_THRESHOLDS,
            "higher",
        ),
        (
            "market_score",
            "market_score_min",
            DEFAULT_MARKET_THRESHOLDS,
            "higher",
        ),
    ]
    rows: list[dict[str, object]] = []
    for profile_name, group in events.groupby("screening_profile", dropna=False):
        profile_key = str(profile_name)
        thresholds = profile_thresholds.get(profile_key, config.default_thresholds)
        profile_zh = profile_names_zh.get(
            profile_key,
            _first_text(group, "screening_profile_zh", "默认规则"),
        )
        for column, threshold_attr, candidates, direction in specs:
            if column not in group.columns:
                continue
            current_threshold = (
                0.58 if threshold_attr is None else float(getattr(thresholds, threshold_attr))
            )
            best = _best_threshold(
                events=group,
                column=column,
                current_threshold=current_threshold,
                candidates=candidates,
                target_window=target_window,
                min_sample_count=min_sample_count,
                direction=direction,
            )
            confidence = _suggestion_confidence(
                sample_count=int(best["sample_count"]),
                win_rate=best["win_rate"],
                avg_return=best["avg_return"],
            )
            rows.append(
                {
                    "screening_profile": profile_key,
                    "screening_profile_zh": profile_zh,
                    "rule": column,
                    "current_threshold": current_threshold,
                    "suggested_threshold": best["threshold"],
                    "sample_count": best["sample_count"],
                    f"win_rate_{target_window}d": best["win_rate"],
                    f"avg_return_{target_window}d": best["avg_return"],
                    "recommendation": best["recommendation"],
                    "recommendation_zh": best["recommendation_zh"],
                    "suggestion_confidence": confidence["level"],
                    "suggestion_confidence_zh": confidence["level_zh"],
                    "suggestion_note": confidence["note"],
                    "suggestion_note_zh": confidence["note_zh"],
                }
            )
    return pd.DataFrame(rows, columns=columns)


def build_overfitting_risk_report(
    profile_calibration: pd.DataFrame,
    min_sample_count: int = MIN_CALIBRATION_SAMPLE_COUNT,
) -> pd.DataFrame:
    """Per-profile overfitting exposure: how many thresholds are being tuned vs. how
    much evidence supports them. A profile tuning many thresholds on few samples is
    the classic overfitting trap."""
    columns = [
        "screening_profile",
        "tunable_param_count",
        "adoptable_param_count",
        "median_sample_count",
        "sample_to_param_ratio",
        "overfit_risk_level",
        "overfit_risk_level_zh",
        "note",
        "note_zh",
    ]
    if (
        profile_calibration is None
        or profile_calibration.empty
        or "screening_profile" not in profile_calibration.columns
    ):
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for profile, group in profile_calibration.groupby("screening_profile"):
        samples = pd.to_numeric(group.get("sample_count"), errors="coerce").fillna(0.0)
        tunable = int(len(group))
        adoptable = int((samples >= min_sample_count).sum())
        median_samples = float(samples.median()) if len(samples) else 0.0
        ratio = round(median_samples / tunable, 2) if tunable else 0.0
        level, level_zh = _overfit_risk_level(adoptable, tunable, median_samples, min_sample_count)
        rows.append(
            {
                "screening_profile": profile,
                "tunable_param_count": tunable,
                "adoptable_param_count": adoptable,
                "median_sample_count": round(median_samples, 1),
                "sample_to_param_ratio": ratio,
                "overfit_risk_level": level,
                "overfit_risk_level_zh": level_zh,
                "note": (
                    f"{adoptable}/{tunable} thresholds have >= {min_sample_count} samples; "
                    f"median sample count {median_samples:.0f}."
                ),
                "note_zh": (
                    f"{tunable}个可调阈值中{adoptable}个样本达到{min_sample_count}；"
                    f"样本中位数{median_samples:.0f}。"
                ),
            }
        )
    result = pd.DataFrame(rows, columns=columns)
    order = {"high": 0, "medium": 1, "low": 2}
    result["_r"] = result["overfit_risk_level"].map(order).fillna(9)
    return result.sort_values(["_r", "screening_profile"]).drop(columns="_r").reset_index(drop=True)


def _overfit_risk_level(
    adoptable: int, tunable: int, median_samples: float, min_sample_count: int
) -> tuple[str, str]:
    if adoptable == 0 or median_samples < min_sample_count / 2:
        return "high", "过拟合风险高"
    if adoptable < tunable or median_samples < min_sample_count:
        return "medium", "过拟合风险中"
    return "low", "过拟合风险低"


def render_suggested_screening_config(
    screening_config: ScreeningConfig,
    profile_calibration: pd.DataFrame,
    benchmark_policy: pd.DataFrame | None = None,
    benchmark_tightening: pd.DataFrame | None = None,
    minimum_sample_guard: pd.DataFrame | None = None,
    historical_threshold_recommendations: pd.DataFrame | None = None,
    min_sample_count: int = MIN_CALIBRATION_SAMPLE_COUNT,
) -> str:
    suggested_thresholds = {
        profile.name: profile.thresholds for profile in screening_config.profiles
    }
    suggested_thresholds["default"] = screening_config.default_thresholds
    applied_notes: list[str] = []
    skipped_notes: list[str] = []
    benchmark_guard = _benchmark_relaxation_guard(benchmark_policy)
    benchmark_tightening_notes = _apply_benchmark_tightening_recommendations(
        suggested_thresholds=suggested_thresholds,
        benchmark_tightening=benchmark_tightening,
        minimum_sample_guard=minimum_sample_guard,
    )
    applied_notes.extend(benchmark_tightening_notes)
    historical_threshold_notes = _apply_historical_threshold_recommendations(
        suggested_thresholds=suggested_thresholds,
        recommendations=historical_threshold_recommendations,
    )
    applied_notes.extend(historical_threshold_notes)

    for row in profile_calibration.itertuples(index=False):
        profile_name = str(row.screening_profile)
        threshold_attr = _threshold_attr_for_rule(str(row.rule))
        if threshold_attr is None:
            continue
        sample_count = int(row.sample_count)
        suggested_value = float(row.suggested_threshold)
        confidence = str(getattr(row, "suggestion_confidence", "insufficient"))
        if (
            sample_count < min_sample_count
            or confidence not in {"medium", "high"}
            or not _is_finite(suggested_value)
        ):
            skipped_notes.append(
                f"{profile_name}.{threshold_attr}: skipped; "
                f"sample_count={sample_count}; confidence={confidence}"
            )
            continue
        current = suggested_thresholds.get(profile_name, screening_config.default_thresholds)
        current_value = float(getattr(current, threshold_attr))
        if suggested_value < current_value and benchmark_guard["block_relaxation"]:
            skipped_notes.append(
                f"{profile_name}.{threshold_attr}: skipped relaxation by benchmark guard; "
                f"{current_value:g} -> {suggested_value:g}; policy={benchmark_guard['policy']}"
            )
            continue
        updated = ScreeningThresholds.from_mapping(
            {
                **current.to_dict(),
                threshold_attr: suggested_value,
            }
        )
        suggested_thresholds[profile_name] = updated
        applied_notes.append(
            f"{profile_name}.{threshold_attr}: {getattr(current, threshold_attr):g} -> "
            f"{suggested_value:g}; confidence={confidence}"
        )

    suggested_config = screening_config.with_profile_thresholds(suggested_thresholds)
    header = [
        "Suggested screening config generated from walk-forward validation.",
        "Review manually before replacing configs/screening.toml.",
        f"Benchmark guard: {benchmark_guard['policy']}",
        f"Benchmark guard note: {benchmark_guard['note']}",
        f"Benchmark tightening suggestions applied: {len(benchmark_tightening_notes)}",
        f"Historical threshold suggestions applied: {len(historical_threshold_notes)}",
        f"Minimum sample guard: {_minimum_sample_guard_summary(minimum_sample_guard)}",
        f"Applied suggestions: {len(applied_notes)}",
        f"Skipped suggestions: {len(skipped_notes)}",
    ]
    if applied_notes:
        header.extend(["", "Applied:"])
        header.extend(f"- {note}" for note in applied_notes)
    if skipped_notes:
        header.extend(["", "Skipped:"])
        header.extend(f"- {note}" for note in skipped_notes[:20])
    return render_screening_config_toml(suggested_config, header_lines=header)


def _apply_historical_threshold_recommendations(
    suggested_thresholds: dict[str, ScreeningThresholds],
    recommendations: pd.DataFrame | None,
) -> list[str]:
    if recommendations is None or recommendations.empty:
        return []
    required = {
        "scope",
        "group_value",
        "threshold_attr",
        "suggested_threshold",
        "priority",
        "allow_auto_apply",
    }
    if not required.issubset(recommendations.columns):
        return []

    notes: list[str] = []
    for row in recommendations.itertuples(index=False):
        if str(row.scope) != "screening_profile":
            continue
        if str(row.priority) not in {"high", "medium"}:
            continue
        if not _coerce_bool(getattr(row, "allow_auto_apply", False)):
            continue
        profile_name = str(row.group_value)
        threshold_attr = str(row.threshold_attr)
        suggested_value = float(row.suggested_threshold)
        current = suggested_thresholds.get(profile_name)
        if current is None or not hasattr(current, threshold_attr) or not _is_finite(suggested_value):
            continue
        current_value = float(getattr(current, threshold_attr))
        if not _historical_suggestion_is_tighter(
            threshold_attr=threshold_attr,
            current_value=current_value,
            suggested_value=suggested_value,
        ):
            continue
        updated = ScreeningThresholds.from_mapping(
            {
                **current.to_dict(),
                threshold_attr: suggested_value,
            }
        )
        suggested_thresholds[profile_name] = updated
        notes.append(
            f"{profile_name}.{threshold_attr}: {current_value:g} -> {suggested_value:g}; "
            f"historical win-rate gate; priority={row.priority}"
        )
    return notes


def _historical_suggestion_is_tighter(
    threshold_attr: str,
    current_value: float,
    suggested_value: float,
) -> bool:
    lower_is_tighter = {"max_backtest_stop_hit_rate", "max_backtest_slippage_pct"}
    if threshold_attr in lower_is_tighter:
        return suggested_value < current_value
    return suggested_value > current_value


def _apply_benchmark_tightening_recommendations(
    suggested_thresholds: dict[str, ScreeningThresholds],
    benchmark_tightening: pd.DataFrame | None,
    minimum_sample_guard: pd.DataFrame | None = None,
) -> list[str]:
    if benchmark_tightening is None or benchmark_tightening.empty:
        return []
    required = {"screening_profile", "threshold_attr", "suggested_threshold", "priority"}
    if not required.issubset(benchmark_tightening.columns):
        return []

    notes: list[str] = []
    for row in benchmark_tightening.itertuples(index=False):
        priority = str(getattr(row, "priority", "low"))
        if priority not in {"high", "medium"}:
            continue
        profile_name = str(row.screening_profile)
        threshold_attr = str(row.threshold_attr)
        suggested_value = float(row.suggested_threshold)
        if not _minimum_sample_guard_allows(
            minimum_sample_guard=minimum_sample_guard,
            profile_name=profile_name,
            threshold_attr=threshold_attr,
            suggested_value=suggested_value,
        ):
            continue
        current = suggested_thresholds.get(profile_name)
        if current is None or not hasattr(current, threshold_attr) or not _is_finite(suggested_value):
            continue
        current_value = float(getattr(current, threshold_attr))
        if suggested_value <= current_value:
            continue
        updated = ScreeningThresholds.from_mapping(
            {
                **current.to_dict(),
                threshold_attr: suggested_value,
            }
        )
        suggested_thresholds[profile_name] = updated
        notes.append(
            f"{profile_name}.{threshold_attr}: {current_value:g} -> {suggested_value:g}; "
            f"benchmark tightening; priority={priority}"
        )
    return notes


def _minimum_sample_guard_allows(
    minimum_sample_guard: pd.DataFrame | None,
    profile_name: str,
    threshold_attr: str,
    suggested_value: float,
) -> bool:
    if minimum_sample_guard is None or minimum_sample_guard.empty:
        return True
    required = {"screening_profile", "threshold_attr", "threshold_value", "allow_adoption"}
    if not required.issubset(minimum_sample_guard.columns):
        return True
    frame = minimum_sample_guard[
        (minimum_sample_guard["screening_profile"].astype(str) == profile_name)
        & (minimum_sample_guard["threshold_attr"].astype(str) == threshold_attr)
    ].copy()
    if frame.empty:
        return False
    frame["threshold_value"] = pd.to_numeric(frame["threshold_value"], errors="coerce")
    matching = frame[(frame["threshold_value"] - suggested_value).abs() < 1e-9]
    if matching.empty:
        return False
    return bool(matching["allow_adoption"].fillna(False).astype(bool).any())


def _minimum_sample_guard_summary(minimum_sample_guard: pd.DataFrame | None) -> str:
    if minimum_sample_guard is None or minimum_sample_guard.empty or "guard_action" not in minimum_sample_guard.columns:
        return "not_available"
    counts = minimum_sample_guard["guard_action"].astype(str).value_counts().to_dict()
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def _benchmark_relaxation_guard(benchmark_policy: pd.DataFrame | None) -> dict[str, object]:
    if benchmark_policy is None or benchmark_policy.empty or "allow_relaxation" not in benchmark_policy.columns:
        return {
            "block_relaxation": False,
            "policy": "no_benchmark_policy",
            "note": "Benchmark policy was not available.",
        }
    allow_relaxation = benchmark_policy["allow_relaxation"].fillna(False).astype(bool)
    actions = ";".join(benchmark_policy["benchmark_policy_action"].astype(str).unique())
    notes = "; ".join(benchmark_policy["policy_note"].astype(str).dropna().unique()[:3])
    if allow_relaxation.any():
        return {
            "block_relaxation": False,
            "policy": actions,
            "note": notes or "At least one portfolio policy allows selective relaxation.",
        }
    return {
        "block_relaxation": True,
        "policy": actions,
        "note": notes or "Benchmark evidence does not support relaxing thresholds.",
    }


def _threshold_attr_for_rule(rule: str) -> str | None:
    return {
        "signal_score": "signal_score_min",
        "high_probability_score": "high_probability_target_score",
        "relative_strength_score": "relative_strength_min",
        "market_score": "market_score_min",
    }.get(rule)


def _suggestion_confidence(
    sample_count: int,
    win_rate: float,
    avg_return: float,
) -> dict[str, str]:
    if sample_count < 5 or not _is_finite(win_rate) or not _is_finite(avg_return):
        return {
            "level": "insufficient",
            "level_zh": "样本不足",
            "note": "Too few validation samples to trust this suggestion.",
            "note_zh": "验证样本太少，暂时不能信任该建议。",
        }
    if sample_count >= 30 and float(avg_return) > 0 and float(win_rate) >= 0.55:
        return {
            "level": "high",
            "level_zh": "高",
            "note": "Sample count, win rate, and average return are supportive.",
            "note_zh": "样本数、胜率和平均收益都支持该建议。",
        }
    if sample_count >= 15 and float(avg_return) > 0:
        return {
            "level": "medium",
            "level_zh": "中",
            "note": "Sample count is acceptable and average return is positive.",
            "note_zh": "样本数基本可用，平均收益为正。",
        }
    if sample_count >= 5:
        return {
            "level": "low",
            "level_zh": "低",
            "note": "Suggestion is based on limited or weak validation performance.",
            "note_zh": "该建议基于有限样本或表现不够强，只能参考。",
        }
    return {
        "level": "insufficient",
        "level_zh": "样本不足",
        "note": "Too few validation samples to trust this suggestion.",
        "note_zh": "验证样本太少，暂时不能信任该建议。",
    }


def _best_threshold(
    events: pd.DataFrame,
    column: str,
    current_threshold: float,
    candidates: tuple[int, ...],
    target_window: int,
    min_sample_count: int,
    direction: str,
) -> dict[str, object]:
    return_column = f"forward_return_{target_window}d"
    candidate_rows: list[dict[str, object]] = []
    for threshold in candidates:
        if direction == "higher":
            subset = events[events[column] >= threshold]
        else:
            subset = events[events[column] <= threshold]
        returns = subset[return_column].dropna()
        sample_count = len(returns)
        win_rate = float((returns > 0).mean()) if sample_count else np.nan
        avg_return = float(returns.mean()) if sample_count else np.nan
        candidate_rows.append(
            {
                "threshold": float(threshold),
                "sample_count": sample_count,
                "win_rate": win_rate,
                "avg_return": avg_return,
            }
        )

    eligible = [row for row in candidate_rows if row["sample_count"] >= min_sample_count]
    if not eligible:
        return {
            "threshold": np.nan,
            "sample_count": 0,
            "win_rate": np.nan,
            "avg_return": np.nan,
            "recommendation": "insufficient validation samples",
            "recommendation_zh": "验证样本不足，暂不建议调整",
        }

    best = sorted(
        eligible,
        key=lambda row: (
            -1 if not _is_finite(row["win_rate"]) else row["win_rate"],
            -1 if not _is_finite(row["avg_return"]) else row["avg_return"],
            row["sample_count"],
        ),
        reverse=True,
    )[0]
    if best["threshold"] > current_threshold:
        recommendation = "tighten threshold"
        recommendation_zh = "建议提高阈值"
    elif best["threshold"] < current_threshold:
        recommendation = "loosen threshold"
        recommendation_zh = "建议放宽阈值"
    else:
        recommendation = "keep threshold"
        recommendation_zh = "建议维持阈值"
    return {
        "threshold": best["threshold"],
        "sample_count": best["sample_count"],
        "win_rate": best["win_rate"],
        "avg_return": best["avg_return"],
        "recommendation": recommendation,
        "recommendation_zh": recommendation_zh,
    }
