from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import RiskConfig


def select_rebalance_dates(dates: pd.Series, frequency: str) -> pd.DatetimeIndex:
    unique_dates = pd.Series(pd.to_datetime(dates).drop_duplicates()).sort_values()
    if unique_dates.empty:
        return pd.DatetimeIndex([])

    date_frame = pd.DataFrame({"date": unique_dates})
    periods = date_frame["date"].dt.to_period(frequency)
    rebalance_dates = date_frame.groupby(periods)["date"].max()
    return pd.DatetimeIndex(rebalance_dates.to_list())


def build_equal_weight_portfolio(
    scored: pd.DataFrame,
    top_n: int,
    rebalance_frequency: str,
) -> pd.DataFrame:
    rebalance_dates = select_rebalance_dates(scored["date"], rebalance_frequency)
    candidates = scored[scored["date"].isin(rebalance_dates)].copy()
    candidates = candidates.dropna(subset=["score"])
    candidates["rank"] = candidates.groupby("date")["score"].rank(
        ascending=False, method="first"
    )
    selected = candidates[candidates["rank"] <= top_n].copy()
    if selected.empty:
        return pd.DataFrame(columns=["date", "ticker", "weight", "score", "rank"])

    counts = selected.groupby("date")["ticker"].transform("count")
    selected["weight"] = 1.0 / counts
    return selected[_selection_columns(selected)].sort_values(["date", "rank"])


def build_risk_managed_portfolio(
    scored: pd.DataFrame,
    prices: pd.DataFrame,
    top_n: int,
    rebalance_frequency: str,
    risk_config: RiskConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    method = risk_config.weighting_method.lower().strip()
    if method not in {"equal", "inverse_volatility"}:
        raise ValueError(f"Unsupported risk weighting_method: {risk_config.weighting_method}")

    rebalance_dates = select_rebalance_dates(scored["date"], rebalance_frequency)
    candidates = scored[scored["date"].isin(rebalance_dates)].copy()
    candidates = candidates.dropna(subset=["score"])
    candidates["rank"] = candidates.groupby("date")["score"].rank(
        ascending=False, method="first"
    )

    returns = prices.sort_values(["ticker", "date"]).copy()
    returns["asset_return"] = returns.groupby("ticker")["adj_close"].pct_change()
    returns_matrix = returns.pivot(index="date", columns="ticker", values="asset_return")
    returns_matrix = returns_matrix.sort_index()

    selection_parts: list[pd.DataFrame] = []
    report_rows: list[dict[str, object]] = []
    for rebalance_date, group in candidates[candidates["rank"] <= top_n].groupby("date"):
        selected = group.sort_values("rank").copy()
        history = returns_matrix[returns_matrix.index < rebalance_date].tail(
            risk_config.volatility_lookback_days
        )
        selected_tickers = selected["ticker"].to_list()
        if method == "equal":
            weights = pd.Series(1.0 / len(selected_tickers), index=selected_tickers)
        else:
            weights = _inverse_volatility_weights(history, selected_tickers)
        weights = _apply_weight_constraints(
            weights,
            max_weight=risk_config.max_position_weight,
            min_weight=risk_config.min_position_weight,
        )
        weights = _apply_sector_cap(weights, selected, risk_config)
        estimated_vol = _estimate_portfolio_volatility(
            history,
            weights,
            annualization_days=risk_config.annualization_days,
        )
        exposure_scale = _target_volatility_scale(
            estimated_vol,
            risk_config.target_annual_volatility,
        )
        weights = weights * exposure_scale
        scaled_estimated_vol = estimated_vol * exposure_scale

        selected["weight"] = selected["ticker"].map(weights).fillna(0.0)
        selection_parts.append(selected[_selection_columns(selected)])
        sector_summary = _sector_summary(weights, selected, risk_config)
        report_rows.append(
            {
                "date": rebalance_date,
                "selected_count": int(len(selected)),
                "gross_exposure": float(weights.sum()),
                "max_position_weight": float(weights.max()) if not weights.empty else 0.0,
                "max_sector_weight": sector_summary["max_sector_weight"],
                "largest_sector": sector_summary["largest_sector"],
                "estimated_annual_volatility": scaled_estimated_vol,
                "unscaled_estimated_annual_volatility": estimated_vol,
                "target_annual_volatility": risk_config.target_annual_volatility,
                "cash_weight": float(max(0.0, 1.0 - weights.sum())),
                "weighting_method": method,
            }
        )

    selections = (
        pd.concat(selection_parts, ignore_index=True)
        if selection_parts
        else pd.DataFrame(columns=["date", "ticker", "weight", "score", "rank"])
    )
    report = pd.DataFrame(report_rows)
    return selections.sort_values(["date", "rank"]).reset_index(drop=True), report


def summarize_portfolio_risk(
    selections: pd.DataFrame,
    prices: pd.DataFrame,
    risk_config: RiskConfig,
) -> pd.DataFrame:
    if selections.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "selected_count",
                "gross_exposure",
                "max_position_weight",
                "max_sector_weight",
                "largest_sector",
                "estimated_annual_volatility",
                "unscaled_estimated_annual_volatility",
                "target_annual_volatility",
                "cash_weight",
                "weighting_method",
            ]
        )

    returns = prices.sort_values(["ticker", "date"]).copy()
    returns["asset_return"] = returns.groupby("ticker")["adj_close"].pct_change()
    returns_matrix = returns.pivot(index="date", columns="ticker", values="asset_return")
    returns_matrix = returns_matrix.sort_index()

    rows: list[dict[str, object]] = []
    for rebalance_date, group in selections.groupby("date"):
        weights = group.set_index("ticker")["weight"].astype(float)
        history = returns_matrix[returns_matrix.index < rebalance_date].tail(
            risk_config.volatility_lookback_days
        )
        estimated_vol = _estimate_portfolio_volatility(
            history,
            weights,
            annualization_days=risk_config.annualization_days,
        )
        sector_summary = _sector_summary(weights, group, risk_config)
        rows.append(
            {
                "date": rebalance_date,
                "selected_count": int(len(group)),
                "gross_exposure": float(weights.sum()),
                "max_position_weight": float(weights.max()) if not weights.empty else 0.0,
                "max_sector_weight": sector_summary["max_sector_weight"],
                "largest_sector": sector_summary["largest_sector"],
                "estimated_annual_volatility": estimated_vol,
                "unscaled_estimated_annual_volatility": estimated_vol,
                "target_annual_volatility": risk_config.target_annual_volatility,
                "cash_weight": float(max(0.0, 1.0 - weights.sum())),
                "weighting_method": risk_config.weighting_method,
            }
        )
    return pd.DataFrame(rows)


def summarize_exposures(
    selections: pd.DataFrame,
    group_column: str = "sector",
) -> pd.DataFrame:
    if selections.empty or group_column not in selections.columns:
        return pd.DataFrame(columns=["date", group_column, "weight", "selected_count"])
    report = (
        selections.groupby(["date", group_column], as_index=False)
        .agg(weight=("weight", "sum"), selected_count=("ticker", "count"))
        .sort_values(["date", "weight"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return report


def _inverse_volatility_weights(
    history: pd.DataFrame,
    tickers: list[str],
) -> pd.Series:
    if not tickers:
        return pd.Series(dtype=float)
    if history.empty:
        return pd.Series(1.0 / len(tickers), index=tickers)

    available = history.reindex(columns=tickers)
    volatility = available.std(ddof=0).replace(0, np.nan)
    inverse_vol = 1.0 / volatility
    inverse_vol = inverse_vol.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if inverse_vol.sum() <= 0:
        return pd.Series(1.0 / len(tickers), index=tickers)
    return inverse_vol / inverse_vol.sum()


def _apply_weight_constraints(
    weights: pd.Series,
    max_weight: float,
    min_weight: float,
) -> pd.Series:
    if weights.empty:
        return weights
    if max_weight <= 0:
        raise ValueError("risk.max_position_weight must be positive.")
    if min_weight < 0:
        raise ValueError("risk.min_position_weight cannot be negative.")
    if min_weight > max_weight:
        raise ValueError("risk.min_position_weight cannot exceed max_position_weight.")

    constrained = weights.astype(float).copy()
    if min_weight > 0:
        constrained = constrained.clip(lower=min_weight)
        constrained = constrained / constrained.sum()

    return _cap_and_redistribute(constrained, max_weight)


def _apply_sector_cap(
    weights: pd.Series,
    selected: pd.DataFrame,
    risk_config: RiskConfig,
) -> pd.Series:
    if risk_config.max_sector_weight is None:
        return weights
    if risk_config.max_sector_weight <= 0:
        raise ValueError("risk.max_sector_weight must be positive when provided.")
    if risk_config.sector_column not in selected.columns:
        return weights

    sectors = selected.set_index("ticker")[risk_config.sector_column].fillna("Unknown")
    constrained = weights.astype(float).copy()
    sector_weights = constrained.groupby(sectors).sum()
    for sector, sector_weight in sector_weights.items():
        if sector_weight > risk_config.max_sector_weight:
            tickers = sectors[sectors == sector].index
            scale = risk_config.max_sector_weight / sector_weight
            constrained.loc[tickers] = constrained.loc[tickers] * scale
    return constrained


def _sector_summary(
    weights: pd.Series,
    selected: pd.DataFrame,
    risk_config: RiskConfig,
) -> dict[str, object]:
    if weights.empty or risk_config.sector_column not in selected.columns:
        return {"max_sector_weight": 0.0, "largest_sector": "Unknown"}
    sectors = selected.set_index("ticker")[risk_config.sector_column].fillna("Unknown")
    sector_weights = weights.groupby(sectors).sum()
    if sector_weights.empty:
        return {"max_sector_weight": 0.0, "largest_sector": "Unknown"}
    largest_sector = str(sector_weights.idxmax())
    return {
        "max_sector_weight": float(sector_weights.max()),
        "largest_sector": largest_sector,
    }


def _cap_and_redistribute(weights: pd.Series, max_weight: float) -> pd.Series:
    constrained = weights.copy()
    active = pd.Series(True, index=constrained.index)

    for _ in range(len(constrained) + 1):
        capped = active & (constrained > max_weight)
        if not capped.any():
            break
        constrained.loc[capped] = max_weight
        active.loc[capped] = False
        remaining_total = 1.0 - constrained.loc[~active].sum()
        if remaining_total <= 0:
            constrained.loc[active] = 0.0
            return constrained
        if not active.any():
            return constrained
        active_sum = constrained.loc[active].sum()
        if active_sum <= 0:
            constrained.loc[active] = remaining_total / int(active.sum())
        else:
            constrained.loc[active] = constrained.loc[active] / active_sum * remaining_total

    if constrained.max() > max_weight:
        constrained = constrained.clip(upper=max_weight)
    return constrained


def _estimate_portfolio_volatility(
    history: pd.DataFrame,
    weights: pd.Series,
    annualization_days: int,
) -> float:
    if history.empty or weights.empty:
        return 0.0
    aligned = history.reindex(columns=weights.index).fillna(0.0)
    portfolio_returns = aligned.mul(weights, axis=1).sum(axis=1)
    daily_vol = float(portfolio_returns.std(ddof=0))
    if not math.isfinite(daily_vol):
        return 0.0
    return daily_vol * math.sqrt(annualization_days)


def _target_volatility_scale(
    estimated_volatility: float,
    target_volatility: float | None,
) -> float:
    if target_volatility is None or target_volatility <= 0 or estimated_volatility <= 0:
        return 1.0
    return min(1.0, target_volatility / estimated_volatility)


def _selection_columns(selected: pd.DataFrame) -> list[str]:
    columns = ["date", "ticker", "weight", "score", "rank"]
    for column in ["sector", "industry", "country", "exchange"]:
        if column in selected.columns:
            columns.append(column)
    return columns
