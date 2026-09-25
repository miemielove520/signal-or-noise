from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import BacktestConfig


def run_backtest(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    config: BacktestConfig,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if config.execution_lag_days < 1:
        raise ValueError("backtest.execution_lag_days must be at least 1.")

    returns = prices.sort_values(["ticker", "date"]).copy()
    returns["asset_return"] = returns.groupby("ticker")["adj_close"].pct_change()
    returns = returns[["date", "ticker", "asset_return"]].dropna()

    all_dates = pd.DatetimeIndex(sorted(prices["date"].unique()))
    weight_matrix = (
        target_weights.pivot(index="date", columns="ticker", values="weight")
        .reindex(all_dates)
        .ffill()
        .fillna(0.0)
    )

    active_weights = weight_matrix.shift(config.execution_lag_days).fillna(0.0)
    returns_matrix = returns.pivot(index="date", columns="ticker", values="asset_return")
    returns_matrix = returns_matrix.reindex(all_dates).fillna(0.0)
    active_weights = active_weights.reindex(columns=returns_matrix.columns, fill_value=0.0)

    gross_returns = (active_weights * returns_matrix).sum(axis=1)
    turnover = active_weights.diff().abs().sum(axis=1).fillna(active_weights.abs().sum(axis=1))
    costs = turnover * (config.transaction_cost_bps / 10_000.0)
    net_returns = gross_returns - costs

    equity = config.initial_capital * (1.0 + net_returns).cumprod()
    curve = pd.DataFrame(
        {
            "date": all_dates,
            "gross_return": gross_returns.to_numpy(),
            "transaction_cost": costs.to_numpy(),
            "net_return": net_returns.to_numpy(),
            "equity": equity.to_numpy(),
            "turnover": turnover.to_numpy(),
            "gross_exposure": active_weights.sum(axis=1).to_numpy(),
        }
    )
    metrics = summarize_performance(curve, config)
    return curve, metrics


def summarize_performance(curve: pd.DataFrame, config: BacktestConfig) -> dict[str, float]:
    if curve.empty:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "annual_volatility": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "average_turnover": 0.0,
            "invested_days_ratio": 0.0,
            "ending_equity": config.initial_capital,
        }

    returns = curve["net_return"].fillna(0.0)
    periods = max(len(curve), 1)
    ending_equity = float(curve["equity"].iloc[-1])
    total_return = ending_equity / config.initial_capital - 1.0
    cagr = (1.0 + total_return) ** (config.annualization_days / periods) - 1.0
    annual_volatility = float(returns.std(ddof=0) * math.sqrt(config.annualization_days))
    excess_daily = returns - config.risk_free_rate / config.annualization_days
    sharpe = 0.0
    if annual_volatility > 0:
        sharpe = float(excess_daily.mean() / returns.std(ddof=0) * math.sqrt(config.annualization_days))

    running_max = curve["equity"].cummax()
    drawdown = curve["equity"] / running_max - 1.0
    max_drawdown = float(drawdown.min())

    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "annual_volatility": annual_volatility,
        "sharpe": sharpe if np.isfinite(sharpe) else 0.0,
        "max_drawdown": max_drawdown,
        "average_turnover": float(curve["turnover"].mean()),
        "invested_days_ratio": float((curve["gross_exposure"] > 0).mean())
        if "gross_exposure" in curve.columns
        else 0.0,
        "ending_equity": ending_equity,
    }
