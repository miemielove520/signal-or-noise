from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .backtest import run_backtest
from .config import ResearchConfig
from .data import load_fundamental_csv, load_macro_csv, load_metadata_csv, load_price_csv
from .factors import add_basic_factors, score_factors
from .ml import RollingMLResult, build_ml_dataset, rolling_train_predict
from .portfolio import build_risk_managed_portfolio, summarize_exposures


@dataclass(frozen=True)
class PipelineResult:
    prices: pd.DataFrame
    scored: pd.DataFrame
    selections: pd.DataFrame
    risk_report: pd.DataFrame
    exposure_report: pd.DataFrame
    equity_curve: pd.DataFrame
    metrics: dict[str, float]


@dataclass(frozen=True)
class MLPipelineResult:
    prices: pd.DataFrame
    scored: pd.DataFrame
    ml: RollingMLResult
    selections: pd.DataFrame
    risk_report: pd.DataFrame
    exposure_report: pd.DataFrame
    equity_curve: pd.DataFrame
    backtest_metrics: dict[str, float]


def build_scored_features(config: ResearchConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = load_price_csv(config.data)
    fundamentals = (
        load_fundamental_csv(config.data.fundamentals_csv)
        if config.data.fundamentals_csv
        else None
    )
    macro = load_macro_csv(config.data.macro_csv) if config.data.macro_csv else None
    factors = add_basic_factors(
        prices,
        config.factors,
        config.universe,
        fundamentals=fundamentals,
        macro=macro,
    )
    scored = score_factors(factors, config.scoring.factor_weights)
    if config.data.metadata_csv:
        metadata = load_metadata_csv(config.data.metadata_csv)
        scored = scored.merge(metadata, on="ticker", how="left")
        for column in ["sector", "industry", "country", "exchange"]:
            scored[column] = scored[column].fillna("Unknown")
    return prices, scored


def run_research_pipeline(config: ResearchConfig) -> PipelineResult:
    prices, scored = build_scored_features(config)
    selections, risk_report = build_risk_managed_portfolio(
        scored,
        prices,
        top_n=config.scoring.top_n,
        rebalance_frequency=config.scoring.rebalance_frequency,
        risk_config=config.risk,
    )
    exposure_report = summarize_exposures(selections, config.risk.sector_column)
    equity_curve, metrics = run_backtest(prices, selections, config.backtest)
    return PipelineResult(
        prices=prices,
        scored=scored,
        selections=selections,
        risk_report=risk_report,
        exposure_report=exposure_report,
        equity_curve=equity_curve,
        metrics=metrics,
    )


def run_ml_pipeline(config: ResearchConfig) -> MLPipelineResult:
    prices, scored = build_scored_features(config)
    dataset = build_ml_dataset(
        scored,
        feature_columns=config.ml.feature_columns,
        label_forward_days=config.ml.label_forward_days,
    )
    ml_result = rolling_train_predict(dataset, config.ml)

    prediction_scores = ml_result.predictions.rename(columns={"ml_prediction": "score"})
    metadata_columns = [
        column
        for column in ["sector", "industry", "country", "exchange"]
        if column in scored.columns
    ]
    if metadata_columns and not prediction_scores.empty:
        prediction_scores = prediction_scores.merge(
            scored[["date", "ticker", *metadata_columns]].drop_duplicates(),
            on=["date", "ticker"],
            how="left",
        )
    selections, risk_report = build_risk_managed_portfolio(
        prediction_scores,
        prices,
        top_n=config.scoring.top_n,
        rebalance_frequency=config.scoring.rebalance_frequency,
        risk_config=config.risk,
    )
    exposure_report = summarize_exposures(selections, config.risk.sector_column)
    equity_curve, backtest_metrics = run_backtest(prices, selections, config.backtest)
    return MLPipelineResult(
        prices=prices,
        scored=scored,
        ml=ml_result,
        selections=selections,
        risk_report=risk_report,
        exposure_report=exposure_report,
        equity_curve=equity_curve,
        backtest_metrics=backtest_metrics,
    )
