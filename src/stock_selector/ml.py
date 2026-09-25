from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from .config import MLConfig


@dataclass(frozen=True)
class RollingMLResult:
    dataset: pd.DataFrame
    predictions: pd.DataFrame
    metrics: dict[str, float]
    feature_importance: pd.DataFrame


def build_ml_dataset(
    features: pd.DataFrame,
    feature_columns: tuple[str, ...],
    label_forward_days: int,
) -> pd.DataFrame:
    if not feature_columns:
        raise ValueError("ml.feature_columns cannot be empty.")

    missing = [column for column in feature_columns if column not in features.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Missing ML feature columns: {missing_text}")

    dataset = features.copy().sort_values(["ticker", "date"]).reset_index(drop=True)
    grouped = dataset.groupby("ticker", group_keys=False)
    future_price = grouped["adj_close"].shift(-label_forward_days)
    dataset["label_date"] = grouped["date"].shift(-label_forward_days)
    dataset["future_return"] = future_price / dataset["adj_close"] - 1.0
    dataset["future_excess_return"] = dataset["future_return"] - dataset.groupby("date")[
        "future_return"
    ].transform("median")

    keep_columns = [
        "date",
        "ticker",
        "adj_close",
        "passes_universe",
        "label_date",
        "future_return",
        "future_excess_return",
        *feature_columns,
    ]
    dataset = dataset[keep_columns]
    feature_list = list(feature_columns)
    dataset[feature_list] = dataset[feature_list].apply(pd.to_numeric, errors="coerce")
    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    dataset = dataset[dataset["passes_universe"].fillna(False)].reset_index(drop=True)
    return dataset


def rolling_train_predict(dataset: pd.DataFrame, config: MLConfig) -> RollingMLResult:
    estimator_factory = _build_estimator_factory(config)
    prediction_rows: list[pd.DataFrame] = []
    importance_rows: list[pd.DataFrame] = []

    dates = pd.DatetimeIndex(sorted(dataset["date"].dropna().unique()))
    for prediction_date in dates:
        test = dataset[dataset["date"] == prediction_date].copy()
        if len(test) < config.min_prediction_assets:
            continue

        train_start = prediction_date - pd.Timedelta(days=config.train_window_days)
        train = dataset[
            (dataset["date"] >= train_start)
            & (dataset["date"] < prediction_date)
            & (dataset["label_date"] < prediction_date)
            & dataset["future_excess_return"].notna()
        ].copy()
        if len(train) < config.min_train_rows:
            continue

        model = estimator_factory()
        model.fit(train[list(config.feature_columns)], train["future_excess_return"])

        test["ml_prediction"] = model.predict(test[list(config.feature_columns)])
        test["model_train_rows"] = len(train)
        test["model_train_start"] = train["date"].min()
        test["model_train_end"] = train["date"].max()
        test["latest_train_label_date"] = train["label_date"].max()
        prediction_rows.append(
            test[
                [
                    "date",
                    "ticker",
                    "ml_prediction",
                    "future_return",
                    "future_excess_return",
                    "label_date",
                    "model_train_rows",
                    "model_train_start",
                    "model_train_end",
                    "latest_train_label_date",
                ]
            ]
        )
        importance_rows.append(_extract_feature_importance(model, config, prediction_date))

    predictions = (
        pd.concat(prediction_rows, ignore_index=True)
        if prediction_rows
        else pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "ml_prediction",
                "future_return",
                "future_excess_return",
                "label_date",
                "model_train_rows",
                "model_train_start",
                "model_train_end",
                "latest_train_label_date",
            ]
        )
    )
    feature_importance = _summarize_feature_importance(importance_rows)
    metrics = evaluate_predictions(
        predictions,
        top_n=config.min_prediction_assets,
        forward_days=config.label_forward_days,
    )
    return RollingMLResult(
        dataset=dataset,
        predictions=predictions,
        metrics=metrics,
        feature_importance=feature_importance,
    )


def evaluate_predictions(
    predictions: pd.DataFrame,
    top_n: int = 5,
    forward_days: int | None = None,
) -> dict[str, float]:
    evaluated = predictions.dropna(subset=["ml_prediction", "future_return"]).copy()
    if evaluated.empty:
        return {
            "prediction_rows": 0.0,
            "prediction_dates": 0.0,
            "rank_ic_mean": 0.0,
            "rank_ic_positive_rate": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "directional_accuracy": 0.0,
            "directional_baseline_accuracy": 0.0,
            "directional_accuracy_advantage": 0.0,
            "top_bucket_mean_return": 0.0,
            "top_bucket_win_rate": 0.0,
            "top_bucket_sharpe": 0.0,
            "top_bucket_annualized_return": 0.0,
            "bottom_bucket_mean_return": 0.0,
            "top_vs_bottom_mean_return": 0.0,
            "top_vs_bottom_spread_sharpe": 0.0,
        }

    rank_ics: list[float] = []
    spreads: list[float] = []
    top_returns: list[float] = []
    bottom_returns: list[float] = []
    for _, group in evaluated.groupby("date"):
        if len(group) < 2:
            continue
        rank_ic = group["ml_prediction"].corr(group["future_return"], method="spearman")
        if pd.notna(rank_ic):
            rank_ics.append(float(rank_ic))
        ranked = group.sort_values("ml_prediction", ascending=False)
        bucket_size = max(1, min(top_n, len(ranked) // 2 or 1))
        top_return = ranked.head(bucket_size)["future_return"].mean()
        bottom_return = ranked.tail(bucket_size)["future_return"].mean()
        top_returns.append(float(top_return))
        bottom_returns.append(float(bottom_return))
        spreads.append(float(top_return - bottom_return))

    errors = evaluated["ml_prediction"] - evaluated["future_excess_return"]
    rmse = math.sqrt(float((errors**2).mean()))
    predicted_up = evaluated["ml_prediction"] > 0
    actual_up = evaluated["future_excess_return"] > 0
    directional_accuracy = float((predicted_up == actual_up).mean())
    positive_rate = float(actual_up.mean())
    directional_baseline = max(positive_rate, 1.0 - positive_rate)
    periods_per_year = 252.0 / forward_days if forward_days and forward_days > 0 else 252.0
    top_returns_array = np.asarray(top_returns, dtype=float)
    spreads_array = np.asarray(spreads, dtype=float)
    return {
        "prediction_rows": float(len(evaluated)),
        "prediction_dates": float(evaluated["date"].nunique()),
        "rank_ic_mean": float(np.mean(rank_ics)) if rank_ics else 0.0,
        "rank_ic_positive_rate": float(np.mean(np.array(rank_ics) > 0)) if rank_ics else 0.0,
        "mae": float(errors.abs().mean()),
        "rmse": rmse,
        "directional_accuracy": directional_accuracy,
        "directional_baseline_accuracy": directional_baseline,
        "directional_accuracy_advantage": directional_accuracy - directional_baseline,
        "top_bucket_mean_return": float(np.mean(top_returns_array))
        if len(top_returns_array)
        else 0.0,
        "top_bucket_win_rate": float(np.mean(top_returns_array > 0))
        if len(top_returns_array)
        else 0.0,
        "top_bucket_sharpe": _sharpe_from_period_returns(top_returns_array, periods_per_year),
        "top_bucket_annualized_return": _annualized_return_from_period_returns(
            top_returns_array,
            periods_per_year,
        ),
        "bottom_bucket_mean_return": float(np.mean(bottom_returns)) if bottom_returns else 0.0,
        "top_vs_bottom_mean_return": float(np.mean(spreads)) if spreads else 0.0,
        "top_vs_bottom_spread_sharpe": _sharpe_from_period_returns(
            spreads_array,
            periods_per_year,
        ),
    }


def _sharpe_from_period_returns(returns: np.ndarray, periods_per_year: float) -> float:
    if len(returns) == 0:
        return 0.0
    std = float(np.std(returns, ddof=0))
    if std <= 0:
        return 0.0
    value = float(np.mean(returns) / std * math.sqrt(periods_per_year))
    return value if math.isfinite(value) else 0.0


def _annualized_return_from_period_returns(
    returns: np.ndarray,
    periods_per_year: float,
) -> float:
    if len(returns) == 0:
        return 0.0
    period_return = float(np.mean(returns))
    if period_return <= -1.0:
        return -1.0
    value = (1.0 + period_return) ** periods_per_year - 1.0
    return float(value) if math.isfinite(value) else 0.0


def _build_estimator_factory(config: MLConfig):
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError("Install the optional 'ml' dependency to use ML modeling.") from exc

    model_type = config.model_type.lower().strip()

    def build():
        if model_type == "ridge":
            return Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", Ridge(alpha=config.ridge_alpha)),
                ]
            )
        if model_type == "random_forest":
            return Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=config.n_estimators,
                            min_samples_leaf=5,
                            random_state=config.random_state,
                            n_jobs=-1,
                        ),
                    ),
                ]
            )
        raise ValueError(f"Unsupported ML model_type: {config.model_type}")

    return build


def _extract_feature_importance(
    model: object,
    config: MLConfig,
    prediction_date: pd.Timestamp,
) -> pd.DataFrame:
    fitted_model = model.named_steps["model"]
    if hasattr(fitted_model, "coef_"):
        values = np.abs(np.asarray(fitted_model.coef_, dtype=float))
    elif hasattr(fitted_model, "feature_importances_"):
        values = np.asarray(fitted_model.feature_importances_, dtype=float)
    else:
        values = np.zeros(len(config.feature_columns))

    return pd.DataFrame(
        {
            "date": prediction_date,
            "feature": list(config.feature_columns),
            "importance": values,
        }
    )


def _summarize_feature_importance(importance_rows: list[pd.DataFrame]) -> pd.DataFrame:
    if not importance_rows:
        return pd.DataFrame(columns=["feature", "importance"])
    raw = pd.concat(importance_rows, ignore_index=True)
    summary = raw.groupby("feature", as_index=False)["importance"].mean()
    total = summary["importance"].sum()
    if total > 0:
        summary["importance"] = summary["importance"] / total
    return summary.sort_values("importance", ascending=False).reset_index(drop=True)
