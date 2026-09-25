from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import MonitorConfig


@dataclass(frozen=True)
class DailyMonitorResult:
    candidates: pd.DataFrame
    checks: pd.DataFrame
    report: str


def build_daily_monitor(
    prices: pd.DataFrame,
    scored: pd.DataFrame,
    selections: pd.DataFrame,
    risk_report: pd.DataFrame,
    exposure_report: pd.DataFrame,
    feature_columns: tuple[str, ...] = (),
    mode: str = "factor",
    as_of_date: str | None = None,
    prediction_metrics: dict[str, float] | None = None,
    monitor_config: MonitorConfig | None = None,
) -> DailyMonitorResult:
    monitor_config = monitor_config or MonitorConfig()
    candidates = latest_candidates(selections)
    checks = monitoring_checks(
        prices=prices,
        scored=scored,
        selections=selections,
        risk_report=risk_report,
        exposure_report=exposure_report,
        feature_columns=feature_columns,
        as_of_date=as_of_date,
        prediction_metrics=prediction_metrics,
        monitor_config=monitor_config,
    )
    report = render_daily_report(
        mode=mode,
        candidates=candidates,
        checks=checks,
        risk_report=risk_report,
        exposure_report=exposure_report,
        prediction_metrics=prediction_metrics,
    )
    return DailyMonitorResult(candidates=candidates, checks=checks, report=report)


def latest_candidates(selections: pd.DataFrame) -> pd.DataFrame:
    if selections.empty:
        return pd.DataFrame(columns=list(selections.columns))
    frame = selections.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    latest_date = frame["date"].max()
    return frame[frame["date"] == latest_date].sort_values("rank").reset_index(drop=True)


def monitoring_checks(
    prices: pd.DataFrame,
    scored: pd.DataFrame,
    selections: pd.DataFrame,
    risk_report: pd.DataFrame,
    exposure_report: pd.DataFrame,
    feature_columns: tuple[str, ...] = (),
    as_of_date: str | None = None,
    prediction_metrics: dict[str, float] | None = None,
    monitor_config: MonitorConfig | None = None,
) -> pd.DataFrame:
    monitor_config = monitor_config or MonitorConfig()
    rows: list[dict[str, object]] = []
    price_latest_date = pd.Timestamp(prices["date"].max())
    monitor_date = pd.Timestamp(as_of_date) if as_of_date else pd.Timestamp.today().normalize()
    data_age_days = max(0, int((monitor_date - price_latest_date).days))
    latest_price_tickers = prices[pd.to_datetime(prices["date"]) == price_latest_date][
        "ticker"
    ].nunique()

    rows.append(
        _check_row(
            "latest_price_date",
            price_latest_date.date().isoformat(),
            "ok",
            "latest date available in price data",
        )
    )
    rows.append(
        _check_row(
            "data_age_days",
            data_age_days,
            _threshold_status(data_age_days, monitor_config.max_data_age_days),
            "calendar days between monitor date and latest price date",
        )
    )
    rows.append(
        _check_row(
            "latest_price_tickers",
            int(latest_price_tickers),
            "warning" if latest_price_tickers == 0 else "ok",
            "ticker count on latest price date",
        )
    )

    candidates = latest_candidates(selections)
    rows.append(
        _check_row(
            "latest_selected_count",
            int(len(candidates)),
            "warning" if candidates.empty else "ok",
            "number of selected tickers in latest portfolio",
        )
    )
    rows.append(
        _check_row(
            "selection_turnover",
            selection_turnover(selections),
            "ok",
            "one-way turnover from previous rebalance target weights",
        )
    )
    rows.append(
        _check_row(
            "candidate_overlap",
            candidate_overlap(selections),
            _minimum_status(candidate_overlap(selections), monitor_config.min_candidate_overlap),
            "ticker overlap between latest and previous rebalance",
        )
    )
    score_concentration_value = score_concentration(candidates)
    rows.append(
        _check_row(
            "score_concentration",
            score_concentration_value,
            _threshold_status(
                score_concentration_value,
                monitor_config.max_score_concentration,
            ),
            "absolute top score divided by total absolute candidate score",
        )
    )

    if not risk_report.empty:
        latest_risk = risk_report.iloc[-1]
        rows.extend(
            [
                _check_row(
                    "gross_exposure",
                    float(latest_risk.get("gross_exposure", 0.0)),
                    "warning" if float(latest_risk.get("gross_exposure", 0.0)) <= 0 else "ok",
                    "latest total non-cash target exposure",
                ),
                _check_row(
                    "cash_weight",
                    float(latest_risk.get("cash_weight", 0.0)),
                    "ok",
                    "latest target cash weight",
                ),
                _check_row(
                    "max_sector_weight",
                    float(latest_risk.get("max_sector_weight", 0.0)),
                    "ok",
                    "largest latest sector exposure",
                ),
                _check_row(
                    "largest_sector",
                    latest_risk.get("largest_sector", "Unknown"),
                    "ok",
                    "largest sector in latest portfolio",
                ),
            ]
        )

    feature_missing = feature_missing_report(scored, feature_columns)
    if not feature_missing.empty:
        average_missing = float(feature_missing["missing_rate"].mean())
        high_missing_count = int(
            (
                feature_missing["missing_rate"]
                > monitor_config.max_single_feature_missing_rate
            ).sum()
        )
        rows.append(
            _check_row(
                "average_feature_missing_rate",
                average_missing,
                _threshold_status(
                    average_missing,
                    monitor_config.max_average_feature_missing_rate,
                ),
                "average missing rate across configured features on latest date",
            )
        )
        rows.append(
            _check_row(
                "high_missing_feature_count",
                high_missing_count,
                "warning" if high_missing_count else "ok",
                "features whose missing rate exceeds configured threshold",
            )
        )

    drift = feature_drift_report(
        scored,
        feature_columns,
        lookback_days=monitor_config.drift_lookback_days,
    )
    if not drift.empty:
        max_abs_drift = float(drift["drift_zscore"].abs().max())
        high_drift_count = int(
            (drift["drift_zscore"].abs() > monitor_config.max_feature_drift_zscore).sum()
        )
        rows.append(
            _check_row(
                "max_feature_drift_zscore",
                max_abs_drift,
                _threshold_status(max_abs_drift, monitor_config.max_feature_drift_zscore),
                "largest absolute latest-vs-history feature mean z-score",
            )
        )
        rows.append(
            _check_row(
                "high_drift_feature_count",
                high_drift_count,
                "warning" if high_drift_count else "ok",
                "features whose drift z-score exceeds configured threshold",
            )
        )

    if prediction_metrics:
        rank_ic_mean = float(prediction_metrics.get("rank_ic_mean", 0.0))
        rows.append(
            _check_row(
                "rank_ic_mean",
                rank_ic_mean,
                _minimum_status(rank_ic_mean, monitor_config.min_rank_ic_mean),
                "mean rank correlation between ML predictions and forward returns",
            )
        )

    return pd.DataFrame(rows)


def feature_missing_report(scored: pd.DataFrame, feature_columns: tuple[str, ...]) -> pd.DataFrame:
    available_features = [column for column in feature_columns if column in scored.columns]
    if scored.empty or not available_features:
        return pd.DataFrame(columns=["feature", "missing_rate", "row_count"])

    frame = scored.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    latest_date = frame["date"].max()
    latest = frame[frame["date"] == latest_date]
    rows = []
    for feature in available_features:
        rows.append(
            {
                "feature": feature,
                "missing_rate": float(latest[feature].isna().mean()),
                "row_count": int(len(latest)),
            }
        )
    return pd.DataFrame(rows).sort_values("missing_rate", ascending=False).reset_index(drop=True)


def feature_drift_report(
    scored: pd.DataFrame,
    feature_columns: tuple[str, ...],
    lookback_days: int,
) -> pd.DataFrame:
    available_features = [column for column in feature_columns if column in scored.columns]
    if scored.empty or not available_features:
        return pd.DataFrame(
            columns=[
                "feature",
                "latest_mean",
                "baseline_mean",
                "baseline_std",
                "drift_zscore",
                "history_days",
            ]
        )

    frame = scored.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    latest_date = frame["date"].max()
    latest = frame[frame["date"] == latest_date]
    history = frame[
        (frame["date"] < latest_date)
        & (frame["date"] >= latest_date - pd.Timedelta(days=lookback_days))
    ]

    rows: list[dict[str, object]] = []
    for feature in available_features:
        daily_history = history.groupby("date")[feature].mean().dropna()
        latest_mean = float(latest[feature].mean()) if latest[feature].notna().any() else 0.0
        baseline_mean = float(daily_history.mean()) if not daily_history.empty else latest_mean
        baseline_std = float(daily_history.std(ddof=0)) if len(daily_history) > 1 else 0.0
        drift_zscore = 0.0
        if baseline_std > 0:
            drift_zscore = (latest_mean - baseline_mean) / baseline_std
        rows.append(
            {
                "feature": feature,
                "latest_mean": latest_mean,
                "baseline_mean": baseline_mean,
                "baseline_std": baseline_std,
                "drift_zscore": float(drift_zscore),
                "history_days": int(daily_history.index.nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "drift_zscore",
        key=lambda series: series.abs(),
        ascending=False,
    ).reset_index(drop=True)


def selection_turnover(selections: pd.DataFrame) -> float:
    if selections.empty:
        return 0.0
    frame = selections.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    dates = sorted(frame["date"].dropna().unique())
    if len(dates) < 2:
        return float(frame[frame["date"] == dates[-1]]["weight"].abs().sum())

    previous = frame[frame["date"] == dates[-2]].set_index("ticker")["weight"].astype(float)
    current = frame[frame["date"] == dates[-1]].set_index("ticker")["weight"].astype(float)
    all_tickers = previous.index.union(current.index)
    turnover = (current.reindex(all_tickers, fill_value=0.0) - previous.reindex(all_tickers, fill_value=0.0)).abs().sum()
    return float(turnover / 2.0)


def candidate_overlap(selections: pd.DataFrame) -> float:
    if selections.empty:
        return 0.0
    frame = selections.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    dates = sorted(frame["date"].dropna().unique())
    if len(dates) < 2:
        return 1.0
    previous = set(frame[frame["date"] == dates[-2]]["ticker"])
    current = set(frame[frame["date"] == dates[-1]]["ticker"])
    if not current:
        return 0.0
    return len(previous.intersection(current)) / len(current)


def score_concentration(candidates: pd.DataFrame) -> float:
    if candidates.empty or "score" not in candidates.columns:
        return 0.0
    scores = pd.to_numeric(candidates["score"], errors="coerce").abs().dropna()
    total = float(scores.sum())
    if total <= 0:
        return 0.0
    return float(scores.max() / total)


def render_daily_report(
    mode: str,
    candidates: pd.DataFrame,
    checks: pd.DataFrame,
    risk_report: pd.DataFrame,
    exposure_report: pd.DataFrame,
    prediction_metrics: dict[str, float] | None = None,
) -> str:
    lines = [
        f"# Daily Stock Selection Monitor ({mode})",
        "",
        "This report is generated from the local research pipeline and is not investment advice.",
        "",
        "## Monitor Checks",
        "",
    ]
    lines.extend(_markdown_table(checks))
    lines.extend(["", "## Latest Candidates", ""])

    candidate_columns = [
        column
        for column in ["ticker", "weight", "score", "rank", "sector", "industry"]
        if column in candidates.columns
    ]
    lines.extend(_markdown_table(candidates[candidate_columns] if candidate_columns else candidates))

    lines.extend(["", "## Latest Risk", ""])
    if risk_report.empty:
        lines.append("No risk report available.")
    else:
        latest_risk = risk_report.tail(1)
        display_columns = [
            column
            for column in [
                "gross_exposure",
                "cash_weight",
                "max_position_weight",
                "max_sector_weight",
                "largest_sector",
                "estimated_annual_volatility",
            ]
            if column in latest_risk.columns
        ]
        lines.extend(_markdown_table(latest_risk[display_columns]))

    lines.extend(["", "## Latest Exposure", ""])
    if exposure_report.empty:
        lines.append("No exposure report available.")
    else:
        frame = exposure_report.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        latest = frame[frame["date"] == frame["date"].max()]
        display_columns = [column for column in latest.columns if column != "date"]
        lines.extend(_markdown_table(latest[display_columns]))

    if prediction_metrics:
        lines.extend(["", "## ML Prediction Metrics", ""])
        metric_frame = pd.DataFrame(
            [{"metric": key, "value": value} for key, value in prediction_metrics.items()]
        )
        lines.extend(_markdown_table(metric_frame))

    return "\n".join(lines).rstrip() + "\n"


def _check_row(metric: str, value: object, status: str, detail: str) -> dict[str, object]:
    return {
        "metric": metric,
        "value": value,
        "status": status,
        "detail": detail,
    }


def _threshold_status(value: float, warning_threshold: float) -> str:
    return "warning" if value > warning_threshold else "ok"


def _minimum_status(value: float, minimum_value: float) -> str:
    return "warning" if value < minimum_value else "ok"


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["No rows."]
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(_format_value(value) for value in row) + " |")
    return lines


def _format_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
