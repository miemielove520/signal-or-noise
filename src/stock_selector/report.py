from __future__ import annotations

import pandas as pd


def render_research_report(
    title: str,
    selections: pd.DataFrame,
    metrics: dict[str, float],
    risk_report: pd.DataFrame,
    exposure_report: pd.DataFrame,
    prediction_metrics: dict[str, float] | None = None,
    feature_importance: pd.DataFrame | None = None,
    survivorship_bias_report: dict[str, object] | None = None,
) -> str:
    lines: list[str] = [
        f"# {title}",
        "",
        "This report is generated from the local research pipeline and is not investment advice.",
        "",
    ]
    lines.extend(_survivorship_section(survivorship_bias_report))
    lines.extend(_metrics_section("Backtest Metrics", metrics))

    if prediction_metrics is not None:
        lines.extend(_metrics_section("Prediction Metrics", prediction_metrics))

    lines.extend(_latest_selection_section(selections))
    lines.extend(_latest_risk_section(risk_report))
    lines.extend(_latest_exposure_section(exposure_report))

    if feature_importance is not None and not feature_importance.empty:
        lines.extend(_feature_importance_section(feature_importance))

    return "\n".join(lines).rstrip() + "\n"


def _survivorship_section(report: dict[str, object] | None) -> list[str]:
    status = report or {
        "survivorship_bias_handled": False,
        "contains_delisted_tickers": False,
        "source": "none",
        "warnings": [
            "No historical universe membership file was supplied; backtest may have survivorship bias."
        ],
    }
    lines = [
        "## Survivorship Bias",
        "",
        f"- `survivorship_bias_handled`: {str(bool(status.get('survivorship_bias_handled'))).lower()}",
        f"- `contains_delisted_tickers`: {str(bool(status.get('contains_delisted_tickers'))).lower()}",
        f"- `historical_universe_source`: {status.get('source', 'none')}",
    ]
    for warning in status.get("warnings", []):
        lines.append(f"- `warning`: {warning}")
    lines.append("")
    return lines


def _metrics_section(title: str, metrics: dict[str, float]) -> list[str]:
    lines = [f"## {title}", ""]
    if not metrics:
        return [*lines, "No metrics available.", ""]
    for key, value in metrics.items():
        lines.append(f"- `{key}`: {_format_number(value)}")
    lines.append("")
    return lines


def _latest_selection_section(selections: pd.DataFrame) -> list[str]:
    lines = ["## Latest Selections", ""]
    if selections.empty:
        return [*lines, "No selections produced.", ""]

    latest_date = selections["date"].max()
    latest = selections[selections["date"] == latest_date].sort_values("rank")
    lines.append(f"Date: `{_format_date(latest_date)}`")
    lines.append("")
    columns = ["ticker", "weight", "score", "rank"]
    for optional in ["sector", "industry"]:
        if optional in latest.columns:
            columns.append(optional)
    lines.extend(_markdown_table(latest[columns]))
    lines.append("")
    return lines


def _latest_risk_section(risk_report: pd.DataFrame) -> list[str]:
    lines = ["## Latest Risk", ""]
    if risk_report.empty:
        return [*lines, "No risk report available.", ""]

    latest = risk_report.iloc[-1]
    for column in [
        "gross_exposure",
        "cash_weight",
        "max_position_weight",
        "max_sector_weight",
        "largest_sector",
        "estimated_annual_volatility",
        "target_annual_volatility",
    ]:
        if column in risk_report.columns:
            lines.append(f"- `{column}`: {_format_number(latest[column])}")
    lines.append("")
    return lines


def _latest_exposure_section(exposure_report: pd.DataFrame) -> list[str]:
    lines = ["## Latest Sector Exposure", ""]
    if exposure_report.empty:
        return [*lines, "No exposure report available.", ""]

    latest_date = exposure_report["date"].max()
    latest = exposure_report[exposure_report["date"] == latest_date].copy()
    latest = latest.sort_values("weight", ascending=False)
    lines.append(f"Date: `{_format_date(latest_date)}`")
    lines.append("")
    display_columns = [column for column in latest.columns if column != "date"]
    lines.extend(_markdown_table(latest[display_columns]))
    lines.append("")
    return lines


def _feature_importance_section(feature_importance: pd.DataFrame) -> list[str]:
    lines = ["## Top Features", ""]
    top = feature_importance.head(10)
    lines.extend(_markdown_table(top[["feature", "importance"]]))
    lines.append("")
    return lines


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["No rows."]
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False):
        values = [_format_number(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _format_number(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return _format_date(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _format_date(value: object) -> str:
    return pd.Timestamp(value).date().isoformat()
