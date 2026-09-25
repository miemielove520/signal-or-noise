from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .json_io import dataframe_records, write_json


@dataclass(frozen=True)
class RunComparisonResult:
    summary: pd.DataFrame
    ticker_comparison: pd.DataFrame
    adoption_decision: pd.DataFrame
    report: str
    output_dir: Path
    output_files: dict[str, str]


def compare_validation_runs(
    previous_dir: str | Path,
    current_dir: str | Path,
    output_dir: str | Path | None = None,
) -> RunComparisonResult:
    previous_path = Path(previous_dir)
    current_path = Path(current_dir)
    root = Path(output_dir) if output_dir is not None else current_path
    root.mkdir(parents=True, exist_ok=True)

    previous_manifest = _read_manifest(previous_path)
    current_manifest = _read_manifest(current_path)
    previous_summary = _read_csv_if_exists(previous_path / "walk_forward_summary.csv")
    current_summary = _read_csv_if_exists(current_path / "walk_forward_summary.csv")
    previous_tickers = _read_csv_if_exists(previous_path / "ticker_validation_ranking.csv")
    current_tickers = _read_csv_if_exists(current_path / "ticker_validation_ranking.csv")
    previous_sample = _read_csv_if_exists(previous_path / "sample_sufficiency_guidance.csv")
    current_sample = _read_csv_if_exists(current_path / "sample_sufficiency_guidance.csv")

    summary = build_run_comparison_summary(
        previous_manifest=previous_manifest,
        current_manifest=current_manifest,
        previous_summary=previous_summary,
        current_summary=current_summary,
        previous_tickers=previous_tickers,
        current_tickers=current_tickers,
        previous_sample=previous_sample,
        current_sample=current_sample,
    )
    ticker_comparison = build_ticker_run_comparison(previous_tickers, current_tickers)
    adoption_decision = build_config_adoption_decision(summary)
    report = render_run_comparison_report(
        previous_dir=previous_path,
        current_dir=current_path,
        summary=summary,
        ticker_comparison=ticker_comparison,
        adoption_decision=adoption_decision,
    )

    summary_path = root / "run_comparison_summary.csv"
    ticker_path = root / "run_comparison_tickers.csv"
    adoption_path = root / "config_adoption_decision.csv"
    report_path = root / "run_comparison.md"
    json_path = root / "run_comparison.json"
    summary.to_csv(summary_path, index=False)
    ticker_comparison.to_csv(ticker_path, index=False)
    adoption_decision.to_csv(adoption_path, index=False)
    report_path.write_text(report, encoding="utf-8")
    output_files = {
        "summary_csv": str(summary_path),
        "ticker_comparison_csv": str(ticker_path),
        "adoption_decision_csv": str(adoption_path),
        "markdown_report": str(report_path),
        "json": str(json_path),
    }
    write_json(
        json_path,
        {
            "previous_dir": str(previous_path),
            "current_dir": str(current_path),
            "summary": dataframe_records(summary),
            "ticker_comparison": dataframe_records(ticker_comparison),
            "adoption_decision": dataframe_records(adoption_decision),
            "output_files": output_files,
        },
    )

    return RunComparisonResult(
        summary=summary,
        ticker_comparison=ticker_comparison,
        adoption_decision=adoption_decision,
        report=report,
        output_dir=root,
        output_files=output_files,
    )


def build_run_comparison_summary(
    previous_manifest: dict[str, object],
    current_manifest: dict[str, object],
    previous_summary: pd.DataFrame,
    current_summary: pd.DataFrame,
    previous_tickers: pd.DataFrame,
    current_tickers: pd.DataFrame,
    previous_sample: pd.DataFrame,
    current_sample: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        _metric_row(
            "event_count",
            "信号样本数",
            _manifest_number(previous_manifest, "event_count"),
            _manifest_number(current_manifest, "event_count"),
            higher_is_better=True,
        ),
        _metric_row(
            "weighted_20d_win_rate",
            "加权20日胜率",
            _weighted_summary_metric(previous_summary, "win_rate_20d"),
            _weighted_summary_metric(current_summary, "win_rate_20d"),
            higher_is_better=True,
            value_type="percent",
        ),
        _metric_row(
            "weighted_20d_avg_return",
            "加权20日平均收益",
            _weighted_summary_metric(previous_summary, "avg_return_20d"),
            _weighted_summary_metric(current_summary, "avg_return_20d"),
            higher_is_better=True,
            value_type="percent",
        ),
        _metric_row(
            "high_probability_sample_count",
            "高概率样本数",
            _bucket_sample_count(previous_summary, "high_probability"),
            _bucket_sample_count(current_summary, "high_probability"),
            higher_is_better=True,
        ),
        _metric_row(
            "priority_candidate_count",
            "优先候选个股数",
            _decision_count(previous_tickers, "priority_candidate"),
            _decision_count(current_tickers, "priority_candidate"),
            higher_is_better=True,
        ),
        _metric_row(
            "watchlist_candidate_count",
            "观察候选个股数",
            _decision_count(previous_tickers, "watchlist_candidate"),
            _decision_count(current_tickers, "watchlist_candidate"),
            higher_is_better=True,
        ),
        _metric_row(
            "avg_ticker_ranking_score",
            "平均个股排名分",
            _column_mean(previous_tickers, "ticker_ranking_score"),
            _column_mean(current_tickers, "ticker_ranking_score"),
            higher_is_better=True,
        ),
        _metric_row(
            "thin_sample_rows",
            "样本不足行数",
            _sample_issue_count(previous_sample),
            _sample_issue_count(current_sample),
            higher_is_better=False,
        ),
        _metric_row(
            "price_warning_count",
            "价格数据警告数",
            len(previous_manifest.get("price_warnings", []) or []),
            len(current_manifest.get("price_warnings", []) or []),
            higher_is_better=False,
        ),
        _metric_row(
            "missing_price_ticker_count",
            "缺失价格股票数",
            len(previous_manifest.get("missing_price_tickers", []) or []),
            len(current_manifest.get("missing_price_tickers", []) or []),
            higher_is_better=False,
        ),
        _metric_row(
            "duration_seconds",
            "运行耗时秒数",
            _manifest_number(previous_manifest, "duration_seconds"),
            _manifest_number(current_manifest, "duration_seconds"),
            higher_is_better=False,
        ),
    ]
    return pd.DataFrame(rows)


def build_ticker_run_comparison(
    previous_tickers: pd.DataFrame,
    current_tickers: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "ticker",
        "change_type",
        "change_type_zh",
        "previous_rank",
        "current_rank",
        "rank_change",
        "previous_score",
        "current_score",
        "score_change",
        "previous_decision",
        "current_decision",
        "previous_decision_zh",
        "current_decision_zh",
        "previous_sample_count",
        "current_sample_count",
        "sample_count_change",
        "comparison_note",
        "comparison_note_zh",
    ]
    if previous_tickers.empty and current_tickers.empty:
        return pd.DataFrame(columns=columns)

    previous = _ticker_map(previous_tickers)
    current = _ticker_map(current_tickers)
    rows: list[dict[str, object]] = []
    for ticker in sorted(set(previous) | set(current)):
        old = previous.get(ticker, {})
        new = current.get(ticker, {})
        if old and new:
            change_type, change_type_zh = "common", "共同存在"
        elif new:
            change_type, change_type_zh = "added", "新增"
        else:
            change_type, change_type_zh = "removed", "移除"
        previous_rank = _safe_number(old.get("rank"))
        current_rank = _safe_number(new.get("rank"))
        previous_score = _safe_number(old.get("ticker_ranking_score"))
        current_score = _safe_number(new.get("ticker_ranking_score"))
        previous_sample = _safe_number(old.get("sample_count"))
        current_sample = _safe_number(new.get("sample_count"))
        rank_change = _rank_change(previous_rank, current_rank)
        score_change = _numeric_change(previous_score, current_score)
        sample_change = _numeric_change(previous_sample, current_sample)
        note, note_zh = _ticker_comparison_note(
            change_type=change_type,
            rank_change=rank_change,
            score_change=score_change,
            sample_change=sample_change,
        )
        rows.append(
            {
                "ticker": ticker,
                "change_type": change_type,
                "change_type_zh": change_type_zh,
                "previous_rank": previous_rank,
                "current_rank": current_rank,
                "rank_change": rank_change,
                "previous_score": previous_score,
                "current_score": current_score,
                "score_change": score_change,
                "previous_decision": old.get("ticker_decision", ""),
                "current_decision": new.get("ticker_decision", ""),
                "previous_decision_zh": old.get("ticker_decision_zh", ""),
                "current_decision_zh": new.get("ticker_decision_zh", ""),
                "previous_sample_count": previous_sample,
                "current_sample_count": current_sample,
                "sample_count_change": sample_change,
                "comparison_note": note,
                "comparison_note_zh": note_zh,
            }
        )
    frame = pd.DataFrame(rows, columns=columns)
    return frame.sort_values(
        ["change_type", "score_change", "sample_count_change"],
        ascending=[True, False, False],
        na_position="last",
    ).reset_index(drop=True)


def build_config_adoption_decision(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "check_name",
        "check_name_zh",
        "status",
        "status_zh",
        "severity",
        "previous_value",
        "current_value",
        "change",
        "detail",
        "detail_zh",
    ]
    if summary.empty:
        return pd.DataFrame(
            [
                _adoption_row(
                    "final_decision",
                    "最终结论",
                    "continue_validation",
                    "继续验证",
                    "caution",
                    np.nan,
                    np.nan,
                    np.nan,
                    "Comparison summary is unavailable.",
                    "缺少对比摘要，不能采用候选配置。",
                )
            ],
            columns=columns,
        )

    metrics = {str(row.metric): row for row in summary.itertuples(index=False)}
    rows = [
        _metric_check(
            metrics,
            metric="event_count",
            check_name="sample_retention",
            check_name_zh="样本保留",
            min_relative_change=-0.20,
            min_absolute_change=-10.0,
            fail_detail="Candidate config removed too many validation samples.",
            fail_detail_zh="候选配置减少了过多验证样本。",
            pass_detail="Validation sample count is acceptable.",
            pass_detail_zh="验证样本数可以接受。",
        ),
        _metric_check(
            metrics,
            metric="weighted_20d_win_rate",
            check_name="win_rate_not_worse",
            check_name_zh="胜率不能变差",
            min_absolute_change=-0.005,
            fail_detail="Weighted 20-day win rate worsened.",
            fail_detail_zh="加权20日胜率变差。",
            pass_detail="Weighted 20-day win rate did not materially worsen.",
            pass_detail_zh="加权20日胜率没有明显变差。",
        ),
        _metric_check(
            metrics,
            metric="weighted_20d_avg_return",
            check_name="average_return_not_worse",
            check_name_zh="平均收益不能变差",
            min_absolute_change=-0.002,
            fail_detail="Weighted 20-day average return worsened.",
            fail_detail_zh="加权20日平均收益变差。",
            pass_detail="Weighted 20-day average return did not materially worsen.",
            pass_detail_zh="加权20日平均收益没有明显变差。",
        ),
        _metric_check(
            metrics,
            metric="high_probability_sample_count",
            check_name="high_probability_sample_retention",
            check_name_zh="高概率样本保留",
            min_relative_change=-0.30,
            min_absolute_change=-5.0,
            fail_detail="High-probability sample count fell too much.",
            fail_detail_zh="高概率样本数下降过多。",
            pass_detail="High-probability sample count is acceptable.",
            pass_detail_zh="高概率样本数可以接受。",
            caution_when_missing=True,
        ),
        _metric_check(
            metrics,
            metric="price_warning_count",
            check_name="data_warnings_not_more",
            check_name_zh="数据警告不能增加",
            max_absolute_change=0.0,
            fail_detail="Data warning count increased.",
            fail_detail_zh="数据警告数量增加。",
            pass_detail="Data warning count did not increase.",
            pass_detail_zh="数据警告数量没有增加。",
        ),
        _metric_check(
            metrics,
            metric="missing_price_ticker_count",
            check_name="missing_prices_not_more",
            check_name_zh="缺失价格股票不能增加",
            max_absolute_change=0.0,
            fail_detail="Missing price ticker count increased.",
            fail_detail_zh="缺失价格股票数量增加。",
            pass_detail="Missing price ticker count did not increase.",
            pass_detail_zh="缺失价格股票数量没有增加。",
        ),
    ]
    final = _final_adoption_row(rows)
    rows.append(final)
    return pd.DataFrame(rows, columns=columns)


def render_run_comparison_report(
    previous_dir: Path,
    current_dir: Path,
    summary: pd.DataFrame,
    ticker_comparison: pd.DataFrame,
    adoption_decision: pd.DataFrame | None = None,
) -> str:
    lines = [
        "# Validation Run Comparison / 验证运行对比",
        "",
        f"- Previous run / 上一次运行: `{previous_dir}`",
        f"- Current run / 当前运行: `{current_dir}`",
        "",
        "## Overall Changes / 整体变化",
        "",
    ]
    lines.extend(_markdown_table(_format_summary_for_report(summary)))
    lines.extend(
        [
            "",
            "## Config Adoption Decision / 配置采用结论",
            "",
            (
                "This section is a safety gate for candidate screening configs. "
                "Do not replace the live config unless the final row says it passed review."
            ),
            "本区块是候选筛选配置的安全门槛；只有最终结论通过时，才考虑替换正式配置。",
            "",
        ]
    )
    adoption_frame = adoption_decision if adoption_decision is not None else pd.DataFrame()
    if adoption_frame.empty:
        lines.append("No adoption decision was available. / 没有配置采用结论。")
    else:
        lines.extend(_markdown_table(_format_numeric_columns_for_report(adoption_frame)))
    lines.extend(
        [
            "",
            "## Ticker Changes / 个股变化",
            "",
        ]
    )
    if ticker_comparison.empty:
        lines.append("No ticker ranking rows were available. / 没有可对比的个股排名。")
    else:
        display_columns = [
            "ticker",
            "change_type_zh",
            "previous_rank",
            "current_rank",
            "rank_change",
            "previous_score",
            "current_score",
            "score_change",
            "previous_decision_zh",
            "current_decision_zh",
            "comparison_note_zh",
        ]
        lines.extend(_markdown_table(ticker_comparison[display_columns].head(30)))
    lines.extend(
        [
            "",
            "## How To Read / 如何阅读",
            "",
            "- Positive `rank_change` means the ticker moved closer to rank 1.",
            "- `rank_change` 为正，表示该股票排名更靠前。",
            "- Positive `score_change` means the ticker ranking score improved.",
            "- `score_change` 为正，表示个股排名分提高。",
            "- Lower warning and missing-price counts are better.",
            "- 数据警告数和缺失价格股票数越低越好。",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _metric_check(
    metrics: dict[str, object],
    metric: str,
    check_name: str,
    check_name_zh: str,
    fail_detail: str,
    fail_detail_zh: str,
    pass_detail: str,
    pass_detail_zh: str,
    min_absolute_change: float | None = None,
    max_absolute_change: float | None = None,
    min_relative_change: float | None = None,
    caution_when_missing: bool = False,
) -> dict[str, object]:
    row = metrics.get(metric)
    previous_value = _safe_number(getattr(row, "previous_value", np.nan))
    current_value = _safe_number(getattr(row, "current_value", np.nan))
    change = _safe_number(getattr(row, "change", np.nan))
    if not np.isfinite(change) or not np.isfinite(previous_value) or not np.isfinite(current_value):
        status = "caution" if caution_when_missing else "failed"
        return _adoption_row(
            check_name,
            check_name_zh,
            status,
            "需要继续验证" if status == "caution" else "未通过",
            "caution" if status == "caution" else "critical",
            previous_value,
            current_value,
            change,
            f"Metric {metric} is unavailable.",
            f"指标 {metric} 不可用。",
        )
    failed = False
    if min_absolute_change is not None and change < min_absolute_change:
        failed = True
    if max_absolute_change is not None and change > max_absolute_change:
        failed = True
    if min_relative_change is not None:
        base = abs(previous_value)
        relative_change = change / base if base > 1e-12 else 0.0
        if relative_change < min_relative_change:
            failed = True
    if failed:
        return _adoption_row(
            check_name,
            check_name_zh,
            "failed",
            "未通过",
            "critical",
            previous_value,
            current_value,
            change,
            fail_detail,
            fail_detail_zh,
        )
    status = "improved" if change > 1e-12 else "passed"
    return _adoption_row(
        check_name,
        check_name_zh,
        status,
        "改善" if status == "improved" else "通过",
        "info",
        previous_value,
        current_value,
        change,
        pass_detail,
        pass_detail_zh,
    )


def _final_adoption_row(rows: list[dict[str, object]]) -> dict[str, object]:
    failed = [row for row in rows if row["status"] == "failed"]
    cautions = [row for row in rows if row["status"] == "caution"]
    improved = [row for row in rows if row["status"] == "improved"]
    if failed:
        return _adoption_row(
            "final_decision",
            "最终结论",
            "reject_candidate_config",
            "拒绝采用候选配置",
            "critical",
            np.nan,
            np.nan,
            np.nan,
            "At least one critical safety check failed.",
            "至少一个关键安全检查未通过。",
        )
    if cautions:
        return _adoption_row(
            "final_decision",
            "最终结论",
            "continue_validation",
            "继续验证",
            "caution",
            np.nan,
            np.nan,
            np.nan,
            "No critical check failed, but more validation is required.",
            "没有关键失败项，但仍需要更多验证。",
        )
    if len(improved) >= 2:
        return _adoption_row(
            "final_decision",
            "最终结论",
            "candidate_config_passed_review",
            "候选配置通过审核",
            "info",
            np.nan,
            np.nan,
            np.nan,
            "Candidate config passed safety checks and improved multiple metrics.",
            "候选配置通过安全检查，并改善多个指标。",
        )
    return _adoption_row(
        "final_decision",
        "最终结论",
        "no_material_improvement",
        "没有明显改善",
        "caution",
        np.nan,
        np.nan,
        np.nan,
        "Candidate config is not worse, but improvement is not strong enough.",
        "候选配置没有明显变差，但改善不够强。",
    )


def _adoption_row(
    check_name: str,
    check_name_zh: str,
    status: str,
    status_zh: str,
    severity: str,
    previous_value: object,
    current_value: object,
    change: object,
    detail: str,
    detail_zh: str,
) -> dict[str, object]:
    return {
        "check_name": check_name,
        "check_name_zh": check_name_zh,
        "status": status,
        "status_zh": status_zh,
        "severity": severity,
        "previous_value": previous_value,
        "current_value": current_value,
        "change": change,
        "detail": detail,
        "detail_zh": detail_zh,
    }


def _read_manifest(run_dir: Path) -> dict[str, object]:
    path = run_dir / "run_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing run manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _manifest_number(manifest: dict[str, object], key: str) -> float:
    return _safe_number(manifest.get(key))


def _weighted_summary_metric(summary: pd.DataFrame, column: str) -> float:
    if summary.empty or column not in summary.columns or "sample_count" not in summary.columns:
        return np.nan
    frame = summary.copy()
    frame["sample_count"] = pd.to_numeric(frame["sample_count"], errors="coerce")
    frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["sample_count", column])
    total = float(frame["sample_count"].sum())
    if total <= 0:
        return np.nan
    return float((frame[column] * frame["sample_count"]).sum() / total)


def _bucket_sample_count(summary: pd.DataFrame, bucket: str) -> float:
    if summary.empty or "bucket" not in summary.columns or "sample_count" not in summary.columns:
        return 0.0
    rows = summary[summary["bucket"] == bucket]
    if rows.empty:
        return 0.0
    return float(pd.to_numeric(rows["sample_count"], errors="coerce").fillna(0).sum())


def _decision_count(tickers: pd.DataFrame, decision: str) -> float:
    if tickers.empty or "ticker_decision" not in tickers.columns:
        return 0.0
    return float((tickers["ticker_decision"] == decision).sum())


def _column_mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if len(values) else np.nan


def _sample_issue_count(sample: pd.DataFrame) -> float:
    if sample.empty or "sample_status" not in sample.columns:
        return 0.0
    return float((sample["sample_status"] != "enough_samples").sum())


def _metric_row(
    metric: str,
    metric_zh: str,
    previous_value: object,
    current_value: object,
    higher_is_better: bool,
    value_type: str = "number",
) -> dict[str, object]:
    previous_number = _safe_number(previous_value)
    current_number = _safe_number(current_value)
    change = _numeric_change(previous_number, current_number)
    assessment, assessment_zh = _assessment(change, higher_is_better)
    return {
        "metric": metric,
        "metric_zh": metric_zh,
        "previous_value": previous_number,
        "current_value": current_number,
        "change": change,
        "value_type": value_type,
        "assessment": assessment,
        "assessment_zh": assessment_zh,
    }


def _assessment(change: float, higher_is_better: bool) -> tuple[str, str]:
    if not np.isfinite(change) or abs(change) < 1e-12:
        return "unchanged", "基本不变"
    improved = change > 0 if higher_is_better else change < 0
    return ("improved", "改善") if improved else ("worsened", "变差")


def _ticker_map(frame: pd.DataFrame) -> dict[str, dict[str, object]]:
    if frame.empty or "ticker" not in frame.columns:
        return {}
    records = frame.to_dict("records")
    return {str(row.get("ticker", "")).upper(): row for row in records if row.get("ticker")}


def _ticker_comparison_note(
    change_type: str,
    rank_change: float,
    score_change: float,
    sample_change: float,
) -> tuple[str, str]:
    if change_type == "added":
        return "Ticker appears only in current run.", "该股票只出现在当前运行。"
    if change_type == "removed":
        return "Ticker appears only in previous run.", "该股票只出现在上一次运行。"
    parts: list[str] = []
    parts_zh: list[str] = []
    if np.isfinite(rank_change) and rank_change > 0:
        parts.append("rank improved")
        parts_zh.append("排名改善")
    elif np.isfinite(rank_change) and rank_change < 0:
        parts.append("rank worsened")
        parts_zh.append("排名下降")
    if np.isfinite(score_change) and score_change > 0:
        parts.append("score improved")
        parts_zh.append("分数提高")
    elif np.isfinite(score_change) and score_change < 0:
        parts.append("score worsened")
        parts_zh.append("分数下降")
    if np.isfinite(sample_change) and sample_change > 0:
        parts.append("more samples")
        parts_zh.append("样本增加")
    elif np.isfinite(sample_change) and sample_change < 0:
        parts.append("fewer samples")
        parts_zh.append("样本减少")
    if not parts:
        return "No material ticker-level change.", "个股层面没有明显变化。"
    return "; ".join(parts) + ".", "；".join(parts_zh) + "。"


def _rank_change(previous_rank: float, current_rank: float) -> float:
    if not np.isfinite(previous_rank) or not np.isfinite(current_rank):
        return np.nan
    return previous_rank - current_rank


def _numeric_change(previous_value: float, current_value: float) -> float:
    if not np.isfinite(previous_value) or not np.isfinite(current_value):
        return np.nan
    return current_value - previous_value


def _safe_number(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def _format_summary_for_report(summary: pd.DataFrame) -> pd.DataFrame:
    frame = summary.copy()
    for column in ["previous_value", "current_value", "change"]:
        frame[column] = [
            _format_metric_value(value, value_type)
            for value, value_type in zip(frame[column], frame["value_type"], strict=False)
        ]
    return frame


def _format_numeric_columns_for_report(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ["previous_value", "current_value", "change"]:
        if column not in result.columns:
            continue
        result[column] = result[column].map(lambda value: _format_metric_value(value, "number"))
    return result


def _format_metric_value(value: object, value_type: str) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(number):
        return "N/A"
    if value_type == "percent":
        return f"{number:.2%}"
    return f"{number:.4f}"


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["No rows / 暂无数据"]
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = [str(row[column]) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return lines
