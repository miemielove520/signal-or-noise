from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .journal import JournalResult, write_daily_journal
from .json_io import dataframe_records, write_json
from .real_data import RealTickerAnalysisResult, normalize_ticker, run_real_ticker_analysis
from .screening_config import ScreeningConfig, ScreeningThresholds, default_screening_config
from .data_sources import scan_data_readiness_summary


@dataclass(frozen=True)
class ScanResult:
    tickers: tuple[str, ...]
    summary: pd.DataFrame
    top_candidates: pd.DataFrame
    data_readiness_summary: pd.DataFrame
    output_dir: Path
    results: tuple[RealTickerAnalysisResult, ...]
    failures: tuple[dict[str, str], ...]
    journal: JournalResult | None = None


def run_high_probability_scan(
    tickers: list[str] | tuple[str, ...],
    period: str = "5y",
    output_dir: str | Path = "outputs/scans/latest",
    data_root: str | Path = "data/real_prices",
    include_snapshot: bool = True,
    write_journal: bool = False,
    journal_root: str | Path = "outputs/journal",
    screening_thresholds: ScreeningThresholds | None = None,
    screening_config: ScreeningConfig | None = None,
) -> ScanResult:
    normalized_tickers = tuple(
        dict.fromkeys(normalize_ticker(ticker) for ticker in tickers if ticker.strip())
    )
    if not normalized_tickers:
        raise ValueError("At least one ticker is required.")

    output_path = Path(output_dir)
    ticker_output_root = output_path / "tickers"
    output_path.mkdir(parents=True, exist_ok=True)

    results: list[RealTickerAnalysisResult] = []
    failures: list[dict[str, str]] = []
    rows: list[dict[str, object]] = []
    for ticker in normalized_tickers:
        try:
            result = run_real_ticker_analysis(
                ticker=ticker,
                period=period,
                output_root=ticker_output_root,
                data_root=data_root,
                include_snapshot=include_snapshot,
                include_peer_comparison=False,
                screening_thresholds=screening_thresholds,
                screening_config=screening_config,
            )
        except Exception as exc:
            failures.append({"ticker": ticker, "error": str(exc)})
            continue
        results.append(result)
        rows.append(_scan_row(result, period=result.effective_period))

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(
            [
                "calibrated_quality_gate_passed",
                "quality_gate_passed",
                "calibrated_high_probability_score",
                "high_probability_score",
                "signal_score",
            ],
            ascending=[False, False, False, False, False],
        ).reset_index(drop=True)
        summary["scan_rank"] = summary.index + 1
        summary = _add_recheck_queue_columns(summary)
    else:
        summary = pd.DataFrame(columns=_scan_columns())

    top_candidates = build_top_candidates(summary)
    data_readiness_summary = scan_data_readiness_summary(summary)
    summary.to_csv(output_path / "high_probability_scan.csv", index=False)
    top_candidates.to_csv(output_path / "top_candidates.csv", index=False)
    data_readiness_summary.to_csv(output_path / "data_readiness_summary.csv", index=False)
    (output_path / "high_probability_scan.md").write_text(
        render_scan_report(summary, failures),
        encoding="utf-8",
    )
    (output_path / "top_candidates.md").write_text(
        render_top_candidates_report(top_candidates, summary, failures),
        encoding="utf-8",
    )
    journal = write_daily_journal(summary, journal_root=journal_root) if write_journal else None
    write_json(
        output_path / "cache_metadata.json",
        _build_scan_cache_metadata(
            tickers=normalized_tickers,
            period=period,
            output_dir=output_path,
            data_root=Path(data_root),
            results=results,
            failures=failures,
        ),
    )
    write_json(
        output_path / "top_candidates.json",
        {
            "generated_at_utc": _utc_now(),
            "tickers_requested": list(normalized_tickers),
            "tickers_analyzed": [result.ticker for result in results],
            "top_candidates": dataframe_records(top_candidates),
            "data_readiness_summary": dataframe_records(data_readiness_summary),
            "source_summary_csv": str(output_path / "high_probability_scan.csv"),
            "source_summary_report": str(output_path / "high_probability_scan.md"),
            "data_readiness_summary_csv": str(output_path / "data_readiness_summary.csv"),
        },
    )
    write_json(
        output_path / "scan_result.json",
        {
            "generated_at_utc": _utc_now(),
            "tickers_requested": list(normalized_tickers),
            "tickers_analyzed": [result.ticker for result in results],
            "requested_period": period,
            "period": period,
            "effective_periods": {
                result.ticker: result.effective_period
                for result in results
            },
            "output_dir": str(output_path),
            "summary": dataframe_records(summary),
            "top_candidates": dataframe_records(top_candidates),
            "data_readiness_summary": dataframe_records(data_readiness_summary),
            "recheck_action_summary": dataframe_records(_recheck_action_summary(summary)),
            "failures": failures,
            "screening_thresholds": (screening_thresholds or ScreeningThresholds()).to_dict(),
            "screening_config": screening_config.to_dict()
            if screening_config is not None
            else default_screening_config().to_dict(),
            "journal": _journal_payload(journal),
            "output_files": {
                "markdown_report": str(output_path / "high_probability_scan.md"),
                "top_candidates_report": str(output_path / "top_candidates.md"),
                "top_candidates_csv": str(output_path / "top_candidates.csv"),
                "top_candidates_json": str(output_path / "top_candidates.json"),
                "data_readiness_summary_csv": str(output_path / "data_readiness_summary.csv"),
                "csv": str(output_path / "high_probability_scan.csv"),
                "json": str(output_path / "scan_result.json"),
                "cache_metadata": str(output_path / "cache_metadata.json"),
            },
        },
    )
    return ScanResult(
        tickers=normalized_tickers,
        summary=summary,
        top_candidates=top_candidates,
        data_readiness_summary=data_readiness_summary,
        output_dir=output_path,
        results=tuple(results),
        failures=tuple(failures),
        journal=journal,
    )


def _build_scan_cache_metadata(
    tickers: tuple[str, ...],
    period: str,
    output_dir: Path,
    data_root: Path,
    results: list[RealTickerAnalysisResult],
    failures: list[dict[str, str]],
) -> dict[str, object]:
    return {
        "generated_at_utc": _utc_now(),
        "period": period,
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "tickers_requested": list(tickers),
        "success_count": len(results),
        "failure_count": len(failures),
        "failures": failures,
        "ticker_cache": {
            result.ticker: getattr(result, "cache_metadata", {})
            for result in results
        },
    }


def _journal_payload(journal: JournalResult | None) -> dict[str, object] | None:
    if journal is None:
        return None
    return {
        "journal_dir": str(journal.journal_dir),
        "snapshot_path": str(journal.snapshot_path),
        "report_path": str(journal.report_path),
        "changes": dataframe_records(journal.changes),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def render_scan_report(summary: pd.DataFrame, failures: list[dict[str, str]] | tuple[dict[str, str], ...]) -> str:
    if not summary.empty and "recheck_priority_score" not in summary.columns:
        summary = _add_recheck_queue_columns(summary)
    lines = [
        "# High Probability Stock Scan / 高概率股票扫描",
        "",
        "This scan ranks tickers by the strict high-probability screening score.",
        "本扫描按照严格高概率筛选分数对股票排序。",
        "",
    ]
    if summary.empty:
        lines.extend(["No tickers were analyzed successfully.", "没有股票成功完成分析。", ""])
    else:
        passed = summary[summary["calibrated_quality_gate_passed"] == True]  # noqa: E712
        near = summary[
            (summary["calibrated_quality_gate_passed"] == False)  # noqa: E712
            & (summary["calibrated_watchlist_status"] == "close_but_not_ready")
        ]
        early = summary[
            (summary["calibrated_quality_gate_passed"] == False)  # noqa: E712
            & (summary["calibrated_watchlist_status"] == "early_watch")
        ]
        filtered = summary[
            (summary["calibrated_quality_gate_passed"] == False)  # noqa: E712
            & (
                ~summary["calibrated_watchlist_status"].isin(
                    ["close_but_not_ready", "early_watch"]
                )
            )
        ]
        lines.extend(_scan_section("## Calibrated High Probability Candidates / 校准后高概率候选", passed))
        lines.extend(_scan_section("## Near Watchlist / 接近机会", near))
        lines.extend(_scan_section("## Early Watchlist / 早期观察", early))
        lines.extend(_scan_section("## Filtered Out / 被过滤", filtered))
        recheck_queue = _recheck_queue(summary)
        lines.extend(_scan_section("## Re-check Queue / 重新检查队列", recheck_queue))
        lines.extend(
            _basic_scan_section(
                "## Re-check Action Breakdown / 重新检查动作分类",
                _recheck_action_summary(summary),
            )
        )
        lines.extend(["", "## Full Ranking / 完整排名", ""])
        lines.extend(_markdown_table(summary[_existing_columns(summary, _scan_display_columns())]))

    if failures:
        lines.extend(["", "## Failures / 失败项", ""])
        lines.extend(_markdown_table(pd.DataFrame(failures)))
    return "\n".join(lines).rstrip() + "\n"


def build_top_candidates(summary: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    columns = _top_candidate_columns()
    if summary.empty:
        return pd.DataFrame(columns=columns)
    if "recheck_priority_score" not in summary.columns:
        summary = _add_recheck_queue_columns(summary)

    rows = [_top_candidate_row(row) for row in summary.itertuples(index=False)]
    candidates = pd.DataFrame(rows)
    candidates = candidates.sort_values(
        ["category_rank", "attention_score", "calibrated_high_probability_score"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    candidates["top_rank"] = candidates.index + 1
    candidates = candidates[columns]
    if limit > 0:
        return candidates.head(limit).reset_index(drop=True)
    return candidates


def render_top_candidates_report(
    top_candidates: pd.DataFrame,
    summary: pd.DataFrame,
    failures: list[dict[str, str]] | tuple[dict[str, str], ...],
) -> str:
    lines = [
        "# Top Candidates / 最佳候选总结",
        "",
        "This page is the first file to read after a batch scan.",
        "批量扫描结束后，优先看这个文件。",
        "",
        "It separates true high-probability candidates from near misses and ordinary filtered names.",
        "它会把真正高概率候选、接近机会、普通过滤项分开，方便你快速判断下一步。",
        "",
    ]
    if summary.empty:
        lines.extend(["No tickers were analyzed successfully.", "没有股票成功完成分析。", ""])
    else:
        lines.extend(_top_candidate_snapshot(summary))
        lines.extend(["", "## Priority List / 优先列表", ""])
        lines.extend(_markdown_table(top_candidates))
        lines.extend(
            [
                "",
                "## How To Use / 怎么用",
                "",
                "- `high_probability`: highest priority; still read the blocker and risk fields before acting.",
                "- `high_probability`: 最高优先级；仍然要先看卡点和风险字段。",
                "- `near_watchlist`: close to passing; monitor the listed trigger and missing items.",
                "- `near_watchlist`: 接近通过；重点看触发条件和缺失项。",
                "- `recheck_queue`: not good enough now, but worth checking again when trigger conditions improve.",
                "- `recheck_queue`: 现在不够好，但触发条件改善后值得重新检查。",
                "- `filtered_out`: low priority; only revisit after data, risk, or signal conditions change materially.",
                "- `filtered_out`: 低优先级；只有数据、风险或信号明显改善后再看。",
                "- `Data needs repair`: do not treat the ticker as a high-confidence candidate until source conflicts or missing layers are fixed.",
                "- `Data needs repair`: 数据需要修复时，不要把这只股票当作高可信候选，先处理数据源冲突或缺失层。",
                "",
            ]
        )
    if failures:
        lines.extend(["", "## Failures / 失败项", ""])
        lines.extend(_markdown_table(pd.DataFrame(failures)))
    return "\n".join(lines).rstrip() + "\n"


def _top_candidate_snapshot(summary: pd.DataFrame) -> list[str]:
    category_frame = build_top_candidates(summary, limit=0)
    counts = (
        category_frame.groupby(["candidate_category", "candidate_category_zh"], dropna=False)
        .size()
        .reset_index(name="ticker_count")
        if not category_frame.empty
        else pd.DataFrame(columns=["candidate_category", "candidate_category_zh", "ticker_count"])
    )
    lines = ["## Scan Snapshot / 扫描快照", ""]
    lines.extend(
        [
            f"- Tickers analyzed / 完成分析数: `{len(summary)}`",
            (
                "- Calibrated high-probability passed / 校准后高概率通过: "
                f"`{int((summary['calibrated_quality_gate_passed'] == True).sum())}`"
            ),
            (
                "- Near watchlist / 接近机会: "
                f"`{int((summary['calibrated_watchlist_status'] == 'close_but_not_ready').sum())}`"
            ),
            (
                "- Re-check queue / 重新检查队列: "
                f"`{int((_recheck_queue(summary)).shape[0])}`"
            ),
            "",
            "Category counts / 分类数量:",
            "",
        ]
    )
    lines.extend(_markdown_table(counts))
    return lines


def _top_candidate_row(row: object) -> dict[str, object]:
    category, category_zh, category_rank = _candidate_category(row)
    attention_score = _attention_score(row, category)
    data_repair_note = _data_repair_display_note(row)
    return {
        "top_rank": 0,
        "ticker": getattr(row, "ticker", ""),
        "candidate_category": category,
        "candidate_category_zh": category_zh,
        "category_rank": category_rank,
        "attention_score": attention_score,
        "final_decision_zh": getattr(row, "final_decision_zh", ""),
        "focus_horizon": getattr(row, "final_focus_horizon", getattr(row, "focus_horizon", "")),
        "latest_price": getattr(row, "latest_price", ""),
        "calibrated_high_probability_score": getattr(row, "calibrated_high_probability_score", ""),
        "calibrated_win_probability": getattr(row, "calibrated_win_probability", ""),
        "calibrated_probability_confidence": getattr(row, "calibrated_probability_confidence", ""),
        "market_score": getattr(row, "market_score", ""),
        "relative_strength_score": getattr(row, "relative_strength_score", ""),
        "data_quality_score": getattr(row, "data_quality_score", ""),
        "data_readiness_level": getattr(row, "data_readiness_level", ""),
        "data_readiness_score": getattr(row, "data_readiness_score", ""),
        "data_repair_note_zh": data_repair_note,
        "overall_risk_level": getattr(row, "overall_risk_level", ""),
        "primary_blocker_zh": getattr(row, "primary_blocker_zh", ""),
        "missing_items_zh": getattr(row, "calibrated_watchlist_missing_items_zh", ""),
        "next_step_zh": getattr(row, "final_next_step_zh", ""),
        "recheck_trigger_price": getattr(row, "recheck_trigger_price", ""),
        "recheck_trigger_distance_pct": getattr(row, "recheck_trigger_distance_pct", ""),
        "recheck_action_type_zh": getattr(row, "recheck_action_type_zh", ""),
        "report_path": getattr(row, "report_path", ""),
    }


def _candidate_category(row: object) -> tuple[str, str, int]:
    if bool(getattr(row, "calibrated_quality_gate_passed", False)):
        return "high_probability", "高概率候选", 0
    watchlist_status = str(getattr(row, "calibrated_watchlist_status", ""))
    if watchlist_status == "close_but_not_ready":
        return "near_watchlist", "接近机会", 1
    if watchlist_status == "early_watch":
        return "early_watchlist", "早期观察", 2
    if float(getattr(row, "recheck_priority_score", 0.0) or 0.0) >= 35.0:
        return "recheck_queue", "重新检查队列", 3
    return "filtered_out", "被过滤", 4


def _attention_score(row: object, category: str) -> float:
    category_bonus = {
        "high_probability": 25.0,
        "near_watchlist": 15.0,
        "early_watchlist": 8.0,
        "recheck_queue": 4.0,
        "filtered_out": 0.0,
    }.get(category, 0.0)
    calibrated_score = float(getattr(row, "calibrated_high_probability_score", 0.0) or 0.0)
    win_probability = float(getattr(row, "calibrated_win_probability", 0.0) or 0.0) * 100.0
    confidence = float(getattr(row, "calibrated_probability_confidence", 0.0) or 0.0)
    recheck_score = float(getattr(row, "recheck_priority_score", 0.0) or 0.0)
    data_quality = float(getattr(row, "data_quality_score", 0.0) or 0.0)
    data_readiness = float(getattr(row, "data_readiness_score", data_quality) or data_quality)
    score = (
        category_bonus
        + calibrated_score * 0.40
        + win_probability * 0.20
        + confidence * 0.15
        + recheck_score * 0.15
        + min(data_quality, data_readiness) * 0.10
    )
    return round(max(0.0, min(100.0, score)), 2)


def _data_repair_display_note(row: object) -> str:
    level = str(getattr(row, "data_readiness_level", ""))
    blockers = str(getattr(row, "data_readiness_primary_blockers_zh", "") or "无")
    if level in {"insufficient", "conflict_warning"}:
        return f"Data needs repair / 数据需要修复：{blockers}"
    if level == "usable_with_warnings":
        return f"Usable with warnings / 可用但有提示：{blockers}"
    return "Ready / 数据可用"


def _top_candidate_columns() -> list[str]:
    return [
        "top_rank",
        "ticker",
        "candidate_category",
        "candidate_category_zh",
        "attention_score",
        "final_decision_zh",
        "focus_horizon",
        "latest_price",
        "calibrated_high_probability_score",
        "calibrated_win_probability",
        "calibrated_probability_confidence",
        "market_score",
        "relative_strength_score",
        "data_quality_score",
        "data_readiness_level",
        "data_readiness_score",
        "data_repair_note_zh",
        "overall_risk_level",
        "primary_blocker_zh",
        "missing_items_zh",
        "next_step_zh",
        "recheck_trigger_price",
        "recheck_trigger_distance_pct",
        "recheck_action_type_zh",
        "report_path",
    ]


def _add_recheck_queue_columns(summary: pd.DataFrame) -> pd.DataFrame:
    enriched = summary.copy()
    if enriched.empty:
        return enriched
    recheck_scores = []
    recheck_levels = []
    recheck_levels_zh = []
    recheck_reasons = []
    recheck_reasons_zh = []
    recheck_trigger_prices = []
    recheck_trigger_distance_pcts = []
    recheck_trigger_distance_labels = []
    recheck_trigger_distance_labels_zh = []
    recheck_trigger_notes = []
    recheck_trigger_notes_zh = []
    recheck_price_trigger_met = []
    recheck_non_price_blocker_counts = []
    recheck_action_types = []
    recheck_action_types_zh = []
    recheck_action_notes = []
    recheck_action_notes_zh = []
    requested_periods = []
    analysis_periods = []
    auto_period_upgraded_values = []
    auto_period_upgrade_reasons = []
    auto_period_upgrade_reasons_zh = []
    data_remediation_actions = []
    data_remediation_actions_zh = []
    data_remediation_priorities = []
    data_remediation_priorities_zh = []
    data_remediation_notes = []
    data_remediation_notes_zh = []
    for row in enriched.itertuples(index=False):
        trigger_profile = _recheck_trigger_profile(row)
        action_profile = _recheck_action_profile(row, trigger_profile)
        score = _recheck_priority_score(row, trigger_profile["distance_pct"])
        level, level_zh = _recheck_priority_level(score)
        recheck_scores.append(score)
        recheck_levels.append(level)
        recheck_levels_zh.append(level_zh)
        recheck_reasons.append(getattr(row, "blocker_recheck_trigger", "No re-check trigger."))
        recheck_reasons_zh.append(getattr(row, "blocker_recheck_trigger_zh", "没有重新检查触发条件。"))
        recheck_trigger_prices.append(trigger_profile["trigger_price"])
        recheck_trigger_distance_pcts.append(trigger_profile["distance_pct"])
        recheck_trigger_distance_labels.append(trigger_profile["label"])
        recheck_trigger_distance_labels_zh.append(trigger_profile["label_zh"])
        recheck_trigger_notes.append(trigger_profile["note"])
        recheck_trigger_notes_zh.append(trigger_profile["note_zh"])
        recheck_price_trigger_met.append(action_profile["price_trigger_met"])
        recheck_non_price_blocker_counts.append(action_profile["non_price_blocker_count"])
        recheck_action_types.append(action_profile["action_type"])
        recheck_action_types_zh.append(action_profile["action_type_zh"])
        recheck_action_notes.append(action_profile["action_note"])
        recheck_action_notes_zh.append(action_profile["action_note_zh"])
        requested_periods.append(getattr(row, "requested_period", getattr(row, "analysis_period", "unknown")))
        analysis_periods.append(getattr(row, "analysis_period", "unknown"))
        auto_period_upgraded_values.append(bool(getattr(row, "auto_period_upgraded", False)))
        auto_period_upgrade_reasons.append(
            getattr(row, "auto_period_upgrade_reason", "Requested period was used without automatic extension.")
        )
        auto_period_upgrade_reasons_zh.append(
            getattr(row, "auto_period_upgrade_reason_zh", "系统使用了你请求的数据周期，没有自动延长。")
        )
        data_remediation_actions.append(getattr(row, "data_remediation_action", "none"))
        data_remediation_actions_zh.append(getattr(row, "data_remediation_action_zh", "无需补数据"))
        data_remediation_priorities.append(getattr(row, "data_remediation_priority", "none"))
        data_remediation_priorities_zh.append(getattr(row, "data_remediation_priority_zh", "无"))
        data_remediation_notes.append(
            getattr(row, "data_remediation_note", "Data quality is acceptable for the current scan.")
        )
        data_remediation_notes_zh.append(
            getattr(row, "data_remediation_note_zh", "当前扫描的数据质量可以接受。")
        )
    enriched["recheck_priority_score"] = recheck_scores
    enriched["recheck_priority_level"] = recheck_levels
    enriched["recheck_priority_level_zh"] = recheck_levels_zh
    enriched["recheck_reason"] = recheck_reasons
    enriched["recheck_reason_zh"] = recheck_reasons_zh
    enriched["recheck_trigger_price"] = recheck_trigger_prices
    enriched["recheck_trigger_distance_pct"] = recheck_trigger_distance_pcts
    enriched["recheck_trigger_distance_label"] = recheck_trigger_distance_labels
    enriched["recheck_trigger_distance_label_zh"] = recheck_trigger_distance_labels_zh
    enriched["recheck_trigger_note"] = recheck_trigger_notes
    enriched["recheck_trigger_note_zh"] = recheck_trigger_notes_zh
    enriched["recheck_price_trigger_met"] = recheck_price_trigger_met
    enriched["recheck_non_price_blocker_count"] = recheck_non_price_blocker_counts
    enriched["recheck_action_type"] = recheck_action_types
    enriched["recheck_action_type_zh"] = recheck_action_types_zh
    enriched["recheck_action_note"] = recheck_action_notes
    enriched["recheck_action_note_zh"] = recheck_action_notes_zh
    enriched["requested_period"] = requested_periods
    enriched["analysis_period"] = analysis_periods
    enriched["auto_period_upgraded"] = auto_period_upgraded_values
    enriched["auto_period_upgrade_reason"] = auto_period_upgrade_reasons
    enriched["auto_period_upgrade_reason_zh"] = auto_period_upgrade_reasons_zh
    enriched["data_remediation_action"] = data_remediation_actions
    enriched["data_remediation_action_zh"] = data_remediation_actions_zh
    enriched["data_remediation_priority"] = data_remediation_priorities
    enriched["data_remediation_priority_zh"] = data_remediation_priorities_zh
    enriched["data_remediation_note"] = data_remediation_notes
    enriched["data_remediation_note_zh"] = data_remediation_notes_zh
    if "data_repair_actions_applied" not in enriched.columns:
        enriched["data_repair_actions_applied"] = "none"
    if "data_repair_actions_applied_zh" not in enriched.columns:
        enriched["data_repair_actions_applied_zh"] = "无"
    enriched["recheck_rank"] = 0

    queue_index = _recheck_queue(enriched).index
    if len(queue_index) > 0:
        ranks = (
            enriched.loc[queue_index]
            .sort_values(
                ["recheck_priority_score", "calibrated_high_probability_score"],
                ascending=[False, False],
            )
            .index
        )
        for rank, index in enumerate(ranks, start=1):
            enriched.at[index, "recheck_rank"] = rank
    return enriched


def _recheck_queue(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or "recheck_priority_score" not in summary.columns:
        return summary.iloc[0:0].copy()
    queue = summary[
        (summary["calibrated_quality_gate_passed"] == False)  # noqa: E712
        & (summary["recheck_priority_score"] > 0)
    ].copy()
    return queue.sort_values(
        ["recheck_priority_score", "calibrated_high_probability_score"],
        ascending=[False, False],
    )


def _recheck_action_summary(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "recheck_action_type",
        "recheck_action_type_zh",
        "ticker_count",
        "avg_recheck_priority_score",
        "max_recheck_priority_score",
        "top_tickers",
        "top_trigger_notes_zh",
    ]
    queue = _recheck_queue(summary)
    if queue.empty or "recheck_action_type" not in queue.columns:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for action_type, frame in queue.groupby("recheck_action_type", sort=False):
        ordered = frame.sort_values("recheck_priority_score", ascending=False)
        rows.append(
            {
                "recheck_action_type": action_type,
                "recheck_action_type_zh": ordered["recheck_action_type_zh"].iloc[0],
                "ticker_count": int(len(ordered)),
                "avg_recheck_priority_score": round(
                    float(ordered["recheck_priority_score"].mean()),
                    2,
                ),
                "max_recheck_priority_score": round(
                    float(ordered["recheck_priority_score"].max()),
                    2,
                ),
                "top_tickers": ", ".join(str(ticker) for ticker in ordered["ticker"].head(5)),
                "top_trigger_notes_zh": "；".join(
                    str(note) for note in ordered["recheck_trigger_note_zh"].head(3)
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["ticker_count", "max_recheck_priority_score"],
        ascending=[False, False],
    )


def _recheck_priority_score(row: object, trigger_distance_pct: float | None = None) -> float:
    if bool(getattr(row, "calibrated_quality_gate_passed", False)):
        return 0.0
    blocker_score = float(getattr(row, "blocker_resolution_score", 0.0))
    calibrated_score = float(getattr(row, "calibrated_high_probability_score", 0.0))
    watchlist_bonus = _watchlist_recheck_bonus(str(getattr(row, "calibrated_watchlist_status", "")))
    trigger_bonus = _trigger_distance_bonus(trigger_distance_pct)
    score = blocker_score * 0.55 + calibrated_score * 0.35 + watchlist_bonus + trigger_bonus
    return round(max(0.0, min(100.0, score)), 2)


def _recheck_action_profile(row: object, trigger_profile: dict[str, object]) -> dict[str, object]:
    blockers = _split_blockers(str(getattr(row, "priority_blockers", "")))
    non_price_blockers = [blocker for blocker in blockers if blocker != "entry_not_executable"]
    distance_pct = trigger_profile.get("distance_pct")
    price_trigger_met = isinstance(distance_pct, (float, int)) and float(distance_pct) <= 0

    risk_blockers = {
        "overall_risk_high",
        "event_risk_blocks_entry",
        "sentiment_risk_high",
        "valuation_risk_high",
    }
    evidence_blockers = {
        "confidence_too_low",
        "signal_score_too_low",
        "backtest_sample_too_small",
        "backtest_win_rate_too_low",
        "backtest_average_return_not_positive",
        "relative_strength_weak",
    }

    if any(blocker in risk_blockers for blocker in blockers):
        action_type = "wait_for_risk_clearance"
        action_type_zh = "等待风险解除"
        action_note = "Do not treat price proximity as enough; a risk filter is still blocking the setup."
        action_note_zh = "不要只看价格接近，当前仍有风险过滤项阻止机会。"
    elif "data_quality_insufficient" in blockers:
        action_type = "refresh_or_extend_data"
        action_type_zh = "刷新或延长数据"
        action_note = "Improve data coverage before trusting the re-check queue ranking."
        action_note_zh = "先提高数据覆盖率，再信任重新检查队列排名。"
    elif "entry_not_executable" in blockers and not price_trigger_met:
        action_type = "wait_for_price_trigger"
        action_type_zh = "等待价格触发"
        action_note = "Price has not reached the re-check trigger yet."
        action_note_zh = "价格还没有到达重新检查触发条件。"
    elif "trade_plan_invalid" in blockers:
        action_type = "wait_for_valid_trade_plan"
        action_type_zh = "等待交易计划有效"
        action_note = "The setup still needs a valid entry, stop, target, and risk/reward plan."
        action_note_zh = "当前仍需要有效入场、止损、目标价和盈亏比计划。"
    elif any(blocker in evidence_blockers for blocker in blockers):
        action_type = "improve_signal_evidence"
        action_type_zh = "等待信号证据改善"
        action_note = "The price condition may be close, but signal or evidence quality is still insufficient."
        action_note_zh = "价格条件可能接近，但信号或证据质量仍不足。"
    else:
        action_type = "monitor"
        action_type_zh = "普通跟踪"
        action_note = "Monitor the trigger and ranking changes."
        action_note_zh = "继续跟踪触发条件和排名变化。"

    return {
        "price_trigger_met": bool(price_trigger_met),
        "non_price_blocker_count": len(non_price_blockers),
        "action_type": action_type,
        "action_type_zh": action_type_zh,
        "action_note": action_note,
        "action_note_zh": action_note_zh,
    }


def _split_blockers(value: str) -> list[str]:
    return [
        blocker.strip()
        for blocker in value.replace("；", ";").split(";")
        if blocker.strip() and blocker.strip() != "none"
    ]


def _recheck_trigger_profile(row: object) -> dict[str, object]:
    latest_price = _safe_float(getattr(row, "latest_price", None))
    trigger_price = _safe_float(getattr(row, "calibrated_watchlist_trigger_price", None))
    if trigger_price is None:
        trigger_price = _safe_float(getattr(row, "watchlist_trigger_price", None))
    if latest_price is None or latest_price <= 0 or trigger_price is None:
        return {
            "trigger_price": trigger_price,
            "distance_pct": None,
            "label": "unknown",
            "label_zh": "未知",
            "note": "Trigger distance is unavailable.",
            "note_zh": "触发价距离不可用。",
        }

    distance_pct = trigger_price / latest_price - 1
    label, label_zh = _trigger_distance_label(distance_pct)
    direction = "above" if distance_pct >= 0 else "below"
    direction_zh = "高于" if distance_pct >= 0 else "低于"
    return {
        "trigger_price": round(trigger_price, 4),
        "distance_pct": round(distance_pct, 6),
        "label": label,
        "label_zh": label_zh,
        "note": (
            f"Re-check trigger is {abs(distance_pct):.2%} {direction} the latest price "
            f"({latest_price:.4f} -> {trigger_price:.4f})."
        ),
        "note_zh": (
            f"重新检查触发价{direction_zh}当前价{abs(distance_pct):.2%}"
            f"（{latest_price:.4f} -> {trigger_price:.4f}）。"
        ),
    }


def _safe_float(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return numeric


def _trigger_distance_label(distance_pct: float) -> tuple[str, str]:
    absolute = abs(distance_pct)
    if absolute <= 0.02:
        return "very_close", "非常接近"
    if absolute <= 0.05:
        return "close", "接近"
    if absolute <= 0.10:
        return "moderate", "中等距离"
    return "far", "距离较远"


def _trigger_distance_bonus(distance_pct: float | None) -> float:
    if distance_pct is None:
        return 0.0
    absolute = abs(distance_pct)
    if absolute <= 0.02:
        return 8.0
    if absolute <= 0.05:
        return 5.0
    if absolute <= 0.10:
        return 2.0
    return 0.0


def _watchlist_recheck_bonus(status: str) -> float:
    if status == "close_but_not_ready":
        return 10.0
    if status == "early_watch":
        return 5.0
    return 0.0


def _recheck_priority_level(score: float) -> tuple[str, str]:
    if score >= 75:
        return "urgent_recheck", "优先重新检查"
    if score >= 55:
        return "watch_soon", "近期观察"
    if score >= 35:
        return "monitor", "普通跟踪"
    return "low_priority", "低优先级"


def _data_remediation_profile(row: pd.Series, period: str) -> dict[str, object]:
    data_quality_score = float(row.get("data_quality_score", 0.0))
    blockers = _split_blockers(str(row.get("priority_blockers", "")))
    has_data_blocker = "data_quality_insufficient" in blockers or data_quality_score < 70.0
    if not has_data_blocker:
        return {
            "action": "none",
            "action_zh": "无需补数据",
            "priority": "none",
            "priority_zh": "无",
            "note": "Data quality is acceptable for the current scan.",
            "note_zh": "当前扫描的数据质量可以接受。",
        }

    if period != "5y":
        return {
            "action": "rerun_with_5y",
            "action_zh": "用5年数据重跑",
            "priority": "high",
            "priority_zh": "高",
            "note": (
                f"Data quality score is {data_quality_score:.1f}. "
                f"The current scan period is {period}; rerun with --period 5y before trusting this ranking."
            ),
            "note_zh": (
                f"数据质量分数为{data_quality_score:.1f}。当前扫描周期是{period}；"
                "建议先用 --period 5y 重新运行，再信任这个排名。"
            ),
        }

    return {
        "action": "refresh_missing_layers",
        "action_zh": "刷新缺失数据层",
        "priority": "high",
        "priority_zh": "高",
        "note": (
            f"Data quality score is {data_quality_score:.1f} even with a 5y scan. "
            "Refresh price, market, sector, fundamental, event, news, analyst, and valuation layers."
        ),
        "note_zh": (
            f"即使用5年扫描，数据质量分数仍为{data_quality_score:.1f}。"
            "建议刷新价格、大盘、板块、基本面、事件、新闻、分析师和估值数据层。"
        ),
    }


def _scan_row(result: RealTickerAnalysisResult, period: str) -> dict[str, object]:
    focus = result.analysis.sort_values(
        ["calibrated_quality_gate_passed", "calibrated_high_probability_score", "signal_score"],
        ascending=[False, False, False],
    ).iloc[0]
    snapshot = result.snapshot or {}
    data_remediation = _data_remediation_profile(focus, period)
    return {
        "ticker": result.ticker,
        "requested_period": getattr(result, "requested_period", period),
        "analysis_period": period,
        "auto_period_upgraded": bool(getattr(result, "auto_period_upgraded", False)),
        "auto_period_upgrade_reason": getattr(result, "auto_period_upgrade_reason", ""),
        "auto_period_upgrade_reason_zh": getattr(result, "auto_period_upgrade_reason_zh", ""),
        "company_name": snapshot.get("company_name") or "",
        "sector": focus.get("sector", "") or snapshot.get("sector") or "",
        "industry": focus.get("industry", "") or snapshot.get("industry") or "",
        "final_decision": focus["final_decision"],
        "final_decision_zh": focus["final_decision_zh"],
        "final_focus_horizon": focus["final_focus_horizon"],
        "final_focus_horizon_zh": focus["final_focus_horizon_zh"],
        "final_score": float(focus["final_score"]),
        "latest_price": float(focus["latest_price"]),
        "final_watchlist_status": focus["final_watchlist_status"],
        "final_watchlist_status_zh": focus["final_watchlist_status_zh"],
        "final_reason": focus["final_reason"],
        "final_reason_zh": focus["final_reason_zh"],
        "final_next_step": focus["final_next_step"],
        "final_next_step_zh": focus["final_next_step_zh"],
        "data_remediation_action": data_remediation["action"],
        "data_remediation_action_zh": data_remediation["action_zh"],
        "data_remediation_priority": data_remediation["priority"],
        "data_remediation_priority_zh": data_remediation["priority_zh"],
        "data_remediation_note": data_remediation["note"],
        "data_remediation_note_zh": data_remediation["note_zh"],
        "data_readiness_level": focus.get("data_readiness_level", "unknown"),
        "data_readiness_level_zh": focus.get("data_readiness_level_zh", "未知"),
        "data_readiness_score": float(focus.get("data_readiness_score", 0.0)),
        "data_readiness_repair_priority": focus.get("data_readiness_repair_priority", "unknown"),
        "data_readiness_repair_priority_zh": focus.get("data_readiness_repair_priority_zh", "未知"),
        "data_readiness_primary_blockers": focus.get("data_readiness_primary_blockers", "none"),
        "data_readiness_primary_blockers_zh": focus.get("data_readiness_primary_blockers_zh", "无"),
        "data_source_validation_status": focus.get("data_source_validation_status", "unknown"),
        "data_source_validation_status_zh": focus.get("data_source_validation_status_zh", "未知"),
        "data_needs_repair": bool(focus.get("data_needs_repair", False)),
        "primary_blocker": focus["primary_blocker"],
        "primary_blocker_zh": focus["primary_blocker_zh"],
        "priority_blockers": focus["priority_blockers"],
        "priority_blockers_zh": focus["priority_blockers_zh"],
        "priority_blocker_count": int(focus["priority_blocker_count"]),
        "priority_blocker_note": focus["priority_blocker_note"],
        "priority_blocker_note_zh": focus["priority_blocker_note_zh"],
        "primary_blocker_resolution": focus["primary_blocker_resolution"],
        "primary_blocker_resolution_zh": focus["primary_blocker_resolution_zh"],
        "blocker_resolution_steps": focus["blocker_resolution_steps"],
        "blocker_resolution_steps_zh": focus["blocker_resolution_steps_zh"],
        "blocker_recheck_trigger": focus["blocker_recheck_trigger"],
        "blocker_recheck_trigger_zh": focus["blocker_recheck_trigger_zh"],
        "primary_blocker_progress": float(focus["primary_blocker_progress"]),
        "blocker_resolution_score": float(focus["blocker_resolution_score"]),
        "blocker_resolution_level": focus["blocker_resolution_level"],
        "blocker_resolution_level_zh": focus["blocker_resolution_level_zh"],
        "blocker_resolution_gap": focus["blocker_resolution_gap"],
        "blocker_resolution_gap_zh": focus["blocker_resolution_gap_zh"],
        "horizon_alignment_score": float(focus["horizon_alignment_score"]),
        "horizon_alignment_label": focus["horizon_alignment_label"],
        "horizon_alignment_label_zh": focus["horizon_alignment_label_zh"],
        "constructive_horizon_count": int(focus["constructive_horizon_count"]),
        "weak_horizon_count": int(focus["weak_horizon_count"]),
        "risk_wait_horizon_count": int(focus["risk_wait_horizon_count"]),
        "horizon_signal_score_spread": float(focus["horizon_signal_score_spread"]),
        "horizon_alignment_note": focus["horizon_alignment_note"],
        "horizon_alignment_note_zh": focus["horizon_alignment_note_zh"],
        "market_regime": focus["market_regime"],
        "market_regime_zh": focus["market_regime_zh"],
        "market_regime_note": focus["market_regime_note"],
        "market_regime_note_zh": focus["market_regime_note_zh"],
        "market_regime_signal_delta": float(focus["market_regime_signal_delta"]),
        "market_regime_confidence_delta": float(
            focus["market_regime_confidence_delta"]
        ),
        "market_regime_sample_delta": int(focus["market_regime_sample_delta"]),
        "market_regime_win_rate_delta": float(focus["market_regime_win_rate_delta"]),
        "market_regime_average_return_delta": float(
            focus["market_regime_average_return_delta"]
        ),
        "focus_horizon": focus["horizon"],
        "screening_action": focus["screening_action"],
        "screening_action_zh": focus["screening_action_zh"],
        "screening_profile": focus.get("screening_profile", "default"),
        "screening_profile_zh": focus.get("screening_profile_zh", "默认规则"),
        "quality_gate_passed": bool(focus["quality_gate_passed"]),
        "high_probability_score": float(focus["high_probability_score"]),
        "high_probability_level": focus["high_probability_level"],
        "calibrated_win_probability": float(focus["calibrated_win_probability"]),
        "calibrated_win_probability_raw": float(focus["calibrated_win_probability_raw"]),
        "probability_calibration_adjustment": float(
            focus["probability_calibration_adjustment"]
        ),
        "probability_calibration_source": focus["probability_calibration_source"],
        "probability_calibration_source_zh": focus["probability_calibration_source_zh"],
        "probability_calibration_sample_count": int(
            focus["probability_calibration_sample_count"]
        ),
        "probability_calibration_action": focus["probability_calibration_action"],
        "probability_calibration_action_zh": focus[
            "probability_calibration_action_zh"
        ],
        "calibrated_probability_level": focus["calibrated_probability_level"],
        "calibrated_probability_level_zh": focus["calibrated_probability_level_zh"],
        "calibrated_probability_confidence": float(
            focus["calibrated_probability_confidence"]
        ),
        "calibrated_probability_confidence_level": focus[
            "calibrated_probability_confidence_level"
        ],
        "calibrated_probability_confidence_level_zh": focus[
            "calibrated_probability_confidence_level_zh"
        ],
        "calibrated_probability_note": focus["calibrated_probability_note"],
        "calibrated_probability_note_zh": focus["calibrated_probability_note_zh"],
        "watchlist_status": focus["watchlist_status"],
        "watchlist_status_zh": focus["watchlist_status_zh"],
        "watchlist_gap_score": float(focus["watchlist_gap_score"]),
        "watchlist_ready_items": focus["watchlist_ready_items"],
        "watchlist_ready_items_zh": focus["watchlist_ready_items_zh"],
        "watchlist_missing_items": focus["watchlist_missing_items"],
        "watchlist_missing_items_zh": focus["watchlist_missing_items_zh"],
        "watchlist_missing_count": int(focus["watchlist_missing_count"]),
        "watchlist_trigger_price": float(focus["watchlist_trigger_price"]),
        "watchlist_recheck_reason": focus["watchlist_recheck_reason"],
        "watchlist_recheck_reason_zh": focus["watchlist_recheck_reason_zh"],
        "signal_score": float(focus["signal_score"]),
        "confidence_score": float(focus["confidence_score"]),
        "data_quality_score": float(focus["data_quality_score"]),
        "data_quality_weakest_layer": focus["data_quality_weakest_layer"],
        "data_quality_weakest_layer_zh": focus["data_quality_weakest_layer_zh"],
        "data_quality_weak_layers": focus["data_quality_weak_layers"],
        "data_quality_weak_layers_zh": focus["data_quality_weak_layers_zh"],
        "data_quality_repair_actions": focus["data_quality_repair_actions"],
        "data_quality_repair_actions_zh": focus["data_quality_repair_actions_zh"],
        "data_quality_repair_priority": focus["data_quality_repair_priority"],
        "data_quality_repair_priority_zh": focus["data_quality_repair_priority_zh"],
        "data_repair_actions_applied": focus.get("data_repair_actions_applied", "none"),
        "data_repair_actions_applied_zh": focus.get("data_repair_actions_applied_zh", "无"),
        "price_health_score": float(focus["price_health_score"]),
        "price_health_level": focus["price_health_level"],
        "price_health_level_zh": focus["price_health_level_zh"],
        "price_health_issue_count": int(focus["price_health_issue_count"]),
        "price_max_calendar_gap_days": int(focus["price_max_calendar_gap_days"]),
        "price_large_gap_count": int(focus["price_large_gap_count"]),
        "price_zero_volume_days": int(focus["price_zero_volume_days"]),
        "price_missing_ohlcv_rows": int(focus["price_missing_ohlcv_rows"]),
        "price_extreme_return_count": int(focus["price_extreme_return_count"]),
        "price_health_note": focus["price_health_note"],
        "price_health_note_zh": focus["price_health_note_zh"],
        "overall_risk_level": focus["overall_risk_level"],
        "market_score": float(focus["market_score"]),
        "relative_strength_score": float(focus["relative_strength_score"]),
        "fundamental_score": float(focus["fundamental_score"]),
        "sector_score": float(focus["sector_score"]),
        "event_risk_level": focus["event_risk_level"],
        "sentiment_score": float(focus.get("sentiment_score", 50.0)),
        "sentiment_risk_level": focus.get("sentiment_risk_level", "unknown"),
        "analyst_score": float(focus.get("analyst_score", 50.0)),
        "analyst_risk_level": focus.get("analyst_risk_level", "unknown"),
        "valuation_score": float(focus.get("valuation_score", 50.0)),
        "valuation_risk_level": focus.get("valuation_risk_level", "unknown"),
        "screening_backtest_entry_type": focus["screening_backtest_entry_type"],
        "screening_backtest_trade_count": int(focus["screening_backtest_trade_count"]),
        "screening_backtest_win_rate": float(focus["screening_backtest_win_rate"]),
        "screening_backtest_stop_hit_rate": float(
            focus["screening_backtest_stop_hit_rate"]
        ),
        "screening_backtest_average_return": float(focus["screening_backtest_average_return"]),
        "backtest_trust_score": float(focus["backtest_trust_score"]),
        "backtest_trust_level": focus["backtest_trust_level"],
        "backtest_trust_level_zh": focus["backtest_trust_level_zh"],
        "backtest_sample_score": float(focus["backtest_sample_score"]),
        "backtest_liquidity_score": float(focus["backtest_liquidity_score"]),
        "backtest_slippage_score": float(focus["backtest_slippage_score"]),
        "backtest_return_evidence_score": float(focus["backtest_return_evidence_score"]),
        "regime_coverage_score": float(focus["regime_coverage_score"]),
        "regime_coverage_level": focus["regime_coverage_level"],
        "regime_coverage_level_zh": focus["regime_coverage_level_zh"],
        "regime_coverage_regime_count": int(focus["regime_coverage_regime_count"]),
        "regime_coverage_dominant_regime": focus["regime_coverage_dominant_regime"],
        "regime_coverage_dominant_regime_zh": focus["regime_coverage_dominant_regime_zh"],
        "regime_coverage_dominant_share": float(focus["regime_coverage_dominant_share"]),
        "regime_coverage_note": focus["regime_coverage_note"],
        "regime_coverage_note_zh": focus["regime_coverage_note_zh"],
        "recent_backtest_score": float(focus["recent_backtest_score"]),
        "recent_backtest_level": focus["recent_backtest_level"],
        "recent_backtest_level_zh": focus["recent_backtest_level_zh"],
        "recent_backtest_trade_count": int(focus["recent_backtest_trade_count"]),
        "recent_backtest_win_rate": float(focus["recent_backtest_win_rate"]),
        "recent_backtest_average_return": float(focus["recent_backtest_average_return"]),
        "recent_backtest_return_delta": float(focus["recent_backtest_return_delta"]),
        "recent_backtest_note": focus["recent_backtest_note"],
        "recent_backtest_note_zh": focus["recent_backtest_note_zh"],
        "backtest_decay_score": float(focus["backtest_decay_score"]),
        "backtest_decay_level": focus["backtest_decay_level"],
        "backtest_decay_level_zh": focus["backtest_decay_level_zh"],
        "backtest_decay_early_trade_count": int(focus["backtest_decay_early_trade_count"]),
        "backtest_decay_late_trade_count": int(focus["backtest_decay_late_trade_count"]),
        "backtest_decay_early_win_rate": float(focus["backtest_decay_early_win_rate"]),
        "backtest_decay_late_win_rate": float(focus["backtest_decay_late_win_rate"]),
        "backtest_decay_early_average_return": float(
            focus["backtest_decay_early_average_return"]
        ),
        "backtest_decay_late_average_return": float(
            focus["backtest_decay_late_average_return"]
        ),
        "backtest_decay_win_rate_delta": float(focus["backtest_decay_win_rate_delta"]),
        "backtest_decay_average_return_delta": float(
            focus["backtest_decay_average_return_delta"]
        ),
        "backtest_decay_note": focus["backtest_decay_note"],
        "backtest_decay_note_zh": focus["backtest_decay_note_zh"],
        "backtest_trust_note": focus["backtest_trust_note"],
        "backtest_trust_note_zh": focus["backtest_trust_note_zh"],
        "entry_readiness_gate_passed": bool(focus["entry_readiness_gate_passed"]),
        "entry_readiness_status": focus["entry_readiness_status"],
        "entry_readiness_status_zh": focus["entry_readiness_status_zh"],
        "entry_readiness_score": float(focus["entry_readiness_score"]),
        "entry_readiness_reason": focus["entry_readiness_reason"],
        "entry_readiness_reason_zh": focus["entry_readiness_reason_zh"],
        "entry_readiness_note": focus["entry_readiness_note"],
        "entry_readiness_note_zh": focus["entry_readiness_note_zh"],
        "trade_plan_quality_gate_passed": bool(focus["trade_plan_quality_gate_passed"]),
        "trade_plan_quality_status": focus["trade_plan_quality_status"],
        "trade_plan_quality_status_zh": focus["trade_plan_quality_status_zh"],
        "trade_plan_quality_score": float(focus["trade_plan_quality_score"]),
        "trade_plan_quality_reason": focus["trade_plan_quality_reason"],
        "trade_plan_quality_reason_zh": focus["trade_plan_quality_reason_zh"],
        "trade_plan_quality_note": focus["trade_plan_quality_note"],
        "trade_plan_quality_note_zh": focus["trade_plan_quality_note_zh"],
        "calibration_action": focus["calibration_action"],
        "calibration_action_zh": focus["calibration_action_zh"],
        "calibration_level": focus["calibration_level"],
        "calibration_level_zh": focus["calibration_level_zh"],
        "recommended_signal_threshold": float(focus["recommended_signal_threshold"]),
        "recommended_confidence_threshold": float(focus["recommended_confidence_threshold"]),
        "recommended_backtest_sample_min": int(focus["recommended_backtest_sample_min"]),
        "recommended_backtest_win_rate_min": float(focus["recommended_backtest_win_rate_min"]),
        "recommended_backtest_average_return_min": float(
            focus["recommended_backtest_average_return_min"]
        ),
        "calibration_note": focus["calibration_note"],
        "calibration_note_zh": focus["calibration_note_zh"],
        "calibrated_screening_action": focus["calibrated_screening_action"],
        "calibrated_screening_action_zh": focus["calibrated_screening_action_zh"],
        "calibrated_quality_gate_passed": bool(focus["calibrated_quality_gate_passed"]),
        "calibrated_high_probability_score": float(focus["calibrated_high_probability_score"]),
        "calibrated_high_probability_level": focus["calibrated_high_probability_level"],
        "calibrated_quality_gate_fail_reasons": focus[
            "calibrated_quality_gate_fail_reasons"
        ],
        "calibrated_quality_gate_fail_reasons_zh": focus[
            "calibrated_quality_gate_fail_reasons_zh"
        ],
        "calibrated_watchlist_status": focus["calibrated_watchlist_status"],
        "calibrated_watchlist_status_zh": focus["calibrated_watchlist_status_zh"],
        "calibrated_watchlist_gap_score": float(focus["calibrated_watchlist_gap_score"]),
        "calibrated_watchlist_missing_items": focus["calibrated_watchlist_missing_items"],
        "calibrated_watchlist_missing_items_zh": focus[
            "calibrated_watchlist_missing_items_zh"
        ],
        "calibrated_watchlist_missing_count": int(
            focus["calibrated_watchlist_missing_count"]
        ),
        "calibrated_watchlist_trigger_price": float(
            focus["calibrated_watchlist_trigger_price"]
        ),
        "calibrated_watchlist_recheck_reason": focus[
            "calibrated_watchlist_recheck_reason"
        ],
        "calibrated_watchlist_recheck_reason_zh": focus[
            "calibrated_watchlist_recheck_reason_zh"
        ],
        "quality_gate_fail_reasons": focus["quality_gate_fail_reasons"],
        "quality_gate_fail_reasons_zh": focus["quality_gate_fail_reasons_zh"],
        "report_path": str(result.output_dir / "ticker_analysis.md"),
    }


def _scan_section(title: str, frame: pd.DataFrame) -> list[str]:
    lines = ["", title, ""]
    if frame.empty:
        lines.append("No rows. / 暂无。")
    else:
        lines.extend(_markdown_table(frame[_existing_columns(frame, _scan_display_columns())]))
    return lines


def _existing_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]


def _basic_scan_section(title: str, frame: pd.DataFrame) -> list[str]:
    lines = ["", title, ""]
    if frame.empty:
        lines.append("No rows. / 暂无。")
    else:
        lines.extend(_markdown_table(frame))
    return lines


def _scan_columns() -> list[str]:
    return [
        "ticker",
        "requested_period",
        "analysis_period",
        "auto_period_upgraded",
        "auto_period_upgrade_reason",
        "auto_period_upgrade_reason_zh",
        "company_name",
        "sector",
        "industry",
        "final_decision",
        "final_decision_zh",
        "final_focus_horizon",
        "final_focus_horizon_zh",
        "final_score",
        "latest_price",
        "final_watchlist_status",
        "final_watchlist_status_zh",
        "final_reason",
        "final_reason_zh",
        "final_next_step",
        "final_next_step_zh",
        "data_remediation_action",
        "data_remediation_action_zh",
        "data_remediation_priority",
        "data_remediation_priority_zh",
        "data_remediation_note",
        "data_remediation_note_zh",
        "data_readiness_level",
        "data_readiness_level_zh",
        "data_readiness_score",
        "data_readiness_repair_priority",
        "data_readiness_repair_priority_zh",
        "data_readiness_primary_blockers",
        "data_readiness_primary_blockers_zh",
        "data_source_validation_status",
        "data_source_validation_status_zh",
        "data_needs_repair",
        "primary_blocker",
        "primary_blocker_zh",
        "priority_blockers",
        "priority_blockers_zh",
        "priority_blocker_count",
        "priority_blocker_note",
        "priority_blocker_note_zh",
        "primary_blocker_resolution",
        "primary_blocker_resolution_zh",
        "blocker_resolution_steps",
        "blocker_resolution_steps_zh",
        "blocker_recheck_trigger",
        "blocker_recheck_trigger_zh",
        "primary_blocker_progress",
        "blocker_resolution_score",
        "blocker_resolution_level",
        "blocker_resolution_level_zh",
        "blocker_resolution_gap",
        "blocker_resolution_gap_zh",
        "recheck_rank",
        "recheck_priority_score",
        "recheck_priority_level",
        "recheck_priority_level_zh",
        "recheck_reason",
        "recheck_reason_zh",
        "recheck_trigger_price",
        "recheck_trigger_distance_pct",
        "recheck_trigger_distance_label",
        "recheck_trigger_distance_label_zh",
        "recheck_trigger_note",
        "recheck_trigger_note_zh",
        "recheck_price_trigger_met",
        "recheck_non_price_blocker_count",
        "recheck_action_type",
        "recheck_action_type_zh",
        "recheck_action_note",
        "recheck_action_note_zh",
        "horizon_alignment_score",
        "horizon_alignment_label",
        "horizon_alignment_label_zh",
        "constructive_horizon_count",
        "weak_horizon_count",
        "risk_wait_horizon_count",
        "horizon_signal_score_spread",
        "horizon_alignment_note",
        "horizon_alignment_note_zh",
        "market_regime",
        "market_regime_zh",
        "market_regime_note",
        "market_regime_note_zh",
        "market_regime_signal_delta",
        "market_regime_confidence_delta",
        "market_regime_sample_delta",
        "market_regime_win_rate_delta",
        "market_regime_average_return_delta",
        "focus_horizon",
        "screening_action",
        "screening_action_zh",
        "screening_profile",
        "screening_profile_zh",
        "quality_gate_passed",
        "high_probability_score",
        "high_probability_level",
        "calibrated_win_probability",
        "calibrated_win_probability_raw",
        "probability_calibration_adjustment",
        "probability_calibration_source",
        "probability_calibration_source_zh",
        "probability_calibration_sample_count",
        "probability_calibration_action",
        "probability_calibration_action_zh",
        "calibrated_probability_level",
        "calibrated_probability_level_zh",
        "calibrated_probability_confidence",
        "calibrated_probability_confidence_level",
        "calibrated_probability_confidence_level_zh",
        "calibrated_probability_note",
        "calibrated_probability_note_zh",
        "watchlist_status",
        "watchlist_status_zh",
        "watchlist_gap_score",
        "watchlist_ready_items",
        "watchlist_ready_items_zh",
        "watchlist_missing_items",
        "watchlist_missing_items_zh",
        "watchlist_missing_count",
        "watchlist_trigger_price",
        "watchlist_recheck_reason",
        "watchlist_recheck_reason_zh",
        "signal_score",
        "confidence_score",
        "data_quality_score",
        "data_quality_weakest_layer",
        "data_quality_weakest_layer_zh",
        "data_quality_weak_layers",
        "data_quality_weak_layers_zh",
        "data_quality_repair_actions",
        "data_quality_repair_actions_zh",
        "data_quality_repair_priority",
        "data_quality_repair_priority_zh",
        "data_repair_actions_applied",
        "data_repair_actions_applied_zh",
        "price_health_score",
        "price_health_level",
        "price_health_level_zh",
        "price_health_issue_count",
        "price_max_calendar_gap_days",
        "price_large_gap_count",
        "price_zero_volume_days",
        "price_missing_ohlcv_rows",
        "price_extreme_return_count",
        "price_health_note",
        "price_health_note_zh",
        "overall_risk_level",
        "market_score",
        "relative_strength_score",
        "fundamental_score",
        "sector_score",
        "event_risk_level",
        "sentiment_score",
        "sentiment_risk_level",
        "analyst_score",
        "analyst_risk_level",
        "valuation_score",
        "valuation_risk_level",
        "screening_backtest_entry_type",
        "screening_backtest_trade_count",
        "screening_backtest_win_rate",
        "screening_backtest_stop_hit_rate",
        "screening_backtest_average_return",
        "backtest_trust_score",
        "backtest_trust_level",
        "backtest_trust_level_zh",
        "backtest_sample_score",
        "backtest_liquidity_score",
        "backtest_slippage_score",
        "backtest_return_evidence_score",
        "regime_coverage_score",
        "regime_coverage_level",
        "regime_coverage_level_zh",
        "regime_coverage_regime_count",
        "regime_coverage_dominant_regime",
        "regime_coverage_dominant_regime_zh",
        "regime_coverage_dominant_share",
        "regime_coverage_note",
        "regime_coverage_note_zh",
        "recent_backtest_score",
        "recent_backtest_level",
        "recent_backtest_level_zh",
        "recent_backtest_trade_count",
        "recent_backtest_win_rate",
        "recent_backtest_average_return",
        "recent_backtest_return_delta",
        "recent_backtest_note",
        "recent_backtest_note_zh",
        "backtest_decay_score",
        "backtest_decay_level",
        "backtest_decay_level_zh",
        "backtest_decay_early_trade_count",
        "backtest_decay_late_trade_count",
        "backtest_decay_early_win_rate",
        "backtest_decay_late_win_rate",
        "backtest_decay_early_average_return",
        "backtest_decay_late_average_return",
        "backtest_decay_win_rate_delta",
        "backtest_decay_average_return_delta",
        "backtest_decay_note",
        "backtest_decay_note_zh",
        "backtest_trust_note",
        "backtest_trust_note_zh",
        "entry_readiness_gate_passed",
        "entry_readiness_status",
        "entry_readiness_status_zh",
        "entry_readiness_score",
        "entry_readiness_reason",
        "entry_readiness_reason_zh",
        "entry_readiness_note",
        "entry_readiness_note_zh",
        "trade_plan_quality_gate_passed",
        "trade_plan_quality_status",
        "trade_plan_quality_status_zh",
        "trade_plan_quality_score",
        "trade_plan_quality_reason",
        "trade_plan_quality_reason_zh",
        "trade_plan_quality_note",
        "trade_plan_quality_note_zh",
        "calibration_action",
        "calibration_action_zh",
        "calibration_level",
        "calibration_level_zh",
        "recommended_signal_threshold",
        "recommended_confidence_threshold",
        "recommended_backtest_sample_min",
        "recommended_backtest_win_rate_min",
        "recommended_backtest_average_return_min",
        "calibration_note",
        "calibration_note_zh",
        "calibrated_screening_action",
        "calibrated_screening_action_zh",
        "calibrated_quality_gate_passed",
        "calibrated_high_probability_score",
        "calibrated_high_probability_level",
        "calibrated_quality_gate_fail_reasons",
        "calibrated_quality_gate_fail_reasons_zh",
        "calibrated_watchlist_status",
        "calibrated_watchlist_status_zh",
        "calibrated_watchlist_gap_score",
        "calibrated_watchlist_missing_items",
        "calibrated_watchlist_missing_items_zh",
        "calibrated_watchlist_missing_count",
        "calibrated_watchlist_trigger_price",
        "calibrated_watchlist_recheck_reason",
        "calibrated_watchlist_recheck_reason_zh",
        "quality_gate_fail_reasons",
        "quality_gate_fail_reasons_zh",
        "report_path",
    ]


def _scan_display_columns() -> list[str]:
    return [
        "scan_rank",
        "ticker",
        "requested_period",
        "analysis_period",
        "auto_period_upgraded",
        "company_name",
        "sector",
        "industry",
        "final_decision",
        "final_score",
        "final_focus_horizon",
        "final_watchlist_status",
        "recheck_rank",
        "recheck_priority_score",
        "recheck_priority_level",
        "latest_price",
        "recheck_trigger_price",
        "recheck_trigger_distance_pct",
        "recheck_trigger_distance_label",
        "recheck_action_type",
        "recheck_non_price_blocker_count",
        "data_remediation_action",
        "data_readiness_level",
        "data_readiness_score",
        "data_source_validation_status",
        "data_needs_repair",
        "primary_blocker",
        "priority_blockers_zh",
        "priority_blocker_count",
        "blocker_resolution_score",
        "blocker_resolution_level",
        "blocker_recheck_trigger_zh",
        "horizon_alignment_label",
        "horizon_alignment_score",
        "market_regime",
        "focus_horizon",
        "screening_action",
        "screening_profile",
        "calibrated_quality_gate_passed",
        "calibrated_high_probability_score",
        "calibrated_win_probability",
        "calibrated_win_probability_raw",
        "probability_calibration_adjustment",
        "probability_calibration_action",
        "calibrated_probability_level",
        "calibrated_probability_confidence",
        "quality_gate_passed",
        "high_probability_score",
        "calibrated_screening_action",
        "calibrated_watchlist_status",
        "calibrated_watchlist_gap_score",
        "calibrated_watchlist_missing_count",
        "watchlist_status",
        "watchlist_gap_score",
        "watchlist_missing_count",
        "entry_readiness_status",
        "entry_readiness_score",
        "trade_plan_quality_status",
        "trade_plan_quality_score",
        "watchlist_trigger_price",
        "signal_score",
        "confidence_score",
        "data_quality_weakest_layer",
        "data_quality_repair_priority",
        "data_readiness_repair_priority",
        "price_health_score",
        "price_health_level",
        "data_repair_actions_applied",
        "sentiment_risk_level",
        "analyst_risk_level",
        "valuation_risk_level",
        "screening_backtest_win_rate",
        "screening_backtest_stop_hit_rate",
        "backtest_trust_score",
        "backtest_trust_level",
        "regime_coverage_score",
        "regime_coverage_level",
        "recent_backtest_score",
        "recent_backtest_level",
        "backtest_decay_score",
        "backtest_decay_level",
        "calibration_action",
        "recommended_signal_threshold",
        "recommended_backtest_win_rate_min",
        "calibrated_quality_gate_fail_reasons",
        "calibrated_watchlist_missing_items",
        "watchlist_missing_items",
    ]


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
    if isinstance(value, float):
        if -1.0 <= value <= 1.0:
            return f"{value:.2%}"
        return f"{value:.2f}"
    return str(value)
