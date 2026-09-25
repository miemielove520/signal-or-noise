from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from .json_io import dataframe_records, write_json


STATUS_RANK = {
    "filtered_out": 0,
    "not_ready": 0,
    "early_watch": 1,
    "close_but_not_ready": 2,
    "ready_high_probability": 3,
}


@dataclass(frozen=True)
class JournalResult:
    journal_dir: Path
    snapshot_path: Path
    report_path: Path
    current: pd.DataFrame
    previous: pd.DataFrame
    changes: pd.DataFrame


def write_daily_journal(
    scan_summary: pd.DataFrame,
    journal_root: str | Path = "outputs/journal",
    run_date: date | None = None,
) -> JournalResult:
    run_date = run_date or date.today()
    root = Path(journal_root)
    snapshots_dir = root / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshots_dir / f"{run_date.isoformat()}_scan.csv"

    current = scan_summary.copy()
    if not current.empty:
        current["journal_date"] = run_date.isoformat()
    previous_path = _latest_previous_snapshot(snapshots_dir, snapshot_path)
    previous = pd.read_csv(previous_path) if previous_path is not None else pd.DataFrame()
    changes = compare_scan_snapshots(current, previous)

    current.to_csv(snapshot_path, index=False)
    changes.to_csv(root / "latest_changes.csv", index=False)
    report_path = root / "daily_journal.md"
    report_path.write_text(render_daily_journal(current, previous, changes, run_date), encoding="utf-8")
    write_json(
        root / "journal_result.json",
        {
            "journal_date": run_date.isoformat(),
            "journal_dir": str(root),
            "snapshot_path": str(snapshot_path),
            "report_path": str(report_path),
            "current": dataframe_records(current),
            "previous": dataframe_records(previous),
            "changes": dataframe_records(changes),
            "output_files": {
                "markdown_report": str(report_path),
                "snapshot_csv": str(snapshot_path),
                "changes_csv": str(root / "latest_changes.csv"),
                "json": str(root / "journal_result.json"),
            },
        },
    )
    return JournalResult(
        journal_dir=root,
        snapshot_path=snapshot_path,
        report_path=report_path,
        current=current,
        previous=previous,
        changes=changes,
    )


def compare_scan_snapshots(current: pd.DataFrame, previous: pd.DataFrame) -> pd.DataFrame:
    if current.empty:
        return pd.DataFrame(columns=_change_columns())

    previous_lookup = (
        previous.set_index("ticker")
        if not previous.empty and "ticker" in previous.columns
        else pd.DataFrame()
    )
    rows: list[dict[str, object]] = []
    for row in current.itertuples(index=False):
        ticker = str(row.ticker)
        current_status = str(row.watchlist_status)
        current_score = float(row.high_probability_score)
        previous_status = "new"
        previous_score = float("nan")
        if not previous_lookup.empty and ticker in previous_lookup.index:
            previous_row = previous_lookup.loc[ticker]
            previous_status = str(previous_row.get("watchlist_status", "unknown"))
            previous_score = _safe_float(previous_row.get("high_probability_score"))

        change_type = _change_type(previous_status, current_status, previous_score, current_score)
        rows.append(
            {
                "ticker": ticker,
                "change_type": change_type,
                "previous_status": previous_status,
                "current_status": current_status,
                "previous_score": previous_score,
                "current_score": current_score,
                "score_change": current_score - previous_score
                if pd.notna(previous_score)
                else float("nan"),
                "quality_gate_passed": bool(row.quality_gate_passed),
                "watchlist_trigger_price": _safe_float(row.watchlist_trigger_price),
                "missing_items_zh": row.watchlist_missing_items_zh,
                "report_path": row.report_path,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["change_type", "current_score"],
        ascending=[True, False],
    ).reset_index(drop=True)


def render_daily_journal(
    current: pd.DataFrame,
    previous: pd.DataFrame,
    changes: pd.DataFrame,
    run_date: date,
) -> str:
    lines = [
        "# Daily Scan Journal / 每日扫描复盘",
        "",
        f"- Journal date / 复盘日期: `{run_date.isoformat()}`",
        f"- Current rows / 今日股票数: `{len(current)}`",
        f"- Previous rows / 上次股票数: `{len(previous)}`",
        "",
        "## Upgrades / 升级",
        "",
    ]
    lines.extend(_change_section(changes, {"new", "upgraded", "score_improved"}))
    lines.extend(["", "## Downgrades / 降级", ""])
    lines.extend(_change_section(changes, {"downgraded", "score_weakened"}))
    lines.extend(["", "## Stable / 稳定", ""])
    lines.extend(_change_section(changes, {"unchanged"}))
    return "\n".join(lines).rstrip() + "\n"


def _change_section(changes: pd.DataFrame, change_types: set[str]) -> list[str]:
    if changes.empty:
        return ["No rows. / 暂无。"]
    frame = changes[changes["change_type"].isin(change_types)]
    if frame.empty:
        return ["No rows. / 暂无。"]
    return _markdown_table(
        frame[
            [
                "ticker",
                "change_type",
                "previous_status",
                "current_status",
                "previous_score",
                "current_score",
                "score_change",
                "watchlist_trigger_price",
                "missing_items_zh",
            ]
        ]
    )


def _latest_previous_snapshot(snapshots_dir: Path, current_path: Path) -> Path | None:
    snapshots = sorted(path for path in snapshots_dir.glob("*_scan.csv") if path != current_path)
    return snapshots[-1] if snapshots else None


def _change_type(
    previous_status: str,
    current_status: str,
    previous_score: float,
    current_score: float,
) -> str:
    if previous_status == "new":
        return "new"
    previous_rank = STATUS_RANK.get(previous_status, 0)
    current_rank = STATUS_RANK.get(current_status, 0)
    if current_rank > previous_rank:
        return "upgraded"
    if current_rank < previous_rank:
        return "downgraded"
    if pd.notna(previous_score) and current_score - previous_score >= 5:
        return "score_improved"
    if pd.notna(previous_score) and current_score - previous_score <= -5:
        return "score_weakened"
    return "unchanged"


def _change_columns() -> list[str]:
    return [
        "ticker",
        "change_type",
        "previous_status",
        "current_status",
        "previous_score",
        "current_score",
        "score_change",
        "quality_gate_passed",
        "watchlist_trigger_price",
        "missing_items_zh",
        "report_path",
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


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _format_number(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)
