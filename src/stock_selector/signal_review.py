from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .json_io import dataframe_records, write_json


FORWARD_WINDOWS = (5, 20, 60)


@dataclass(frozen=True)
class SignalReviewResult:
    review_dir: Path
    history_path: Path
    report_path: Path
    history: pd.DataFrame
    ticker_history: pd.DataFrame
    summary: pd.DataFrame


@dataclass(frozen=True)
class SignalReviewDueScanResult:
    review_root: Path
    output_dir: Path
    due_items: pd.DataFrame
    summary: pd.DataFrame
    report_path: Path
    csv_path: Path
    json_path: Path


@dataclass(frozen=True)
class SignalReviewFeedbackContext:
    status: str
    score: float
    adjustment: float
    sample_count: int
    focus_window: str
    win_rate: float | None
    avg_return: float | None
    median_return: float | None
    level: str
    level_zh: str
    note: str
    note_zh: str

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "score": self.score,
            "adjustment": self.adjustment,
            "sample_count": self.sample_count,
            "focus_window": self.focus_window,
            "win_rate": self.win_rate,
            "avg_return": self.avg_return,
            "median_return": self.median_return,
            "level": self.level,
            "level_zh": self.level_zh,
            "note": self.note,
            "note_zh": self.note_zh,
        }


NEUTRAL_SIGNAL_REVIEW_FEEDBACK = SignalReviewFeedbackContext(
    status="insufficient_history",
    score=50.0,
    adjustment=0.0,
    sample_count=0,
    focus_window="none",
    win_rate=None,
    avg_return=None,
    median_return=None,
    level="insufficient_history",
    level_zh="历史样本不足",
    note="Not enough completed prior signals are available for feedback scoring.",
    note_zh="已完成的历史信号样本不足，暂不用于反向调分。",
)


def build_signal_review_feedback_context(
    ticker: str,
    review_root: str | Path = "outputs/signal_review",
) -> SignalReviewFeedbackContext:
    ticker = ticker.upper().strip()
    if not ticker:
        return NEUTRAL_SIGNAL_REVIEW_FEEDBACK
    history = _read_history(Path(review_root) / "signal_history.csv")
    if history.empty or "ticker" not in history.columns:
        return NEUTRAL_SIGNAL_REVIEW_FEEDBACK
    ticker_history = history[history["ticker"].astype(str).str.upper() == ticker].copy()
    if ticker_history.empty:
        return NEUTRAL_SIGNAL_REVIEW_FEEDBACK
    summary = summarize_signal_history(ticker_history)
    metrics = summarize_ticker_signal_review(ticker_history, summary)
    return _feedback_from_metrics(metrics)


def scan_signal_review_due_items(
    review_root: str | Path = "outputs/signal_review",
    output_dir: str | Path | None = None,
    as_of_date: str | pd.Timestamp | None = None,
) -> SignalReviewDueScanResult:
    review_path = Path(review_root)
    output_path = Path(output_dir) if output_dir is not None else review_path
    output_path.mkdir(parents=True, exist_ok=True)
    history_path = review_path / "signal_history.csv"
    history = _read_history(history_path)
    as_of = pd.Timestamp(as_of_date).normalize() if as_of_date is not None else pd.Timestamp.today().normalize()
    due_items = _build_due_items(history, as_of)
    summary = _build_due_summary(due_items, history)
    csv_path = output_path / "signal_review_due.csv"
    report_path = output_path / "signal_review_due.md"
    json_path = output_path / "signal_review_due.json"
    due_items.to_csv(csv_path, index=False)
    report_path.write_text(
        render_signal_review_due_report(due_items, summary, as_of),
        encoding="utf-8",
    )
    write_json(
        json_path,
        {
            "generated_at_utc": _utc_now(),
            "as_of_date": as_of.date().isoformat(),
            "review_root": review_path,
            "history_path": history_path,
            "due_items": dataframe_records(due_items),
            "summary": dataframe_records(summary),
            "output_files": {
                "due_csv": csv_path,
                "markdown_report": report_path,
                "json": json_path,
            },
        },
    )
    return SignalReviewDueScanResult(
        review_root=review_path,
        output_dir=output_path,
        due_items=due_items,
        summary=summary,
        report_path=report_path,
        csv_path=csv_path,
        json_path=json_path,
    )


def write_signal_review(
    analysis: pd.DataFrame,
    prices: pd.DataFrame,
    review_root: str | Path = "outputs/signal_review",
    source_report_path: str | Path | None = None,
) -> SignalReviewResult:
    review_dir = Path(review_root)
    review_dir.mkdir(parents=True, exist_ok=True)
    history_path = review_dir / "signal_history.csv"
    report_path = review_dir / "signal_review.md"

    existing = _read_history(history_path)
    current_signal = _build_current_signal_row(analysis, prices, source_report_path)
    combined = pd.concat([existing, pd.DataFrame([current_signal])], ignore_index=True)
    combined = _deduplicate_history(combined)
    combined = _update_forward_outcomes(combined, prices)
    combined = combined.sort_values(["ticker", "signal_date"]).reset_index(drop=True)

    ticker = str(current_signal["ticker"])
    ticker_history = combined[combined["ticker"] == ticker].copy().reset_index(drop=True)
    summary = summarize_signal_history(combined)

    combined.to_csv(history_path, index=False)
    ticker_history.to_csv(review_dir / f"{ticker}_signal_history.csv", index=False)
    summary.to_csv(review_dir / "signal_review_summary.csv", index=False)
    report_path.write_text(render_signal_review_report(ticker_history, summary), encoding="utf-8")
    write_json(
        review_dir / "signal_review_result.json",
        {
            "generated_at_utc": _utc_now(),
            "review_dir": review_dir,
            "history_path": history_path,
            "report_path": report_path,
            "current_ticker": ticker,
            "history": dataframe_records(combined),
            "ticker_history": dataframe_records(ticker_history),
            "summary": dataframe_records(summary),
            "output_files": {
                "history_csv": history_path,
                "ticker_history_csv": review_dir / f"{ticker}_signal_history.csv",
                "summary_csv": review_dir / "signal_review_summary.csv",
                "markdown_report": report_path,
                "json": review_dir / "signal_review_result.json",
            },
        },
    )
    return SignalReviewResult(
        review_dir=review_dir,
        history_path=history_path,
        report_path=report_path,
        history=combined,
        ticker_history=ticker_history,
        summary=summary,
    )


def summarize_signal_history(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(columns=_summary_columns())
    rows: list[dict[str, object]] = []
    for ticker, group in history.groupby("ticker", dropna=False):
        row: dict[str, object] = {
            "ticker": ticker,
            "signal_count": len(group),
        }
        for window in FORWARD_WINDOWS:
            column = f"forward_return_{window}d"
            status_column = f"outcome_status_{window}d"
            if column in group.columns:
                numeric_values = pd.to_numeric(group[column], errors="coerce")
                values = numeric_values.dropna()
            else:
                numeric_values = pd.Series([np.nan] * len(group), index=group.index, dtype=float)
                values = pd.Series(dtype=float)
            row[f"completed_{window}d_count"] = len(values)
            row[f"pending_{window}d_count"] = _pending_outcome_count(
                group=group,
                numeric_values=numeric_values,
                status_column=status_column,
            )
            row[f"min_trading_days_remaining_{window}d"] = _min_remaining_days(
                group,
                window,
                numeric_values,
            )
            row[f"next_estimated_review_date_{window}d"] = _next_estimated_review_date(
                group,
                window,
                numeric_values,
            )
            row[f"win_rate_{window}d"] = float((values > 0).mean()) if len(values) else np.nan
            row[f"avg_return_{window}d"] = float(values.mean()) if len(values) else np.nan
            row[f"median_return_{window}d"] = float(values.median()) if len(values) else np.nan
        row.update(_review_learning_state_from_summary_row(row))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)


def render_signal_review_report(ticker_history: pd.DataFrame, summary: pd.DataFrame) -> str:
    ticker = str(ticker_history["ticker"].iloc[-1]) if not ticker_history.empty else "N/A"
    lines = [
        "# Signal Review / 信号复盘",
        "",
        f"- Ticker / 股票: `{ticker}`",
        f"- Signal rows for ticker / 该股票信号记录数: `{len(ticker_history)}`",
        "",
        "## How To Read / 怎么看",
        "",
        "- `forward_return_5d`: close-to-close return after 5 trading days / 5个交易日后的收盘到收盘收益。",
        "- `forward_return_20d`: close-to-close return after 20 trading days / 20个交易日后的收盘到收盘收益。",
        "- `forward_return_60d`: close-to-close return after 60 trading days / 60个交易日后的收盘到收盘收益。",
        "- Blank values mean not enough future trading days yet / 空值代表未来交易日还不够。",
        "",
        "## Summary / 汇总",
        "",
    ]
    ticker_summary = summary[summary["ticker"] == ticker] if not summary.empty else pd.DataFrame()
    lines.extend(_markdown_table(ticker_summary if not ticker_summary.empty else summary))
    lines.extend(["", "## Recent Signals / 最近信号", ""])
    display_columns = [
        "signal_date",
        "ticker",
        "focus_horizon",
        "final_decision",
        "watchlist_status",
        "signal_price",
        "high_probability_score",
        "calibrated_win_probability",
        "forward_return_5d",
        "forward_return_20d",
        "forward_return_60d",
    ]
    display_history = ticker_history.copy()
    for column in display_columns:
        if column not in display_history.columns:
            display_history[column] = np.nan
    lines.extend(_markdown_table(display_history.tail(20)[display_columns] if not display_history.empty else display_history))
    return "\n".join(lines).rstrip() + "\n"


def render_signal_review_due_report(
    due_items: pd.DataFrame,
    summary: pd.DataFrame,
    as_of_date: str | pd.Timestamp,
) -> str:
    as_of = pd.Timestamp(as_of_date).date().isoformat()
    lines = [
        "# Signal Review Due Scan / 信号复盘到期扫描",
        "",
        f"- As of date / 截止日期: `{as_of}`",
        "",
        "## Summary / 汇总",
        "",
    ]
    lines.extend(_markdown_table(summary))
    lines.extend(["", "## Due Now / 已到复盘时间", ""])
    due_now = _due_items_by_status(due_items, "due_now")
    lines.extend(_markdown_table(_due_display_frame(due_now)))
    lines.extend(["", "## Pending / 等待中", ""])
    pending = due_items[due_items["review_status"].isin(["pending", "pending_unknown"])] if not due_items.empty else due_items
    lines.extend(_markdown_table(_due_display_frame(pending)))
    lines.extend(
        [
            "",
            "How to read / 怎么看:",
            "",
            "- `due_now`: the review window has arrived, but the signal still needs a refreshed ticker run to fill the outcome.",
            "- `due_now`: 已到复盘窗口，但还需要重新运行对应 ticker 才能填入结果。",
            "- `pending`: the review window has not arrived yet.",
            "- `pending`: 复盘窗口还没到。",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def summarize_ticker_signal_review(ticker_history: pd.DataFrame, summary: pd.DataFrame) -> dict[str, object]:
    ticker = str(ticker_history["ticker"].iloc[-1]).upper() if not ticker_history.empty else ""
    metrics: dict[str, object] = {
        "ticker_signal_count": int(len(ticker_history)),
    }
    ticker_summary = _ticker_summary_row(ticker, summary)
    for window in FORWARD_WINDOWS:
        metrics[f"completed_{window}d_count"] = _safe_int(
            ticker_summary.get(f"completed_{window}d_count") if ticker_summary is not None else None
        )
        metrics[f"pending_{window}d_count"] = _safe_int(
            ticker_summary.get(f"pending_{window}d_count") if ticker_summary is not None else None
        )
        metrics[f"min_trading_days_remaining_{window}d"] = _safe_int(
            ticker_summary.get(f"min_trading_days_remaining_{window}d")
            if ticker_summary is not None
            else None
        )
        metrics[f"next_estimated_review_date_{window}d"] = (
            ticker_summary.get(f"next_estimated_review_date_{window}d")
            if ticker_summary is not None
            else ""
        )
        metrics[f"win_rate_{window}d"] = _safe_optional_float(
            ticker_summary.get(f"win_rate_{window}d") if ticker_summary is not None else None
        )
        metrics[f"avg_return_{window}d"] = _safe_optional_float(
            ticker_summary.get(f"avg_return_{window}d") if ticker_summary is not None else None
        )
        metrics[f"median_return_{window}d"] = _safe_optional_float(
            ticker_summary.get(f"median_return_{window}d") if ticker_summary is not None else None
        )
    for key in _review_learning_state_columns():
        metrics[key] = ticker_summary.get(key) if ticker_summary is not None else _default_learning_state_value(key)
    return metrics


def render_signal_review_section(
    ticker_history: pd.DataFrame,
    summary: pd.DataFrame,
    report_path: str | Path | None = None,
) -> str:
    metrics = summarize_ticker_signal_review(ticker_history, summary)
    lines = [
        "## Signal Review / 信号复盘",
        "",
        "This section checks earlier signals for this ticker against later close-to-close returns.",
        "这一部分会把这个股票之前生成过的信号，和之后实际收盘到收盘收益进行对照。",
        "",
        f"- Signal records / 信号记录数: `{metrics['ticker_signal_count']}`",
        f"- Review learning score / 复盘学习分数: `{_format_number(metrics['review_learning_score'])}`",
        (
            f"- Review learning level / 复盘学习等级: `{metrics['review_learning_level']}` / "
            f"`{metrics['review_learning_level_zh']}`"
        ),
        (
            f"- Review learning focus / 复盘重点窗口: `{metrics['review_learning_focus_window']}`, "
            f"samples / 样本 `{metrics['review_learning_sample_count']}`, "
            f"adjustment / 调分 `{_format_signed_number(metrics['review_learning_adjustment'])}`"
        ),
        (
            f"- Review learning note / 复盘学习说明: {metrics['review_learning_note']} / "
            f"{metrics['review_learning_note_zh']}"
        ),
    ]
    for window in FORWARD_WINDOWS:
        lines.append(
            f"- {window} trading days / {window}个交易日: "
            f"completed samples / 已完成样本 `{metrics[f'completed_{window}d_count']}`, "
            f"pending samples / 等待中样本 `{metrics[f'pending_{window}d_count']}`, "
            f"next review estimate / 预计下次可复盘 `{metrics[f'next_estimated_review_date_{window}d'] or 'N/A'}`, "
            f"min remaining trading days / 最少剩余交易日 `{metrics[f'min_trading_days_remaining_{window}d']}`, "
            f"win rate / 胜率 `{_format_optional_percent(metrics[f'win_rate_{window}d'])}`, "
            f"average return / 平均收益 `{_format_optional_percent(metrics[f'avg_return_{window}d'])}`, "
            f"median return / 中位收益 `{_format_optional_percent(metrics[f'median_return_{window}d'])}`"
        )
    if report_path is not None:
        lines.append(f"- Full review report / 完整复盘报告: `{report_path}`")
    lines.extend(
        [
            "",
            "Blank or N/A values usually mean the model has not accumulated enough future trading days yet.",
            "空值或 N/A 通常代表未来交易日还不够，暂时不能评价那一档复盘结果。",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _feedback_from_metrics(metrics: dict[str, object]) -> SignalReviewFeedbackContext:
    sample_count = _safe_int(metrics.get("review_learning_sample_count"))
    if sample_count >= 3:
        return SignalReviewFeedbackContext(
            status="ok",
            score=round(float(_safe_optional_float(metrics.get("review_learning_score")) or 50.0), 2),
            adjustment=round(float(_safe_optional_float(metrics.get("review_learning_adjustment")) or 0.0), 2),
            sample_count=sample_count,
            focus_window=str(metrics.get("review_learning_focus_window") or "none"),
            win_rate=_safe_optional_float(metrics.get("review_learning_win_rate")),
            avg_return=_safe_optional_float(metrics.get("review_learning_avg_return")),
            median_return=_safe_optional_float(metrics.get("review_learning_median_return")),
            level=str(metrics.get("review_learning_level") or "neutral"),
            level_zh=str(metrics.get("review_learning_level_zh") or "历史复盘中性"),
            note=str(metrics.get("review_learning_note") or ""),
            note_zh=str(metrics.get("review_learning_note_zh") or ""),
        )

    candidates: list[dict[str, object]] = []
    for window, weight in [(5, 0.25), (20, 0.45), (60, 0.30)]:
        count = _safe_int(metrics.get(f"completed_{window}d_count"))
        win_rate = _safe_optional_float(metrics.get(f"win_rate_{window}d"))
        avg_return = _safe_optional_float(metrics.get(f"avg_return_{window}d"))
        median_return = _safe_optional_float(metrics.get(f"median_return_{window}d"))
        if count <= 0 or win_rate is None or avg_return is None:
            continue
        candidates.append(
            {
                "window": window,
                "weight": weight,
                "count": count,
                "win_rate": win_rate,
                "avg_return": avg_return,
                "median_return": median_return,
            }
        )
    completed_count = int(sum(int(item["count"]) for item in candidates))
    if completed_count < 3:
        return NEUTRAL_SIGNAL_REVIEW_FEEDBACK

    weighted_score = 0.0
    weight_sum = 0.0
    for item in candidates:
        sample_weight = min(float(item["count"]) / 10.0, 1.0)
        weight = float(item["weight"]) * sample_weight
        win_rate = float(item["win_rate"])
        avg_return = float(item["avg_return"])
        window_score = 50.0 + (win_rate - 0.50) * 90.0 + avg_return * 120.0
        weighted_score += max(0.0, min(window_score, 100.0)) * weight
        weight_sum += weight
    if weight_sum <= 0:
        return NEUTRAL_SIGNAL_REVIEW_FEEDBACK

    score = round(float(weighted_score / weight_sum), 2)
    focus = max(candidates, key=lambda item: (int(item["count"]), float(item["weight"])))
    sample_scale = min(completed_count / 20.0, 1.0)
    adjustment = round(float(max(-7.0, min((score - 50.0) * 0.22 * sample_scale, 7.0))), 2)
    level, level_zh = _signal_review_feedback_level(score, completed_count)
    note = (
        f"Prior signal feedback uses {completed_count} completed outcomes. "
        f"Focus window={focus['window']}d, win rate={_format_optional_percent(focus['win_rate'])}, "
        f"average return={_format_optional_percent(focus['avg_return'])}. "
        f"Score adjustment={adjustment:+.2f}."
    )
    note_zh = (
        f"历史信号反馈使用{completed_count}个已完成结果。"
        f"重点窗口={focus['window']}日，胜率={_format_optional_percent(focus['win_rate'])}，"
        f"平均收益={_format_optional_percent(focus['avg_return'])}。"
        f"分数调整={adjustment:+.2f}。"
    )
    return SignalReviewFeedbackContext(
        status="ok",
        score=score,
        adjustment=adjustment,
        sample_count=completed_count,
        focus_window=f"{focus['window']}d",
        win_rate=_safe_optional_float(focus["win_rate"]),
        avg_return=_safe_optional_float(focus["avg_return"]),
        median_return=_safe_optional_float(focus["median_return"]),
        level=level,
        level_zh=level_zh,
        note=note,
        note_zh=note_zh,
    )


def _build_due_items(history: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    columns = [
        "ticker",
        "signal_date",
        "focus_horizon",
        "review_window",
        "review_status",
        "review_status_zh",
        "estimated_review_date",
        "trading_days_remaining",
        "forward_return",
        "outcome_status",
        "final_decision",
        "watchlist_status",
        "high_probability_score",
        "calibrated_win_probability",
        "primary_blocker",
        "recommended_command",
        "report_path",
    ]
    if history.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for signal in history.itertuples(index=False):
        ticker = str(getattr(signal, "ticker", "")).upper().strip()
        if not ticker:
            continue
        for window in FORWARD_WINDOWS:
            forward_return = _safe_optional_float(getattr(signal, f"forward_return_{window}d", None))
            if forward_return is not None:
                continue
            status, status_zh = _due_status_for_signal(signal, window, as_of)
            rows.append(
                {
                    "ticker": ticker,
                    "signal_date": getattr(signal, "signal_date", ""),
                    "focus_horizon": getattr(signal, "focus_horizon", ""),
                    "review_window": f"{window}d",
                    "review_status": status,
                    "review_status_zh": status_zh,
                    "estimated_review_date": getattr(signal, f"estimated_review_date_{window}d", ""),
                    "trading_days_remaining": _safe_optional_float(
                        getattr(signal, f"trading_days_remaining_{window}d", None)
                    ),
                    "forward_return": forward_return,
                    "outcome_status": getattr(signal, f"outcome_status_{window}d", ""),
                    "final_decision": getattr(signal, "final_decision", ""),
                    "watchlist_status": getattr(signal, "watchlist_status", ""),
                    "high_probability_score": _safe_optional_float(
                        getattr(signal, "high_probability_score", None)
                    ),
                    "calibrated_win_probability": _safe_optional_float(
                        getattr(signal, "calibrated_win_probability", None)
                    ),
                    "primary_blocker": getattr(signal, "primary_blocker", ""),
                    "recommended_command": f"python3 run.py {ticker} --no-snapshot --no-peers",
                    "report_path": getattr(signal, "report_path", ""),
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows)
    status_order = {"due_now": 0, "pending": 1, "pending_unknown": 2}
    frame["_status_order"] = frame["review_status"].map(status_order).fillna(9)
    frame["_estimated_date_sort"] = pd.to_datetime(frame["estimated_review_date"], errors="coerce")
    frame["_remaining_sort"] = pd.to_numeric(frame["trading_days_remaining"], errors="coerce")
    frame = frame.sort_values(
        ["_status_order", "_estimated_date_sort", "_remaining_sort", "ticker", "review_window"],
        na_position="last",
    )
    return frame[columns].reset_index(drop=True)


def _due_status_for_signal(signal: object, window: int, as_of: pd.Timestamp) -> tuple[str, str]:
    remaining = _safe_optional_float(getattr(signal, f"trading_days_remaining_{window}d", None))
    estimated_date = pd.to_datetime(
        getattr(signal, f"estimated_review_date_{window}d", ""),
        errors="coerce",
    )
    if remaining is not None and remaining <= 0:
        return "due_now", "已到复盘时间"
    if not pd.isna(estimated_date) and estimated_date.normalize() <= as_of:
        return "due_now", "已到复盘时间"
    if remaining is not None or not pd.isna(estimated_date):
        return "pending", "等待中"
    return "pending_unknown", "等待信息不足"


def _build_due_summary(due_items: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    signal_count = int(len(history)) if not history.empty else 0
    completed = 0
    if not history.empty:
        for window in FORWARD_WINDOWS:
            column = f"forward_return_{window}d"
            if column in history.columns:
                completed += int(pd.to_numeric(history[column], errors="coerce").notna().sum())
    if due_items.empty:
        return pd.DataFrame(
            [
                {
                    "signal_count": signal_count,
                    "due_now_count": 0,
                    "pending_count": 0,
                    "pending_unknown_count": 0,
                    "completed_outcome_count": completed,
                    "next_due_date": "",
                    "next_due_ticker": "",
                }
            ]
        )
    due_now = _due_items_by_status(due_items, "due_now")
    next_due = due_items.copy()
    next_due["_estimated"] = pd.to_datetime(next_due["estimated_review_date"], errors="coerce")
    next_due = next_due.dropna(subset=["_estimated"]).sort_values("_estimated")
    first_due = next_due.iloc[0] if not next_due.empty else None
    return pd.DataFrame(
        [
            {
                "signal_count": signal_count,
                "due_now_count": int(len(due_now)),
                "pending_count": int((due_items["review_status"] == "pending").sum()),
                "pending_unknown_count": int((due_items["review_status"] == "pending_unknown").sum()),
                "completed_outcome_count": completed,
                "next_due_date": first_due["estimated_review_date"] if first_due is not None else "",
                "next_due_ticker": first_due["ticker"] if first_due is not None else "",
            }
        ]
    )


def _due_items_by_status(due_items: pd.DataFrame, status: str) -> pd.DataFrame:
    if due_items.empty or "review_status" not in due_items.columns:
        return due_items
    return due_items[due_items["review_status"] == status].copy()


def _due_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "ticker",
        "signal_date",
        "review_window",
        "review_status",
        "review_status_zh",
        "estimated_review_date",
        "trading_days_remaining",
        "focus_horizon",
        "high_probability_score",
        "calibrated_win_probability",
        "recommended_command",
        "report_path",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    display = frame.copy()
    for column in columns:
        if column not in display.columns:
            display[column] = np.nan
    return display[columns]


def _build_current_signal_row(
    analysis: pd.DataFrame,
    prices: pd.DataFrame,
    source_report_path: str | Path | None,
) -> dict[str, object]:
    if analysis.empty:
        raise ValueError("analysis cannot be empty")
    if prices.empty:
        raise ValueError("prices cannot be empty")
    focus = analysis.sort_values("high_probability_score", ascending=False).iloc[0]
    normalized_prices = _normal_price_frame(prices)
    latest = normalized_prices.iloc[-1]
    signal_date = pd.Timestamp(latest["date"]).date().isoformat()
    signal_price = _safe_float(focus.get("latest_price"))
    if not np.isfinite(signal_price) or signal_price <= 0:
        signal_price = float(latest["adj_close"])
    return {
        "created_at_utc": _utc_now(),
        "signal_date": signal_date,
        "ticker": str(focus.get("ticker") or latest["ticker"]).upper().strip(),
        "focus_horizon": focus.get("final_focus_horizon", focus.get("horizon", "")),
        "final_decision": focus.get("final_decision", ""),
        "final_decision_zh": focus.get("final_decision_zh", ""),
        "watchlist_status": focus.get("calibrated_watchlist_status", focus.get("watchlist_status", "")),
        "screening_action": focus.get("calibrated_screening_action", focus.get("screening_action", "")),
        "quality_gate_passed": bool(focus.get("calibrated_quality_gate_passed", focus.get("quality_gate_passed", False))),
        "signal_price": round(float(signal_price), 6),
        "high_probability_score": _safe_float(focus.get("calibrated_high_probability_score", focus.get("high_probability_score"))),
        "signal_score": _safe_float(focus.get("signal_score")),
        "calibrated_win_probability": _safe_float(focus.get("calibrated_win_probability")),
        "primary_blocker": focus.get("primary_blocker", ""),
        "quality_gate_fail_reasons_zh": focus.get("calibrated_quality_gate_fail_reasons_zh", focus.get("quality_gate_fail_reasons_zh", "")),
        "report_path": str(source_report_path or ""),
    }


def _update_forward_outcomes(history: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    if history.empty or prices.empty:
        return history
    updated = history.copy()
    normalized_prices = _normal_price_frame(prices)
    ticker = str(normalized_prices["ticker"].iloc[-1])
    ticker_prices = normalized_prices[normalized_prices["ticker"] == ticker].reset_index(drop=True)
    if ticker_prices.empty:
        return updated

    date_values = pd.to_datetime(ticker_prices["date"]).dt.normalize()
    close_values = pd.to_numeric(ticker_prices["adj_close"], errors="coerce")
    for index, row in updated[updated["ticker"] == ticker].iterrows():
        signal_date = pd.to_datetime(row.get("signal_date"), errors="coerce")
        if pd.isna(signal_date):
            continue
        eligible = date_values[date_values <= signal_date.normalize()]
        if eligible.empty:
            continue
        signal_index = int(eligible.index[-1])
        signal_price = _safe_float(row.get("signal_price"))
        if not np.isfinite(signal_price) or signal_price <= 0:
            signal_price = float(close_values.iloc[signal_index])
        for window in FORWARD_WINDOWS:
            target_index = signal_index + window
            return_column = f"forward_return_{window}d"
            date_column = f"outcome_date_{window}d"
            status_column = f"outcome_status_{window}d"
            remaining_column = f"trading_days_remaining_{window}d"
            due_column = f"estimated_review_date_{window}d"
            if target_index >= len(ticker_prices):
                remaining_days = int(target_index - (len(ticker_prices) - 1))
                updated.at[index, status_column] = "pending"
                updated.at[index, remaining_column] = remaining_days
                updated.at[index, due_column] = _estimated_business_date(
                    ticker_prices.loc[len(ticker_prices) - 1, "date"],
                    remaining_days,
                )
                continue
            target_price = float(close_values.iloc[target_index])
            forward_return = target_price / signal_price - 1.0
            updated.at[index, return_column] = round(float(forward_return), 6)
            updated.at[index, date_column] = pd.Timestamp(ticker_prices.loc[target_index, "date"]).date().isoformat()
            updated.at[index, status_column] = "win" if forward_return > 0 else "loss"
            updated.at[index, remaining_column] = 0
            updated.at[index, due_column] = pd.Timestamp(ticker_prices.loc[target_index, "date"]).date().isoformat()
    return updated


def _deduplicate_history(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return history
    required = ["ticker", "signal_date", "focus_horizon"]
    for column in required:
        if column not in history.columns:
            history[column] = ""
    return history.drop_duplicates(required, keep="last").reset_index(drop=True)


def _read_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _normal_price_frame(prices: pd.DataFrame) -> pd.DataFrame:
    frame = prices.copy()
    frame.columns = [str(column).lower() for column in frame.columns]
    if "adj_close" not in frame.columns and "close" in frame.columns:
        frame["adj_close"] = frame["close"]
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    frame["adj_close"] = pd.to_numeric(frame["adj_close"], errors="coerce")
    frame = frame.dropna(subset=["date", "ticker", "adj_close"])
    return frame.sort_values(["ticker", "date"]).reset_index(drop=True)


def _summary_columns() -> list[str]:
    columns = ["ticker", "signal_count"]
    for window in FORWARD_WINDOWS:
        columns.extend(
            [
                f"completed_{window}d_count",
                f"pending_{window}d_count",
                f"min_trading_days_remaining_{window}d",
                f"next_estimated_review_date_{window}d",
                f"win_rate_{window}d",
                f"avg_return_{window}d",
                f"median_return_{window}d",
            ]
        )
    columns.extend(_review_learning_state_columns())
    return columns


def _review_learning_state_columns() -> list[str]:
    return [
        "review_learning_score",
        "review_learning_adjustment",
        "review_learning_sample_count",
        "review_learning_focus_window",
        "review_learning_win_rate",
        "review_learning_avg_return",
        "review_learning_median_return",
        "review_learning_level",
        "review_learning_level_zh",
        "review_learning_note",
        "review_learning_note_zh",
    ]


def _review_learning_state_from_summary_row(row: dict[str, object]) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    for window, weight in [(5, 0.25), (20, 0.45), (60, 0.30)]:
        count = _safe_int(row.get(f"completed_{window}d_count"))
        win_rate = _safe_optional_float(row.get(f"win_rate_{window}d"))
        avg_return = _safe_optional_float(row.get(f"avg_return_{window}d"))
        median_return = _safe_optional_float(row.get(f"median_return_{window}d"))
        if count <= 0 or win_rate is None or avg_return is None:
            continue
        candidates.append(
            {
                "window": window,
                "weight": weight,
                "count": count,
                "win_rate": win_rate,
                "avg_return": avg_return,
                "median_return": median_return,
            }
        )
    completed_count = int(sum(int(item["count"]) for item in candidates))
    if completed_count < 3:
        return _neutral_review_learning_state(completed_count)

    weighted_score = 0.0
    weight_sum = 0.0
    for item in candidates:
        sample_weight = min(float(item["count"]) / 10.0, 1.0)
        weight = float(item["weight"]) * sample_weight
        win_rate = float(item["win_rate"])
        avg_return = float(item["avg_return"])
        window_score = 50.0 + (win_rate - 0.50) * 90.0 + avg_return * 120.0
        weighted_score += max(0.0, min(window_score, 100.0)) * weight
        weight_sum += weight
    if weight_sum <= 0:
        return _neutral_review_learning_state(completed_count)

    score = round(float(weighted_score / weight_sum), 2)
    focus = max(candidates, key=lambda item: (int(item["count"]), float(item["weight"])))
    sample_scale = min(completed_count / 20.0, 1.0)
    adjustment = round(float(max(-7.0, min((score - 50.0) * 0.22 * sample_scale, 7.0))), 2)
    level, level_zh = _signal_review_feedback_level(score, completed_count)
    note = (
        f"Learning state uses {completed_count} completed outcomes. "
        f"Focus window={focus['window']}d, win rate={_format_optional_percent(focus['win_rate'])}, "
        f"average return={_format_optional_percent(focus['avg_return'])}."
    )
    note_zh = (
        f"复盘学习状态使用{completed_count}个已完成结果。"
        f"重点窗口={focus['window']}日，胜率={_format_optional_percent(focus['win_rate'])}，"
        f"平均收益={_format_optional_percent(focus['avg_return'])}。"
    )
    return {
        "review_learning_score": score,
        "review_learning_adjustment": adjustment,
        "review_learning_sample_count": completed_count,
        "review_learning_focus_window": f"{focus['window']}d",
        "review_learning_win_rate": _safe_optional_float(focus["win_rate"]),
        "review_learning_avg_return": _safe_optional_float(focus["avg_return"]),
        "review_learning_median_return": _safe_optional_float(focus["median_return"]),
        "review_learning_level": level,
        "review_learning_level_zh": level_zh,
        "review_learning_note": note,
        "review_learning_note_zh": note_zh,
    }


def _neutral_review_learning_state(sample_count: int = 0) -> dict[str, object]:
    return {
        "review_learning_score": 50.0,
        "review_learning_adjustment": 0.0,
        "review_learning_sample_count": sample_count,
        "review_learning_focus_window": "none",
        "review_learning_win_rate": np.nan,
        "review_learning_avg_return": np.nan,
        "review_learning_median_return": np.nan,
        "review_learning_level": "insufficient_history",
        "review_learning_level_zh": "历史样本不足",
        "review_learning_note": "Not enough completed prior signals are available for learning-state scoring.",
        "review_learning_note_zh": "已完成的历史信号样本不足，暂不能形成复盘学习状态。",
    }


def _default_learning_state_value(key: str) -> object:
    return _neutral_review_learning_state().get(key, np.nan)


def _pending_outcome_count(
    group: pd.DataFrame,
    numeric_values: pd.Series,
    status_column: str,
) -> int:
    missing_count = int(numeric_values.isna().sum())
    if status_column not in group.columns:
        return missing_count
    statuses = group[status_column].astype(str).str.lower().str.strip()
    pending_count = int((statuses == "pending").sum())
    return max(missing_count, pending_count)


def _min_remaining_days(
    group: pd.DataFrame,
    window: int,
    numeric_values: pd.Series,
) -> float:
    pending_mask = numeric_values.isna()
    if not pending_mask.any():
        return np.nan
    column = f"trading_days_remaining_{window}d"
    if column not in group.columns:
        return np.nan
    values = pd.to_numeric(group.loc[pending_mask, column], errors="coerce").dropna()
    if values.empty:
        return np.nan
    return int(values.min())


def _next_estimated_review_date(
    group: pd.DataFrame,
    window: int,
    numeric_values: pd.Series,
) -> str:
    pending_mask = numeric_values.isna()
    if not pending_mask.any():
        return ""
    column = f"estimated_review_date_{window}d"
    if column not in group.columns:
        return ""
    values = pd.to_datetime(group.loc[pending_mask, column], errors="coerce").dropna()
    if values.empty:
        return ""
    return pd.Timestamp(values.min()).date().isoformat()


def _ticker_summary_row(ticker: str, summary: pd.DataFrame) -> dict[str, object] | None:
    if not ticker or summary.empty or "ticker" not in summary.columns:
        return None
    matches = summary[summary["ticker"].astype(str).str.upper() == ticker]
    if matches.empty:
        return None
    return matches.iloc[-1].to_dict()


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["No rows. / 暂无。"]
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False):
        values = [
            _format_value_for_column(column, value)
            for column, value in zip(columns, row)
        ]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _format_value_for_column(column: str, value: object) -> str:
    if pd.isna(value):
        return ""
    if column.endswith("_score") or column.endswith("_adjustment"):
        numeric = _safe_optional_float(value)
        return f"{numeric:.2f}" if numeric is not None else str(value)
    if (
        column.endswith("_rate")
        or column.endswith("_return")
        or "_return_" in column
        or column.endswith("_probability")
    ):
        numeric = _safe_optional_float(value)
        return f"{numeric:.2%}" if numeric is not None else str(value)
    return _format_value(value)


def _format_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.2%}" if -1.0 <= value <= 1.0 else f"{value:.2f}"
    return str(value)


def _format_optional_percent(value: object) -> str:
    numeric = _safe_optional_float(value)
    if numeric is None:
        return "N/A"
    return f"{numeric:.2%}"


def _format_number(value: object) -> str:
    numeric = _safe_optional_float(value)
    if numeric is None:
        return "N/A"
    return f"{numeric:.2f}"


def _format_signed_number(value: object) -> str:
    numeric = _safe_optional_float(value)
    if numeric is None:
        return "N/A"
    return f"{numeric:+.2f}"


def _signal_review_feedback_level(score: float, sample_count: int) -> tuple[str, str]:
    if sample_count < 3:
        return "insufficient_history", "历史样本不足"
    if score >= 65:
        return "supportive", "历史复盘支持"
    if score >= 55:
        return "slightly_supportive", "历史复盘略支持"
    if score >= 45:
        return "neutral", "历史复盘中性"
    if score >= 35:
        return "weak", "历史复盘偏弱"
    return "poor", "历史复盘较差"


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _safe_optional_float(value: object) -> float | None:
    numeric = _safe_float(value)
    if not np.isfinite(numeric):
        return None
    return float(numeric)


def _safe_int(value: object) -> int:
    numeric = _safe_float(value)
    if not np.isfinite(numeric):
        return 0
    return int(numeric)


def _estimated_business_date(start_date: object, remaining_days: int) -> str:
    if remaining_days <= 0:
        return pd.Timestamp(start_date).date().isoformat()
    estimated = pd.Timestamp(start_date) + pd.tseries.offsets.BDay(remaining_days)
    return estimated.date().isoformat()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
