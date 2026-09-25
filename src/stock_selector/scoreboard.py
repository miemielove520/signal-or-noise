"""Scoreboard analytics for the human-vs-model experiment.

Win rate alone is a misleading scorecard: a high win rate with a poor payoff
ratio still loses money, and a modest win rate with big winners still compounds.
This module computes expectancy, payoff ratio, profit factor, and a cumulative
drawdown so the scoreboard judges *quality of decisions*, not just direction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .human_log import DEFAULT_BENCHMARK, FORWARD_WINDOWS, fill_human_trade_outcomes


def compute_outcome_metrics(
    returns: pd.Series | list[float],
    order: pd.Series | list | None = None,
) -> dict[str, object]:
    """Per-trade metrics for a set of (excess) returns.

    ``returns`` are per-trade returns — excess-vs-benchmark for the scoreboard.
    ``order`` optionally orders trades in time before the drawdown is computed
    (drawdown is only meaningful on a time-ordered equity curve).
    """
    series = pd.to_numeric(pd.Series(list(returns)), errors="coerce")
    if order is not None:
        order_series = pd.Series(list(order)).reset_index(drop=True)
        series = series.reset_index(drop=True)
        sort_index = pd.to_datetime(order_series, errors="coerce").sort_values(
            kind="stable"
        ).index
        series = series.iloc[sort_index]
    series = series.dropna().reset_index(drop=True)

    count = int(len(series))
    if count == 0:
        return _empty_metrics()

    wins = series[series > 0]
    losses = series[series < 0]
    win_count = int(len(wins))
    loss_count = int(len(losses))

    expectancy = float(series.mean())
    avg_win = float(wins.mean()) if win_count else 0.0
    avg_loss = float(losses.mean()) if loss_count else 0.0  # negative or 0
    beat_rate = win_count / count

    payoff_ratio = float(avg_win / abs(avg_loss)) if avg_loss != 0.0 else np.nan
    gross_profit = float(wins.sum())
    gross_loss = float(abs(losses.sum()))
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else np.nan

    std = float(series.std(ddof=0))
    consistency = float(expectancy / std) if std > 0 else np.nan

    # Equity starts at a 0 baseline so an immediate losing streak counts as
    # drawdown from the start, rather than treating the first loss as the peak.
    equity = np.concatenate([[0.0], series.cumsum().to_numpy()])
    running_max = np.maximum.accumulate(equity)
    drawdown = equity - running_max
    max_drawdown = float(drawdown.min())

    return {
        "completed_count": count,
        "beat_count": win_count,
        "lag_count": loss_count,
        "beat_rate": beat_rate,
        "expectancy": expectancy,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": payoff_ratio,
        "profit_factor": profit_factor,
        "consistency": consistency,
        "total_excess": float(series.sum()),
        "max_drawdown": max_drawdown,
    }


def summarize_outcomes_by_window(
    frame: pd.DataFrame,
    return_col_template: str,
    date_col_template: str | None = None,
    windows: tuple[int, ...] = FORWARD_WINDOWS,
    pending_status_template: str | None = None,
) -> pd.DataFrame:
    """One metrics row per forward window for a single actor's outcome frame."""
    rows: list[dict[str, object]] = []
    for window in windows:
        return_col = return_col_template.format(window=window)
        row: dict[str, object] = {"window": f"{window}d"}
        if frame is None or frame.empty or return_col not in frame.columns:
            row.update(_empty_metrics())
            row["pending_count"] = 0
            rows.append(row)
            continue
        returns = frame[return_col]
        order = None
        if date_col_template is not None:
            date_col = date_col_template.format(window=window)
            if date_col in frame.columns:
                order = frame[date_col]
        metrics = compute_outcome_metrics(returns, order=order)
        metrics["pending_count"] = _pending_count(
            frame, return_col, pending_status_template, window
        )
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_human_scoreboard(trades: pd.DataFrame) -> pd.DataFrame:
    """Per-window expectancy/drawdown scoreboard for logged human trades.

    Scored on excess return vs the benchmark, so "winning" means beating the
    market, not merely finishing positive.
    """
    return summarize_outcomes_by_window(
        trades,
        return_col_template="excess_return_{window}d",
        date_col_template="outcome_date_{window}d",
        pending_status_template="outcome_status_{window}d",
    )


def prepare_model_outcomes(
    model_signals: pd.DataFrame,
    price_lookup: dict[str, pd.DataFrame],
    benchmark_lookup: dict[str, pd.DataFrame],
    benchmark: str = DEFAULT_BENCHMARK,
) -> pd.DataFrame:
    """Compute excess-vs-benchmark outcomes for model signals.

    Reuses the exact same filling primitive as human trades so both actors are
    scored identically. Model signals lack a benchmark column, so one is applied.
    """
    if model_signals is None or model_signals.empty:
        return model_signals if model_signals is not None else pd.DataFrame()
    prepared = model_signals.copy()
    if "benchmark" not in prepared.columns:
        prepared["benchmark"] = str(benchmark).upper().strip()
    else:
        prepared["benchmark"] = prepared["benchmark"].fillna(str(benchmark).upper().strip())
    return fill_human_trade_outcomes(prepared, price_lookup, benchmark_lookup)


def summarize_actor_excess(filled: pd.DataFrame) -> pd.DataFrame:
    """Per-window excess-return scoreboard for one actor's filled outcomes."""
    return summarize_outcomes_by_window(
        filled,
        return_col_template="excess_return_{window}d",
        date_col_template="outcome_date_{window}d",
        pending_status_template="outcome_status_{window}d",
    )


def build_head_to_head(human_filled: pd.DataFrame, model_filled: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side human-vs-model comparison, one row per forward window.

    The per-window leader is decided by expectancy (average excess return per
    decision), and only when both actors have at least one completed outcome.
    """
    human = summarize_actor_excess(human_filled).add_prefix("human_")
    model = summarize_actor_excess(model_filled).add_prefix("model_")
    human = human.rename(columns={"human_window": "window"})
    model = model.rename(columns={"model_window": "window"})
    merged = human.merge(model, on="window", how="outer")

    rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        human_n = int(row.get("human_completed_count") or 0)
        model_n = int(row.get("model_completed_count") or 0)
        human_exp = row.get("human_expectancy")
        model_exp = row.get("model_expectancy")
        leader, edge = _decide_leader(human_n, model_n, human_exp, model_exp)
        rows.append(
            {
                "window": row["window"],
                "human_completed": human_n,
                "human_expectancy": human_exp,
                "human_beat_rate": row.get("human_beat_rate"),
                "human_profit_factor": row.get("human_profit_factor"),
                "human_max_drawdown": row.get("human_max_drawdown"),
                "model_completed": model_n,
                "model_expectancy": model_exp,
                "model_beat_rate": row.get("model_beat_rate"),
                "model_profit_factor": row.get("model_profit_factor"),
                "model_max_drawdown": row.get("model_max_drawdown"),
                "expectancy_edge": edge,
                "leader": leader,
            }
        )
    order = {f"{window}d": index for index, window in enumerate(FORWARD_WINDOWS)}
    result = pd.DataFrame(rows)
    result["_order"] = result["window"].map(order).fillna(99)
    return result.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def render_head_to_head_markdown(head_to_head: pd.DataFrame) -> str:
    """Human-readable bilingual head-to-head report."""
    lines = [
        "# Human vs Model Scoreboard / 人机对照记分台",
        "",
        "Scored on excess return vs the benchmark (QQQ/SPY). \"Winning\" means beating the market.",
        "按相对基准（QQQ/SPY）的超额收益打分。“赢”指跑赢大盘，而不是只要正收益。",
        "",
        "- `expectancy`: average excess return per decision / 每次决策的平均超额收益（越高越好）。",
        "- `beat_rate`: share of decisions that beat the benchmark / 跑赢基准的比例。",
        "- `profit_factor`: gross excess win / gross excess loss / 超额盈亏比（>1 才划算）。",
        "- `max_drawdown`: worst cumulative excess drawdown / 最差累计超额回撤。",
        "- `leader`: higher expectancy, only when both sides have samples / 期望值更高者，需两边都有样本。",
        "",
    ]
    if head_to_head is None or head_to_head.empty:
        lines.append("No data yet. / 暂无数据。")
        return "\n".join(lines) + "\n"

    headline = _headline_verdict(head_to_head)
    lines.extend([f"**Verdict / 结论: {headline}**", ""])

    columns = [
        "window",
        "human_completed",
        "human_expectancy",
        "human_beat_rate",
        "model_completed",
        "model_expectancy",
        "model_beat_rate",
        "expectancy_edge",
        "leader",
    ]
    percent_columns = {
        "human_expectancy",
        "human_beat_rate",
        "model_expectancy",
        "model_beat_rate",
        "expectancy_edge",
    }
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for _, row in head_to_head.iterrows():
        cells = []
        for column in columns:
            value = row.get(column)
            if column in percent_columns:
                cells.append(_format_percent(value))
            else:
                cells.append("" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _decide_leader(
    human_n: int,
    model_n: int,
    human_exp: object,
    model_exp: object,
) -> tuple[str, object]:
    if human_n < 1 or model_n < 1:
        return "insufficient", np.nan
    human_value = float(human_exp) if human_exp is not None and np.isfinite(float(human_exp)) else None
    model_value = float(model_exp) if model_exp is not None and np.isfinite(float(model_exp)) else None
    if human_value is None or model_value is None:
        return "insufficient", np.nan
    edge = human_value - model_value
    if abs(edge) < 1e-9:
        return "tie", edge
    return ("human" if edge > 0 else "model"), edge


def _headline_verdict(head_to_head: pd.DataFrame) -> str:
    leaders = head_to_head["leader"].astype(str).tolist()
    human_wins = leaders.count("human")
    model_wins = leaders.count("model")
    if human_wins == 0 and model_wins == 0:
        return "Not enough completed outcomes yet / 已完成样本还不够，暂无定论"
    if human_wins > model_wins:
        return f"You lead {human_wins}-{model_wins} across windows / 你在多个窗口领先 {human_wins}-{model_wins}"
    if model_wins > human_wins:
        return f"Model leads {model_wins}-{human_wins} across windows / 模型领先 {model_wins}-{human_wins}"
    return f"Even {human_wins}-{model_wins} across windows / 打平 {human_wins}-{model_wins}"


def _format_percent(value: object) -> str:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "N/A"
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "N/A"


def _pending_count(
    frame: pd.DataFrame,
    return_col: str,
    pending_status_template: str | None,
    window: int,
) -> int:
    if pending_status_template is not None:
        status_col = pending_status_template.format(window=window)
        if status_col in frame.columns:
            statuses = frame[status_col].astype(str).str.lower().str.strip()
            return int((statuses == "pending").sum())
    return int(pd.to_numeric(frame[return_col], errors="coerce").isna().sum())


def _empty_metrics() -> dict[str, object]:
    return {
        "completed_count": 0,
        "beat_count": 0,
        "lag_count": 0,
        "beat_rate": np.nan,
        "expectancy": np.nan,
        "avg_win": np.nan,
        "avg_loss": np.nan,
        "payoff_ratio": np.nan,
        "profit_factor": np.nan,
        "consistency": np.nan,
        "total_excess": 0.0,
        "max_drawdown": 0.0,
    }
