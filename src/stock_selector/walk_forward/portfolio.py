"""Top-N portfolio replays of the validation signals: rebalances, equity curves and benchmark comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import (
    _is_finite,
)
from .config import (
    DEFAULT_BENCHMARK_TICKERS,
)
from .events import (
    _normalize_prices,
)


def summarize_walk_forward_portfolios(
    events: pd.DataFrame,
    target_window: int = 20,
    top_n: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_columns = [
        "portfolio_name",
        "portfolio_name_zh",
        "target_window_days",
        "top_n",
        "rebalance_count",
        "avg_position_count",
        "avg_forward_return",
        "median_forward_return",
        "win_rate",
        "best_forward_return",
        "worst_forward_return",
        "compounded_forward_return",
        "avg_selected_probability",
        "avg_selected_high_probability_score",
        "avg_selected_drawdown",
        "portfolio_note",
        "portfolio_note_zh",
    ]
    rebalance_columns = [
        "date",
        "portfolio_name",
        "portfolio_name_zh",
        "selected_count",
        "selected_tickers",
        f"portfolio_forward_return_{target_window}d",
        "portfolio_max_drawdown_after_signal",
        "avg_selected_probability",
        "avg_selected_high_probability_score",
    ]
    required = {
        "date",
        "ticker",
        f"forward_return_{target_window}d",
        "high_probability_score",
    }
    if events.empty or not required.issubset(events.columns):
        return (
            pd.DataFrame(columns=summary_columns),
            pd.DataFrame(columns=rebalance_columns),
        )

    strategies = [
        {
            "portfolio_name": "top_calibrated_probability",
            "portfolio_name_zh": "校准胜率前N",
            "score_column": "calibrated_win_probability",
            "buckets": None,
            "note": "Selects the highest calibrated win-probability candidates each signal date.",
            "note_zh": "每个信号日选择校准后胜率最高的候选。",
        },
        {
            "portfolio_name": "top_high_probability_score",
            "portfolio_name_zh": "高概率分数前N",
            "score_column": "high_probability_score",
            "buckets": None,
            "note": "Selects the highest high-probability-score candidates each signal date.",
            "note_zh": "每个信号日选择高概率分数最高的候选。",
        },
        {
            "portfolio_name": "strict_high_probability_only",
            "portfolio_name_zh": "只选严格高概率",
            "score_column": "calibrated_win_probability",
            "buckets": {"high_probability"},
            "note": "Selects only candidates that passed the strict high-probability gate.",
            "note_zh": "只选择通过严格高概率门槛的候选。",
        },
        {
            "portfolio_name": "watchlist_or_better",
            "portfolio_name_zh": "观察名单及以上",
            "score_column": "calibrated_win_probability",
            "buckets": {"high_probability", "near_watchlist"},
            "note": "Selects high-probability and near-watchlist candidates.",
            "note_zh": "选择高概率候选和接近观察名单的候选。",
        },
    ]

    all_rebalances: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for strategy in strategies:
        score_column = str(strategy["score_column"])
        if score_column not in events.columns:
            continue
        strategy_rebalances = _portfolio_rebalance_rows(
            events=events,
            target_window=target_window,
            top_n=top_n,
            portfolio_name=strategy["portfolio_name"],
            portfolio_name_zh=strategy["portfolio_name_zh"],
            score_column=score_column,
            buckets=strategy["buckets"],
        )
        all_rebalances.extend(strategy_rebalances)
        summary_rows.append(
            _portfolio_summary_row(
                strategy_rebalances,
                portfolio_name=strategy["portfolio_name"],
                portfolio_name_zh=strategy["portfolio_name_zh"],
                target_window=target_window,
                top_n=top_n,
                note=strategy["note"],
                note_zh=strategy["note_zh"],
            )
        )

    return (
        pd.DataFrame(summary_rows, columns=summary_columns),
        pd.DataFrame(all_rebalances, columns=rebalance_columns),
    )


def _portfolio_rebalance_rows(
    events: pd.DataFrame,
    target_window: int,
    top_n: int,
    portfolio_name: object,
    portfolio_name_zh: object,
    score_column: str,
    buckets: set[str] | None,
) -> list[dict[str, object]]:
    frame = events.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame[f"forward_return_{target_window}d"] = pd.to_numeric(
        frame[f"forward_return_{target_window}d"],
        errors="coerce",
    )
    frame[score_column] = pd.to_numeric(frame[score_column], errors="coerce")
    frame["high_probability_score"] = pd.to_numeric(
        frame["high_probability_score"],
        errors="coerce",
    )
    if "calibrated_win_probability" in frame.columns:
        frame["calibrated_win_probability"] = pd.to_numeric(
            frame["calibrated_win_probability"],
            errors="coerce",
        )
    if buckets is not None:
        frame = frame[frame["validation_bucket"].isin(buckets)].copy()
    frame = frame.dropna(subset=["date", "ticker", f"forward_return_{target_window}d", score_column])
    if frame.empty:
        return []

    rows: list[dict[str, object]] = []
    return_column = f"forward_return_{target_window}d"
    for date, group in frame.groupby("date", dropna=False):
        selected = (
            group.sort_values([score_column, "high_probability_score"], ascending=False)
            .drop_duplicates("ticker", keep="first")
            .head(top_n)
        )
        if selected.empty:
            continue
        drawdown = (
            float(pd.to_numeric(selected["max_drawdown_after_signal"], errors="coerce").mean())
            if "max_drawdown_after_signal" in selected.columns
            else np.nan
        )
        rows.append(
            {
                "date": date,
                "portfolio_name": portfolio_name,
                "portfolio_name_zh": portfolio_name_zh,
                "selected_count": int(len(selected)),
                "selected_tickers": ", ".join(selected["ticker"].astype(str).to_list()),
                f"portfolio_forward_return_{target_window}d": float(selected[return_column].mean()),
                "portfolio_max_drawdown_after_signal": drawdown,
                "avg_selected_probability": float(
                    selected["calibrated_win_probability"].mean()
                )
                if "calibrated_win_probability" in selected.columns
                else np.nan,
                "avg_selected_high_probability_score": float(
                    selected["high_probability_score"].mean()
                ),
            }
        )
    return rows


def _portfolio_summary_row(
    rebalance_rows: list[dict[str, object]],
    portfolio_name: object,
    portfolio_name_zh: object,
    target_window: int,
    top_n: int,
    note: object,
    note_zh: object,
) -> dict[str, object]:
    return_column = f"portfolio_forward_return_{target_window}d"
    if not rebalance_rows:
        return {
            "portfolio_name": portfolio_name,
            "portfolio_name_zh": portfolio_name_zh,
            "target_window_days": int(target_window),
            "top_n": int(top_n),
            "rebalance_count": 0,
            "avg_position_count": 0.0,
            "avg_forward_return": np.nan,
            "median_forward_return": np.nan,
            "win_rate": np.nan,
            "best_forward_return": np.nan,
            "worst_forward_return": np.nan,
            "compounded_forward_return": np.nan,
            "avg_selected_probability": np.nan,
            "avg_selected_high_probability_score": np.nan,
            "avg_selected_drawdown": np.nan,
            "portfolio_note": note,
            "portfolio_note_zh": note_zh,
        }
    frame = pd.DataFrame(rebalance_rows)
    returns = pd.to_numeric(frame[return_column], errors="coerce").dropna()
    return {
        "portfolio_name": portfolio_name,
        "portfolio_name_zh": portfolio_name_zh,
        "target_window_days": int(target_window),
        "top_n": int(top_n),
        "rebalance_count": int(len(returns)),
        "avg_position_count": float(pd.to_numeric(frame["selected_count"], errors="coerce").mean()),
        "avg_forward_return": float(returns.mean()) if len(returns) else np.nan,
        "median_forward_return": float(returns.median()) if len(returns) else np.nan,
        "win_rate": float((returns > 0).mean()) if len(returns) else np.nan,
        "best_forward_return": float(returns.max()) if len(returns) else np.nan,
        "worst_forward_return": float(returns.min()) if len(returns) else np.nan,
        "compounded_forward_return": float((1.0 + returns).prod() - 1.0)
        if len(returns)
        else np.nan,
        "avg_selected_probability": float(
            pd.to_numeric(frame["avg_selected_probability"], errors="coerce").mean()
        ),
        "avg_selected_high_probability_score": float(
            pd.to_numeric(frame["avg_selected_high_probability_score"], errors="coerce").mean()
        ),
        "avg_selected_drawdown": float(
            pd.to_numeric(frame["portfolio_max_drawdown_after_signal"], errors="coerce").mean()
        ),
        "portfolio_note": note,
        "portfolio_note_zh": note_zh,
    }


def build_walk_forward_portfolio_equity(
    prices: pd.DataFrame,
    portfolio_rebalances: pd.DataFrame,
    target_window: int = 20,
    transaction_cost: float = 0.001,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_columns = [
        "portfolio_name",
        "portfolio_name_zh",
        "daily_rows",
        "start_date",
        "end_date",
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "max_drawdown",
        "positive_day_rate",
        "avg_daily_return",
        "best_daily_return",
        "worst_daily_return",
        "avg_position_count",
        "avg_turnover",
        "transaction_cost",
        "equity_note",
        "equity_note_zh",
    ]
    curve_columns = [
        "date",
        "portfolio_name",
        "portfolio_name_zh",
        "selected_tickers",
        "selected_count",
        "daily_return_before_cost",
        "turnover",
        "transaction_cost_impact",
        "daily_return",
        "equity",
        "drawdown",
    ]
    required_rebalances = {"date", "portfolio_name", "portfolio_name_zh", "selected_tickers"}
    required_prices = {"date", "ticker", "adj_close"}
    if (
        prices.empty
        or portfolio_rebalances.empty
        or not required_rebalances.issubset(portfolio_rebalances.columns)
        or not required_prices.issubset(prices.columns)
    ):
        return (
            pd.DataFrame(columns=summary_columns),
            pd.DataFrame(columns=curve_columns),
        )

    price_frame = _normalize_prices(prices)
    price_pivot = (
        price_frame.pivot_table(
            index="date",
            columns="ticker",
            values="adj_close",
            aggfunc="last",
        )
        .sort_index()
        .astype(float)
    )
    daily_returns = price_pivot.pct_change().replace([np.inf, -np.inf], np.nan)
    if daily_returns.empty:
        return (
            pd.DataFrame(columns=summary_columns),
            pd.DataFrame(columns=curve_columns),
        )

    rebalance_frame = portfolio_rebalances.copy()
    rebalance_frame["date"] = pd.to_datetime(rebalance_frame["date"], errors="coerce")
    rebalance_frame = rebalance_frame.dropna(subset=["date", "portfolio_name", "selected_tickers"])
    if rebalance_frame.empty:
        return (
            pd.DataFrame(columns=summary_columns),
            pd.DataFrame(columns=curve_columns),
        )

    all_curve_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    group_columns = ["portfolio_name", "portfolio_name_zh"]
    for keys, group in rebalance_frame.groupby(group_columns, dropna=False):
        portfolio_name, portfolio_name_zh = keys
        curve_rows = _portfolio_equity_rows(
            daily_returns=daily_returns,
            rebalances=group.sort_values("date"),
            portfolio_name=portfolio_name,
            portfolio_name_zh=portfolio_name_zh,
            target_window=target_window,
            transaction_cost=transaction_cost,
        )
        all_curve_rows.extend(curve_rows)
        summary_rows.append(
            _portfolio_equity_summary_row(
                curve_rows=curve_rows,
                portfolio_name=portfolio_name,
                portfolio_name_zh=portfolio_name_zh,
                transaction_cost=transaction_cost,
            )
        )

    return (
        pd.DataFrame(summary_rows, columns=summary_columns),
        pd.DataFrame(all_curve_rows, columns=curve_columns),
    )


def _portfolio_equity_rows(
    daily_returns: pd.DataFrame,
    rebalances: pd.DataFrame,
    portfolio_name: object,
    portfolio_name_zh: object,
    target_window: int,
    transaction_cost: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rebalance_rows = list(rebalances.itertuples(index=False))
    if not rebalance_rows:
        return rows

    equity = 1.0
    previous_weights: dict[str, float] = {}
    last_written_date: pd.Timestamp | None = None
    available_dates = pd.Index(daily_returns.index)

    for index, rebalance in enumerate(rebalance_rows):
        rebalance_date = pd.Timestamp(rebalance.date)
        selected_tickers = _parse_selected_tickers(getattr(rebalance, "selected_tickers", ""))
        selected_tickers = [ticker for ticker in selected_tickers if ticker in daily_returns.columns]
        if not selected_tickers:
            continue
        start_position = int(available_dates.searchsorted(rebalance_date, side="left"))
        if start_position >= len(available_dates):
            continue
        if index + 1 < len(rebalance_rows):
            next_date = pd.Timestamp(rebalance_rows[index + 1].date)
            end_position = int(available_dates.searchsorted(next_date, side="left"))
        else:
            end_position = min(start_position + target_window + 1, len(available_dates))
        if end_position <= start_position:
            continue

        current_weights = {ticker: 1.0 / len(selected_tickers) for ticker in selected_tickers}
        turnover = _portfolio_turnover(previous_weights, current_weights)
        previous_weights = current_weights
        dates = available_dates[start_position:end_position]
        for date_offset, date in enumerate(dates):
            date = pd.Timestamp(date)
            if last_written_date is not None and date <= last_written_date:
                continue
            selected_returns = daily_returns.loc[date, selected_tickers].dropna()
            daily_return_before_cost = (
                float(selected_returns.mean())
                if len(selected_returns) and date_offset > 0
                else 0.0
            )
            cost_impact = turnover * transaction_cost if date_offset == 0 else 0.0
            daily_return = daily_return_before_cost - cost_impact
            equity *= 1.0 + daily_return
            rows.append(
                {
                    "date": date,
                    "portfolio_name": portfolio_name,
                    "portfolio_name_zh": portfolio_name_zh,
                    "selected_tickers": ", ".join(selected_tickers),
                    "selected_count": int(len(selected_tickers)),
                    "daily_return_before_cost": daily_return_before_cost,
                    "turnover": float(turnover) if date_offset == 0 else 0.0,
                    "transaction_cost_impact": float(cost_impact),
                    "daily_return": float(daily_return),
                    "equity": float(equity),
                    "drawdown": np.nan,
                }
            )
            last_written_date = date

    if rows:
        equity_series = pd.Series([row["equity"] for row in rows], dtype=float)
        drawdowns = equity_series / equity_series.cummax() - 1.0
        for row, drawdown in zip(rows, drawdowns):
            row["drawdown"] = float(drawdown)
    return rows


def _portfolio_equity_summary_row(
    curve_rows: list[dict[str, object]],
    portfolio_name: object,
    portfolio_name_zh: object,
    transaction_cost: float,
) -> dict[str, object]:
    if not curve_rows:
        return {
            "portfolio_name": portfolio_name,
            "portfolio_name_zh": portfolio_name_zh,
            "daily_rows": 0,
            "start_date": pd.NaT,
            "end_date": pd.NaT,
            "total_return": np.nan,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "positive_day_rate": np.nan,
            "avg_daily_return": np.nan,
            "best_daily_return": np.nan,
            "worst_daily_return": np.nan,
            "avg_position_count": np.nan,
            "avg_turnover": np.nan,
            "transaction_cost": transaction_cost,
            "equity_note": "No daily portfolio equity curve could be built.",
            "equity_note_zh": "没有生成逐日组合净值曲线。",
        }
    frame = pd.DataFrame(curve_rows)
    returns = pd.to_numeric(frame["daily_return"], errors="coerce").fillna(0.0)
    equity = pd.to_numeric(frame["equity"], errors="coerce").dropna()
    total_return = float(equity.iloc[-1] - 1.0) if len(equity) else np.nan
    day_count = len(returns)
    annualized_return = (
        float((1.0 + total_return) ** (252 / day_count) - 1.0)
        if day_count > 0 and _is_finite(total_return) and total_return > -1.0
        else np.nan
    )
    volatility = float(returns.std(ddof=0) * np.sqrt(252)) if day_count > 1 else np.nan
    sharpe = (
        float((returns.mean() / returns.std(ddof=0)) * np.sqrt(252))
        if day_count > 1 and returns.std(ddof=0) > 0
        else np.nan
    )
    return {
        "portfolio_name": portfolio_name,
        "portfolio_name_zh": portfolio_name_zh,
        "daily_rows": int(day_count),
        "start_date": frame["date"].min(),
        "end_date": frame["date"].max(),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": float(pd.to_numeric(frame["drawdown"], errors="coerce").min()),
        "positive_day_rate": float((returns > 0).mean()) if day_count else np.nan,
        "avg_daily_return": float(returns.mean()) if day_count else np.nan,
        "best_daily_return": float(returns.max()) if day_count else np.nan,
        "worst_daily_return": float(returns.min()) if day_count else np.nan,
        "avg_position_count": float(pd.to_numeric(frame["selected_count"], errors="coerce").mean()),
        "avg_turnover": float(pd.to_numeric(frame["turnover"], errors="coerce").mean()),
        "transaction_cost": transaction_cost,
        "equity_note": (
            "Daily curve holds each equal-weight basket until the next rebalance "
            "or the target validation window, including transaction-cost drag."
        ),
        "equity_note_zh": (
            "逐日曲线按等权组合持有到下一次调仓或目标验证窗口，"
            "并计入交易成本拖累。"
        ),
    }


def _parse_selected_tickers(value: object) -> list[str]:
    return [
        str(item).strip().upper()
        for item in str(value).split(",")
        if str(item).strip()
    ]


def _portfolio_turnover(
    previous_weights: dict[str, float],
    current_weights: dict[str, float],
) -> float:
    tickers = set(previous_weights) | set(current_weights)
    if not tickers:
        return 0.0
    return float(
        0.5
        * sum(abs(current_weights.get(ticker, 0.0) - previous_weights.get(ticker, 0.0)) for ticker in tickers)
    )


def build_walk_forward_benchmark_comparison(
    prices: pd.DataFrame,
    portfolio_equity_curve: pd.DataFrame,
    benchmark_tickers: tuple[str, ...] = DEFAULT_BENCHMARK_TICKERS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_columns = [
        "portfolio_name",
        "portfolio_name_zh",
        "benchmark_ticker",
        "daily_rows",
        "start_date",
        "end_date",
        "portfolio_total_return",
        "benchmark_total_return",
        "excess_total_return",
        "portfolio_annualized_return",
        "benchmark_annualized_return",
        "excess_annualized_return",
        "portfolio_max_drawdown",
        "benchmark_max_drawdown",
        "drawdown_advantage",
        "portfolio_sharpe",
        "benchmark_sharpe",
        "daily_correlation",
        "benchmark_note",
        "benchmark_note_zh",
    ]
    curve_columns = [
        "date",
        "portfolio_name",
        "portfolio_name_zh",
        "benchmark_ticker",
        "portfolio_daily_return",
        "benchmark_daily_return",
        "excess_daily_return",
        "portfolio_equity",
        "benchmark_equity",
        "relative_equity",
        "portfolio_drawdown",
        "benchmark_drawdown",
    ]
    required_curve = {"date", "portfolio_name", "portfolio_name_zh", "daily_return", "equity", "drawdown"}
    required_prices = {"date", "ticker", "adj_close"}
    if (
        prices.empty
        or portfolio_equity_curve.empty
        or not required_curve.issubset(portfolio_equity_curve.columns)
        or not required_prices.issubset(prices.columns)
    ):
        return pd.DataFrame(columns=summary_columns), pd.DataFrame(columns=curve_columns)

    price_frame = _normalize_prices(prices)
    benchmark_tickers = tuple(dict.fromkeys(str(ticker).upper().strip() for ticker in benchmark_tickers if str(ticker).strip()))
    price_frame = price_frame[price_frame["ticker"].isin(benchmark_tickers)].copy()
    if price_frame.empty:
        return pd.DataFrame(columns=summary_columns), pd.DataFrame(columns=curve_columns)

    benchmark_prices = (
        price_frame.pivot_table(
            index="date",
            columns="ticker",
            values="adj_close",
            aggfunc="last",
        )
        .sort_index()
        .astype(float)
    )
    benchmark_returns = benchmark_prices.pct_change().replace([np.inf, -np.inf], np.nan)

    curve_frame = portfolio_equity_curve.copy()
    curve_frame["date"] = pd.to_datetime(curve_frame["date"], errors="coerce")
    curve_frame["daily_return"] = pd.to_numeric(curve_frame["daily_return"], errors="coerce")
    curve_frame["equity"] = pd.to_numeric(curve_frame["equity"], errors="coerce")
    curve_frame["drawdown"] = pd.to_numeric(curve_frame["drawdown"], errors="coerce")
    curve_frame = curve_frame.dropna(subset=["date", "portfolio_name", "daily_return", "equity"])
    if curve_frame.empty:
        return pd.DataFrame(columns=summary_columns), pd.DataFrame(columns=curve_columns)

    all_curve_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    group_columns = ["portfolio_name", "portfolio_name_zh"]
    for keys, group in curve_frame.groupby(group_columns, dropna=False):
        portfolio_name, portfolio_name_zh = keys
        group = group.sort_values("date")
        for benchmark_ticker in benchmark_tickers:
            if benchmark_ticker not in benchmark_returns.columns:
                summary_rows.append(
                    _missing_benchmark_summary_row(
                        portfolio_name,
                        portfolio_name_zh,
                        benchmark_ticker,
                    )
                )
                continue
            comparison_rows = _benchmark_comparison_rows(
                portfolio_group=group,
                benchmark_returns=benchmark_returns[benchmark_ticker],
                portfolio_name=portfolio_name,
                portfolio_name_zh=portfolio_name_zh,
                benchmark_ticker=benchmark_ticker,
            )
            all_curve_rows.extend(comparison_rows)
            summary_rows.append(
                _benchmark_summary_row(
                    comparison_rows=comparison_rows,
                    portfolio_name=portfolio_name,
                    portfolio_name_zh=portfolio_name_zh,
                    benchmark_ticker=benchmark_ticker,
                )
            )

    return (
        pd.DataFrame(summary_rows, columns=summary_columns),
        pd.DataFrame(all_curve_rows, columns=curve_columns),
    )


def _benchmark_comparison_rows(
    portfolio_group: pd.DataFrame,
    benchmark_returns: pd.Series,
    portfolio_name: object,
    portfolio_name_zh: object,
    benchmark_ticker: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    benchmark_equity = 1.0
    benchmark_values: list[float] = []
    for index, row in enumerate(portfolio_group.itertuples(index=False)):
        date = pd.Timestamp(row.date)
        if date not in benchmark_returns.index:
            continue
        portfolio_daily_return = float(row.daily_return)
        benchmark_daily_return = 0.0 if index == 0 else float(benchmark_returns.loc[date])
        if not _is_finite(benchmark_daily_return):
            benchmark_daily_return = 0.0
        benchmark_equity *= 1.0 + benchmark_daily_return
        benchmark_values.append(benchmark_equity)
        benchmark_series = pd.Series(benchmark_values, dtype=float)
        benchmark_drawdown = float(benchmark_series.iloc[-1] / benchmark_series.cummax().iloc[-1] - 1.0)
        rows.append(
            {
                "date": date,
                "portfolio_name": portfolio_name,
                "portfolio_name_zh": portfolio_name_zh,
                "benchmark_ticker": benchmark_ticker,
                "portfolio_daily_return": portfolio_daily_return,
                "benchmark_daily_return": benchmark_daily_return,
                "excess_daily_return": portfolio_daily_return - benchmark_daily_return,
                "portfolio_equity": float(row.equity),
                "benchmark_equity": float(benchmark_equity),
                "relative_equity": float(row.equity / benchmark_equity - 1.0)
                if benchmark_equity > 0
                else np.nan,
                "portfolio_drawdown": float(row.drawdown)
                if _is_finite(row.drawdown)
                else np.nan,
                "benchmark_drawdown": benchmark_drawdown,
            }
        )
    return rows


def _benchmark_summary_row(
    comparison_rows: list[dict[str, object]],
    portfolio_name: object,
    portfolio_name_zh: object,
    benchmark_ticker: str,
) -> dict[str, object]:
    if not comparison_rows:
        return _missing_benchmark_summary_row(
            portfolio_name,
            portfolio_name_zh,
            benchmark_ticker,
        )
    frame = pd.DataFrame(comparison_rows)
    portfolio_returns = pd.to_numeric(frame["portfolio_daily_return"], errors="coerce").fillna(0.0)
    benchmark_returns = pd.to_numeric(frame["benchmark_daily_return"], errors="coerce").fillna(0.0)
    daily_rows = len(frame)
    portfolio_total_return = float(frame["portfolio_equity"].iloc[-1] - 1.0)
    benchmark_total_return = float(frame["benchmark_equity"].iloc[-1] - 1.0)
    excess_total_return = portfolio_total_return - benchmark_total_return
    portfolio_annualized = _annualized_return(portfolio_total_return, daily_rows)
    benchmark_annualized = _annualized_return(benchmark_total_return, daily_rows)
    correlation = (
        float(portfolio_returns.corr(benchmark_returns))
        if daily_rows > 2 and portfolio_returns.std(ddof=0) > 0 and benchmark_returns.std(ddof=0) > 0
        else np.nan
    )
    return {
        "portfolio_name": portfolio_name,
        "portfolio_name_zh": portfolio_name_zh,
        "benchmark_ticker": benchmark_ticker,
        "daily_rows": int(daily_rows),
        "start_date": frame["date"].min(),
        "end_date": frame["date"].max(),
        "portfolio_total_return": portfolio_total_return,
        "benchmark_total_return": benchmark_total_return,
        "excess_total_return": excess_total_return,
        "portfolio_annualized_return": portfolio_annualized,
        "benchmark_annualized_return": benchmark_annualized,
        "excess_annualized_return": portfolio_annualized - benchmark_annualized
        if _is_finite(portfolio_annualized) and _is_finite(benchmark_annualized)
        else np.nan,
        "portfolio_max_drawdown": float(pd.to_numeric(frame["portfolio_drawdown"], errors="coerce").min()),
        "benchmark_max_drawdown": float(pd.to_numeric(frame["benchmark_drawdown"], errors="coerce").min()),
        "drawdown_advantage": float(
            pd.to_numeric(frame["portfolio_drawdown"], errors="coerce").min()
            - pd.to_numeric(frame["benchmark_drawdown"], errors="coerce").min()
        ),
        "portfolio_sharpe": _sharpe(portfolio_returns),
        "benchmark_sharpe": _sharpe(benchmark_returns),
        "daily_correlation": correlation,
        "benchmark_note": (
            f"Compared daily portfolio equity against {benchmark_ticker} over matching dates."
        ),
        "benchmark_note_zh": (
            f"在相同日期上，将组合逐日净值与{benchmark_ticker}进行对比。"
        ),
    }


def _missing_benchmark_summary_row(
    portfolio_name: object,
    portfolio_name_zh: object,
    benchmark_ticker: str,
) -> dict[str, object]:
    return {
        "portfolio_name": portfolio_name,
        "portfolio_name_zh": portfolio_name_zh,
        "benchmark_ticker": benchmark_ticker,
        "daily_rows": 0,
        "start_date": pd.NaT,
        "end_date": pd.NaT,
        "portfolio_total_return": np.nan,
        "benchmark_total_return": np.nan,
        "excess_total_return": np.nan,
        "portfolio_annualized_return": np.nan,
        "benchmark_annualized_return": np.nan,
        "excess_annualized_return": np.nan,
        "portfolio_max_drawdown": np.nan,
        "benchmark_max_drawdown": np.nan,
        "drawdown_advantage": np.nan,
        "portfolio_sharpe": np.nan,
        "benchmark_sharpe": np.nan,
        "daily_correlation": np.nan,
        "benchmark_note": f"{benchmark_ticker} benchmark price data is unavailable.",
        "benchmark_note_zh": f"缺少{benchmark_ticker}基准价格数据。",
    }


def _annualized_return(total_return: float, day_count: int) -> float:
    if day_count <= 0 or not _is_finite(total_return) or total_return <= -1.0:
        return np.nan
    return float((1.0 + total_return) ** (252 / day_count) - 1.0)


def _sharpe(daily_returns: pd.Series) -> float:
    returns = pd.to_numeric(daily_returns, errors="coerce").dropna()
    if len(returns) < 2 or returns.std(ddof=0) <= 0:
        return np.nan
    return float((returns.mean() / returns.std(ddof=0)) * np.sqrt(252))
