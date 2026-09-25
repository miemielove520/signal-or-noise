"""Long-term paper-trading performance tracker.

``paper.py`` generates simulated orders and holds a paper portfolio state. This
module turns that into a *forward evaluation system*: mark the paper portfolio to
market each day, record an equity curve, compare it to a benchmark, and produce a
readiness verdict. The point is to prove — with money at risk only on paper, over
weeks — that the model's picks actually hold up before any real or IBKR trading.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


CASH_TICKER = "CASH"
# Minimum forward trading days before a paper track is worth judging at all.
PAPER_MIN_TRACK_DAYS = 20
EQUITY_HISTORY_COLUMNS = ("date", "equity", "benchmark_close")


@dataclass(frozen=True)
class PaperPerformance:
    status: str
    status_zh: str
    days_tracked: int
    start_date: str | None
    latest_date: str | None
    start_equity: float | None
    latest_equity: float | None
    total_return: float | None
    benchmark_return: float | None
    excess_return: float | None
    max_drawdown: float | None
    readiness: str
    readiness_zh: str
    note: str
    note_zh: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def mark_to_market(
    state: pd.DataFrame,
    price_lookup: dict[str, float],
    fallback_price_lookup: dict[str, float] | None = None,
) -> float:
    """Value the portfolio without silently treating missing prices as zero."""
    if state is None or state.empty:
        return 0.0
    frame = state.copy()
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    if "ticker" not in frame.columns or "quantity" not in frame.columns:
        return 0.0
    prices = {str(k).upper().strip(): float(v) for k, v in (price_lookup or {}).items()}
    fallback_prices = {
        str(k).upper().strip(): float(v) for k, v in (fallback_price_lookup or {}).items()
    }
    equity = 0.0
    for _, row in frame.iterrows():
        ticker = str(row["ticker"]).upper().strip()
        qty = _f(row["quantity"]) or 0.0
        if ticker == CASH_TICKER:
            equity += qty
        else:
            price = prices.get(ticker, fallback_prices.get(ticker))
            if price is None or not np.isfinite(price) or price <= 0:
                raise ValueError(f"Missing valuation price for held ticker {ticker}.")
            equity += qty * price
    return round(equity, 2)


def record_equity_point(
    history: pd.DataFrame | None,
    date: str,
    equity: float,
    benchmark_close: float | None,
) -> pd.DataFrame:
    """Append (or replace) one day's equity point, keeping one row per date."""
    row = {"date": str(date), "equity": float(equity), "benchmark_close": benchmark_close}
    base = history.copy() if history is not None and not history.empty else pd.DataFrame(
        columns=list(EQUITY_HISTORY_COLUMNS)
    )
    base = base[base["date"].astype(str) != str(date)] if "date" in base.columns else base
    updated = pd.concat([base, pd.DataFrame([row])], ignore_index=True)
    return updated.sort_values("date").reset_index(drop=True)


def summarize_paper_performance(
    history: pd.DataFrame,
    min_days: int = PAPER_MIN_TRACK_DAYS,
) -> PaperPerformance:
    if history is None or history.empty or "equity" not in history.columns:
        return _insufficient(0)
    frame = history.copy().sort_values("date").reset_index(drop=True)
    equity = pd.to_numeric(frame["equity"], errors="coerce").dropna()
    days = int(len(equity))
    if days < 2:
        latest = round(float(equity.iloc[-1]), 2) if days == 1 else None
        latest_date = str(frame["date"].iloc[-1]) if days == 1 else None
        return _insufficient(days, latest_equity=latest, latest_date=latest_date)

    start_equity = float(equity.iloc[0])
    latest_equity = float(equity.iloc[-1])
    total_return = latest_equity / start_equity - 1.0 if start_equity else None

    benchmark_return = None
    if "benchmark_close" in frame.columns:
        bench = pd.to_numeric(frame["benchmark_close"], errors="coerce").dropna()
        if len(bench) >= 2 and float(bench.iloc[0]) > 0:
            benchmark_return = float(bench.iloc[-1]) / float(bench.iloc[0]) - 1.0
    excess = (
        total_return - benchmark_return
        if total_return is not None and benchmark_return is not None
        else None
    )

    running_max = equity.cummax()
    max_drawdown = float((equity / running_max - 1.0).min())

    readiness, readiness_zh, status, status_zh, note, note_zh = _verdict(
        days, min_days, excess, max_drawdown, total_return
    )
    return PaperPerformance(
        status=status,
        status_zh=status_zh,
        days_tracked=days,
        start_date=str(frame["date"].iloc[0]),
        latest_date=str(frame["date"].iloc[-1]),
        start_equity=round(start_equity, 2),
        latest_equity=round(latest_equity, 2),
        total_return=_r(total_return),
        benchmark_return=_r(benchmark_return),
        excess_return=_r(excess),
        max_drawdown=round(max_drawdown, 4),
        readiness=readiness,
        readiness_zh=readiness_zh,
        note=note,
        note_zh=note_zh,
    )


def expected_annual_from_walk_forward(
    summary: pd.DataFrame, window: int = 20
) -> float | None:
    """Backtest's expected annual return, from the walk-forward per-signal return.

    ``avg_return_{window}d`` is the mean return per signal over ``window`` trading
    days; sample-weighted across rows and annualized to a comparable yearly figure.
    """
    if summary is None or summary.empty:
        return None
    col = f"avg_return_{window}d"
    if col not in summary.columns:
        return None
    returns = pd.to_numeric(summary[col], errors="coerce")
    weights = (
        pd.to_numeric(summary["sample_count"], errors="coerce")
        if "sample_count" in summary.columns
        else pd.Series(1.0, index=summary.index)
    )
    mask = returns.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return None
    per_period = float((returns[mask] * weights[mask]).sum() / weights[mask].sum())
    periods_per_year = 252.0 / window
    return (1.0 + per_period) ** periods_per_year - 1.0


def compare_paper_to_backtest(
    paper: PaperPerformance,
    expected_annual_return: float | None,
    min_days: int = PAPER_MIN_TRACK_DAYS,
) -> dict[str, object]:
    """Gate: does the paper run live up to what the backtest predicted? If the
    backtest expects a positive edge and the paper run is flat/negative over a
    meaningful window, the backtest was likely overfit — do not progress to IBKR."""
    out: dict[str, object] = {
        "backtest_expected_annual": _r(expected_annual_return),
        "paper_annual": None,
        "gate": "no_backtest",
        "gate_zh": "无回测预期",
        "note_zh": "还没有回测预期可对比；先跑 validate.py 生成 walk_forward_summary。",
    }
    if expected_annual_return is None:
        return out
    if paper.days_tracked < min_days or paper.total_return is None:
        out["gate"], out["gate_zh"] = "accumulating", "样本积累中"
        out["note_zh"] = f"追踪不足{min_days}个交易日，暂不与回测对比。"
        return out

    paper_annual = (1.0 + float(paper.total_return)) ** (252.0 / paper.days_tracked) - 1.0
    out["paper_annual"] = _r(paper_annual)
    if expected_annual_return <= 0:
        out["gate"], out["gate_zh"] = "no_edge_expected", "回测本身无正预期"
        out["note_zh"] = "回测预期收益非正，无从判断是否达标。"
    elif paper_annual <= 0:
        out["gate"], out["gate_zh"] = "divergent", "严重偏离（疑似过拟合）"
        out["note_zh"] = "回测预期为正、实盘却在亏——回测很可能过拟合，**不要上实盘/IBKR**。"
    elif paper_annual < 0.4 * expected_annual_return:
        out["gate"], out["gate_zh"] = "below_expectation", "明显低于预期"
        out["note_zh"] = "实盘年化远低于回测预期（不足40%），警惕过拟合，继续观察别急着上实盘。"
    else:
        out["gate"], out["gate_zh"] = "consistent", "与回测基本一致"
        out["note_zh"] = "实盘表现和回测预期基本吻合，策略可信度提升。"
    return out


def _verdict(
    days: int,
    min_days: int,
    excess: float | None,
    max_drawdown: float,
    total_return: float | None,
) -> tuple[str, str, str, str, str, str]:
    if days < min_days:
        return (
            "accumulating",
            "样本积累中",
            "tracking",
            "追踪中",
            f"Only {days}/{min_days} trading days tracked — too early to judge; keep the paper run going.",
            f"只追踪了{days}/{min_days}个交易日，样本还太少，暂不能下结论；继续跑模拟盘。",
        )
    # Enough days: judge vs benchmark and drawdown.
    if max_drawdown <= -0.25 or (excess is not None and excess <= -0.10):
        return (
            "not_ready",
            "暂不达标",
            "underperforming",
            "表现偏弱",
            "Paper performance lags the benchmark or drew down hard — do NOT progress to live/IBKR yet.",
            "模拟盘明显跑输基准或回撤过大——先别进入实盘/IBKR。",
        )
    if excess is not None and excess >= 0.0:
        return (
            "on_track",
            "达标",
            "on_track",
            "达标",
            "Paper performance is holding up vs the benchmark over a meaningful window.",
            "在足够长的窗口里，模拟盘表现不输基准，站得住。",
        )
    return (
        "borderline",
        "临界",
        "borderline",
        "临界",
        "Trailing the benchmark slightly but drawdown is contained — keep tracking before deciding.",
        "略微跑输基准但回撤可控——继续追踪再决定。",
    )


def _insufficient(
    days: int, latest_equity: float | None = None, latest_date: str | None = None
) -> PaperPerformance:
    return PaperPerformance(
        status="tracking",
        status_zh="追踪中",
        days_tracked=days,
        start_date=None,
        latest_date=latest_date,
        start_equity=None,
        latest_equity=latest_equity,
        total_return=None,
        benchmark_return=None,
        excess_return=None,
        max_drawdown=None,
        readiness="accumulating",
        readiness_zh="样本积累中",
        note="Not enough equity history yet; run the daily paper mark-to-market to build it.",
        note_zh="净值历史还不够；每天给模拟盘盯市来积累。",
    )


def _f(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _r(value: float | None) -> float | None:
    return round(float(value), 4) if value is not None else None
