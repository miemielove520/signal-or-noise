"""Record the human investor's own trade decisions for the human-vs-model scoreboard.

This module intentionally keeps a schema that is compatible with the model
signal history in :mod:`stock_selector.signal_review`, so the same forward-outcome
filling and summary logic can later be reused for both actors. Human decisions are
stored in a separate file (``human_trades.csv``) so they never contaminate the
model's own review-learning feedback loop.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


HUMAN_ACTOR = "human"
VALID_ACTIONS = ("BUY", "SELL", "ADD", "TRIM")
DEFAULT_BENCHMARK = "QQQ"
VALID_BENCHMARKS = ("QQQ", "SPY")

# Forward-outcome windows in trading days. Kept identical to
# ``stock_selector.signal_review.FORWARD_WINDOWS`` so both actors are comparable.
FORWARD_WINDOWS = (5, 20, 60)

HUMAN_TRADES_FILENAME = "human_trades.csv"

# Columns written for every human decision. Forward-outcome columns
# (forward_return_5d, benchmark_return_5d, excess_return_5d, ...) are added later
# by the outcome-filling step and are intentionally not created here.
CORE_COLUMNS = (
    "created_at_utc",
    "signal_date",
    "ticker",
    "actor",
    "action",
    "weight_pct",
    "signal_price",
    "benchmark",
    "reason",
    "report_path",
)


def append_human_trade(
    ticker: str,
    action: str,
    weight_pct: float,
    signal_price: float,
    signal_date: str | pd.Timestamp | None = None,
    reason: str = "",
    benchmark: str = DEFAULT_BENCHMARK,
    review_root: str | Path = "outputs/signal_review",
    report_path: str | Path | None = None,
) -> pd.DataFrame:
    """Validate and append one human trade decision to ``human_trades.csv``.

    Returns the full, de-duplicated human-trades frame after the append.
    Raises ``ValueError`` on invalid input so callers can show a friendly error.
    """

    row = _build_human_trade_row(
        ticker=ticker,
        action=action,
        weight_pct=weight_pct,
        signal_price=signal_price,
        signal_date=signal_date,
        reason=reason,
        benchmark=benchmark,
        report_path=report_path,
    )

    review_dir = Path(review_root)
    review_dir.mkdir(parents=True, exist_ok=True)
    path = review_dir / HUMAN_TRADES_FILENAME

    existing = load_human_trades(review_dir)
    combined = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    combined = _deduplicate_human_trades(combined)
    combined = combined.sort_values(["signal_date", "ticker", "action"]).reset_index(drop=True)
    combined = _order_columns(combined)
    combined.to_csv(path, index=False)
    return combined


def load_human_trades(review_root: str | Path = "outputs/signal_review") -> pd.DataFrame:
    """Load the human trades file, or an empty frame with the core schema."""
    path = Path(review_root) / HUMAN_TRADES_FILENAME
    if not path.exists():
        return pd.DataFrame(columns=list(CORE_COLUMNS))
    try:
        frame = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=list(CORE_COLUMNS))
    return _order_columns(frame)


def update_human_trade_outcomes(
    price_lookup: dict[str, pd.DataFrame],
    benchmark_lookup: dict[str, pd.DataFrame],
    review_root: str | Path = "outputs/signal_review",
) -> pd.DataFrame:
    """Load human trades, fill forward/excess outcomes, and save back.

    ``price_lookup`` maps a ticker to its daily price frame (needs ``date`` and a
    close column). ``benchmark_lookup`` maps a benchmark symbol (``QQQ``/``SPY``)
    to its price frame. Both are passed in so this function stays offline-testable;
    a caller is responsible for fetching prices.
    """
    review_dir = Path(review_root)
    trades = load_human_trades(review_dir)
    filled = fill_human_trade_outcomes(trades, price_lookup, benchmark_lookup)
    if not filled.empty:
        review_dir.mkdir(parents=True, exist_ok=True)
        filled.to_csv(review_dir / HUMAN_TRADES_FILENAME, index=False)
    return filled


def fill_human_trade_outcomes(
    trades: pd.DataFrame,
    price_lookup: dict[str, pd.DataFrame],
    benchmark_lookup: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compute forward return, benchmark return, and excess return per window.

    A "win" is defined as beating the benchmark (excess return > 0), matching the
    scoreboard's chosen success metric. Absolute forward return is kept alongside
    so the raw move is still visible. No lookahead: the stock's outcome date is the
    signal date plus ``window`` trading days on the stock's own calendar, and the
    benchmark is measured close-to-close over that same calendar span.
    """
    if trades is None or trades.empty:
        return trades if trades is not None else pd.DataFrame(columns=list(CORE_COLUMNS))

    normalized_prices = {
        str(ticker).upper().strip(): _normalize_price_frame(frame)
        for ticker, frame in (price_lookup or {}).items()
    }
    normalized_benchmarks = {
        str(symbol).upper().strip(): _normalize_price_frame(frame)
        for symbol, frame in (benchmark_lookup or {}).items()
    }

    updated = trades.copy().reset_index(drop=True)
    for index, row in updated.iterrows():
        ticker = str(row.get("ticker", "")).upper().strip()
        action = str(row.get("action", "BUY")).upper().strip()
        direction = -1.0 if action in {"SELL", "TRIM"} else 1.0
        updated.at[index, "decision_direction"] = int(direction)
        prices = normalized_prices.get(ticker)
        if prices is None or prices.empty:
            continue
        signal_date = pd.to_datetime(row.get("signal_date"), errors="coerce")
        if pd.isna(signal_date):
            continue
        eligible = prices[prices["date"] <= signal_date.normalize()]
        if eligible.empty:
            continue
        signal_index = int(eligible.index[-1])
        signal_price = _safe_finite_float(row.get("signal_price"))
        if signal_price is None or signal_price <= 0:
            signal_price = float(prices.loc[signal_index, "adj_close"])
        benchmark_symbol = str(row.get("benchmark", DEFAULT_BENCHMARK)).upper().strip()
        benchmark_prices = normalized_benchmarks.get(benchmark_symbol)
        benchmark_signal_close = _asof_close(benchmark_prices, signal_date)

        last_index = len(prices) - 1
        for window in FORWARD_WINDOWS:
            target_index = signal_index + window
            if target_index > last_index:
                remaining = int(target_index - last_index)
                updated.at[index, f"outcome_status_{window}d"] = "pending"
                updated.at[index, f"trading_days_remaining_{window}d"] = remaining
                updated.at[index, f"estimated_review_date_{window}d"] = _estimated_business_date(
                    prices.loc[last_index, "date"], remaining
                )
                continue

            outcome_date = pd.Timestamp(prices.loc[target_index, "date"])
            target_price = float(prices.loc[target_index, "adj_close"])
            underlying_return = target_price / signal_price - 1.0
            forward_return = direction * underlying_return

            benchmark_return = None
            excess_return = None
            benchmark_outcome_close = _asof_close(benchmark_prices, outcome_date)
            if (
                benchmark_signal_close is not None
                and benchmark_outcome_close is not None
                and benchmark_signal_close > 0
            ):
                benchmark_return = benchmark_outcome_close / benchmark_signal_close - 1.0
                excess_return = direction * (underlying_return - benchmark_return)

            updated.at[index, f"underlying_forward_return_{window}d"] = round(
                float(underlying_return), 6
            )
            updated.at[index, f"forward_return_{window}d"] = round(float(forward_return), 6)
            updated.at[index, f"benchmark_return_{window}d"] = (
                round(float(benchmark_return), 6) if benchmark_return is not None else np.nan
            )
            updated.at[index, f"excess_return_{window}d"] = (
                round(float(excess_return), 6) if excess_return is not None else np.nan
            )
            updated.at[index, f"outcome_date_{window}d"] = outcome_date.date().isoformat()
            updated.at[index, f"trading_days_remaining_{window}d"] = 0
            updated.at[index, f"estimated_review_date_{window}d"] = outcome_date.date().isoformat()
            if excess_return is not None:
                updated.at[index, f"outcome_status_{window}d"] = (
                    "beat_benchmark" if excess_return > 0 else "lagged_benchmark"
                )
            else:
                # No benchmark data: fall back to absolute direction so the row is not blank.
                updated.at[index, f"outcome_status_{window}d"] = (
                    "up_no_benchmark" if forward_return > 0 else "down_no_benchmark"
                )
    return updated


def _normalize_price_frame(prices: pd.DataFrame) -> pd.DataFrame:
    if prices is None or prices.empty:
        return pd.DataFrame(columns=["date", "adj_close"])
    frame = prices.copy()
    frame.columns = [str(column).lower() for column in frame.columns]
    if "adj_close" not in frame.columns and "close" in frame.columns:
        frame["adj_close"] = frame["close"]
    if "date" not in frame.columns or "adj_close" not in frame.columns:
        return pd.DataFrame(columns=["date", "adj_close"])
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["adj_close"] = pd.to_numeric(frame["adj_close"], errors="coerce")
    frame = frame.dropna(subset=["date", "adj_close"])
    # One row per date and a clean 0..n-1 index: forward outcomes count trading days
    # by position (signal_index + window), so duplicate dates would skew the window.
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    return frame.reset_index(drop=True)


def _asof_close(prices: pd.DataFrame | None, target_date: pd.Timestamp) -> float | None:
    if prices is None or prices.empty:
        return None
    eligible = prices[prices["date"] <= pd.Timestamp(target_date).normalize()]
    if eligible.empty:
        return None
    return float(eligible.iloc[-1]["adj_close"])


def _estimated_business_date(start_date: object, remaining_days: int) -> str:
    if remaining_days <= 0:
        return pd.Timestamp(start_date).date().isoformat()
    estimated = pd.Timestamp(start_date) + pd.tseries.offsets.BDay(remaining_days)
    return estimated.date().isoformat()


def _safe_finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _build_human_trade_row(
    ticker: str,
    action: str,
    weight_pct: float,
    signal_price: float,
    signal_date: str | pd.Timestamp | None,
    reason: str,
    benchmark: str,
    report_path: str | Path | None,
) -> dict[str, object]:
    clean_ticker = str(ticker or "").upper().strip()
    if not clean_ticker:
        raise ValueError("Ticker cannot be empty. / 股票代码不能为空。")

    clean_action = str(action or "").upper().strip()
    if clean_action not in VALID_ACTIONS:
        allowed = ", ".join(VALID_ACTIONS)
        raise ValueError(
            f"Action must be one of {allowed}. / 操作必须是其中之一：{allowed}。"
        )

    weight_value = _require_finite_float(weight_pct, "weight_pct / 仓位百分比")
    if not 0.0 <= weight_value <= 100.0:
        raise ValueError("weight_pct must be between 0 and 100. / 仓位百分比必须在0到100之间。")

    price_value = _require_finite_float(signal_price, "signal_price / 信号价格")
    if price_value <= 0.0:
        raise ValueError("signal_price must be positive. / 信号价格必须大于0。")

    clean_benchmark = str(benchmark or DEFAULT_BENCHMARK).upper().strip()
    if clean_benchmark not in VALID_BENCHMARKS:
        allowed = ", ".join(VALID_BENCHMARKS)
        raise ValueError(
            f"Benchmark must be one of {allowed}. / 基准必须是其中之一：{allowed}。"
        )

    resolved_date = _normalize_signal_date(signal_date)

    return {
        "created_at_utc": _utc_now(),
        "signal_date": resolved_date,
        "ticker": clean_ticker,
        "actor": HUMAN_ACTOR,
        "action": clean_action,
        "weight_pct": round(weight_value, 4),
        "signal_price": round(price_value, 6),
        "benchmark": clean_benchmark,
        "reason": str(reason or "").strip(),
        "report_path": str(report_path or ""),
    }


def _normalize_signal_date(signal_date: str | pd.Timestamp | None) -> str:
    if signal_date is None or (isinstance(signal_date, str) and not signal_date.strip()):
        return pd.Timestamp.today().normalize().date().isoformat()
    parsed = pd.to_datetime(signal_date, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(
            f"signal_date could not be parsed: {signal_date!r}. / 无法解析信号日期。"
        )
    return pd.Timestamp(parsed).normalize().date().isoformat()


def _deduplicate_human_trades(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    keys = ["ticker", "signal_date", "action"]
    for column in keys:
        if column not in frame.columns:
            frame[column] = ""
    return frame.drop_duplicates(keys, keep="last").reset_index(drop=True)


def _order_columns(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = list(CORE_COLUMNS)
    extras = [column for column in frame.columns if column not in ordered]
    for column in ordered:
        if column not in frame.columns:
            frame[column] = "" if column in {"reason", "report_path", "benchmark", "actor", "action"} else np.nan
    return frame[ordered + extras]


def _require_finite_float(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a number. / {label} 必须是数字。") from None
    if not np.isfinite(number):
        raise ValueError(f"{label} must be a finite number. / {label} 必须是有限数字。")
    return number


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
