from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PaperTradingConfig


CASH_TICKER = "CASH"


@dataclass(frozen=True)
class PaperTradeResult:
    as_of_date: pd.Timestamp
    orders: pd.DataFrame
    pre_trade_state: pd.DataFrame
    post_trade_state: pd.DataFrame
    summary: dict[str, float | str]
    report: str


@dataclass(frozen=True)
class PaperTargetConfig:
    max_positions: int = 5
    max_position_weight: float = 0.20
    min_target_weight: float = 0.02
    cash_reserve_weight: float = 0.10
    allow_near_watchlist: bool = False
    base_signal_score_min: float = 65.0
    base_confidence_score_min: float = 65.0
    base_high_probability_score_min: float = 65.0


def build_paper_targets_from_analysis(
    analysis: pd.DataFrame,
    config: PaperTargetConfig | None = None,
    as_of_date: str | pd.Timestamp | None = None,
    market_regime_policy: pd.DataFrame | None = None,
) -> pd.DataFrame:
    target_config = config or PaperTargetConfig()
    if analysis.empty:
        return _empty_paper_targets()

    policy_lookup = _market_regime_policy_lookup(market_regime_policy)
    rows: list[dict[str, object]] = []
    for row in analysis.to_dict(orient="records"):
        target_row = _analysis_row_to_paper_candidate(
            row,
            target_config,
            as_of_date,
            policy_lookup,
        )
        if target_row is not None:
            rows.append(target_row)

    if not rows:
        return _empty_paper_targets()

    targets = pd.DataFrame(rows)
    targets = targets.sort_values(
        by=["paper_quality_gate_passed", "paper_score", "ticker"],
        ascending=[False, False, True],
    ).head(max(int(target_config.max_positions), 0))
    targets = targets.reset_index(drop=True)
    targets["paper_rank"] = targets.index + 1
    weights = _capped_target_weights(
        scores=targets["paper_score"].astype(float).to_numpy(),
        investable_weight=max(0.0, 1.0 - float(target_config.cash_reserve_weight)),
        max_position_weight=max(0.0, float(target_config.max_position_weight)),
    )
    targets["weight"] = weights
    targets["target_weight"] = weights
    targets["paper_target_weight"] = weights
    if "market_regime_position_scale" in targets.columns:
        scale = pd.to_numeric(targets["market_regime_position_scale"], errors="coerce").fillna(1.0)
        targets["weight"] = targets["weight"] * scale
        targets["target_weight"] = targets["target_weight"] * scale
        targets["paper_target_weight"] = targets["paper_target_weight"] * scale
    targets = targets[targets["paper_target_weight"] >= target_config.min_target_weight]
    if targets.empty:
        return _empty_paper_targets()
    targets["paper_rank"] = range(1, len(targets) + 1)
    return targets.reset_index(drop=True)


def load_portfolio_state(
    path: str | Path | None,
    initial_cash: float,
) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        return pd.DataFrame([{"ticker": CASH_TICKER, "quantity": float(initial_cash)}])

    state = pd.read_csv(path)
    state.columns = [column.strip().lower() for column in state.columns]
    missing = {"ticker", "quantity"}.difference(state.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Missing required portfolio state columns: {missing_text}")

    state["ticker"] = state["ticker"].astype("string").str.upper().str.strip()
    state["quantity"] = pd.to_numeric(state["quantity"], errors="coerce")
    state = state.dropna(subset=["ticker", "quantity"])
    if CASH_TICKER not in set(state["ticker"]):
        state = pd.concat(
            [pd.DataFrame([{"ticker": CASH_TICKER, "quantity": 0.0}]), state],
            ignore_index=True,
        )
    return state.groupby("ticker", as_index=False)["quantity"].sum()


def build_latest_price_table(prices: pd.DataFrame, as_of_date: str | None = None) -> pd.DataFrame:
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if as_of_date is not None:
        frame = frame[frame["date"] <= pd.Timestamp(as_of_date)]
    if frame.empty:
        raise ValueError("No prices available for paper trading.")

    latest_date = frame["date"].max()
    latest = frame[frame["date"] == latest_date][["date", "ticker", "adj_close"]].copy()
    latest = latest.rename(columns={"adj_close": "price"})
    latest["ticker"] = latest["ticker"].astype(str).str.upper()
    return latest.sort_values("ticker").reset_index(drop=True)


def run_paper_rebalance(
    selections: pd.DataFrame,
    prices: pd.DataFrame,
    state: pd.DataFrame,
    config: PaperTradingConfig,
    as_of_date: str | None = None,
) -> PaperTradeResult:
    latest_prices = build_latest_price_table(prices, as_of_date=as_of_date)
    price_date = pd.Timestamp(latest_prices["date"].max())
    target_weights = _latest_target_weights(selections, price_date)
    orders, post_state, summary = _rebalance_to_targets(
        target_weights=target_weights,
        latest_prices=latest_prices,
        state=state,
        config=config,
    )
    report = render_paper_trade_report(
        as_of_date=price_date,
        orders=orders,
        pre_trade_state=state,
        post_trade_state=post_state,
        summary=summary,
    )
    return PaperTradeResult(
        as_of_date=price_date,
        orders=orders,
        pre_trade_state=state,
        post_trade_state=post_state,
        summary=summary,
        report=report,
    )


def _latest_target_weights(selections: pd.DataFrame, price_date: pd.Timestamp) -> pd.DataFrame:
    if selections.empty:
        return pd.DataFrame(columns=["ticker", "target_weight"])
    frame = selections.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    eligible = frame[frame["date"] <= price_date]
    if eligible.empty:
        return pd.DataFrame(columns=["ticker", "target_weight"])
    latest_date = eligible["date"].max()
    latest = eligible[eligible["date"] == latest_date][["ticker", "weight"]].copy()
    latest["ticker"] = latest["ticker"].astype(str).str.upper()
    latest = latest.rename(columns={"weight": "target_weight"})
    return latest.groupby("ticker", as_index=False)["target_weight"].sum()


def _rebalance_to_targets(
    target_weights: pd.DataFrame,
    latest_prices: pd.DataFrame,
    state: pd.DataFrame,
    config: PaperTradingConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | str]]:
    price_map = latest_prices.set_index("ticker")["price"].astype(float)
    positions = _state_to_positions(state)
    cash = float(positions.pop(CASH_TICKER, 0.0))

    # Unpriced holdings must remain visible because silently valuing them at zero
    # would corrupt both portfolio value and the next allocation.
    unpriced_positions = sorted(
        ticker
        for ticker, quantity in positions.items()
        if ticker not in price_map.index and abs(quantity) > 1e-9
    )
    current_values = {
        ticker: quantity * float(price_map.get(ticker, np.nan))
        for ticker, quantity in positions.items()
        if ticker in price_map.index
    }
    portfolio_value = cash + sum(current_values.values())
    if portfolio_value <= 0:
        raise ValueError("Paper portfolio value must be positive.")

    targets = target_weights.set_index("ticker")["target_weight"].astype(float)
    order_rows: list[dict[str, object]] = []
    all_tickers = sorted(set(current_values) | set(targets.index))
    for ticker in all_tickers:
        if ticker not in price_map.index:
            continue
        price = float(price_map[ticker])
        target_value = portfolio_value * float(targets.get(ticker, 0.0))
        current_value = float(current_values.get(ticker, 0.0))
        delta_value = target_value - current_value
        if abs(delta_value) < config.min_trade_value:
            continue
        if abs(delta_value) / portfolio_value < config.trade_buffer_pct:
            continue
        # Bid-ask spread: buys fill above the close, sells below it. Sizing and cash
        # use this realistic fill price; the position is later marked at the close,
        # so the spread shows up as a real cost (not the optimistic mid/close fill).
        direction = 1.0 if delta_value > 0 else -1.0
        half_spread = config.spread_bps / 10_000.0
        fill_price = price * (1.0 + direction * half_spread)
        quantity = delta_value / fill_price
        if quantity < 0:
            # A sell fill below the close can otherwise calculate slightly more
            # shares than the holding during full liquidation. Never cross short.
            quantity = -min(abs(quantity), float(positions.get(ticker, 0.0)))
        if not config.allow_fractional_shares:
            quantity = np.sign(quantity) * np.floor(abs(quantity))
        if abs(quantity) <= 0:
            continue
        action = "BUY" if quantity > 0 else "SELL"
        notional = abs(quantity * fill_price)
        spread_cost = abs(quantity) * abs(fill_price - price)
        slippage_cost = notional * config.slippage_bps / 10_000.0
        commission_cost = notional * config.commission_bps / 10_000.0
        estimated_cost = slippage_cost + commission_cost
        order_rows.append(
            {
                "ticker": ticker,
                "action": action,
                "quantity": abs(float(quantity)),
                "signed_quantity": float(quantity),
                "price": float(fill_price),
                "close_price": float(price),
                "notional": float(notional),
                "spread_cost": float(spread_cost),
                "slippage_cost": float(slippage_cost),
                "commission_cost": float(commission_cost),
                "estimated_cost": float(estimated_cost),
                "total_transaction_cost": float(spread_cost + estimated_cost),
                "target_weight": float(targets.get(ticker, 0.0)),
                "current_weight": current_value / portfolio_value,
                "target_value": target_value,
                "current_value": current_value,
            }
        )

    orders = pd.DataFrame(order_rows)
    if orders.empty:
        post_state = _positions_to_state(positions, cash)
        summary = _paper_summary(
            portfolio_value, portfolio_value, cash, 0.0, 0, unpriced_positions
        )
        return orders, post_state, summary

    orders = _apply_cash_constraint(orders, cash)
    orders = _add_order_balance_audit(orders, positions, cash)
    post_positions, post_cash = _apply_orders(positions, cash, orders)
    post_value = post_cash + sum(
        quantity * float(price_map.get(ticker, 0.0))
        for ticker, quantity in post_positions.items()
    )
    post_state = _positions_to_state(post_positions, post_cash)
    summary = _paper_summary(
        pre_trade_value=portfolio_value,
        post_trade_value=post_value,
        post_trade_cash=post_cash,
        total_estimated_cost=float(orders["estimated_cost"].sum()),
        order_count=len(orders),
        unpriced_positions=unpriced_positions,
        total_spread_cost=float(orders["spread_cost"].sum()),
        total_slippage_cost=float(orders["slippage_cost"].sum()),
        total_commission_cost=float(orders["commission_cost"].sum()),
        total_transaction_cost=float(orders["total_transaction_cost"].sum()),
    )
    return orders, post_state, summary


def _apply_cash_constraint(orders: pd.DataFrame, cash: float) -> pd.DataFrame:
    sells = orders[orders["action"] == "SELL"].copy()
    buys = orders[orders["action"] == "BUY"].copy()
    sell_proceeds = float((sells["notional"] - sells["estimated_cost"]).sum())
    available_cash = cash + sell_proceeds
    buy_cost = float((buys["notional"] + buys["estimated_cost"]).sum())
    if buy_cost <= available_cash or buys.empty:
        return pd.concat([sells, buys], ignore_index=True)

    scale = available_cash / buy_cost if buy_cost > 0 else 0.0
    scale_columns = [
        "quantity",
        "signed_quantity",
        "notional",
        "spread_cost",
        "slippage_cost",
        "commission_cost",
        "estimated_cost",
        "total_transaction_cost",
    ]
    buys[scale_columns] = buys[scale_columns] * scale
    return pd.concat([sells, buys], ignore_index=True)


def _add_order_balance_audit(
    orders: pd.DataFrame,
    positions: dict[str, float],
    cash: float,
) -> pd.DataFrame:
    if orders.empty:
        return orders
    audited = orders.copy().reset_index(drop=True)
    running_positions = dict(positions)
    running_cash = float(cash)
    before_quantities: list[float] = []
    after_quantities: list[float] = []
    cash_before: list[float] = []
    cash_after: list[float] = []
    for order in audited.itertuples(index=False):
        ticker = str(order.ticker)
        signed_quantity = float(order.signed_quantity)
        position_before = float(running_positions.get(ticker, 0.0))
        next_position = position_before + signed_quantity
        next_cash = (
            running_cash
            - signed_quantity * float(order.price)
            - float(order.estimated_cost)
        )
        before_quantities.append(position_before)
        after_quantities.append(next_position)
        cash_before.append(running_cash)
        cash_after.append(next_cash)
        running_positions[ticker] = next_position
        running_cash = next_cash
    audited["position_quantity_before"] = before_quantities
    audited["position_quantity_after"] = after_quantities
    audited["cash_before"] = cash_before
    audited["cash_after"] = cash_after
    return audited


def _apply_orders(
    positions: dict[str, float],
    cash: float,
    orders: pd.DataFrame,
) -> tuple[dict[str, float], float]:
    updated = dict(positions)
    updated_cash = cash
    for order in orders.itertuples(index=False):
        signed_quantity = float(order.signed_quantity)
        updated[order.ticker] = updated.get(order.ticker, 0.0) + signed_quantity
        cash_delta = -signed_quantity * float(order.price)
        updated_cash += cash_delta - float(order.estimated_cost)
    updated = {ticker: quantity for ticker, quantity in updated.items() if abs(quantity) > 1e-9}
    return updated, updated_cash


def _state_to_positions(state: pd.DataFrame) -> dict[str, float]:
    frame = state.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["quantity"] = pd.to_numeric(frame["quantity"], errors="coerce").fillna(0.0)
    return frame.groupby("ticker")["quantity"].sum().to_dict()


def _positions_to_state(positions: dict[str, float], cash: float) -> pd.DataFrame:
    rows = [{"ticker": CASH_TICKER, "quantity": float(cash)}]
    for ticker, quantity in sorted(positions.items()):
        rows.append({"ticker": ticker, "quantity": float(quantity)})
    return pd.DataFrame(rows)


def _paper_summary(
    pre_trade_value: float,
    post_trade_value: float,
    post_trade_cash: float,
    total_estimated_cost: float,
    order_count: int,
    unpriced_positions: list[str] | None = None,
    total_spread_cost: float = 0.0,
    total_slippage_cost: float = 0.0,
    total_commission_cost: float = 0.0,
    total_transaction_cost: float = 0.0,
) -> dict[str, float | str]:
    summary: dict[str, float | str] = {
        "pre_trade_value": float(pre_trade_value),
        "post_trade_value": float(post_trade_value),
        "post_trade_cash": float(post_trade_cash),
        "cash_weight": float(post_trade_cash / post_trade_value) if post_trade_value else 0.0,
        "total_estimated_cost": float(total_estimated_cost),
        "total_spread_cost": float(total_spread_cost),
        "total_slippage_cost": float(total_slippage_cost),
        "total_commission_cost": float(total_commission_cost),
        "total_transaction_cost": float(total_transaction_cost),
        "order_count": float(order_count),
    }
    if unpriced_positions:
        # Unpriced holdings are excluded from this run's valuation and trading.
        summary["unpriced_positions"] = ", ".join(unpriced_positions)
        summary["unpriced_positions_note_zh"] = (
            "以上持仓本次没有报价，无法估值或卖出；组合价值未包含它们，请检查数据源。"
        )
    return summary


def render_paper_trade_report(
    as_of_date: pd.Timestamp,
    orders: pd.DataFrame,
    pre_trade_state: pd.DataFrame,
    post_trade_state: pd.DataFrame,
    summary: dict[str, float | str],
) -> str:
    lines = [
        "# Paper Trade Rebalance Report / 纸面模拟交易再平衡报告",
        "",
        "This is a local simulation report, not a live brokerage order ticket.",
        "这是本地模拟报告，不是真实券商订单，也不会自动下单。",
        "",
        f"Date / 日期: `{as_of_date.date().isoformat()}`",
        "",
        "## Summary / 摘要",
        "",
    ]
    for key, value in summary.items():
        lines.append(f"- `{key}`: {_format_value(value)}")
    lines.extend(["", "## Pre Trade State / 模拟交易前仓位", ""])
    lines.extend(_markdown_table(pre_trade_state))
    lines.extend(["", "## Orders / 模拟订单", ""])
    lines.extend(_markdown_table(orders))
    lines.extend(["", "## Post Trade State / 模拟交易后仓位", ""])
    lines.extend(_markdown_table(post_trade_state))
    return "\n".join(lines).rstrip() + "\n"


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
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _analysis_row_to_paper_candidate(
    row: dict[str, object],
    config: PaperTargetConfig,
    as_of_date: str | pd.Timestamp | None,
    policy_lookup: dict[str, dict[str, object]] | None = None,
) -> dict[str, object] | None:
    ticker = _safe_str(row.get("ticker")).upper()
    if not ticker:
        return None

    gate_passed = _safe_bool(
        _first_existing(row, ["calibrated_quality_gate_passed", "quality_gate_passed"])
    )
    watchlist_status = _safe_str(
        _first_existing(row, ["calibrated_watchlist_status", "watchlist_status"])
    )
    overall_risk_level = _safe_str(row.get("overall_risk_level")).lower()
    data_quality_score = _safe_float(row.get("data_quality_score"), default=100.0)
    confidence_score = _safe_float(row.get("confidence_score"), default=100.0)
    high_probability_score = _safe_float(
        _first_existing(
            row,
            ["calibrated_high_probability_score", "high_probability_score"],
        ),
        default=np.nan,
    )
    signal_score = _safe_float(row.get("signal_score"), default=np.nan)
    market_policy = _resolve_market_regime_policy(row, policy_lookup or {})

    eligible = bool(gate_passed) or (
        bool(config.allow_near_watchlist) and watchlist_status == "close_but_not_ready"
    )
    if not eligible:
        return None
    if not _market_policy_allows_candidate(
        gate_passed=gate_passed,
        watchlist_status=watchlist_status,
        signal_score=signal_score,
        confidence_score=confidence_score,
        high_probability_score=high_probability_score,
        config=config,
        policy=market_policy,
    ):
        return None
    if overall_risk_level == "high":
        return None
    if data_quality_score < 70 or confidence_score < 65:
        return None

    score = _paper_score(row)
    if score <= 0:
        return None

    latest_price = _safe_float(row.get("latest_price"), default=np.nan)
    date_value = _row_date(row, as_of_date)
    reason, reason_zh = _paper_reason(row, gate_passed)
    focus_horizon = _safe_str(_first_existing(row, ["final_focus_horizon", "focus_horizon"]))

    return {
        "date": date_value,
        "ticker": ticker,
        "weight": 0.0,
        "target_weight": 0.0,
        "paper_target_weight": 0.0,
        "paper_rank": 0,
        "paper_score": score,
        "paper_quality_gate_passed": bool(gate_passed),
        "paper_eligibility": "selected",
        "paper_eligibility_zh": "进入模拟目标",
        "paper_reason": reason,
        "paper_reason_zh": reason_zh,
        "latest_price": latest_price,
        "final_decision": _safe_str(row.get("final_decision")),
        "final_decision_zh": _safe_str(row.get("final_decision_zh")),
        "focus_horizon": focus_horizon,
        "screening_profile": _safe_str(row.get("screening_profile")),
        "screening_profile_zh": _safe_str(row.get("screening_profile_zh")),
        "quality_gate_passed": bool(gate_passed),
        "high_probability_score": high_probability_score,
        "calibrated_win_probability": _safe_float(
            row.get("calibrated_win_probability"),
            default=np.nan,
        ),
        "overall_risk_level": overall_risk_level,
        "data_quality_score": data_quality_score,
        "confidence_score": confidence_score,
        "watchlist_status": watchlist_status,
        "watchlist_status_zh": _safe_str(
            _first_existing(
                row,
                ["calibrated_watchlist_status_zh", "watchlist_status_zh"],
            )
        ),
        "market_regime": market_policy["market_regime"],
        "market_regime_policy_action": market_policy["protection_action"],
        "market_regime_policy_action_zh": market_policy["protection_action_zh"],
        "market_regime_allowed_signal_bucket": market_policy["allowed_signal_bucket"],
        "market_regime_position_scale": market_policy["position_scale"],
        "market_regime_policy_note_zh": market_policy["policy_note_zh"],
        "report_path": _safe_str(row.get("report_path")),
    }


def _market_policy_allows_candidate(
    gate_passed: bool,
    watchlist_status: str,
    signal_score: float,
    confidence_score: float,
    high_probability_score: float,
    config: PaperTargetConfig,
    policy: dict[str, object],
) -> bool:
    if not bool(policy.get("allow_new_entries", True)):
        return False
    bucket = str(policy.get("allowed_signal_bucket", "normal"))
    if bucket == "none":
        return False
    if bucket == "strict_high_probability_only" and not gate_passed:
        return False
    if bucket == "high_probability_or_near_only" and not (
        gate_passed or watchlist_status == "close_but_not_ready"
    ):
        return False

    signal_delta = _safe_float(policy.get("signal_score_delta"), default=0.0)
    confidence_delta = _safe_float(policy.get("confidence_score_delta"), default=0.0)
    high_probability_delta = _safe_float(
        policy.get("high_probability_score_delta"),
        default=0.0,
    )
    if _is_finite_number(signal_score) and signal_score < config.base_signal_score_min + signal_delta:
        return False
    if (
        _is_finite_number(confidence_score)
        and confidence_score < config.base_confidence_score_min + confidence_delta
    ):
        return False
    if (
        _is_finite_number(high_probability_score)
        and high_probability_score < config.base_high_probability_score_min + high_probability_delta
    ):
        return False
    return True


def _market_regime_policy_lookup(
    policy: pd.DataFrame | None,
) -> dict[str, dict[str, object]]:
    if policy is None or policy.empty or "validation_market_regime" not in policy.columns:
        return {}
    lookup: dict[str, dict[str, object]] = {}
    for row in policy.to_dict(orient="records"):
        regime = _normalise_market_regime(row.get("validation_market_regime"))
        if not regime:
            continue
        lookup[regime] = {
            "market_regime": regime,
            "protection_action": _safe_str(row.get("protection_action")) or "normal_rules",
            "protection_action_zh": _safe_str(row.get("protection_action_zh")) or "使用正常规则",
            "allow_new_entries": _safe_bool_default(row.get("allow_new_entries"), True),
            "allowed_signal_bucket": _safe_str(row.get("allowed_signal_bucket")) or "normal",
            "signal_score_delta": _safe_float(row.get("signal_score_delta"), default=0.0),
            "confidence_score_delta": _safe_float(row.get("confidence_score_delta"), default=0.0),
            "high_probability_score_delta": _safe_float(
                row.get("high_probability_score_delta"),
                default=0.0,
            ),
            "position_scale": max(
                0.0,
                min(1.0, _safe_float(row.get("position_scale"), default=1.0)),
            ),
            "policy_note_zh": _safe_str(row.get("policy_note_zh")),
        }
    return lookup


def _resolve_market_regime_policy(
    row: dict[str, object],
    policy_lookup: dict[str, dict[str, object]],
) -> dict[str, object]:
    raw_regime = _first_existing(
        row,
        [
            "validation_market_regime",
            "current_market_regime",
            "market_regime",
        ],
    )
    regime = _normalise_market_regime(raw_regime)
    policy = policy_lookup.get(regime, _default_market_policy(regime))
    return {**_default_market_policy(regime), **policy}


def _default_market_policy(regime: str) -> dict[str, object]:
    return {
        "market_regime": regime or "unknown",
        "protection_action": "normal_rules",
        "protection_action_zh": "使用正常规则",
        "allow_new_entries": True,
        "allowed_signal_bucket": "normal",
        "signal_score_delta": 0.0,
        "confidence_score_delta": 0.0,
        "high_probability_score_delta": 0.0,
        "position_scale": 1.0,
        "policy_note_zh": "没有匹配的市场状态保护规则，使用正常规则。",
    }


def _normalise_market_regime(value: object) -> str:
    text = _safe_str(value).lower().strip()
    mapping = {
        "supportive": "bull_uptrend",
        "strong": "bull_uptrend",
        "bull": "bull_uptrend",
        "weak": "bear_downtrend",
        "bear": "bear_downtrend",
        "neutral": "sideways_mixed",
        "mixed": "sideways_mixed",
        "sideways": "sideways_mixed",
        "volatile": "high_volatility",
        "high_vol": "high_volatility",
    }
    return mapping.get(text, text)


def _paper_score(row: dict[str, object]) -> float:
    win_probability = _safe_float(row.get("calibrated_win_probability"), default=np.nan)
    high_probability_score = _safe_float(
        _first_existing(row, ["calibrated_high_probability_score", "high_probability_score"]),
        default=np.nan,
    )
    backtest_trust_score = _safe_float(row.get("backtest_trust_score"), default=np.nan)
    signal_score = _safe_float(row.get("signal_score"), default=np.nan)
    confidence_score = _safe_float(row.get("confidence_score"), default=np.nan)
    data_quality_score = _safe_float(row.get("data_quality_score"), default=np.nan)

    probability_component = win_probability if np.isfinite(win_probability) else np.nan
    if not np.isfinite(probability_component):
        probability_component = high_probability_score / 100.0 if np.isfinite(high_probability_score) else 0.50

    quality_component = _bounded_fraction(high_probability_score, default=60.0)
    trust_component = _bounded_fraction(backtest_trust_score, default=70.0)
    signal_component = _bounded_fraction(signal_score, default=65.0)
    confidence_component = _bounded_fraction(confidence_score, default=70.0)
    data_component = _bounded_fraction(data_quality_score, default=80.0)
    score = (
        probability_component * 0.35
        + quality_component * 0.25
        + trust_component * 0.15
        + signal_component * 0.10
        + confidence_component * 0.10
        + data_component * 0.05
    )
    return float(max(0.0, min(1.0, score)))


def _capped_target_weights(
    scores: np.ndarray,
    investable_weight: float,
    max_position_weight: float,
) -> np.ndarray:
    if len(scores) == 0 or investable_weight <= 0 or max_position_weight <= 0:
        return np.zeros(len(scores), dtype=float)

    scores = np.nan_to_num(scores.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    scores = np.maximum(scores, 0.0)
    if float(scores.sum()) <= 0:
        scores = np.ones(len(scores), dtype=float)

    target_total = min(float(investable_weight), float(max_position_weight) * len(scores))
    weights = np.zeros(len(scores), dtype=float)
    remaining = np.ones(len(scores), dtype=bool)
    remaining_total = target_total

    while remaining.any() and remaining_total > 1e-12:
        remaining_scores = scores[remaining]
        if float(remaining_scores.sum()) <= 0:
            proposed = np.full(remaining_scores.shape, remaining_total / len(remaining_scores))
        else:
            proposed = remaining_scores / remaining_scores.sum() * remaining_total
        capped = proposed > max_position_weight
        remaining_indices = np.where(remaining)[0]
        if not capped.any():
            weights[remaining_indices] = proposed
            break
        capped_indices = remaining_indices[capped]
        weights[capped_indices] = max_position_weight
        remaining[capped_indices] = False
        remaining_total = target_total - float(weights.sum())

    return np.round(weights, 8)


def _paper_reason(row: dict[str, object], gate_passed: bool) -> tuple[str, str]:
    horizon = _safe_str(_first_existing(row, ["final_focus_horizon", "focus_horizon"])) or "focus"
    profile = _safe_str(row.get("screening_profile")) or "default"
    if gate_passed:
        return (
            f"{horizon} signal passed strict high-probability gate under {profile} profile.",
            f"{horizon} 周期信号通过严格高概率筛选，使用 {profile} 大类规则。",
        )
    return (
        f"{horizon} signal is near watchlist and allowed by paper target settings.",
        f"{horizon} 周期信号接近观察名单，本次模拟设置允许纳入。",
    )


def _row_date(row: dict[str, object], as_of_date: str | pd.Timestamp | None) -> pd.Timestamp:
    if as_of_date is not None:
        return pd.Timestamp(as_of_date).normalize()
    value = _first_existing(row, ["date", "analysis_date", "latest_date"])
    if value is not None and not _is_missing(value):
        return pd.Timestamp(value).normalize()
    return pd.Timestamp.today().normalize()


def _empty_paper_targets() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "ticker",
            "weight",
            "target_weight",
            "paper_target_weight",
            "paper_rank",
            "paper_score",
            "paper_quality_gate_passed",
            "paper_eligibility",
            "paper_eligibility_zh",
            "paper_reason",
            "paper_reason_zh",
            "latest_price",
            "final_decision",
            "final_decision_zh",
            "focus_horizon",
            "screening_profile",
            "screening_profile_zh",
            "quality_gate_passed",
            "high_probability_score",
            "calibrated_win_probability",
            "overall_risk_level",
            "data_quality_score",
            "confidence_score",
            "watchlist_status",
            "watchlist_status_zh",
            "market_regime",
            "market_regime_policy_action",
            "market_regime_policy_action_zh",
            "market_regime_allowed_signal_bucket",
            "market_regime_position_scale",
            "market_regime_policy_note_zh",
            "report_path",
        ]
    )


def _first_existing(row: dict[str, object], columns: list[str]) -> object | None:
    for column in columns:
        value = row.get(column)
        if not _is_missing(value):
            return value
    return None


def _safe_bool(value: object) -> bool:
    if _is_missing(value):
        return False
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "passed"}


def _safe_bool_default(value: object, default: bool) -> bool:
    if _is_missing(value):
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "passed"}:
        return True
    if text in {"0", "false", "no", "n", "failed"}:
        return False
    return default


def _safe_float(value: object, default: float = 0.0) -> float:
    if _is_missing(value):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _is_finite_number(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _bounded_fraction(value: object, default: float) -> float:
    numeric = _safe_float(value, default=default)
    if not np.isfinite(numeric):
        numeric = default
    return float(max(0.0, min(1.0, numeric / 100.0)))


def _safe_str(value: object) -> str:
    if _is_missing(value):
        return ""
    return str(value).strip()


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False
