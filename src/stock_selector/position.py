"""Position-aware context from the user's real holdings (``portfolio.csv``).

When a ticker is actually held, generic "should I enter" analysis is the wrong
frame — the relevant questions are cost basis (in profit or loss?), position
weight (concentration risk), and where a stop sits versus the entry. This module
turns the portfolio file into that context. It states facts and risk, not buy/sell
advice — the decision stays with the user.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class PositionContext:
    ticker: str
    is_held: bool
    shares: float | None = None
    avg_cost: float | None = None
    latest_price: float | None = None
    market_value: float | None = None
    weight_pct: float | None = None
    unrealized_pl: float | None = None
    unrealized_pl_pct: float | None = None
    concentration_level: str = "none"
    concentration_level_zh: str = "无"
    position_note: str = ""
    position_note_zh: str = ""
    next_step: str = ""
    next_step_zh: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_position_context(
    ticker: str,
    latest_price: float | None = None,
    portfolio_path: str | Path = "portfolio.csv",
    weight_pct_override: float | None = None,
) -> PositionContext:
    ticker = str(ticker).upper().strip()
    not_held = PositionContext(
        ticker=ticker,
        is_held=False,
        position_note="Not a current holding; standard entry analysis applies.",
        position_note_zh="非当前持仓；按常规买入分析。",
    )
    path = Path(portfolio_path)
    if not path.exists():
        return not_held
    try:
        frame = pd.read_csv(path)
    except Exception:
        return not_held
    if frame.empty or "ticker" not in frame.columns:
        return not_held
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    match = frame[frame["ticker"] == ticker]
    if match.empty or ticker == "CASH":
        return not_held

    row = match.iloc[0]
    shares = _f(row.get("shares"))
    avg_cost = _f(row.get("avg_cost"))
    # 权重优先用调用方按最新价重算的值；csv 里的 weight_pct 是手工导出时的静态快照。
    weight = weight_pct_override if weight_pct_override is not None else _f(row.get("weight_pct"))
    # Prefer today's analysis price (live snapshot / latest close) — it is refreshed
    # every run. The csv's current_price is a manual export snapshot that goes stale
    # within a day; keep it only as a fallback when no fresh price is available.
    price = _f(latest_price)
    if price is None or price <= 0:
        price = _f(row.get("current_price"))
    market_value = (
        shares * price if shares is not None and price is not None else _f(row.get("market_value"))
    )
    upl_pct = (
        price / avg_cost - 1.0
        if avg_cost is not None and price is not None and avg_cost > 0
        else None
    )
    upl = (
        (price - avg_cost) * shares
        if avg_cost is not None and price is not None and shares is not None
        else None
    )
    level, level_zh = _concentration_level(weight)
    note, note_zh = _build_note(shares, avg_cost, price, upl, upl_pct, weight, level_zh)
    next_step, next_step_zh = _build_next_step(avg_cost, upl_pct, weight, level, level_zh)

    return PositionContext(
        ticker=ticker,
        is_held=True,
        shares=shares,
        avg_cost=avg_cost,
        latest_price=price,
        market_value=round(market_value, 2) if market_value is not None else None,
        weight_pct=weight,
        unrealized_pl=round(upl, 2) if upl is not None else None,
        unrealized_pl_pct=round(upl_pct, 4) if upl_pct is not None else None,
        concentration_level=level,
        concentration_level_zh=level_zh,
        position_note=note,
        position_note_zh=note_zh,
        next_step=next_step,
        next_step_zh=next_step_zh,
    )


def _concentration_level(weight_pct: float | None) -> tuple[str, str]:
    if weight_pct is None:
        return "unknown", "未知"
    if weight_pct >= 30.0:
        return "extreme", "极高"
    if weight_pct >= 20.0:
        return "high", "高"
    if weight_pct >= 10.0:
        return "elevated", "偏高"
    return "normal", "正常"


def _build_note(
    shares: float | None,
    avg_cost: float | None,
    price: float | None,
    upl: float | None,
    upl_pct: float | None,
    weight: float | None,
    level_zh: str,
) -> tuple[str, str]:
    pl_en = f"{upl_pct:+.2%}" if upl_pct is not None else "N/A"
    pl_zh = f"{upl_pct:+.2%}" if upl_pct is not None else "未知"
    upl_en = f" (${upl:+,.0f})" if upl is not None else ""
    weight_en = f"{weight:.1f}%" if weight is not None else "N/A"
    note = (
        f"Held: {_num(shares)} sh @ avg cost {_money(avg_cost)}, now {_money(price)}; "
        f"unrealized {pl_en}{upl_en}. Position is {weight_en} of the portfolio "
        f"(concentration: {level_zh})."
    )
    note_zh = (
        f"持仓：{_num(shares)}股，均价{_money(avg_cost)}，现价{_money(price)}；"
        f"浮动盈亏{pl_zh}{upl_en}。该仓位占组合{weight_en}（集中度{level_zh}）。"
    )
    return note, note_zh


def _build_next_step(
    avg_cost: float | None,
    upl_pct: float | None,
    weight: float | None,
    level: str,
    level_zh: str,
) -> tuple[str, str]:
    """Concrete scenario options for a holder — several plans with their trade-offs,
    not a single buy/sell command. The final decision is the user's."""
    zh: list[str] = []
    en: list[str] = []
    if level in {"high", "extreme"}:
        wtxt = f"{weight:.0f}%" if weight is not None else "很大"
        zh.append(f"降集中度：分批把这只从{wtxt}降到20%以下，组合更分散、更抗波动")
        en.append(f"trim to spread risk (currently {wtxt} of the book)")
    if avg_cost is not None:
        zh.append(f"守成本：跌破成本线${avg_cost:,.2f}就考虑减仓止损，控制亏损")
        en.append(f"stop-loss reference at cost ${avg_cost:,.2f}")
    if upl_pct is not None and upl_pct <= -0.15:
        zh.append("重估逻辑：已明显浮亏，重新检视当初买入理由；逻辑不成立就离场，不要死扛")
        en.append("thesis is underwater — re-check the reason to hold or exit")
    elif upl_pct is not None and upl_pct >= 0.20:
        zh.append("锁利：浮盈较多，可考虑部分止盈、落袋一部分")
        en.append("consider taking partial profit")
    if not zh:
        zh.append("正常持有，设好止损位跟踪即可")
        en.append("hold and track with a stop in place")

    zh_text = (
        "持有观察 · 方案参考："
        + "；".join(f"{i + 1}) {item}" for i, item in enumerate(zh))
        + "。以上为参考方案，是否操作由你决定。"
    )
    en_text = "Holding — options: " + "; ".join(en) + ". Your call."
    return en_text, zh_text


def _f(value: object) -> float | None:
    try:
        if value is None:
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None  # drop NaN


def _num(value: float | None) -> str:
    return f"{value:g}" if value is not None else "N/A"


def _money(value: float | None) -> str:
    return f"${value:,.2f}" if value is not None else "N/A"
