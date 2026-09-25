from __future__ import annotations

import pandas as pd


PEER_GROUPS = {
    "software": ["NOW", "CRM", "MSFT", "ADBE", "ORCL", "INTU", "SNOW", "DDOG", "PANW", "PLTR"],
    "cloud": ["NOW", "CRM", "MSFT", "ADBE", "ORCL", "SNOW", "DDOG", "NET", "MDB"],
    "semiconductor": ["NVDA", "AMD", "AVGO", "INTC", "QCOM", "AMAT", "MU", "TSM", "ASML"],
    "consumer electronics": ["AAPL", "SONY", "HPQ", "DELL", "LOGI"],
    "internet retail": ["AMZN", "MELI", "SHOP", "EBAY", "BABA", "JD"],
    "automobiles": ["TSLA", "GM", "F", "RIVN", "LCID", "TM"],
    "building products": ["AAON", "CARR", "LII", "TT", "JCI", "IR", "ETN", "HON", "PH"],
    "hvac": ["AAON", "CARR", "LII", "TT", "JCI", "IR"],
    "banks": ["JPM", "BAC", "WFC", "C", "GS", "MS"],
    "biotechnology": ["AMGN", "GILD", "VRTX", "REGN", "BIIB", "MRNA"],
}


SECTOR_PEERS = {
    "technology": ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "ADBE", "CRM", "NOW"],
    "communication services": ["META", "GOOGL", "NFLX", "DIS", "TMUS", "VZ"],
    "consumer cyclical": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX"],
    "consumer defensive": ["WMT", "COST", "PG", "KO", "PEP", "PM"],
    "financial services": ["JPM", "BAC", "WFC", "C", "GS", "MS"],
    "healthcare": ["LLY", "UNH", "JNJ", "MRK", "ABBV", "TMO"],
    "industrials": ["GE", "CAT", "HON", "UNP", "RTX", "BA", "AAON", "CARR", "LII"],
    "energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC"],
    "basic materials": ["LIN", "SHW", "FCX", "NEM", "APD", "ECL"],
    "real estate": ["PLD", "AMT", "EQIX", "SPG", "O", "CCI"],
    "utilities": ["NEE", "SO", "DUK", "AEP", "SRE", "D"],
}


def choose_peer_tickers(
    ticker: str,
    sector: str | None,
    industry: str | None,
    max_peers: int = 6,
) -> list[str]:
    ticker = ticker.upper().strip()
    sector_text = str(sector or "").lower().strip()
    industry_text = str(industry or "").lower().strip()

    candidates: list[str] = []
    for keyword, tickers in PEER_GROUPS.items():
        if keyword in industry_text:
            candidates.extend(tickers)
    if not candidates:
        candidates.extend(SECTOR_PEERS.get(sector_text, []))

    peers: list[str] = []
    for peer in candidates:
        normalized = peer.upper().strip()
        if normalized and normalized != ticker and normalized not in peers:
            peers.append(normalized)
        if len(peers) >= max_peers:
            break
    return peers


def build_peer_comparison_frame(
    target_ticker: str,
    target_analysis: pd.DataFrame,
    target_snapshot: dict[str, object] | None,
    peer_results: list[object],
) -> pd.DataFrame:
    rows = [
        _comparison_row(
            ticker=target_ticker,
            analysis=target_analysis,
            snapshot=target_snapshot,
            comparison_role="target",
        )
    ]
    for result in peer_results:
        rows.append(
            _comparison_row(
                ticker=result.ticker,
                analysis=result.analysis,
                snapshot=result.snapshot,
                comparison_role="peer",
            )
        )

    frame = pd.DataFrame(rows)
    frame = frame.sort_values(
        ["high_probability_score", "signal_score"],
        ascending=[False, False],
    ).reset_index(drop=True)
    frame["peer_rank"] = frame.index + 1
    target_rows = frame[frame["ticker"] == target_ticker.upper().strip()]
    target_rank = int(target_rows["peer_rank"].iloc[0]) if not target_rows.empty else 0
    frame["target_peer_rank"] = target_rank
    return _attach_peer_valuation_metrics(frame, target_ticker)


def render_peer_comparison_report(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "## Peer Comparison / 同业比较\n\nNo peer comparison rows were produced.\n"

    target = frame[frame["comparison_role"] == "target"]
    target_row = target.iloc[0] if not target.empty else frame.iloc[0]
    lines = [
        "## Peer Comparison / 同业比较",
        "",
        (
            f"- Target rank / 目标排名: `{int(target_row['peer_rank'])}` "
            f"of `{len(frame)}`"
        ),
        (
            "- Target high probability score / 目标高概率分数: "
            f"`{_format_number(target_row['high_probability_score'])}`"
        ),
        (
            "- Target screening / 目标筛选结论: "
            f"`{target_row['screening_action']}` / {target_row['screening_action_zh']}"
        ),
        "- Peer note / 同业说明: higher rank means stronger current high-probability screening score.",
        "  排名越靠前，代表当前高概率筛选分数相对同业更强。",
        "",
        "## Peer Valuation / 同业估值",
        "",
        (
            "- Target valuation rank / 目标估值排名: "
            f"`{_format_number(target_row['valuation_peer_rank'])}` of `{len(frame)}`"
        ),
        (
            "- Peer valuation label / 同业估值标签: "
            f"`{target_row['peer_valuation_label']}` / {target_row['peer_valuation_label_zh']}"
        ),
        (
            "- Forward PE vs peer median / 预期市盈率相对同业中位数: "
            f"`{_format_percent(target_row['forward_pe_vs_peer_median_pct'])}`"
        ),
        (
            "- PEG vs peer median / PEG相对同业中位数: "
            f"`{_format_percent(target_row['peg_vs_peer_median_pct'])}`"
        ),
        (
            "- Peer valuation note / 同业估值说明: "
            f"{target_row['peer_valuation_note']} / {target_row['peer_valuation_note_zh']}"
        ),
        "",
        "## Peer Ranking Table / 同业排名表",
        "",
    ]
    display_columns = [
        "peer_rank",
        "valuation_peer_rank",
        "ticker",
        "comparison_role",
        "company_name",
        "sector",
        "industry",
        "focus_horizon",
        "screening_action",
        "calibrated_screening_action",
        "screening_profile",
        "high_probability_score",
        "calibrated_high_probability_score",
        "quality_gate_passed",
        "calibrated_quality_gate_passed",
        "signal_score",
        "confidence_score",
        "data_quality_score",
        "market_score",
        "relative_strength_score",
        "fundamental_score",
        "valuation_score",
        "valuation_risk_level",
        "valuation_forward_pe",
        "valuation_peg_ratio",
        "peer_valuation_label",
        "forward_pe_vs_peer_median_pct",
        "sector_score",
        "quality_gate_fail_reasons",
        "calibrated_quality_gate_fail_reasons",
    ]
    lines.extend(_markdown_table(frame[display_columns]))
    return "\n".join(lines).rstrip() + "\n"


def _comparison_row(
    ticker: str,
    analysis: pd.DataFrame,
    snapshot: dict[str, object] | None,
    comparison_role: str,
) -> dict[str, object]:
    focus = analysis.sort_values("high_probability_score", ascending=False).iloc[0]
    snapshot = snapshot or {}
    return {
        "ticker": ticker.upper().strip(),
        "comparison_role": comparison_role,
        "company_name": snapshot.get("company_name") or "",
        "sector": focus.get("sector", "") or snapshot.get("sector") or "",
        "industry": focus.get("industry", "") or snapshot.get("industry") or "",
        "focus_horizon": focus["horizon"],
        "screening_action": focus["screening_action"],
        "screening_action_zh": focus["screening_action_zh"],
        "calibrated_screening_action": focus.get(
            "calibrated_screening_action",
            focus["screening_action"],
        ),
        "calibrated_screening_action_zh": focus.get(
            "calibrated_screening_action_zh",
            focus["screening_action_zh"],
        ),
        "screening_profile": focus.get("screening_profile", "default"),
        "screening_profile_zh": focus.get("screening_profile_zh", "默认规则"),
        "high_probability_score": float(focus["high_probability_score"]),
        "calibrated_high_probability_score": float(
            focus.get("calibrated_high_probability_score", focus["high_probability_score"])
        ),
        "high_probability_level": focus["high_probability_level"],
        "quality_gate_passed": bool(focus["quality_gate_passed"]),
        "calibrated_quality_gate_passed": bool(
            focus.get("calibrated_quality_gate_passed", focus["quality_gate_passed"])
        ),
        "quality_gate_fail_reasons": focus["quality_gate_fail_reasons"],
        "quality_gate_fail_reasons_zh": focus["quality_gate_fail_reasons_zh"],
        "calibrated_quality_gate_fail_reasons": focus.get(
            "calibrated_quality_gate_fail_reasons",
            focus["quality_gate_fail_reasons"],
        ),
        "calibrated_quality_gate_fail_reasons_zh": focus.get(
            "calibrated_quality_gate_fail_reasons_zh",
            focus["quality_gate_fail_reasons_zh"],
        ),
        "signal_score": float(focus["signal_score"]),
        "confidence_score": float(focus["confidence_score"]),
        "data_quality_score": float(focus["data_quality_score"]),
        "market_score": float(focus["market_score"]),
        "relative_strength_score": float(focus["relative_strength_score"]),
        "fundamental_score": float(focus["fundamental_score"]),
        "valuation_score": _safe_float(focus.get("valuation_score")),
        "valuation_risk_level": focus.get("valuation_risk_level", "unknown"),
        "valuation_forward_pe": _safe_float(focus.get("valuation_forward_pe")),
        "valuation_peg_ratio": _safe_float(focus.get("valuation_peg_ratio")),
        "valuation_growth_reference": _safe_float(focus.get("valuation_growth_reference")),
        "valuation_profit_margin": _safe_float(focus.get("valuation_profit_margin")),
        "sector_score": float(focus["sector_score"]),
    }


def _attach_peer_valuation_metrics(frame: pd.DataFrame, target_ticker: str) -> pd.DataFrame:
    frame = frame.copy()
    target_ticker = target_ticker.upper().strip()
    peer_only = frame[frame["ticker"] != target_ticker]
    benchmark = peer_only if not peer_only.empty else frame

    peer_forward_pe_median = _median_or_nan(benchmark["valuation_forward_pe"])
    peer_peg_median = _median_or_nan(benchmark["valuation_peg_ratio"])
    peer_score_median = _median_or_nan(benchmark["valuation_score"])

    frame["peer_forward_pe_median"] = peer_forward_pe_median
    frame["peer_peg_median"] = peer_peg_median
    frame["peer_valuation_score_median"] = peer_score_median
    frame["forward_pe_vs_peer_median_pct"] = frame["valuation_forward_pe"].apply(
        lambda value: _ratio_vs_median(value, peer_forward_pe_median)
    )
    frame["peg_vs_peer_median_pct"] = frame["valuation_peg_ratio"].apply(
        lambda value: _ratio_vs_median(value, peer_peg_median)
    )
    frame["valuation_score_vs_peer_median"] = frame["valuation_score"].apply(
        lambda value: _diff_vs_median(value, peer_score_median)
    )

    frame = frame.sort_values(
        ["valuation_score", "high_probability_score"],
        ascending=[False, False],
    ).reset_index(drop=True)
    frame["valuation_peer_rank"] = frame.index + 1
    frame = frame.sort_values(
        ["high_probability_score", "signal_score"],
        ascending=[False, False],
    ).reset_index(drop=True)
    frame["peer_rank"] = frame.index + 1
    target_rows = frame[frame["ticker"] == target_ticker]
    target_rank = int(target_rows["peer_rank"].iloc[0]) if not target_rows.empty else 0
    frame["target_peer_rank"] = target_rank

    labels = frame.apply(_peer_valuation_label, axis=1)
    frame["peer_valuation_label"] = [item[0] for item in labels]
    frame["peer_valuation_label_zh"] = [item[1] for item in labels]
    notes = frame.apply(_peer_valuation_note, axis=1)
    frame["peer_valuation_note"] = [item[0] for item in notes]
    frame["peer_valuation_note_zh"] = [item[1] for item in notes]
    return frame


def _peer_valuation_label(row: pd.Series) -> tuple[str, str]:
    score_diff = _safe_float(row.get("valuation_score_vs_peer_median"))
    pe_diff = _safe_float(row.get("forward_pe_vs_peer_median_pct"))
    peg_diff = _safe_float(row.get("peg_vs_peer_median_pct"))
    available = [value for value in [score_diff, pe_diff, peg_diff] if value is not None]
    if not available:
        return "unknown", "未知"
    cheaper_signal = (score_diff is not None and score_diff >= 8) or (
        pe_diff is not None and pe_diff <= -0.20
    ) or (peg_diff is not None and peg_diff <= -0.20)
    richer_signal = (
        (score_diff is not None and score_diff <= -8)
        or (pe_diff is not None and pe_diff >= 0.25)
        or (peg_diff is not None and peg_diff >= 0.35)
    )
    if cheaper_signal and richer_signal:
        return "mixed_peer_valuation", "估值信号分化"
    if cheaper_signal:
        return "cheaper_than_peers", "低于同行估值"
    if richer_signal:
        return "richer_than_peers", "高于同行估值"
    return "near_peer_median", "接近同行中位数"


def _peer_valuation_note(row: pd.Series) -> tuple[str, str]:
    label, label_zh = _peer_valuation_label(row)
    return (
        f"Peer valuation label={label}; valuation score is "
        f"{_format_number(row.get('valuation_score_vs_peer_median'))} points vs peer median; "
        f"forward PE is {_format_percent(row.get('forward_pe_vs_peer_median_pct'))} vs peer median; "
        f"PEG is {_format_percent(row.get('peg_vs_peer_median_pct'))} vs peer median.",
        f"同业估值标签={label_zh}；估值分数相对同业中位数"
        f"{_format_number(row.get('valuation_score_vs_peer_median'))}分；"
        f"forward PE相对同业中位数{_format_percent(row.get('forward_pe_vs_peer_median_pct'))}；"
        f"PEG相对同业中位数{_format_percent(row.get('peg_vs_peer_median_pct'))}。",
    )


def _median_or_nan(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return float(numeric.median())


def _ratio_vs_median(value: object, median: float) -> float:
    numeric = _safe_float(value)
    if numeric is None or pd.isna(median) or median <= 0:
        return float("nan")
    return numeric / median - 1.0


def _diff_vs_median(value: object, median: float) -> float:
    numeric = _safe_float(value)
    if numeric is None or pd.isna(median):
        return float("nan")
    return numeric - median


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
        return f"{value:.4f}"
    return str(value)


def _format_percent(value: object) -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return ""
    return f"{numeric:.2%}"


def _safe_float(value: object) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        result = float(value)
    except Exception:
        return None
    return result if pd.notna(result) else None
