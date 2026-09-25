"""Render a self-contained HTML dashboard from existing analysis outputs.

This is a pure read-render layer on top of the analysis pipeline: it reads the JSON
results the analysis already writes and produces one browser-openable HTML file. It
does no network I/O and does not touch the core analysis code, so it cannot affect
the rest of the project.
"""

from __future__ import annotations

import html
import json
from datetime import date
from pathlib import Path

import pandas as pd


PRICES_DIR = "data/real_prices"


SUMMARY_FIELDS = (
    "ticker",
    "company_name",
    "current_price",
    "date",
    "latest_price",
    "final_decision",
    "final_decision_zh",
    "final_score",
    "final_watchlist_status_zh",
    "final_focus_horizon_zh",
    "primary_blocker_zh",
    "final_next_step_zh",
    "fundamental_score",
    "fundamental_trend_direction_zh",
    "positive_news_level_zh",
    "positive_news_drivers_zh",
    "risk_news_level_zh",
    "risk_news_drivers_zh",
    "sentiment_risk_level_zh",
    "event_risk_level_zh",
    "news_titles",
    "generated_at_utc",
)


def collect_ticker_summaries(
    real_ticker_dir: str | Path,
    tickers: list[str] | None = None,
) -> list[dict]:
    """Load one summary per ticker. When ``tickers`` is given, only those are shown
    (in that order) — this keeps stale/leftover analysis directories off the page."""
    root = Path(real_ticker_dir)
    if not root.exists():
        return []
    if tickers:
        wanted = [t.upper().strip() for t in tickers if t.strip()]
        dirs = [root / t for t in wanted if (root / t).is_dir()]
    else:
        dirs = sorted(p for p in root.iterdir() if p.is_dir())
    summaries: list[dict] = []
    for ticker_dir in dirs:
        row = _load_summary_row(ticker_dir)
        if row is not None:
            summaries.append({field: row.get(field) for field in SUMMARY_FIELDS})
    return summaries


def _load_summary_row(ticker_dir: Path) -> dict | None:
    # Prefer the full CSV row (has fundamental depth/trend); fall back to the
    # curated JSON, which lacks those fields.
    row: dict | None = None
    csv_path = ticker_dir / "ticker_analysis.csv"
    if csv_path.exists():
        try:
            frame = pd.read_csv(csv_path)
            if not frame.empty:
                row = frame.iloc[0].to_dict()
        except Exception:
            row = None
    if row is None:
        json_path = ticker_dir / "analysis_result.json"
        if json_path.exists():
            try:
                row = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                row = None
    if row is None:
        return None
    # Company name and the live/intraday current price live only in the snapshot,
    # not the analysis row (which is built on closing prices).
    if not row.get("company_name") or not row.get("current_price"):
        snapshot_path = ticker_dir / "external_snapshot.json"
        if snapshot_path.exists():
            try:
                snap = json.loads(snapshot_path.read_text(encoding="utf-8"))
                if not row.get("company_name"):
                    row["company_name"] = snap.get("company_name")
                if not row.get("current_price"):
                    row["current_price"] = snap.get("current_price")
                titles = snap.get("news_titles")
                if isinstance(titles, list) and titles:
                    row["news_titles"] = titles
            except Exception:
                pass
    return row


def recent_returns(ticker: str, prices_dir: str | Path = PRICES_DIR) -> dict | None:
    """1/5/20-trading-day returns from the cached daily price file, for a quick read."""
    root = Path(prices_dir)
    matches = sorted(root.glob(f"{ticker.upper()}_*.csv")) if root.exists() else []
    if not matches:
        return None
    try:
        frame = pd.read_csv(matches[0])
    except Exception:
        return None
    close_col = "adj_close" if "adj_close" in frame.columns else "close"
    if close_col not in frame.columns or len(frame) < 2:
        return None
    closes = pd.to_numeric(frame[close_col], errors="coerce").dropna().to_numpy()
    if len(closes) < 2:
        return None
    latest = float(closes[-1])

    def ret(n: int) -> float | None:
        if len(closes) <= n or closes[-1 - n] == 0:
            return None
        return latest / float(closes[-1 - n]) - 1.0

    return {"r1": ret(1), "r5": ret(5), "r20": ret(20), "latest_close": latest}


SECTOR_NAMES = {
    "XLK": "科技", "XLF": "金融", "XLE": "能源", "XLV": "医疗", "XLY": "可选消费",
    "XLP": "必需消费", "XLI": "工业", "XLU": "公用事业", "XLB": "原材料",
    "XLRE": "房地产", "XLC": "通讯", "SMH": "半导体",
}


def money_flow_index(ticker: str, prices_dir: str | Path = PRICES_DIR, period: int = 14) -> float | None:
    """Money Flow Index (0-100) from cached OHLCV — a free-data volume proxy for
    inflow/outflow. >60 leans inflow (accumulation), <40 leans outflow. NOT
    institutional/dark-pool flow, which needs paid data."""
    root = Path(prices_dir)
    matches = sorted(root.glob(f"{ticker.upper()}_*.csv")) if root.exists() else []
    if not matches:
        return None
    try:
        f = pd.read_csv(matches[0])
    except Exception:
        return None
    need = {"high", "low", "close", "volume"}
    if not need.issubset(f.columns) or len(f) < period + 1:
        return None
    high = pd.to_numeric(f["high"], errors="coerce")
    low = pd.to_numeric(f["low"], errors="coerce")
    close = pd.to_numeric(f["close"], errors="coerce")
    vol = pd.to_numeric(f["volume"], errors="coerce")
    typical = (high + low + close) / 3.0
    raw_flow = typical * vol
    direction = typical.diff()
    pos = raw_flow.where(direction > 0, 0.0).rolling(period).sum()
    neg = raw_flow.where(direction < 0, 0.0).rolling(period).sum()
    denom = float(neg.iloc[-1])
    pos_v = float(pos.iloc[-1])
    if pos_v != pos_v or denom != denom:
        return None
    if denom <= 0:
        return 100.0 if pos_v > 0 else 50.0
    return round(100.0 - 100.0 / (1.0 + pos_v / denom), 1)


def collect_sector_returns(sectors: list[str], prices_dir: str | Path = PRICES_DIR) -> list[dict]:
    """Per-sector recent returns, sorted strongest→weakest (money rotation)."""
    rows = []
    for t in sectors:
        t = str(t).upper().strip()
        if not t:
            continue
        rets = recent_returns(t, prices_dir)
        if not rets:
            continue
        rows.append(
            {
                "ticker": t,
                "name": SECTOR_NAMES.get(t, t),
                "r1": rets.get("r1"),
                "r5": rets.get("r5"),
                "r20": rets.get("r20"),
                "mfi": money_flow_index(t, prices_dir),
            }
        )
    rows.sort(key=lambda x: (x["r5"] if x["r5"] is not None else -9), reverse=True)
    return rows


def _render_sectors(sectors: list[dict]) -> str:
    if not sectors:
        return ""
    rows = ["<tr><th>板块</th><th>1日</th><th>1周</th><th>1月</th><th>资金流(MFI)</th></tr>"]
    for s in sectors:
        r5 = s.get("r5")
        cls = "pl-up" if (r5 or 0) >= 0 else "pl-down"
        mfi = s.get("mfi")
        mfi_txt = "—"
        if isinstance(mfi, (int, float)):
            flow = "流入" if mfi >= 60 else ("流出" if mfi <= 40 else "中性")
            mfi_txt = f"{mfi:.0f} {flow}"
        rows.append(
            f'<tr><td><b>{html.escape(str(s.get("name") or ""))}</b> '
            f'<span class="dim">{html.escape(str(s.get("ticker") or ""))}</span></td>'
            f"<td>{_pctx(s.get('r1'))}</td>"
            f'<td class="{cls}">{_pctx(s.get("r5"))}</td>'
            f"<td>{_pctx(s.get('r20'))}</td><td>{mfi_txt}</td></tr>"
        )
    top = [s["name"] for s in sectors[:3] if (s.get("r5") or 0) > 0]
    bottom = [s["name"] for s in sectors[-3:] if (s.get("r5") or 0) < 0]
    read = ""
    if top or bottom:
        parts = []
        if top:
            parts.append("资金轮入：" + "、".join(top))
        if bottom:
            parts.append("资金轮出：" + "、".join(bottom))
        read = f'<div class="market-read">近一周 {" ｜ ".join(parts)}（板块轮动=散户可见的资金流向；MFI 为量能代理，非机构级真实流向）</div>'
    return f'<div class="news-label">板块表现与资金流 / Sector rotation & flow</div><table class="board">{"".join(rows)}</table>{read}'


def _business_days_behind(date_str: object, today: date | None) -> int | None:
    """分析基准（最后一个收盘日）落后今天几个交易日。

    周一看上周五的收盘 = 落后 1（正常：当天收盘还没发生/管线还没跑）；
    落后 >=2 才说明真的漏跑了一个完整交易日。日历天数会把周末误报成过期。"""
    if today is None or not date_str:
        return None
    try:
        analysis_date = date.fromisoformat(str(date_str)[:10])
    except (ValueError, TypeError):
        return None
    if analysis_date >= today:
        return 0
    return max(0, len(pd.bdate_range(analysis_date, today)) - 1)


def _decision_tone(decision: str) -> str:
    text = (decision or "").lower()
    if any(word in text for word in ("buy", "constructive", "long", "breakout")):
        if "avoid" in text or "wait" in text or "downtrend" in text:
            return "warn"
        return "good"
    if "avoid" in text or "downtrend" in text or "reduce" in text or "exit" in text:
        return "bad"
    return "warn"


def _fmt(value: object, kind: str = "text") -> str:
    # None, empty, or NaN (from a CSV cell) all render as an em dash.
    if value is None or value == "" or (isinstance(value, float) and value != value):
        return "—"
    if kind == "price":
        try:
            return f"${float(value):,.2f}"
        except (TypeError, ValueError):
            return html.escape(str(value))
    if kind == "score":
        try:
            return f"{float(value):.1f}"
        except (TypeError, ValueError):
            return html.escape(str(value))
    return html.escape(str(value))


def render_dashboard_html(
    summaries: list[dict],
    generated_at: str = "",
    market: list[dict] | None = None,
    prices_dir: str | Path = PRICES_DIR,
    today: date | None = None,
    candidates: list[dict] | None = None,
    paper: dict | None = None,
    sectors: list[dict] | None = None,
    banner: str = "",
) -> str:
    cards = "\n".join(_render_card(s, today) for s in summaries) or (
        '<p class="empty">还没有分析结果。先运行 <code>python3 run.py AVGO</code>。</p>'
    )
    market_html = _render_market(market or [], prices_dir, today)
    sectors_html = _render_sectors(sectors or [])
    candidates_html = _render_candidates(candidates or [])
    paper_html = _render_paper(paper)
    stamp = html.escape(generated_at) if generated_at else "—"
    return _PAGE.format(
        banner=banner,
        market=market_html,
        sectors=sectors_html,
        cards=cards,
        candidates=candidates_html,
        paper=paper_html,
        stamp=stamp,
        count=len(summaries),
    )


def _render_market(market: list[dict], prices_dir: str | Path, today: date | None) -> str:
    if not market:
        return '<p class="empty">未配置大盘参考。</p>'
    tiles = []
    for m in market:
        ticker = str(m.get("ticker") or "")
        rets = recent_returns(ticker, prices_dir)
        r1 = rets.get("r1") if rets else None
        r5 = rets.get("r5") if rets else None
        r20 = rets.get("r20") if rets else None
        tone = "good" if (r5 or 0) >= 0 else "bad"
        tiles.append(
            _MARKET_TILE.format(
                tone=tone,
                ticker=_fmt(ticker),
                company=_fmt(m.get("company_name")),
                price=_fmt(m.get("current_price") if m.get("current_price") is not None else m.get("latest_price"), "price"),
                r1=_pct(r1),
                r5=_pct(r5),
                r20=_pct(r20),
            )
        )
    return f'<div class="market">{"".join(tiles)}</div><div class="market-read">{_market_read(market, prices_dir)}</div>'


def _market_read(market: list[dict], prices_dir: str | Path) -> str:
    reads = []
    for m in market:
        ticker = str(m.get("ticker") or "")
        rets = recent_returns(ticker, prices_dir)
        if not rets:
            continue
        r5, r20 = rets.get("r5"), rets.get("r20")
        name = "标普500(VOO)" if ticker == "VOO" else ("纳指100(QQQM)" if ticker == "QQQM" else ticker)
        if r20 is None:
            continue
        if r20 >= 0.03:
            tone = "偏强，近一个月上行"
        elif r20 <= -0.03:
            tone = "偏弱，近一个月回落"
        else:
            tone = "震荡整理"
        wk = f"，近一周{_pct(r5)}" if r5 is not None else ""
        reads.append(f"{name}：{tone}{wk}。")
    if not reads:
        return "暂无足够数据判断大盘。"
    return "简要判断 / Read： " + " ".join(reads)


def _pct(value: object) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "—"
    try:
        v = float(value)
        sign = "+" if v >= 0 else ""
        return f"{sign}{v:.1%}"
    except (TypeError, ValueError):
        return "—"


def _render_card(summary: dict, today: date | None = None) -> str:
    ticker = _fmt(summary.get("ticker"))
    tone = _decision_tone(str(summary.get("final_decision") or ""))
    # Big number = live/intraday snapshot price (what the user sees on their broker);
    # the basis line makes clear the analysis itself runs on the last daily close.
    current = summary.get("current_price")
    close = summary.get("latest_price")
    headline = _fmt(current if current is not None else close, "price")
    behind = _business_days_behind(summary.get("date"), today)
    stale = (
        f' <span class="stale">⚠️ 数据滞后 {behind} 个交易日，请检查每日任务</span>'
        if (behind is not None and behind >= 2)
        else ""
    )
    return _CARD.format(
        tone=tone,
        ticker=ticker,
        company=_fmt(summary.get("company_name")),
        price=headline,
        close=_fmt(close, "price"),
        position=_position_line(summary),
        date=_fmt(summary.get("date")),
        stale=stale,
        decision=_fmt(summary.get("final_decision_zh")),
        score=_fmt(summary.get("final_score"), "score"),
        watchlist=_fmt(summary.get("final_watchlist_status_zh")),
        horizon=_fmt(summary.get("final_focus_horizon_zh")),
        fscore=_fmt(summary.get("fundamental_score"), "score"),
        ftrend=_fmt(summary.get("fundamental_trend_direction_zh")),
        blocker=_fmt(summary.get("primary_blocker_zh")),
        # For a holding, prefer the signal-aware read; fall back to the generic
        # holder note, then (for non-holdings) the entry-based next step.
        nextstep=(
            _holder_next_step(summary)
            or _fmt(summary.get("pos_next_step_zh") or summary.get("final_next_step_zh"))
        ),
        news=_news_block(summary),
    )


# English catalyst/risk keywords → 中文 description. The raw source headlines are in
# English and cannot be auto-translated offline, so the news read is presented as a
# Chinese interpretation of the graded catalysts/risks instead.
_KEYWORD_ZH = {
    "chip deal": "大额芯片订单", "billion deal": "大额交易", "deal worth": "大额交易",
    "supply deal": "供货订单", "supply agreement": "供货协议", "wins order": "获得订单",
    "secures order": "获得订单", "record order": "创纪录订单", "order worth": "大额订单",
    "major order": "大额订单", "large order": "大额订单", "wins contract": "获得合同",
    "major contract": "大额合同", "large contract": "大额合同", "record backlog": "创纪录订单积压",
    "data center backlog": "数据中心订单积压", "beats estimates": "财报超预期",
    "beat estimates": "财报超预期", "earnings beat": "财报超预期", "revenue beat": "营收超预期",
    "upgrade": "分析师上调", "upgraded": "分析师上调", "price target raised": "上调目标价",
    "raises price target": "上调目标价", "outperform": "跑赢评级", "guidance raised": "上调指引",
    "raises guidance": "上调指引", "record revenue": "创纪录营收", "record profit": "创纪录利润",
    "strong demand": "需求强劲", "regulatory approval": "监管批准", "fda approval": "FDA批准",
    "buyout": "收购", "acquisition offer": "收购要约", "strategic partnership": "战略合作",
    "expands partnership": "扩大合作", "growth": "增长", "momentum": "势头强",
    "offering": "增发稀释", "dilution": "股本稀释", "shelf": "储架增发", "short seller": "做空报告",
    "reverse split": "反向拆股", "pump": "炒作", "meme": "题材炒作", "going concern": "持续经营风险",
}


def _drivers_zh(drivers: object) -> list[str]:
    text = str(drivers or "").replace("；", ";")
    found: list[str] = []
    lower = text.lower()
    for eng, zh in _KEYWORD_ZH.items():
        if eng in lower and zh not in found:
            found.append(zh)
    return found


def _news_block(summary: dict) -> str:
    """Chinese news read: graded catalysts/risks translated to 中文 (no English headlines)."""
    tags: list[str] = []
    pos = str(summary.get("positive_news_level_zh") or "").strip()
    risk = str(summary.get("risk_news_level_zh") or "").strip()
    senti = str(summary.get("sentiment_risk_level_zh") or "").strip()
    event = str(summary.get("event_risk_level_zh") or "").strip()
    if pos and pos not in {"无明显利好", "未知"}:
        tags.append(f'<span class="tag good">利好·{html.escape(pos)}</span>')
    if risk and risk not in {"无明显风险", "未知"}:
        tags.append(f'<span class="tag bad">风险·{html.escape(risk)}</span>')
    if senti == "高":
        tags.append('<span class="tag bad">情绪风险高</span>')
    if event == "高":
        tags.append('<span class="tag warn">事件风险高</span>')

    titles = summary.get("news_titles")
    headlines = ""
    if isinstance(titles, list) and titles:
        items = "".join(f"<li>{html.escape(str(t))}</li>" for t in titles[:4])
        headlines = f'<div class="news-label">近期消息 / Recent news</div><ul class="news">{items}</ul>'

    if not tags and not headlines:
        return ""
    tagline = f'<div class="tags">{"".join(tags)}</div>' if tags else ""
    return f'<div class="newsblock">{tagline}{headlines}</div>'


def collect_candidates(
    scan_csv: str | Path,
    exclude: list[str] | None = None,
    top_n: int = 8,
) -> list[dict]:
    """Top model-screened high-probability candidates, excluding current holdings."""
    p = Path(scan_csv)
    if not p.exists():
        return []
    try:
        df = pd.read_csv(p)
    except Exception:
        return []
    if df.empty or "ticker" not in df.columns:
        return []
    excluded = {t.upper().strip() for t in (exclude or [])}
    df = df[~df["ticker"].astype(str).str.upper().isin(excluded)]
    if "high_probability_score" in df.columns:
        df = df.sort_values("high_probability_score", ascending=False)
    return df.head(top_n).to_dict("records")


def _render_candidates(candidates: list[dict]) -> str:
    if not candidates:
        return (
            '<p class="empty">今晚自动扫描后这里会出现模型筛选的潜力股。'
            '手动跑：<code>python3 scan.py --universe growth-core</code></p>'
        )
    rows = ["<tr><th>代码</th><th>公司</th><th>高概率分</th><th>档位</th><th>状态</th></tr>"]
    for c in candidates:
        rows.append(
            "<tr>"
            f"<td><b>{_fmt(c.get('ticker'))}</b></td>"
            f"<td>{_fmt(c.get('company_name'))}</td>"
            f"<td>{_fmt(c.get('high_probability_score'), 'score')}</td>"
            f"<td>{_fmt(c.get('high_probability_level'))}</td>"
            f"<td>{_fmt(c.get('watchlist_status_zh'))}</td>"
            "</tr>"
        )
    return f'<table class="board">{"".join(rows)}</table>'


def _holder_next_step(summary: dict) -> str | None:
    """Signal-aware next step for a holding: combine the model's read (fundamental
    trend, news, P/L, concentration) into a specific leaning — not a template."""
    pl = summary.get("pos_pl_pct")
    if pl is None or (isinstance(pl, float) and pl != pl):
        return None
    profit = float(pl) >= 0
    conc = str(summary.get("pos_conc") or "")
    high_conc = conc in {"high", "extreme"}
    weight = summary.get("pos_weight")
    ftrend = str(summary.get("fundamental_trend_direction_zh") or "")
    pos_news = str(summary.get("positive_news_level_zh") or "")
    risk_news = str(summary.get("risk_news_level_zh") or "")

    reads: list[str] = ["浮盈" if profit else "浮亏"]
    pos_n = 1 if profit else 0
    neg_n = 0 if profit else 1
    if ftrend == "改善中":
        pos_n += 1
        reads.append("基本面改善中")
    elif ftrend == "恶化中":
        neg_n += 1
        reads.append("基本面恶化中")
    if pos_news and pos_news not in {"无明显利好", "未知"}:
        pos_n += 1
        reads.append(f"消息面{pos_news}")
    if risk_news and risk_news not in {"无明显风险", "未知"}:
        neg_n += 1
        reads.append(f"风险:{risk_news}")

    if neg_n >= 2 and high_conc:
        lean = "多个负面叠加 + 仓位过大 → 优先考虑减仓降风险"
    elif neg_n >= 2:
        lean = "负面信号偏多 → 考虑减仓或收紧止损"
    elif pos_n >= 2 and high_conc:
        lean = "信号偏正面但仓位过大 → 可锁部分利润、把集中度降下来，而非全卖"
    elif pos_n >= 2:
        lean = "信号偏正面 → 可继续持有，设好止损"
    else:
        lean = "信号中性 → 持有观察，守住成本线"

    tail = ""
    if high_conc and weight is not None:
        tail = f"（集中度{'极高' if conc == 'extreme' else '高'}，占{float(weight):.0f}%，是当前最大风险）"
    return f"信号解读：{html.escape('，'.join(reads))}。{lean}{tail}。是否操作由你决定。"


def _position_line(summary: dict) -> str:
    pl = summary.get("pos_pl_pct")
    if pl is None or (isinstance(pl, float) and pl != pl):
        return ""
    weight = summary.get("pos_weight")
    conc = str(summary.get("pos_conc") or "")
    conc_zh = str(summary.get("pos_conc_zh") or "")
    warn = " ⚠️" if conc in {"high", "extreme"} else ""
    cls = "pl-up" if float(pl) >= 0 else "pl-down"
    wtxt = f"{float(weight):.1f}%" if weight is not None else "—"
    return (
        f'<div class="position">持仓 · 浮盈 <span class="{cls}">{float(pl):+.2%}</span>'
        f' · 仓位 {wtxt} · 集中度 {html.escape(conc_zh)}{warn}</div>'
    )


def collect_paper_detail(outputs_dir: str | Path) -> dict:
    """What the paper portfolio decided: target allocations + reasons, and holdings."""
    outputs_root = Path(outputs_dir)
    root = outputs_root / "paper_latest"
    detail: dict = {
        "targets": [],
        "holdings": [],
        "cash": None,
        "orders": [],
        "audit_run_count": 0,
        "audit_valuation_count": 0,
        "audit_journal_available": False,
    }
    tp = root / "paper_targets.csv"
    if tp.exists():
        try:
            df = pd.read_csv(tp)
            for _, r in df.iterrows():
                detail["targets"].append(
                    {
                        "ticker": str(r.get("ticker") or "").upper(),
                        "weight": r.get("paper_target_weight"),
                        "score": _f(r.get("paper_score")),
                        "reason": r.get("paper_reason_zh") or r.get("paper_reason"),
                    }
                )
        except Exception:
            pass
    op = root / "orders.csv"
    if op.exists():
        try:
            df = pd.read_csv(op)
            for _, r in df.iterrows():
                detail["orders"].append(
                    {
                        "ticker": str(r.get("ticker") or "").upper(),
                        "action": str(r.get("action") or "").upper(),
                        "quantity": _f(r.get("quantity")),
                        "price": _f(r.get("price")),
                        "notional": _f(r.get("notional")),
                    }
                )
        except Exception:
            pass
    sp = root / "post_trade_state.csv"
    if sp.exists():
        try:
            df = pd.read_csv(sp)
            for _, r in df.iterrows():
                t = str(r.get("ticker") or "").upper()
                if t == "CASH":
                    detail["cash"] = _f(r.get("quantity"))
                elif t:
                    detail["holdings"].append({"ticker": t, "quantity": _f(r.get("quantity"))})
        except Exception:
            pass
    audit_root = outputs_root / "paper_history"
    detail["audit_journal_available"] = (audit_root / "paper_journal.md").exists()
    for key, filename in [
        ("audit_run_count", "paper_runs.csv"),
        ("audit_valuation_count", "paper_valuations.csv"),
    ]:
        path = audit_root / filename
        if path.exists():
            try:
                detail[key] = int(len(pd.read_csv(path)))
            except Exception:
                detail[key] = 0
    return detail


def _f(value: object) -> float | None:
    try:
        n = float(value)
        return n if n == n else None
    except (TypeError, ValueError):
        return None


def _render_paper(paper: dict | None) -> str:
    if not paper:
        return '<p class="empty">模拟盘还没启动。每天自动跑会初始化并追踪净值。</p>'
    days = int(paper.get("days_tracked") or 0)
    eq = paper.get("latest_equity")
    eq_txt = f"${float(eq):,.2f}" if isinstance(eq, (int, float)) else "—"
    readiness = html.escape(str(paper.get("readiness_zh") or ""))
    note = html.escape(str(paper.get("note_zh") or ""))

    decisions = _render_paper_decisions(paper)
    orders = _render_paper_orders(paper)
    audit = _render_paper_audit(paper)
    logic = _render_paper_logic()
    return (
        '<div class="mtile">'
        f'<div class="mhead">模拟盘净值 <span class="mprice">{eq_txt}</span></div>'
        f'<div class="mrets"><span>已追踪 {days} 天</span>'
        f'<span>收益 {_pctx(paper.get("total_return"))}</span>'
        f'<span>基准 {_pctx(paper.get("benchmark_return"))}</span>'
        f'<span>超额 {_pctx(paper.get("excess_return"))}</span>'
        f'<span>最大回撤 {_pctx(paper.get("max_drawdown"))}</span></div>'
        f'<div class="market-read">是否可进下一步：<b>{readiness}</b> — {note}</div>'
        f"{_render_backtest_gate(paper.get('backtest_gate'))}"
        f"{decisions}"
        f"{orders}"
        f"{audit}"
        f"{logic}"
        "</div>"
    )


def _render_paper_orders(paper: dict) -> str:
    """This run's actual simulated buy/sell tickets."""
    orders = paper.get("orders") or []
    if not orders:
        return (
            '<div class="news-label">本期买卖 / Orders</div>'
            '<div class="paper-read">本期没有触发买卖单（目标与现有仓位差异低于最小交易额 $100，或已无更优配置）——不动也是一种纪律。</div>'
        )
    rows = ["<tr><th>动作</th><th>代码</th><th>股数</th><th>成交价</th><th>金额</th></tr>"]
    for o in orders[:8]:
        act = str(o.get("action") or "")
        act_zh = "买入" if act == "BUY" else ("卖出" if act == "SELL" else act)
        cls = "pl-up" if act == "BUY" else "pl-down"
        q = o.get("quantity")
        qtxt = f"{float(q):.4f}" if isinstance(q, (int, float)) and q == q else "—"
        p = o.get("price")
        ptxt = f"${float(p):,.2f}" if isinstance(p, (int, float)) and p == p else "—"
        n = o.get("notional")
        ntxt = f"${float(n):,.0f}" if isinstance(n, (int, float)) and n == n else "—"
        rows.append(
            f'<tr><td><span class="{cls}">{act_zh}</span></td>'
            f"<td><b>{html.escape(str(o.get('ticker') or ''))}</b></td>"
            f"<td>{qtxt}</td><td>{ptxt}</td><td>{ntxt}</td></tr>"
        )
    return (
        '<div class="news-label">本期买卖 / Orders（模拟单，不是真实下单）</div>'
        f'<table class="board">{"".join(rows)}</table>'
    )


def _render_paper_audit(paper: dict) -> str:
    if not paper.get("audit_journal_available"):
        return ""
    run_count = int(paper.get("audit_run_count") or 0)
    valuation_count = int(paper.get("audit_valuation_count") or 0)
    return (
        '<div class="news-label">完整记录 / Audit Trail</div>'
        '<div class="paper-read">'
        f"已保存 {run_count} 次再平衡、{valuation_count} 次估值；每次都有独立且不会被覆盖的快照。 "
        '<a href="../paper_history/paper_journal.md">打开完整审计日志</a>'
        "</div>"
    )


def _render_paper_logic() -> str:
    """Static explainer: the buy/sell rules the paper engine follows, step by step."""
    steps = [
        ("1. 选池", "从每日高概率扫描 <code>high_probability_scan.csv</code>（~86 只成长股）里逐只过滤。"),
        ("2. 过关门槛", "必须<b>通过质量闸门</b>，或（本次开了 <code>--allow-near-watchlist</code>）状态为「接近但未达标」；且 风险≠高、数据质量≥70、置信度≥65；再受市场状态保护规则约束。"),
        ("3. 打分", "综合分 = 胜率 35% + 高概率分 25% + 回测可信度 15% + 信号分 10% + 置信度 10% + 数据质量 5%。"),
        ("4. 定标的", "按 (是否过闸门, 综合分) 排序，最多取 <b>5 只</b>。"),
        ("5. 定仓位", "留 <b>10% 现金</b>，其余 90% 按分数分配，单只≤<b>20%</b>；低于 2% 的目标丢弃。"),
        ("6. 生成买卖", "对比目标市值与当前持仓：差额&lt;$100 或&lt;0.1% 不动；买单成交价含半个买卖价差（买贵卖便宜），扣 5bps 滑点；现金不够则按比例缩小买单。"),
        ("7. 盯市", "用当日收盘价重估净值，与基准（QQQ）比，满 20 个交易日才与回测年化对比。"),
    ]
    items = "".join(
        f'<tr><td><b>{html.escape(t)}</b></td><td>{body}</td></tr>' for t, body in steps
    )
    return (
        '<div class="news-label">买卖逻辑 / How it decides（固定规则，每天照此执行）</div>'
        f'<table class="board paper-logic">{items}</table>'
        '<div class="paper-read">初始资金 $1000 · 最多 5 只 · 单只≤20% · 留 10% 现金 · 分数越高分得越多。规则固定、可复现，正是要和你的人工判断做对照。</div>'
    )


def _render_backtest_gate(gate: dict | None) -> str:
    if not gate:
        return ""
    verdict = str(gate.get("gate") or "")
    cls = "pl-down" if verdict in {"divergent", "below_expectation"} else (
        "pl-up" if verdict == "consistent" else ""
    )
    exp = _pctx(gate.get("backtest_expected_annual"))
    paper_annual = _pctx(gate.get("paper_annual"))
    compare = f"（回测预期年化 {exp} vs 模拟盘年化 {paper_annual}）" if gate.get("paper_annual") is not None else ""
    return (
        f'<div class="paper-read">回测对比闸门：'
        f'<span class="{cls}">{html.escape(str(gate.get("gate_zh") or ""))}</span> '
        f'{compare} — {html.escape(str(gate.get("note_zh") or ""))}</div>'
    )


def _render_paper_decisions(paper: dict) -> str:
    targets = paper.get("targets") or []
    holdings = paper.get("holdings") or []
    cash = paper.get("cash")
    if not targets and not holdings:
        return (
            '<div class="news-label">本期决策 / What it decided</div>'
            '<div class="paper-read">本期模型没有找到过关的候选，保持现金观望——这本身就是一个决定（不乱买）。</div>'
        )
    rows = ["<tr><th>代码</th><th>分配</th><th>理由</th></tr>"]
    for t in targets[:8]:
        w = t.get("weight")
        wtxt = f"{float(w):.0%}" if isinstance(w, (int, float)) and w == w else "—"
        rows.append(
            f"<tr><td><b>{html.escape(str(t.get('ticker') or ''))}</b></td>"
            f"<td>{wtxt}</td><td>{html.escape(str(t.get('reason') or ''))}</td></tr>"
        )
    cash_txt = f"（现金 ${float(cash):,.0f}）" if isinstance(cash, (int, float)) else ""
    table = f'<table class="board">{"".join(rows)}</table>' if targets else ""
    hold_txt = ""
    if holdings:
        names = "、".join(f"{h['ticker']}×{h.get('quantity')}" for h in holdings[:8])
        hold_txt = f'<div class="paper-read">当前持仓：{html.escape(names)} {cash_txt}</div>'
    return f'<div class="news-label">本期决策 / What it decided（怎么分配、为什么）</div>{table}{hold_txt}'


def _pctx(value: object) -> str:
    try:
        if value is None or (isinstance(value, float) and value != value):
            return "—"
        return f"{float(value):+.2%}"
    except (TypeError, ValueError):
        return "—"


_CARD = """
<div class="card {tone}">
  <div class="card-head"><span class="ticker">{ticker}</span>
    <span class="price"><small>现价</small> {price}</span></div>
  <div class="company">{company}</div>
  <div class="decision">{decision}</div>
  {position}
  <div class="meta">
    <span>综合分 <b>{score}</b></span><span>观察 {watchlist}</span><span>周期 {horizon}</span>
  </div>
  <div class="meta">
    <span>基本面 <b>{fscore}</b></span><span>趋势 {ftrend}</span>
  </div>
  <div class="basis">分析基准：收盘 {close} · {date}{stale}</div>
  <div class="blocker">卡点：{blocker}</div>
  <div class="nextstep">下一步：{nextstep}</div>
  {news}
</div>"""


_MARKET_TILE = """
<div class="mtile {tone}">
  <div class="mhead"><b>{ticker}</b> <span class="mprice">{price}</span></div>
  <div class="mname">{company}</div>
  <div class="mrets"><span>1日 {r1}</span><span>1周 {r5}</span><span>1月 {r20}</span></div>
</div>"""


_PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>选股看板 / Stock Dashboard</title>
<style>
:root {{ color-scheme: light dark; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  background:#0f1115; color:#e6e8eb; padding:24px; }}
@media (prefers-color-scheme: light) {{ body {{ background:#f5f6f8; color:#1a1d22; }} }}
h1 {{ font-size:22px; margin:0 0 4px; }}
.sub {{ opacity:.6; font-size:13px; margin-bottom:20px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:14px; }}
.card {{ border-radius:12px; padding:16px; background:#1a1d24; border-left:4px solid #667; }}
@media (prefers-color-scheme: light) {{ .card {{ background:#fff; box-shadow:0 1px 4px rgba(0,0,0,.08); }} }}
.card.good {{ border-left-color:#2ecc71; }}
.card.warn {{ border-left-color:#f1c40f; }}
.card.bad {{ border-left-color:#e74c3c; }}
.card-head {{ display:flex; justify-content:space-between; align-items:baseline; }}
.ticker {{ font-size:20px; font-weight:700; }}
.price {{ font-size:16px; opacity:.9; }}
.price small {{ font-size:10px; opacity:.6; }}
.company {{ font-size:12px; opacity:.55; margin-top:2px; }}
.basis {{ font-size:11px; opacity:.5; margin-top:8px; }}
.position {{ font-size:12px; margin:6px 0; padding:4px 8px; border-radius:6px; background:#2226; }}
.pl-up {{ color:#2ecc71; font-weight:600; }}
.pl-down {{ color:#e74c3c; font-weight:600; }}
.newsblock {{ margin-top:10px; border-top:1px solid #3335; padding-top:8px; }}
.tags {{ display:flex; gap:6px; flex-wrap:wrap; margin-bottom:6px; }}
.tag {{ font-size:11px; padding:2px 7px; border-radius:10px; }}
.tag.good {{ background:rgba(46,204,113,.18); color:#2ecc71; }}
.tag.bad {{ background:rgba(231,76,60,.18); color:#e74c3c; }}
.tag.warn {{ background:rgba(241,196,15,.18); color:#d4a70a; }}
.news-label {{ font-size:11px; opacity:.5; margin-bottom:3px; }}
.news {{ margin:0; padding-left:16px; font-size:11px; opacity:.75; }}
.news li {{ margin-bottom:2px; }}
.stale {{ color:#e67e22; opacity:1; font-weight:600; }}
.banner {{ padding:10px 14px; border-radius:8px; margin:10px 0; font-size:14px; font-weight:600; }}
.banner-bad {{ background:#c0392b22; border:1px solid #c0392b; }}
.banner-warn {{ background:#e67e2222; border:1px solid #e67e22; }}
.market {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(180px,1fr)); gap:12px; margin-bottom:8px; }}
.mtile {{ border-radius:10px; padding:12px 14px; background:#1a1d24; border-left:4px solid #667; }}
@media (prefers-color-scheme: light) {{ .mtile {{ background:#fff; box-shadow:0 1px 4px rgba(0,0,0,.08); }} }}
.mtile.good {{ border-left-color:#2ecc71; }}
.mtile.bad {{ border-left-color:#e74c3c; }}
.mhead {{ font-size:16px; }} .mprice {{ opacity:.8; font-size:14px; }}
.mname {{ font-size:11px; opacity:.5; margin:2px 0 6px; }}
.mrets {{ display:flex; gap:12px; font-size:12px; opacity:.85; }}
.market-read {{ font-size:13px; opacity:.8; margin:4px 0 8px; padding:8px 12px; background:#2224; border-radius:8px; }}
.paper-read {{ font-size:12px; opacity:.8; margin:4px 0; }}
.decision {{ margin:8px 0; font-size:15px; font-weight:600; }}
.meta {{ display:flex; gap:14px; flex-wrap:wrap; font-size:12px; opacity:.8; margin:4px 0; }}
.meta b {{ opacity:1; }}
.dim {{ opacity:.5; margin-left:auto; }}
.blocker,.nextstep {{ font-size:12px; margin-top:6px; opacity:.75; }}
h2 {{ font-size:17px; margin:28px 0 12px; }}
h2 .note {{ font-size:12px; font-weight:400; opacity:.5; }}
table.board {{ border-collapse:collapse; width:100%; font-size:13px; overflow-x:auto; display:block; }}
table.board th, table.board td {{ border:1px solid #333a; padding:6px 10px; text-align:left; white-space:nowrap; }}
@media (prefers-color-scheme: light) {{ table.board th, table.board td {{ border-color:#ddd; }} }}
table.board th {{ background:#2226; }}
table.paper-logic {{ display:table; table-layout:fixed; }}
table.paper-logic td {{ white-space:normal; vertical-align:top; line-height:1.5; }}
table.paper-logic td:first-child {{ width:74px; white-space:nowrap; opacity:.85; }}
.empty {{ opacity:.6; font-size:14px; }}
code {{ background:#3336; padding:1px 5px; border-radius:4px; }}
.foot {{ margin-top:28px; font-size:12px; opacity:.5; }}
</style></head>
<body>
<h1>选股看板 / Stock Dashboard</h1>
<div class="sub">{count} 只持仓 · 更新于 {stamp} · 本页只读，不影响任何分析逻辑</div>
{banner}
<h2>大盘概况 / Market</h2>
{market}
{sectors}
<h2>我的持仓 / Holdings</h2>
<div class="grid">{cards}</div>
<h2>潜力股候选 / Candidates <span class="note">（模型筛选，非个人推荐；仅供研究）</span></h2>
{candidates}
<h2>模拟盘 / Paper Trading <span class="note">（模型用假钱向前跑，验证策略；不是你的操作）</span></h2>
{paper}
<div class="foot">本地生成的静态页面。重新运行分析后再次生成即可刷新。</div>
</body></html>"""


def _enrich_with_positions(summaries: list[dict], portfolio_path: str | Path) -> None:
    """Attach cost-basis P/L, weight, and concentration to holding cards.

    浮盈/市值/权重全部用今天的分析价重算——portfolio.csv 里的 current_price /
    market_value / weight_pct 是手工导出时的静态快照，隔天就过期；csv 只作为
    shares + avg_cost（真正不变的东西）的事实来源。"""
    from .position import build_position_context

    fresh_price: dict[str, float] = {}
    for summary in summaries:
        ticker = str(summary.get("ticker") or "").upper()
        price = summary.get("current_price") or summary.get("latest_price")
        try:
            if ticker and price is not None and float(price) > 0:
                fresh_price[ticker] = float(price)
        except (TypeError, ValueError):
            continue
    fresh_weights = _fresh_portfolio_weights(portfolio_path, fresh_price)

    for summary in summaries:
        ticker = str(summary.get("ticker") or "").upper()
        if not ticker:
            continue
        pos = build_position_context(
            ticker,
            latest_price=fresh_price.get(ticker),
            portfolio_path=portfolio_path,
            weight_pct_override=fresh_weights.get(ticker),
        )
        if pos.is_held:
            summary["pos_pl_pct"] = pos.unrealized_pl_pct
            summary["pos_weight"] = pos.weight_pct
            summary["pos_conc"] = pos.concentration_level
            summary["pos_conc_zh"] = pos.concentration_level_zh
            summary["pos_next_step_zh"] = pos.next_step_zh


def _fresh_portfolio_weights(
    portfolio_path: str | Path, fresh_price: dict[str, float]
) -> dict[str, float]:
    """按最新价重算各持仓权重。没有新价的行（含 CASH）退回 csv 的 market_value。"""
    path = Path(portfolio_path)
    if not path.exists():
        return {}
    try:
        frame = pd.read_csv(path)
    except Exception:
        return {}
    if frame.empty or "ticker" not in frame.columns:
        return {}
    values: dict[str, float] = {}
    for row in frame.to_dict(orient="records"):
        ticker = str(row.get("ticker") or "").upper().strip()
        if not ticker:
            continue
        shares = row.get("shares")
        price = fresh_price.get(ticker)
        value = None
        try:
            if price is not None and shares is not None and float(shares) == float(shares):
                value = float(shares) * price
        except (TypeError, ValueError):
            value = None
        if value is None:
            raw = row.get("market_value")
            try:
                value = float(raw) if raw is not None and float(raw) == float(raw) else None
            except (TypeError, ValueError):
                value = None
        if value is not None:
            values[ticker] = value
    total = sum(values.values())
    if total <= 0:
        return {}
    return {t: v / total * 100.0 for t, v in values.items() if t != "CASH"}


def _build_page_banner(
    summaries: list[dict],
    market: list[dict],
    today: date | None,
    outputs_dir: Path,
) -> str:
    """页面级健康横幅：红=最近一次自动更新有失败步骤；黄=数据滞后≥2个交易日。
    有了它，管线静默失败时打开页面第一眼就能看到，而不是继续读旧数据。"""
    parts: list[str] = []
    status_path = outputs_dir / "dashboard" / "last_run_status.json"
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            status = None
        if status and status.get("ok") is False:
            fails = html.escape(str(status.get("failures") or "未知步骤"))
            finished = html.escape(str(status.get("finished_at") or ""))
            parts.append(
                f'<div class="banner banner-bad">🔴 最近一次自动更新有失败步骤：{fails}'
                f'（{finished}）。页面可能缺数据，详见 outputs/daily_update.log</div>'
            )
    scan_result_path = outputs_dir / "scans" / "latest" / "scan_result.json"
    if scan_result_path.exists():
        try:
            scan_result = json.loads(scan_result_path.read_text(encoding="utf-8"))
            scan_failures = scan_result.get("failures") or []
        except Exception:
            scan_failures = []
        if scan_failures:
            failed_tickers = ", ".join(
                str(item.get("ticker") or "?") for item in scan_failures if isinstance(item, dict)
            )
            parts.append(
                '<div class="banner banner-bad">🔴 本次候选扫描有股票缺失数据：'
                f'{html.escape(failed_tickers)}。这些股票没有进入排序，请检查扫描报告。</div>'
            )
    dates = sorted(
        str(s.get("date"))
        for s in list(summaries) + list(market)
        if s.get("date")
    )
    newest = dates[-1] if dates else None
    behind = _business_days_behind(newest, today)
    if behind is not None and behind >= 2:
        parts.append(
            f'<div class="banner banner-warn">⚠️ 全页数据已滞后 {behind} 个交易日'
            f'（最新分析基准 {html.escape(str(newest))}）。每日任务可能没在跑，'
            f'检查：launchctl print gui/$(id -u)/com.stockselector.daily</div>'
        )
    return "".join(parts)


def _read_watchlist(path: str | Path) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    tickers: list[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            # Allow one-per-line or several space-separated tickers per line.
            tickers.extend(tok.upper() for tok in line.split())
    return tickers


def build_dashboard(
    outputs_dir: str | Path = "outputs",
    generated_at: str = "",
    watchlist_path: str | Path | None = "watchlist.txt",
    market_path: str | Path | None = "market.txt",
    prices_dir: str | Path = PRICES_DIR,
    today: date | None = None,
    portfolio_path: str | Path = "portfolio.csv",
) -> Path:
    outputs = Path(outputs_dir)
    tickers = _read_watchlist(watchlist_path) if watchlist_path else []
    summaries = collect_ticker_summaries(outputs / "real_ticker", tickers=tickers or None)
    _enrich_with_positions(summaries, portfolio_path)
    market_tickers = _read_watchlist(market_path) if market_path else []
    market = collect_ticker_summaries(outputs / "real_ticker", tickers=market_tickers or None)
    sector_tickers = _read_watchlist("sectors.txt") if Path("sectors.txt").exists() else []
    sectors = collect_sector_returns(sector_tickers, prices_dir) if sector_tickers else []
    candidates = collect_candidates(
        outputs / "scans" / "latest" / "high_probability_scan.csv",
        exclude=tickers + market_tickers,
    )
    paper_path = outputs / "paper_latest" / "paper_performance.json"
    paper = None
    if paper_path.exists():
        try:
            paper = json.loads(paper_path.read_text(encoding="utf-8"))
        except Exception:
            paper = None
    detail = collect_paper_detail(outputs)
    if paper is not None:
        paper.update(detail)
    elif detail.get("targets") or detail.get("holdings") or detail.get("cash") is not None:
        paper = detail
    banner = _build_page_banner(summaries, market, today, outputs)
    page = render_dashboard_html(
        summaries,
        generated_at,
        market=market,
        prices_dir=prices_dir,
        today=today,
        candidates=candidates,
        paper=paper,
        sectors=sectors,
        banner=banner,
    )
    dashboard_dir = outputs / "dashboard"
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    index_path = dashboard_dir / "index.html"
    index_path.write_text(page, encoding="utf-8")
    return index_path
