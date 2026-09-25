from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class SentimentContext:
    ticker: str
    sentiment_score: float
    sentiment_label: str
    sentiment_label_zh: str
    sentiment_risk_level: str
    sentiment_risk_level_zh: str
    sentiment_block_new_entries: bool
    sentiment_positive_count: int
    sentiment_negative_count: int
    sentiment_high_risk_count: int
    positive_news_level: str
    positive_news_level_zh: str
    positive_news_score: float
    positive_news_major_count: int
    positive_news_strong_count: int
    positive_news_moderate_count: int
    positive_news_drivers: str
    positive_news_drivers_zh: str
    sentiment_titles_used: int
    sentiment_note: str
    sentiment_note_zh: str
    sentiment_warning: str
    source: str
    # Negative-side grading. Fake catalysts (pump/short-seller/reverse split) and
    # dilution (offerings) are news that look like activity but hurt retail buyers.
    risk_news_level: str = "none"
    risk_news_level_zh: str = "无明显风险"
    risk_news_score: float = 0.0
    fake_catalyst_count: int = 0
    dilution_count: int = 0
    risk_news_drivers: str = "none"
    risk_news_drivers_zh: str = "无"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


POSITIVE_KEYWORDS = (
    "beat",
    "beats",
    "upgrade",
    "upgraded",
    "outperform",
    "raises",
    "raised",
    "record",
    "growth",
    "strong",
    "surge",
    "profit",
    "margin",
    "guidance raised",
    "buy rating",
    "price target raised",
)

NEGATIVE_KEYWORDS = (
    "miss",
    "misses",
    "missed",
    "downgrade",
    "downgraded",
    "underperform",
    "cuts",
    "cut",
    "lowered",
    "weak",
    "decline",
    "slump",
    "concern",
    "risk",
    "bearish",
    "price target cut",
)

HIGH_RISK_KEYWORDS = (
    "lawsuit",
    "investigation",
    "sec probe",
    "doj",
    "fraud",
    "recall",
    "bankruptcy",
    "guidance cut",
    "guidance lowered",
    "accounting issue",
    "resigns",
    "layoffs",
    "halts",
    "delisting",
)


POSITIVE_NEWS_CATALYSTS = (
    (
        "major",
        "重大利好",
        18.0,
        (
            "guidance raised",
            "raises guidance",
            "record backlog",
            "major contract",
            "large contract",
            "wins contract",
            "chip deal",
            "supply deal",
            "supply agreement",
            "multiyear deal",
            "multi-year deal",
            "multiyear agreement",
            "deal worth",
            "billion deal",
            "landmark deal",
            "wins order",
            "secures order",
            "major order",
            "large order",
            "record order",
            "order worth",
            "expands partnership",
            "regulatory approval",
            "fda approval",
            "buyout",
            "acquisition offer",
            "strategic partnership",
            "data center backlog",
        ),
    ),
    (
        "strong",
        "强利好",
        10.0,
        (
            "beats estimates",
            "beat estimates",
            "earnings beat",
            "revenue beat",
            "upgrade",
            "upgraded",
            "price target raised",
            "raises price target",
            "outperform",
            "record revenue",
            "record profit",
            "strong demand",
        ),
    ),
    (
        "moderate",
        "普通利好",
        4.0,
        (
            "growth",
            "strong",
            "profit",
            "margin",
            "trend",
            "outpacing",
            "buy now",
            "good stock",
            "positive",
            "momentum",
        ),
    ),
)

POSITIVE_NEWS_LEVEL_LABELS = {
    "major": "major positive",
    "strong": "strong positive",
    "moderate": "ordinary positive",
}


# Negative catalysts that are frequently dressed up as "news". These keyword sets
# are intentionally disjoint from NEGATIVE/HIGH_RISK keywords above so the composite
# score does not double-count the same headline.
RISK_NEWS_CATALYSTS = (
    (
        "fake_catalyst",
        "假利好/炒作",
        -12.0,
        (
            "pump and dump",
            "pump-and-dump",
            "short seller report",
            "short-seller report",
            "short seller",
            "reverse split",
            "reverse stock split",
            "meme stock",
            "social media hype",
            "stock promotion",
            "promoted stock",
            "retail frenzy",
            "going concern",
        ),
    ),
    (
        "dilution",
        "稀释/增发",
        -8.0,
        (
            "stock offering",
            "share offering",
            "public offering",
            "secondary offering",
            "at-the-market offering",
            "atm offering",
            "shelf registration",
            "shelf offering",
            "dilution",
            "dilutive",
            "convertible notes",
            "registered direct",
            "capital raise",
            "priced offering",
        ),
    ),
)

RISK_NEWS_LEVEL_LABELS = {
    "fake_catalyst": "fake catalyst",
    "dilution": "dilution risk",
}


def build_sentiment_context(
    ticker: str,
    snapshot: dict[str, Any] | None,
    source: str = "yfinance_news_titles",
) -> SentimentContext:
    ticker = ticker.upper().strip()
    if not snapshot:
        return _unknown_context(ticker, "snapshot unavailable", source)

    titles = _clean_titles(snapshot.get("news_titles"))
    if not titles:
        return _unknown_context(ticker, "news titles unavailable", source)

    text = " ".join(titles).lower()
    positive_count = _keyword_count(text, POSITIVE_KEYWORDS)
    negative_count = _keyword_count(text, NEGATIVE_KEYWORDS)
    high_risk_count = _keyword_count(text, HIGH_RISK_KEYWORDS)
    catalyst = _grade_positive_news(titles)
    risk_news = _grade_risk_news(titles)

    score = (
        50.0
        + positive_count * 5.0
        + catalyst["score"]
        - negative_count * 6.0
        - high_risk_count * 12.0
        + risk_news["score"]
    )
    score = round(float(max(0.0, min(score, 100.0))), 2)
    label, label_zh = _sentiment_label(score)
    risk_level, risk_level_zh = _sentiment_risk_level(score, high_risk_count)
    fake_catalyst_count = int(risk_news["fake_catalyst_count"])
    # A pump / short-seller report / reverse split is a genuine reason not to chase,
    # even when the raw score still looks acceptable.
    block_new_entries = (
        risk_level == "high" or high_risk_count >= 2 or fake_catalyst_count >= 1
    )

    note = (
        f"Used {len(titles)} recent headline(s); positive keywords={positive_count}, "
        f"negative keywords={negative_count}, high-risk keywords={high_risk_count}; "
        f"positive news level={catalyst['level']}; risk news level={risk_news['level']}."
    )
    note_zh = (
        f"使用{len(titles)}条近期新闻标题；正面关键词={positive_count}，"
        f"负面关键词={negative_count}，高风险关键词={high_risk_count}；"
        f"利好等级={catalyst['level_zh']}；风险新闻等级={risk_news['level_zh']}。"
    )

    return SentimentContext(
        ticker=ticker,
        sentiment_score=score,
        sentiment_label=label,
        sentiment_label_zh=label_zh,
        sentiment_risk_level=risk_level,
        sentiment_risk_level_zh=risk_level_zh,
        sentiment_block_new_entries=block_new_entries,
        sentiment_positive_count=positive_count,
        sentiment_negative_count=negative_count,
        sentiment_high_risk_count=high_risk_count,
        positive_news_level=str(catalyst["level"]),
        positive_news_level_zh=str(catalyst["level_zh"]),
        positive_news_score=float(catalyst["score"]),
        positive_news_major_count=int(catalyst["major_count"]),
        positive_news_strong_count=int(catalyst["strong_count"]),
        positive_news_moderate_count=int(catalyst["moderate_count"]),
        positive_news_drivers=str(catalyst["drivers"]),
        positive_news_drivers_zh=str(catalyst["drivers_zh"]),
        sentiment_titles_used=len(titles),
        sentiment_note=note,
        sentiment_note_zh=note_zh,
        sentiment_warning="",
        source=source,
        risk_news_level=str(risk_news["level"]),
        risk_news_level_zh=str(risk_news["level_zh"]),
        risk_news_score=float(risk_news["score"]),
        fake_catalyst_count=fake_catalyst_count,
        dilution_count=int(risk_news["dilution_count"]),
        risk_news_drivers=str(risk_news["drivers"]),
        risk_news_drivers_zh=str(risk_news["drivers_zh"]),
    )


def _unknown_context(ticker: str, warning: str, source: str) -> SentimentContext:
    return SentimentContext(
        ticker=ticker,
        sentiment_score=50.0,
        sentiment_label="unknown",
        sentiment_label_zh="未知",
        sentiment_risk_level="unknown",
        sentiment_risk_level_zh="未知",
        sentiment_block_new_entries=False,
        sentiment_positive_count=0,
        sentiment_negative_count=0,
        sentiment_high_risk_count=0,
        positive_news_level="unknown",
        positive_news_level_zh="未知",
        positive_news_score=0.0,
        positive_news_major_count=0,
        positive_news_strong_count=0,
        positive_news_moderate_count=0,
        positive_news_drivers="none",
        positive_news_drivers_zh="无",
        sentiment_titles_used=0,
        sentiment_note="Recent news title data is unavailable; neutral score 50 is used.",
        sentiment_note_zh="近期新闻标题不可用；使用中性分数50。",
        sentiment_warning=warning,
        source=source,
    )


def _clean_titles(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    titles: list[str] = []
    for item in value:
        title = str(item).strip()
        if title and title not in titles:
            titles.append(title)
    return titles[:10]


def _keyword_count(text: str, keywords: Iterable[str]) -> int:
    return sum(text.count(keyword) for keyword in keywords)


def _grade_positive_news(titles: list[str]) -> dict[str, object]:
    counts = {"major": 0, "strong": 0, "moderate": 0}
    matched_drivers: list[str] = []
    matched_drivers_zh: list[str] = []
    score = 0.0

    for title in titles:
        title_text = title.lower()
        for level, level_zh, weight, keywords in POSITIVE_NEWS_CATALYSTS:
            matched = [keyword for keyword in keywords if keyword in title_text]
            if not matched:
                continue
            counts[level] += 1
            score += weight
            if len(matched_drivers) < 5:
                matched_drivers.append(f"{POSITIVE_NEWS_LEVEL_LABELS[level]}: {matched[0]}")
                matched_drivers_zh.append(f"{level_zh}: {matched[0]}")
            break

    if counts["major"] > 0:
        level, level_zh = "major", "重大利好"
    elif counts["strong"] > 0:
        level, level_zh = "strong", "强利好"
    elif counts["moderate"] > 0:
        level, level_zh = "moderate", "普通利好"
    else:
        level, level_zh = "none", "无明显利好"

    return {
        "level": level,
        "level_zh": level_zh,
        "score": round(min(score, 30.0), 2),
        "major_count": counts["major"],
        "strong_count": counts["strong"],
        "moderate_count": counts["moderate"],
        "drivers": "; ".join(matched_drivers) or "none",
        "drivers_zh": "；".join(matched_drivers_zh) or "无",
    }


def _grade_risk_news(titles: list[str]) -> dict[str, object]:
    counts = {"fake_catalyst": 0, "dilution": 0}
    matched_drivers: list[str] = []
    matched_drivers_zh: list[str] = []
    score = 0.0

    for title in titles:
        title_text = title.lower()
        for level, level_zh, weight, keywords in RISK_NEWS_CATALYSTS:
            matched = [keyword for keyword in keywords if keyword in title_text]
            if not matched:
                continue
            counts[level] += 1
            score += weight
            if len(matched_drivers) < 5:
                matched_drivers.append(f"{RISK_NEWS_LEVEL_LABELS[level]}: {matched[0]}")
                matched_drivers_zh.append(f"{level_zh}: {matched[0]}")
            break

    if counts["fake_catalyst"] > 0:
        level, level_zh = "fake_catalyst", "假利好/炒作"
    elif counts["dilution"] > 0:
        level, level_zh = "dilution", "稀释/增发"
    else:
        level, level_zh = "none", "无明显风险"

    return {
        "level": level,
        "level_zh": level_zh,
        "score": round(max(score, -30.0), 2),
        "fake_catalyst_count": counts["fake_catalyst"],
        "dilution_count": counts["dilution"],
        "drivers": "; ".join(matched_drivers) or "none",
        "drivers_zh": "；".join(matched_drivers_zh) or "无",
    }


def _sentiment_label(score: float) -> tuple[str, str]:
    if score >= 65:
        return "positive", "正面"
    if score <= 40:
        return "negative", "负面"
    return "neutral", "中性"


def _sentiment_risk_level(score: float, high_risk_count: int) -> tuple[str, str]:
    if high_risk_count >= 2 or score <= 30:
        return "high", "高"
    if high_risk_count == 1 or score <= 45:
        return "medium", "中"
    return "low", "低"
