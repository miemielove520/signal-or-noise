from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from .analysis import HORIZON_SPECS, analyze_ticker, render_ticker_analysis
from .analyst import AnalystContext, build_analyst_context
from .data import download_prices_for_period_multi_source
from .data_sources import (
    DataReadinessReport,
    build_data_readiness_report,
    data_readiness_frame,
    data_readiness_payload,
    render_data_readiness_report,
)
from .events import EventRiskContext, fetch_yfinance_event_risk
from .fundamentals import FundamentalContext, build_fundamental_context
from .position import build_position_context
from .json_io import dataframe_records, write_json
from .market import (
    SectorContext,
    build_market_context,
    build_relative_strength_contexts,
    build_sector_context,
    choose_sector_etf,
)
from .peer import (
    build_peer_comparison_frame,
    choose_peer_tickers,
    render_peer_comparison_report,
)
from .screening_config import (
    ScreeningConfig,
    ScreeningProfile,
    ScreeningThresholds,
    default_screening_config,
)
from .sentiment import SentimentContext, build_sentiment_context
from .sec_data import SecFundamentalSnapshot, fetch_sec_fundamental_snapshot
from .signal_review import (
    build_signal_review_feedback_context,
    render_signal_review_section,
    summarize_ticker_signal_review,
    write_signal_review,
)
from .snapshot import fetch_yfinance_snapshot
from .valuation import ValuationContext, build_valuation_context


TICKER_ALIASES = {
    "APPL": "AAPL",
}


TICKER_CLASSIFICATION_FALLBACKS = {
    "AAPL": {"sector": "Technology", "industry": "Consumer Electronics"},
    "MSFT": {"sector": "Technology", "industry": "Software"},
    "NVDA": {"sector": "Technology", "industry": "Semiconductors"},
    "AMD": {"sector": "Technology", "industry": "Semiconductors"},
    "AVGO": {"sector": "Technology", "industry": "Semiconductors"},
    "INTC": {"sector": "Technology", "industry": "Semiconductors"},
    "QCOM": {"sector": "Technology", "industry": "Semiconductors"},
    "NOW": {"sector": "Technology", "industry": "Software"},
    "CRM": {"sector": "Technology", "industry": "Software"},
    "ADBE": {"sector": "Technology", "industry": "Software"},
    "ORCL": {"sector": "Technology", "industry": "Software"},
    "SNOW": {"sector": "Technology", "industry": "Software"},
    "DDOG": {"sector": "Technology", "industry": "Software"},
    "PANW": {"sector": "Technology", "industry": "Software"},
    "PLTR": {"sector": "Technology", "industry": "Software"},
    "META": {"sector": "Communication Services", "industry": "Internet Content & Information"},
    "GOOGL": {"sector": "Communication Services", "industry": "Internet Content & Information"},
    "GOOG": {"sector": "Communication Services", "industry": "Internet Content & Information"},
    "AMZN": {"sector": "Consumer Cyclical", "industry": "Internet Retail"},
    "TSLA": {"sector": "Consumer Cyclical", "industry": "Automobiles"},
    "JPM": {"sector": "Financial Services", "industry": "Banks"},
    "BAC": {"sector": "Financial Services", "industry": "Banks"},
    "LLY": {"sector": "Healthcare", "industry": "Drug Manufacturers"},
    "UNH": {"sector": "Healthcare", "industry": "Healthcare Plans"},
    "AAON": {"sector": "Industrials", "industry": "Building Products"},
}


SNAPSHOT_CACHE_MAX_AGE_DAYS = 14


@dataclass(frozen=True)
class RealTickerAnalysisResult:
    ticker: str
    requested_period: str
    effective_period: str
    auto_period_upgraded: bool
    auto_period_upgrade_reason: str
    auto_period_upgrade_reason_zh: str
    prices: pd.DataFrame
    scored: pd.DataFrame
    analysis: pd.DataFrame
    output_dir: Path
    snapshot: dict[str, object] | None
    benchmark_prices: dict[str, pd.DataFrame]
    data_sources: dict[str, str]
    data_source_warnings: dict[str, tuple[str, ...]]
    event_risk: EventRiskContext
    fundamentals: FundamentalContext
    sentiment: SentimentContext
    analyst: AnalystContext
    valuation: ValuationContext
    sector_context: SectorContext
    peer_comparison: pd.DataFrame
    data_readiness: DataReadinessReport
    cache_metadata: dict[str, object]


def _attach_position_context(analysis: pd.DataFrame, ticker: str) -> None:
    """Add position_* columns to every row of the analysis frame from portfolio.csv."""
    latest_price = None
    if "latest_price" in analysis.columns and not analysis.empty:
        try:
            latest_price = float(analysis["latest_price"].iloc[0])
        except (TypeError, ValueError):
            latest_price = None
    position = build_position_context(ticker, latest_price=latest_price)
    for key, value in position.to_dict().items():
        analysis[f"position_{key}" if not key.startswith("position_") else key] = value


def _build_fundamental_trend_context(
    ticker: str,
    data_root: str | Path,
    as_of_date: object,
    allow_fetch: bool,
):
    """Analyze the multi-period SEC fundamental trend for a ticker. Returns None on
    any failure (offline, missing SEC_USER_AGENT, unknown ticker) so the caller
    degrades to a no-trend fundamental context."""
    if not allow_fetch:
        return None
    try:
        from .fundamental_trends import fetch_fundamental_trend_context

        return fetch_fundamental_trend_context(
            ticker, data_root=str(data_root), as_of_date=as_of_date
        )
    except Exception as exc:
        # 降级但别静默：基本面趋势缺席会拉低打分深度，日志里要能看出来。
        print(f"- {ticker}: fundamental trend context unavailable ({exc})", file=sys.stderr)
        return None


def run_real_ticker_analysis(
    ticker: str,
    period: str = "5y",
    horizons: tuple[str, ...] = ("all",),
    output_root: str | Path = "outputs/real_ticker",
    data_root: str | Path = "data/real_prices",
    include_snapshot: bool = True,
    include_peer_comparison: bool = True,
    peer_limit: int = 6,
    screening_thresholds: ScreeningThresholds | None = None,
    screening_config: ScreeningConfig | None = None,
    auto_period_upgrade: bool = True,
    probability_calibration_path: str | Path | None = None,
    _requested_period: str | None = None,
    _auto_period_upgraded: bool = False,
    _auto_period_upgrade_reason: str = "",
    _auto_period_upgrade_reason_zh: str = "",
) -> RealTickerAnalysisResult:
    ticker = normalize_ticker(ticker)
    if not ticker:
        raise ValueError("Ticker cannot be empty.")
    requested_period = _requested_period or period

    output_dir = Path(output_root) / ticker
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(data_root) / f"{ticker}_{period}.csv"

    price_result = download_prices_for_period_multi_source(
        [ticker],
        period=period,
        output_path=data_path,
    )
    prices = price_result.prices
    benchmark_prices, benchmark_sources, benchmark_warnings = _download_benchmark_prices(
        period=period,
        data_root=Path(data_root),
    )
    market_context = build_market_context(benchmark_prices)
    relative_strength_contexts = build_relative_strength_contexts(
        target_prices=prices,
        benchmark_prices=benchmark_prices,
        horizon_windows={
            name: spec.momentum_window
            for name, spec in HORIZON_SPECS.items()
        },
    )
    scored = build_single_ticker_scored_frame(prices)
    event_risk = fetch_yfinance_event_risk(
        ticker=ticker,
        as_of_date=prices["date"].max(),
    )
    snapshot: dict[str, object] | None = None
    if include_snapshot:
        snapshot = fetch_yfinance_snapshot(ticker)
        snapshot, cache_repair_actions, cache_repair_actions_zh = _repair_snapshot_from_cache(
            ticker=ticker,
            snapshot=snapshot,
            data_root=Path(data_root),
        )
    else:
        cache_repair_actions = ()
        cache_repair_actions_zh = ()
    snapshot, data_repair_actions, data_repair_actions_zh = _repair_snapshot_classification(
        ticker=ticker,
        snapshot=snapshot,
    )
    snapshot, financial_repair_actions, financial_repair_actions_zh = (
        _repair_snapshot_financial_fields(
            ticker=ticker,
            snapshot=snapshot,
            allow_fetch=include_snapshot,
        )
    )
    snapshot, sec_repair_actions, sec_repair_actions_zh, sec_snapshot = (
        _repair_snapshot_from_sec(
            ticker=ticker,
            snapshot=snapshot,
            data_root=Path(data_root),
            allow_fetch=include_snapshot,
        )
    )
    data_repair_actions = (
        cache_repair_actions
        + data_repair_actions
        + financial_repair_actions
        + sec_repair_actions
    )
    data_repair_actions_zh = (
        cache_repair_actions_zh
        + data_repair_actions_zh
        + financial_repair_actions_zh
        + sec_repair_actions_zh
    )
    if include_snapshot:
        _save_snapshot_cache(ticker=ticker, snapshot=snapshot, data_root=Path(data_root))
    fundamental_trend = _build_fundamental_trend_context(
        ticker=ticker,
        data_root=data_root,
        as_of_date=prices["date"].max(),
        allow_fetch=include_snapshot,
    )
    fundamentals = build_fundamental_context(ticker, snapshot, trend_context=fundamental_trend)
    sentiment = build_sentiment_context(ticker, snapshot)
    analyst = build_analyst_context(ticker, snapshot)
    valuation = build_valuation_context(ticker, snapshot)
    sector = str((snapshot or {}).get("sector") or "")
    industry = str((snapshot or {}).get("industry") or "")
    screening_profile = _resolve_screening_profile(
        ticker=ticker,
        sector=sector,
        industry=industry,
        screening_thresholds=screening_thresholds,
        screening_config=screening_config,
    )
    sector_etf = choose_sector_etf(sector, industry)
    sector_prices, sector_source, sector_warnings = _download_sector_prices(
        sector_etf=sector_etf,
        period=period,
        data_root=Path(data_root),
    )
    sector_context = build_sector_context(
        target_prices=prices,
        sector_prices=sector_prices,
        sector=sector,
        industry=industry,
        sector_etf=sector_etf,
        lookback_days=60,
    )
    data_readiness = build_data_readiness_report(
        ticker=ticker,
        prices=prices,
        price_provider=price_result.provider,
        price_attempts=price_result.attempts,
        price_warnings=price_result.warnings,
        price_missing_tickers=price_result.missing_tickers,
        price_source_validation=price_result.source_validation,
        benchmark_prices=benchmark_prices,
        benchmark_sources=benchmark_sources,
        benchmark_warnings=benchmark_warnings,
        sector_etf=sector_etf,
        sector_prices=sector_prices,
        sector_source=sector_source,
        sector_warnings=sector_warnings,
        snapshot=snapshot,
        contexts={
            "fundamental": fundamentals,
            "valuation": valuation,
            "sentiment": sentiment,
            "event": event_risk,
            "analyst": analyst,
        },
    )
    probability_calibration_context = _load_probability_calibration_context(
        probability_calibration_path
    )
    signal_review_feedback_context = build_signal_review_feedback_context(
        ticker=ticker,
        review_root=Path(output_root).parent / "signal_review",
    )
    analysis = analyze_ticker(
        scored=scored,
        prices=prices,
        ticker=ticker,
        horizons=horizons,
        market_context=market_context,
        relative_strength_contexts=relative_strength_contexts,
        event_risk_context=event_risk,
        fundamental_context=fundamentals,
        sentiment_context=sentiment,
        analyst_context=analyst,
        valuation_context=valuation,
        sector_context=sector_context,
        screening_thresholds=screening_profile.thresholds,
        trading_rules=screening_profile.trading_rules,
        probability_calibration_context=probability_calibration_context,
        signal_review_feedback_context=signal_review_feedback_context,
    )
    analysis["screening_profile"] = screening_profile.name
    analysis["screening_profile_zh"] = screening_profile.name_zh

    # Position-aware context: if this ticker is a real holding, attach cost-basis,
    # weight, and concentration so the report speaks to the position, not just entry.
    _attach_position_context(analysis, ticker)

    should_upgrade, upgrade_reason, upgrade_reason_zh = _auto_period_upgrade_decision(
        requested_period=requested_period,
        effective_period=period,
        analysis=analysis,
    )
    if auto_period_upgrade and should_upgrade:
        return run_real_ticker_analysis(
            ticker=ticker,
            period="5y",
            horizons=horizons,
            output_root=output_root,
            data_root=data_root,
            include_snapshot=include_snapshot,
            include_peer_comparison=include_peer_comparison,
            peer_limit=peer_limit,
            screening_thresholds=screening_thresholds,
            screening_config=screening_config,
            auto_period_upgrade=False,
            probability_calibration_path=probability_calibration_path,
            _requested_period=requested_period,
            _auto_period_upgraded=True,
            _auto_period_upgrade_reason=upgrade_reason,
            _auto_period_upgrade_reason_zh=upgrade_reason_zh,
        )

    period_reason = (
        _auto_period_upgrade_reason
        if _auto_period_upgraded
        else "Requested period was used without automatic extension."
    )
    period_reason_zh = (
        _auto_period_upgrade_reason_zh
        if _auto_period_upgraded
        else "系统使用了你请求的数据周期，没有自动延长。"
    )
    analysis["requested_period"] = requested_period
    analysis["analysis_period"] = period
    analysis["auto_period_upgraded"] = _auto_period_upgraded
    analysis["auto_period_upgrade_reason"] = period_reason
    analysis["auto_period_upgrade_reason_zh"] = period_reason_zh
    analysis["data_repair_actions_applied"] = "; ".join(data_repair_actions) or "none"
    analysis["data_repair_actions_applied_zh"] = "；".join(data_repair_actions_zh) or "无"
    analysis = _apply_data_readiness_to_analysis(analysis, data_readiness)

    peer_comparison = pd.DataFrame()
    if include_peer_comparison and include_snapshot and snapshot:
        peer_comparison = _build_real_peer_comparison(
            ticker=ticker,
            period=period,
            horizons=horizons,
            output_dir=output_dir,
            data_root=Path(data_root),
            target_analysis=analysis,
            target_snapshot=snapshot,
            sector=sector,
            industry=industry,
            peer_limit=peer_limit,
            screening_config=screening_config,
        )

    analysis.to_csv(output_dir / "ticker_analysis.csv", index=False)
    data_readiness_report = render_data_readiness_report(data_readiness)
    report = render_ticker_analysis(analysis).rstrip() + "\n\n" + data_readiness_report
    if not peer_comparison.empty:
        report = report.rstrip() + "\n\n" + render_peer_comparison_report(peer_comparison)
    (output_dir / "ticker_analysis.md").write_text(report, encoding="utf-8")
    if not peer_comparison.empty:
        peer_comparison.to_csv(output_dir / "peer_comparison.csv", index=False)
        (output_dir / "peer_comparison.md").write_text(
            render_peer_comparison_report(peer_comparison),
            encoding="utf-8",
        )
    data_sources = {
        ticker: price_result.provider,
        **benchmark_sources,
        **({sector_etf: sector_source} if sector_etf else {}),
    }
    data_source_warnings = {
        ticker: price_result.warnings,
        **benchmark_warnings,
        **({sector_etf: sector_warnings} if sector_etf else {}),
    }
    write_json(
        output_dir / "data_sources.json",
        {
            "providers": data_sources,
            "warnings": {key: list(value) for key, value in data_source_warnings.items()},
        },
    )
    write_json(output_dir / "data_readiness.json", data_readiness_payload(data_readiness))
    data_readiness_frame(data_readiness).to_csv(output_dir / "data_readiness.csv", index=False)
    (output_dir / "data_readiness.md").write_text(data_readiness_report, encoding="utf-8")
    write_json(output_dir / "event_risk.json", event_risk.to_dict())
    write_json(output_dir / "fundamental_quality.json", fundamentals.to_dict())
    write_json(output_dir / "sentiment_risk.json", sentiment.to_dict())
    write_json(output_dir / "analyst_expectations.json", analyst.to_dict())
    write_json(output_dir / "valuation_risk.json", valuation.to_dict())

    if include_snapshot:
        write_json(output_dir / "external_snapshot.json", snapshot)
    if sec_snapshot is not None:
        write_json(output_dir / "sec_fundamentals.json", sec_snapshot.to_dict())

    signal_review_metadata: dict[str, object]
    try:
        signal_review = write_signal_review(
            analysis=analysis,
            prices=prices,
            review_root=Path(output_root).parent / "signal_review",
            source_report_path=output_dir / "ticker_analysis.md",
        )
        review_section = render_signal_review_section(
            signal_review.ticker_history,
            signal_review.summary,
            signal_review.report_path,
        )
        with (output_dir / "ticker_analysis.md").open("a", encoding="utf-8") as file:
            file.write("\n\n" + review_section)
        signal_review_metadata = {
            "status": "ok",
            "review_dir": str(signal_review.review_dir),
            "history_path": str(signal_review.history_path),
            "report_path": str(signal_review.report_path),
            **summarize_ticker_signal_review(signal_review.ticker_history, signal_review.summary),
        }
    except Exception as exc:
        signal_review_metadata = {
            "status": "failed",
            "warning": str(exc),
        }

    cache_metadata = _build_cache_metadata(
        ticker=ticker,
        requested_period=requested_period,
        period=period,
        data_root=Path(data_root),
        output_dir=output_dir,
        price_file=data_path,
        prices=prices,
        price_provider=price_result.provider,
        price_warnings=price_result.warnings,
        benchmark_prices=benchmark_prices,
        benchmark_sources=benchmark_sources,
        benchmark_warnings=benchmark_warnings,
        sector_etf=sector_etf,
        sector_prices=sector_prices,
        sector_source=sector_source,
        sector_warnings=sector_warnings,
        data_repair_actions=data_repair_actions,
        data_repair_actions_zh=data_repair_actions_zh,
        data_readiness=data_readiness,
        signal_review_metadata=signal_review_metadata,
    )
    write_json(output_dir / "cache_metadata.json", cache_metadata)
    write_json(
        output_dir / "analysis_result.json",
        _build_analysis_result_payload(
            ticker=ticker,
            requested_period=requested_period,
            period=period,
            output_dir=output_dir,
            analysis=analysis,
            peer_comparison=peer_comparison,
            snapshot=snapshot,
            data_sources=data_sources,
            data_source_warnings=data_source_warnings,
            event_risk=event_risk,
            fundamentals=fundamentals,
            sentiment=sentiment,
            analyst=analyst,
            valuation=valuation,
            sector_context=sector_context,
            data_readiness=data_readiness,
            sec_snapshot=sec_snapshot,
            data_repair_actions=data_repair_actions,
            data_repair_actions_zh=data_repair_actions_zh,
            cache_metadata=cache_metadata,
            screening_profile=screening_profile,
            screening_config=screening_config,
        ),
    )

    return RealTickerAnalysisResult(
        ticker=ticker,
        requested_period=requested_period,
        effective_period=period,
        auto_period_upgraded=_auto_period_upgraded,
        auto_period_upgrade_reason=period_reason,
        auto_period_upgrade_reason_zh=period_reason_zh,
        prices=prices,
        scored=scored,
        analysis=analysis,
        output_dir=output_dir,
        snapshot=snapshot,
        benchmark_prices=benchmark_prices,
        data_sources=data_sources,
        data_source_warnings=data_source_warnings,
        event_risk=event_risk,
        fundamentals=fundamentals,
        sentiment=sentiment,
        analyst=analyst,
        valuation=valuation,
        sector_context=sector_context,
        peer_comparison=peer_comparison,
        data_readiness=data_readiness,
        cache_metadata=cache_metadata,
    )


def normalize_ticker(ticker: str) -> str:
    normalized = ticker.upper().strip()
    return TICKER_ALIASES.get(normalized, normalized)


def _repair_snapshot_classification(
    ticker: str,
    snapshot: dict[str, object] | None,
) -> tuple[dict[str, object] | None, tuple[str, ...], tuple[str, ...]]:
    fallback = TICKER_CLASSIFICATION_FALLBACKS.get(ticker.upper().strip())
    if not fallback:
        return snapshot, (), ()
    repaired = dict(snapshot or {})
    actions: list[str] = []
    actions_zh: list[str] = []
    if not str(repaired.get("sector") or "").strip():
        repaired["sector"] = fallback["sector"]
        actions.append("filled missing sector from local ticker classification")
        actions_zh.append("用本地ticker分类补齐缺失的板块")
    if not str(repaired.get("industry") or "").strip():
        repaired["industry"] = fallback["industry"]
        actions.append("filled missing industry from local ticker classification")
        actions_zh.append("用本地ticker分类补齐缺失的行业")
    if actions:
        repaired["classification_repair_source"] = "local_ticker_classification"
    return repaired, tuple(actions), tuple(actions_zh)


FUNDAMENTAL_REPAIR_FIELDS = (
    "revenue_growth",
    "earnings_growth",
    "profit_margin",
    "return_on_equity",
    "free_cash_flow",
    "revenue",
    "net_income",
    "operating_cash_flow",
    "capital_expenditure",
    "shareholders_equity",
    "forward_pe",
    "trailing_pe",
    "peg_ratio",
    "debt_to_equity",
    "sec_cik",
    "sec_data_source",
    "sec_fetched_at_utc",
    "sec_latest_revenue_period",
    "sec_latest_net_income_period",
    "sec_latest_balance_sheet_period",
)

VALUATION_REPAIR_FIELDS = (
    "forward_pe",
    "trailing_pe",
    "peg_ratio",
    "revenue_growth",
    "earnings_growth",
    "profit_margin",
    "free_cash_flow_yield",
)

SNAPSHOT_CACHE_REPAIR_FIELDS = tuple(
    dict.fromkeys(
        (
            "company_name",
            "sector",
            "industry",
            "current_price",
            "market_cap",
            "trailing_pe",
            "gross_margin",
            "operating_margin",
            "total_cash",
            "total_debt",
            "shareholders_equity",
        )
        + FUNDAMENTAL_REPAIR_FIELDS
        + VALUATION_REPAIR_FIELDS
        + (
            "target_mean_price",
            "target_high_price",
            "target_low_price",
            "recommendation_mean",
            "recommendation_key",
            "number_of_analysts",
            "analyst_upside",
            "news_titles",
            "sec_cik",
            "sec_data_source",
            "sec_fetched_at_utc",
            "sec_latest_revenue_period",
            "sec_latest_net_income_period",
            "sec_latest_balance_sheet_period",
            "sec_fundamental_coverage",
            "sec_fundamental_warnings",
            "sec_raw_cache_path",
            "sec_extracted_cache_path",
            "free_cash_flow_yield",
        )
    )
)


def _repair_snapshot_from_cache(
    ticker: str,
    snapshot: dict[str, object] | None,
    data_root: Path,
) -> tuple[dict[str, object] | None, tuple[str, ...], tuple[str, ...]]:
    cached_snapshot = _load_snapshot_cache(ticker=ticker, data_root=data_root)
    if not cached_snapshot:
        return snapshot, (), ()

    repaired = dict(snapshot or {})
    filled_fields = _merge_missing_snapshot_fields(repaired, cached_snapshot)
    if not filled_fields:
        return snapshot, (), ()

    repaired["snapshot_cache_repair_source"] = str(_snapshot_cache_path(data_root, ticker))
    filled_text = ", ".join(filled_fields)
    filled_text_zh = "、".join(filled_fields)
    return (
        repaired,
        (f"filled missing snapshot fields from local cache: {filled_text}",),
        (f"通过本地快照缓存补齐字段：{filled_text_zh}",),
    )


def _load_snapshot_cache(ticker: str, data_root: Path) -> dict[str, object] | None:
    path = _snapshot_cache_path(data_root, ticker)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"- {ticker}: snapshot cache unreadable, refetching ({exc})", file=sys.stderr)
        return None
    if not isinstance(payload, dict):
        return None
    cached_ticker = str(payload.get("ticker") or "").upper().strip()
    if cached_ticker and cached_ticker != ticker.upper().strip():
        return None
    if _snapshot_cache_is_stale(payload):
        return None
    return payload


def _save_snapshot_cache(
    ticker: str,
    snapshot: dict[str, object] | None,
    data_root: Path,
) -> None:
    if not _snapshot_cache_worth_saving(snapshot):
        return
    payload = dict(snapshot or {})
    payload["ticker"] = ticker.upper().strip()
    payload["snapshot_cache_saved_at_utc"] = _utc_now()
    write_json(_snapshot_cache_path(data_root, ticker), payload)


def _snapshot_cache_worth_saving(snapshot: dict[str, object] | None) -> bool:
    if not snapshot:
        return False
    return any(_has_snapshot_value(snapshot.get(field)) for field in SNAPSHOT_CACHE_REPAIR_FIELDS)


def _snapshot_cache_path(data_root: Path, ticker: str) -> Path:
    cache_root = data_root.parent if data_root.name == "real_prices" else data_root
    return cache_root / "real_snapshots" / f"{ticker.upper().strip()}_snapshot.json"


def _snapshot_cache_is_stale(payload: dict[str, object]) -> bool:
    timestamp = (
        payload.get("snapshot_cache_saved_at_utc")
        or payload.get("fetched_at_utc")
    )
    if not isinstance(timestamp, str) or not timestamp.strip():
        return True
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return True
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    age = datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)
    return age > timedelta(days=SNAPSHOT_CACHE_MAX_AGE_DAYS)


def _repair_snapshot_financial_fields(
    ticker: str,
    snapshot: dict[str, object] | None,
    allow_fetch: bool,
) -> tuple[dict[str, object] | None, tuple[str, ...], tuple[str, ...]]:
    if not allow_fetch:
        return snapshot, (), ()

    fundamental_coverage = _field_coverage(snapshot, FUNDAMENTAL_REPAIR_FIELDS)
    valuation_coverage = _field_coverage(snapshot, VALUATION_REPAIR_FIELDS)
    if fundamental_coverage >= 0.25 and valuation_coverage >= 0.25:
        return snapshot, (), ()

    repaired = dict(snapshot or {})
    try:
        retry_snapshot = fetch_yfinance_snapshot(ticker)
    except Exception:
        return (
            repaired,
            ("retried yfinance snapshot for financial fields but fetch failed",),
            ("已尝试重新拉取yfinance财务字段，但拉取失败",),
        )

    filled_fields = _merge_missing_snapshot_fields(repaired, retry_snapshot)
    if not filled_fields:
        return (
            repaired,
            ("retried yfinance snapshot for financial fields but no missing fields improved",),
            ("已尝试重新拉取yfinance财务字段，但缺失字段没有改善",),
        )

    repaired["financial_repair_source"] = "yfinance_snapshot_retry"
    filled_text = ", ".join(filled_fields)
    filled_text_zh = "、".join(filled_fields)
    return (
        repaired,
        (f"filled missing financial fields from yfinance retry: {filled_text}",),
        (f"通过yfinance二次拉取补齐财务字段：{filled_text_zh}",),
    )


def _repair_snapshot_from_sec(
    ticker: str,
    snapshot: dict[str, object] | None,
    data_root: Path,
    allow_fetch: bool,
) -> tuple[
    dict[str, object] | None,
    tuple[str, ...],
    tuple[str, ...],
    SecFundamentalSnapshot | None,
]:
    if not allow_fetch:
        return snapshot, (), (), None

    repaired = dict(snapshot or {})
    try:
        sec_snapshot = fetch_sec_fundamental_snapshot(
            ticker=ticker,
            data_root=data_root,
        )
    except Exception as exc:
        return (
            repaired or snapshot,
            (f"SEC companyfacts fetch failed: {exc}",),
            (f"SEC companyfacts 拉取失败：{exc}",),
            None,
        )

    filled_fields = list(_merge_missing_snapshot_fields(repaired, sec_snapshot.fields))
    _attach_sec_metadata(repaired, sec_snapshot)
    derived_fields = list(_derive_snapshot_valuation_fields(repaired))
    if sec_snapshot.data_coverage > 0:
        current_source = str(repaired.get("data_source") or "current_snapshot")
        repaired["fundamental_data_source"] = _combine_source_names(
            current_source,
            "sec_companyfacts_current_snapshot",
        )
        repaired["valuation_data_source"] = _combine_source_names(
            current_source,
            "sec_companyfacts_current_snapshot",
        )

    changed_fields = tuple(dict.fromkeys(filled_fields + derived_fields))
    if not changed_fields:
        return (
            repaired,
            ("SEC companyfacts checked but no missing fields improved",),
            ("已检查SEC companyfacts，但缺失字段没有改善",),
            sec_snapshot,
        )

    changed_text = ", ".join(changed_fields)
    changed_text_zh = "、".join(changed_fields)
    return (
        repaired,
        (f"filled missing financial fields from SEC companyfacts: {changed_text}",),
        (f"通过SEC companyfacts补齐财务字段：{changed_text_zh}",),
        sec_snapshot,
    )


def _attach_sec_metadata(
    target: dict[str, object],
    sec_snapshot: SecFundamentalSnapshot,
) -> None:
    target["sec_cik"] = sec_snapshot.cik
    target["sec_data_source"] = sec_snapshot.source
    target["sec_fetched_at_utc"] = sec_snapshot.fetched_at_utc
    target["sec_fundamental_coverage"] = sec_snapshot.data_coverage
    target["sec_fundamental_warnings"] = list(sec_snapshot.warnings)
    target["sec_raw_cache_path"] = sec_snapshot.raw_cache_path
    target["sec_extracted_cache_path"] = sec_snapshot.extracted_cache_path


def _derive_snapshot_valuation_fields(snapshot: dict[str, object]) -> tuple[str, ...]:
    filled: list[str] = []
    market_cap = _safe_float(snapshot.get("market_cap"))
    net_income = _safe_float(snapshot.get("net_income"))
    free_cash_flow = _safe_float(snapshot.get("free_cash_flow"))
    if (
        market_cap is not None
        and market_cap > 0
        and net_income is not None
        and net_income > 0
        and not _has_snapshot_value(snapshot.get("trailing_pe"))
    ):
        snapshot["trailing_pe"] = market_cap / net_income
        filled.append("trailing_pe")
    if (
        market_cap is not None
        and market_cap > 0
        and free_cash_flow is not None
        and not _has_snapshot_value(snapshot.get("free_cash_flow_yield"))
    ):
        snapshot["free_cash_flow_yield"] = free_cash_flow / market_cap
        filled.append("free_cash_flow_yield")
    return tuple(filled)


def _combine_source_names(left: str, right: str) -> str:
    parts = []
    for value in (left, right):
        for part in str(value).replace("+", ",").split(","):
            cleaned = part.strip()
            if cleaned and cleaned not in parts:
                parts.append(cleaned)
    return "+".join(parts) if parts else right


def _field_coverage(snapshot: dict[str, object] | None, fields: tuple[str, ...]) -> float:
    if not snapshot:
        return 0.0
    available = sum(_has_snapshot_value(snapshot.get(field)) for field in fields)
    return available / len(fields) if fields else 0.0


def _merge_missing_snapshot_fields(
    target: dict[str, object],
    source: dict[str, object] | None,
) -> tuple[str, ...]:
    if not source:
        return ()
    filled: list[str] = []
    for field in SNAPSHOT_CACHE_REPAIR_FIELDS:
        if _has_snapshot_value(target.get(field)):
            continue
        value = source.get(field)
        if not _has_snapshot_value(value):
            continue
        target[field] = value
        filled.append(field)
    return tuple(filled)


def _has_snapshot_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        return True
    return True


def _safe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.replace(",", "").replace("%", "").strip()
        if pd.isna(value):
            return None
        numeric = float(value)
    except Exception:
        return None
    return numeric if np.isfinite(numeric) else None


def _auto_period_upgrade_decision(
    requested_period: str,
    effective_period: str,
    analysis: pd.DataFrame,
    data_quality_floor: float = 70.0,
) -> tuple[bool, str, str]:
    if not _period_is_shorter_than_5y(effective_period):
        return False, "", ""
    if analysis.empty or "data_quality_score" not in analysis.columns:
        return False, "", ""
    focus = analysis.sort_values("high_probability_score", ascending=False).iloc[0]
    data_quality_score = float(focus.get("data_quality_score", 0.0))
    if data_quality_score >= data_quality_floor:
        return False, "", ""
    reason = (
        f"Data quality score was {data_quality_score:.1f} using {effective_period}. "
        "The system automatically re-ran the analysis with 5y data before writing the final report."
    )
    reason_zh = (
        f"使用{effective_period}数据时，数据质量分数为{data_quality_score:.1f}。"
        "系统已自动改用5年数据重新分析，再生成最终报告。"
    )
    return True, reason, reason_zh


def _period_is_shorter_than_5y(period: str) -> bool:
    months = _period_to_months(period)
    return months is not None and months < 60


def _period_to_months(period: str) -> int | None:
    value = period.strip().lower()
    if value in {"max", "ytd"}:
        return None
    if value.endswith("mo"):
        try:
            return int(value[:-2])
        except ValueError:
            return None
    try:
        number = int(value[:-1])
    except (TypeError, ValueError):
        return None
    unit = value[-1:]
    if unit == "d":
        return max(1, round(number / 21))
    if unit == "m":
        return number
    if unit == "y":
        return number * 12
    return None


def _load_probability_calibration_context(path: str | Path | None = None) -> pd.DataFrame:
    calibration_path = _resolve_probability_calibration_path(path)
    if calibration_path is None or not calibration_path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(calibration_path)
    except Exception as exc:
        # 校准表读不出来会静默退回未校准概率——必须在日志里可见。
        print(f"- probability calibration unreadable ({calibration_path}): {exc}", file=sys.stderr)
        return pd.DataFrame()
    required = {
        "probability_bucket",
        "sample_count",
        "recommended_probability_adjustment",
    }
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    return frame


def _resolve_probability_calibration_path(path: str | Path | None = None) -> Path | None:
    if path is not None:
        return Path(path)
    preferred = Path("outputs/walk_forward/latest/probability_calibration.csv")
    if preferred.exists():
        return preferred
    candidates = sorted(
        Path("outputs/walk_forward").glob("*/probability_calibration.csv"),
        key=lambda item: item.stat().st_mtime if item.exists() else 0.0,
        reverse=True,
    )
    return candidates[0] if candidates else preferred


def _download_benchmark_prices(
    period: str,
    data_root: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, str], dict[str, tuple[str, ...]]]:
    ticker_to_file = {
        "SPY": "SPY",
        "QQQ": "QQQ",
        "^VIX": "VIX",
    }
    benchmark_prices: dict[str, pd.DataFrame] = {}
    benchmark_sources: dict[str, str] = {}
    benchmark_warnings: dict[str, tuple[str, ...]] = {}
    for ticker, file_stem in ticker_to_file.items():
        try:
            result = download_prices_for_period_multi_source(
                [ticker],
                period=period,
                output_path=data_root / f"{file_stem}_{period}.csv",
            )
            benchmark_prices[ticker] = result.prices
            benchmark_sources[ticker] = result.provider
            benchmark_warnings[ticker] = result.warnings
        except Exception as exc:
            print(f"- benchmark {ticker}: price fetch failed ({exc})", file=sys.stderr)
            benchmark_prices[ticker] = pd.DataFrame()
            benchmark_sources[ticker] = "unavailable"
            benchmark_warnings[ticker] = ("all providers failed",)
    return benchmark_prices, benchmark_sources, benchmark_warnings


def _download_sector_prices(
    sector_etf: str,
    period: str,
    data_root: Path,
) -> tuple[pd.DataFrame, str, tuple[str, ...]]:
    if not sector_etf:
        return pd.DataFrame(), "unavailable", ("sector ETF unavailable",)
    try:
        result = download_prices_for_period_multi_source(
            [sector_etf],
            period=period,
            output_path=data_root / f"{sector_etf}_{period}.csv",
        )
        return result.prices, result.provider, result.warnings
    except Exception:
        return pd.DataFrame(), "unavailable", ("all providers failed",)


def _build_real_peer_comparison(
    ticker: str,
    period: str,
    horizons: tuple[str, ...],
    output_dir: Path,
    data_root: Path,
    target_analysis: pd.DataFrame,
    target_snapshot: dict[str, object],
    sector: str,
    industry: str,
    peer_limit: int,
    screening_config: ScreeningConfig | None = None,
) -> pd.DataFrame:
    peer_tickers = choose_peer_tickers(
        ticker=ticker,
        sector=sector,
        industry=industry,
        max_peers=peer_limit,
    )
    if not peer_tickers:
        return pd.DataFrame()

    peer_results: list[RealTickerAnalysisResult] = []
    peer_output_root = output_dir / "peers"
    for peer_ticker in peer_tickers:
        try:
            peer_results.append(
                run_real_ticker_analysis(
                    ticker=peer_ticker,
                    period=period,
                    horizons=horizons,
                    output_root=peer_output_root,
                    data_root=data_root,
                    include_snapshot=True,
                    include_peer_comparison=False,
                    peer_limit=0,
                    screening_config=screening_config,
                    probability_calibration_path=None,
                )
            )
        except Exception:
            continue

    if not peer_results:
        return pd.DataFrame()

    return build_peer_comparison_frame(
        target_ticker=ticker,
        target_analysis=target_analysis,
        target_snapshot=target_snapshot,
        peer_results=peer_results,
    )


def _resolve_screening_profile(
    ticker: str,
    sector: str,
    industry: str,
    screening_thresholds: ScreeningThresholds | None,
    screening_config: ScreeningConfig | None,
) -> ScreeningProfile:
    if screening_thresholds is not None:
        return ScreeningProfile(
            name="custom",
            name_zh="自定义规则",
            thresholds=screening_thresholds,
        )
    config = screening_config or default_screening_config()
    return config.resolve_profile(ticker=ticker, sector=sector, industry=industry)


def _apply_data_readiness_to_analysis(
    analysis: pd.DataFrame,
    data_readiness: DataReadinessReport,
) -> pd.DataFrame:
    result = analysis.copy()
    blockers = "; ".join(data_readiness.primary_blockers) or "none"
    blockers_zh = "；".join(data_readiness.primary_blockers_zh) or "无"
    result["data_readiness_level"] = data_readiness.overall_status
    result["data_readiness_level_zh"] = data_readiness.overall_status_zh
    result["data_readiness_score"] = float(data_readiness.overall_score)
    result["data_readiness_repair_priority"] = data_readiness.repair_priority
    result["data_readiness_repair_priority_zh"] = data_readiness.repair_priority_zh
    result["data_readiness_primary_blockers"] = blockers
    result["data_readiness_primary_blockers_zh"] = blockers_zh
    result["data_source_validation_status"] = str(
        data_readiness.source_validation.get("status", "unknown")
    )
    result["data_source_validation_status_zh"] = str(
        data_readiness.source_validation.get("status_zh", "未知")
    )
    result["data_needs_repair"] = data_readiness.overall_status in {
        "insufficient",
        "conflict_warning",
    }

    if not _data_readiness_blocks_high_probability(data_readiness):
        return result

    reason, reason_zh = _data_readiness_gate_reason(data_readiness)
    for index in result.index:
        result.at[index, "quality_gate_passed"] = False
        result.at[index, "screening_action"] = "not_high_probability_setup_now"
        result.at[index, "screening_action_zh"] = "当前不是高概率机会"
        result.at[index, "calibrated_quality_gate_passed"] = False
        result.at[index, "calibrated_screening_action"] = "not_calibrated_high_probability_now"
        result.at[index, "calibrated_screening_action_zh"] = "校准后当前仍不是高概率机会"
        result.at[index, "quality_gate_fail_reasons"] = _append_reason(
            result.at[index, "quality_gate_fail_reasons"],
            reason,
        )
        result.at[index, "quality_gate_fail_reasons_zh"] = _append_reason(
            result.at[index, "quality_gate_fail_reasons_zh"],
            reason_zh,
            separator="；",
        )
        result.at[index, "calibrated_quality_gate_fail_reasons"] = _append_reason(
            result.at[index, "calibrated_quality_gate_fail_reasons"],
            reason,
        )
        result.at[index, "calibrated_quality_gate_fail_reasons_zh"] = _append_reason(
            result.at[index, "calibrated_quality_gate_fail_reasons_zh"],
            reason_zh,
            separator="；",
        )
    return result


def _data_readiness_blocks_high_probability(data_readiness: DataReadinessReport) -> bool:
    if data_readiness.overall_status in {"insufficient", "conflict_warning"}:
        return True
    return float(data_readiness.overall_score) < 70.0


def _data_readiness_gate_reason(
    data_readiness: DataReadinessReport,
) -> tuple[str, str]:
    if data_readiness.overall_status == "conflict_warning":
        return (
            "data source conflict warning",
            "数据源存在冲突",
        )
    if data_readiness.primary_blockers:
        return (
            f"data source readiness insufficient: {', '.join(data_readiness.primary_blockers)}",
            f"数据源准备度不足：{'、'.join(data_readiness.primary_blockers_zh)}",
        )
    return (
        f"data source readiness score below 70: {data_readiness.overall_score:.1f}",
        f"数据源准备度分数低于70：{data_readiness.overall_score:.1f}",
    )


def _append_reason(value: object, reason: str, separator: str = "; ") -> str:
    text = str(value or "").strip()
    if not text or text in {"none", "all strict quality gates passed", "所有严格质量门槛通过"}:
        return reason
    if reason in text:
        return text
    return f"{text}{separator}{reason}"


def _build_analysis_result_payload(
    ticker: str,
    requested_period: str,
    period: str,
    output_dir: Path,
    analysis: pd.DataFrame,
    peer_comparison: pd.DataFrame,
    snapshot: dict[str, object] | None,
    data_sources: dict[str, str],
    data_source_warnings: dict[str, tuple[str, ...]],
    event_risk: EventRiskContext,
    fundamentals: FundamentalContext,
    sentiment: SentimentContext,
    analyst: AnalystContext,
    valuation: ValuationContext,
    sector_context: SectorContext,
    data_readiness: DataReadinessReport,
    sec_snapshot: SecFundamentalSnapshot | None,
    data_repair_actions: tuple[str, ...],
    data_repair_actions_zh: tuple[str, ...],
    cache_metadata: dict[str, object],
    screening_profile: ScreeningProfile,
    screening_config: ScreeningConfig | None,
) -> dict[str, object]:
    focus = analysis.sort_values("high_probability_score", ascending=False).iloc[0]
    return {
        "generated_at_utc": _utc_now(),
        "ticker": ticker,
        "requested_period": requested_period,
        "period": period,
        "effective_period": period,
        "auto_period_upgraded": bool(focus.get("auto_period_upgraded", False)),
        "auto_period_upgrade_reason": focus.get("auto_period_upgrade_reason", ""),
        "auto_period_upgrade_reason_zh": focus.get("auto_period_upgrade_reason_zh", ""),
        "latest_date": _latest_date(analysis, "date"),
        "latest_price": float(focus["latest_price"]),
        "final_decision": focus["final_decision"],
        "final_decision_zh": focus["final_decision_zh"],
        "final_focus_horizon": focus["final_focus_horizon"],
        "final_focus_horizon_zh": focus["final_focus_horizon_zh"],
        "final_score": float(focus["final_score"]),
        "final_watchlist_status": focus["final_watchlist_status"],
        "final_watchlist_status_zh": focus["final_watchlist_status_zh"],
        "final_reason": focus["final_reason"],
        "final_reason_zh": focus["final_reason_zh"],
        "final_next_step": focus["final_next_step"],
        "final_next_step_zh": focus["final_next_step_zh"],
        "primary_blocker": focus["primary_blocker"],
        "primary_blocker_zh": focus["primary_blocker_zh"],
        "priority_blockers": focus["priority_blockers"],
        "priority_blockers_zh": focus["priority_blockers_zh"],
        "priority_blocker_count": int(focus["priority_blocker_count"]),
        "priority_blocker_note": focus["priority_blocker_note"],
        "priority_blocker_note_zh": focus["priority_blocker_note_zh"],
        "primary_blocker_resolution": focus["primary_blocker_resolution"],
        "primary_blocker_resolution_zh": focus["primary_blocker_resolution_zh"],
        "blocker_resolution_steps": focus["blocker_resolution_steps"],
        "blocker_resolution_steps_zh": focus["blocker_resolution_steps_zh"],
        "blocker_recheck_trigger": focus["blocker_recheck_trigger"],
        "blocker_recheck_trigger_zh": focus["blocker_recheck_trigger_zh"],
        "primary_blocker_progress": float(focus["primary_blocker_progress"]),
        "blocker_resolution_score": float(focus["blocker_resolution_score"]),
        "blocker_resolution_level": focus["blocker_resolution_level"],
        "blocker_resolution_level_zh": focus["blocker_resolution_level_zh"],
        "blocker_resolution_gap": focus["blocker_resolution_gap"],
        "blocker_resolution_gap_zh": focus["blocker_resolution_gap_zh"],
        "horizon_alignment_score": float(focus["horizon_alignment_score"]),
        "horizon_alignment_label": focus["horizon_alignment_label"],
        "horizon_alignment_label_zh": focus["horizon_alignment_label_zh"],
        "constructive_horizon_count": int(focus["constructive_horizon_count"]),
        "weak_horizon_count": int(focus["weak_horizon_count"]),
        "risk_wait_horizon_count": int(focus["risk_wait_horizon_count"]),
        "horizon_signal_score_spread": float(focus["horizon_signal_score_spread"]),
        "horizon_alignment_note": focus["horizon_alignment_note"],
        "horizon_alignment_note_zh": focus["horizon_alignment_note_zh"],
        "market_regime": focus["market_regime"],
        "market_regime_zh": focus["market_regime_zh"],
        "market_regime_note": focus["market_regime_note"],
        "market_regime_note_zh": focus["market_regime_note_zh"],
        "market_regime_signal_delta": float(focus["market_regime_signal_delta"]),
        "market_regime_confidence_delta": float(focus["market_regime_confidence_delta"]),
        "market_regime_sample_delta": int(focus["market_regime_sample_delta"]),
        "market_regime_win_rate_delta": float(focus["market_regime_win_rate_delta"]),
        "market_regime_average_return_delta": float(
            focus["market_regime_average_return_delta"]
        ),
        "entry_readiness_gate_passed": bool(focus["entry_readiness_gate_passed"]),
        "entry_readiness_status": focus["entry_readiness_status"],
        "entry_readiness_status_zh": focus["entry_readiness_status_zh"],
        "entry_readiness_score": float(focus["entry_readiness_score"]),
        "entry_readiness_reason": focus["entry_readiness_reason"],
        "entry_readiness_reason_zh": focus["entry_readiness_reason_zh"],
        "entry_readiness_note": focus["entry_readiness_note"],
        "entry_readiness_note_zh": focus["entry_readiness_note_zh"],
        "trade_plan_quality_gate_passed": bool(focus["trade_plan_quality_gate_passed"]),
        "trade_plan_quality_status": focus["trade_plan_quality_status"],
        "trade_plan_quality_status_zh": focus["trade_plan_quality_status_zh"],
        "trade_plan_quality_score": float(focus["trade_plan_quality_score"]),
        "trade_plan_quality_reason": focus["trade_plan_quality_reason"],
        "trade_plan_quality_reason_zh": focus["trade_plan_quality_reason_zh"],
        "trade_plan_quality_note": focus["trade_plan_quality_note"],
        "trade_plan_quality_note_zh": focus["trade_plan_quality_note_zh"],
        "primary_decision": focus["primary_decision"],
        "primary_decision_zh": focus["primary_decision_zh"],
        "focus_horizon": focus["horizon"],
        "screening_action": focus["screening_action"],
        "screening_action_zh": focus["screening_action_zh"],
        "screening_profile": screening_profile.name,
        "screening_profile_zh": screening_profile.name_zh,
        "trading_rule_entry_style": focus["trading_rule_entry_style"],
        "trading_rule_entry_style_zh": focus["trading_rule_entry_style_zh"],
        "trading_rule_atr_stop_multiple": float(focus["trading_rule_atr_stop_multiple"]),
        "trading_rule_target_r_multiple": float(focus["trading_rule_target_r_multiple"]),
        "trading_rule_max_chase_pct": float(focus["trading_rule_max_chase_pct"]),
        "trading_rule_time_stop_days": int(focus["trading_rule_time_stop_days"]),
        "trading_rule_trailing_stop_trigger_r": float(
            focus["trading_rule_trailing_stop_trigger_r"]
        ),
        "trading_rule_trailing_stop_lock_r": float(
            focus["trading_rule_trailing_stop_lock_r"]
        ),
        "trading_rule_sell_rule": focus["trading_rule_sell_rule"],
        "trading_rule_sell_rule_zh": focus["trading_rule_sell_rule_zh"],
        "quality_gate_passed": bool(focus["quality_gate_passed"]),
        "high_probability_score": float(focus["high_probability_score"]),
        "calibrated_win_probability": float(focus["calibrated_win_probability"]),
        "calibrated_win_probability_raw": float(
            focus["calibrated_win_probability_raw"]
        ),
        "probability_calibration_adjustment": float(
            focus["probability_calibration_adjustment"]
        ),
        "probability_calibration_source": focus["probability_calibration_source"],
        "probability_calibration_source_zh": focus["probability_calibration_source_zh"],
        "probability_calibration_sample_count": int(
            focus["probability_calibration_sample_count"]
        ),
        "probability_calibration_action": focus["probability_calibration_action"],
        "probability_calibration_action_zh": focus[
            "probability_calibration_action_zh"
        ],
        "probability_calibration_note": focus["probability_calibration_note"],
        "probability_calibration_note_zh": focus["probability_calibration_note_zh"],
        "calibrated_probability_level": focus["calibrated_probability_level"],
        "calibrated_probability_level_zh": focus["calibrated_probability_level_zh"],
        "calibrated_probability_confidence": float(
            focus["calibrated_probability_confidence"]
        ),
        "calibrated_probability_confidence_level": focus[
            "calibrated_probability_confidence_level"
        ],
        "calibrated_probability_confidence_level_zh": focus[
            "calibrated_probability_confidence_level_zh"
        ],
        "calibrated_probability_note": focus["calibrated_probability_note"],
        "calibrated_probability_note_zh": focus["calibrated_probability_note_zh"],
        "screening_backtest_entry_type": focus["screening_backtest_entry_type"],
        "screening_backtest_trade_count": int(focus["screening_backtest_trade_count"]),
        "screening_backtest_win_rate": float(focus["screening_backtest_win_rate"]),
        "screening_backtest_stop_hit_rate": float(
            focus["screening_backtest_stop_hit_rate"]
        ),
        "screening_backtest_average_return": float(
            focus["screening_backtest_average_return"]
        ),
        "calibrated_screening_action": focus["calibrated_screening_action"],
        "calibrated_screening_action_zh": focus["calibrated_screening_action_zh"],
        "calibrated_quality_gate_passed": bool(focus["calibrated_quality_gate_passed"]),
        "calibrated_high_probability_score": float(
            focus["calibrated_high_probability_score"]
        ),
        "calibrated_quality_gate_fail_reasons": focus[
            "calibrated_quality_gate_fail_reasons"
        ],
        "calibrated_quality_gate_fail_reasons_zh": focus[
            "calibrated_quality_gate_fail_reasons_zh"
        ],
        "watchlist_status": focus["watchlist_status"],
        "watchlist_status_zh": focus["watchlist_status_zh"],
        "quality_gate_fail_reasons": focus["quality_gate_fail_reasons"],
        "quality_gate_fail_reasons_zh": focus["quality_gate_fail_reasons_zh"],
        "data_quality_score": float(focus["data_quality_score"]),
        "data_quality_level": focus["data_quality_level"],
        "data_quality_level_zh": focus["data_quality_level_zh"],
        "data_quality_weakest_layer": focus["data_quality_weakest_layer"],
        "data_quality_weakest_layer_zh": focus["data_quality_weakest_layer_zh"],
        "data_quality_weak_layers": focus["data_quality_weak_layers"],
        "data_quality_weak_layers_zh": focus["data_quality_weak_layers_zh"],
        "data_quality_repair_actions": focus["data_quality_repair_actions"],
        "data_quality_repair_actions_zh": focus["data_quality_repair_actions_zh"],
        "data_quality_repair_priority": focus["data_quality_repair_priority"],
        "data_quality_repair_priority_zh": focus["data_quality_repair_priority_zh"],
        "data_repair_actions_applied": list(data_repair_actions),
        "data_repair_actions_applied_zh": list(data_repair_actions_zh),
        "price_health_score": float(focus["price_health_score"]),
        "price_health_level": focus["price_health_level"],
        "price_health_level_zh": focus["price_health_level_zh"],
        "price_health_issue_count": int(focus["price_health_issue_count"]),
        "price_max_calendar_gap_days": int(focus["price_max_calendar_gap_days"]),
        "price_large_gap_count": int(focus["price_large_gap_count"]),
        "price_zero_volume_days": int(focus["price_zero_volume_days"]),
        "price_missing_ohlcv_rows": int(focus["price_missing_ohlcv_rows"]),
        "price_extreme_return_count": int(focus["price_extreme_return_count"]),
        "price_health_note": focus["price_health_note"],
        "price_health_note_zh": focus["price_health_note_zh"],
        "backtest_trust_score": float(focus["backtest_trust_score"]),
        "backtest_time_stop_days": int(focus["backtest_time_stop_days"]),
        "backtest_trailing_stop_trigger_r": float(
            focus["backtest_trailing_stop_trigger_r"]
        ),
        "backtest_trailing_stop_lock_r": float(focus["backtest_trailing_stop_lock_r"]),
        "backtest_trust_level": focus["backtest_trust_level"],
        "backtest_trust_level_zh": focus["backtest_trust_level_zh"],
        "backtest_sample_score": float(focus["backtest_sample_score"]),
        "backtest_liquidity_score": float(focus["backtest_liquidity_score"]),
        "backtest_slippage_score": float(focus["backtest_slippage_score"]),
        "backtest_return_evidence_score": float(focus["backtest_return_evidence_score"]),
        "regime_coverage_score": float(focus["regime_coverage_score"]),
        "regime_coverage_level": focus["regime_coverage_level"],
        "regime_coverage_level_zh": focus["regime_coverage_level_zh"],
        "regime_coverage_regime_count": int(focus["regime_coverage_regime_count"]),
        "regime_coverage_dominant_regime": focus["regime_coverage_dominant_regime"],
        "regime_coverage_dominant_regime_zh": focus[
            "regime_coverage_dominant_regime_zh"
        ],
        "regime_coverage_dominant_share": float(focus["regime_coverage_dominant_share"]),
        "regime_coverage_note": focus["regime_coverage_note"],
        "regime_coverage_note_zh": focus["regime_coverage_note_zh"],
        "recent_backtest_score": float(focus["recent_backtest_score"]),
        "recent_backtest_level": focus["recent_backtest_level"],
        "recent_backtest_level_zh": focus["recent_backtest_level_zh"],
        "recent_backtest_trade_count": int(focus["recent_backtest_trade_count"]),
        "recent_backtest_win_rate": float(focus["recent_backtest_win_rate"]),
        "recent_backtest_average_return": float(focus["recent_backtest_average_return"]),
        "recent_backtest_return_delta": float(focus["recent_backtest_return_delta"]),
        "recent_backtest_note": focus["recent_backtest_note"],
        "recent_backtest_note_zh": focus["recent_backtest_note_zh"],
        "backtest_decay_score": float(focus["backtest_decay_score"]),
        "backtest_decay_level": focus["backtest_decay_level"],
        "backtest_decay_level_zh": focus["backtest_decay_level_zh"],
        "backtest_decay_early_trade_count": int(focus["backtest_decay_early_trade_count"]),
        "backtest_decay_late_trade_count": int(focus["backtest_decay_late_trade_count"]),
        "backtest_decay_early_win_rate": float(focus["backtest_decay_early_win_rate"]),
        "backtest_decay_late_win_rate": float(focus["backtest_decay_late_win_rate"]),
        "backtest_decay_early_average_return": float(
            focus["backtest_decay_early_average_return"]
        ),
        "backtest_decay_late_average_return": float(
            focus["backtest_decay_late_average_return"]
        ),
        "backtest_decay_win_rate_delta": float(focus["backtest_decay_win_rate_delta"]),
        "backtest_decay_average_return_delta": float(
            focus["backtest_decay_average_return_delta"]
        ),
        "backtest_decay_note": focus["backtest_decay_note"],
        "backtest_decay_note_zh": focus["backtest_decay_note_zh"],
        "backtest_trust_note": focus["backtest_trust_note"],
        "backtest_trust_note_zh": focus["backtest_trust_note_zh"],
        "analysis_rows": dataframe_records(analysis),
        "peer_comparison": dataframe_records(peer_comparison),
        "snapshot": snapshot or {},
        "data_sources": data_sources,
        "data_source_warnings": {
            key: list(value) for key, value in data_source_warnings.items()
        },
        "data_readiness": data_readiness.to_dict(),
        "event_risk": event_risk.to_dict(),
        "fundamental_quality": fundamentals.to_dict(),
        "sentiment_risk": sentiment.to_dict(),
        "analyst_expectations": analyst.to_dict(),
        "valuation_risk": valuation.to_dict(),
        "sec_fundamentals": sec_snapshot.to_dict() if sec_snapshot is not None else {},
        "sector_context": asdict(sector_context),
        "screening_thresholds": screening_profile.thresholds.to_dict(),
        "screening_profile_detail": screening_profile.to_dict(),
        "screening_config": screening_config.to_dict()
        if screening_config is not None
        else default_screening_config().to_dict(),
        "cache_metadata": cache_metadata,
        "output_files": {
            "markdown_report": str(output_dir / "ticker_analysis.md"),
            "csv": str(output_dir / "ticker_analysis.csv"),
            "json": str(output_dir / "analysis_result.json"),
            "data_readiness_json": str(output_dir / "data_readiness.json"),
            "data_readiness_markdown": str(output_dir / "data_readiness.md"),
            "data_readiness_csv": str(output_dir / "data_readiness.csv"),
            "sec_fundamentals": str(output_dir / "sec_fundamentals.json"),
            "cache_metadata": str(output_dir / "cache_metadata.json"),
            "peer_csv": str(output_dir / "peer_comparison.csv"),
            "peer_markdown": str(output_dir / "peer_comparison.md"),
        },
    }


def _build_cache_metadata(
    ticker: str,
    requested_period: str,
    period: str,
    data_root: Path,
    output_dir: Path,
    price_file: Path,
    prices: pd.DataFrame,
    price_provider: str,
    price_warnings: tuple[str, ...],
    benchmark_prices: dict[str, pd.DataFrame],
    benchmark_sources: dict[str, str],
    benchmark_warnings: dict[str, tuple[str, ...]],
    sector_etf: str,
    sector_prices: pd.DataFrame,
    sector_source: str,
    sector_warnings: tuple[str, ...],
    data_repair_actions: tuple[str, ...],
    data_repair_actions_zh: tuple[str, ...],
    data_readiness: DataReadinessReport,
    signal_review_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    benchmark_files = {
        "SPY": data_root / f"SPY_{period}.csv",
        "QQQ": data_root / f"QQQ_{period}.csv",
        "^VIX": data_root / f"VIX_{period}.csv",
    }
    symbols: dict[str, object] = {
        ticker: _cache_symbol_metadata(
            symbol=ticker,
            prices=prices,
            provider=price_provider,
            warnings=price_warnings,
            cache_file=price_file,
        )
    }
    for symbol, frame in benchmark_prices.items():
        symbols[symbol] = _cache_symbol_metadata(
            symbol=symbol,
            prices=frame,
            provider=benchmark_sources.get(symbol, "unavailable"),
            warnings=benchmark_warnings.get(symbol, ()),
            cache_file=benchmark_files[symbol],
        )
    if sector_etf:
        symbols[sector_etf] = _cache_symbol_metadata(
            symbol=sector_etf,
            prices=sector_prices,
            provider=sector_source,
            warnings=sector_warnings,
            cache_file=data_root / f"{sector_etf}_{period}.csv",
        )
    return {
        "generated_at_utc": _utc_now(),
        "ticker": ticker,
        "requested_period": requested_period,
        "period": period,
        "effective_period": period,
        "auto_period_upgraded": requested_period != period,
        "data_repair_actions_applied": list(data_repair_actions),
        "data_repair_actions_applied_zh": list(data_repair_actions_zh),
        "data_readiness": data_readiness.to_dict(),
        "signal_review": signal_review_metadata or {},
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "symbols": symbols,
    }


def _cache_symbol_metadata(
    symbol: str,
    prices: pd.DataFrame,
    provider: str,
    warnings: tuple[str, ...],
    cache_file: Path,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "provider": provider,
        "row_count": int(len(prices)),
        "latest_price_date": _latest_date(prices, "date"),
        "cache_file": str(cache_file),
        "cache_file_exists": cache_file.exists(),
        "cache_file_modified_utc": _file_modified_utc(cache_file),
        "warnings": list(warnings),
    }


def _latest_date(frame: pd.DataFrame, column: str) -> str | None:
    if frame.empty or column not in frame.columns:
        return None
    value = pd.to_datetime(frame[column], errors="coerce").max()
    if pd.isna(value):
        return None
    return value.date().isoformat()


def _file_modified_utc(path: Path) -> str | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_single_ticker_scored_frame(
    prices: pd.DataFrame,
    min_history_days: int = 60,
    liquidity_window: int = 20,
    min_avg_dollar_volume: float = 1_000_000.0,
) -> pd.DataFrame:
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=False)
    frame["ticker"] = frame["ticker"].astype("string").str.upper().str.strip()
    frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)
    grouped = frame.groupby("ticker", group_keys=False)

    frame["dollar_volume"] = frame["adj_close"].astype(float) * frame["volume"].astype(float)
    frame["history_days"] = grouped.cumcount() + 1
    frame["liquidity"] = grouped["dollar_volume"].rolling(liquidity_window).mean().reset_index(
        level=0,
        drop=True,
    )
    frame["passes_universe"] = (
        (frame["history_days"] >= min_history_days)
        & (frame["liquidity"] >= min_avg_dollar_volume)
    )
    frame["score"] = np.nan
    return frame[
        [
            "date",
            "ticker",
            "adj_close",
            "volume",
            "passes_universe",
            "score",
            "history_days",
            "liquidity",
        ]
    ]
