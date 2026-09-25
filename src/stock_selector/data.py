from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from .config import DataConfig


REQUIRED_PRICE_COLUMNS = {
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
}

REQUIRED_FUNDAMENTAL_COLUMNS = {
    "report_date",
    "period_end",
    "ticker",
    "revenue",
    "gross_profit",
    "operating_income",
    "net_income",
    "book_value",
    "total_assets",
    "total_liabilities",
    "operating_cash_flow",
    "capital_expenditure",
    "shares_outstanding",
}

REQUIRED_METADATA_COLUMNS = {
    "ticker",
    "sector",
}


@dataclass(frozen=True)
class PriceDownloadResult:
    prices: pd.DataFrame
    provider: str
    attempts: tuple[str, ...]
    warnings: tuple[str, ...]
    missing_tickers: tuple[str, ...] = ()
    source_validation: dict[str, object] | None = None


def load_price_csv(config: DataConfig) -> pd.DataFrame:
    prices = pd.read_csv(config.prices_csv)
    prices.columns = [column.strip().lower() for column in prices.columns]

    missing = REQUIRED_PRICE_COLUMNS.difference(prices.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Missing required price columns: {missing_text}")

    prices["date"] = pd.to_datetime(prices["date"], utc=False)
    prices["ticker"] = prices["ticker"].astype("string").str.upper().str.strip()
    prices["ticker"] = prices["ticker"].replace("", pd.NA)
    numeric_columns = ["open", "high", "low", "close", "adj_close", "volume"]
    prices[numeric_columns] = prices[numeric_columns].apply(pd.to_numeric, errors="coerce")
    prices = prices.dropna(subset=["date", "ticker", "adj_close", "volume"])
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    duplicate_mask = prices.duplicated(["date", "ticker"], keep=False)
    if duplicate_mask.any():
        count = int(duplicate_mask.sum())
        raise ValueError(f"Found duplicate date/ticker rows: {count}")

    return prices


def load_fundamental_csv(path: str | Path) -> pd.DataFrame:
    fundamentals = pd.read_csv(path)
    fundamentals.columns = [column.strip().lower() for column in fundamentals.columns]

    missing = REQUIRED_FUNDAMENTAL_COLUMNS.difference(fundamentals.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Missing required fundamental columns: {missing_text}")

    fundamentals["report_date"] = pd.to_datetime(fundamentals["report_date"], utc=False)
    fundamentals["period_end"] = pd.to_datetime(fundamentals["period_end"], utc=False)
    fundamentals["ticker"] = fundamentals["ticker"].astype("string").str.upper().str.strip()
    fundamentals["ticker"] = fundamentals["ticker"].replace("", pd.NA)
    if "fundamentals_source" not in fundamentals.columns:
        fundamentals["fundamentals_source"] = "yfinance_restated"
    else:
        fundamentals["fundamentals_source"] = (
            fundamentals["fundamentals_source"].fillna("").astype(str).str.strip()
        )
        fundamentals.loc[fundamentals["fundamentals_source"] == "", "fundamentals_source"] = (
            "yfinance_restated"
        )
    numeric_columns = sorted(REQUIRED_FUNDAMENTAL_COLUMNS - {"report_date", "period_end", "ticker"})
    fundamentals[numeric_columns] = fundamentals[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    fundamentals = fundamentals.dropna(subset=["report_date", "period_end", "ticker"])
    fundamentals = fundamentals.sort_values(["ticker", "report_date"]).reset_index(drop=True)

    duplicate_mask = fundamentals.duplicated(["report_date", "ticker"], keep=False)
    if duplicate_mask.any():
        count = int(duplicate_mask.sum())
        raise ValueError(f"Found duplicate report_date/ticker rows: {count}")

    return fundamentals


def load_macro_csv(path: str | Path) -> pd.DataFrame:
    macro = pd.read_csv(path)
    macro.columns = [column.strip().lower() for column in macro.columns]
    if "date" not in macro.columns:
        raise ValueError("Missing required macro column: date")

    macro["date"] = pd.to_datetime(macro["date"], utc=False)
    value_columns = [column for column in macro.columns if column != "date"]
    if not value_columns:
        raise ValueError("Macro CSV must include at least one value column.")

    macro[value_columns] = macro[value_columns].apply(pd.to_numeric, errors="coerce")
    macro = macro.dropna(subset=["date"])
    macro = macro.sort_values("date").reset_index(drop=True)
    duplicate_mask = macro.duplicated(["date"], keep=False)
    if duplicate_mask.any():
        count = int(duplicate_mask.sum())
        raise ValueError(f"Found duplicate macro date rows: {count}")
    return macro


def load_metadata_csv(path: str | Path) -> pd.DataFrame:
    metadata = pd.read_csv(path)
    metadata.columns = [column.strip().lower() for column in metadata.columns]

    missing = REQUIRED_METADATA_COLUMNS.difference(metadata.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Missing required metadata columns: {missing_text}")

    for column in ["ticker", "sector", "industry", "country", "exchange"]:
        if column not in metadata.columns:
            metadata[column] = "Unknown"
        metadata[column] = metadata[column].astype("string").str.strip()
        metadata[column] = metadata[column].replace("", pd.NA).fillna("Unknown")

    metadata["ticker"] = metadata["ticker"].str.upper()
    metadata = metadata[["ticker", "sector", "industry", "country", "exchange"]]
    duplicate_mask = metadata.duplicated(["ticker"], keep=False)
    if duplicate_mask.any():
        count = int(duplicate_mask.sum())
        raise ValueError(f"Found duplicate metadata ticker rows: {count}")
    return metadata.sort_values("ticker").reset_index(drop=True)


def download_yfinance_prices(
    tickers: list[str],
    start: str,
    end: str | None = None,
    output_path: str | Path | None = None,
    timeout_seconds: int = 30,
) -> pd.DataFrame:
    tickers = [ticker.upper().strip() for ticker in tickers if ticker.strip()]
    if not tickers:
        raise ValueError("At least one ticker is required.")

    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install the optional 'data' dependency to use yfinance.") from exc

    raw = _download_yfinance_silently(
        yf,
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=True,
        timeout=timeout_seconds,
    )
    if raw.empty:
        raise ValueError("No price data returned from yfinance.")

    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        if isinstance(raw.columns, pd.MultiIndex):
            if ticker not in raw.columns.get_level_values(0):
                continue
            frame = raw[ticker].copy()
        else:
            frame = raw.copy()

        frame = frame.reset_index()
        frame.columns = [str(column).lower().replace(" ", "_") for column in frame.columns]
        if "date" not in frame.columns and "index" in frame.columns:
            frame = frame.rename(columns={"index": "date"})
        frame["ticker"] = ticker.upper()
        frame = frame.rename(columns={"adj_close": "adj_close"})
        frames.append(frame)

    prices = pd.concat(frames, ignore_index=True)
    prices = prices.rename(columns={"date": "date"})
    prices = prices[
        ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    ]
    prices = prices.dropna(subset=["date", "ticker", "adj_close", "volume"])
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(output_path, index=False)

    return prices


def download_yfinance_prices_for_period(
    tickers: list[str],
    period: str = "5y",
    output_path: str | Path | None = None,
    timeout_seconds: int = 30,
) -> pd.DataFrame:
    tickers = [ticker.upper().strip() for ticker in tickers if ticker.strip()]
    if not tickers:
        raise ValueError("At least one ticker is required.")

    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "Install yfinance before using real-data analysis: python3 -m pip install yfinance"
        ) from exc

    raw = _download_yfinance_silently(
        yf,
        tickers=tickers,
        period=period,
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=True,
        timeout=timeout_seconds,
    )
    if raw.empty:
        raise ValueError("No price data returned from yfinance.")

    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        if isinstance(raw.columns, pd.MultiIndex):
            if ticker in raw.columns.get_level_values(0):
                frame = raw[ticker].copy()
            elif ticker in raw.columns.get_level_values(-1):
                frame = raw.xs(ticker, axis=1, level=-1).copy()
            else:
                continue
        else:
            frame = raw.copy()

        frame = frame.reset_index()
        frame.columns = [str(column).lower().replace(" ", "_") for column in frame.columns]
        if "date" not in frame.columns and "index" in frame.columns:
            frame = frame.rename(columns={"index": "date"})
        frame["ticker"] = ticker.upper()
        frames.append(frame)

    if not frames:
        raise ValueError("No usable ticker data returned from yfinance.")

    prices = pd.concat(frames, ignore_index=True)
    if "date" not in prices.columns and "datetime" in prices.columns:
        prices = prices.rename(columns={"datetime": "date"})
    if "adj_close" not in prices.columns:
        prices["adj_close"] = prices["close"]

    prices = prices[
        ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    ]
    prices = prices.dropna(subset=["date", "ticker", "adj_close", "volume"])
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(output_path, index=False)

    return prices


def download_prices_for_period_multi_source(
    tickers: list[str],
    period: str = "5y",
    output_path: str | Path | None = None,
    provider_order: list[str] | tuple[str, ...] | None = None,
    timeout_seconds: int | None = None,
    max_attempts: int | None = None,
) -> PriceDownloadResult:
    tickers = [ticker.upper().strip() for ticker in tickers if ticker.strip()]
    if not tickers:
        raise ValueError("At least one ticker is required.")

    providers = _provider_order(provider_order)
    timeout_seconds = _download_timeout_seconds(timeout_seconds)
    max_attempts = _download_max_attempts(max_attempts)
    attempts: list[str] = []
    warnings: list[str] = []
    for provider in providers:
        attempts.append(provider)
        prices = pd.DataFrame()
        if not _is_known_provider(provider):
            warnings.append(f"Unknown provider skipped: {provider}")
            continue

        prices = _download_prices_from_provider_with_retries(
            provider=provider,
            tickers=tickers,
            period=period,
            timeout_seconds=timeout_seconds,
            max_attempts=max_attempts,
            warnings=warnings,
        )

        if prices.empty:
            warnings.append(f"{provider}: no rows returned")
            continue

        prices = normalize_price_frame(prices)
        missing_tickers = _missing_price_tickers(tickers, prices)
        if missing_tickers:
            warnings.append(
                f"{provider}: missing price rows for {', '.join(missing_tickers)}"
            )
        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            prices.to_csv(output_path, index=False)
        source_validation = _cross_validate_price_sources(
            primary_prices=prices,
            primary_provider=provider,
            providers=providers,
            tickers=tickers,
            period=period,
            timeout_seconds=timeout_seconds,
            warnings=warnings,
        )
        return PriceDownloadResult(
            prices=prices,
            provider=provider,
            attempts=tuple(attempts),
            warnings=tuple(warnings),
            missing_tickers=tuple(missing_tickers),
            source_validation=source_validation,
        )

    warning_text = "; ".join(warnings) if warnings else "No providers were attempted."
    raise ValueError(f"No price data returned from configured providers. {warning_text}")


def _download_prices_from_provider_with_retries(
    provider: str,
    tickers: list[str],
    period: str,
    timeout_seconds: int,
    max_attempts: int,
    warnings: list[str],
) -> pd.DataFrame:
    prices = pd.DataFrame()
    for attempt_number in range(1, max_attempts + 1):
        try:
            prices = _download_prices_from_provider(
                provider=provider,
                tickers=tickers,
                period=period,
                timeout_seconds=timeout_seconds,
            )
            break
        except Exception as exc:
            warnings.append(f"{provider} attempt {attempt_number}/{max_attempts}: {exc}")
            if _non_retryable_download_error(exc):
                break
    return prices


def _download_prices_from_provider(
    provider: str,
    tickers: list[str],
    period: str,
    timeout_seconds: int,
) -> pd.DataFrame:
    entry = PRICE_PROVIDERS.get(provider)
    if entry is None:
        raise ValueError(f"Unknown provider: {provider}")
    return entry.fetch(tickers, period=period, timeout_seconds=timeout_seconds)


def _cross_validate_price_sources(
    primary_prices: pd.DataFrame,
    primary_provider: str,
    providers: tuple[str, ...],
    tickers: list[str],
    period: str,
    timeout_seconds: int,
    warnings: list[str],
) -> dict[str, object]:
    comparison_rows: list[dict[str, object]] = []
    validation_warnings: list[str] = []
    for provider in providers:
        if provider == primary_provider:
            continue
        if not _is_known_provider(provider):
            continue
        try:
            comparison = normalize_price_frame(
                _download_prices_from_provider(
                    provider=provider,
                    tickers=tickers,
                    period=period,
                    timeout_seconds=timeout_seconds,
                )
            )
        except Exception as exc:
            validation_warnings.append(f"{provider}: validation fetch failed: {exc}")
            continue
        if comparison.empty:
            validation_warnings.append(f"{provider}: validation fetch returned no rows")
            continue
        comparison_rows.extend(
            _price_source_comparison_rows(primary_prices, comparison, provider)
        )

    if not comparison_rows:
        return {
            "status": "single_source_available",
            "status_zh": "只有单一价格源可用",
            "comparison_provider_count": 0,
            "max_close_diff_pct": None,
            "max_volume_diff_pct": None,
            "max_missing_date_count": 0,
            "comparisons": [],
            "warnings": validation_warnings,
        }

    max_close_diff = _max_numeric(
        row["latest_close_diff_pct"] for row in comparison_rows
    )
    max_volume_diff = _max_numeric(
        row["latest_volume_diff_pct"] for row in comparison_rows
    )
    max_missing_dates = max(int(row["missing_date_count"]) for row in comparison_rows)
    conflict = max_close_diff > 0.005 or max_volume_diff > 0.25 or max_missing_dates > 5
    status = "conflict_warning" if conflict else "validated"
    status_zh = "多源价格存在冲突" if conflict else "多源价格已验证"
    if conflict:
        warnings.append("price source validation conflict detected")
    return {
        "status": status,
        "status_zh": status_zh,
        "comparison_provider_count": len({row["provider"] for row in comparison_rows}),
        "max_close_diff_pct": round(float(max_close_diff), 6),
        "max_volume_diff_pct": round(float(max_volume_diff), 6),
        "max_missing_date_count": max_missing_dates,
        "comparisons": comparison_rows,
        "warnings": validation_warnings,
    }


def _price_source_comparison_rows(
    primary: pd.DataFrame,
    comparison: pd.DataFrame,
    provider: str,
) -> list[dict[str, object]]:
    primary_frame = normalize_price_frame(primary)
    comparison_frame = normalize_price_frame(comparison)
    rows: list[dict[str, object]] = []
    for ticker in sorted(set(primary_frame["ticker"]) & set(comparison_frame["ticker"])):
        primary_ticker = primary_frame[primary_frame["ticker"] == ticker].sort_values("date")
        comparison_ticker = comparison_frame[comparison_frame["ticker"] == ticker].sort_values("date")
        if primary_ticker.empty or comparison_ticker.empty:
            continue
        primary_by_date = primary_ticker.assign(_d=pd.to_datetime(primary_ticker["date"]).dt.date)
        comparison_by_date = comparison_ticker.assign(
            _d=pd.to_datetime(comparison_ticker["date"]).dt.date
        )
        primary_dates = set(primary_by_date["_d"])
        comparison_dates = set(comparison_by_date["_d"])
        # Compare on the latest COMMON trading day. Otherwise one source having an
        # extra intraday/partial bar for "today" (which the other lacks) would be
        # compared against a different date and raise a spurious conflict.
        common_dates = primary_dates & comparison_dates
        if common_dates:
            latest_common = max(common_dates)
            primary_latest = primary_by_date[primary_by_date["_d"] == latest_common].iloc[-1]
            comparison_latest = comparison_by_date[comparison_by_date["_d"] == latest_common].iloc[-1]
            close_diff = _relative_difference(
                primary_latest.get("adj_close"), comparison_latest.get("adj_close")
            )
            volume_diff = _relative_difference(
                primary_latest.get("volume"), comparison_latest.get("volume")
            )
            primary_latest_date = latest_common.isoformat()
            comparison_latest_date = latest_common.isoformat()
        else:
            close_diff = None
            volume_diff = None
            primary_latest_date = pd.Timestamp(primary_ticker.iloc[-1]["date"]).date().isoformat()
            comparison_latest_date = pd.Timestamp(comparison_ticker.iloc[-1]["date"]).date().isoformat()
        rows.append(
            {
                "ticker": ticker,
                "provider": provider,
                "primary_latest_date": primary_latest_date,
                "comparison_latest_date": comparison_latest_date,
                "latest_close_diff_pct": close_diff,
                "latest_volume_diff_pct": volume_diff,
                "missing_date_count": len(primary_dates.symmetric_difference(comparison_dates)),
            }
        )
    return rows


def _relative_difference(left: object, right: object) -> float | None:
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError):
        return None
    denominator = max(abs(left_value), abs(right_value), 1e-9)
    return abs(left_value - right_value) / denominator


def _max_numeric(values: object) -> float:
    numeric_values: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            numeric_values.append(float(value))
        except (TypeError, ValueError):
            continue
    return max(numeric_values) if numeric_values else 0.0


def _download_yfinance_silently(yfinance_module, **kwargs: object) -> pd.DataFrame:
    stderr_buffer = io.StringIO()
    with redirect_stderr(stderr_buffer):
        return yfinance_module.download(**kwargs)


def download_polygon_prices_for_period(
    tickers: list[str],
    period: str = "5y",
    timeout_seconds: int = 30,
) -> pd.DataFrame:
    api_key = os.environ.get("POLYGON_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("POLYGON_API_KEY is not set")

    start_date, end_date = _period_date_range(period)
    frames: list[pd.DataFrame] = []
    for requested_ticker in tickers:
        provider_ticker = _polygon_ticker(requested_ticker)
        query = urlencode(
            {
                "adjusted": "true",
                "sort": "asc",
                "limit": 50000,
                "apiKey": api_key,
            }
        )
        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{provider_ticker}/range/1/day/"
            f"{start_date.isoformat()}/{end_date.isoformat()}?{query}"
        )
        payload = _get_json(url, timeout_seconds=timeout_seconds)
        results = payload.get("results") or []
        if not results:
            continue

        rows = []
        for item in results:
            timestamp = int(item["t"]) / 1000
            rows.append(
                {
                    "date": datetime.fromtimestamp(timestamp, tz=timezone.utc).date().isoformat(),
                    "ticker": requested_ticker,
                    "open": item.get("o"),
                    "high": item.get("h"),
                    "low": item.get("l"),
                    "close": item.get("c"),
                    "adj_close": item.get("c"),
                    "volume": item.get("v", 0),
                }
            )
        frames.append(pd.DataFrame(rows))

    if not frames:
        raise ValueError("No price data returned from Polygon.")
    return normalize_price_frame(pd.concat(frames, ignore_index=True))


def download_alpaca_prices_for_period(
    tickers: list[str],
    period: str = "5y",
    timeout_seconds: int = 30,
) -> pd.DataFrame:
    api_key = os.environ.get("ALPACA_API_KEY_ID", "").strip()
    api_secret = os.environ.get("ALPACA_API_SECRET_KEY", "").strip()
    if not api_key or not api_secret:
        raise RuntimeError("ALPACA_API_KEY_ID or ALPACA_API_SECRET_KEY is not set")

    stock_tickers = [ticker for ticker in tickers if not ticker.startswith("^")]
    if not stock_tickers:
        raise ValueError("Alpaca stock data does not support this benchmark ticker.")

    start_date, end_date = _period_date_range(period)
    query = urlencode(
        {
            "symbols": ",".join(stock_tickers),
            "timeframe": "1Day",
            "start": f"{start_date.isoformat()}T00:00:00Z",
            "end": f"{end_date.isoformat()}T23:59:59Z",
            "adjustment": "all",
            "feed": os.environ.get("ALPACA_DATA_FEED", "iex"),
            "limit": 10000,
        }
    )
    payload = _get_json(
        f"https://data.alpaca.markets/v2/stocks/bars?{query}",
        headers={
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret,
        },
        timeout_seconds=timeout_seconds,
    )
    bars = payload.get("bars") or {}
    frames: list[pd.DataFrame] = []
    for ticker, items in bars.items():
        rows = [
            {
                "date": str(item.get("t", ""))[:10],
                "ticker": ticker,
                "open": item.get("o"),
                "high": item.get("h"),
                "low": item.get("l"),
                "close": item.get("c"),
                "adj_close": item.get("c"),
                "volume": item.get("v", 0),
            }
            for item in items
        ]
        if rows:
            frames.append(pd.DataFrame(rows))

    if not frames:
        raise ValueError("No price data returned from Alpaca.")
    return normalize_price_frame(pd.concat(frames, ignore_index=True))


def download_tiingo_prices_for_period(
    tickers: list[str],
    period: str = "5y",
    timeout_seconds: int = 30,
) -> pd.DataFrame:
    """Daily adjusted prices from Tiingo (free API token). Reliable from datacenter
    IPs, so it works where Stooq's anti-bot challenge blocks cloud servers."""
    token = os.environ.get("TIINGO_API_TOKEN", "").strip() or os.environ.get(
        "TIINGO_API_KEY", ""
    ).strip()
    if not token:
        raise RuntimeError("TIINGO_API_TOKEN is not set")

    stock_tickers = [ticker for ticker in tickers if not ticker.startswith("^")]
    if not stock_tickers:
        raise ValueError("Tiingo does not support this benchmark ticker.")

    start_date, end_date = _period_date_range(period)
    frames: list[pd.DataFrame] = []
    for requested_ticker in stock_tickers:
        query = urlencode(
            {"startDate": start_date.isoformat(), "endDate": end_date.isoformat(), "format": "json"}
        )
        url = f"https://api.tiingo.com/tiingo/daily/{requested_ticker.lower()}/prices?{query}"
        text = _get_text(
            url,
            headers={"Content-Type": "application/json", "Authorization": f"Token {token}"},
            timeout_seconds=timeout_seconds,
        )
        frame = _parse_tiingo_json(text, requested_ticker)
        if not frame.empty:
            frames.append(frame)

    if not frames:
        raise ValueError("No price data returned from Tiingo.")
    return normalize_price_frame(pd.concat(frames, ignore_index=True))


def _parse_tiingo_json(text: str, requested_ticker: str) -> pd.DataFrame:
    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        return pd.DataFrame()
    if not isinstance(payload, list) or not payload:
        return pd.DataFrame()
    rows = [
        {
            "date": str(item.get("date", ""))[:10],
            "ticker": requested_ticker,
            # Raw OHLC + a separate adjusted close, matching the frame convention
            # (close is raw, adj_close carries split/dividend adjustment).
            "open": item.get("open"),
            "high": item.get("high"),
            "low": item.get("low"),
            "close": item.get("close"),
            "adj_close": item.get("adjClose", item.get("close")),
            "volume": item.get("volume", 0),
        }
        for item in payload
        if isinstance(item, dict)
    ]
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def download_stooq_prices_for_period(
    tickers: list[str],
    period: str = "5y",
    timeout_seconds: int = 30,
) -> pd.DataFrame:
    """Free daily prices from Stooq (no API key). Used as a second source so that
    cross-source validation runs even without paid providers configured."""
    start_date, end_date = _period_date_range(period)
    frames: list[pd.DataFrame] = []
    for requested_ticker in tickers:
        symbol = _stooq_symbol(requested_ticker)
        query = urlencode(
            {
                "s": symbol,
                "d1": start_date.strftime("%Y%m%d"),
                "d2": end_date.strftime("%Y%m%d"),
                "i": "d",
            }
        )
        text = _get_text(f"https://stooq.com/q/d/l/?{query}", timeout_seconds=timeout_seconds)
        frame = _parse_stooq_csv(text, requested_ticker)
        if not frame.empty:
            frames.append(frame)

    if not frames:
        raise ValueError("No price data returned from Stooq.")
    return normalize_price_frame(pd.concat(frames, ignore_index=True))


def _stooq_symbol(ticker: str) -> str:
    clean = ticker.strip().lower()
    if clean.startswith("^"):
        return clean
    return f"{clean}.us"


def _parse_stooq_csv(text: str, requested_ticker: str) -> pd.DataFrame:
    if not text or "date" not in text[:64].lower():
        # Stooq returns "No data" (or an HTML error) for unknown symbols.
        return pd.DataFrame()
    try:
        raw = pd.read_csv(io.StringIO(text))
    except Exception:
        return pd.DataFrame()
    raw.columns = [str(column).strip().lower() for column in raw.columns]
    required = {"date", "open", "high", "low", "close"}
    if not required.issubset(raw.columns):
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "date": raw["date"],
            "ticker": requested_ticker,
            "open": raw["open"],
            "high": raw["high"],
            "low": raw["low"],
            "close": raw["close"],
            "adj_close": raw["close"],
            "volume": raw.get("volume", 0),
        }
    )


def normalize_price_frame(prices: pd.DataFrame) -> pd.DataFrame:
    frame = prices.copy()
    frame.columns = [str(column).strip().lower().replace(" ", "_") for column in frame.columns]
    if "adj_close" not in frame.columns and "close" in frame.columns:
        frame["adj_close"] = frame["close"]
    missing = REQUIRED_PRICE_COLUMNS.difference(frame.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Missing required price columns: {missing_text}")

    frame = frame[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]]
    frame["date"] = pd.to_datetime(frame["date"], utc=False)
    frame["ticker"] = frame["ticker"].astype("string").str.upper().str.strip()
    numeric_columns = ["open", "high", "low", "close", "adj_close", "volume"]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=["date", "ticker", "adj_close", "volume"])
    frame = frame.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    return frame.reset_index(drop=True)


def _provider_order(provider_order: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if provider_order is None:
        raw_value = os.environ.get("STOCK_SELECTOR_DATA_PROVIDER_ORDER") or os.environ.get(
            "STOCK_SELECTOR_DATA_PROVIDER"
        )
        if raw_value:
            provider_order = tuple(value.strip() for value in raw_value.split(","))
        else:
            provider_order = _available_default_providers()
    providers = tuple(str(provider).lower().strip() for provider in provider_order if str(provider).strip())
    return providers or ("yfinance",)


def _available_default_providers() -> tuple[str, ...]:
    providers: list[str] = []
    for name, provider in PRICE_PROVIDERS.items():
        if provider.requires_key and provider.is_available():
            providers.append(name)
    # yfinance is the primary free source; stooq is a free second source so that
    # cross-source validation runs even when no paid API keys are configured.
    providers.append("yfinance")
    if "stooq" in PRICE_PROVIDERS and os.environ.get("STOCK_SELECTOR_DISABLE_STOOQ", "").strip() not in {"1", "true", "yes"}:
        providers.append("stooq")
    return tuple(providers)


def _download_timeout_seconds(value: int | None) -> int:
    if value is not None:
        return max(1, int(value))
    raw_value = os.environ.get("STOCK_SELECTOR_DOWNLOAD_TIMEOUT_SECONDS", "").strip()
    if raw_value:
        try:
            return max(1, int(raw_value))
        except ValueError:
            return 30
    return 30


def _download_max_attempts(value: int | None) -> int:
    if value is not None:
        return max(1, int(value))
    raw_value = os.environ.get("STOCK_SELECTOR_DOWNLOAD_MAX_ATTEMPTS", "").strip()
    if raw_value:
        try:
            return max(1, int(raw_value))
        except ValueError:
            return 2
    return 2


def _non_retryable_download_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(
        text in message
        for text in [
            "api_key",
            "api key",
            "secret_key",
            "not set",
            "install yfinance",
            "install the optional",
        ]
    )


def _missing_price_tickers(tickers: list[str], prices: pd.DataFrame) -> list[str]:
    if prices.empty or "ticker" not in prices.columns:
        return list(dict.fromkeys(tickers))
    returned = set(prices["ticker"].astype("string").str.upper().str.strip().dropna())
    return [ticker for ticker in dict.fromkeys(tickers) if ticker not in returned]


def _period_date_range(period: str) -> tuple[date, date]:
    end_date = date.today()
    value = period.strip().lower()
    if value.endswith("mo"):
        amount_text = value[:-2]
        amount = int(amount_text) if amount_text.isdigit() else 1
        return end_date - timedelta(days=amount * 31), end_date
    if len(value) >= 2 and value[:-1].isdigit():
        amount = int(value[:-1])
        unit = value[-1]
    else:
        amount = 5
        unit = "y"

    if unit == "d":
        start_date = end_date - timedelta(days=amount)
    elif unit == "w":
        start_date = end_date - timedelta(weeks=amount)
    elif unit == "y":
        start_date = end_date.replace(year=end_date.year - amount)
    else:
        start_date = end_date.replace(year=end_date.year - 5)
    return start_date, end_date


def _polygon_ticker(ticker: str) -> str:
    if ticker == "^VIX":
        return "I:VIX"
    return ticker


def _get_json(
    url: str,
    headers: dict[str, str] | None = None,
    timeout_seconds: int = 30,
) -> dict[str, object]:
    request = Request(url, headers=headers or {})
    with urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_text(
    url: str,
    headers: dict[str, str] | None = None,
    timeout_seconds: int = 30,
) -> str:
    request = Request(url, headers=headers or {"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=timeout_seconds) as response:
        return response.read().decode("utf-8", errors="replace")


@dataclass(frozen=True)
class PriceProvider:
    """A pluggable daily-price source. Register one to make it selectable by name
    in the provider order and the cross-source validation.

    ``fetch_name`` is the name of a module-level ``(tickers, period, timeout_seconds)``
    downloader. It is resolved at call time (late binding) so that tests and callers
    can monkeypatch the underlying function on the module."""

    name: str
    fetch_name: str
    is_available: Callable[[], bool]
    requires_key: bool = False

    def fetch(self, tickers: list[str], period: str, timeout_seconds: int) -> pd.DataFrame:
        func = globals().get(self.fetch_name)
        if not callable(func):
            raise ValueError(f"Provider function not found: {self.fetch_name}")
        return func(tickers, period=period, timeout_seconds=timeout_seconds)


def _polygon_available() -> bool:
    return bool(os.environ.get("POLYGON_API_KEY", "").strip())


def _alpaca_available() -> bool:
    return bool(
        os.environ.get("ALPACA_API_KEY_ID", "").strip()
        and os.environ.get("ALPACA_API_SECRET_KEY", "").strip()
    )


def _tiingo_available() -> bool:
    return bool(
        os.environ.get("TIINGO_API_TOKEN", "").strip()
        or os.environ.get("TIINGO_API_KEY", "").strip()
    )


def _price_provider_registry() -> dict[str, PriceProvider]:
    providers = [
        PriceProvider("polygon", "download_polygon_prices_for_period", _polygon_available, True),
        PriceProvider("alpaca", "download_alpaca_prices_for_period", _alpaca_available, True),
        PriceProvider("tiingo", "download_tiingo_prices_for_period", _tiingo_available, True),
        PriceProvider("yfinance", "download_yfinance_prices_for_period", lambda: True, False),
        PriceProvider("stooq", "download_stooq_prices_for_period", lambda: True, False),
    ]
    return {provider.name: provider for provider in providers}


PRICE_PROVIDERS: dict[str, PriceProvider] = _price_provider_registry()


def _is_known_provider(provider: str) -> bool:
    return provider in PRICE_PROVIDERS
