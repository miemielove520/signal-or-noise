from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SEC_CACHE_MAX_AGE_DAYS = 30
SEC_FUNDAMENTAL_HISTORY_COLUMNS = [
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
    "fundamentals_source",
    "sec_cik",
    "sec_form",
    "sec_accession",
]


@dataclass(frozen=True)
class SecFundamentalSnapshot:
    ticker: str
    cik: str
    company_name: str
    fetched_at_utc: str
    source: str
    data_coverage: float
    fields: dict[str, object]
    warnings: tuple[str, ...]
    raw_cache_path: str
    extracted_cache_path: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["warnings"] = list(self.warnings)
        return payload


def fetch_sec_fundamental_snapshot(
    ticker: str,
    data_root: str | Path = "data",
    max_cache_age_days: int = SEC_CACHE_MAX_AGE_DAYS,
    timeout_seconds: int = 30,
) -> SecFundamentalSnapshot:
    ticker = ticker.upper().strip()
    if not ticker:
        raise ValueError("ticker cannot be empty.")

    root = _sec_root(Path(data_root))
    root.mkdir(parents=True, exist_ok=True)
    cik, company_name = _lookup_cik(
        ticker=ticker,
        root=root,
        max_cache_age_days=max_cache_age_days,
        timeout_seconds=timeout_seconds,
    )
    raw_path = root / "companyfacts" / f"CIK{cik}.json"
    facts = _load_or_fetch_json(
        url=SEC_COMPANYFACTS_URL.format(cik=cik),
        path=raw_path,
        max_cache_age_days=max_cache_age_days,
        timeout_seconds=timeout_seconds,
    )
    fields, coverage, warnings = _extract_snapshot_fields(ticker, facts)
    fields["ticker"] = ticker
    fields["company_name"] = fields.get("company_name") or company_name
    fields["sec_cik"] = cik
    fields["sec_data_source"] = "sec_companyfacts"
    fields["sec_fetched_at_utc"] = _utc_now()
    extracted_path = root / "extracted" / f"{ticker}_sec_fundamentals.json"
    extracted_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = SecFundamentalSnapshot(
        ticker=ticker,
        cik=cik,
        company_name=str(fields.get("company_name") or company_name),
        fetched_at_utc=str(fields["sec_fetched_at_utc"]),
        source="sec_companyfacts",
        data_coverage=coverage,
        fields=_json_safe(fields),
        warnings=tuple(warnings),
        raw_cache_path=str(raw_path),
        extracted_cache_path=str(extracted_path),
    )
    extracted_path.write_text(json.dumps(snapshot.to_dict(), indent=2), encoding="utf-8")
    return snapshot


def fetch_sec_fundamental_history(
    ticker: str,
    data_root: str | Path = "data",
    max_cache_age_days: int = SEC_CACHE_MAX_AGE_DAYS,
    timeout_seconds: int = 30,
) -> pd.DataFrame:
    ticker = ticker.upper().strip()
    if not ticker:
        raise ValueError("ticker cannot be empty.")

    root = _sec_root(Path(data_root))
    root.mkdir(parents=True, exist_ok=True)
    cik, _company_name = _lookup_cik(
        ticker=ticker,
        root=root,
        max_cache_age_days=max_cache_age_days,
        timeout_seconds=timeout_seconds,
    )
    raw_path = root / "companyfacts" / f"CIK{cik}.json"
    facts = _load_or_fetch_json(
        url=SEC_COMPANYFACTS_URL.format(cik=cik),
        path=raw_path,
        max_cache_age_days=max_cache_age_days,
        timeout_seconds=timeout_seconds,
    )
    history = extract_sec_fundamental_history(
        ticker=ticker,
        cik=cik,
        facts_payload=facts,
    )
    extracted_path = root / "extracted" / f"{ticker}_sec_fundamental_history.csv"
    extracted_path.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(extracted_path, index=False)
    return history


def extract_sec_fundamental_history(
    ticker: str,
    cik: str,
    facts_payload: dict[str, Any],
) -> pd.DataFrame:
    ticker = ticker.upper().strip()
    facts = ((facts_payload.get("facts") or {}).get("us-gaap") or {})
    annual: dict[str, dict[str, Any]] = {}

    _merge_fact_values(
        annual,
        "revenue",
        _best_fact_series(
            facts,
            [
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "Revenues",
                "SalesRevenueNet",
            ],
            duration=True,
        ),
    )
    _merge_fact_values(annual, "gross_profit", _best_fact_series(facts, ["GrossProfit"], duration=True))
    _merge_fact_values(
        annual,
        "operating_income",
        _best_fact_series(facts, ["OperatingIncomeLoss"], duration=True),
    )
    _merge_fact_values(annual, "net_income", _best_fact_series(facts, ["NetIncomeLoss"], duration=True))
    _merge_fact_values(
        annual,
        "operating_cash_flow",
        _best_fact_series(facts, ["NetCashProvidedByUsedInOperatingActivities"], duration=True),
    )
    _merge_fact_values(
        annual,
        "capital_expenditure",
        _best_fact_series(facts, ["PaymentsToAcquirePropertyPlantAndEquipment"], duration=True),
        transform=lambda value: abs(value),
    )
    _merge_fact_values(
        annual,
        "book_value",
        _best_fact_series(
            facts,
            [
                "StockholdersEquity",
                "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
            ],
            duration=False,
        ),
    )
    _merge_fact_values(annual, "total_assets", _best_fact_series(facts, ["Assets"], duration=False))
    _merge_fact_values(
        annual,
        "total_liabilities",
        _best_fact_series(facts, ["Liabilities"], duration=False),
    )
    _merge_fact_values(
        annual,
        "shares_outstanding",
        _best_fact_series(
            facts,
            [
                "EntityCommonStockSharesOutstanding",
                "CommonStocksIncludingAdditionalPaidInCapitalMember",
                "WeightedAverageNumberOfDilutedSharesOutstanding",
            ],
            duration=False,
        ),
    )

    rows: list[dict[str, object]] = []
    for period_end, values in sorted(annual.items()):
        report_date = _safe_date(values.get("filed"))
        period_date = _safe_date(period_end)
        if report_date is None or period_date is None:
            continue
        row = {
            "report_date": report_date,
            "period_end": period_date,
            "ticker": ticker,
            "fundamentals_source": "sec_pit",
            "sec_cik": cik,
            "sec_form": values.get("form") or "",
            "sec_accession": values.get("accn") or "",
        }
        for column in SEC_FUNDAMENTAL_HISTORY_COLUMNS:
            if column not in row:
                row[column] = values.get(column, np.nan)
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=SEC_FUNDAMENTAL_HISTORY_COLUMNS)
    frame = pd.DataFrame(rows)
    frame = frame[SEC_FUNDAMENTAL_HISTORY_COLUMNS]
    frame["report_date"] = pd.to_datetime(frame["report_date"], utc=False)
    frame["period_end"] = pd.to_datetime(frame["period_end"], utc=False)
    numeric_columns = [
        column
        for column in SEC_FUNDAMENTAL_HISTORY_COLUMNS
        if column
        not in {
            "report_date",
            "period_end",
            "ticker",
            "fundamentals_source",
            "sec_cik",
            "sec_form",
            "sec_accession",
        }
    ]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    return frame.sort_values(["ticker", "report_date"]).drop_duplicates(
        ["ticker", "report_date"],
        keep="last",
    ).reset_index(drop=True)


def _lookup_cik(
    ticker: str,
    root: Path,
    max_cache_age_days: int,
    timeout_seconds: int,
) -> tuple[str, str]:
    path = root / "company_tickers.json"
    payload = _load_or_fetch_json(
        url=SEC_COMPANY_TICKERS_URL,
        path=path,
        max_cache_age_days=max_cache_age_days,
        timeout_seconds=timeout_seconds,
    )
    records = payload.values() if isinstance(payload, dict) else payload
    for item in records:
        if not isinstance(item, dict):
            continue
        if str(item.get("ticker") or "").upper().strip() != ticker:
            continue
        cik = str(item.get("cik_str") or "").strip()
        if not cik:
            break
        return cik.zfill(10), str(item.get("title") or "")
    raise ValueError(f"SEC CIK not found for ticker {ticker}.")


def _load_or_fetch_json(
    url: str,
    path: Path,
    max_cache_age_days: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    if _cache_is_fresh(path, max_cache_age_days):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass

    path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(
        url,
        headers={
            "User-Agent": _sec_user_agent(),
            "Host": "data.sec.gov" if "data.sec.gov" in url else "www.sec.gov",
        },
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _sec_user_agent() -> str:
    return (
        os.environ.get("SEC_USER_AGENT", "").strip()
        or "stock-selector-local-research/1.0 contact@example.com"
    )


def _cache_is_fresh(path: Path, max_cache_age_days: int) -> bool:
    if not path.exists():
        return False
    age = datetime.now(timezone.utc) - datetime.fromtimestamp(
        path.stat().st_mtime,
        tz=timezone.utc,
    )
    return age <= timedelta(days=max_cache_age_days)


def _extract_snapshot_fields(
    ticker: str,
    facts_payload: dict[str, Any],
) -> tuple[dict[str, object], float, list[str]]:
    facts = ((facts_payload.get("facts") or {}).get("us-gaap") or {})
    company_name = str(facts_payload.get("entityName") or "")
    warnings: list[str] = []

    revenue_series = _best_fact_series(
        facts,
        [
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues",
            "SalesRevenueNet",
        ],
        duration=True,
    )
    net_income_series = _best_fact_series(facts, ["NetIncomeLoss"], duration=True)
    operating_cash_flow_series = _best_fact_series(
        facts,
        ["NetCashProvidedByUsedInOperatingActivities"],
        duration=True,
    )
    capex_series = _best_fact_series(
        facts,
        ["PaymentsToAcquirePropertyPlantAndEquipment"],
        duration=True,
    )
    equity_series = _best_fact_series(
        facts,
        [
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ],
        duration=False,
    )
    debt_series = _combined_debt_series(facts)

    latest_revenue = _latest_value(revenue_series)
    previous_revenue = _previous_value(revenue_series)
    latest_net_income = _latest_value(net_income_series)
    previous_net_income = _previous_value(net_income_series)
    latest_operating_cash_flow = _latest_value(operating_cash_flow_series)
    latest_capex = _latest_value(capex_series)
    latest_equity = _latest_value(equity_series)
    latest_debt = _latest_value(debt_series)

    free_cash_flow = (
        latest_operating_cash_flow - abs(latest_capex)
        if latest_operating_cash_flow is not None and latest_capex is not None
        else None
    )
    revenue_growth = _growth(latest_revenue, previous_revenue)
    earnings_growth = _growth(latest_net_income, previous_net_income)
    profit_margin = _ratio(latest_net_income, latest_revenue)
    return_on_equity = _ratio(latest_net_income, latest_equity)
    debt_to_equity = (
        latest_debt / latest_equity * 100.0
        if latest_debt is not None and latest_equity and latest_equity > 0
        else None
    )

    fields: dict[str, object] = {
        "company_name": company_name,
        "revenue": latest_revenue,
        "net_income": latest_net_income,
        "operating_cash_flow": latest_operating_cash_flow,
        "capital_expenditure": latest_capex,
        "free_cash_flow": free_cash_flow,
        "shareholders_equity": latest_equity,
        "total_debt": latest_debt,
        "revenue_growth": revenue_growth,
        "earnings_growth": earnings_growth,
        "profit_margin": profit_margin,
        "return_on_equity": return_on_equity,
        "debt_to_equity": debt_to_equity,
        "sec_latest_revenue_period": _latest_period(revenue_series),
        "sec_latest_net_income_period": _latest_period(net_income_series),
        "sec_latest_balance_sheet_period": _latest_period(equity_series),
    }
    tracked = [
        "revenue",
        "net_income",
        "operating_cash_flow",
        "free_cash_flow",
        "shareholders_equity",
        "revenue_growth",
        "profit_margin",
        "return_on_equity",
        "debt_to_equity",
    ]
    coverage = sum(_has_value(fields.get(field)) for field in tracked) / len(tracked)
    for field in tracked:
        if not _has_value(fields.get(field)):
            warnings.append(f"SEC field unavailable: {field}")
    if coverage == 0:
        warnings.append(f"No usable SEC companyfacts metrics extracted for {ticker}.")
    return fields, round(float(coverage), 2), warnings


def _best_fact_series(
    facts: dict[str, Any],
    names: list[str],
    duration: bool,
) -> list[dict[str, Any]]:
    for name in names:
        series = _fact_series(facts.get(name), duration=duration)
        if series:
            return series
    return []


def _fact_series(fact: Any, duration: bool) -> list[dict[str, Any]]:
    if not isinstance(fact, dict):
        return []
    units = fact.get("units") or {}
    values = units.get("USD") or units.get("shares") or []
    if not isinstance(values, list):
        return []
    rows = []
    for item in values:
        if not isinstance(item, dict):
            continue
        if item.get("form") not in {"10-K", "10-KT"}:
            continue
        if duration and item.get("fp") != "FY":
            continue
        value = _safe_float(item.get("val"))
        if value is None:
            continue
        rows.append(item | {"val": value})
    return sorted(rows, key=lambda row: (str(row.get("end") or ""), str(row.get("filed") or "")))


def _combined_debt_series(facts: dict[str, Any]) -> list[dict[str, Any]]:
    current = _best_fact_series(
        facts,
        [
            "LongTermDebtAndFinanceLeaseObligationsCurrent",
            "LongTermDebtCurrent",
            "DebtCurrent",
            "ShortTermBorrowings",
        ],
        duration=False,
    )
    noncurrent = _best_fact_series(
        facts,
        [
            "LongTermDebtAndFinanceLeaseObligationsNoncurrent",
            "LongTermDebtNoncurrent",
            "LongTermDebt",
        ],
        duration=False,
    )
    combined: dict[str, dict[str, Any]] = {}
    for row in current:
        key = str(row.get("end") or "")
        combined[key] = row | {"val": float(row["val"])}
    for row in noncurrent:
        key = str(row.get("end") or "")
        existing = combined.get(key)
        if existing:
            existing["val"] = float(existing["val"]) + float(row["val"])
        else:
            combined[key] = row | {"val": float(row["val"])}
    return sorted(combined.values(), key=lambda row: (str(row.get("end") or ""), str(row.get("filed") or "")))


def _merge_fact_values(
    rows_by_period: dict[str, dict[str, Any]],
    output_column: str,
    series: list[dict[str, Any]],
    transform=None,
) -> None:
    for item in series:
        period_end = str(item.get("end") or "").strip()
        if not period_end:
            continue
        value = _safe_float(item.get("val"))
        if value is None:
            continue
        if transform is not None:
            value = transform(value)
        row = rows_by_period.setdefault(period_end, {})
        row[output_column] = value
        filed = str(item.get("filed") or "").strip()
        current_filed = str(row.get("filed") or "").strip()
        if filed and (not current_filed or filed > current_filed):
            row["filed"] = filed
            row["form"] = str(item.get("form") or "")
            row["accn"] = str(item.get("accn") or "")


def _safe_date(value: object) -> str | None:
    try:
        if value is None or pd.isna(value):
            return None
        timestamp = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(timestamp):
        return None
    return timestamp.date().isoformat()


def _latest_value(series: list[dict[str, Any]]) -> float | None:
    return _safe_float(series[-1].get("val")) if series else None


def _previous_value(series: list[dict[str, Any]]) -> float | None:
    return _safe_float(series[-2].get("val")) if len(series) >= 2 else None


def _latest_period(series: list[dict[str, Any]]) -> str | None:
    return str(series[-1].get("end") or "") if series else None


def _growth(latest: float | None, previous: float | None) -> float | None:
    if latest is None or previous is None or previous == 0:
        return None
    return latest / abs(previous) - 1.0


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.replace(",", "").replace("%", "").strip()
        if pd.isna(value):
            return None
        result = float(value)
    except Exception:
        return None
    return result if np.isfinite(result) else None


def _has_value(value: object) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        return True
    return True


def _sec_root(data_root: Path) -> Path:
    root = data_root.parent if data_root.name == "real_prices" else data_root
    return root / "sec"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return [_json_safe(nested) for nested in value]
    if isinstance(value, tuple):
        return [_json_safe(nested) for nested in value]
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
