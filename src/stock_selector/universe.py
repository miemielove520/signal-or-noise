from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .real_data import normalize_ticker


BUILT_IN_UNIVERSES = {
    "mega-cap-tech": [
        "AAPL",
        "MSFT",
        "NVDA",
        "GOOGL",
        "META",
        "AMZN",
        "AVGO",
        "TSLA",
        "NFLX",
        "ORCL",
    ],
    "software": [
        "NOW",
        "CRM",
        "MSFT",
        "ADBE",
        "ORCL",
        "INTU",
        "SNOW",
        "DDOG",
        "PANW",
        "PLTR",
    ],
    "semiconductors": [
        "NVDA",
        "AMD",
        "AVGO",
        "INTC",
        "QCOM",
        "AMAT",
        "MU",
        "TSM",
        "ASML",
    ],
    "ai-infrastructure": [
        "NVDA",
        "AVGO",
        "AMD",
        "ANET",
        "MRVL",
        "SMCI",
        "DELL",
        "ARM",
        "TSM",
        "ASML",
        "AMAT",
    ],
    "cybersecurity": [
        "PANW",
        "CRWD",
        "ZS",
        "FTNT",
        "OKTA",
        "S",
        "QLYS",
        "NET",
    ],
    "fintech-high-beta": [
        "PYPL",
        "XYZ",
        "COIN",
        "HOOD",
        "SOFI",
        "AFRM",
        "UPST",
    ],
    "defensive-quality": [
        "COST",
        "WMT",
        "PG",
        "KO",
        "PEP",
        "MCD",
        "MDLZ",
        "CL",
        "CLX",
    ],
    "industrial-quality": [
        "AAON",
        "CARR",
        "LII",
        "TT",
        "JCI",
        "IR",
        "ETN",
        "HON",
        "PH",
    ],
    "saas-software": [
        "NOW",
        "CRM",
        "SNOW",
        "DDOG",
        "MDB",
        "ADBE",
        "INTU",
        "PLTR",
    ],
    "healthcare-quality": [
        "UNH",
        "LLY",
        "JNJ",
        "ABBV",
        "MRK",
        "TMO",
        "ISRG",
        "SYK",
        "BSX",
        "VRTX",
    ],
    "financial-quality": [
        "JPM",
        "V",
        "MA",
        "BAC",
        "WFC",
        "GS",
        "MS",
        "AXP",
        "CME",
        "ICE",
    ],
    "consumer-discretionary": [
        "AMZN",
        "TSLA",
        "HD",
        "LOW",
        "NKE",
        "SBUX",
        "BKNG",
        "TJX",
        "ORLY",
        "MAR",
    ],
    "consumer-staples": [
        "COST",
        "WMT",
        "PG",
        "KO",
        "PEP",
        "MCD",
        "MDLZ",
        "CL",
        "KMB",
        "GIS",
    ],
    "energy-industrials": [
        "XOM",
        "CVX",
        "COP",
        "EOG",
        "SLB",
        "CAT",
        "DE",
        "GE",
        "ETN",
        "PH",
    ],
    "small-mid-growth": [
        "ALAB",
        "AAON",
        "APP",
        "BILL",
        "CELH",
        "DUOL",
        "ESTC",
        "HUBS",
        "TOST",
        "TTD",
    ],
}

BUILT_IN_UNIVERSES["growth-core"] = list(
    dict.fromkeys(
        BUILT_IN_UNIVERSES["mega-cap-tech"]
        + BUILT_IN_UNIVERSES["software"]
        + BUILT_IN_UNIVERSES["semiconductors"]
        + BUILT_IN_UNIVERSES["ai-infrastructure"]
        + BUILT_IN_UNIVERSES["cybersecurity"]
    )
)

BUILT_IN_UNIVERSES["balanced-core"] = list(
    dict.fromkeys(
        BUILT_IN_UNIVERSES["mega-cap-tech"]
        + BUILT_IN_UNIVERSES["software"]
        + BUILT_IN_UNIVERSES["semiconductors"]
        + BUILT_IN_UNIVERSES["cybersecurity"]
        + BUILT_IN_UNIVERSES["defensive-quality"]
        + BUILT_IN_UNIVERSES["industrial-quality"]
    )
)

BUILT_IN_UNIVERSES["research-core"] = list(
    dict.fromkeys(
        ticker
        for universe_name in [
            "mega-cap-tech",
            "software",
            "semiconductors",
            "ai-infrastructure",
            "cybersecurity",
            "fintech-high-beta",
            "defensive-quality",
            "industrial-quality",
            "saas-software",
            "healthcare-quality",
            "financial-quality",
            "consumer-discretionary",
            "consumer-staples",
            "energy-industrials",
            "small-mid-growth",
        ]
        for ticker in BUILT_IN_UNIVERSES[universe_name]
    )
)

BUILT_IN_UNIVERSES["sector-core"] = list(
    dict.fromkeys(
        ticker
        for universe_name in [
            "mega-cap-tech",
            "semiconductors",
            "saas-software",
            "cybersecurity",
            "financial-quality",
            "healthcare-quality",
            "consumer-staples",
            "consumer-discretionary",
            "industrial-quality",
            "energy-industrials",
        ]
        for ticker in BUILT_IN_UNIVERSES[universe_name]
    )
)


@dataclass(frozen=True)
class HistoricalUniverseMembership:
    records: pd.DataFrame
    source_path: Path | None = None

    @property
    def point_in_time_enabled(self) -> bool:
        return not self.records.empty and "start_date" in self.records.columns

    @property
    def contains_delisted_tickers(self) -> bool:
        if self.records.empty:
            return False
        if "delisted_date" in self.records.columns and self.records["delisted_date"].notna().any():
            return True
        if "status" in self.records.columns:
            status = self.records["status"].fillna("").astype(str).str.lower()
            return bool(status.str.contains("delist").any())
        return False

    @property
    def delisted_ticker_count(self) -> int:
        if self.records.empty:
            return 0
        delisted = pd.Series(False, index=self.records.index)
        if "delisted_date" in self.records.columns:
            delisted = delisted | self.records["delisted_date"].notna()
        if "status" in self.records.columns:
            delisted = delisted | self.records["status"].fillna("").astype(str).str.lower().str.contains(
                "delist"
            )
        return int(self.records.loc[delisted, "ticker"].nunique())

    def tickers(self) -> list[str]:
        return self.records["ticker"].dropna().astype(str).drop_duplicates().to_list()

    def active_record(self, ticker: str, as_of_date: object) -> dict[str, object] | None:
        if self.records.empty:
            return None
        clean_ticker = normalize_ticker(str(ticker))
        signal_date = pd.Timestamp(as_of_date).normalize()
        rows = self.records[self.records["ticker"] == clean_ticker]
        if rows.empty:
            return None
        active = rows[
            (rows["start_date"].isna() | (rows["start_date"] <= signal_date))
            & (rows["end_date"].isna() | (signal_date <= rows["end_date"]))
        ]
        if active.empty:
            return None
        return active.sort_values("start_date").iloc[-1].to_dict()

    def survivorship_report(self, tickers: list[str] | tuple[str, ...]) -> dict[str, object]:
        requested = {normalize_ticker(str(ticker)) for ticker in tickers if str(ticker).strip()}
        relevant = self.records[self.records["ticker"].isin(requested)] if requested else self.records
        contains_delisted = False
        if not relevant.empty:
            contains_delisted = HistoricalUniverseMembership(relevant).contains_delisted_tickers
        handled = self.point_in_time_enabled
        warnings: list[str] = []
        warnings_zh: list[str] = []
        if not handled:
            warnings.append("Historical universe membership is missing point-in-time dates.")
            warnings_zh.append("历史成分股文件缺少时点日期，无法确认幸存者偏差已处理。")
        if handled and not contains_delisted:
            warnings.append("No delisted tickers are marked in the current historical universe sample.")
            warnings_zh.append("当前历史股票池样本没有标记退市股票，退市覆盖仍需继续补充。")
        return {
            "survivorship_bias_handled": handled,
            "point_in_time_universe": handled,
            "contains_delisted_tickers": contains_delisted,
            "historical_constituent_count": int(relevant["ticker"].nunique()) if not relevant.empty else 0,
            "delisted_ticker_count": HistoricalUniverseMembership(relevant).delisted_ticker_count
            if not relevant.empty
            else 0,
            "source": str(self.source_path) if self.source_path else "in_memory",
            "warnings": warnings,
            "warnings_zh": warnings_zh,
        }


def load_universe_tickers(
    tickers: list[str] | tuple[str, ...] | None = None,
    universe_name: str | None = None,
    universe_file: str | Path | None = None,
) -> list[str]:
    values: list[str] = []
    if universe_name:
        key = universe_name.lower().strip()
        if key not in BUILT_IN_UNIVERSES:
            choices = ", ".join(sorted(BUILT_IN_UNIVERSES))
            raise ValueError(f"Unknown built-in universe '{universe_name}'. Choices: {choices}.")
        values.extend(BUILT_IN_UNIVERSES[key])
    if universe_file:
        values.extend(_load_tickers_from_file(Path(universe_file)))
    if tickers:
        values.extend(tickers)

    normalized = []
    for ticker in values:
        clean = normalize_ticker(str(ticker))
        if clean and clean not in normalized:
            normalized.append(clean)
    if not normalized:
        raise ValueError("At least one ticker or universe is required.")
    return normalized


def load_historical_universe_membership(path: str | Path) -> HistoricalUniverseMembership:
    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"Historical universe file not found: {source_path}")
    frame = pd.read_csv(source_path)
    if frame.empty:
        raise ValueError("Historical universe file is empty.")
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    ticker_column = _ticker_column(frame)
    frame["ticker"] = frame[ticker_column].astype(str).map(normalize_ticker)
    frame = frame[frame["ticker"] != ""].copy()
    if frame.empty:
        raise ValueError("Historical universe file does not contain valid tickers.")

    if "as_of_date" in frame.columns and "start_date" not in frame.columns:
        records = _membership_from_snapshots(frame)
    else:
        records = frame.copy()
        if "start_date" not in records.columns:
            raise ValueError(
                "Historical universe CSV must include start_date/end_date or as_of_date snapshots."
            )
        records["start_date"] = _optional_datetime(records["start_date"])
        if "end_date" in records.columns:
            records["end_date"] = _optional_datetime(records["end_date"])
        else:
            records["end_date"] = pd.NaT
        if "as_of_date" in records.columns:
            records["as_of_date"] = _optional_datetime(records["as_of_date"])
        else:
            records["as_of_date"] = records["start_date"]

    if "delisted_date" in records.columns:
        records["delisted_date"] = _optional_datetime(records["delisted_date"])
        records["end_date"] = records["end_date"].where(
            records["end_date"].notna(),
            records["delisted_date"],
        )
    else:
        records["delisted_date"] = pd.NaT
    if "delisting_return" in records.columns:
        records["delisting_return"] = pd.to_numeric(records["delisting_return"], errors="coerce")
    else:
        records["delisting_return"] = pd.NA
    if "status" not in records.columns:
        records["status"] = ""

    records = records[
        [
            "ticker",
            "start_date",
            "end_date",
            "as_of_date",
            "delisted_date",
            "delisting_return",
            "status",
        ]
    ].sort_values(["ticker", "start_date", "end_date"])
    return HistoricalUniverseMembership(records.reset_index(drop=True), source_path=source_path)


def survivorship_report_without_historical_membership() -> dict[str, object]:
    return {
        "survivorship_bias_handled": False,
        "point_in_time_universe": False,
        "contains_delisted_tickers": False,
        "historical_constituent_count": 0,
        "delisted_ticker_count": 0,
        "source": "none",
        "warnings": [
            "No historical universe membership file was supplied; validation may have survivorship bias."
        ],
        "warnings_zh": [
            "没有提供历史成分股文件；验证结果可能存在幸存者偏差。"
        ],
    }


def _load_tickers_from_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Universe file not found: {path}")
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        ticker_column = _ticker_column(frame)
        return frame[ticker_column].dropna().astype(str).to_list()

    text = path.read_text(encoding="utf-8")
    tickers: list[str] = []
    for line in text.splitlines():
        clean = line.split("#", 1)[0].strip()
        if not clean:
            continue
        tickers.extend(part.strip() for part in clean.split(",") if part.strip())
    return tickers


def _ticker_column(frame: pd.DataFrame) -> str:
    lower_to_original = {column.lower().strip(): column for column in frame.columns}
    for candidate in ["ticker", "symbol"]:
        if candidate in lower_to_original:
            return lower_to_original[candidate]
    return frame.columns[0]


def _membership_from_snapshots(frame: pd.DataFrame) -> pd.DataFrame:
    snapshots = frame.copy()
    snapshots["as_of_date"] = _optional_datetime(snapshots["as_of_date"])
    snapshots = snapshots.dropna(subset=["as_of_date"])
    if snapshots.empty:
        raise ValueError("Historical universe snapshots do not contain valid as_of_date values.")
    snapshot_dates = sorted(pd.Timestamp(date).normalize() for date in snapshots["as_of_date"].unique())
    rows: list[dict[str, object]] = []
    for index, snapshot_date in enumerate(snapshot_dates):
        next_date = snapshot_dates[index + 1] if index + 1 < len(snapshot_dates) else pd.NaT
        end_date = next_date - pd.Timedelta(days=1) if pd.notna(next_date) else pd.NaT
        snapshot_rows = snapshots[snapshots["as_of_date"] == snapshot_date]
        for row in snapshot_rows.itertuples(index=False):
            values = row._asdict()
            values["start_date"] = snapshot_date
            values["end_date"] = end_date
            rows.append(values)
    return pd.DataFrame(rows)


def _optional_datetime(values: object) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", utc=False).dt.normalize()
