from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.bdate_range("2024-01-02", periods=140)
    specs = {
        "ALFA": {
            "sector": "Technology",
            "industry": "Software",
            "start_price": 100.0,
            "drift": 0.0016,
            "base_volume": 1_200_000,
            "revenue": 520_000_000,
            "growth": 0.035,
            "gross_margin": 0.55,
            "net_margin": 0.17,
            "book_value": 1_650_000_000,
            "debt_ratio": 0.45,
            "shares": 22_000_000,
        },
        "BRAV": {
            "sector": "Industrials",
            "industry": "Machinery",
            "start_price": 65.0,
            "drift": 0.0007,
            "base_volume": 2_000_000,
            "revenue": 780_000_000,
            "growth": 0.018,
            "gross_margin": 0.40,
            "net_margin": 0.10,
            "book_value": 2_100_000_000,
            "debt_ratio": 0.70,
            "shares": 38_000_000,
        },
        "CHAR": {
            "sector": "Consumer",
            "industry": "Retail",
            "start_price": 42.0,
            "drift": -0.0001,
            "base_volume": 1_800_000,
            "revenue": 430_000_000,
            "growth": -0.006,
            "gross_margin": 0.32,
            "net_margin": 0.04,
            "book_value": 1_200_000_000,
            "debt_ratio": 1.10,
            "shares": 31_000_000,
        },
        "DELT": {
            "sector": "Technology",
            "industry": "Semiconductors",
            "start_price": 88.0,
            "drift": 0.0011,
            "base_volume": 900_000,
            "revenue": 610_000_000,
            "growth": 0.026,
            "gross_margin": 0.48,
            "net_margin": 0.13,
            "book_value": 1_900_000_000,
            "debt_ratio": 0.55,
            "shares": 26_000_000,
        },
        "ECHO": {
            "sector": "Healthcare",
            "industry": "Medical Devices",
            "start_price": 120.0,
            "drift": 0.0003,
            "base_volume": 2_500_000,
            "revenue": 900_000_000,
            "growth": 0.010,
            "gross_margin": 0.36,
            "net_margin": 0.07,
            "book_value": 2_400_000_000,
            "debt_ratio": 0.85,
            "shares": 30_000_000,
        },
    }

    price_rows = build_price_rows(dates, specs)
    price_path = data_dir / "sample_prices.csv"
    pd.DataFrame(price_rows).to_csv(price_path, index=False)
    print(f"Wrote {price_path}")

    fundamental_rows = build_fundamental_rows(specs)
    fundamental_path = data_dir / "sample_fundamentals.csv"
    pd.DataFrame(fundamental_rows).to_csv(fundamental_path, index=False)
    print(f"Wrote {fundamental_path}")

    macro_rows = build_macro_rows()
    macro_path = data_dir / "sample_macro.csv"
    pd.DataFrame(macro_rows).to_csv(macro_path, index=False)
    print(f"Wrote {macro_path}")

    metadata_rows = build_metadata_rows(specs)
    metadata_path = data_dir / "sample_metadata.csv"
    pd.DataFrame(metadata_rows).to_csv(metadata_path, index=False)
    print(f"Wrote {metadata_path}")

    state_path = data_dir / "sample_portfolio_state.csv"
    pd.DataFrame([{"ticker": "CASH", "quantity": 100_000.0}]).to_csv(
        state_path,
        index=False,
    )
    print(f"Wrote {state_path}")
    return 0


def build_price_rows(
    dates: pd.DatetimeIndex,
    specs: dict[str, dict[str, float]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ticker, spec in specs.items():
        price = spec["start_price"]
        for index, date in enumerate(dates):
            seasonal = 0.0025 * np.sin(index / 7.0)
            idiosyncratic = 0.0015 * np.cos((index + len(ticker)) / 11.0)
            price *= 1.0 + spec["drift"] + seasonal + idiosyncratic
            volume = int(spec["base_volume"] * (1.0 + 0.08 * np.sin(index / 9.0)))
            rows.append(
                {
                    "date": date.date().isoformat(),
                    "ticker": ticker,
                    "open": round(price * 0.995, 4),
                    "high": round(price * 1.012, 4),
                    "low": round(price * 0.988, 4),
                    "close": round(price, 4),
                    "adj_close": round(price, 4),
                    "volume": volume,
                }
            )
    return rows


def build_fundamental_rows(specs: dict[str, dict[str, float]]) -> list[dict[str, object]]:
    quarter_ends = pd.date_range("2021-12-31", periods=11, freq="QE")
    rows: list[dict[str, object]] = []
    for ticker, spec in specs.items():
        for index, period_end in enumerate(quarter_ends):
            seasonal = 1.0 + 0.025 * np.sin(index / 2.0)
            revenue = spec["revenue"] * ((1.0 + spec["growth"]) ** index) * seasonal
            gross_profit = revenue * spec["gross_margin"]
            operating_income = revenue * (spec["net_margin"] + 0.055)
            net_income = revenue * spec["net_margin"]
            book_value = spec["book_value"] * (1.0 + 0.015 * index)
            total_liabilities = book_value * spec["debt_ratio"]
            total_assets = book_value + total_liabilities
            operating_cash_flow = net_income * 1.15
            capital_expenditure = revenue * 0.035
            report_date = period_end + pd.Timedelta(days=35)
            rows.append(
                {
                    "report_date": report_date.date().isoformat(),
                    "period_end": period_end.date().isoformat(),
                    "ticker": ticker,
                    "revenue": round(revenue, 2),
                    "gross_profit": round(gross_profit, 2),
                    "operating_income": round(operating_income, 2),
                    "net_income": round(net_income, 2),
                    "book_value": round(book_value, 2),
                    "total_assets": round(total_assets, 2),
                    "total_liabilities": round(total_liabilities, 2),
                    "operating_cash_flow": round(operating_cash_flow, 2),
                    "capital_expenditure": round(capital_expenditure, 2),
                    "shares_outstanding": int(spec["shares"]),
                }
            )
    return rows


def build_macro_rows() -> list[dict[str, object]]:
    dates = pd.date_range("2023-01-01", "2024-08-01", freq="MS")
    rows: list[dict[str, object]] = []
    for index, date in enumerate(dates):
        rows.append(
            {
                "date": date.date().isoformat(),
                "fed_funds_rate": round(4.25 + 0.025 * index, 4),
                "cpi_yoy": round(3.8 - 0.035 * index, 4),
                "unemployment_rate": round(3.7 + 0.015 * np.sin(index / 2.5), 4),
            }
        )
    return rows


def build_metadata_rows(specs: dict[str, dict[str, float]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ticker, spec in specs.items():
        rows.append(
            {
                "ticker": ticker,
                "sector": spec["sector"],
                "industry": spec["industry"],
                "country": "US",
                "exchange": "SAMPLE",
            }
        )
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
