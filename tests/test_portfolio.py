from __future__ import annotations

import unittest

import pandas as pd

from stock_selector.config import RiskConfig
from stock_selector.portfolio import build_risk_managed_portfolio, summarize_exposures


def scored_frame() -> pd.DataFrame:
    rows = []
    dates = pd.bdate_range("2024-01-02", periods=50)
    for ticker_index, ticker in enumerate(["AAA", "BBB", "CCC"]):
        for index, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "score": 10 - ticker_index + index * 0.01,
                    "sector": "Tech" if ticker in {"AAA", "BBB"} else "Health",
                }
            )
    return pd.DataFrame(rows)


def price_frame() -> pd.DataFrame:
    rows = []
    dates = pd.bdate_range("2024-01-02", periods=50)
    for ticker_index, ticker in enumerate(["AAA", "BBB", "CCC"]):
        price = 100.0
        for index, date in enumerate(dates):
            drift = 0.001 * (ticker_index + 1)
            wobble = 0.001 * (ticker_index + 1) * ((index % 3) - 1)
            price *= 1.0 + drift + wobble
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "adj_close": price,
                    "volume": 1_000_000,
                }
            )
    return pd.DataFrame(rows)


class PortfolioRiskTest(unittest.TestCase):
    def test_inverse_volatility_portfolio_caps_weights_and_reports_risk(self) -> None:
        selections, report = build_risk_managed_portfolio(
            scored_frame(),
            price_frame(),
            top_n=3,
            rebalance_frequency="M",
            risk_config=RiskConfig(
                weighting_method="inverse_volatility",
                volatility_lookback_days=20,
                target_annual_volatility=0.05,
                max_position_weight=0.50,
                annualization_days=252,
            ),
        )

        self.assertFalse(selections.empty)
        self.assertFalse(report.empty)
        self.assertLessEqual(selections["weight"].max(), 0.50)
        self.assertTrue((report["gross_exposure"] <= 1.0).all())
        self.assertIn("estimated_annual_volatility", report.columns)

    def test_infeasible_position_cap_leaves_cash(self) -> None:
        selections, report = build_risk_managed_portfolio(
            scored_frame(),
            price_frame(),
            top_n=3,
            rebalance_frequency="M",
            risk_config=RiskConfig(
                weighting_method="inverse_volatility",
                volatility_lookback_days=20,
                max_position_weight=0.25,
            ),
        )

        self.assertLessEqual(selections["weight"].max(), 0.25)
        self.assertTrue((report["gross_exposure"] < 1.0).all())
        self.assertTrue((report["cash_weight"] > 0.0).all())

    def test_sector_cap_limits_group_exposure(self) -> None:
        selections, report = build_risk_managed_portfolio(
            scored_frame(),
            price_frame(),
            top_n=3,
            rebalance_frequency="M",
            risk_config=RiskConfig(
                weighting_method="equal",
                max_position_weight=0.60,
                max_sector_weight=0.55,
            ),
        )
        exposures = summarize_exposures(selections, "sector")

        self.assertLessEqual(report["max_sector_weight"].max(), 0.55)
        self.assertLessEqual(exposures["weight"].max(), 0.55)
        self.assertIn("sector", selections.columns)


if __name__ == "__main__":
    unittest.main()
