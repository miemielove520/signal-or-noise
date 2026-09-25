from __future__ import annotations

import unittest

import pandas as pd

from stock_selector.factors import add_fundamental_factors, add_macro_features


def base_price_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-15"),
                "ticker": "AAA",
                "adj_close": 100.0,
                "passes_universe": True,
            },
            {
                "date": pd.Timestamp("2024-02-20"),
                "ticker": "AAA",
                "adj_close": 110.0,
                "passes_universe": True,
            },
        ]
    )


def fundamental_frame() -> pd.DataFrame:
    report_dates = pd.to_datetime(
        [
            "2022-04-30",
            "2022-07-31",
            "2022-10-31",
            "2023-01-31",
            "2023-04-30",
            "2023-07-31",
            "2023-10-31",
            "2024-01-31",
        ]
    )
    rows = []
    for index, report_date in enumerate(report_dates):
        revenue = 100.0 + index * 10.0
        rows.append(
            {
                "report_date": report_date,
                "period_end": report_date - pd.Timedelta(days=30),
                "ticker": "AAA",
                "revenue": revenue,
                "gross_profit": revenue * 0.5,
                "operating_income": revenue * 0.2,
                "net_income": revenue * 0.1,
                "book_value": 1_000.0 + index * 20.0,
                "total_assets": 1_600.0 + index * 20.0,
                "total_liabilities": 600.0,
                "operating_cash_flow": revenue * 0.12,
                "capital_expenditure": revenue * 0.02,
                "shares_outstanding": 10.0,
            }
        )
    return pd.DataFrame(rows)


class FeatureEnrichmentTest(unittest.TestCase):
    def test_fundamental_merge_is_point_in_time(self) -> None:
        enriched = add_fundamental_factors(base_price_frame(), fundamental_frame())

        first = enriched[enriched["date"] == pd.Timestamp("2024-01-15")].iloc[0]
        second = enriched[enriched["date"] == pd.Timestamp("2024-02-20")].iloc[0]

        self.assertEqual(first["report_date"], pd.Timestamp("2023-10-31"))
        self.assertEqual(second["report_date"], pd.Timestamp("2024-01-31"))
        self.assertGreaterEqual(first["fundamental_age_days"], 0)
        self.assertGreater(first["earnings_yield"], 0)
        self.assertIn("revenue_growth_yoy", enriched.columns)

    def test_macro_features_are_prefixed_and_asof_merged(self) -> None:
        macro = pd.DataFrame(
            [
                {"date": pd.Timestamp("2024-01-01"), "fed_funds_rate": 4.5},
                {"date": pd.Timestamp("2024-02-01"), "fed_funds_rate": 4.75},
            ]
        )

        enriched = add_macro_features(base_price_frame(), macro)

        first = enriched[enriched["date"] == pd.Timestamp("2024-01-15")].iloc[0]
        second = enriched[enriched["date"] == pd.Timestamp("2024-02-20")].iloc[0]
        self.assertEqual(first["macro_fed_funds_rate"], 4.5)
        self.assertEqual(second["macro_fed_funds_rate"], 4.75)


if __name__ == "__main__":
    unittest.main()
