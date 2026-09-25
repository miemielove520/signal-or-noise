from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from stock_selector.config import MLConfig
from stock_selector.ml import build_ml_dataset, rolling_train_predict


def feature_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=70)
    rows = []
    for ticker_index, ticker in enumerate(["AAA", "BBB", "CCC"]):
        price = 100.0 + ticker_index * 20
        for index, date in enumerate(dates):
            price *= 1.0 + 0.001 * (ticker_index + 1) + 0.0005 * np.sin(index / 3)
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "adj_close": price,
                    "passes_universe": index >= 5,
                    "momentum_5": 0.01 * ticker_index + 0.001 * index,
                    "quality": 0.2 + 0.03 * ticker_index,
                }
            )
    return pd.DataFrame(rows)


class MLTest(unittest.TestCase):
    def test_build_ml_dataset_adds_label_date_and_future_return(self) -> None:
        dataset = build_ml_dataset(feature_frame(), ("momentum_5", "quality"), 5)

        first = dataset[dataset["ticker"] == "AAA"].iloc[0]

        self.assertIn("label_date", dataset.columns)
        self.assertGreater(first["label_date"], first["date"])
        self.assertIn("future_excess_return", dataset.columns)

    def test_rolling_train_predict_respects_label_availability(self) -> None:
        dataset = build_ml_dataset(feature_frame(), ("momentum_5", "quality"), 5)
        result = rolling_train_predict(
            dataset,
            MLConfig(
                feature_columns=("momentum_5", "quality"),
                label_forward_days=5,
                train_window_days=80,
                min_train_rows=30,
                min_prediction_assets=3,
                model_type="ridge",
            ),
        )

        self.assertFalse(result.predictions.empty)
        self.assertTrue(
            (result.predictions["latest_train_label_date"] < result.predictions["date"]).all()
        )
        self.assertIn("rank_ic_mean", result.metrics)
        self.assertIn("top_bucket_mean_return", result.metrics)
        self.assertIn("directional_accuracy_advantage", result.metrics)
        self.assertEqual(result.feature_importance["feature"].nunique(), 2)


if __name__ == "__main__":
    unittest.main()
