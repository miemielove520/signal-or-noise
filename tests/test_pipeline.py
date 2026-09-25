from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from stock_selector.config import (
    BacktestConfig,
    DataConfig,
    FactorConfig,
    MLConfig,
    MonitorConfig,
    PaperTradingConfig,
    ResearchConfig,
    RiskConfig,
    ScoringConfig,
    UniverseConfig,
)
from stock_selector.pipeline import run_research_pipeline


def make_prices() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=90)
    rows = []
    specs = {
        "AAA": (100.0, 0.0018, 1_000_000),
        "BBB": (50.0, 0.0008, 1_500_000),
        "CCC": (80.0, -0.0002, 2_000_000),
    }
    for ticker, (start_price, drift, volume) in specs.items():
        price = start_price
        for index, date in enumerate(dates):
            price *= 1.0 + drift + 0.002 * np.sin(index / 5)
            rows.append(
                {
                    "date": date.date().isoformat(),
                    "ticker": ticker,
                    "open": price * 0.99,
                    "high": price * 1.01,
                    "low": price * 0.98,
                    "close": price,
                    "adj_close": price,
                    "volume": volume,
                }
            )
    return pd.DataFrame(rows)


class PipelineTest(unittest.TestCase):
    def test_pipeline_produces_selections_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            price_path = Path(tmpdir) / "prices.csv"
            make_prices().to_csv(price_path, index=False)

            config = ResearchConfig(
                data=DataConfig(prices_csv=price_path),
                universe=UniverseConfig(min_history_days=40, min_avg_dollar_volume=100_000),
                factors=FactorConfig(
                    momentum_windows=(20, 40),
                    volatility_window=20,
                    trend_window=30,
                    liquidity_window=20,
                ),
                scoring=ScoringConfig(
                    top_n=2,
                    rebalance_frequency="M",
                    factor_weights={
                        "momentum_20": 0.4,
                        "momentum_40": 0.3,
                        "low_volatility_20": 0.2,
                        "trend_30": 0.1,
                    },
                ),
                risk=RiskConfig(),
                backtest=BacktestConfig(initial_capital=100_000),
                ml=MLConfig(),
                paper=PaperTradingConfig(),
                monitor=MonitorConfig(),
            )

            result = run_research_pipeline(config)

            self.assertFalse(result.selections.empty)
            self.assertFalse(result.risk_report.empty)
            self.assertIn("ending_equity", result.metrics)
            self.assertGreater(result.metrics["ending_equity"], 0)
            self.assertTrue((result.selections["weight"] > 0).all())


if __name__ == "__main__":
    unittest.main()
