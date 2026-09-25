# Stock Selector Long-Term Roadmap

The goal is to build a broad but verifiable stock selection system. The core principles are reliable data, reproducible workflows, lookahead-bias prevention, backtesting before paper trading, and paper trading before live trading.

## 1. Research Scope

- Market scope: start with US daily equities; later expand to A-shares, Hong Kong stocks, and ETFs.
- Universe scope: start from a tradable stock universe, then add filters for listing age, liquidity, market cap, sector, halts, and delistings.
- Output format: generate daily or weekly candidate lists, portfolio weights, risk notes, and reasons for changes.
- Trading assumption: focus on low-to-medium frequency stock selection, not high-frequency trading.

## 2. Data Layer

- Price data: OHLCV, adjusted prices, corporate actions, and delisting data.
- Fundamental data: income statement, balance sheet, cash flow, share count, and valuation data.
- Macro data: interest rates, inflation, employment, credit spreads, US dollar index, and related series.
- Sector and style data: sector classification, market-cap buckets, and style exposures.
- Event and text data: earnings release dates, filings, news, analyst expectations, and sentiment.

Potential data sources:

- SEC EDGAR API: official filings and XBRL financial data through `data.sec.gov`.
- FRED API: official macroeconomic time series.
- Nasdaq Data Link: professional market, fundamental, and alternative data entry point.
- yfinance: useful for research prototypes, but should be replaced or cross-checked before live trading.

## 3. Factor Layer

First stage:

- Momentum: 1, 3, 6, and 12-month returns, including variants that skip the most recent month.
- Volatility and risk: historical volatility, max drawdown, and downside volatility.
- Liquidity: dollar volume, turnover, and spread proxy metrics.
- Trend: moving-average distance, moving-average slope, and breakout strength.

Second stage:

- Value: PE, PB, EV/EBITDA, and FCF yield.
- Quality: ROE, ROIC, gross margin, leverage, and earnings stability.
- Growth: revenue and earnings growth, plus analyst estimate revisions.
- Sentiment: filing, news, and social sentiment with strict data-availability timing.
- Macro regime: interest rates, inflation, and economic-cycle regime.

## 4. Model Layer

- Baseline: factor z-score weighted scoring.
- Machine learning: ElasticNet, Random Forest, Gradient Boosting, LightGBM, and XGBoost after dependency review.
- Training method: rolling-window training with strict time ordering.
- Label design: 1, 4, and 12-week excess return, risk-adjusted return, and quantile classification.
- Evaluation metrics: IC, Rank IC, bucket returns, long-short spread, turnover, and out-of-sample Sharpe.
- Explainability: factor contribution, feature importance, and single-ticker selection rationale.

## 5. Backtest Layer

- Lookahead prevention: all features must use only data available by the signal date.
- Cost model: commission, slippage, market impact, and taxes where applicable.
- Trading constraints: minimum dollar volume, position caps, sector caps, halts, and limit-move rules.
- Bias control: survivorship bias, delisting returns, and filing-date lag.
- Outputs: equity curve, drawdown, annual returns, holdings, trades, and exposure analysis.

## 6. Risk And Portfolio

- Weighting methods: equal weight, score weight, risk parity, and constrained optimization.
- Risk controls: max drawdown, volatility targeting, single-name, sector, and style exposure caps.
- Portfolio review: concentration, liquidity, event risk, and earnings-window risk.

## 7. Productionization

- Daily tasks: data update, factor calculation, model scoring, and report generation.
- Monitoring: missing data, abnormal prices, model drift, and backtest-versus-paper divergence.
- Audit: store input data version, configuration, code version, and outputs for every run.
- Human review: the model provides recommendations; a human confirms final action.

## 8. Current Milestones

- M0: local CSV, baseline factors, and simple backtest. Initial version complete.
- M1: real price download and data quality report. yfinance prototype and local price audit complete.
- M2: fundamental and macro data integration. Schema, sample data, point-in-time merge, and first valuation/quality/growth factors complete.
- M3: machine learning training and rolling validation. Future-return labels, label availability checks, rolling out-of-sample training, Rank IC, error metrics, top-bottom spread, and feature importance complete.
- M4: portfolio optimization, risk controls, and research report. Equal/inverse-volatility weights, position caps, sector caps, target-volatility scaling, cash weight, sector exposure, and Markdown summary complete.
- M5: paper trading workflow. Local portfolio state, rebalance orders, transaction cost estimate, post-trade state, and paper-trading Markdown report complete.
- M6: daily candidate and monitoring report. Latest candidates, data freshness, risk/exposure, turnover, candidate stability, score concentration, feature missing rate, feature drift, ML Rank IC, and threshold warnings complete.
- M7: single-ticker trade plans. Short, medium, and long horizon entry, stop, target, position sizing, and execution-timing labels complete.
