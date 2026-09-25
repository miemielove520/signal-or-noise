# Stock Selector Research Project

This is a local stock selection research framework. It reads daily price data, builds factors, scores stocks, constructs risk-managed portfolios, runs backtests, produces paper-trading orders, and generates single-ticker trade plans.

> This project is for research and engineering validation only. It is not financial advice. Before live trading, validate data licensing, point-in-time data, out-of-sample behavior, transaction costs, slippage, risk controls, and manual review.

## Current Capabilities

- Load local CSV daily OHLCV data.
- Audit price data quality:
  - Missing required columns.
  - Duplicate date/ticker rows.
  - Missing or unparseable values.
  - Non-positive prices, negative volume, and invalid OHLC ranges.
  - Abnormal one-day moves, long gaps, and stale prices.
- Download research prototype prices with a provider order:
  - Optional Polygon and Alpaca price data when API keys are configured.
  - yfinance as the default free fallback source.
  - Price source cross-checks for close price, volume, and missing trading dates.
- Merge point-in-time fundamentals using `report_date <= signal date`.
- Merge macro time series into trading-day feature rows.
- Merge ticker metadata such as sector, industry, country, and exchange.
- Build baseline factors:
  - Momentum: 20-day and 60-day returns.
  - Volatility: 20-day return volatility, where lower volatility scores better.
  - Trend: price versus the 50-day moving average.
  - Liquidity: 20-day average dollar volume.
- Build fundamental factors:
  - Valuation: earnings yield, book-to-market, free cash flow yield.
  - Quality: ROE, gross margin, low debt-to-equity.
  - Growth: TTM revenue growth.
- Run rolling machine learning validation:
  - Build future-return labels and `label_date`.
  - Train only on labels known before each prediction date.
  - Report Rank IC, top bucket return/win rate/Sharpe, top-bottom spread, and feature importance.
- Build cross-sectional factor scores.
- Select top-ranked stocks on a rebalance schedule.
- Support equal-weight and inverse-volatility portfolio weights.
- Support maximum position weight, sector weight cap, target volatility, and cash weight.
- Generate local paper-trading rebalance orders.
- Generate daily candidate and monitoring reports.
- Generate single-ticker short, medium, and long horizon trade plans:
  - Trend, momentum, support, resistance, ATR, and volume state.
  - Conditional entry price, stop loss, take profit, and risk/reward.
  - Position sizing based on capital and risk budget.
- Track prior ticker signals and convert completed 5/20/60 trading-day outcomes into a review learning score.
- Optionally save a current yfinance company/analyst/news snapshot. This snapshot is not used in historical backtests.
- Apply backtest signals on the next trading day by default to avoid same-close lookahead bias.
- Report CAGR, annual volatility, Sharpe, max drawdown, turnover, invested-days ratio, and ending equity.
- Unit tests cover core workflows.

## Quick Start

Create the project environment once:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[ml,dev]'
```

Analyze a real stock with the simple entry point:

```bash
python3 run.py
```

Then type a ticker symbol such as `AAPL`, `NVDA`, or `MSFT`.

You can also pass the ticker directly:

```bash
python3 run.py AAPL
```

This downloads real daily price data, writes a cached CSV under `data/real_prices/`, caches current company snapshot fields under `data/real_snapshots/`, writes the report under `outputs/real_ticker/<TICKER>/`, and compares the ticker with a small peer group when current company metadata is available.

Data providers are pluggable. Prices come from a registry of sources tried in order; if no paid keys are configured, the app still runs on free sources:

```bash
export STOCK_SELECTOR_DATA_PROVIDER_ORDER=tiingo,polygon,alpaca,yfinance,stooq
export TIINGO_API_TOKEN=...
export POLYGON_API_KEY=...
export ALPACA_API_KEY_ID=...
export ALPACA_API_SECRET_KEY=...
export SEC_USER_AGENT="your-name your-email@example.com"
export FRED_API_KEY=...
```

Price providers, in default preference order:

- `polygon`, `alpaca`, `tiingo`: used first when their API keys are set (all free-tier friendly; Tiingo needs a free `TIINGO_API_TOKEN`).
- `yfinance`: the default free primary source, no key required.
- `stooq`: a free second source (no key) added so cross-source price validation runs even without paid keys. Note: Stooq blocks datacenter/cloud IPs with an anti-bot challenge, so it typically only returns data from a residential IP; it degrades gracefully to yfinance otherwise. Disable it with `STOCK_SELECTOR_DISABLE_STOOQ=1`.

When two or more sources return data, close price, volume, and missing trading dates are cross-checked on the latest common trading day and reported under `source_validation` (`validated` or `conflict_warning`). Adding a new source is a one-line registry entry plus a `download_*_prices_for_period(tickers, period, timeout_seconds)` function in `src/stock_selector/data.py`.

`TIINGO_API_TOKEN`, `POLYGON_API_KEY`, `ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY`, `SEC_USER_AGENT`, and `FRED_API_KEY` are optional. SEC companyfacts is used to fill missing fundamental fields when available. SEC filing-date history can also be exported as point-in-time fundamentals with `fundamentals_source=sec_pit`; yfinance-derived fundamental CSVs are treated as `yfinance_restated`. FRED readiness is currently reported as a planned framework field; it does not replace the scoring model yet.

Single-ticker real analysis now writes both human-readable and app-ready outputs:

```text
outputs/real_ticker/<TICKER>/ticker_analysis.md
outputs/real_ticker/<TICKER>/ticker_analysis.csv
outputs/real_ticker/<TICKER>/analysis_result.json
outputs/real_ticker/<TICKER>/cache_metadata.json
outputs/real_ticker/<TICKER>/data_readiness.json
outputs/real_ticker/<TICKER>/data_readiness.md
outputs/real_ticker/<TICKER>/data_readiness.csv
outputs/real_ticker/<TICKER>/sec_fundamentals.json
data/sec/extracted/<TICKER>_sec_fundamental_history.csv
```

The strict high-probability screening rules are configurable. The default file is:

```text
configs/screening.toml
```

You normally do not need to pass it because the built-in defaults match this file. The real ticker workflow automatically resolves a screening profile from ticker, sector, and industry:

- `ai_infrastructure` / AI基建规则
- `cybersecurity` / 网络安全规则
- `saas_software` / SaaS软件规则
- `fintech_high_beta` / 金融科技高波动规则
- `defensive_quality` / 防御质量股规则
- `industrial_quality` / 工业质量股规则
- `semiconductor` / 半导体规则
- `mega_cap_tech` / 大盘科技规则
- `default` / 默认规则

Each report and JSON result includes `screening_profile` and `screening_profile_zh`, so you can see which rule set was used. To test a stricter or looser rule set:

```bash
python3 run.py AAPL --screening-config configs/screening.toml
```

Each profile can also carry its own trading rules under `[profiles.<name>.trading]`.
每个大类也可以在 `[profiles.<name>.trading]` 下面配置自己的交易规则：

- `preferred_entry_style`: `breakout`, `pullback`, or `balanced`
- `short_atr_stop_multiple`, `medium_atr_stop_multiple`, `long_atr_stop_multiple`
- `short_target_r_multiple`, `medium_target_r_multiple`, `long_target_r_multiple`
- `short_max_chase_pct`, `medium_max_chase_pct`, `long_max_chase_pct`
- `short_time_stop_days`, `medium_time_stop_days`, `long_time_stop_days`
- `trailing_stop_trigger_r`, `trailing_stop_lock_r`
- `sell_rule`, `sell_rule_zh`

The ticker report shows these fields in `Profile Trading Rules / 大类交易规则`.
单股报告会在 `Profile Trading Rules / 大类交易规则` 里显示当前股票实际使用的买点、止损、止盈和卖出规则。

Scan multiple tickers for strict high-probability candidates:

```bash
python3 scan.py AAPL MSFT NVDA NOW CRM AVGO AAON
```

Scan a built-in universe and write a daily journal:

```bash
python3 scan.py --universe software --journal
```

Supported built-in universes:

```text
mega-cap-tech
software
semiconductors
ai-infrastructure
cybersecurity
saas-software
fintech-high-beta
defensive-quality
industrial-quality
healthcare-quality
financial-quality
consumer-discretionary
consumer-staples
energy-industrials
small-mid-growth
growth-core
balanced-core
research-core
sector-core
```

The scan writes:

```text
outputs/scans/latest/high_probability_scan.csv
outputs/scans/latest/high_probability_scan.md
outputs/scans/latest/top_candidates.csv
outputs/scans/latest/top_candidates.md
outputs/scans/latest/top_candidates.json
outputs/scans/latest/data_readiness_summary.csv
outputs/scans/latest/scan_result.json
outputs/scans/latest/cache_metadata.json
outputs/journal/daily_journal.md
outputs/journal/journal_result.json
```

Start with `top_candidates.md` after a batch scan. It is the short first-read page that shows the best candidates, near misses, main blockers, trigger prices, and next steps.
批量扫描后优先看 `top_candidates.md`。它是简化版首页，会显示最佳候选、接近机会、主要卡点、触发价和下一步。

Use `data_readiness_summary.csv` when you want to find tickers whose results need repair before they can be trusted as high-confidence candidates.
如果你想先找出哪些股票的数据结果不够可靠，就看 `data_readiness_summary.csv`。

Generate a local paper-trading simulation from the latest scan:

```bash
python3 paper.py
```

This reads `outputs/scans/latest/high_probability_scan.csv`, converts strict high-probability candidates into target weights, and writes simulated orders without sending anything to a broker.
这会读取 `outputs/scans/latest/high_probability_scan.csv`，把严格高概率候选转成模拟目标仓位，并生成本地模拟订单；它不会连接券商，也不会真实下单。

Useful options:

```bash
python3 paper.py --allow-near-watchlist
python3 paper.py --initial-cash 10000 --max-positions 5 --max-position-weight 0.20
python3 paper.py --market-regime-policy outputs/walk_forward/latest/market_regime_policy.csv
python3 paper.py --ignore-market-regime-policy
python3 paper.py --update-state
```

`--update-state` is the only mode that writes the simulated post-trade portfolio back to `outputs/paper_latest/paper_state.csv`. Without it, the run is a dry run.
只有加 `--update-state` 时，程序才会把模拟交易后的仓位写回 `outputs/paper_latest/paper_state.csv`。不加这个参数时，只是演练。

By default, `paper.py` tries to read `outputs/walk_forward/latest/market_regime_policy.csv` when it exists. If the policy says the current regime should block new entries or reduce exposure, paper targets are filtered or scaled before simulated orders are generated.
默认情况下，如果 `outputs/walk_forward/latest/market_regime_policy.csv` 存在，`paper.py` 会读取它。如果保护规则要求暂停新开仓或降低暴露，模拟目标会在生成订单前被过滤或缩小。

The paper simulation writes:

```text
outputs/paper_latest/paper_targets.csv
outputs/paper_latest/prices_used.csv
outputs/paper_latest/orders.csv
outputs/paper_latest/pre_trade_state.csv
outputs/paper_latest/post_trade_state.csv
outputs/paper_latest/paper_trade_report.md
outputs/paper_latest/paper_trade_result.json
```

`outputs/paper_latest/` is the current view and is replaced by the next run. Every
rebalance and mark-to-market is also written to the append-only audit folder:

`outputs/paper_latest/` 是当前视图，下一次运行会覆盖它。每次再平衡和每日估值还会
同时写入不会被后续运行覆盖的完整审计目录：

```text
outputs/paper_history/paper_journal.md
outputs/paper_history/paper_runs.csv
outputs/paper_history/paper_orders.csv
outputs/paper_history/paper_positions.csv
outputs/paper_history/paper_valuations.csv
outputs/paper_history/runs/<RUN_ID>/
outputs/paper_history/valuations/<VALUATION_ID>/
```

Each immutable rebalance snapshot includes the full analysis input, exact prices
used, targets, orders, pre/post states, valued positions, configuration, code
revision, transaction-cost breakdown, checksums, and the generated report. Failed
runs are recorded as failures instead of disappearing silently.

每个不可覆盖的再平衡快照都会保存完整输入、实际使用价格、目标仓位、模拟订单、
交易前后状态、估值仓位、配置、代码版本、成本拆分、文件校验值和报告。运行失败也会
记录失败原因，不会静默消失。

Run one ticker without peer comparison when you want faster output:

```bash
python3 run.py NOW --no-peers
```

Check which recorded ticker signals are due for 5/20/60 trading-day review:

```bash
python3 review_due.py
```

Refresh only the tickers whose review windows are already due:

```bash
python3 review_due.py --refresh-due
```

This writes:

```text
outputs/signal_review/signal_review_due.csv
outputs/signal_review/signal_review_due.md
outputs/signal_review/signal_review_due.json
```

It separates `due_now` signals from still-pending signals, so you can see which tickers need a refreshed run to fill their review outcome.

The signal review summary also reports a learning state:

- `review_learning_score`: overall historical signal feedback score / 历史信号反馈总分。
- `review_learning_level`: whether prior signals are supportive, neutral, weak, or still insufficient / 历史信号是支持、中性、偏弱，还是样本不足。
- `review_learning_adjustment`: the score adjustment fed back into future ticker analysis / 反馈到之后个股分析里的调分。
- `review_learning_focus_window`: the 5d, 20d, or 60d window with the most useful completed evidence / 当前最有参考价值的复盘窗口。


## Data Format

Price CSV files must include:

```text
date,ticker,open,high,low,close,adj_close,volume
```

Fundamental CSV files must include:

```text
report_date,period_end,ticker,revenue,gross_profit,operating_income,net_income,book_value,total_assets,total_liabilities,operating_cash_flow,capital_expenditure,shares_outstanding
```

Use `report_date` as the date when the market could know the filing. For SEC companyfacts history, this is the SEC `filed` date, not the fiscal `period_end`. Do not replace it with `period_end`, because that creates lookahead bias. Optional `fundamentals_source` values are preserved; missing source values default to `yfinance_restated`.

Macro CSV files must include `date` and at least one numeric value column, for example:

```text
date,fed_funds_rate,cpi_yoy,unemployment_rate
```

Metadata CSV files must include:

```text
ticker,sector,industry,country,exchange
```

Paper-trading state CSV files use:

```text
ticker,quantity
```

The `CASH` row stores cash. Other rows store share quantities.

## Backtest Configuration

The `[backtest]` section controls capital, cost, and signal execution timing:

```toml
[backtest]
initial_capital = 100000
annualization_days = 252
risk_free_rate = 0.0
transaction_cost_bps = 5
execution_lag_days = 1
```

- `execution_lag_days`: delays signal execution by at least one trading day. The default is `1`.
- `transaction_cost_bps`: deducts costs from actual executed turnover.

## Risk Configuration

The `[risk]` section controls portfolio construction:

```toml
[risk]
weighting_method = "inverse_volatility"
volatility_lookback_days = 40
target_annual_volatility = 0.12
max_position_weight = 0.60
max_sector_weight = 0.70
sector_column = "sector"
min_position_weight = 0.0
annualization_days = 252
```

- `weighting_method`: `equal` or `inverse_volatility`.
- `target_annual_volatility`: scales exposure down when estimated volatility is above target.
- `max_position_weight`: caps single-position weight.
- `max_sector_weight`: caps sector exposure.
- `risk_report.csv`: records gross exposure, max position weight, max sector weight, estimated volatility, and cash weight.

## Paper Trading Configuration

The `[paper]` section controls local paper-trading order generation:

```toml
[paper]
state_csv = "data/sample_portfolio_state.csv"
initial_cash = 100000
min_trade_value = 100
trade_buffer_pct = 0.001
slippage_bps = 5
commission_bps = 0
allow_fractional_shares = true
```

- `state_csv`: local paper portfolio state file.
- `min_trade_value` and `trade_buffer_pct`: filter small rebalance orders.
- `slippage_bps` and `commission_bps`: estimate trading costs.
- `allow_fractional_shares`: enables fractional share sizing.

## Monitor Configuration

The `[monitor]` section controls daily monitoring thresholds:

```toml
[monitor]
max_data_age_days = 7
max_average_feature_missing_rate = 0.10
max_single_feature_missing_rate = 0.25
drift_lookback_days = 60
max_feature_drift_zscore = 3.0
min_candidate_overlap = 0.50
max_score_concentration = 0.80
min_rank_ic_mean = 0.0
```

## Project Structure

```text
configs/default.toml              Default research configuration
run.py                            Simple real-stock analysis entry point
data/real_prices/                 Cached real yfinance price files
data/real_snapshots/              Cached current company snapshot fallbacks
data/sample_prices.csv            Sample price data
data/sample_fundamentals.csv      Sample fundamental data
data/sample_macro.csv             Sample macro data
data/sample_metadata.csv          Sample ticker metadata
data/sample_portfolio_state.csv   Sample paper portfolio state
scripts/generate_sample_data.py   Sample data generator
src/stock_selector/               Core source code
tests/                            Unit tests
README.md                         Project overview (docs/USAGE.md = this guide)
docs/ROADMAP.md                   Long-term roadmap
```

## Output Files

Factor backtests write to `outputs/latest/` by default:

- `selections.csv`
- `risk_report.csv`
- `exposure_report.csv`
- `equity_curve.csv`
- `summary_report.md`

ML validation writes to `outputs/ml_latest/` by default:

- `ml_dataset.csv`
- `ml_predictions.csv`
- `feature_importance.csv`
- `ml_selections.csv`
- `ml_risk_report.csv`
- `ml_exposure_report.csv`
- `ml_equity_curve.csv`
- `ml_summary_report.md`

Paper trading writes to `outputs/paper_latest/` by default:

- `pre_trade_state.csv`
- `prices_used.csv`
- `orders.csv`
- `post_trade_state.csv`
- `paper_trade_report.md`

The append-only audit trail is under `outputs/paper_history/`; start with
`paper_journal.md` for the human-readable index.

Daily monitoring writes to `outputs/daily_latest/` by default:

- `daily_candidates.csv`
- `monitor_checks.csv`
- `feature_missing_report.csv`
- `feature_drift_report.csv`
- `daily_report.md`

Single-ticker analysis writes to `outputs/ticker_latest/` by default:

- `ticker_analysis.csv`
- `ticker_analysis.md`
- `external_snapshot.json`, only when `--include-external-snapshot` is used.

Real ticker analysis writes to `outputs/real_ticker/<TICKER>/` by default:

- `ticker_analysis.csv`
- `ticker_analysis.md`
- `analysis_result.json`
- `cache_metadata.json`
- `data_readiness.json`
- `data_readiness.md`
- `data_readiness.csv`
- `sec_fundamentals.json`, when SEC companyfacts is available.
- `external_snapshot.json`, unless `--no-snapshot` is used.

Successful snapshot fetches are also cached under `data/real_snapshots/` and used only to fill missing current snapshot fields in future single-ticker reports. Snapshot cache entries older than 14 days are ignored.

Each single-ticker run also updates signal review files:

- `outputs/signal_review/signal_history.csv`
- `outputs/signal_review/<TICKER>_signal_history.csv`
- `outputs/signal_review/signal_review_summary.csv`
- `outputs/signal_review/signal_review.md`

Signal review records the model's decision at the signal date and later fills in 5, 20, and 60 trading day close-to-close returns when enough future price data exists.

## Next Priorities

1. Expand SEC/FRED from readiness framework fields into validated scoring inputs.
2. Add dataset versioning for every external source.
3. Add sector-neutral, sentiment, event, and analyst revision factors.
4. Expand backtesting for halts, limit moves, dividends, delistings, and survivorship bias.
5. Add prediction explanations, drift charts, and alerts.
