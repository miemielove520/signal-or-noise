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

## Human vs Model Scoreboard / 人机对照记分台

Log your own trade decisions and score them, on the same basis as the model, to see whose decisions are more disciplined over time. Wins are measured as **excess return vs a benchmark (QQQ/SPY)** — beating the market, not merely finishing positive — and summarized with expectancy, payoff ratio, profit factor, and drawdown rather than win rate alone.
记录你自己的交易决策，用和模型一样的口径打分，看长期下来谁更有纪律。“赢”按**相对基准（QQQ/SPY）的超额收益**衡量，即跑赢大盘，而不是只要正收益；用期望值、盈亏比、盈利因子和回撤汇总，而不是只看胜率。

```bash
python3 log_trade.py NVDA --action BUY --weight 12 --reason "breakout pullback"   # log a decision
python3 review_human.py            # fill 5/20/60-day forward and excess-vs-benchmark outcomes
python3 scoreboard_report.py       # write the side-by-side human-vs-model report
```

The scoreboard writes:

```text
outputs/signal_review/human_trades.csv
outputs/signal_review/model_trades.csv
outputs/signal_review/human_vs_model.md
outputs/signal_review/human_vs_model.csv
```

The model side uses only state-updating paper BUY/SELL orders from
`outputs/paper_latest/paper_trade_result.json`. Scan-only observations and
`avoid_for_now` rows are not scored as model trades. SELL/TRIM decisions are
direction-adjusted, while the underlying stock return is retained separately.

Validate the screening rules historically before any paper trading:

```bash
python3 validate.py AAPL MSFT NVDA NOW
```

For more realistic validation, use point-in-time universe membership instead of today's
surviving ticker list:

```bash
python3 validate.py --historical-universe-file data/universes/sample_historical_universe.csv --period 5y
```

Historical universe CSV schema:

```csv
ticker,start_date,end_date,delisted_date,delisting_return,status
AAPL,2010-01-01,,,,
OLDX,2010-01-01,2020-06-30,2020-06-30,-1.0,delisted
```

You can also provide complete `as_of_date` snapshots. In that mode, every snapshot date
is treated as a full constituent list until the next snapshot.

Validation reports print:

- `survivorship_bias_handled`: whether point-in-time membership was supplied.
- `contains_delisted_tickers`: whether the sample explicitly includes delisted tickers.
- `historical_universe_source`: the file used for membership.

If a delisted ticker lacks enough future prices during a validation holding window, the
validator uses `delisting_return` when supplied, otherwise the last available price.

Validation presets:

```bash
python3 validate.py --preset quick
python3 validate.py --preset standard
python3 validate.py --preset deep
python3 validate.py --preset extreme
```

- `quick`: smoke test on `research-core`.
- `standard`: 5-year research validation.
- `deep`: 10-year all-universe validation.
- `extreme`: 10-year all-universe validation with denser 10-trading-day sampling.

The validation output includes:

```text
segment_validation_summary.csv
market_regime_validation_summary.csv
market_regime_policy.csv
validation_coverage_plan.csv
validation_coverage_plan.md
```

`segment_validation_summary.csv` breaks results down by profile, horizon, entry type, and validation bucket. `market_regime_validation_summary.csv` breaks historical signals into bull uptrend, bear downtrend, sideways/mixed, high-volatility, and unknown regimes using benchmark data available at each historical signal date. `market_regime_policy.csv` converts those regime results into conservative protection actions such as normal rules, strict-only entries, tighter entries, or blocking new entries. `validation_coverage_plan.md` shows which stock pools need more sample coverage and gives standard and deep validation commands.

Safe config workflow:

```bash
python3 validate.py --preset deep
python3 validate.py --preset deep --screening-config outputs/walk_forward/preset_deep/suggested_screening.toml --output-dir outputs/walk_forward/candidate_config
python3 compare_runs.py outputs/walk_forward/preset_deep outputs/walk_forward/candidate_config
```

Only consider replacing `configs/screening.toml` if `compare_runs.py` says the candidate config passed review.

Default validation settings are research-oriented:

```text
period=5y
step_days=20
min_history_days=170
```

默认验证参数偏研究用途：使用5年数据、每20个交易日取一个历史信号、至少170天历史后才开始验证。

Use a validation preset when you do not want to choose period, step size, and universe manually:

```bash
python3 validate.py --preset quick
python3 validate.py --preset standard
python3 validate.py --preset deep
```

- `quick`: fast smoke test, lower sample confidence / 快速粗测，样本可信度较低。
- `standard`: research-grade default, stronger sample coverage / 研究级默认验证，样本覆盖更充分。
- `deep`: slower all-universe validation / 更慢的全部股票池深度验证。

Validate a built-in universe without typing every ticker:

```bash
python3 validate.py --universe software
python3 validate.py --universe semiconductors
python3 validate.py --universe mega-cap-tech
python3 validate.py --universe ai-infrastructure
python3 validate.py --universe cybersecurity
python3 validate.py --universe industrial-quality
python3 validate.py --universe research-core
```

Validate every built-in universe and compare them in one report:

```bash
python3 validate.py --all-universes
```

Run a faster first pass when you only want a quick diagnosis:

```bash
python3 validate.py --all-universes --period 1y --step-days 80 --min-history-days 120
```

The validation writes:

```text
outputs/walk_forward/latest/walk_forward_events.csv
outputs/walk_forward/latest/walk_forward_summary.csv
outputs/walk_forward/latest/ticker_validation_ranking.csv
outputs/walk_forward/latest/sample_sufficiency_guidance.csv
outputs/walk_forward/latest/profile_validation_summary.csv
outputs/walk_forward/latest/profile_health_dashboard.csv
outputs/walk_forward/latest/profile_health_dashboard.md
outputs/walk_forward/latest/profile_action_recommendations.csv
outputs/walk_forward/latest/profile_action_recommendations.md
outputs/walk_forward/latest/profile_blocker_dashboard.csv
outputs/walk_forward/latest/profile_blocker_dashboard.md
outputs/walk_forward/latest/profile_rule_calibration.csv
outputs/walk_forward/latest/suggested_screening.toml
outputs/walk_forward/latest/rule_calibration.csv
outputs/walk_forward/latest/walk_forward_report.md
outputs/walk_forward/latest/validation_result.json
outputs/walk_forward/latest/run_manifest.json
```

Compare two validation runs:

```bash
python3 compare_runs.py outputs/walk_forward/previous outputs/walk_forward/latest
```

The comparison writes:

```text
run_comparison_summary.csv
run_comparison_tickers.csv
config_adoption_decision.csv
run_comparison.md
run_comparison.json
```

`config_adoption_decision.csv` is a safety gate for candidate configs. It checks whether the candidate run keeps enough samples, avoids worse 20-day win rate and average return, and does not increase data warnings or missing prices. Only consider replacing `configs/screening.toml` when the final decision is `candidate_config_passed_review`.
`config_adoption_decision.csv` 是候选配置安全门槛。它会检查候选运行是否保留足够样本、20日胜率和平均收益没有变差、数据警告和缺失价格没有增加。只有最终结论是 `candidate_config_passed_review` 时，才考虑替换 `configs/screening.toml`。

The all-universe validation writes:

```text
outputs/walk_forward/all_universes/all_universe_validation_summary.csv
outputs/walk_forward/all_universes/all_universe_validation_report.md
outputs/walk_forward/all_universes/all_universe_validation_result.json
outputs/walk_forward/all_universes/all_universe_suggested_screening.toml
```

The all-universe report includes a primary diagnostic, optimization priority, top quality-gate blockers, suggested threshold changes, and bilingual recommendations for each universe.
全股票池报告会给出主要诊断、优化优先级、最常见质量门槛卡点、建议阈值调整，以及每个股票池的中英文改进建议。
Suggested threshold changes are review notes only; they do not automatically replace `configs/screening.toml`.
建议阈值调整只是复盘建议，不会自动替换 `configs/screening.toml`。
The generated `all_universe_suggested_screening.toml` is an experimental config for follow-up validation.
生成的 `all_universe_suggested_screening.toml` 是实验配置，用于下一轮验证。

Validate a suggested experiment config:

```bash
python3 validate.py --all-universes --universe semiconductors --screening-config outputs/walk_forward/all_universes/all_universe_suggested_screening.toml
```

The command prints progress as each universe starts and completes, and updates the aggregate files after every universe.
命令会在每个股票池开始和完成时打印进度，并在每个股票池完成后更新总表文件。

`profile_validation_summary.csv` splits validation results by screening profile, such as `ai_infrastructure`, `saas_software`, `cybersecurity`, `fintech_high_beta`, `defensive_quality`, `industrial_quality`, `semiconductor`, and `mega_cap_tech`.
`profile_rule_calibration.csv` suggests whether each profile's key thresholds should be tightened, loosened, or kept based on walk-forward samples.
Each profile rule suggestion includes a confidence level: `high`, `medium`, `low`, or `insufficient`.
`suggested_screening.toml` applies only medium/high-confidence, sample-supported threshold suggestions into a reviewable config file; inspect it manually before replacing `configs/screening.toml`.

Safe config review flow:

```bash
python3 validate.py --preset standard --output-dir outputs/walk_forward/current_config
python3 validate.py --preset standard --screening-config outputs/walk_forward/current_config/suggested_screening.toml --output-dir outputs/walk_forward/suggested_config
python3 compare_runs.py outputs/walk_forward/current_config outputs/walk_forward/suggested_config
```

安全配置审核流程：

```text
先验证当前配置
再用 suggested_screening.toml 验证候选配置
最后用 compare_runs.py 查看是否通过 config_adoption_decision
```

Run the factor backtest:

```bash
PYTHONPATH=src python3 -m stock_selector.cli run --config configs/default.toml
```

Run tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
.venv/bin/python -m pytest
```

For launchd automation on macOS, grant Full Disk Access to `/bin/bash` under
System Settings > Privacy & Security > Full Disk Access. Without this permission,
background jobs cannot read a project stored under `~/Documents` and launchd exits
with `EX_CONFIG`/`Operation not permitted`.

Audit the configured price data:

```bash
PYTHONPATH=src python3 -m stock_selector.cli audit --config configs/default.toml
```

Run rolling ML validation and ML-ranked backtest:

```bash
PYTHONPATH=src python3 -m stock_selector.cli ml-run --config configs/default.toml
```

Generate local paper-trading rebalance orders:

```bash
PYTHONPATH=src python3 -m stock_selector.cli paper-trade --config configs/default.toml
```

Generate paper-trading orders from ML targets:

```bash
PYTHONPATH=src python3 -m stock_selector.cli paper-trade --config configs/default.toml --mode ml
```

Generate the daily candidate and monitoring report:

```bash
PYTHONPATH=src python3 -m stock_selector.cli daily-report --config configs/default.toml
```

Generate the ML daily candidate and monitoring report:

```bash
PYTHONPATH=src python3 -m stock_selector.cli daily-report --config configs/default.toml --mode ml
```

Analyze one ticker across short, medium, and long horizons:

```bash
PYTHONPATH=src python3 -m stock_selector.cli real AAPL
```

Save a separate current external snapshot:

```bash
PYTHONPATH=src python3 -m stock_selector.cli real AAPL
```

The generated `external_snapshot.json` is a current snapshot only. It is not used in historical backtests.

Analyze sample data from `configs/default.toml`:

```bash
PYTHONPATH=src python3 -m stock_selector.cli analyze-ticker \
  --config configs/default.toml \
  --ticker ALFA \
  --horizon all
```

Single-ticker analysis focuses on analysis, screening, entry, and risk. It no longer assumes a fixed capital amount or simulates a share count.
单股分析聚焦于分析、筛选、买点和风险，不再假设固定本金或模拟买多少股。

Download real prices for research prototyping:

```bash
PYTHONPATH=src python3 -m stock_selector.cli download-yfinance \
  --tickers AAPL MSFT NVDA \
  --start 2020-01-01 \
  --output data/us_prices.csv
```

Regenerate sample data:

```bash
python3 scripts/generate_sample_data.py
```

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
