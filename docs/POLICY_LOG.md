# 策略版本日志 / Policy Log

## English summary (added October 2026 for readers)

The Chinese log below is the original record, written at the time; this summary translates it. File paths are as of July 2026. After the September refactor, `src/stock_selector/analysis.py` became the package `src/stock_selector/analysis/`, and `candidates_universe.txt` became `universes/candidates.txt`.

- **Rule.** Every change to the paper portfolio's strategy is logged here. While an evaluation window is open, strategy logic (scores, entries, thresholds, universe) is frozen. Bug fixes are allowed at any time.
- **v1 (2026-07-08 → 07-09), discarded.** Ran for 2 trading days (equity $999.40) and was archived as statistically meaningless, because the scoring changed mid-test.
- **v2 (frozen 2026-07-09).** The paper portfolio was reset to $1,000. The pre-specified evaluation window was 20 trading days (to about 2026-08-06). Changes from v1:
  1. A confirmed volume breakout is executable even when extended above its trend baseline; only parabolic moves (beyond 2.5 × the horizon's chase limit) are refused.
  2. The high-probability score no longer double-counts six factors already inside the signal score. New weights: signal .32, confidence .14, data quality .10, entry readiness .10, backtest quality .09, backtest trust .09, horizon alignment .08, plan quality .08.
  3. When the chosen entry type has too few backtest trades, breakout and pullback evidence are pooled instead of failing the gate.
  4. Operations only: a single 150-ticker universe file, and a weekday 15:00 PT pipeline with a day-over-day journal.
- **v2 backtest baseline (2026-07-10).** 6,219 signal rows, 52.2 % 20-day win rate, sample-weighted expectation of 26.0 % a year. This is an optimistic yardstick: it includes survivorship bias and is not net of costs.
- **Freeze-period fixes (2026-07-10, no strategy change).** Stale-price handling with warnings; failures surfaced on the dashboard; a catch-up guard for the scheduler; an immutable audit trail of every run's inputs, prices, orders, costs and positions.
- **v2 closed (2026-09-26).** See the bilingual entry below. It includes the as-run differences found in the post-hoc audit: the calibration table was active, regime protection was off, a Labor Day rebalance happened, and run dates differ from price dates.
- **v3 backlog (logged, never implemented).** Volatility-normalised technical thresholds; remove the cross-sectionally constant market score from the ranking; re-estimate probabilities once more samples exist.

---

模拟盘的"模型策略"每次变更都要在这里登记。实验窗口(20 个交易日)内策略逻辑**冻结**:
bug 修复随时可以;打分/买点/门槛/股票池这类策略变更必须攒进下一版,窗口结束后统一发布。

## v2 前向测试结束 / Forward test closed — 2026-09-26

- **终点 / End point:** 最后一次成功运行是 2026-09-26 06:31 PT,使用 2026-09-25 收盘价;
  The last successful run was 2026-09-26 06:31 PT and marked the portfolio at the 2026-09-25 close.
  之后 9/28、9/29、9/30 三次运行都因电脑在运行中进入睡眠、网络中断而失败。没有补跑,实验在此结束。
  The runs on 9/28–9/30 failed because the machine went to sleep mid-run; they were not re-run, and the
  experiment was closed at that point. 分析见 / Analysis: [`results/REPORT.md`](../results/REPORT.md).
- **实际运行与上文记录的差异 / As-run differences from the notes below** (发现于结束后的复核 / found in the post-hoc audit):
  1. **概率校准表实际已启用。** 下文写"未启用",但 `real_data._resolve_probability_calibration_path` 在
     `latest/` 缺少该文件时会回退到最新的 `outputs/walk_forward/*/probability_calibration.csv`,
     即 v2 基准表(2026-07-10 16:54 生成,早于第一次 v2 调仓)。整个 v2 期间都用了它。
     *The calibration table was in effect for all of v2 via a fallback lookup, although the notes say it was not.*
  2. **市场状态保护实际未启用。** `paper.py` 只读取 `outputs/walk_forward/latest/market_regime_policy.csv`,
     该文件不存在,所以 v2 期间没有市场状态保护。
     *Market-regime protection was off for all of v2: the only path the paper engine reads did not exist.*
  3. **9/7(劳动节,休市)也运行并调仓了**,用的是 9/4 的收盘价。分析时把同一收盘价对应的两条记录合并,保留调仓后的那条。
     *A rebalance also ran on Labor Day at stale 9/4 prices; the analysis keeps the post-rebalance mark.*
  4. **运行日期 ≠ 价格日期。** 9/24 和 9/26 两次在开盘前运行,用的是前一交易日收盘价;
     分析按实际价格日期对齐(`paper_tracker.price_dates_for_runs`)。
     *Two early-morning runs used the previous close; the analysis aligns marks on the close actually used.*

## v2 — 2026-07-09 起(已结束 / closed 2026-09-26)

模拟盘于 2026-07-09 重置($1000 现金起步),v2 从 2026-07-10 的 13:20 PT 调仓开始计。
评估窗口:20 个交易日(约至 2026-08-06),期间策略冻结。

相对 v1 的变更(均在 `src/stock_selector/analysis.py`):

1. **突破买点可执行**(修复自相矛盾):`breakout` 风格下,放量确认突破即使高于趋势基线
   (extended)也判可执行;仅当涨幅超过 `max_chase_pct × BREAKOUT_EXTENSION_MULTIPLE(2.5)`
   的抛物线状态才拒绝。此前确认突破必被 extended 检查打回"等回调",强势股永远不可买。
2. **高概率分去重**:`_high_probability_score` 移除与 `signal_score` 重复的 6 个独立因子项
   (大盘/相对强弱/基本面/板块/分析师/估值)及 `fail_count×4` 的三重惩罚;新权重合计 1.0:
   信号 .32 / 置信 .14 / 数据质量 .10 / 买点就绪 .10 / 回测质量 .09 / 回测可信 .09 /
   周期一致 .08 / 计划质量 .08。
3. **回测样本合并**:被选买点类型样本 < 下限时,按笔数加权合并突破+回调两类回测作为证据
   (`_pooled_entry_backtest`),标签"综合买点(样本合并)";不再因单桶样本稀少一票否决。
4. **运营变更**(不影响策略):股票池统一为 `candidates_universe.txt`(150 只,一行一个);
   流水线在工作日 15:00 PT 运行并加 `--journal`;页面从同一次扫描刷新。

v1 模拟盘档案:`outputs/paper_archive/v1-ended-2026-07-09/`(仅 2 个交易日,无统计意义)。

**v2 回测基准(2026-07-10 生成,供 20 天闸门用)**:walk-forward 用 v2 代码在同一 150 只
候选池上重放(2y,步长 20 天),存档 `outputs/walk_forward/v2_baseline_2026-07-10/`,
仅 `walk_forward_summary.csv` + 报告发布至 `outputs/walk_forward/latest/`。
6,219 个信号样本,20 日胜率 52.2%,样本加权年化预期 **26.0%**(注意:含幸存者偏差、
未扣交易成本,是偏乐观的尺子;闸门只用它判断模拟盘是否严重跑输预期)。
`probability_calibration.csv` 与 `rule_calibration.csv` 均**未启用**(留在存档目录,
启用属于策略变更,归 v3)。latest/ 里 6 月 15 日的旧残留文件已清除。

**冻结期 bug/运营修复（2026-07-10，不改变策略）**:

- 人类 SELL/TRIM 按决策方向结算，保留原始股票涨跌供审计。
- 模型记分只使用实际写回模拟仓位的 paper BUY/SELL 订单，不再把全部扫描信号当做多。
- 记分板和到期复盘接入每日流水线；扫描缺失 ticker 会让流水线失败并在 dashboard 显示。
- 模拟盘缺当前价格时使用上一已知价并告警；连旧价也没有则明确失败，不再按零估值。
- dashboard 构建结果纳入最终运行状态；新鲜度以本次运行开始时间判断。
- launchd 增加每30分钟轻量补跑守卫，当天已运行时不会重复执行完整流水线。
- 模拟盘增加不可覆盖的完整审计流水：保存每次输入、价格、目标、订单、成本、交易前后
  仓位、现金、配置、代码版本、文件校验值和每日估值；失败运行也永久记录。

## v3 待办(冻结期内只登记、不实施)

- 技术分阈值按波动率(ATR)归一化,替代固定百分比(`trend_distance/0.10` 等)。
- `market_score` 移出高概率分(横截面上是常数,浪费权重),仅保留为环境门槛。
- 胜率校准样本不足问题:等 walk-forward/signal_review 样本积累后重估显示口径。

## v1 — 2026-07-08 至 2026-07-09(已归档)

原始逻辑:重复计权的高概率分、单桶回测样本硬门槛、突破风格下 extended 一票否决。
仅运行 2 个交易日,净值 $999.40,无统计意义。
