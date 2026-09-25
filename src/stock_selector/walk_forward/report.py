"""Markdown rendering of the walk-forward validation report."""

from __future__ import annotations

import pandas as pd

from ..universe import survivorship_report_without_historical_membership

from ._common import (
    _format_metric_frame,
    _markdown_table,
)
from .config import (
    DEFAULT_FORWARD_WINDOWS,
)


def render_walk_forward_report(
    events: pd.DataFrame,
    summary: pd.DataFrame,
    calibration: pd.DataFrame,
    ticker_ranking: pd.DataFrame | None = None,
    sample_sufficiency: pd.DataFrame | None = None,
    profile_summary: pd.DataFrame | None = None,
    segment_summary: pd.DataFrame | None = None,
    market_regime_summary: pd.DataFrame | None = None,
    market_regime_policy: pd.DataFrame | None = None,
    profile_calibration: pd.DataFrame | None = None,
    probability_calibration: pd.DataFrame | None = None,
    portfolio_summary: pd.DataFrame | None = None,
    portfolio_rebalances: pd.DataFrame | None = None,
    portfolio_equity_summary: pd.DataFrame | None = None,
    portfolio_equity_curve: pd.DataFrame | None = None,
    benchmark_summary: pd.DataFrame | None = None,
    benchmark_curve: pd.DataFrame | None = None,
    benchmark_policy: pd.DataFrame | None = None,
    benchmark_tightening: pd.DataFrame | None = None,
    tightening_impact: pd.DataFrame | None = None,
    threshold_sensitivity: pd.DataFrame | None = None,
    minimum_sample_guard: pd.DataFrame | None = None,
    forward_windows: tuple[int, ...] = DEFAULT_FORWARD_WINDOWS,
    survivorship_bias_report: dict[str, object] | None = None,
) -> str:
    survivorship = survivorship_bias_report or survivorship_report_without_historical_membership()
    lines = [
        "# Walk-Forward Validation / 滚动历史验证",
        "",
        "This report replays historical signal dates using only price data available up to each signal date.",
        "本报告按历史信号日期回放，只使用每个信号日以前可见的价格数据。",
        "",
        "Validation mode uses neutral historical-safe market, sector, fundamental, and event contexts.",
        "验证模式使用中性、历史安全的大盘、板块、基本面和事件上下文，避免把当前快照回填到历史。",
        "",
        f"- Event rows / 信号样本行数: `{len(events)}`",
        f"- Forward windows / 未来观察窗口: `{', '.join(str(window) + 'd' for window in forward_windows)}`",
        f"- survivorship_bias_handled: `{str(bool(survivorship.get('survivorship_bias_handled'))).lower()}`",
        f"- contains_delisted_tickers / 样本包含退市股票: `{str(bool(survivorship.get('contains_delisted_tickers'))).lower()}`",
        f"- historical_universe_source / 历史股票池来源: `{survivorship.get('source', 'none')}`",
        "",
        "## Survivorship Bias / 幸存者偏差",
        "",
        (
            "This section states whether the validation used point-in-time universe "
            "membership instead of today's surviving ticker list."
        ),
        "本区块说明验证是否使用了历史时点成分股，而不是今天仍然存在的股票列表。",
        "",
        f"- survivorship_bias_handled: `{str(bool(survivorship.get('survivorship_bias_handled'))).lower()}`",
        f"- point_in_time_universe: `{str(bool(survivorship.get('point_in_time_universe'))).lower()}`",
        f"- contains_delisted_tickers: `{str(bool(survivorship.get('contains_delisted_tickers'))).lower()}`",
        f"- historical_constituent_count: `{int(survivorship.get('historical_constituent_count') or 0)}`",
        f"- delisted_ticker_count: `{int(survivorship.get('delisted_ticker_count') or 0)}`",
        f"- source: `{survivorship.get('source', 'none')}`",
        "",
        *[f"- warning: {warning}" for warning in survivorship.get("warnings", [])],
        *[f"- 警告: {warning}" for warning in survivorship.get("warnings_zh", [])],
        "",
        "## Performance Summary / 表现摘要",
        "",
    ]
    lines.extend(_markdown_table(_format_metric_frame(summary)))
    lines.extend(
        [
            "",
            "## Ticker Validation Ranking / 个股验证排名",
            "",
            (
                "This section ranks tickers inside the validated universe using "
                "historical signal quality, win rate, average return, and drawdown."
            ),
            "本区块按历史信号质量、胜率、平均收益和回撤，对股票池内部 ticker 排名。",
            "",
        ]
    )
    ticker_ranking_frame = ticker_ranking if ticker_ranking is not None else pd.DataFrame()
    if not ticker_ranking_frame.empty:
        lines.extend(_markdown_table(_format_metric_frame(ticker_ranking_frame.head(25))))
    else:
        lines.extend(_markdown_table(ticker_ranking_frame))
    lines.extend(
        [
            "",
            "## Sample Sufficiency Guidance / 样本充分性建议",
            "",
            (
                "This section explains whether the validation has enough samples and "
                "which parameters to adjust when the ranking is too thin."
            ),
            "本区块说明验证样本是否足够；当排名样本太少时，给出下一次验证应调整的参数。",
            "",
        ]
    )
    sample_sufficiency_frame = (
        sample_sufficiency if sample_sufficiency is not None else pd.DataFrame()
    )
    if not sample_sufficiency_frame.empty:
        lines.extend(_markdown_table(_format_metric_frame(sample_sufficiency_frame.head(30))))
    else:
        lines.extend(_markdown_table(sample_sufficiency_frame))
    lines.extend(
        [
            "",
            "## Profile Summary / 分类规则表现",
            "",
        ]
    )
    profile_frame = profile_summary if profile_summary is not None else pd.DataFrame()
    lines.extend(_markdown_table(_format_metric_frame(profile_frame)))
    lines.extend(
        [
            "",
            "## Segment Validation Summary / 分层验证表现",
            "",
            (
                "This section breaks validation down by profile, horizon, entry type, "
                "and validation bucket so profile-specific rules are not judged only by averages."
            ),
            "本区块按规则大类、周期、买点类型和验证结果拆分表现，避免只看总体平均值。",
            "",
        ]
    )
    segment_frame = segment_summary if segment_summary is not None else pd.DataFrame()
    if not segment_frame.empty:
        lines.extend(_markdown_table(_format_metric_frame(segment_frame.head(40))))
    else:
        lines.extend(_markdown_table(segment_frame))
    lines.extend(
        [
            "",
            "## Market Regime Validation / 市场状态验证",
            "",
            (
                "This section splits historical signals by market regime using benchmark "
                "data available at each historical signal date."
            ),
            "本区块用每个历史信号日当时可见的基准数据，把信号拆成牛市、熊市、震荡和高波动环境。",
            "",
        ]
    )
    regime_frame = market_regime_summary if market_regime_summary is not None else pd.DataFrame()
    lines.extend(_markdown_table(_format_metric_frame(regime_frame)))
    lines.extend(
        [
            "",
            "## Market Regime Protection Policy / 市场状态保护规则",
            "",
            (
                "This section converts market-regime validation into conservative "
                "entry protection rules for bear and high-volatility regimes."
            ),
            "本区块把市场状态验证结果转换成保护规则，尤其针对熊市和高波动环境。",
            "",
        ]
    )
    regime_policy_frame = market_regime_policy if market_regime_policy is not None else pd.DataFrame()
    lines.extend(_markdown_table(_format_metric_frame(regime_policy_frame)))
    lines.extend(
        [
            "",
            "## Probability Calibration / 概率校准",
            "",
            (
                "This section compares estimated win probability against actual "
                "walk-forward win rate by probability bucket."
            ),
            "本区块按概率分组，对比模型估计胜率与滚动验证中的实际胜率。",
            "",
        ]
    )
    probability_frame = (
        probability_calibration if probability_calibration is not None else pd.DataFrame()
    )
    lines.extend(_markdown_table(_format_metric_frame(probability_frame)))
    lines.extend(
        [
            "",
            "## Portfolio Validation / 组合验证",
            "",
            (
                "This section forms equal-weight baskets from the top-ranked candidates "
                "on each historical signal date."
            ),
            "本区块在每个历史信号日从排名最高的候选中构建等权组合。",
            "",
        ]
    )
    portfolio_frame = portfolio_summary if portfolio_summary is not None else pd.DataFrame()
    lines.extend(_markdown_table(_format_metric_frame(portfolio_frame)))
    lines.extend(
        [
            "",
            "## Portfolio Equity Curve / 组合逐日净值曲线",
            "",
            (
                "This section simulates daily equal-weight portfolio returns from "
                "historical rebalance selections, including transaction-cost drag."
            ),
            "本区块根据历史调仓选择模拟逐日等权组合收益，并计入交易成本拖累。",
            "",
        ]
    )
    equity_summary_frame = (
        portfolio_equity_summary if portfolio_equity_summary is not None else pd.DataFrame()
    )
    lines.extend(_markdown_table(_format_metric_frame(equity_summary_frame)))
    lines.extend(
        [
            "",
            "## Portfolio Equity Curve Sample / 组合净值曲线样本",
            "",
        ]
    )
    equity_curve_frame = portfolio_equity_curve if portfolio_equity_curve is not None else pd.DataFrame()
    if not equity_curve_frame.empty:
        lines.extend(_markdown_table(_format_metric_frame(equity_curve_frame.head(25))))
    else:
        lines.extend(_markdown_table(equity_curve_frame))
    lines.extend(
        [
            "",
            "## Benchmark Comparison / 基准对比",
            "",
            (
                "This section compares portfolio equity against SPY and QQQ "
                "on the same daily dates."
            ),
            "本区块在相同每日日期上，把组合净值与SPY和QQQ进行对比。",
            "",
        ]
    )
    benchmark_summary_frame = benchmark_summary if benchmark_summary is not None else pd.DataFrame()
    lines.extend(_markdown_table(_format_metric_frame(benchmark_summary_frame)))
    lines.extend(
        [
            "",
            "## Benchmark-Aware Rule Policy / 基准感知规则建议",
            "",
            (
                "This section converts benchmark comparison into a threshold policy: "
                "tighten, keep, or selectively relax."
            ),
            "本区块把基准对比转换成阈值策略：收紧、维持或谨慎放宽。",
            "",
        ]
    )
    benchmark_policy_frame = benchmark_policy if benchmark_policy is not None else pd.DataFrame()
    lines.extend(_markdown_table(_format_metric_frame(benchmark_policy_frame)))
    lines.extend(
        [
            "",
            "## Specific Tightening Recommendations / 具体收紧建议",
            "",
            (
                "This section lists concrete threshold changes when benchmark-aware "
                "policy says the model should tighten."
            ),
            "当基准感知策略要求收紧时，本区块列出具体应该提高的门槛。",
            "",
        ]
    )
    benchmark_tightening_frame = benchmark_tightening if benchmark_tightening is not None else pd.DataFrame()
    lines.extend(_markdown_table(_format_metric_frame(benchmark_tightening_frame)))
    lines.extend(
        [
            "",
            "## Tightening Impact Validation / 收紧效果验证",
            "",
            (
                "This section compares validation metrics before and after applying "
                "medium/high priority tightening recommendations."
            ),
            "本区块对比应用中高优先级收紧建议前后的验证指标。",
            "",
        ]
    )
    tightening_impact_frame = tightening_impact if tightening_impact is not None else pd.DataFrame()
    lines.extend(_markdown_table(_format_metric_frame(tightening_impact_frame)))
    lines.extend(
        [
            "",
            "## Threshold Sensitivity Grid / 阈值敏感度网格",
            "",
            (
                "This section tests multiple threshold values and threshold pairs "
                "to find filters that actually improve validation metrics."
            ),
            "本区块测试多个门槛值和门槛组合，寻找真正改善验证指标的过滤条件。",
            "",
        ]
    )
    threshold_sensitivity_frame = threshold_sensitivity if threshold_sensitivity is not None else pd.DataFrame()
    if not threshold_sensitivity_frame.empty:
        lines.extend(_markdown_table(_format_metric_frame(threshold_sensitivity_frame.head(30))))
    else:
        lines.extend(_markdown_table(threshold_sensitivity_frame))
    lines.extend(
        [
            "",
            "## Minimum Sample Guard / 最小样本保护",
            "",
            (
                "This section blocks threshold changes that leave too few validation "
                "samples or do not show clear improvement."
            ),
            "本区块阻止样本太少或改善不明确的门槛被自动采用。",
            "",
        ]
    )
    guard_frame = minimum_sample_guard if minimum_sample_guard is not None else pd.DataFrame()
    if not guard_frame.empty:
        lines.extend(_markdown_table(_format_metric_frame(guard_frame.head(30))))
    else:
        lines.extend(_markdown_table(guard_frame))
    lines.extend(
        [
            "",
            "## Benchmark Comparison Sample / 基准对比样本",
            "",
        ]
    )
    benchmark_curve_frame = benchmark_curve if benchmark_curve is not None else pd.DataFrame()
    if not benchmark_curve_frame.empty:
        lines.extend(_markdown_table(_format_metric_frame(benchmark_curve_frame.head(25))))
    else:
        lines.extend(_markdown_table(benchmark_curve_frame))
    lines.extend(
        [
            "",
            "## Portfolio Rebalances / 组合历史调仓",
            "",
        ]
    )
    rebalance_frame = portfolio_rebalances if portfolio_rebalances is not None else pd.DataFrame()
    if not rebalance_frame.empty:
        lines.extend(_markdown_table(_format_metric_frame(rebalance_frame.head(25))))
    else:
        lines.extend(_markdown_table(rebalance_frame))
    lines.extend(
        [
            "",
            "## Profile Rule Calibration / 分类规则阈值建议",
            "",
        ]
    )
    profile_calibration_frame = (
        profile_calibration if profile_calibration is not None else pd.DataFrame()
    )
    lines.extend(_markdown_table(_format_metric_frame(profile_calibration_frame)))
    lines.extend(
        [
            "",
            "## Rule Calibration / 规则校准",
            "",
        ]
    )
    lines.extend(_markdown_table(_format_metric_frame(calibration)))
    return "\n".join(lines).rstrip() + "\n"
