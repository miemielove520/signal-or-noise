"""Signal or noise? The statistical post-mortem of the frozen v2 model.

Reads only ``results/data/`` (see ``scripts/export_results.py``) and writes
``results/summary.json`` plus the figures in ``results/figures/``.

    python research/forward_test.py

Sections (same order as results/REPORT.md):
  1. Forward test: the paper portfolio vs QQQ and vs its own universe, with
     robustness checks, the pre-specified 20-session checkpoint and a market-beta fit.
  2. Backtest-implied predictive check: how unusual is the result if the backtest were right?
  3. Implementation audit: holding periods, turnover, and a counterfactual that holds
     the same entries for the backtest's 20 sessions.
  4. Ranking skill in the backtest: rank IC, top-5 spread, multiple testing.
  5. Probability calibration: Brier skill, Murphy decomposition, date-clustered intervals.
  6. Sample-size audit: duplicated rows, within-date correlation, effective sample size.
  7. Power: how much data a forward test needs.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stock_selector.paper_tracker import price_dates_for_runs  # noqa: E402
from stock_selector.stats_tests import (  # noqa: E402
    bonferroni_t_threshold,
    bootstrap_statistic,
    brier_decomposition,
    brier_skill,
    cluster_bootstrap,
    compounded_return,
    duplication_factor,
    information_coefficients,
    intraclass_correlation,
    newey_west_t,
    one_sample_mean_test,
    reliability_table,
    required_sample_size,
    simulate_window_returns,
)

DATA = ROOT / "results" / "data"
FIGURES = ROOT / "results" / "figures"
SEED = 20260709  # paper portfolio start date
EXPERIMENT_END = pd.Timestamp("2026-09-25")  # last close used by the frozen v2 forward test
PRESPECIFIED_SESSIONS = 20  # docs/POLICY_LOG.md: v2 evaluation window of 20 trading days
BACKTEST_HOLD_SESSIONS = 20  # the walk-forward measures 20-session forward returns
N_RESAMPLES = 10_000
N_CLUSTER_RESAMPLES = 4_000
MEAN_BLOCK_DAYS = 5.0
BLOCK_SENSITIVITY = (1.0, 2.0, 5.0, 10.0, 20.0)
BACKTEST_PORTFOLIO = "top_high_probability_score"  # closest backtest analogue of the paper rules
# Walk-forward runs saved to disk during development (many were code smoke tests,
# not strategy variants), so the true number of independent "looks" is uncertain.
SAVED_WALK_FORWARD_RUNS = 36
CALIBRATION_BINS = [0, 0.5, 0.55, 0.6, 0.65, 1.0]

# ---- chart style (reference palette, light surface) ------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
GRID_BAR = "#a8a7a1"  # neutral bar for the in-sample period
GRID_BAR_LIGHT = "#d4d3ce"
MODEL = "#2a78d6"  # slot 1 blue
QQQ = "#eb6834"  # slot 2 orange
UNIVERSE = "#1baf7a"  # slot 3 aqua
NEG = "#e34948"  # diverging negative pole
POS = MODEL  # diverging positive pole


def _style(ax, title: str, subtitle: str = "") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold", pad=22 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=INK_2, va="bottom")


def _figure(width=8.0, height=4.2):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def _save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, facecolor=SURFACE)
    print(f"wrote results/figures/{name}")


def _month_axis(ax, fmt: str = "%b %d") -> None:
    import matplotlib.dates as mdates

    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))


def _label_line_ends(ax, ends: list[tuple[pd.Timestamp, float]], min_gap_frac: float = 0.06) -> None:
    """Label each line's final value, nudging labels apart so they never overlap."""
    low, high = ax.get_ylim()
    gap = (high - low) * min_gap_frac
    previous = None
    for x, value in sorted(ends, key=lambda item: item[1]):
        y = value if previous is None else max(value, previous + gap)
        previous = y
        ax.text(x + pd.Timedelta(days=1), y, f"{value:+.1f}%", ha="left", va="center", fontsize=9, color=INK)


def _round(value, digits: int = 8):
    """Round floats to ``digits`` significant digits so summary.json diffs are stable."""
    if isinstance(value, dict):
        return {k: _round(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round(v, digits) for v in value]
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if not math.isfinite(value) or value == 0:
            return value
        return float(f"{value:.{digits}g}")
    if isinstance(value, np.integer):
        return int(value)
    return value


# ---------------------------------------------------------------------------
# 1. Forward test
# ---------------------------------------------------------------------------


def _load_forward() -> dict:
    px = pd.read_csv(DATA / "forward_window_prices.csv", parse_dates=["date"]).set_index("date")
    runs = pd.read_csv(DATA / "paper_equity_history.csv")
    # Runs are stamped with the day they ran, but early-morning and holiday runs marked
    # the portfolio at the previous session's close. Align everything on that close.
    runs["price_date"] = price_dates_for_runs(runs["executed_at"], px["QQQ"].dropna().index)
    eq = runs.drop_duplicates("price_date", keep="last").set_index("price_date").sort_index()
    eq = eq[eq.index <= EXPERIMENT_END]
    window = px.loc[eq.index[0] : eq.index[-1]].ffill()
    return {"px": px, "runs": runs, "eq": eq, "window": window}


def forward_test(loaded: dict) -> dict:
    px, runs, eq, window = loaded["px"], loaded["runs"], loaded["eq"], loaded["window"]
    start, end = eq.index[0], eq.index[-1]
    universe_cols = [c for c in window.columns if c not in ("QQQ", "SPY") and window[c].notna().all()]
    unpriced = [c for c in px.columns if c not in ("QQQ", "SPY") and px[c].notna().sum() == 0]

    universe_curve = (window[universe_cols] / window[universe_cols].iloc[0]).mean(axis=1)
    model_curve = eq["equity"] / eq["equity"].iloc[0]
    # The pipeline's stored benchmark_close has gaps on a few days; use the
    # downloaded QQQ closes on the same price dates instead.
    qqq_px = window["QQQ"].reindex(eq.index)
    qqq_curve = qqq_px / qqq_px.iloc[0]

    model_daily = model_curve.pct_change().dropna()
    qqq_daily = qqq_curve.pct_change().dropna()
    excess_daily = (model_daily - qqq_daily).dropna()

    boot = bootstrap_statistic(
        excess_daily, np.mean, np.random.default_rng(SEED), N_RESAMPLES, MEAN_BLOCK_DAYS
    )
    block_sensitivity = {}
    for block in BLOCK_SENSITIVITY:
        res = bootstrap_statistic(excess_daily, np.mean, np.random.default_rng(SEED), N_RESAMPLES, block)
        block_sensitivity[f"{block:g}"] = [res.ci_low, res.ci_high]
    iid = one_sample_mean_test(excess_daily)

    beta, alpha, r_value, _, _ = stats.linregress(qqq_daily.to_numpy(), model_daily.to_numpy())

    # Pre-specified checkpoint: the close PRESPECIFIED_SESSIONS sessions after the start.
    gate_date = window.index[PRESPECIFIED_SESSIONS]
    gate_mark = eq.index[eq.index <= gate_date][-1]

    sd = float(excess_daily.std(ddof=1))
    modest_edge = 0.0005  # +0.05 % per day, about +13 % a year
    sessions_per_year = 252

    fig, ax = _figure()
    ends = []
    for curve, color, label in (
        (model_curve, MODEL, "Model paper portfolio ($1,000 start)"),
        (qqq_curve, QQQ, "QQQ (Nasdaq-100 fund)"),
        (universe_curve, UNIVERSE, f"The model's own {len(universe_cols)} stocks, equal weight"),
    ):
        y = (curve - 1) * 100
        ax.plot(y.index, y.values, color=color, linewidth=2, label=label)
        ends.append((y.index[-1], float(y.iloc[-1])))
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.axvline(gate_date, color=INK_2, linewidth=1, linestyle=":")
    ax.text(
        gate_date + pd.Timedelta(days=1),
        ax.get_ylim()[1] * 0.92,
        "pre-specified\n20-session check",
        fontsize=8,
        color=INK_2,
        va="top",
    )
    _label_line_ends(ax, ends)
    ax.set_ylabel("Return since start (%)", color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=INK)
    _month_axis(ax)
    _style(
        ax,
        "Forward test: the model lost while its own stock list gained",
        f"Rules frozen; simulated $1,000 portfolio marked at market closes, {start:%b %d} – {end:%b %d, %Y}",
    )
    ax.margins(x=0.08)
    _save(fig, "forward_equity.png")

    return {
        "start": str(start.date()),
        "end": str(end.date()),
        "runs_recorded": int(len(runs[runs["price_date"] <= EXPERIMENT_END])),
        "runs_remapped_to_previous_close": int(
            (runs["price_date"].dt.strftime("%Y-%m-%d") != runs["date"]).sum()
        ),
        "holiday_reruns_merged": int(runs["price_date"].duplicated(keep="last").sum()),
        # Market sessions between the first and last close, and how many of them have a mark.
        "trading_days": int(len(window.index) - 1),
        "daily_marks": int(len(model_daily)),
        "mark_intervals_spanning_several_sessions": int(
            (pd.Series(window.index.get_indexer(eq.index)).diff().dropna() > 1).sum()
        ),
        "start_equity": float(eq["equity"].iloc[0]),
        "end_equity": float(eq["equity"].iloc[-1]),
        "model_return": float(model_curve.iloc[-1] - 1),
        "qqq_return": float(qqq_curve.iloc[-1] - 1),
        "relative_excess_vs_qqq": float(model_curve.iloc[-1] / qqq_curve.iloc[-1] - 1),
        "universe_equal_weight_return": float(universe_curve.iloc[-1] - 1),
        "universe_size_listed": int(len(universe_cols) + len(unpriced)),
        "universe_size_priced": len(universe_cols),
        "universe_unpriced_tickers": unpriced,
        "max_drawdown": float((model_curve / model_curve.cummax() - 1).min()),
        "prespecified_check": {
            "sessions": PRESPECIFIED_SESSIONS,
            "date": str(gate_date.date()),
            "model_return": float(model_curve.loc[gate_mark] - 1),
            "qqq_return": float(qqq_curve.loc[gate_mark] - 1),
        },
        "mean_daily_excess_vs_qqq": boot.estimate,
        "sd_daily_excess_vs_qqq": sd,
        "mean_daily_excess_ci95": [boot.ci_low, boot.ci_high],
        "share_of_resamples_with_mean_excess_above_zero": 1 - boot.p_value_le_zero,
        "ci95_by_mean_block_length": block_sensitivity,
        "iid_t_stat": iid.t_stat,
        "iid_p_value": iid.p_value_two_sided,
        "newey_west_t_stat_5_lags": newey_west_t(excess_daily, 5),
        "lag1_autocorrelation_of_excess": float(excess_daily.autocorr(lag=1)),
        "beta_vs_qqq": float(beta),
        "alpha_per_mark": float(alpha),
        "r_squared_vs_qqq": float(r_value**2),
        "power": {
            "observations_to_detect_observed_effect": required_sample_size(boot.estimate, sd),
            "modest_edge_per_day": modest_edge,
            "observations_to_detect_modest_edge": required_sample_size(modest_edge, sd),
            "years_to_detect_modest_edge": required_sample_size(modest_edge, sd) / sessions_per_year,
        },
    }


# ---------------------------------------------------------------------------
# 2. Backtest-implied predictive check
# ---------------------------------------------------------------------------


def luck_test(forward: dict) -> dict:
    curves = pd.read_csv(DATA / "backtest_portfolio_vs_benchmark.csv", parse_dates=["date"])
    variants = {}
    for name, g in curves[curves["benchmark_ticker"] == "QQQ"].groupby("portfolio_name"):
        variants[name] = {
            "start": str(g["date"].min().date()),
            "end": str(g["date"].max().date()),
            "total_return": compounded_return(g["portfolio_daily_return"].to_numpy()),
            "qqq_return": compounded_return(g["benchmark_daily_return"].to_numpy()),
        }
    bt = curves[(curves["portfolio_name"] == BACKTEST_PORTFOLIO) & (curves["benchmark_ticker"] == "QQQ")]
    days = forward["trading_days"]
    observed_excess = forward["relative_excess_vs_qqq"]

    def tail_probability(block: float) -> tuple[float, np.ndarray]:
        rng = np.random.default_rng(SEED + 1)
        port = simulate_window_returns(bt["portfolio_daily_return"], days, rng, N_RESAMPLES, block)
        rng = np.random.default_rng(SEED + 1)  # same blocks for the benchmark leg
        bench = simulate_window_returns(bt["benchmark_daily_return"], days, rng, N_RESAMPLES, block)
        sim = (1 + port) / (1 + bench) - 1
        return float(np.mean(sim <= observed_excess)), sim

    p_as_bad, sim_excess = tail_probability(MEAN_BLOCK_DAYS)
    sensitivity = {f"{b:g}": tail_probability(b)[0] for b in BLOCK_SENSITIVITY}
    backtest_excess = (bt["portfolio_daily_return"] - bt["benchmark_daily_return"]).to_numpy()
    backtest_test = one_sample_mean_test(backtest_excess)

    fig, ax = _figure()
    values = np.clip(sim_excess * 100, -60, 110)
    bins = np.linspace(-60, 110, 69)
    counts, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    colors = [NEG if c <= observed_excess * 100 else MODEL for c in centers]
    ax.bar(centers, counts, width=np.diff(edges) * 0.9, color=colors)
    ax.axvline(observed_excess * 100, color=INK, linewidth=1.5)
    ax.text(
        observed_excess * 100 - 1.5,
        counts.max() * 0.97,
        f"live test\n{observed_excess:+.1%}",
        color=INK,
        fontsize=9,
        va="top",
        ha="right",
    )
    ax.text(
        observed_excess * 100 - 1.5,
        counts.max() * 0.6,
        f"{p_as_bad:.0%} of\nsimulations are\nthis bad or worse",
        color=NEG,
        fontsize=8.5,
        va="top",
        ha="right",
    )
    ax.axvline(0, color=INK_2, linewidth=0.8)
    ax.set_xlim(-60, 110)
    ax.set_xlabel(f"Model minus QQQ over {days} trading sessions (%)", color=INK_2, fontsize=9)
    ax.set_ylabel("Simulated periods", color=INK_2, fontsize=9)
    _style(
        ax,
        f"If the backtest were right, this result would occur about {p_as_bad:.0%} of the time",
        f"{N_RESAMPLES:,} periods re-sampled from the backtest's own daily returns (stationary block bootstrap)",
    )
    _save(fig, "backtest_vs_forward.png")

    return {
        "backtest_portfolio": BACKTEST_PORTFOLIO,
        "backtest_start": variants[BACKTEST_PORTFOLIO]["start"],
        "backtest_end": variants[BACKTEST_PORTFOLIO]["end"],
        "backtest_total_return": variants[BACKTEST_PORTFOLIO]["total_return"],
        "backtest_qqq_return": variants[BACKTEST_PORTFOLIO]["qqq_return"],
        "backtest_variants": variants,
        "backtest_sessions": int(len(bt)),
        "backtest_mean_daily_excess": backtest_test.mean,
        "backtest_sd_daily_excess": backtest_test.std,
        "backtest_t_stat_daily_excess": backtest_test.t_stat,
        "backtest_p_value_daily_excess": backtest_test.p_value_two_sided,
        "observed_forward_excess": observed_excess,
        "simulated_excess_median": float(np.median(sim_excess)),
        "simulated_excess_5th_percentile": float(np.quantile(sim_excess, 0.05)),
        "p_value_as_bad_as_observed": p_as_bad,
        "monte_carlo_standard_error": math.sqrt(p_as_bad * (1 - p_as_bad) / N_RESAMPLES),
        "p_value_by_mean_block_length": sensitivity,
    }


# ---------------------------------------------------------------------------
# 3. Implementation audit
# ---------------------------------------------------------------------------


def _round_trips(orders: pd.DataFrame) -> pd.DataFrame:
    """First buy to full exit for every ticker, using fill prices and run dates."""
    rows = []
    for ticker, g in orders.sort_values(["as_of_date", "order_sequence"]).groupby("ticker"):
        opened = None
        for _, row in g.iterrows():
            if row["action"] == "BUY" and opened is None:
                opened = row
            elif (
                row["action"] == "SELL" and opened is not None and abs(row["position_quantity_after"]) < 1e-9
            ):
                rows.append(
                    {
                        "ticker": ticker,
                        "open_run": opened["as_of_date"],
                        "close_run": row["as_of_date"],
                        "days": int((row["as_of_date"] - opened["as_of_date"]).days),
                        "realized_return": float(row["fill_price"] / opened["fill_price"] - 1),
                    }
                )
                opened = None
    return pd.DataFrame(rows)


def implementation_audit(loaded: dict, forward: dict) -> dict:
    px, runs = loaded["px"], loaded["runs"]
    orders = pd.read_csv(DATA / "paper_orders.csv", parse_dates=["as_of_date"])
    trips = _round_trips(orders)
    capital = forward["start_equity"]

    # Map each run date to the close it used, then ask: what if every opening buy had been
    # held for the backtest's 20 sessions instead?
    run_to_price = dict(zip(pd.to_datetime(runs["date"]), runs["price_date"], strict=False))
    sessions = px["QQQ"].dropna().index
    sessions = sessions[sessions <= EXPERIMENT_END]
    held = []
    for _, trip in trips.iterrows():
        entry = run_to_price.get(trip["open_run"])
        if entry is None or entry not in sessions:
            continue
        i = sessions.get_loc(entry)
        if i + BACKTEST_HOLD_SESSIONS >= len(sessions):
            continue
        exit_day = sessions[i + BACKTEST_HOLD_SESSIONS]
        stock = px[trip["ticker"]]
        held.append(
            {
                "ticker": trip["ticker"],
                "realized_return": trip["realized_return"],
                "held_20_return": float(stock.loc[exit_day] / stock.loc[entry] - 1),
                "qqq_20_return": float(px["QQQ"].loc[exit_day] / px["QQQ"].loc[entry] - 1),
            }
        )
    held = pd.DataFrame(held)
    held["held_minus_qqq"] = held["held_20_return"] - held["qqq_20_return"]
    open_at_end = int(orders.groupby("ticker")["position_quantity_after"].last().gt(1e-9).sum())

    fig, ax = _figure(8, 3.6)
    bins = np.arange(0, trips["days"].max() + 2) - 0.5
    ax.hist(trips["days"], bins=bins, color=MODEL, edgecolor=SURFACE, linewidth=2)
    ax.axvline(28, color=INK, linewidth=1.2, linestyle="--")
    ax.text(
        27.5,
        ax.get_ylim()[1] * 0.9,
        "backtest holds\n20 trading sessions\n(≈28 calendar days)",
        ha="right",
        va="top",
        fontsize=9,
        color=INK,
    )
    ax.set_xlabel("Calendar days from first buy to full sale", color=INK_2, fontsize=9)
    ax.set_ylabel("Completed trades", color=INK_2, fontsize=9)
    _style(
        ax,
        "The live strategy was not the one that was backtested",
        f"{len(trips)} completed trades, median hold {trips['days'].median():.0f} days; "
        f"{open_at_end} positions still open at the end are not counted",
    )
    _save(fig, "holding_periods.png")

    fig, ax = _figure(8.0, 3.4)
    labels = [
        "Actual trades\n(as executed)",
        "Same entries,\nheld 20 sessions",
        "QQQ over the\nsame 20 sessions",
    ]
    means = [held["realized_return"].mean(), held["held_20_return"].mean(), held["qqq_20_return"].mean()]
    medians = [
        held["realized_return"].median(),
        held["held_20_return"].median(),
        held["qqq_20_return"].median(),
    ]
    ax.barh(range(3), [m * 100 for m in means], color=[NEG, MODEL, QQQ], height=0.55)
    for i, (mean, median) in enumerate(zip(means, medians, strict=True)):
        ax.text(
            max(mean, 0) * 100 + 0.2,
            i,
            f"mean {mean:+.1%}  (median {median:+.1%})",
            va="center",
            ha="left",
            fontsize=9,
            color=INK,
        )
    ax.set_yticks(range(3), labels)
    ax.invert_yaxis()
    ax.axvline(0, color=INK_2, linewidth=0.8)
    span = max(abs(m) for m in means) * 100
    ax.set_xlim(-span * 1.25, span * 2.4)
    ax.set_xlabel("Average return per trade (%)", color=INK_2, fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    _style(
        ax,
        "Same picks, different holding rule",
        f"{len(held)} trades that had 20 sessions left before the end; equal weight, before costs",
    )
    _save(fig, "counterfactual_hold.png")

    return {
        "orders": int(len(orders)),
        "turnover_multiple_of_capital": float(orders["notional"].abs().sum() / capital),
        "transaction_costs_pct_of_capital": float(orders["total_transaction_cost"].sum() / capital),
        "average_cost_bps_per_order": float(
            (orders["total_transaction_cost"] / orders["notional"].abs()).mean() * 10_000
        ),
        "average_fill_vs_close_bps": float(
            ((orders["fill_price"] / orders["close_price"] - 1).abs()).mean() * 10_000
        ),
        "closed_round_trips": int(len(trips)),
        "median_holding_days": float(trips["days"].median()),
        "positions_open_at_end": open_at_end,
        "counterfactual": {
            "trades_with_full_window": int(len(held)),
            "mean_realized_return": float(held["realized_return"].mean()),
            "median_realized_return": float(held["realized_return"].median()),
            "mean_held_20_return": float(held["held_20_return"].mean()),
            "median_held_20_return": float(held["held_20_return"].median()),
            "mean_qqq_20_return": float(held["qqq_20_return"].mean()),
            "median_held_minus_qqq": float(held["held_minus_qqq"].median()),
            "held_minus_qqq_test": _test_dict(one_sample_mean_test(held["held_minus_qqq"])),
            "held_minus_realized_test": _test_dict(
                one_sample_mean_test(held["held_20_return"] - held["realized_return"])
            ),
        },
    }


def _test_dict(t) -> dict:
    return {
        "mean": t.mean,
        "ci95": [t.ci_low, t.ci_high],
        "t_stat": t.t_stat,
        "p_value": t.p_value_two_sided,
        "n": t.n,
    }


# ---------------------------------------------------------------------------
# 4-6. Backtest signals: ranking skill, calibration, sample-size audit
# ---------------------------------------------------------------------------

# The backtest scores every (date, stock) three times -- once per horizon
# profile -- but all three rows share the same forward return. We test each
# horizon separately and headline "short": its 20-day time stop matches the
# 20-day outcome being measured.
HEADLINE_HORIZON = "short"


def _horizon_signals(raw: pd.DataFrame, horizon: str) -> tuple[pd.DataFrame, int]:
    sig = raw[raw["horizon"] == horizon].copy()
    per_date = sig.groupby("date")["ticker"].transform("size")
    stray_rows = int((per_date < 100).sum())
    sig = sig[per_date >= 100].copy()  # full cross-sections only
    sig["win"] = (sig["forward_return_20d"] > 0).astype(int)
    return sig, stray_rows


def _top5_minus_universe(sig: pd.DataFrame) -> pd.Series:
    universe = sig.groupby("date")["forward_return_20d"].mean()
    top5 = (
        sig.sort_values("high_probability_score", ascending=False)
        .groupby("date")
        .head(5)
        .groupby("date")["forward_return_20d"]
        .mean()
    )
    return top5 - universe


def _horizon_row(sig: pd.DataFrame) -> dict:
    ic = one_sample_mean_test(information_coefficients(sig, "high_probability_score", "forward_return_20d"))
    top = one_sample_mean_test(_top5_minus_universe(sig))
    brier = brier_skill(sig["calibrated_win_probability"], sig["win"])
    return {
        "ic_mean": ic.mean,
        "ic_ci95": [ic.ci_low, ic.ci_high],
        "ic_t_stat": ic.t_stat,
        "ic_p_value": ic.p_value_two_sided,
        "top5_minus_universe_mean_20d": top.mean,
        "top5_minus_universe_ci95": [top.ci_low, top.ci_high],
        "top5_minus_universe_t_stat": top.t_stat,
        "top5_minus_universe_p_value": top.p_value_two_sided,
        "brier_skill_score": brier.skill_score,
    }


def _skill(frame: pd.DataFrame) -> float:
    return brier_skill(frame["calibrated_win_probability"], frame["win"]).skill_score


def signal_tests() -> dict:
    raw = pd.read_csv(DATA / "walk_forward_signals.csv", parse_dates=["date"], low_memory=False)
    dup = duplication_factor(raw, ["date", "ticker"])
    by_horizon = {h: _horizon_row(_horizon_signals(raw, h)[0]) for h in ("short", "medium", "long")}
    sig, stray_rows = _horizon_signals(raw, HEADLINE_HORIZON)
    n_dates = sig["date"].nunique()
    stocks_per_date = float(sig.groupby("date").size().mean())

    ic = information_coefficients(sig, "high_probability_score", "forward_return_20d")
    ic_test = one_sample_mean_test(ic)
    top_test = one_sample_mean_test(_top5_minus_universe(sig))

    universe = sig.groupby("date")["forward_return_20d"].mean()
    qqq = pd.read_csv(DATA / "qqq_history.csv", parse_dates=["date"]).set_index("date")["QQQ"]
    qqq_20d = (qqq.shift(-20) / qqq - 1).reindex(universe.index)
    universe_vs_qqq = one_sample_mean_test(universe - qqq_20d)

    brier = brier_skill(sig["calibrated_win_probability"], sig["win"])
    murphy = brier_decomposition(sig["calibrated_win_probability"], sig["win"], CALIBRATION_BINS)
    rng = np.random.default_rng(SEED + 2)
    skill_ci = cluster_bootstrap(sig, "date", _skill, rng, N_CLUSTER_RESAMPLES)
    skill_by_date = sig.groupby("date")[["calibrated_win_probability", "win"]].apply(_skill)
    rel = reliability_table(sig["calibrated_win_probability"], sig["win"], CALIBRATION_BINS)
    sig["bin"] = pd.cut(sig["calibrated_win_probability"], bins=CALIBRATION_BINS, include_lowest=True)
    cluster_ci = []
    for _, g in sig.groupby("bin", observed=True):
        res = cluster_bootstrap(g, "date", lambda f: f["win"].mean(), rng, N_CLUSTER_RESAMPLES)
        cluster_ci.append([res.ci_low, res.ci_high])
    rel["cluster_ci_low"] = [c[0] for c in cluster_ci]
    rel["cluster_ci_high"] = [c[1] for c in cluster_ci]

    icc_return = intraclass_correlation(sig, "date", "forward_return_20d")
    icc_win = intraclass_correlation(sig, "date", "win")
    design_effect = 1 + (stocks_per_date - 1) * icc_return

    thresholds = {n: bonferroni_t_threshold(n, n_dates - 1) for n in (1, 3, 4, 10, SAVED_WALK_FORWARD_RUNS)}
    ic_sd = ic_test.std

    # Figures -----------------------------------------------------------------
    fig, ax = _figure(8, 3.8)
    colors = [POS if v > 0 else NEG for v in ic.values]
    positions = np.arange(len(ic))
    ax.bar(positions, ic.values, color=colors, width=0.7)
    ax.set_xticks(positions, [d.strftime("%-d %b\n%Y") for d in ic.index])
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.axhline(ic_test.mean, color=INK, linewidth=1.2, linestyle="--")
    ax.text(
        -0.45,
        ic_test.mean + 0.012,
        f"average {ic_test.mean:+.2f}",
        va="bottom",
        ha="left",
        fontsize=9,
        color=INK,
    )
    ax.set_ylabel("Rank correlation (IC)", color=INK_2, fontsize=9)
    ax.tick_params(axis="x", rotation=0, labelsize=7)
    _style(
        ax,
        "Does a higher score mean a higher 20-day return?",
        f"Positive on {ic_test.positive_count} of {n_dates} dates; average {ic_test.mean:+.2f}, "
        f"95% CI {ic_test.ci_low:+.2f} to {ic_test.ci_high:+.2f}: not distinguishable from zero",
    )
    _save(fig, "information_coefficient.png")

    fig, ax = _figure(5.8, 5.2)
    ax.plot([0, 1], [0, 1], color=INK_2, linewidth=1, linestyle="--")
    ax.text(0.92, 0.95, "perfect\ncalibration", fontsize=8, color=INK_2, ha="right")
    ax.axhline(brier.base_rate, color=INK_2, linewidth=0.8, linestyle=":")
    ax.text(0.02, brier.base_rate + 0.012, f"{brier.base_rate:.0%} of stocks rose", fontsize=8, color=INK_2)
    ax.errorbar(
        rel["mean_predicted"],
        rel["observed"],
        yerr=[rel["observed"] - rel["cluster_ci_low"], rel["cluster_ci_high"] - rel["observed"]],
        fmt="o",
        color=MODEL,
        ecolor=MODEL,
        elinewidth=1.5,
        capsize=3,
        markersize=8,
        markerfacecolor=SURFACE,
        markeredgewidth=2,
        zorder=3,
    )
    for i, (_, r) in enumerate(rel.iterrows()):
        above = i % 2 == 1
        ax.annotate(
            f"n={r['n']:,}",
            (r["mean_predicted"], r["cluster_ci_high"] if above else r["cluster_ci_low"]),
            xytext=(0, 6 if above else -6),
            textcoords="offset points",
            ha="center",
            va="bottom" if above else "top",
            fontsize=7.5,
            color=INK_2,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Model's stated win probability", color=INK_2, fontsize=9)
    ax.set_ylabel("Share of stocks that actually rose (20 days)", color=INK_2, fontsize=9)
    top_bin, low_bin = rel.iloc[-1], rel.iloc[0]
    _style(
        ax,
        "Stated probabilities vs what happened",
        f"Said {top_bin['mean_predicted']:.0%}: {top_bin['observed']:.0%} rose. "
        f"Said {low_bin['mean_predicted']:.0%}: {low_bin['observed']:.0%} rose. "
        "Bars: 95% CI resampling whole dates",
    )
    _save(fig, "calibration.png")

    return {
        "raw_signal_rows": int(len(raw)),
        "duplication_factor": dup,
        "distinct_date_ticker_outcomes": int(len(raw.drop_duplicates(["date", "ticker"]))),
        "independent_signals": int(len(sig)),
        "stray_single_ticker_rows_dropped": stray_rows,
        "signal_dates": n_dates,
        "stocks_per_date": stocks_per_date,
        "ic_mean": ic_test.mean,
        "ic_sd_across_dates": ic_sd,
        "ic_ci95": [ic_test.ci_low, ic_test.ci_high],
        "ic_t_stat": ic_test.t_stat,
        "ic_p_value": ic_test.p_value_two_sided,
        "ic_positive_dates": ic_test.positive_count,
        "dates_needed_for_ic": {
            "0.05": required_sample_size(0.05, ic_sd),
            "0.03": required_sample_size(0.03, ic_sd),
        },
        "top5_minus_universe_mean_20d": top_test.mean,
        "top5_minus_universe_ci95": [top_test.ci_low, top_test.ci_high],
        "top5_minus_universe_t_stat": top_test.t_stat,
        "top5_minus_universe_p_value": top_test.p_value_two_sided,
        "top5_beat_universe_dates": top_test.positive_count,
        "top5_sign_test_p_two_sided": top_test.sign_test_p_two_sided,
        "top5_wilcoxon_p_two_sided": top_test.wilcoxon_p_two_sided,
        "universe_minus_qqq_mean_20d": universe_vs_qqq.mean,
        "universe_minus_qqq_t_stat": universe_vs_qqq.t_stat,
        "universe_minus_qqq_p_value": universe_vs_qqq.p_value_two_sided,
        "brier": brier.brier,
        "brier_reference": brier.reference_brier,
        "brier_skill_score": brier.skill_score,
        "brier_skill_ci95_date_clustered": [skill_ci.ci_low, skill_ci.ci_high],
        "brier_skill_share_of_resamples_at_or_above_zero": float(1 - skill_ci.p_value_le_zero),
        "brier_skill_negative_dates": int((skill_by_date < 0).sum()),
        "murphy_reliability": murphy.reliability,
        "murphy_resolution": murphy.resolution,
        "murphy_uncertainty": murphy.uncertainty,
        "mean_forecast": murphy.mean_forecast,
        "base_win_rate": brier.base_rate,
        "reliability": rel.to_dict(orient="records"),
        "intraclass_correlation_20d_return": icc_return,
        "intraclass_correlation_win": icc_win,
        "design_effect": design_effect,
        "effective_independent_outcomes": float(len(sig) / design_effect),
        "bonferroni_t_thresholds": {str(k): v for k, v in thresholds.items()},
        "saved_walk_forward_runs": SAVED_WALK_FORWARD_RUNS,
        "headline_horizon": HEADLINE_HORIZON,
        "by_horizon": by_horizon,
    }


# ---------------------------------------------------------------------------
# Study design figure
# ---------------------------------------------------------------------------


def study_design(forward: dict, luck: dict, signals_dates: list[pd.Timestamp]) -> None:
    """One-glance timeline: backtest window, design period, freeze and forward test."""
    import matplotlib.dates as mdates

    fig, ax = _figure(9.0, 3.2)
    rows = [
        (
            "Walk-forward backtest",
            luck["backtest_start"],
            luck["backtest_end"],
            GRID_BAR,
            f"top picks {luck['backtest_total_return']:+.0%}  vs  QQQ {luck['backtest_qqq_return']:+.0%}",
        ),
        (
            "Score weights designed",
            "2026-05-28",
            "2026-07-08",
            GRID_BAR_LIGHT,
            "after the backtest period had already happened",
        ),
        (
            "Frozen forward test",
            forward["start"],
            forward["end"],
            MODEL,
            f"paper portfolio {forward['model_return']:+.1%}  vs  QQQ {forward['qqq_return']:+.1%}",
        ),
    ]
    for i, (name, start, end, color, note) in enumerate(rows):
        y = len(rows) - 1 - i
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        ax.barh(y, (end_ts - start_ts).days, left=start_ts, height=0.46, color=color)
        ax.text(start_ts, y + 0.36, name, fontsize=9.5, color=INK, fontweight="bold", va="bottom")
        if i == 0:  # long bar: write the result inside it
            ax.text(
                start_ts + pd.Timedelta(days=12),
                y,
                note,
                fontsize=9,
                color=SURFACE,
                va="center",
                fontweight="bold",
                zorder=4,
            )
        else:
            ax.text(end_ts + pd.Timedelta(days=6), y, note, fontsize=9, color=INK_2, va="center")
    backtest_y = len(rows) - 1
    for date in signals_dates:
        ax.plot([date, date], [backtest_y - 0.36, backtest_y - 0.27], color=INK_2, linewidth=1.4)
    freeze = pd.Timestamp(forward["start"])
    gate = pd.Timestamp(forward["prespecified_check"]["date"])
    ax.axvline(freeze, color=INK, linewidth=1, linestyle="--")
    ax.text(freeze, len(rows) - 0.25, "rules frozen  ", fontsize=8.5, color=INK, va="bottom", ha="right")
    ax.axvline(gate, color=INK_2, linewidth=1, linestyle=":")
    ax.text(gate, -0.55, "  20-session check", fontsize=8, color=INK_2, va="bottom")
    ax.set_yticks([])
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_xlim(pd.Timestamp("2025-02-15"), pd.Timestamp("2027-03-15"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    _style(
        ax,
        "Study design",
        f"Ticks: the {len(signals_dates)} backtest signal dates. "
        "The backtest period was known while the score was designed; the forward test was not.",
    )
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    _save(fig, "study_design.png")


def main() -> None:
    loaded = _load_forward()
    forward = forward_test(loaded)
    luck = luck_test(forward)
    implementation = implementation_audit(loaded, forward)
    signals = signal_tests()
    raw = pd.read_csv(DATA / "walk_forward_signals.csv", parse_dates=["date"], low_memory=False)
    sig, _ = _horizon_signals(raw, HEADLINE_HORIZON)
    study_design(forward, luck, sorted(sig["date"].unique()))
    summary = {
        "generated_by": "research/forward_test.py",
        "units": "returns are fractions (0.01 = 1 %); daily statistics are per mark-to-mark interval",
        "forward_test": forward,
        "luck_test": luck,
        "implementation": implementation,
        "signal_tests": signals,
        "settings": {
            "seed": SEED,
            "bootstrap_resamples": N_RESAMPLES,
            "cluster_bootstrap_resamples": N_CLUSTER_RESAMPLES,
            "mean_block_days": MEAN_BLOCK_DAYS,
            "experiment_end": str(EXPERIMENT_END.date()),
        },
    }
    out = ROOT / "results" / "summary.json"
    out.write_text(json.dumps(_round(summary), indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
