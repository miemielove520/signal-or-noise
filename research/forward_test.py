"""Signal or noise? The statistical post-mortem of the frozen v2 model.

Reads only ``results/data/`` (see ``scripts/export_results.py``) and writes
``results/summary.json`` plus the figures in ``results/figures/``.

    python research/forward_test.py

Five questions, one section each:
  1. Forward test: how did the paper portfolio do against QQQ and its own universe?
  2. Was the loss bad luck? Compare the result with what the backtest implied.
  3. Does the score rank stocks at all? (information coefficient, top-5 vs universe)
  4. Are the model's win probabilities honest? (Brier skill, reliability)
  5. Sample-size hygiene: duplicated observations and multiple testing.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stock_selector.stats_tests import (  # noqa: E402
    bonferroni_t_threshold,
    bootstrap_statistic,
    brier_skill,
    compounded_return,
    duplication_factor,
    information_coefficients,
    one_sample_mean_test,
    reliability_table,
    simulate_window_returns,
)

DATA = ROOT / "results" / "data"
FIGURES = ROOT / "results" / "figures"
SEED = 20260709  # paper portfolio start date
N_RESAMPLES = 10_000
MEAN_BLOCK_DAYS = 5.0
BACKTEST_PORTFOLIO = "top_high_probability_score"  # closest backtest analogue of the paper rules
# Walk-forward runs saved to disk during development (many were code smoke tests,
# not strategy variants), so the true number of independent "looks" is uncertain.
SAVED_WALK_FORWARD_RUNS = 36

# ---- chart style (reference palette, light surface) ------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
MODEL = "#2a78d6"      # slot 1 blue
QQQ = "#eb6834"        # slot 2 orange
UNIVERSE = "#1baf7a"   # slot 3 aqua
NEG = "#e34948"        # diverging negative pole
POS = MODEL            # diverging positive pole


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


# ---------------------------------------------------------------------------
# 1. Forward test
# ---------------------------------------------------------------------------

def forward_test() -> dict:
    eq = pd.read_csv(DATA / "paper_equity_history.csv", parse_dates=["date"]).set_index("date")
    px = pd.read_csv(DATA / "forward_window_prices.csv", parse_dates=["date"]).set_index("date")
    start, end = eq.index[0], eq.index[-1]
    window = px.loc[start:end].ffill()
    universe_cols = [c for c in window.columns if c not in ("QQQ", "SPY") and window[c].notna().all()]
    delisted = [c for c in px.columns if c not in ("QQQ", "SPY") and px[c].notna().sum() == 0]

    universe_curve = (window[universe_cols] / window[universe_cols].iloc[0]).mean(axis=1)
    model_curve = eq["equity"] / eq["equity"].iloc[0]
    # The pipeline's stored benchmark_close has gaps on a few days; use the
    # downloaded QQQ closes on the same dates instead.
    qqq_px = window["QQQ"].reindex(eq.index).ffill()
    qqq_curve = qqq_px / qqq_px.iloc[0]

    model_daily = model_curve.pct_change().dropna()
    qqq_daily = qqq_curve.pct_change().dropna()
    excess_daily = (model_daily - qqq_daily).dropna()

    rng = np.random.default_rng(SEED)
    boot = bootstrap_statistic(excess_daily, np.mean, rng, N_RESAMPLES, MEAN_BLOCK_DAYS)

    orders = pd.read_csv(DATA / "paper_orders.csv", parse_dates=["as_of_date"])
    holds = _holding_periods(orders)
    capital = float(eq["equity"].iloc[0])

    fig, ax = _figure()
    for curve, color, label in (
        (model_curve, MODEL, "Model paper portfolio"),
        (qqq_curve, QQQ, "QQQ"),
        (universe_curve, UNIVERSE, f"Its own {len(universe_cols)}-stock universe, equal weight"),
    ):
        y = (curve - 1) * 100
        ax.plot(y.index, y.values, color=color, linewidth=2, label=label)
        ax.annotate(f"{y.iloc[-1]:+.1f}%", (y.index[-1], y.iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set_ylabel("Return since start (%)", color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=INK)
    _style(ax, "Forward test: the model lost while its own universe gained",
           f"Frozen policy v2, $1,000 paper portfolio, {start:%b %d} – {end:%b %d, %Y}")
    ax.margins(x=0.08)
    _save(fig, "forward_equity.png")

    fig, ax = _figure(8, 3.6)
    bins = np.arange(0, max(holds) + 2) - 0.5
    ax.hist(holds, bins=bins, color=MODEL, edgecolor=SURFACE, linewidth=2)
    ax.axvline(28, color=INK, linewidth=1.2, linestyle="--")
    ax.text(27.5, ax.get_ylim()[1] * 0.9, "backtest assumes\n20 trading days\n(≈28 calendar days)",
            ha="right", va="top", fontsize=9, color=INK)
    ax.set_xlabel("Calendar days from buy to full exit", color=INK_2, fontsize=9)
    ax.set_ylabel("Round trips", color=INK_2, fontsize=9)
    _style(ax, "The live strategy was not the one that was backtested",
           f"{len(holds)} closed round trips; median hold {np.median(holds):.0f} days")
    _save(fig, "holding_periods.png")

    return {
        "start": str(start.date()),
        "end": str(end.date()),
        "trading_days": int(len(model_daily)),
        "model_return": float(model_curve.iloc[-1] - 1),
        "qqq_return": float(qqq_curve.iloc[-1] - 1),
        "universe_equal_weight_return": float(universe_curve.iloc[-1] - 1),
        "universe_size_priced": len(universe_cols),
        "universe_delisted_or_unpriced": delisted,
        "max_drawdown": float((model_curve / model_curve.cummax() - 1).min()),
        "mean_daily_excess_vs_qqq": boot.estimate,
        "mean_daily_excess_ci95": [boot.ci_low, boot.ci_high],
        "share_of_resamples_with_mean_excess_above_zero": 1 - boot.p_value_le_zero,
        "orders": int(len(orders)),
        "turnover_multiple_of_capital": float(orders["notional"].abs().sum() / capital),
        "transaction_costs_pct_of_capital": float(orders["total_transaction_cost"].sum() / capital),
        "closed_round_trips": len(holds),
        "median_holding_days": float(np.median(holds)),
    }


def _holding_periods(orders: pd.DataFrame) -> list[int]:
    holds: list[int] = []
    for _, g in orders.sort_values(["as_of_date", "order_sequence"]).groupby("ticker"):
        opened = None
        for _, row in g.iterrows():
            if row["action"] == "BUY" and opened is None:
                opened = row["as_of_date"]
            elif row["action"] == "SELL" and opened is not None and abs(row["position_quantity_after"]) < 1e-9:
                holds.append(int((row["as_of_date"] - opened).days))
                opened = None
    return holds


# ---------------------------------------------------------------------------
# 2. Bad luck? Compare with the backtest's own distribution
# ---------------------------------------------------------------------------

def luck_test(forward: dict) -> dict:
    bt = pd.read_csv(DATA / "backtest_portfolio_vs_benchmark.csv", parse_dates=["date"])
    bt = bt[(bt["portfolio_name"] == BACKTEST_PORTFOLIO) & (bt["benchmark_ticker"] == "QQQ")]
    days = forward["trading_days"]
    observed_excess = (1 + forward["model_return"]) / (1 + forward["qqq_return"]) - 1

    rng = np.random.default_rng(SEED + 1)
    port = simulate_window_returns(bt["portfolio_daily_return"], days, rng, N_RESAMPLES, MEAN_BLOCK_DAYS)
    rng = np.random.default_rng(SEED + 1)  # same blocks for the benchmark leg
    bench = simulate_window_returns(bt["benchmark_daily_return"], days, rng, N_RESAMPLES, MEAN_BLOCK_DAYS)
    sim_excess = (1 + port) / (1 + bench) - 1
    p_as_bad = float(np.mean(sim_excess <= observed_excess))

    fig, ax = _figure()
    ax.hist(sim_excess * 100, bins=60, color=MODEL, edgecolor=SURFACE, linewidth=0.5)
    ax.axvline(observed_excess * 100, color=NEG, linewidth=2)
    ax.text(observed_excess * 100 - 1.5, ax.get_ylim()[1] * 0.95, f"forward test\n{observed_excess:+.1%}",
            color=INK, fontsize=9, va="top", ha="right")
    ax.axvline(0, color=INK_2, linewidth=0.8)
    ax.set_xlabel(f"Excess return vs QQQ over {days} trading days (%)", color=INK_2, fontsize=9)
    ax.set_ylabel("Simulated windows", color=INK_2, fontsize=9)
    _style(ax, f"If the backtest were true, a result this bad would happen {p_as_bad:.0%} of the time",
           f"{N_RESAMPLES:,} block-bootstrap windows drawn from the backtest's daily returns; "
           f"{p_as_bad:.2%} are as bad as the forward test")
    _save(fig, "backtest_vs_forward.png")

    total = compounded_return(bt["portfolio_daily_return"].to_numpy())
    total_b = compounded_return(bt["benchmark_daily_return"].to_numpy())
    return {
        "backtest_portfolio": BACKTEST_PORTFOLIO,
        "backtest_start": str(bt["date"].min().date()),
        "backtest_end": str(bt["date"].max().date()),
        "backtest_total_return": total,
        "backtest_qqq_return": total_b,
        "observed_forward_excess": observed_excess,
        "simulated_excess_median": float(np.median(sim_excess)),
        "simulated_excess_5th_percentile": float(np.quantile(sim_excess, 0.05)),
        "p_value_as_bad_as_observed": p_as_bad,
    }


# ---------------------------------------------------------------------------
# 3 & 4 & 5. Backtest signals: ranking skill, calibration, sample hygiene
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


def _horizon_row(sig: pd.DataFrame) -> dict:
    ic = one_sample_mean_test(information_coefficients(sig, "high_probability_score", "forward_return_20d"))
    universe = sig.groupby("date")["forward_return_20d"].mean()
    top5 = (sig.sort_values("high_probability_score", ascending=False)
            .groupby("date").head(5).groupby("date")["forward_return_20d"].mean())
    top = one_sample_mean_test(top5 - universe)
    brier = brier_skill(sig["calibrated_win_probability"], sig["win"])
    return {
        "ic_mean": ic.mean, "ic_t_stat": ic.t_stat, "ic_p_value": ic.p_value_two_sided,
        "top5_minus_universe_mean_20d": top.mean, "top5_minus_universe_t_stat": top.t_stat,
        "top5_minus_universe_p_value": top.p_value_two_sided,
        "brier_skill_score": brier.skill_score,
    }


def signal_tests() -> dict:
    raw = pd.read_csv(DATA / "walk_forward_signals.csv", parse_dates=["date"], low_memory=False)
    dup = duplication_factor(raw, ["date", "ticker"])
    by_horizon = {h: _horizon_row(_horizon_signals(raw, h)[0]) for h in ("short", "medium", "long")}
    sig, stray_rows = _horizon_signals(raw, HEADLINE_HORIZON)
    n_dates = sig["date"].nunique()

    ic = information_coefficients(sig, "high_probability_score", "forward_return_20d")
    ic_test = one_sample_mean_test(ic)

    universe = sig.groupby("date")["forward_return_20d"].mean()
    top5 = (sig.sort_values("high_probability_score", ascending=False)
            .groupby("date").head(5).groupby("date")["forward_return_20d"].mean())
    top_test = one_sample_mean_test(top5 - universe)

    qqq = pd.read_csv(DATA / "qqq_history.csv", parse_dates=["date"]).set_index("date")["QQQ"]
    qqq_20d = (qqq.shift(-20) / qqq - 1).reindex(universe.index)
    universe_vs_qqq = one_sample_mean_test(universe - qqq_20d)

    brier = brier_skill(sig["calibrated_win_probability"], sig["win"])
    rel = reliability_table(sig["calibrated_win_probability"], sig["win"], [0, 0.5, 0.55, 0.6, 0.65, 1.0])

    thresholds = {n: bonferroni_t_threshold(n, n_dates - 1) for n in (1, 10, SAVED_WALK_FORWARD_RUNS)}

    # Figures -----------------------------------------------------------------
    fig, ax = _figure(8, 3.8)
    colors = [POS if v > 0 else NEG for v in ic.values]
    ax.bar(ic.index.strftime("%Y-%m-%d"), ic.values, color=colors, width=0.7)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.axhline(ic_test.mean, color=INK, linewidth=1.2, linestyle="--")
    ax.text(-0.4, ic_test.mean + 0.01, f"mean {ic_test.mean:+.3f}", va="bottom", ha="left", fontsize=9, color=INK)
    ax.set_ylabel("Rank correlation (IC)", color=INK_2, fontsize=9)
    ax.tick_params(axis="x", rotation=45)
    _style(ax, "Does a higher score mean a higher 20-day return?",
           f"Spearman IC per signal date, {n_dates} dates × ~{len(sig) // n_dates} stocks; "
           f"t = {ic_test.t_stat:.2f}, p = {ic_test.p_value_two_sided:.2f}")
    _save(fig, "information_coefficient.png")

    fig, ax = _figure(5.6, 5.0)
    ax.plot([0, 1], [0, 1], color=INK_2, linewidth=1, linestyle="--")
    ax.text(0.9, 0.93, "perfect\ncalibration", fontsize=8, color=INK_2, ha="right")
    ax.axhline(brier.base_rate, color=GRID, linewidth=1)
    ax.errorbar(rel["mean_predicted"], rel["observed"],
                yerr=[rel["observed"] - rel["ci_low"], rel["ci_high"] - rel["observed"]],
                fmt="o", color=MODEL, ecolor=MODEL, elinewidth=1.5, capsize=0, markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2)
    for _, r in rel.iterrows():
        ax.annotate(f"n={r['n']}", (r["mean_predicted"], r["ci_high"]), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8, color=INK_2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Model's predicted win probability", color=INK_2, fontsize=9)
    ax.set_ylabel("Observed share of winners (20 days)", color=INK_2, fontsize=9)
    _style(ax, "The probabilities are close to a coin flip",
           f"Brier skill score {brier.skill_score:+.2f} vs. always guessing {brier.base_rate:.0%}")
    _save(fig, "calibration.png")

    return {
        "raw_signal_rows": int(len(raw)),
        "duplication_factor": dup,
        "independent_signals": int(len(sig)),
        "stray_single_ticker_rows_dropped": stray_rows,
        "signal_dates": n_dates,
        "ic_mean": ic_test.mean,
        "ic_t_stat": ic_test.t_stat,
        "ic_p_value": ic_test.p_value_two_sided,
        "ic_positive_dates": ic_test.positive_count,
        "top5_minus_universe_mean_20d": top_test.mean,
        "top5_minus_universe_t_stat": top_test.t_stat,
        "top5_minus_universe_p_value": top_test.p_value_two_sided,
        "top5_beat_universe_dates": top_test.positive_count,
        "top5_sign_test_p_one_sided": top_test.sign_test_p_one_sided,
        "universe_minus_qqq_mean_20d": universe_vs_qqq.mean,
        "universe_minus_qqq_t_stat": universe_vs_qqq.t_stat,
        "brier": brier.brier,
        "brier_reference": brier.reference_brier,
        "brier_skill_score": brier.skill_score,
        "base_win_rate": brier.base_rate,
        "reliability": rel.to_dict(orient="records"),
        "bonferroni_t_thresholds": {str(k): v for k, v in thresholds.items()},
        "saved_walk_forward_runs": SAVED_WALK_FORWARD_RUNS,
        "headline_horizon": HEADLINE_HORIZON,
        "by_horizon": by_horizon,
    }


def main() -> None:
    forward = forward_test()
    summary = {
        "forward_test": forward,
        "luck_test": luck_test(forward),
        "signal_tests": signal_tests(),
        "settings": {"seed": SEED, "bootstrap_resamples": N_RESAMPLES, "mean_block_days": MEAN_BLOCK_DAYS},
    }
    out = ROOT / "results" / "summary.json"
    out.write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")
    print(json.dumps({k: v for k, v in summary.items() if k != "signal_tests"}, indent=2, default=float))
    print(json.dumps({k: v for k, v in summary["signal_tests"].items() if k != "reliability"}, indent=2, default=float))


if __name__ == "__main__":
    main()
