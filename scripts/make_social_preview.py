"""Render the 1280x640 GitHub social-preview card (docs/social-preview.png).

Numbers come from results/summary.json and the curve from results/data, so the
card always matches the report. Upload it under Settings -> Social preview.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stock_selector.paper_tracker import price_dates_for_runs  # noqa: E402

BG = "#0f1724"
INK = "#f4f6f8"
INK_2 = "#a9b4c2"
MODEL = "#5a9cf0"
QQQ = "#f08a5d"


def main() -> None:
    summary = json.loads((ROOT / "results" / "summary.json").read_text())
    forward, luck = summary["forward_test"], summary["luck_test"]
    px = pd.read_csv(ROOT / "results/data/forward_window_prices.csv", parse_dates=["date"]).set_index("date")
    runs = pd.read_csv(ROOT / "results/data/paper_equity_history.csv")
    runs["price_date"] = price_dates_for_runs(runs["executed_at"], px["QQQ"].dropna().index)
    eq = runs.drop_duplicates("price_date", keep="last").set_index("price_date").sort_index()
    eq = eq[eq.index <= pd.Timestamp(forward["end"])]
    model = (eq["equity"] / eq["equity"].iloc[0] - 1) * 100
    qqq = (px["QQQ"].reindex(eq.index) / px["QQQ"].reindex(eq.index).iloc[0] - 1) * 100

    backtest_days = (pd.Timestamp(luck["backtest_end"]) - pd.Timestamp(luck["backtest_start"])).days
    forward_days = (pd.Timestamp(forward["end"]) - pd.Timestamp(forward["start"])).days

    def pct(value: float, digits: int) -> str:
        return f"{value:+.{digits}%}".replace("-", "\u2212")

    fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
    fig.patch.set_facecolor(BG)
    fig.text(0.06, 0.82, "Signal or Noise?", fontsize=46, color=INK, fontweight="bold")
    fig.text(0.06, 0.74, "A frozen forward test of a multi-factor stock model", fontsize=19, color=INK_2)
    blocks = [
        (
            pct(luck["backtest_total_return"], 0),
            "walk-forward backtest",
            f"QQQ {pct(luck['backtest_qqq_return'], 0)}  ·  {round(backtest_days / 30.4)} months",
        ),
        (
            pct(forward["model_return"], 1),
            "frozen live paper test",
            f"QQQ {pct(forward['qqq_return'], 1)}  ·  {round(forward_days / 7)} weeks",
        ),
    ]
    for i, (value, label, context) in enumerate(blocks):
        x = 0.06 + i * 0.24
        fig.text(x, 0.5, value, fontsize=34, color=INK, fontweight="bold")
        fig.text(x, 0.445, label, fontsize=13, color=INK)
        fig.text(x, 0.405, context, fontsize=12, color=INK_2)
    fig.text(0.06, 0.25, "Block bootstrap  ·  rank IC  ·  Brier skill", fontsize=13, color=INK_2)
    fig.text(0.06, 0.205, "Multiple testing  ·  pseudo-replication", fontsize=13, color=INK_2)
    fig.text(0.06, 0.08, "github.com/miemielove520/signal-or-noise", fontsize=13, color=INK_2)

    ax = fig.add_axes([0.62, 0.17, 0.28, 0.5])
    ax.set_facecolor(BG)
    ax.plot(model.index, model.values, color=MODEL, linewidth=3)
    ax.plot(qqq.index, qqq.values, color=QQQ, linewidth=3)
    ax.axhline(0, color=INK_2, linewidth=0.8, alpha=0.6)
    for series, color, name in ((model, MODEL, "model"), (qqq, QQQ, "QQQ")):
        ax.annotate(
            name,
            (series.index[-1], series.iloc[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            color=color,
            fontsize=13,
            va="center",
            annotation_clip=False,
        )
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xticks([])
    ax.tick_params(colors=INK_2, labelsize=11, length=0)
    ax.set_yticks([-20, -10, 0])
    ax.set_yticklabels(["\u221220%", "\u221210%", "0%"])
    ax.set_title(f"{forward['start']} \u2192 {forward['end']}", color=INK_2, fontsize=12, loc="left")

    out = ROOT / "docs" / "social-preview.png"
    fig.savefig(out, facecolor=BG)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
