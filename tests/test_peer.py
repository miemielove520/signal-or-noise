from __future__ import annotations

import unittest

import pandas as pd

from stock_selector.peer import (
    build_peer_comparison_frame,
    choose_peer_tickers,
    render_peer_comparison_report,
)


class PeerTest(unittest.TestCase):
    def test_choose_peer_tickers_uses_software_industry_for_now(self) -> None:
        peers = choose_peer_tickers("NOW", "Technology", "Software - Application", max_peers=4)

        self.assertNotIn("NOW", peers)
        self.assertEqual(peers[:3], ["CRM", "MSFT", "ADBE"])
        self.assertEqual(len(peers), 4)

    def test_choose_peer_tickers_uses_building_products_for_aaon(self) -> None:
        peers = choose_peer_tickers("AAON", "Industrials", "Building Products", max_peers=4)

        self.assertNotIn("AAON", peers)
        self.assertEqual(peers[:3], ["CARR", "LII", "TT"])
        self.assertEqual(len(peers), 4)

    def test_peer_comparison_ranks_by_high_probability_score(self) -> None:
        target = make_analysis(
            "NOW",
            70.0,
            passed=False,
            valuation_score=52.0,
            forward_pe=58.0,
            peg_ratio=3.2,
        )
        peer = StubResult(
            "CRM",
            make_analysis(
                "CRM",
                82.0,
                passed=True,
                valuation_score=68.0,
                forward_pe=34.0,
                peg_ratio=2.0,
            ),
            {"company_name": "Salesforce"},
        )

        frame = build_peer_comparison_frame(
            target_ticker="NOW",
            target_analysis=target,
            target_snapshot={"company_name": "ServiceNow"},
            peer_results=[peer],
        )
        report = render_peer_comparison_report(frame)

        self.assertEqual(frame.iloc[0]["ticker"], "CRM")
        self.assertEqual(int(frame[frame["ticker"] == "NOW"]["target_peer_rank"].iloc[0]), 2)
        self.assertIn("## Peer Comparison / 同业比较", report)
        self.assertIn("## Peer Valuation / 同业估值", report)
        self.assertIn("Target rank / 目标排名", report)
        self.assertIn("valuation_peer_rank", frame.columns)
        self.assertIn("peer_valuation_label", frame.columns)
        self.assertEqual(
            frame[frame["ticker"] == "NOW"]["peer_valuation_label"].iloc[0],
            "richer_than_peers",
        )

    def test_peer_valuation_marks_conflicting_metrics_as_mixed(self) -> None:
        target = make_analysis(
            "HPQ",
            60.0,
            passed=False,
            valuation_score=55.0,
            forward_pe=8.0,
            peg_ratio=9.0,
        )
        peer = StubResult(
            "DELL",
            make_analysis(
                "DELL",
                65.0,
                passed=False,
                valuation_score=65.0,
                forward_pe=18.0,
                peg_ratio=1.2,
            ),
            {"company_name": "Dell"},
        )

        frame = build_peer_comparison_frame(
            target_ticker="HPQ",
            target_analysis=target,
            target_snapshot={"company_name": "HP"},
            peer_results=[peer],
        )

        self.assertEqual(
            frame[frame["ticker"] == "HPQ"]["peer_valuation_label"].iloc[0],
            "mixed_peer_valuation",
        )


class StubResult:
    def __init__(self, ticker: str, analysis: pd.DataFrame, snapshot: dict[str, object]) -> None:
        self.ticker = ticker
        self.analysis = analysis
        self.snapshot = snapshot


def make_analysis(
    ticker: str,
    high_probability_score: float,
    passed: bool,
    valuation_score: float = 65.0,
    forward_pe: float = 30.0,
    peg_ratio: float = 2.0,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": ticker,
                "horizon": "short",
                "sector": "Technology",
                "industry": "Software",
                "screening_action": "high_probability_watchlist_candidate"
                if passed
                else "not_high_probability_setup_now",
                "screening_action_zh": "高概率观察候选" if passed else "当前不是高概率机会",
                "high_probability_score": high_probability_score,
                "high_probability_level": "high" if passed else "medium",
                "quality_gate_passed": passed,
                "quality_gate_fail_reasons": "all strict quality gates passed"
                if passed
                else "backtest win rate too low",
                "quality_gate_fail_reasons_zh": "所有严格质量门槛通过"
                if passed
                else "回测胜率不足",
                "signal_score": 75.0,
                "confidence_score": 80.0,
                "data_quality_score": 90.0,
                "market_score": 70.0,
                "relative_strength_score": 65.0,
                "fundamental_score": 72.0,
                "valuation_score": valuation_score,
                "valuation_risk_level": "low" if valuation_score >= 55 else "medium",
                "valuation_forward_pe": forward_pe,
                "valuation_peg_ratio": peg_ratio,
                "valuation_growth_reference": 0.12,
                "valuation_profit_margin": 0.20,
                "sector_score": 60.0,
            }
        ]
    )


if __name__ == "__main__":
    unittest.main()
