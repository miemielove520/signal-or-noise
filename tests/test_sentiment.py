from __future__ import annotations

import unittest

from stock_selector.sentiment import build_sentiment_context


class SentimentTest(unittest.TestCase):
    def test_positive_headlines_score_above_neutral(self) -> None:
        context = build_sentiment_context(
            "AAPL",
            {
                "news_titles": [
                    "Apple beats estimates as services growth stays strong",
                    "Analyst raises price target after record profit",
                ]
            },
        )

        self.assertGreater(context.sentiment_score, 50)
        self.assertEqual(context.sentiment_label, "positive")
        self.assertEqual(context.sentiment_risk_level, "low")
        self.assertEqual(context.positive_news_level, "strong")
        self.assertGreater(context.positive_news_score, 0)
        self.assertFalse(context.sentiment_block_new_entries)

    def test_major_positive_news_is_graded_above_regular_positive_news(self) -> None:
        context = build_sentiment_context(
            "XYZ",
            {
                "news_titles": [
                    "XYZ raises guidance after major contract and record backlog",
                    "XYZ announces strategic partnership for data center backlog",
                ]
            },
        )

        self.assertEqual(context.positive_news_level, "major")
        self.assertEqual(context.positive_news_level_zh, "重大利好")
        self.assertGreaterEqual(context.positive_news_major_count, 1)
        self.assertIn("major", context.positive_news_drivers)

    def test_large_order_or_deal_is_major_positive(self) -> None:
        context = build_sentiment_context(
            "AVGO",
            {"news_titles": ["Apple announces chip deal with Broadcom worth more than $30 billion"]},
        )
        self.assertEqual(context.positive_news_level, "major")
        self.assertEqual(context.positive_news_level_zh, "重大利好")
        self.assertGreaterEqual(context.positive_news_major_count, 1)

    def test_high_risk_headlines_block_new_entries(self) -> None:
        context = build_sentiment_context(
            "XYZ",
            {
                "news_titles": [
                    "XYZ faces SEC probe over accounting issue",
                    "Analysts downgrade XYZ after guidance cut and lawsuit",
                ]
            },
        )

        self.assertEqual(context.sentiment_risk_level, "high")
        self.assertTrue(context.sentiment_block_new_entries)
        self.assertGreaterEqual(context.sentiment_high_risk_count, 2)

    def test_fake_catalyst_is_flagged_and_blocks_entries(self) -> None:
        context = build_sentiment_context(
            "ZZZ",
            {
                "news_titles": [
                    "ZZZ soars after reverse split and social media hype",
                    "Short seller report questions ZZZ accounting",
                ]
            },
        )

        self.assertEqual(context.risk_news_level, "fake_catalyst")
        self.assertEqual(context.risk_news_level_zh, "假利好/炒作")
        self.assertGreaterEqual(context.fake_catalyst_count, 1)
        self.assertLess(context.risk_news_score, 0)
        self.assertTrue(context.sentiment_block_new_entries)
        self.assertIn("fake catalyst", context.risk_news_drivers)

    def test_dilution_is_graded_but_does_not_hard_block(self) -> None:
        context = build_sentiment_context(
            "ZZZ",
            {
                "news_titles": [
                    "ZZZ announces public offering to raise capital",
                    "ZZZ files shelf registration for secondary offering",
                ]
            },
        )

        self.assertEqual(context.risk_news_level, "dilution")
        self.assertGreaterEqual(context.dilution_count, 1)
        self.assertLess(context.risk_news_score, 0)

    def test_fake_catalyst_penalizes_an_otherwise_positive_headline(self) -> None:
        pumped = build_sentiment_context(
            "ZZZ",
            {"news_titles": ["ZZZ surges on strong momentum amid social media hype and reverse split"]},
        )
        clean = build_sentiment_context(
            "ZZZ",
            {"news_titles": ["ZZZ surges on strong momentum"]},
        )
        self.assertLess(pumped.sentiment_score, clean.sentiment_score)

    def test_clean_positive_news_has_no_risk_flag(self) -> None:
        context = build_sentiment_context(
            "AAPL",
            {"news_titles": ["Apple beats estimates as services growth stays strong"]},
        )
        self.assertEqual(context.risk_news_level, "none")
        self.assertEqual(context.fake_catalyst_count, 0)
        self.assertEqual(context.dilution_count, 0)

    def test_missing_news_uses_unknown_neutral_context(self) -> None:
        context = build_sentiment_context("XYZ", None)

        self.assertEqual(context.sentiment_score, 50.0)
        self.assertEqual(context.sentiment_label, "unknown")
        self.assertEqual(context.sentiment_risk_level, "unknown")
        self.assertIn("snapshot unavailable", context.sentiment_warning)


if __name__ == "__main__":
    unittest.main()
