from __future__ import annotations

import unittest

from stock_selector.snapshot import extract_news_titles


class SnapshotTest(unittest.TestCase):
    def test_extract_news_titles_from_nested_payload(self) -> None:
        titles = extract_news_titles(
            [
                {"content": {"title": "First headline"}},
                {"title": "Second headline"},
                {"nested": [{"title": "First headline"}]},
            ]
        )

        self.assertEqual(titles, ["First headline", "Second headline"])


if __name__ == "__main__":
    unittest.main()
