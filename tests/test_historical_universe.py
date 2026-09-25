from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from stock_selector.historical_universe import (
    filter_to_members_as_of,
    load_historical_membership,
    members_as_of,
    survivorship_status,
)


def _membership() -> pd.DataFrame:
    # ENRON left in 2001 (delisted); AAPL still a member; NFLX joined 2010.
    return pd.DataFrame(
        {
            "ticker": ["AAPL", "ENRON", "NFLX"],
            "start_date": ["1990-01-01", "1995-01-01", "2010-01-01"],
            "end_date": ["", "2001-12-01", ""],
        }
    )


class MembershipTests(unittest.TestCase):
    def test_members_as_of_includes_then_current(self) -> None:
        m = load_membership_frame(_membership())
        # In 2000, ENRON was still a member; NFLX not yet.
        self.assertEqual(members_as_of(m, "2000-06-01"), ["AAPL", "ENRON"])

    def test_delisted_ticker_excluded_after_end(self) -> None:
        m = load_membership_frame(_membership())
        # In 2015, ENRON is gone (delisted), NFLX is in.
        self.assertEqual(members_as_of(m, "2015-06-01"), ["AAPL", "NFLX"])

    def test_filter_preserves_order_and_drops_nonmembers(self) -> None:
        m = load_membership_frame(_membership())
        got = filter_to_members_as_of(["NFLX", "AAPL", "ENRON"], m, "2015-06-01")
        self.assertEqual(got, ["NFLX", "AAPL"])

    def test_no_data_cannot_filter(self) -> None:
        empty = pd.DataFrame()
        self.assertEqual(filter_to_members_as_of(["AAPL", "ZZZ"], empty, "2015-01-01"), ["AAPL", "ZZZ"])

    def test_survivorship_status_flags_missing_data(self) -> None:
        self.assertFalse(survivorship_status(None)["handled"])
        self.assertEqual(survivorship_status(pd.DataFrame())["status"], "not_handled")
        self.assertTrue(survivorship_status(load_membership_frame(_membership()))["handled"])

    def test_load_missing_file_is_empty(self) -> None:
        self.assertTrue(load_historical_membership("/nope/x.csv").empty)

    def test_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hist.csv"
            _membership().to_csv(path, index=False)
            loaded = load_historical_membership(path)
            self.assertEqual(len(loaded), 3)
            self.assertIn("AAPL", set(loaded["ticker"]))


def load_membership_frame(raw: pd.DataFrame) -> pd.DataFrame:
    # Normalize via the loader by round-tripping through CSV (parses dates).
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.csv"
        raw.to_csv(path, index=False)
        return load_historical_membership(path)


if __name__ == "__main__":
    unittest.main()
