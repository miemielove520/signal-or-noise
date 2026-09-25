from __future__ import annotations

from datetime import date, datetime
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from stock_selector.json_io import dataframe_records, json_safe, write_json


class JsonSafeTest(unittest.TestCase):
    def test_numpy_scalars_are_converted_to_python_types(self) -> None:
        result = json_safe({"count": np.int64(3), "ratio": np.float64(1.5)})

        self.assertEqual(result, {"count": 3, "ratio": 1.5})
        self.assertIsInstance(result["count"], int)
        self.assertIsInstance(result["ratio"], float)

    def test_nan_and_inf_become_none(self) -> None:
        self.assertIsNone(json_safe(float("nan")))
        self.assertIsNone(json_safe(float("inf")))
        self.assertIsNone(json_safe(np.float64("nan")))
        self.assertIsNone(json_safe(np.float64("inf")))

    def test_dates_and_paths_are_stringified(self) -> None:
        self.assertEqual(json_safe(date(2024, 1, 2)), "2024-01-02")
        self.assertEqual(json_safe(datetime(2024, 1, 2, 3, 4, 5)), "2024-01-02T03:04:05")
        self.assertEqual(json_safe(pd.Timestamp("2024-01-02")), "2024-01-02T00:00:00")
        self.assertEqual(json_safe(Path("a") / "b"), "a/b")

    def test_containers_and_non_string_keys_are_normalized(self) -> None:
        result = json_safe({1: (np.int64(2), {3}), "nested": [Path("x")]})

        # Keys are coerced to strings; tuples and sets become lists.
        self.assertEqual(set(result.keys()), {"1", "nested"})
        self.assertEqual(result["1"], [2, [3]])
        self.assertEqual(result["nested"], ["x"])

    def test_series_and_dataframe_are_recordized(self) -> None:
        self.assertEqual(json_safe(pd.Series({"a": np.int64(1)})), {"a": 1})

        frame = pd.DataFrame({"ticker": ["AAA"], "value": [np.float64(2.0)]})
        self.assertEqual(dataframe_records(frame), [{"ticker": "AAA", "value": 2.0}])
        self.assertEqual(json_safe(frame), [{"ticker": "AAA", "value": 2.0}])


class WriteJsonTest(unittest.TestCase):
    def test_write_json_creates_parent_dirs_and_valid_utf8(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "nested" / "out.json"
            write_json(target, {"name": "市场", "value": np.int64(7)})

            self.assertTrue(target.exists())
            loaded = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(loaded, {"name": "市场", "value": 7})
            # ensure_ascii=False keeps non-ASCII readable rather than escaped.
            self.assertIn("市场", target.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
