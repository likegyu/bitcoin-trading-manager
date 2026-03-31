import tempfile
import unittest
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

from macro_history import (
    MAX_ENTRIES,
    MacroHistoryTimeline,
    _snapshot_from_macro,
)


def _macro_sample(day_index: int) -> dict:
    return {
        "DFII10": {"value": 1.50 + day_index * 0.03},
        "DGS2": {"value": 3.80 - day_index * 0.02},
        "DTWEXBGS": {"value": 118.0 + day_index * 0.40},
        "STABLE_MCAP": {"value": 205.0 + day_index * 1.50},
        "USDT_DOM": {"value": 63.0 - day_index * 0.20},
        "BTC_DOM": {"value": 58.0 + day_index * 0.35},
    }


class MacroHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.timeline = MacroHistoryTimeline(Path(self.temp_dir.name) / "macro_history.jsonl")
        self.timeline._loaded = True

        base = datetime(2026, 3, 24, 0, 0, tzinfo=timezone.utc)
        snapshots = []
        for idx in range(8):
            observed_at = base + timedelta(hours=24 * idx)
            snapshots.append(_snapshot_from_macro(_macro_sample(idx), observed_at))

        self.timeline._entries = deque(snapshots, maxlen=MAX_ENTRIES)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_window_changes_are_calculated_correctly(self):
        summary = self.timeline._build_summary_locked(None)

        day = summary["windows"]["day"]["metrics"]
        swing = summary["windows"]["swing"]["metrics"]
        week = summary["windows"]["week"]["metrics"]

        self.assertAlmostEqual(day["STABLE_MCAP"]["change"], 1.50, places=6)
        self.assertAlmostEqual(day["USDT_DOM"]["change"], -0.20, places=6)
        self.assertAlmostEqual(day["BTC_DOM"]["change"], 0.35, places=6)

        self.assertAlmostEqual(swing["STABLE_MCAP"]["change"], 4.50, places=6)
        self.assertAlmostEqual(swing["USDT_DOM"]["change"], -0.60, places=6)
        self.assertAlmostEqual(swing["BTC_DOM"]["change"], 1.05, places=6)

        self.assertAlmostEqual(week["STABLE_MCAP"]["change"], 10.50, places=6)
        self.assertAlmostEqual(week["USDT_DOM"]["change"], -1.40, places=6)
        self.assertAlmostEqual(week["BTC_DOM"]["change"], 2.45, places=6)

    def test_trend_classification_uses_metric_thresholds(self):
        summary = self.timeline._build_summary_locked(None)
        week = summary["windows"]["week"]["metrics"]

        self.assertEqual(week["DFII10"]["trend"], "상승")
        self.assertEqual(week["DGS2"]["trend"], "하락")
        self.assertEqual(week["BTC_DOM"]["trend"], "상승")
        self.assertEqual(week["USDT_DOM"]["trend"], "하락")

    def test_attach_to_macro_populates_change_fields(self):
        latest_macro = _macro_sample(7)
        summary = self.timeline._build_summary_locked(None)

        self.timeline._attach_to_macro(latest_macro, summary)

        self.assertAlmostEqual(latest_macro["STABLE_MCAP"]["change24h"], 1.50, places=6)
        self.assertAlmostEqual(latest_macro["STABLE_MCAP"]["change72h"], 4.50, places=6)
        self.assertAlmostEqual(latest_macro["STABLE_MCAP"]["change7d"], 10.50, places=6)

        self.assertAlmostEqual(latest_macro["BTC_DOM"]["change24h"], 0.35, places=6)
        self.assertAlmostEqual(latest_macro["BTC_DOM"]["change72h"], 1.05, places=6)
        self.assertAlmostEqual(latest_macro["BTC_DOM"]["change7d"], 2.45, places=6)
        self.assertEqual(latest_macro["BTC_DOM"]["trend7d"], "상승")
        self.assertIn("_history_summary", latest_macro)


if __name__ == "__main__":
    unittest.main()
