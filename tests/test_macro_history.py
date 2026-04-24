import tempfile
import unittest
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

from macro_history import (
    MAX_ENTRIES,
    MacroHistoryTimeline,
    _normalize_legacy_yield_units,
    _snapshot_from_macro,
)


def _macro_sample(day_index: int) -> dict:
    return {
        "TNX_10Y": {"value": 4.20 + day_index * 0.03},
        "FVX_5Y": {"value": 3.80 - day_index * 0.02},
        "DXY": {"value": 103.0 + day_index * 0.40},
        "STABLE_MCAP": {"value": 205.0 + day_index * 1.50},
        "USDT_DOM": {"value": 63.0 - day_index * 0.20},
        "BTC_DOM": {"value": 58.0 + day_index * 0.35},
        "HYG_LQD": {"value": 0.8200 + day_index * 0.0015},
        "IBIT_PX": {"value": 55.00 + day_index * 0.80},
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

        self.assertEqual(week["TNX_10Y"]["trend"], "상승")
        self.assertEqual(week["FVX_5Y"]["trend"], "하락")
        self.assertEqual(week["BTC_DOM"]["trend"], "상승")
        self.assertEqual(week["USDT_DOM"]["trend"], "하락")
        # HYG_LQD 주간 변화 = 7*0.0015 = 0.0105, threshold=0.002 → 상승
        self.assertEqual(week["HYG_LQD"]["trend"], "상승")
        # IBIT_PX 주간 변화 = 7*0.80 = 5.6, threshold=0.50 → 상승
        self.assertEqual(week["IBIT_PX"]["trend"], "상승")

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

    def test_normalizes_legacy_over_scaled_yields_only_before_cutoff(self):
        legacy = {
            "observed_ts": datetime(2026, 4, 24, 17, 20, tzinfo=timezone.utc).timestamp(),
            "TNX_10Y": 0.4304,
            "FVX_5Y": 0.3913,
        }
        self.assertTrue(_normalize_legacy_yield_units(legacy))
        self.assertAlmostEqual(legacy["TNX_10Y"], 4.304, places=6)
        self.assertAlmostEqual(legacy["FVX_5Y"], 3.913, places=6)

        future_low_rate = {
            "observed_ts": datetime(2026, 4, 25, 0, 0, tzinfo=timezone.utc).timestamp(),
            "TNX_10Y": 0.95,
            "FVX_5Y": 0.80,
        }
        self.assertFalse(_normalize_legacy_yield_units(future_low_rate))
        self.assertAlmostEqual(future_low_rate["TNX_10Y"], 0.95, places=6)
        self.assertAlmostEqual(future_low_rate["FVX_5Y"], 0.80, places=6)


if __name__ == "__main__":
    unittest.main()
