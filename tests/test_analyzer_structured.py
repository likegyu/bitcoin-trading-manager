import unittest

from analyzer import (
    _extract_analysis_json,
    _levels_from_structured,
    _signal_from_structured,
    _strip_analysis_json_block,
)


class AnalyzerStructuredOutputTests(unittest.TestCase):
    def test_extracts_json_and_strips_machine_block(self):
        raw = (
            '<analysis_json>{"view":"상방 우위","confidence":74,'
            '"trade":{"entry":"$100,000","stop":99000,"target":102000,"leverage":3},'
            '"levels":{"resistance":103000,"support":98500,'
            '"bull_trigger":101000,"bear_trigger":98000}}</analysis_json>\n'
            "📊 관점: 상방 우위\n"
            "💯 확신도: 74%"
        )

        parsed = _extract_analysis_json(raw)
        report = _strip_analysis_json_block(raw)

        self.assertEqual(parsed["confidence"], 74)
        self.assertEqual(_signal_from_structured(parsed), "매수")
        self.assertTrue(report.startswith("📊 관점"))
        self.assertNotIn("analysis_json", report)

    def test_structured_levels_normalize_prices(self):
        parsed = {
            "levels": {
                "resistance": "$103,500",
                "support": "N/A",
                "bull_trigger": 101000,
                "bear_trigger": "98,750.5",
            },
            "trade": {
                "entry": "$100,000",
                "stop": 99000,
                "target": None,
            },
        }

        levels = _levels_from_structured(parsed)

        self.assertEqual(levels["resistance"], 103500.0)
        self.assertIsNone(levels["support"])
        self.assertEqual(levels["bear_trigger"], 98750.5)
        self.assertEqual(levels["entry"], 100000.0)


if __name__ == "__main__":
    unittest.main()
