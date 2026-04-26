import unittest

from analyzer import (
    _extract_analysis_json,
    _levels_from_structured,
    _normalize_analysis_json,
    _render_report_from_structured,
    _signal_from_structured,
    _strip_analysis_json_block,
    parse_leverage,
    parse_report_sections,
    parse_signal,
    parse_trade_levels,
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

    def test_renders_report_when_tool_use_omits_body(self):
        parsed = {
            "view": "하방 우위",
            "confidence": 55,
            "regime": "박스",
            "confidence_breakdown": {
                "price_structure": 16,
                "momentum": 11,
                "derivatives": 8,
                "macro": 7,
                "account_risk_fit": 2,
                "data_quality_penalty": -3,
                "counter_scenario_penalty": -6,
            },
            "data_quality_notes": ["마지막 캔들은 미완성봉"],
            "key_facts": ["1h 가격이 단기 지지에 근접"],
            "inferences": ["반등 실패 시 하방 압력이 우세"],
            "counter_scenario": ["78,550 회복 시 하방 관점 약화"],
            "levels": {
                "resistance": 78347.99,
                "support": 77207,
                "bull_trigger": 78550,
                "bear_trigger": 77207,
            },
            "trade": {
                "entry": None,
                "stop": 78550,
                "target": 76505,
                "leverage": 3,
            },
            "actions": {
                "aggressive": "77,207 이탈 확인 시 소액 숏",
                "conservative": "이탈 후 되돌림 실패까지 대기",
            },
            "invalidation": "78,550 회복",
            "summary": "박스 하단 이탈 여부가 핵심입니다.",
        }

        report = _render_report_from_structured(parsed)
        meta = parse_report_sections(report)
        signal, confidence = parse_signal(report)
        levels = parse_trade_levels(report)

        self.assertTrue(meta["format_ok"])
        self.assertEqual(signal, "매도")
        self.assertEqual(confidence, 35)
        self.assertEqual(parse_leverage(report), 3)
        self.assertEqual(levels["stop"], 78550.0)
        self.assertEqual(levels["target"], 76505.0)
        self.assertNotIn("record_analysis", report)

    def test_normalizes_confidence_to_breakdown_sum(self):
        parsed = {
            "confidence": 55,
            "confidence_breakdown": {
                "price_structure": 16,
                "momentum": 11,
                "derivatives": 8,
                "macro": 7,
                "account_risk_fit": 2,
                "data_quality_penalty": -3,
                "counter_scenario_penalty": -6,
            },
        }

        normalized, adjustments = _normalize_analysis_json(parsed)

        self.assertEqual(normalized["confidence"], 35)
        self.assertTrue(adjustments)

    def test_parse_signal_tolerates_markdown_wrapped_fields(self):
        report = (
            "📊 **관점:** **상방 우위**\n"
            "💯 **확신도:** **74%**\n"
        )

        signal, confidence = parse_signal(report)

        self.assertEqual(signal, "매수")
        self.assertEqual(confidence, 74)


if __name__ == "__main__":
    unittest.main()
