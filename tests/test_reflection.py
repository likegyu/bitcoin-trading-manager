import unittest

from agents.reflection import _compact_decision_meta, _directional_evaluation


class ReflectionDecisionMetaTests(unittest.TestCase):
    def test_directional_evaluation_uses_stored_signal(self):
        meta = {"signal": "매수", "confidence": 72, "pair": "BTC/USDC"}

        self.assertIn("적중", _directional_evaluation("analyst", meta, 1.2))
        self.assertIn("오판", _directional_evaluation("analyst", meta, -0.8))

    def test_compact_decision_meta_includes_quality_and_breakdown(self):
        meta = {
            "signal": "매도",
            "confidence": 68,
            "pair": "BTC/USDC",
            "confidence_breakdown": {"price_structure": 18},
            "data_quality": {"grade": "high", "score": 86},
            "derived_features_summary": {
                "higher_tf_bias": "bearish",
                "lower_tf_bias": "bearish",
                "conflicts": [],
            },
        }

        block = _compact_decision_meta("analyst", meta)

        self.assertIn("signal=매도", block)
        self.assertIn("data_quality=high(86/100)", block)
        self.assertIn("price_structure", block)


if __name__ == "__main__":
    unittest.main()
