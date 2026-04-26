import unittest

from agents.consistency_check import _check_confidence_breakdown


class ConsistencyCheckTests(unittest.TestCase):
    def test_confidence_breakdown_uses_clamped_sum(self):
        issues = _check_confidence_breakdown({
            "confidence": 1,
            "confidence_breakdown": {
                "price_structure": 0,
                "momentum": 0,
                "derivatives": 0,
                "macro": 0,
                "account_risk_fit": 0,
                "data_quality_penalty": -15,
                "counter_scenario_penalty": -10,
            },
        })

        self.assertEqual(issues, [])

    def test_confidence_zero_is_invalid(self):
        issues = _check_confidence_breakdown({
            "confidence": 0,
            "confidence_breakdown": {
                "price_structure": 0,
                "momentum": 0,
                "derivatives": 0,
                "macro": 0,
                "account_risk_fit": 0,
                "data_quality_penalty": 0,
                "counter_scenario_penalty": 0,
            },
        })

        self.assertTrue(issues)

    def test_breakdown_component_out_of_range_is_reported(self):
        issues = _check_confidence_breakdown({
            "confidence": 100,
            "confidence_breakdown": {
                "price_structure": 999,
                "momentum": 20,
                "derivatives": 20,
                "macro": 15,
                "account_risk_fit": 15,
                "data_quality_penalty": 0,
                "counter_scenario_penalty": 0,
            },
        })

        self.assertTrue(any("price_structure" in issue for issue in issues))


if __name__ == "__main__":
    unittest.main()
