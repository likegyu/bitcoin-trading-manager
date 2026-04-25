import unittest

from agents.judge import _parse_judge_output, format_judge_block, JudgeResult


class JudgeTests(unittest.TestCase):
    def test_parse_judge_rubric_scores(self):
        parsed = _parse_judge_output(
            "판정: 하방 우위\n"
            "점수: price_structure=2, momentum=1, derivatives=-1, macro=0, "
            "account_risk_fit=1, counter_scenario=-2\n"
            "이유: 구조가 약합니다.\n"
            "Bull 핵심: 장기 지지\n"
            "Bear 핵심: 단기 이탈"
        )

        self.assertEqual(parsed["verdict"], "하방 우위")
        self.assertEqual(parsed["rubric_scores"]["price_structure"], 2)
        self.assertEqual(parsed["rubric_scores"]["counter_scenario"], -2)

    def test_format_judge_block_includes_scores(self):
        block = format_judge_block(
            JudgeResult(
                enabled=True,
                verdict="상방 우위",
                reasoning="구조 우세",
                bull_key="저점 상승",
                bear_key="저항 근접",
                raw_text="",
                rubric_scores={"price_structure": 2, "momentum": 1},
            )
        )

        self.assertIn("점수:", block)
        self.assertIn("price_structure=2", block)


if __name__ == "__main__":
    unittest.main()
