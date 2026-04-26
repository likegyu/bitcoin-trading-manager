import unittest
from types import SimpleNamespace

from agents.signal_processing import extract_trading_signal


class SignalProcessingTests(unittest.TestCase):
    def test_judge_alignment_tolerates_markdown_verdict(self):
        report = (
            "📊 **관점:** **상방 우위**\n"
            "💯 **확신도:** **74%**\n"
        )
        judge = SimpleNamespace(enabled=True, verdict="**상방 우위**")

        signal = extract_trading_signal(report, judge_result=judge)

        self.assertEqual(signal.signal_kr, "매수")
        self.assertEqual(signal.confidence, 74)
        self.assertEqual(signal.judge_verdict, "상방 우위")
        self.assertTrue(signal.judge_aligned)

    def test_judge_alignment_detects_markdown_mismatch(self):
        report = "📊 **관점:** **상방 우위**\n💯 확신도: 74%"
        judge = SimpleNamespace(enabled=True, verdict="- **하방 우위**")

        signal = extract_trading_signal(report, judge_result=judge)

        self.assertEqual(signal.judge_verdict, "하방 우위")
        self.assertFalse(signal.judge_aligned)


if __name__ == "__main__":
    unittest.main()
