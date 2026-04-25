import unittest

import pandas as pd

from analysis_context import build_analysis_context, build_derived_features


def _tf_frame(close: float, sma200: float, sma50: float, ema9: float) -> pd.DataFrame:
    idx = pd.date_range("2026-04-25", periods=120, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "open": [close] * 120,
            "high": [close * 1.01] * 120,
            "low": [close * 0.99] * 120,
            "close": [close] * 120,
            "volume": [100.0] * 120,
            "rsi": [58.0 if close > sma200 else 42.0] * 120,
            "macd_hist": [1.0 if close > sma200 else -1.0] * 119 + [1.5 if close > sma200 else -1.5],
            "bb_pct": [0.62 if close > sma200 else 0.38] * 120,
            "sma_200": [sma200] * 120,
            "sma_50": [sma50] * 120,
            "ema_9": [ema9] * 120,
            "volume_ma": [100.0] * 120,
            "atr": [close * 0.015] * 120,
        },
        index=idx,
    )


class AnalysisContextTests(unittest.TestCase):
    def test_derived_features_flag_tf_conflict(self):
        multi_tf = {
            "1d": _tf_frame(104, 100, 102, 105),
            "4h": _tf_frame(104, 100, 102, 105),
            "1h": _tf_frame(104, 100, 102, 105),
            "15m": _tf_frame(96, 100, 98, 95),
            "5m": _tf_frame(96, 100, 98, 95),
        }

        derived = build_derived_features(multi_tf)

        self.assertEqual(derived["higher_tf_bias"], "bullish")
        self.assertEqual(derived["lower_tf_bias"], "bearish")
        self.assertTrue(derived["conflicts"])

    def test_analysis_context_emits_quality_auditor_and_derived_blocks(self):
        multi_tf = {"1h": _tf_frame(104, 100, 102, 105)}

        ctx = build_analysis_context(
            multi_tf,
            macro_snapshot={},
            market_ctx={},
            account_ctx={},
        )

        self.assertIn("<data_quality>", ctx["text"])
        self.assertIn("<data_auditor>", ctx["text"])
        self.assertIn("<derived_features>", ctx["text"])
        self.assertIn("TNX_10Y 값 없음", ctx["quality"]["no_use"])


if __name__ == "__main__":
    unittest.main()
