"""Realized Volatility 연환산 계수 회귀 테스트.

과거 버전에서는 전통 주식의 영업일 252 를 썼으나, BTC 는 24/7 거래이므로
실제 연환산 계수는 365 이어야 한다. 252 를 쓰면 DVOL 내재변동성 대비 약
√(252/365) ≈ 83%, 즉 RV 가 약 17% 저평가돼 IV 프리미엄 판정이 편향된다.
"""
from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from indicators import _RV_ANNUALIZE, add_realized_vol


class RealizedVolAnnualizationTests(unittest.TestCase):
    def test_daily_rv_uses_365_not_252(self):
        # 고정 시드로 일정한 로그수익률을 만들어 계산 결과가 365 연환산과
        # 일치하는지 검증한다. 252 로 돌리면 결과가 약 17% 작아져 실패한다.
        rng = np.random.default_rng(42)
        log_returns = rng.normal(loc=0.0, scale=0.02, size=400)
        prices = 30000.0 * np.exp(np.cumsum(log_returns))
        df = pd.DataFrame({"close": prices})

        out = add_realized_vol(df.copy(), tf="1d", period=20)
        period = 20

        # 마지막 20 봉 로그수익률 표준편차 * sqrt(365) * 100
        log_ret = np.log(df["close"] / df["close"].shift(1))
        expected_rv = (
            log_ret.tail(period).std(ddof=1) * math.sqrt(365) * 100
        )
        observed_rv = float(out[f"rv_{period}"].iloc[-1])

        self.assertAlmostEqual(observed_rv, round(expected_rv, 2), places=2)

        # 만약 과거 버그(252 연환산)였다면 값이 약 √(252/365) ≈ 0.831 배로
        # 줄어들 것이고, round(2) 를 감안해도 차이가 최소 0.1 이상 벌어진다.
        bug_rv = log_ret.tail(period).std(ddof=1) * math.sqrt(252) * 100
        self.assertGreater(
            observed_rv - bug_rv,
            0.1,
            msg="RV 가 여전히 252 연환산(또는 그와 구분되지 않는 값)으로 보임",
        )

    def test_annualize_table_uses_365_basis(self):
        # 각 TF 의 연환산 계수가 365 를 기반으로 하는지 확인.
        self.assertEqual(_RV_ANNUALIZE["1d"], 365)
        self.assertEqual(_RV_ANNUALIZE["4h"], 365 * 6)
        self.assertEqual(_RV_ANNUALIZE["1h"], 365 * 24)
        self.assertEqual(_RV_ANNUALIZE["15m"], 365 * 96)
        self.assertEqual(_RV_ANNUALIZE["5m"], 365 * 288)


if __name__ == "__main__":
    unittest.main()
