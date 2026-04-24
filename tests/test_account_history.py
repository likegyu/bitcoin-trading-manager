from __future__ import annotations

import tempfile
import unittest
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

from account_history import (
    MAX_ENTRIES,
    AccountContextTimeline,
    _snapshot_from_context,
)
from time_utils import start_of_kst_day


def _position(symbol: str = "BTCUSDC", side: str = "롱", notional: float = 10000.0) -> dict:
    return {
        "symbol": symbol,
        "side": side,
        "size": 1.0,
        "leverage": 10.0,
        "notional": notional,
    }


def _ctx(
    equity: float,
    cash_pnl: float,
    positions: list[dict] | None = None,
    wallet_balance: float | None = None,
    upnl: float | None = None,
) -> dict:
    position_list = list(positions or [])
    total_notional = sum(float(pos.get("notional") or 0.0) for pos in position_list)
    total_upnl = upnl if upnl is not None else 0.0
    if wallet_balance is None:
        wallet_balance = equity - total_upnl

    return {
        "wallet_balance": wallet_balance,
        "available_balance": wallet_balance * 0.7,
        "margin_balance": equity,
        "account_equity": equity,
        "daily_target_pct": 1.0,
        "daily_loss_limit_pct": -2.0,
        "today_cash_pnl": cash_pnl,
        "today_realized_pnl": cash_pnl,
        "today_funding_fee": 0.0,
        "today_commission_fee": 0.0,
        "today_total_pnl": cash_pnl,
        "today_total_mode": "cash",
        "today_total_label": "금일 현금손익",
        "today_pnl_pct": 0.0,
        "open_positions": position_list,
        "open_position_count": len(position_list),
        "open_position_notional": total_notional,
        "open_position_upnl": total_upnl,
        "effective_leverage": 10.0 if position_list else None,
        "risk_status": "active",
        "carryover_positions": [],
    }


class AccountHistoryDayAnchorTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.timeline = AccountContextTimeline(Path(self.temp_dir.name) / "account_history.jsonl")
        self.timeline._loaded = True
        self.timeline._entries = deque(maxlen=MAX_ENTRIES)
        self.cutoff = start_of_kst_day(datetime.now(timezone.utc)).astimezone(timezone.utc)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _append_snapshot(self, ctx: dict, observed_at: datetime):
        self.timeline._entries.append(_snapshot_from_context(ctx, observed_at=observed_at))

    def test_late_first_snapshot_uses_cash_mode(self):
        self._append_snapshot(
            _ctx(1000.0, -5.0, [_position()], upnl=20.0),
            self.cutoff - timedelta(hours=4),
        )

        current = _ctx(1070.0, 12.0, [_position()], upnl=55.0)
        self.timeline.observe(current)

        self.assertEqual(current["today_total_mode"], "cash")
        self.assertEqual(current["day_anchor_source"], "cash_fallback")
        self.assertIsNone(current["today_eval_pnl"])
        self.assertEqual(current["carryover_positions"], [])
        self.assertAlmostEqual(current["today_total_pnl"], 12.0, places=6)

        # cash-mode 분모는 wallet_balance - cash_pnl 이어야 함.
        # current: equity=1070, upnl=55 → wallet_balance=1015, cash_pnl=12
        # 올바른 분모 = 1015 - 12 = 1003, pct = 12/1003 * 100 ≈ 1.1964%
        # 과거 버그(current_equity - cash_pnl = 1058) 면 pct ≈ 1.1342% 가 나와야 실패함
        self.assertAlmostEqual(current["today_pnl_pct"], 12.0 / 1003.0 * 100, places=4)

    def test_recent_prev_close_can_anchor_evaluation(self):
        self._append_snapshot(
            _ctx(1000.0, -5.0, [_position()], upnl=20.0),
            self.cutoff - timedelta(minutes=10),
        )

        current = _ctx(1070.0, 12.0, [_position()], upnl=55.0)
        self.timeline.observe(current)

        self.assertEqual(current["today_total_mode"], "evaluation")
        self.assertEqual(current["day_anchor_source"], "prev_close")
        self.assertAlmostEqual(current["today_total_pnl"], 70.0, places=6)
        self.assertEqual(current["carryover_positions"], ["BTCUSDC 롱 $10,000.00"])

    def test_reopened_position_is_not_marked_as_carryover(self):
        self._append_snapshot(
            _ctx(1000.0, -5.0, [_position()], upnl=20.0),
            self.cutoff - timedelta(minutes=5),
        )
        self._append_snapshot(
            _ctx(1010.0, 0.0, [], upnl=0.0),
            self.cutoff + timedelta(minutes=5),
        )

        current = _ctx(1060.0, 8.0, [_position()], upnl=40.0)
        self.timeline.observe(current)

        self.assertEqual(current["today_total_mode"], "evaluation")
        self.assertEqual(current["day_anchor_source"], "day_start")
        self.assertEqual(current["carryover_positions"], [])

    def test_position_spanning_midnight_is_marked_as_carryover(self):
        self._append_snapshot(
            _ctx(1000.0, -5.0, [_position()], upnl=20.0),
            self.cutoff - timedelta(minutes=5),
        )
        self._append_snapshot(
            _ctx(1010.0, 0.0, [_position()], upnl=20.0),
            self.cutoff + timedelta(minutes=5),
        )

        current = _ctx(1060.0, 8.0, [_position()], upnl=40.0)
        self.timeline.observe(current)

        self.assertEqual(current["today_total_mode"], "evaluation")
        self.assertEqual(current["day_anchor_source"], "day_start")
        self.assertEqual(current["carryover_positions"], ["BTCUSDC 롱 $10,000.00"])


if __name__ == "__main__":
    unittest.main()
