# =============================================
# 자동매매 엔진
# =============================================
# 흐름:
#   분석 완료 → TradingSignal 판독
#   → 진입 조건 충족 여부 확인
#   → 기존 포지션 확인
#   → 포지션 사이징
#   → 시장가 진입 + 손절/익절 주문
#   → 매매 로그 기록
#
# 환경변수 / config 로 제어:
#   AUTO_TRADE_ENABLED   : 1|0       자동매매 ON/OFF
#   AUTO_TRADE_DRY_RUN   : 1|0       드라이런 모드 (주문 미전송)
#   AUTO_TRADE_MIN_CONFIDENCE : 65   최소 확신도
#   AUTO_TRADE_MIN_STRENGTH   : 2    최소 시그널 강도 (BUY=2, STRONG_BUY=3)
#   AUTO_TRADE_RISK_PCT       : 0.02 진입 당 계좌 리스크 비율 (2%)
#   AUTO_TRADE_SL_ATR_MULT    : 1.5  손절 = ATR × 배수
#   AUTO_TRADE_TP_RR          : 2.0  익절 = 손절폭 × Risk:Reward
#   AUTO_TRADE_MAX_LEVERAGE   : 5    레버리지 상한
#   AUTO_TRADE_COOLDOWN_MIN   : 30   연속 거래 쿨다운 (분)
#   AUTO_TRADE_FLIP_GUARD     : 1    반대 포지션 즉시 전환 방지
# =============================================
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

import config as _cfg
import trader as _trader

logger = logging.getLogger(__name__)

# ── 매매 로그 경로 ────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRADE_LOG_PATH   = os.path.join(_BASE_DIR, "data", "trade_log.jsonl")
AT_STATE_PATH    = os.path.join(_BASE_DIR, "data", "auto_trader_state.json")


# ══════════════════════════════════════════════
# 설정 헬퍼
# ══════════════════════════════════════════════

def _cfg_bool(name: str, default: bool) -> bool:
    val = os.getenv(name, str(int(getattr(_cfg, name, default)))).lower()
    return val not in ("0", "false", "no")

def _cfg_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(getattr(_cfg, name, default))))

def _cfg_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(getattr(_cfg, name, default))))


# ══════════════════════════════════════════════
# 상태 / 로그
# ══════════════════════════════════════════════

@dataclass
class TradeRecord:
    """단일 거래 기록."""
    ts:           str        # ISO 타임스탬프
    symbol:       str
    action:       str        # "OPEN_LONG" | "OPEN_SHORT" | "CLOSE" | "SKIP" | "ERROR"
    signal_en:    str        # "BUY" | "SELL" | "HOLD" …
    strength:     int
    confidence:   int
    entry_price:  float
    sl_price:     float
    tp_price:     float
    quantity:     float
    leverage:     int
    dry_run:      bool
    reason:       str        # skip/error 이유 또는 실행 요약
    order_ids:    list[int]  = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_trade_log(record: TradeRecord):
    try:
        os.makedirs(os.path.dirname(TRADE_LOG_PATH), exist_ok=True)
        with open(TRADE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.error("매매 로그 기록 실패: %s", exc)


def load_trade_log(limit: int = 50) -> list[dict]:
    """최근 매매 기록 반환."""
    try:
        if not os.path.exists(TRADE_LOG_PATH):
            return []
        with open(TRADE_LOG_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
        records = []
        for line in reversed(lines[-limit * 2:]):
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
        return records[:limit]
    except Exception:
        return []


# ══════════════════════════════════════════════
# AutoTrader 상태 (서버 재시작 후 복원)
# ══════════════════════════════════════════════

class _ATState:
    """자동매매 설정 상태 — 파일 기반 영속."""

    def __init__(self):
        self._lock = threading.Lock()
        self.enabled: bool = _cfg_bool("AUTO_TRADE_ENABLED", False)
        self.dry_run: bool = _cfg_bool("AUTO_TRADE_DRY_RUN", True)
        self._load()

    def _load(self):
        try:
            if os.path.exists(AT_STATE_PATH):
                with open(AT_STATE_PATH, "r", encoding="utf-8") as f:
                    d = json.load(f)
                self.enabled = bool(d.get("enabled", self.enabled))
                self.dry_run = bool(d.get("dry_run", self.dry_run))
        except Exception:
            pass

    def _save(self):
        try:
            os.makedirs(os.path.dirname(AT_STATE_PATH), exist_ok=True)
            with open(AT_STATE_PATH, "w", encoding="utf-8") as f:
                json.dump({"enabled": self.enabled, "dry_run": self.dry_run}, f)
        except Exception as exc:
            logger.warning("ATState 저장 실패: %s", exc)

    def update(self, enabled: bool, dry_run: bool):
        with self._lock:
            self.enabled = enabled
            self.dry_run = dry_run
            # config 의 DRY_RUN 도 동기화 (trader.py 가 참조)
            _cfg.AUTO_TRADE_DRY_RUN = dry_run
            self._save()

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "dry_run": self.dry_run,
            "min_confidence":  _cfg_int("AUTO_TRADE_MIN_CONFIDENCE", 65),
            "min_strength":    _cfg_int("AUTO_TRADE_MIN_STRENGTH", 2),
            "risk_pct":        _cfg_float("AUTO_TRADE_RISK_PCT", 0.02),
            "sl_atr_mult":     _cfg_float("AUTO_TRADE_SL_ATR_MULT", 1.5),
            "tp_rr":           _cfg_float("AUTO_TRADE_TP_RR", 2.0),
            "max_leverage":    _cfg_int("AUTO_TRADE_MAX_LEVERAGE", 5),
            "cooldown_min":    _cfg_int("AUTO_TRADE_COOLDOWN_MIN", 30),
        }


_at_state = _ATState()

# 마지막 거래 시각 (쿨다운)
_last_trade_ts: float = 0.0
_last_trade_lock = threading.Lock()


# ══════════════════════════════════════════════
# 진입 조건 판단
# ══════════════════════════════════════════════

def _is_cooldown_ok() -> tuple[bool, str]:
    cooldown_sec = _cfg_int("AUTO_TRADE_COOLDOWN_MIN", 30) * 60
    with _last_trade_lock:
        elapsed = time.time() - _last_trade_ts
    if elapsed < cooldown_sec:
        remain = int((cooldown_sec - elapsed) / 60)
        return False, f"쿨다운 중 — 다음 거래까지 약 {remain}분 남음"
    return True, ""


def _check_signal(trading_signal: dict) -> tuple[bool, str, str]:
    """
    TradingSignal dict 를 검사해 진입 가능 여부와 방향을 반환.
    Returns: (ok, direction, reason)
      direction: "LONG" | "SHORT" | ""
    """
    min_conf     = _cfg_int("AUTO_TRADE_MIN_CONFIDENCE", 65)
    min_strength = _cfg_int("AUTO_TRADE_MIN_STRENGTH", 2)

    signal_en  = trading_signal.get("signal_en", "HOLD")
    strength   = int(trading_signal.get("strength", 0))
    confidence = int(trading_signal.get("confidence", 0))
    aligned    = bool(trading_signal.get("judge_aligned", True))

    if abs(strength) < min_strength:
        return False, "", f"시그널 강도 부족 (strength={strength}, min={min_strength})"
    if confidence < min_conf:
        return False, "", f"확신도 부족 (confidence={confidence}, min={min_conf})"
    if not aligned:
        return False, "", "애널리스트-심판 방향 불일치 → 진입 보류"

    if strength > 0:
        return True, "LONG", f"{signal_en} / conf={confidence} / strength={strength}"
    else:
        return True, "SHORT", f"{signal_en} / conf={confidence} / strength={strength}"


def _calc_atr(trade_levels: dict, tf_preference: list[str] | None = None) -> float:
    """
    trade_levels 에 ATR 정보가 없을 경우 손절폭 fallback 사용.
    trade_levels 예: {"entry": 95000, "stop_loss": 94000, "take_profit": 97000}
    """
    # trade_levels 에서 직접 손절폭 계산
    entry = float(trade_levels.get("entry") or 0)
    sl    = float(trade_levels.get("stop_loss") or 0)
    if entry and sl:
        return abs(entry - sl)
    return 0.0


# ══════════════════════════════════════════════
# 포지션 사이징
# ══════════════════════════════════════════════

def _calc_position(
    direction: str,
    entry_price: float,
    atr: float,
    symbol: str,
) -> tuple[float, float, float, int, float, float]:
    """
    Returns: (quantity, sl_price, tp_price, leverage, sl_dist, tp_dist)
    """
    risk_pct   = _cfg_float("AUTO_TRADE_RISK_PCT", 0.02)
    sl_mult    = _cfg_float("AUTO_TRADE_SL_ATR_MULT", 1.5)
    tp_rr      = _cfg_float("AUTO_TRADE_TP_RR", 2.0)
    max_lev    = _cfg_int("AUTO_TRADE_MAX_LEVERAGE", 5)
    target_lev = min(getattr(_cfg, "DEFAULT_LEVERAGE", 3), max_lev)

    # 가용 잔고 조회
    try:
        balance = _trader.get_account_balance("USDT")
    except Exception:
        try:
            balance = _trader.get_account_balance("USDC")
        except Exception:
            balance = 0.0

    if balance <= 0:
        raise ValueError("계좌 잔고를 조회할 수 없습니다 (balance=0)")

    # ATR 기반 손절 거리 계산
    sl_dist = atr * sl_mult if atr > 0 else entry_price * 0.01   # fallback: 1%
    tp_dist = sl_dist * tp_rr

    if direction == "LONG":
        sl_price = entry_price - sl_dist
        tp_price = entry_price + tp_dist
    else:  # SHORT
        sl_price = entry_price + sl_dist
        tp_price = entry_price - tp_dist

    # 리스크 금액 기반 수량 계산
    risk_usdt = balance * risk_pct
    qty_raw   = risk_usdt / sl_dist if sl_dist > 0 else 0.0

    # stepSize 에 맞춰 내림
    step  = _trader.get_lot_step_size(symbol)
    qty   = _trader.round_qty(qty_raw, step)

    if qty <= 0:
        raise ValueError(f"계산된 수량이 0 (balance={balance:.2f}, sl_dist={sl_dist:.2f})")

    return qty, sl_price, tp_price, target_lev, sl_dist, tp_dist


# ══════════════════════════════════════════════
# 메인 진입 함수
# ══════════════════════════════════════════════

def execute_trade(payload: dict) -> TradeRecord:
    """
    분석 payload 를 받아 자동매매를 실행하고 TradeRecord 를 반환한다.

    Parameters
    ----------
    payload : dict
        server._build_payload 반환값 (trading_signal, trade_levels 등 포함)
    """
    symbol = getattr(_cfg, "DEFAULT_SYMBOL", "BTCUSDC")
    ts     = _now_iso()

    trading_signal: dict = payload.get("trading_signal") or {}
    trade_levels:   dict = payload.get("trade_levels") or {}

    # ── 0. 자동매매 활성화 확인 ──────────────────────
    if not _at_state.enabled:
        rec = TradeRecord(
            ts=ts, symbol=symbol, action="SKIP",
            signal_en=trading_signal.get("signal_en", ""),
            strength=trading_signal.get("strength", 0),
            confidence=trading_signal.get("confidence", 0),
            entry_price=0, sl_price=0, tp_price=0,
            quantity=0, leverage=0,
            dry_run=_at_state.dry_run,
            reason="자동매매 비활성화 상태",
        )
        _append_trade_log(rec)
        return rec

    # ── 1. 시그널 조건 확인 ──────────────────────────
    ok, direction, signal_reason = _check_signal(trading_signal)
    if not ok:
        rec = TradeRecord(
            ts=ts, symbol=symbol, action="SKIP",
            signal_en=trading_signal.get("signal_en", "HOLD"),
            strength=trading_signal.get("strength", 0),
            confidence=trading_signal.get("confidence", 0),
            entry_price=0, sl_price=0, tp_price=0,
            quantity=0, leverage=0,
            dry_run=_at_state.dry_run,
            reason=f"[조건 미충족] {signal_reason}",
        )
        _append_trade_log(rec)
        logger.info("자동매매 SKIP — %s", signal_reason)
        return rec

    # ── 2. 쿨다운 확인 ──────────────────────────────
    cd_ok, cd_reason = _is_cooldown_ok()
    if not cd_ok:
        rec = TradeRecord(
            ts=ts, symbol=symbol, action="SKIP",
            signal_en=trading_signal.get("signal_en", ""),
            strength=trading_signal.get("strength", 0),
            confidence=trading_signal.get("confidence", 0),
            entry_price=0, sl_price=0, tp_price=0,
            quantity=0, leverage=0,
            dry_run=_at_state.dry_run,
            reason=f"[쿨다운] {cd_reason}",
        )
        _append_trade_log(rec)
        logger.info("자동매매 SKIP — %s", cd_reason)
        return rec

    # ── 3. 현재 포지션 확인 ──────────────────────────
    try:
        existing = _trader.get_position(symbol)
    except Exception as exc:
        existing = None
        logger.warning("포지션 조회 실패 (진입 계속): %s", exc)

    if existing:
        amt = float(existing.get("positionAmt", 0))
        existing_dir = "LONG" if amt > 0 else "SHORT"
        flip_guard = _cfg_bool("AUTO_TRADE_FLIP_GUARD", True)

        if existing_dir == direction:
            rec = TradeRecord(
                ts=ts, symbol=symbol, action="SKIP",
                signal_en=trading_signal.get("signal_en", ""),
                strength=trading_signal.get("strength", 0),
                confidence=trading_signal.get("confidence", 0),
                entry_price=float(existing.get("entryPrice", 0)),
                sl_price=0, tp_price=0, quantity=abs(amt),
                leverage=int(float(existing.get("leverage", 1))),
                dry_run=_at_state.dry_run,
                reason=f"이미 {existing_dir} 포지션 보유 중 — 중복 진입 건너뜀",
            )
            _append_trade_log(rec)
            logger.info("자동매매 SKIP — 기존 %s 포지션 존재", existing_dir)
            return rec

        if flip_guard:
            rec = TradeRecord(
                ts=ts, symbol=symbol, action="SKIP",
                signal_en=trading_signal.get("signal_en", ""),
                strength=trading_signal.get("strength", 0),
                confidence=trading_signal.get("confidence", 0),
                entry_price=float(existing.get("entryPrice", 0)),
                sl_price=0, tp_price=0, quantity=abs(amt),
                leverage=int(float(existing.get("leverage", 1))),
                dry_run=_at_state.dry_run,
                reason=(
                    f"반전 방지(flip_guard): 기존 {existing_dir} → 신규 {direction} 방향 전환 보류. "
                    "포지션을 직접 청산 후 다음 분석을 기다리세요."
                ),
            )
            _append_trade_log(rec)
            logger.info("자동매매 SKIP — flip_guard: %s → %s", existing_dir, direction)
            return rec

        # flip_guard 비활성 시 기존 포지션 청산 후 진입
        logger.info("기존 %s 포지션 청산 후 %s 진입", existing_dir, direction)
        try:
            _trader.cancel_all_open_orders(symbol)
            _trader.close_position(symbol)
        except Exception as exc:
            logger.error("기존 포지션 청산 실패: %s", exc)

    # ── 4. 진입가 결정 ───────────────────────────────
    current_price = float(payload.get("current_price") or payload.get("price") or 0)
    entry_hint    = float(trade_levels.get("entry") or 0)
    entry_price   = entry_hint if entry_hint > 0 else current_price
    if entry_price <= 0:
        rec = TradeRecord(
            ts=ts, symbol=symbol, action="ERROR",
            signal_en=trading_signal.get("signal_en", ""),
            strength=trading_signal.get("strength", 0),
            confidence=trading_signal.get("confidence", 0),
            entry_price=0, sl_price=0, tp_price=0,
            quantity=0, leverage=0,
            dry_run=_at_state.dry_run,
            reason="진입가를 결정할 수 없음 (price=0)",
        )
        _append_trade_log(rec)
        return rec

    # ── 5. ATR / 포지션 사이징 ───────────────────────
    try:
        atr = _calc_atr(trade_levels)
        qty, sl_price, tp_price, leverage, sl_dist, tp_dist = _calc_position(
            direction, entry_price, atr, symbol
        )
    except Exception as exc:
        rec = TradeRecord(
            ts=ts, symbol=symbol, action="ERROR",
            signal_en=trading_signal.get("signal_en", ""),
            strength=trading_signal.get("strength", 0),
            confidence=trading_signal.get("confidence", 0),
            entry_price=entry_price, sl_price=0, tp_price=0,
            quantity=0, leverage=0,
            dry_run=_at_state.dry_run,
            reason=f"포지션 사이징 실패: {exc}",
        )
        _append_trade_log(rec)
        logger.error("포지션 사이징 실패: %s", exc)
        return rec

    # ── 6. 레버리지 설정 ─────────────────────────────
    try:
        _trader.set_leverage(symbol, leverage)
    except Exception as exc:
        logger.warning("레버리지 설정 실패 (계속 진행): %s", exc)

    # ── 7. 시장가 진입 ───────────────────────────────
    entry_side = "BUY" if direction == "LONG" else "SELL"
    order_ids: list[int] = []

    try:
        entry_result = _trader.place_market_order(symbol, entry_side, qty)
        oid = entry_result.get("orderId", -1)
        if oid != -1:
            order_ids.append(oid)
        actual_entry = float(entry_result.get("avgPrice") or entry_result.get("price") or entry_price)
    except Exception as exc:
        rec = TradeRecord(
            ts=ts, symbol=symbol, action="ERROR",
            signal_en=trading_signal.get("signal_en", ""),
            strength=trading_signal.get("strength", 0),
            confidence=trading_signal.get("confidence", 0),
            entry_price=entry_price, sl_price=sl_price, tp_price=tp_price,
            quantity=qty, leverage=leverage,
            dry_run=_at_state.dry_run,
            reason=f"시장가 주문 실패: {exc}",
        )
        _append_trade_log(rec)
        logger.error("시장가 주문 실패: %s", exc)
        return rec

    # ── 8. 손절 / 익절 주문 ──────────────────────────
    sl_side = "SELL" if direction == "LONG" else "BUY"
    try:
        sl_result = _trader.place_stop_order(
            symbol, sl_side, qty, sl_price, order_type="STOP_MARKET"
        )
        oid = sl_result.get("orderId", -1)
        if oid != -1:
            order_ids.append(oid)
    except Exception as exc:
        logger.warning("손절 주문 실패 (계속 진행): %s", exc)

    try:
        tp_result = _trader.place_stop_order(
            symbol, sl_side, qty, tp_price, order_type="TAKE_PROFIT_MARKET"
        )
        oid = tp_result.get("orderId", -1)
        if oid != -1:
            order_ids.append(oid)
    except Exception as exc:
        logger.warning("익절 주문 실패 (계속 진행): %s", exc)

    # ── 9. 쿨다운 타이머 갱신 ───────────────────────
    with _last_trade_lock:
        global _last_trade_ts
        _last_trade_ts = time.time()

    action = "OPEN_LONG" if direction == "LONG" else "OPEN_SHORT"
    dry = _at_state.dry_run or _trader._dry_run()
    rec = TradeRecord(
        ts=ts, symbol=symbol, action=action,
        signal_en=trading_signal.get("signal_en", ""),
        strength=trading_signal.get("strength", 0),
        confidence=trading_signal.get("confidence", 0),
        entry_price=actual_entry,
        sl_price=round(sl_price, 2),
        tp_price=round(tp_price, 2),
        quantity=qty, leverage=leverage,
        dry_run=dry,
        reason=signal_reason,
        order_ids=order_ids,
    )
    _append_trade_log(rec)
    logger.info(
        "✅ 자동매매 실행 [%s] %s qty=%.6f entry=%.2f sl=%.2f tp=%.2f lev=x%d dry=%s",
        action, symbol, qty, actual_entry, sl_price, tp_price, leverage, dry
    )
    return rec


# ══════════════════════════════════════════════
# 공개 인터페이스
# ══════════════════════════════════════════════

def get_status() -> dict:
    """자동매매 현재 설정 + 최근 10건 기록 반환."""
    return {
        **_at_state.to_dict(),
        "last_trades": load_trade_log(limit=10),
    }


def set_config(enabled: bool, dry_run: bool) -> dict:
    """자동매매 ON/OFF 및 드라이런 설정 변경."""
    _at_state.update(enabled=enabled, dry_run=dry_run)
    mode = "DRY-RUN" if dry_run else "실거래"
    state = "활성화" if enabled else "비활성화"
    logger.info("자동매매 설정 변경 → %s (%s)", state, mode)
    return _at_state.to_dict()
