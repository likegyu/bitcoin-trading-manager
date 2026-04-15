# =============================================
# 자동매매 엔진 v2
# =============================================
# 새로 추가된 기능:
#   1. Claude 권장 레버리지 적용 (AUTO_TRADE_CLAUDE_LEVERAGE)
#   2. 확신도 기반 동적 포지션 사이징 (AUTO_TRADE_DYNAMIC_SIZING)
#   3. 신호 반전 시 자동 청산 후 재진입 (AUTO_TRADE_REVERSAL_ENABLED)
#
# 안전장치:
#   - 레버리지 상한 강제 적용 (AUTO_TRADE_MAX_LEVERAGE)
#   - 최소 보유 시간 체크 — 휩소 방지 (AUTO_TRADE_REVERSAL_MIN_HOLD)
#   - 반전 확신도 기준을 진입보다 높게 설정 (AUTO_TRADE_REVERSAL_MIN_CONF)
#   - 하루 최대 반전 횟수 제한 (AUTO_TRADE_REVERSAL_MAX_PER_DAY)
#   - flip_guard: 반전 기능 비활성화 시에도 즉시 전환 차단
#   - 쿨다운: 진입 간 최소 간격 (반전 후 재진입은 쿨다운 우회)
# =============================================
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

import config as _cfg
import trader as _trader

logger = logging.getLogger(__name__)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRADE_LOG_PATH = os.path.join(_BASE_DIR, "data", "trade_log.jsonl")
AT_STATE_PATH  = os.path.join(_BASE_DIR, "data", "auto_trader_state.json")


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
# 매매 기록
# ══════════════════════════════════════════════

@dataclass
class TradeRecord:
    ts:           str
    symbol:       str
    action:       str    # OPEN_LONG | OPEN_SHORT | REVERSAL_LONG | REVERSAL_SHORT | CLOSE | SKIP | ERROR
    signal_en:    str
    strength:     int
    confidence:   int
    entry_price:  float
    sl_price:     float
    tp_price:     float
    quantity:     float
    leverage:     int
    dry_run:      bool
    reason:       str
    order_ids:    list[int] = field(default_factory=list)

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
# 상태 관리
# ══════════════════════════════════════════════

class _ATState:
    """자동매매 설정 + 반전 횟수 추적 — 파일 기반 영속."""

    def __init__(self):
        self._lock = threading.Lock()
        self.enabled: bool = _cfg_bool("AUTO_TRADE_ENABLED", False)
        self.dry_run: bool = _cfg_bool("AUTO_TRADE_DRY_RUN", True)
        self._reversal_date: str = ""
        self._reversal_count: int = 0
        self._load()

    def _load(self):
        try:
            if os.path.exists(AT_STATE_PATH):
                with open(AT_STATE_PATH, "r", encoding="utf-8") as f:
                    d = json.load(f)
                self.enabled = bool(d.get("enabled", self.enabled))
                self.dry_run = bool(d.get("dry_run", self.dry_run))
                self._reversal_date  = d.get("reversal_date", "")
                self._reversal_count = int(d.get("reversal_count", 0))
        except Exception:
            pass

    def _save(self):
        try:
            os.makedirs(os.path.dirname(AT_STATE_PATH), exist_ok=True)
            with open(AT_STATE_PATH, "w", encoding="utf-8") as f:
                json.dump({
                    "enabled":        self.enabled,
                    "dry_run":        self.dry_run,
                    "reversal_date":  self._reversal_date,
                    "reversal_count": self._reversal_count,
                }, f)
        except Exception as exc:
            logger.warning("ATState 저장 실패: %s", exc)

    def update(self, enabled: bool, dry_run: bool):
        with self._lock:
            self.enabled = enabled
            self.dry_run = dry_run
            _cfg.AUTO_TRADE_DRY_RUN = dry_run
            self._save()

    # ── 반전 카운터 ──────────────────────────────
    def reversal_count_today(self) -> int:
        today = datetime.now(timezone.utc).date().isoformat()
        with self._lock:
            if self._reversal_date != today:
                self._reversal_date  = today
                self._reversal_count = 0
            return self._reversal_count

    def increment_reversal(self):
        today = datetime.now(timezone.utc).date().isoformat()
        with self._lock:
            if self._reversal_date != today:
                self._reversal_date  = today
                self._reversal_count = 0
            self._reversal_count += 1
            self._save()

    def to_dict(self) -> dict:
        return {
            "enabled":              self.enabled,
            "dry_run":              self.dry_run,
            "min_confidence":       _cfg_int("AUTO_TRADE_MIN_CONFIDENCE", 60),
            "min_strength":         _cfg_int("AUTO_TRADE_MIN_STRENGTH", 2),
            "risk_pct":             _cfg_float("AUTO_TRADE_RISK_PCT", 0.04),
            "risk_min_pct":         _cfg_float("AUTO_TRADE_RISK_MIN_PCT", 0.02),
            "dynamic_sizing":       _cfg_bool("AUTO_TRADE_DYNAMIC_SIZING", True),
            "sl_atr_mult":          _cfg_float("AUTO_TRADE_SL_ATR_MULT", 2.0),
            "tp_rr":                _cfg_float("AUTO_TRADE_TP_RR", 1.5),
            "max_leverage":         _cfg_int("AUTO_TRADE_MAX_LEVERAGE", 7),
            "claude_leverage":      _cfg_bool("AUTO_TRADE_CLAUDE_LEVERAGE", True),
            "cooldown_min":         _cfg_int("AUTO_TRADE_COOLDOWN_MIN", 120),
            "reversal_enabled":     _cfg_bool("AUTO_TRADE_REVERSAL_ENABLED", True),
            "reversal_min_hold":    _cfg_int("AUTO_TRADE_REVERSAL_MIN_HOLD", 30),
            "reversal_min_conf":    _cfg_int("AUTO_TRADE_REVERSAL_MIN_CONF", 67),
            "reversal_max_per_day": _cfg_int("AUTO_TRADE_REVERSAL_MAX_PER_DAY", 2),
            "reversal_count_today": self.reversal_count_today(),
        }


_at_state = _ATState()

_last_trade_ts: float = 0.0
_last_trade_lock = threading.Lock()


# ══════════════════════════════════════════════
# ① 신호 조건 판단
# ══════════════════════════════════════════════

def _check_signal(trading_signal: dict) -> tuple[bool, str, str]:
    """
    Returns: (ok, direction, reason)
    direction: "LONG" | "SHORT" | ""
    """
    min_conf     = _cfg_int("AUTO_TRADE_MIN_CONFIDENCE", 60)
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

    direction = "LONG" if strength > 0 else "SHORT"
    return True, direction, f"{signal_en} / conf={confidence} / strength={strength}"


# ══════════════════════════════════════════════
# ② 쿨다운
# ══════════════════════════════════════════════

def _is_cooldown_ok() -> tuple[bool, str]:
    cooldown_sec = _cfg_int("AUTO_TRADE_COOLDOWN_MIN", 120) * 60
    with _last_trade_lock:
        elapsed = time.time() - _last_trade_ts
    if elapsed < cooldown_sec:
        remain = int((cooldown_sec - elapsed) / 60)
        return False, f"쿨다운 중 — 다음 진입까지 약 {remain}분 남음"
    return True, ""


def _update_cooldown():
    global _last_trade_ts
    with _last_trade_lock:
        _last_trade_ts = time.time()


# ══════════════════════════════════════════════
# ③ 반전 매매 판단 (안전장치 포함)
# ══════════════════════════════════════════════

def _should_reverse(existing: dict, direction: str, trading_signal: dict) -> tuple[bool, str]:
    """
    기존 포지션과 반대 방향 신호 시 반전(청산 후 재진입) 가능 여부 판단.

    안전장치:
      A. reversal_enabled 플래그
      B. 최소 보유 시간 (updateTime 기반) — 휩소 방지
      C. 반전 확신도 ≥ reversal_min_conf (진입 기준보다 높음)
      D. 하루 최대 반전 횟수
    """
    if not _cfg_bool("AUTO_TRADE_REVERSAL_ENABLED", True):
        return False, "반전 매매 비활성 (AUTO_TRADE_REVERSAL_ENABLED=0)"

    # A. 최소 보유 시간 ────────────────────────
    min_hold_min = _cfg_int("AUTO_TRADE_REVERSAL_MIN_HOLD", 30)
    update_time_ms = int(existing.get("updateTime", 0))
    hold_min = 0.0
    if update_time_ms > 0:
        hold_min = (time.time() * 1000 - update_time_ms) / 60_000
        if hold_min < min_hold_min:
            remain = int(min_hold_min - hold_min)
            return False, (
                f"[휩소 방지] 최소 보유 {min_hold_min}분 미충족 "
                f"(현재 {hold_min:.0f}분 보유, {remain}분 더 보유 필요)"
            )

    # B. 반전 확신도 ───────────────────────────
    min_conf = _cfg_int("AUTO_TRADE_REVERSAL_MIN_CONF", 67)
    confidence = int(trading_signal.get("confidence", 0))
    if confidence < min_conf:
        return False, f"[반전 확신도 부족] {confidence} < {min_conf}"

    # C. Judge 정렬 필수 ───────────────────────
    if not bool(trading_signal.get("judge_aligned", True)):
        return False, "[반전 차단] 반전 신호에서 애널리스트-심판 불일치 — 반전 보류"

    # D. 하루 최대 반전 횟수 ───────────────────
    max_per_day = _cfg_int("AUTO_TRADE_REVERSAL_MAX_PER_DAY", 2)
    count = _at_state.reversal_count_today()
    if count >= max_per_day:
        return False, f"[일일 한도] 하루 최대 반전 횟수 도달 ({count}/{max_per_day})"

    existing_dir = "LONG" if float(existing.get("positionAmt", 0)) > 0 else "SHORT"
    return True, (
        f"반전 조건 충족 — {existing_dir}→{direction} / "
        f"conf={confidence} / hold={hold_min:.0f}분 / 오늘 반전 {count+1}/{max_per_day}회"
    )


# ══════════════════════════════════════════════
# ④ 동적 포지션 사이징
# ══════════════════════════════════════════════

def _dynamic_risk_pct(confidence: int) -> float:
    """
    확신도에 비례해 리스크 비율을 선형 보간.

    확신도 MIN_CONF(60) → RISK_MIN_PCT(2%)
    확신도 90%+         → RISK_PCT(4%)

    AUTO_TRADE_DYNAMIC_SIZING=0 이면 RISK_PCT 고정 반환.
    """
    if not _cfg_bool("AUTO_TRADE_DYNAMIC_SIZING", True):
        return _cfg_float("AUTO_TRADE_RISK_PCT", 0.04)

    min_conf = _cfg_int("AUTO_TRADE_MIN_CONFIDENCE", 60)
    max_conf = 90          # 90% 이상은 최대 리스크로 처리
    min_risk = _cfg_float("AUTO_TRADE_RISK_MIN_PCT", 0.02)
    max_risk = _cfg_float("AUTO_TRADE_RISK_PCT", 0.04)

    ratio = (confidence - min_conf) / max(max_conf - min_conf, 1)
    ratio = max(0.0, min(1.0, ratio))
    result = min_risk + (max_risk - min_risk) * ratio
    logger.debug("동적 리스크: conf=%d → %.1f%%", confidence, result * 100)
    return result


# ══════════════════════════════════════════════
# ⑤ Claude 권장 레버리지 적용
# ══════════════════════════════════════════════

def _resolve_leverage(claude_leverage: Optional[int]) -> int:
    """
    최종 레버리지 결정.

    우선순위:
      1) AUTO_TRADE_CLAUDE_LEVERAGE=1 이고 Claude가 유효한 값을 제안 → 상한 클리핑 후 사용
      2) 그 외 → DEFAULT_LEVERAGE (상한 클리핑)

    안전장치: MAX_LEVERAGE 이하로 반드시 클리핑.
    """
    max_lev     = _cfg_int("AUTO_TRADE_MAX_LEVERAGE", 7)
    default_lev = getattr(_cfg, "DEFAULT_LEVERAGE", 3)

    if _cfg_bool("AUTO_TRADE_CLAUDE_LEVERAGE", True) and claude_leverage is not None:
        lev = max(1, min(claude_leverage, max_lev))
        if lev != claude_leverage:
            logger.info("Claude 권장 레버리지 %dx → 상한 클리핑 → %dx", claude_leverage, lev)
        else:
            logger.info("Claude 권장 레버리지 적용: %dx", lev)
        return lev

    return min(default_lev, max_lev)


# ══════════════════════════════════════════════
# ⑥ ATR 계산
# ══════════════════════════════════════════════

def _calc_atr(trade_levels: dict) -> float:
    """
    Claude 분석 결과의 trade_levels 에서 손절 거리(ATR 대용)를 계산.
    parse_trade_levels() 반환 키: "entry", "stop" ("stop_loss" 폴백)
    """
    entry = float(trade_levels.get("entry") or 0)
    sl    = float(trade_levels.get("stop") or trade_levels.get("stop_loss") or 0)
    if entry and sl:
        return abs(entry - sl)
    return 0.0


# ══════════════════════════════════════════════
# ⑦ 포지션 사이징
# ══════════════════════════════════════════════

def _calc_position(
    direction: str,
    entry_price: float,
    atr: float,
    symbol: str,
    confidence: int = 70,
    claude_leverage: Optional[int] = None,
) -> tuple[float, float, float, int, float, float]:
    """
    Returns: (quantity, sl_price, tp_price, leverage, sl_dist, tp_dist)
    """
    risk_pct  = _dynamic_risk_pct(confidence)     # ← 확신도 기반 동적 비율
    sl_mult   = _cfg_float("AUTO_TRADE_SL_ATR_MULT", 2.0)
    tp_rr     = _cfg_float("AUTO_TRADE_TP_RR", 1.5)
    leverage  = _resolve_leverage(claude_leverage) # ← Claude 권장 레버리지

    # 가용 잔고 조회
    is_dry = _cfg_bool("AUTO_TRADE_DRY_RUN", True)
    balance = 0.0

    if is_dry:
        # ── 드라이런: 항상 가상 잔고만 사용, 실계좌 조회 안 함 ──
        try:
            import backtester as _bt
            balance = _bt.get_dry_balance()
        except Exception:
            pass
        if balance <= 0:
            # 파일 없으면 env 초기값으로 생성
            balance = float(os.getenv("AUTO_TRADE_DRY_RUN_BALANCE", "10000"))
            try:
                _bt.reset_dry_balance(balance)
            except Exception:
                pass
        logger.info("드라이런 가상 잔고: $%.2f", balance)
    else:
        # ── 실매매: 실계좌 잔고 조회 ──
        try:
            balance = _trader.get_account_balance("USDT")
        except Exception:
            pass
        if balance <= 0:
            try:
                balance = _trader.get_account_balance("USDC")
            except Exception:
                pass
        if balance <= 0:
            raise ValueError("계좌 잔고를 조회할 수 없습니다 (balance=0)")

    # 손절/익절 거리 계산
    sl_dist = atr * sl_mult if atr > 0 else entry_price * 0.01  # fallback: 1%
    tp_dist = sl_dist * tp_rr

    if direction == "LONG":
        sl_price = entry_price - sl_dist
        tp_price = entry_price + tp_dist
    else:
        sl_price = entry_price + sl_dist
        tp_price = entry_price - tp_dist

    # 리스크 기반 수량
    risk_usdt = balance * risk_pct
    qty_raw   = risk_usdt / sl_dist if sl_dist > 0 else 0.0

    step = _trader.get_lot_step_size(symbol)
    qty  = _trader.round_qty(qty_raw, step)

    if qty <= 0:
        raise ValueError(
            f"계산된 수량이 0 — "
            f"balance={balance:.2f}, risk={risk_pct*100:.1f}%, sl_dist={sl_dist:.2f}"
        )

    logger.info(
        "포지션 사이징 — dir=%s conf=%d risk=%.1f%% balance=%.2f "
        "sl_dist=%.2f qty=%.6f lev=%dx",
        direction, confidence, risk_pct * 100, balance, sl_dist, qty, leverage,
    )
    return qty, sl_price, tp_price, leverage, sl_dist, tp_dist


# ══════════════════════════════════════════════
# ⑧ 메인 실행 함수
# ══════════════════════════════════════════════

def execute_trade(payload: dict) -> TradeRecord:
    """
    분석 payload를 받아 자동매매를 실행한다.

    흐름:
      0. 활성화 확인
      1. 시그널 조건 판단
      2. 현재 포지션 조회
      3. 기존 포지션 처리
         ├─ 같은 방향 → SKIP (중복 방지)
         └─ 반대 방향 → 반전 조건 판단
            ├─ 반전 가능 → 청산 후 재진입 (쿨다운 우회)
            └─ 반전 불가 → SKIP (flip_guard)
      4. 신규 진입 (쿨다운 확인)
      5. 포지션 사이징 + 레버리지 설정
      6. 시장가 진입 + 손절/익절 주문
      7. 기록
    """
    symbol = getattr(_cfg, "DEFAULT_SYMBOL", "BTCUSDC")
    ts     = _now_iso()

    trading_signal: dict    = payload.get("trading_signal") or {}
    trade_levels:   dict    = payload.get("trade_levels") or {}
    claude_leverage: Optional[int] = payload.get("claude_leverage")
    confidence: int         = int(trading_signal.get("confidence", 50))

    def _skip(reason: str) -> TradeRecord:
        rec = TradeRecord(
            ts=ts, symbol=symbol, action="SKIP",
            signal_en=trading_signal.get("signal_en", ""),
            strength=trading_signal.get("strength", 0),
            confidence=confidence,
            entry_price=0, sl_price=0, tp_price=0,
            quantity=0, leverage=0,
            dry_run=_at_state.dry_run,
            reason=reason,
        )
        _append_trade_log(rec)
        logger.info("자동매매 SKIP — %s", reason)
        return rec

    def _error(reason: str, entry=0.0, sl=0.0, tp=0.0, qty=0.0, lev=0) -> TradeRecord:
        rec = TradeRecord(
            ts=ts, symbol=symbol, action="ERROR",
            signal_en=trading_signal.get("signal_en", ""),
            strength=trading_signal.get("strength", 0),
            confidence=confidence,
            entry_price=entry, sl_price=sl, tp_price=tp,
            quantity=qty, leverage=lev,
            dry_run=_at_state.dry_run,
            reason=reason,
        )
        _append_trade_log(rec)
        logger.error("자동매매 ERROR — %s", reason)
        return rec

    # ── 0. 활성화 확인 ───────────────────────────────
    if not _at_state.enabled:
        return _skip("자동매매 비활성화 상태")

    # ── 1. 시그널 조건 판단 ──────────────────────────
    ok, direction, signal_reason = _check_signal(trading_signal)
    if not ok:
        return _skip(f"[조건 미충족] {signal_reason}")

    # ── 2. 현재 포지션 / 대기 조회 ──────────────────────
    is_dry = _cfg_bool("AUTO_TRADE_DRY_RUN", True) or _at_state.dry_run
    existing = None          # 실매매 포지션 (Binance 형식)
    dry_live = None          # 드라이런 포지션/PENDING (backtester 형식)

    if is_dry:
        try:
            import backtester as _bt
            dry_live = _bt.get_live_position()
        except Exception as exc:
            logger.warning("드라이런 포지션 조회 실패: %s", exc)
    else:
        try:
            existing = _trader.get_position(symbol)
        except Exception as exc:
            logger.warning("포지션 조회 실패 (신규 진입으로 계속): %s", exc)

    bypass_cooldown = False   # 반전 재진입 시 쿨다운 우회 플래그
    current_price = float(payload.get("current_price") or payload.get("price") or 0)

    # ── 3. 기존 포지션 처리 ──────────────────────────
    if is_dry and dry_live:
        live_status = dry_live.get("status")       # "open" | "pending"
        live_dir    = dry_live.get("direction")     # "LONG" | "SHORT"

        if live_status == "pending":
            # PENDING 중 → 같은 방향이면 SKIP, 반대 방향이면 기존 PENDING 취소 후 새 PENDING
            if live_dir == direction:
                return _skip(f"이미 {live_dir} 진입 대기(PENDING) 중 — 중복 건너뜀")
            else:
                # 반대 방향 신호 → 기존 PENDING 취소
                import backtester as _bt
                cancel_rec = TradeRecord(
                    ts=ts, symbol=symbol, action="CANCEL_PENDING",
                    signal_en=trading_signal.get("signal_en", ""),
                    strength=trading_signal.get("strength", 0),
                    confidence=confidence,
                    entry_price=0, sl_price=0, tp_price=0,
                    quantity=0, leverage=0,
                    dry_run=True,
                    reason=f"[반대 신호로 취소] {live_dir}→{direction}",
                )
                _append_trade_log(cancel_rec)
                logger.info("[DRY-RUN] 기존 %s PENDING 취소 → 새 %s PENDING 생성", live_dir, direction)
                bypass_cooldown = True   # 취소-재진입은 쿨다운 면제

        elif live_status == "open":
            # OPEN 포지션 → 기존 반전 로직 적용
            try:
                from datetime import datetime as _dt2, timezone as _tz2
                _ts_dt = _dt2.fromisoformat(dry_live["ts"].replace("Z", "+00:00"))
                _update_ms = int(_ts_dt.timestamp() * 1000)
            except Exception:
                _update_ms = 0
            existing = {
                "positionAmt": dry_live["quantity"] if live_dir == "LONG" else -dry_live["quantity"],
                "entryPrice":  dry_live["entry_price"],
                "updateTime":  _update_ms,
            }

    if existing:
        amt          = float(existing.get("positionAmt", 0))
        existing_dir = "LONG" if amt > 0 else "SHORT"

        if existing_dir == direction:
            return _skip(f"이미 {existing_dir} 포지션 보유 중 — 중복 진입 건너뜀")

        can_reverse, rev_reason = _should_reverse(existing, direction, trading_signal)
        if not can_reverse:
            return _skip(f"[반전 차단] {rev_reason}")

        logger.info("반전 실행: %s → %s / %s", existing_dir, direction, rev_reason)
        if is_dry:
            close_rec = TradeRecord(
                ts=ts, symbol=symbol, action="CLOSE",
                signal_en=trading_signal.get("signal_en", ""),
                strength=trading_signal.get("strength", 0),
                confidence=confidence,
                entry_price=current_price, sl_price=0, tp_price=0,
                quantity=0, leverage=0,
                dry_run=True,
                reason=f"[드라이런 반전 청산] {rev_reason}",
            )
            _append_trade_log(close_rec)
            logger.info("[DRY-RUN] 반전 청산 기록 완료")
        else:
            try:
                _trader.cancel_all_open_orders(symbol)
            except Exception as exc:
                logger.warning("미결 주문 취소 실패 (계속 진행): %s", exc)
            try:
                _trader.close_position(symbol)
            except Exception as exc:
                return _error(f"반전 청산 실패 — 재진입 중단: {exc}")

        _at_state.increment_reversal()
        bypass_cooldown = True

    # ── 4. 쿨다운 확인 ──────────────────────────────
    if not bypass_cooldown:
        cd_ok, cd_reason = _is_cooldown_ok()
        if not cd_ok:
            return _skip(f"[쿨다운] {cd_reason}")

    # ── 5. 진입가 결정 ───────────────────────────────
    entry_hint  = float(trade_levels.get("entry") or 0)
    # 드라이런: Claude 제안 진입가 우선 (지정가 시뮬레이션), 없으면 현재가 (즉시 체결)
    # 실매매:  Claude 제안 진입가 우선, 없으면 현재가
    entry_price = entry_hint if entry_hint > 0 else current_price
    if entry_price <= 0:
        return _error("진입가를 결정할 수 없음 (price=0)")

    # ── 6. 포지션 사이징 ─────────────────────────────
    try:
        atr = _calc_atr(trade_levels)
        qty, sl_price, tp_price, leverage, sl_dist, tp_dist = _calc_position(
            direction, entry_price, atr, symbol,
            confidence=confidence,
            claude_leverage=claude_leverage,
        )
    except Exception as exc:
        return _error(f"포지션 사이징 실패: {exc}")

    # ── 7. 드라이런: PENDING 기록 후 종료 ────────────
    if is_dry:
        pending_action = f"PENDING_{direction}"
        _update_cooldown()
        rec = TradeRecord(
            ts=ts, symbol=symbol, action=pending_action,
            signal_en=trading_signal.get("signal_en", ""),
            strength=trading_signal.get("strength", 0),
            confidence=confidence,
            entry_price=entry_price,
            sl_price=round(sl_price, 2),
            tp_price=round(tp_price, 2),
            quantity=qty, leverage=leverage,
            dry_run=True,
            reason=f"[지정가 대기] {signal_reason} / entry={entry_price:.2f}",
        )
        _append_trade_log(rec)
        logger.info(
            "⏳ [%s] %s entry=%.2f sl=%.2f tp=%.2f lev=x%d — 체결 대기 중",
            pending_action, symbol, entry_price, sl_price, tp_price, leverage,
        )
        return rec

    # ── 8. 실매매: 레버리지 설정 + 시장가 진입 ────────
    try:
        _trader.set_leverage(symbol, leverage)
    except Exception as exc:
        logger.warning("레버리지 설정 실패 (계속 진행): %s", exc)

    entry_side = "BUY" if direction == "LONG" else "SELL"
    order_ids: list[int] = []

    try:
        entry_result = _trader.place_market_order(symbol, entry_side, qty)
        oid = entry_result.get("orderId", -1)
        if oid != -1:
            order_ids.append(oid)
        actual_entry = float(
            entry_result.get("avgPrice") or entry_result.get("price") or entry_price
        )
    except Exception as exc:
        return _error(
            f"시장가 주문 실패: {exc}",
            entry=entry_price, sl=sl_price, tp=tp_price, qty=qty, lev=leverage,
        )

    # ── 9. 손절 / 익절 주문 ──────────────────────────
    sl_side = "SELL" if direction == "LONG" else "BUY"
    try:
        sl_res = _trader.place_stop_order(symbol, sl_side, qty, sl_price, "STOP_MARKET")
        oid = sl_res.get("orderId", -1)
        if oid != -1:
            order_ids.append(oid)
    except Exception as exc:
        logger.warning("손절 주문 실패 (포지션은 유지): %s", exc)

    try:
        tp_res = _trader.place_stop_order(symbol, sl_side, qty, tp_price, "TAKE_PROFIT_MARKET")
        oid = tp_res.get("orderId", -1)
        if oid != -1:
            order_ids.append(oid)
    except Exception as exc:
        logger.warning("익절 주문 실패 (포지션은 유지): %s", exc)

    # ── 10. 쿨다운 갱신 ─────────────────────────────
    _update_cooldown()

    action = (
        f"REVERSAL_{direction}" if (bypass_cooldown and existing)
        else f"OPEN_{direction}"
    )
    dry = False  # 실매매 경로 (드라이런은 step 7에서 이미 반환됨)

    rec = TradeRecord(
        ts=ts, symbol=symbol, action=action,
        signal_en=trading_signal.get("signal_en", ""),
        strength=trading_signal.get("strength", 0),
        confidence=confidence,
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
        "✅ [%s] %s qty=%.6f entry=%.2f sl=%.2f tp=%.2f lev=x%d "
        "risk=%.1f%% claude_lev=%s dry=%s",
        action, symbol, qty, actual_entry, sl_price, tp_price, leverage,
        _dynamic_risk_pct(confidence) * 100,
        claude_leverage, dry,
    )
    return rec


# ══════════════════════════════════════════════
# 공개 인터페이스
# ══════════════════════════════════════════════

def get_status() -> dict:
    return {
        **_at_state.to_dict(),
        "last_trades": load_trade_log(limit=10),
    }


def set_config(enabled: bool, dry_run: bool) -> dict:
    _at_state.update(enabled=enabled, dry_run=dry_run)
    mode  = "DRY-RUN" if dry_run else "실거래"
    state = "활성화" if enabled else "비활성화"
    logger.info("자동매매 설정 변경 → %s (%s)", state, mode)
    return _at_state.to_dict()
