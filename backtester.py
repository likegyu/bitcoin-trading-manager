# =============================================
# 백테스터 — 드라이런/실매매 로그 기반 손익 검증
# =============================================
# 동작 방식:
#   1. data/trade_log.jsonl 에서 진입 액션(OPEN_LONG/SHORT, REVERSAL_*)을 읽음
#   2. Binance Futures 1분봉 캔들 조회 (entry ts 이후)
#   3. 캔들 high/low 를 순차 스캔 → SL / TP 중 어느 것이 먼저 닿는지 판단
#   4. 손익, 승패, 보유 시간 계산
#   5. 전체 통계 집계 (승률, 총손익, 최대낙폭, 평균 보유시간)
# =============================================
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

import requests

import config as _cfg

logger = logging.getLogger(__name__)

_BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
TRADE_LOG_PATH   = os.path.join(_BASE_DIR, "data", "trade_log.jsonl")
CACHE_PATH       = os.path.join(_BASE_DIR, "data", "backtest_cache.json")
DRY_BALANCE_PATH = os.path.join(_BASE_DIR, "data", "dry_run_balance.json")
FUTURES_URL      = _cfg.BINANCE_FUTURES_URL  # https://fapi.binance.com

ENTRY_ACTIONS = {"OPEN_LONG", "OPEN_SHORT", "REVERSAL_LONG", "REVERSAL_SHORT"}
MAX_CANDLES   = 500   # 1분봉 500개 = 약 8시간 시뮬레이션
MAX_WAIT_MIN  = 480   # 최대 대기 시간 (분) — 미결 처리 기준


# ══════════════════════════════════════════════
# 결과 데이터 클래스
# ══════════════════════════════════════════════

@dataclass
class TradeResult:
    # 원본 로그 필드
    ts:           str
    symbol:       str
    action:       str
    signal_en:    str
    confidence:   int
    entry_price:  float
    sl_price:     float
    tp_price:     float
    quantity:     float
    leverage:     int
    dry_run:      bool

    # 백테스트 결과
    outcome:      str     = "PENDING"   # WIN | LOSS | TIMEOUT | INVALID
    exit_price:   float   = 0.0
    exit_ts:      str     = ""
    hold_min:     float   = 0.0         # 보유 시간 (분)
    pnl_usd:      float   = 0.0         # 손익 (USDT 기준)
    pnl_pct:      float   = 0.0         # 손익률 (레버리지 포함)
    margin_usd:   float   = 0.0         # 투입 증거금
    note:         str     = ""


@dataclass
class BacktestSummary:
    total:        int   = 0
    wins:         int   = 0
    losses:       int   = 0
    timeouts:     int   = 0
    invalids:     int   = 0
    win_rate:     float = 0.0
    total_pnl:    float = 0.0
    avg_pnl:      float = 0.0
    max_drawdown: float = 0.0
    avg_hold_min: float = 0.0
    trades:       list  = field(default_factory=list)


# ══════════════════════════════════════════════
# 포지션 감시 상태 (메모리, 프로세스 재시작 시 초기화)
# ══════════════════════════════════════════════

# 현재 감시 중인 포지션의 "마지막으로 검사한 캔들 종료 시각(ms)"
# key: trade_ts(str), value: last_checked_close_ms(int)
_watcher_state: dict = {}


# ══════════════════════════════════════════════
# Binance 캔들 조회
# ══════════════════════════════════════════════

def _fetch_klines(symbol: str, start_ms: int, limit: int = MAX_CANDLES) -> list[dict]:
    """
    Binance Futures 1m 캔들 조회.
    반환: [{"open_time": int, "high": float, "low": float, "close_time": int}, ...]
    """
    url = f"{FUTURES_URL}/fapi/v1/klines"
    params = {
        "symbol":    symbol.replace("/", ""),
        "interval":  "1m",
        "startTime": start_ms,
        "limit":     limit,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        raw = resp.json()
        return [
            {
                "open_time":  c[0],
                "open":       float(c[1]),
                "high":       float(c[2]),
                "low":        float(c[3]),
                "close":      float(c[4]),
                "close_time": c[6],
            }
            for c in raw
        ]
    except Exception as exc:
        logger.warning("캔들 조회 실패 (%s): %s", symbol, exc)
        return []


# ══════════════════════════════════════════════
# 개별 트레이드 시뮬레이션
# ══════════════════════════════════════════════

def _simulate(record: dict, candles: list[dict]) -> tuple[str, float, str, float]:
    """
    캔들 목록을 순차 스캔해서 SL/TP 히트 여부 결정.
    반환: (outcome, exit_price, exit_ts, hold_min)
    outcome: "WIN" | "LOSS" | "TIMEOUT"
    """
    direction = "LONG" if "LONG" in record["action"] else "SHORT"
    sl = record["sl_price"]
    tp = record["tp_price"]
    entry_ms = _ts_to_ms(record["ts"])

    for c in candles:
        high = c["high"]
        low  = c["low"]
        candle_close_ts = _ms_to_iso(c["close_time"])
        hold_min = (c["close_time"] - entry_ms) / 60_000

        if direction == "LONG":
            # 같은 캔들에서 동시 닿을 경우: 하락→SL 먼저 (보수적)
            if low <= sl:
                return "LOSS", sl, candle_close_ts, hold_min
            if high >= tp:
                return "WIN", tp, candle_close_ts, hold_min
        else:  # SHORT
            if high >= sl:
                return "LOSS", sl, candle_close_ts, hold_min
            if low <= tp:
                return "WIN", tp, candle_close_ts, hold_min

    # 캔들 범위 내에서 결론 없음 → TIMEOUT
    if candles:
        last = candles[-1]
        hold_min = (last["close_time"] - entry_ms) / 60_000
        return "TIMEOUT", last["close"], _ms_to_iso(last["close_time"]), hold_min
    return "TIMEOUT", record["entry_price"], "", 0.0


def _calc_pnl(direction: str, entry: float, exit_price: float,
              qty: float, leverage: int) -> tuple[float, float, float]:
    """
    손익 계산.
    반환: (pnl_usd, pnl_pct, margin_usd)
    pnl_usd: (exit - entry) * qty   for LONG
             (entry - exit) * qty   for SHORT
    margin_usd: qty * entry / leverage
    pnl_pct: pnl_usd / margin_usd * 100
    """
    if entry <= 0 or qty <= 0:
        return 0.0, 0.0, 0.0
    if direction == "LONG":
        pnl_usd = (exit_price - entry) * qty
    else:
        pnl_usd = (entry - exit_price) * qty
    margin_usd = (entry * qty) / max(leverage, 1)
    pnl_pct = (pnl_usd / margin_usd * 100) if margin_usd > 0 else 0.0
    return round(pnl_usd, 4), round(pnl_pct, 2), round(margin_usd, 4)


# ══════════════════════════════════════════════
# 유틸
# ══════════════════════════════════════════════

def _ts_to_ms(ts_str: str) -> int:
    """ISO 8601 문자열 → 유닉스 밀리초"""
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1000)
    except Exception:
        return int(time.time() * 1000)


def _ms_to_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def _is_future(ts_str: str, buffer_min: int = 5) -> bool:
    """진입 시각이 현재보다 미래(+buffer_min)인지 확인 (아직 완료 불가)"""
    entry_ms = _ts_to_ms(ts_str)
    return entry_ms > (time.time() * 1000 - buffer_min * 60_000)


# ══════════════════════════════════════════════
# 오늘 요약 (캔들 조회 없이 로그만으로 빠르게)
# ══════════════════════════════════════════════

def get_today_summary() -> dict:
    """
    오늘(UTC) trade_log 에서 진입 레코드만 읽어 빠른 통계를 반환.
    캔들 조회 없이 로그에 기록된 값만 사용하므로 즉시 응답 가능.

    반환:
        entries_today  : 오늘 진입 횟수
        longs_today    : LONG 진입 수
        shorts_today   : SHORT 진입 수
        dry_run        : 마지막 레코드의 dry_run 값
        last_action    : 마지막 진입 액션
        last_ts        : 마지막 진입 시각
        last_signal    : 마지막 signal_en
        last_confidence: 마지막 확신도
        live_position  : get_live_position() 결과 (None 가능)
    """
    today_utc = datetime.now(timezone.utc).date().isoformat()  # "2026-04-15"

    entries_today = 0
    longs_today   = 0
    shorts_today  = 0
    last_rec: Optional[dict] = None

    try:
        with open(TRADE_LOG_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("action") not in ENTRY_ACTIONS:
                    continue
                ts_str = rec.get("ts", "")
                if ts_str[:10] == today_utc:
                    entries_today += 1
                    if "LONG" in rec.get("action", ""):
                        longs_today += 1
                    else:
                        shorts_today += 1
                last_rec = rec
    except FileNotFoundError:
        pass

    live_pos = get_live_position()
    balance_data = load_dry_balance()

    return {
        "entries_today":   entries_today,
        "longs_today":     longs_today,
        "shorts_today":    shorts_today,
        "dry_run":         last_rec.get("dry_run", True) if last_rec else True,
        "last_action":     last_rec.get("action")        if last_rec else None,
        "last_ts":         last_rec.get("ts")            if last_rec else None,
        "last_signal":     last_rec.get("signal_en")     if last_rec else None,
        "last_confidence": last_rec.get("confidence")    if last_rec else None,
        "live_position":   live_pos,
        "balance":         balance_data,
    }


# ══════════════════════════════════════════════
# 라이브 포지션 (현재 열려있는 드라이런 포지션)
# ══════════════════════════════════════════════

def get_live_position() -> Optional[dict]:
    """
    trade_log.jsonl 에서 현재 오픈 중인 드라이런 포지션을 반환.
    마지막 진입 액션 이후 CLOSE / 반대 방향 REVERSAL 이 없으면 "open"으로 판단.

    반환 dict:
      action, symbol, direction, entry_price, sl_price, tp_price,
      quantity, leverage, confidence, ts, held_min, dry_run
    """
    ALL_ACTIONS  = ENTRY_ACTIONS | {"CLOSE", "SKIP", "ERROR"}
    CLOSE_ACTS   = {"CLOSE"}

    all_records: list[dict] = []
    try:
        with open(TRADE_LOG_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("action") in ALL_ACTIONS:
                    all_records.append(rec)
    except FileNotFoundError:
        return None

    if not all_records:
        return None

    # 시간 오름차순 정렬
    all_records.sort(key=lambda r: r.get("ts", ""))

    # 마지막 진입 액션 찾기
    last_entry: Optional[dict] = None
    for rec in all_records:
        action = rec.get("action", "")
        if action in ENTRY_ACTIONS:
            last_entry = rec
        elif action in CLOSE_ACTS:
            last_entry = None   # 명시적 청산 → 포지션 없음

    if last_entry is None:
        return None

    # REVERSAL 은 새 포지션이므로 유효 — 단 entry_price / sl / tp 유효성 체크
    if last_entry.get("entry_price", 0) <= 0:
        return None

    # 최대 시뮬레이션 범위(8시간) 초과 시 타임아웃 처리 → 열린 포지션 아님
    entry_ms    = _ts_to_ms(last_entry.get("ts", ""))
    now_ms      = time.time() * 1000
    held_min    = (now_ms - entry_ms) / 60_000
    if held_min > MAX_WAIT_MIN:
        return None

    direction = "LONG" if "LONG" in last_entry.get("action", "") else "SHORT"

    return {
        "action":       last_entry.get("action"),
        "symbol":       last_entry.get("symbol", ""),
        "direction":    direction,
        "entry_price":  last_entry.get("entry_price", 0),
        "sl_price":     last_entry.get("sl_price", 0),
        "tp_price":     last_entry.get("tp_price", 0),
        "quantity":     last_entry.get("quantity", 0),
        "leverage":     last_entry.get("leverage", 1),
        "confidence":   last_entry.get("confidence", 0),
        "signal_en":    last_entry.get("signal_en", ""),
        "ts":           last_entry.get("ts", ""),
        "held_min":     round(held_min, 1),
        "dry_run":      last_entry.get("dry_run", True),
    }


# ══════════════════════════════════════════════
# 드라이런 가상 잔고 관리
# ══════════════════════════════════════════════

def _default_balance() -> float:
    return float(os.getenv("AUTO_TRADE_DRY_RUN_BALANCE",
                           str(getattr(_cfg, "AUTO_TRADE_DRY_RUN_BALANCE", 10000))))


def load_dry_balance() -> dict:
    """가상 잔고 파일 로드. 없으면 초기값으로 생성."""
    try:
        with open(DRY_BALANCE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        # 필드 무결성 보장
        data.setdefault("initial_balance", data.get("balance", _default_balance()))
        data.setdefault("peak_balance",    data.get("balance", _default_balance()))
        data.setdefault("total_pnl",       0.0)
        data.setdefault("trade_count",     0)
        data.setdefault("wins",            0)
        data.setdefault("losses",          0)
        return data
    except (FileNotFoundError, json.JSONDecodeError):
        initial = _default_balance()
        return {
            "balance":         initial,
            "initial_balance": initial,
            "peak_balance":    initial,
            "total_pnl":       0.0,
            "trade_count":     0,
            "wins":            0,
            "losses":          0,
            "updated_at":      datetime.now(timezone.utc).isoformat(),
        }


def _save_dry_balance(data: dict):
    try:
        os.makedirs(os.path.dirname(DRY_BALANCE_PATH), exist_ok=True)
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        with open(DRY_BALANCE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning("가상 잔고 저장 실패: %s", exc)


def get_dry_balance() -> float:
    """현재 가상 잔고만 빠르게 반환."""
    return load_dry_balance()["balance"]


def _apply_pnl_to_balance(pnl_usd: float, outcome: str):
    """트레이드 확정 시 가상 잔고에 손익 반영."""
    data = load_dry_balance()
    data["balance"]     = round(data["balance"] + pnl_usd, 4)
    data["peak_balance"] = max(data["peak_balance"], data["balance"])
    data["total_pnl"]   = round(data["total_pnl"] + pnl_usd, 4)
    data["trade_count"] = data.get("trade_count", 0) + 1
    if outcome == "WIN":
        data["wins"]   = data.get("wins", 0) + 1
    elif outcome == "LOSS":
        data["losses"] = data.get("losses", 0) + 1
    _save_dry_balance(data)
    logger.info(
        "[DryBalance] %s pnl=%+.2f  잔고 %.2f → %.2f",
        outcome, pnl_usd,
        data["balance"] - pnl_usd, data["balance"],
    )


def reset_dry_balance(initial: Optional[float] = None):
    """가상 잔고 초기화 (디버그/재시작용)."""
    amt = initial if initial is not None else _default_balance()
    _save_dry_balance({
        "balance":         amt,
        "initial_balance": amt,
        "peak_balance":    amt,
        "total_pnl":       0.0,
        "trade_count":     0,
        "wins":            0,
        "losses":          0,
    })


# ══════════════════════════════════════════════
# 캐시 I/O
# ══════════════════════════════════════════════

def _load_cache() -> dict:
    """캐시 파일 로드. {ts_str: TradeResult_dict} 형태."""
    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_cache(cache: dict):
    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
    except Exception as exc:
        logger.warning("캐시 저장 실패: %s", exc)


def _build_summary_from_results(results: list[dict]) -> BacktestSummary:
    """TradeResult dict 목록 → BacktestSummary"""
    summary = BacktestSummary()
    running_pnl  = 0.0
    peak_pnl     = 0.0
    max_drawdown = 0.0
    total_hold   = 0.0
    evaluated    = 0

    for t in results:
        oc = t.get("outcome", "")
        if oc == "WIN":     summary.wins     += 1
        elif oc == "LOSS":  summary.losses   += 1
        elif oc == "TIMEOUT": summary.timeouts += 1
        elif oc == "INVALID": summary.invalids += 1

        pnl = t.get("pnl_usd", 0) or 0
        running_pnl  += pnl
        peak_pnl      = max(peak_pnl, running_pnl)
        max_drawdown  = max(max_drawdown, peak_pnl - running_pnl)

        if oc in ("WIN", "LOSS", "TIMEOUT"):
            total_hold += t.get("hold_min", 0) or 0
            evaluated  += 1

    ev = summary.wins + summary.losses + summary.timeouts
    summary.total        = len(results)
    summary.win_rate     = round(summary.wins / ev * 100, 1) if ev > 0 else 0.0
    summary.total_pnl    = round(sum(t.get("pnl_usd", 0) or 0 for t in results), 4)
    summary.avg_pnl      = round(summary.total_pnl / ev, 4) if ev > 0 else 0.0
    summary.max_drawdown = round(max_drawdown, 4)
    summary.avg_hold_min = round(total_hold / evaluated, 1) if evaluated > 0 else 0.0
    summary.trades       = results
    return summary


# ══════════════════════════════════════════════
# 서버 사이드 증분 백테스트 (캐시 활용)
# ══════════════════════════════════════════════

def run_backtest_incremental(dry_run_only: bool = False) -> BacktestSummary:
    """
    캐시를 활용한 증분 백테스트. 서버에서 분석 완료 후 자동 실행.

    - 이미 WIN/LOSS/TIMEOUT 으로 확정된 트레이드는 캐시를 그대로 사용 (캔들 재조회 없음)
    - PENDING 또는 캐시 미존재 트레이드만 Binance 에서 신규 캔들 조회
    - 결과를 CACHE_PATH 에 저장
    """
    cache = _load_cache()
    RESOLVED = {"WIN", "LOSS", "TIMEOUT", "INVALID"}

    # ── 로그에서 진입 레코드 읽기 ──
    records: list[dict] = []
    try:
        with open(TRADE_LOG_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("action") not in ENTRY_ACTIONS:
                    continue
                if dry_run_only and not rec.get("dry_run", True):
                    continue
                records.append(rec)
    except FileNotFoundError:
        return BacktestSummary()

    results: list[dict] = []
    cache_updated = False

    for rec in records:
        ts  = rec.get("ts", "")
        cached = cache.get(ts)

        # 이미 확정된 결과 → 캐시 그대로 사용
        if cached and cached.get("outcome") in RESOLVED:
            results.append(cached)
            continue

        # 유효성 체크
        if rec.get("entry_price", 0) <= 0 or rec.get("sl_price", 0) <= 0 or rec.get("tp_price", 0) <= 0:
            result = asdict(TradeResult(
                ts=ts, symbol=rec.get("symbol",""), action=rec.get("action",""),
                signal_en=rec.get("signal_en",""), confidence=rec.get("confidence",0),
                entry_price=rec.get("entry_price",0), sl_price=rec.get("sl_price",0),
                tp_price=rec.get("tp_price",0), quantity=rec.get("quantity",0),
                leverage=rec.get("leverage",1), dry_run=rec.get("dry_run",True),
                outcome="INVALID", note="entry/SL/TP 가격 없음",
            ))
            cache[ts] = result
            results.append(result)
            cache_updated = True
            continue

        # 아직 완료되지 않은 최신 트레이드 (진입 후 5분 이내)
        if _is_future(ts, buffer_min=5):
            result = asdict(TradeResult(
                ts=ts, symbol=rec.get("symbol",""), action=rec.get("action",""),
                signal_en=rec.get("signal_en",""), confidence=rec.get("confidence",0),
                entry_price=rec.get("entry_price",0), sl_price=rec.get("sl_price",0),
                tp_price=rec.get("tp_price",0), quantity=rec.get("quantity",0),
                leverage=rec.get("leverage",1), dry_run=rec.get("dry_run",True),
                outcome="PENDING", note="진입 직후 대기",
            ))
            results.append(result)
            continue

        # ── 신규/PENDING → Binance 캔들 조회 ──────────────────
        entry_ms = _ts_to_ms(ts)
        candles  = _fetch_klines(rec["symbol"], start_ms=entry_ms, limit=MAX_CANDLES)

        outcome, exit_price, exit_ts, hold_min = _simulate(rec, candles)
        direction = "LONG" if "LONG" in rec.get("action","") else "SHORT"
        pnl_usd, pnl_pct, margin_usd = _calc_pnl(
            direction, rec["entry_price"], exit_price, rec.get("quantity",0), rec.get("leverage",1)
        )

        result = asdict(TradeResult(
            ts=ts, symbol=rec.get("symbol",""), action=rec.get("action",""),
            signal_en=rec.get("signal_en",""), confidence=rec.get("confidence",0),
            entry_price=rec.get("entry_price",0), sl_price=rec.get("sl_price",0),
            tp_price=rec.get("tp_price",0), quantity=rec.get("quantity",0),
            leverage=rec.get("leverage",1), dry_run=rec.get("dry_run",True),
            outcome=outcome, exit_price=exit_price, exit_ts=exit_ts,
            hold_min=round(hold_min,1), pnl_usd=pnl_usd, pnl_pct=pnl_pct,
            margin_usd=margin_usd,
        ))

        # PENDING 은 캐시에 저장하지 않음 (다음 실행에서 재평가)
        if outcome in RESOLVED:
            cache[ts] = result
            cache_updated = True

        results.append(result)

    if cache_updated:
        _save_cache(cache)

    # 최신순 정렬
    results.sort(key=lambda r: r.get("ts",""), reverse=True)
    return _build_summary_from_results(results)


def get_cached_backtest() -> Optional[BacktestSummary]:
    """캐시만 읽어서 즉시 반환. 캐시 없으면 None."""
    cache = _load_cache()
    if not cache:
        return None
    results = sorted(cache.values(), key=lambda r: r.get("ts",""), reverse=True)
    return _build_summary_from_results(results)


# ══════════════════════════════════════════════
# 실시간 1분봉 감시 (서버 백그라운드용)
# ══════════════════════════════════════════════

def tick_open_position() -> Optional[dict]:
    """
    현재 열린 드라이런 포지션이 있으면 **신규 1분봉만** 조회해 SL/TP 체크.
    매 분 서버에서 호출. 결과가 확정되면 캐시에 저장하고 결과 dict 반환.
    포지션 없거나 아직 미확정이면 None 반환.

    반환 dict (확정 시):
        outcome, exit_price, pnl_usd, pnl_pct, hold_min, ts, symbol
    """
    pos = get_live_position()
    if pos is None:
        return None

    ts      = pos["ts"]
    symbol  = pos["symbol"]
    sl      = pos["sl_price"]
    tp      = pos["tp_price"]
    entry   = pos["entry_price"]
    qty     = pos["quantity"]
    lev     = pos["leverage"]
    isLong  = pos["direction"] == "LONG"
    entry_ms = _ts_to_ms(ts)

    # 마지막으로 확인한 캔들 종료 시각 이후부터만 조회
    last_checked_ms = _watcher_state.get(ts, entry_ms)
    # 다음 캔들 시작 = last_checked_ms + 1
    fetch_from_ms   = last_checked_ms + 1

    # 현재 완료된 캔들만 가져옴 (limit=5: 최대 5분치, 보통 1-2개)
    candles = _fetch_klines(symbol, start_ms=fetch_from_ms, limit=5)

    # 아직 새 완료 캔들 없음
    if not candles:
        return None

    # 캔들 순차 스캔
    outcome = exit_price = exit_ts = hold_min = None
    for c in candles:
        high = c["high"]
        low  = c["low"]
        close_ms  = c["close_time"]
        hold_min  = (close_ms - entry_ms) / 60_000

        if isLong:
            if low <= sl:
                outcome, exit_price = "LOSS", sl; exit_ts = _ms_to_iso(close_ms); break
            if high >= tp:
                outcome, exit_price = "WIN",  tp; exit_ts = _ms_to_iso(close_ms); break
        else:
            if high >= sl:
                outcome, exit_price = "LOSS", sl; exit_ts = _ms_to_iso(close_ms); break
            if low <= tp:
                outcome, exit_price = "WIN",  tp; exit_ts = _ms_to_iso(close_ms); break

        # 이 캔들에서 닿지 않음 → 마지막 확인 시각 업데이트
        _watcher_state[ts] = close_ms

    # 타임아웃 체크 (8시간 초과)
    if outcome is None and hold_min is not None and hold_min > MAX_WAIT_MIN:
        last_c   = candles[-1]
        outcome  = "TIMEOUT"
        exit_price = last_c["close"]
        exit_ts  = _ms_to_iso(last_c["close_time"])
        hold_min = (last_c["close_time"] - entry_ms) / 60_000

    if outcome is None:
        # 아직 미확정
        return None

    # ── 결과 확정 → 캐시에 저장 ──────────────────
    direction = "LONG" if isLong else "SHORT"
    pnl_usd, pnl_pct, margin_usd = _calc_pnl(direction, entry, exit_price, qty, lev)

    result = asdict(TradeResult(
        ts=ts, symbol=symbol, action=pos["action"],
        signal_en=pos["signal_en"], confidence=pos["confidence"],
        entry_price=entry, sl_price=sl, tp_price=tp,
        quantity=qty, leverage=lev, dry_run=pos["dry_run"],
        outcome=outcome, exit_price=exit_price, exit_ts=exit_ts,
        hold_min=round(hold_min, 1), pnl_usd=pnl_usd, pnl_pct=pnl_pct,
        margin_usd=margin_usd,
    ))

    cache = _load_cache()
    cache[ts] = result
    _save_cache(cache)
    _watcher_state.pop(ts, None)  # 감시 상태 정리

    # 가상 잔고 복리 반영 (드라이런 전용)
    if pos.get("dry_run", True):
        _apply_pnl_to_balance(pnl_usd, outcome)

    logger.info(
        "[Watcher] %s %s → %s  exit=%.1f  pnl=%+.2f USD  hold=%.0f분",
        symbol, direction, outcome, exit_price, pnl_usd, hold_min
    )
    return result


# ══════════════════════════════════════════════
# 메인 백테스트 함수
# ══════════════════════════════════════════════

def run_backtest(limit: int = 200, dry_run_only: bool = False) -> BacktestSummary:
    """
    trade_log.jsonl 를 읽어 백테스트를 실행하고 BacktestSummary 를 반환.

    Parameters
    ----------
    limit       : 최신 N개 진입 레코드만 처리 (0 = 전체)
    dry_run_only: True 이면 dry_run=True 인 레코드만 대상
    """
    summary = BacktestSummary()

    # ── 로그 읽기 ──────────────────────────────
    records: list[dict] = []
    try:
        with open(TRADE_LOG_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("action") not in ENTRY_ACTIONS:
                    continue
                if dry_run_only and not rec.get("dry_run", True):
                    continue
                # 유효성: entry_price / sl_price / tp_price 모두 양수여야 함
                if rec.get("entry_price", 0) <= 0 or rec.get("sl_price", 0) <= 0 or rec.get("tp_price", 0) <= 0:
                    result = TradeResult(
                        ts=rec.get("ts", ""),
                        symbol=rec.get("symbol", ""),
                        action=rec.get("action", ""),
                        signal_en=rec.get("signal_en", ""),
                        confidence=rec.get("confidence", 0),
                        entry_price=rec.get("entry_price", 0),
                        sl_price=rec.get("sl_price", 0),
                        tp_price=rec.get("tp_price", 0),
                        quantity=rec.get("quantity", 0),
                        leverage=rec.get("leverage", 1),
                        dry_run=rec.get("dry_run", True),
                        outcome="INVALID",
                        note="entry/SL/TP 가격 정보 없음",
                    )
                    summary.invalids += 1
                    summary.trades.append(asdict(result))
                    continue
                records.append(rec)
    except FileNotFoundError:
        logger.warning("trade_log.jsonl 없음: %s", TRADE_LOG_PATH)
        return summary

    # 최신순 정렬 후 limit 적용
    records.sort(key=lambda r: r.get("ts", ""), reverse=True)
    if limit > 0:
        records = records[:limit]
    records.reverse()  # 시간 오름차순으로 복원

    # ── 개별 시뮬레이션 ────────────────────────
    running_pnl    = 0.0
    peak_pnl       = 0.0
    max_drawdown   = 0.0
    total_hold_min = 0.0
    evaluated      = 0

    for rec in records:
        ts_str = rec.get("ts", "")

        # 아직 완료되지 않을 수 있는 최신 트레이드
        if _is_future(ts_str, buffer_min=MAX_WAIT_MIN):
            result = TradeResult(
                ts=ts_str,
                symbol=rec.get("symbol", ""),
                action=rec.get("action", ""),
                signal_en=rec.get("signal_en", ""),
                confidence=rec.get("confidence", 0),
                entry_price=rec.get("entry_price", 0),
                sl_price=rec.get("sl_price", 0),
                tp_price=rec.get("tp_price", 0),
                quantity=rec.get("quantity", 0),
                leverage=rec.get("leverage", 1),
                dry_run=rec.get("dry_run", True),
                outcome="PENDING",
                note="아직 완료 대기 중",
            )
            summary.trades.append(asdict(result))
            continue

        # 캔들 조회
        entry_ms = _ts_to_ms(ts_str)
        candles  = _fetch_klines(rec["symbol"], start_ms=entry_ms, limit=MAX_CANDLES)

        # 시뮬레이션
        outcome, exit_price, exit_ts, hold_min = _simulate(rec, candles)

        # 손익 계산
        direction = "LONG" if "LONG" in rec["action"] else "SHORT"
        pnl_usd, pnl_pct, margin_usd = _calc_pnl(
            direction, rec["entry_price"], exit_price, rec["quantity"], rec["leverage"]
        )

        result = TradeResult(
            ts=ts_str,
            symbol=rec.get("symbol", ""),
            action=rec.get("action", ""),
            signal_en=rec.get("signal_en", ""),
            confidence=rec.get("confidence", 0),
            entry_price=rec.get("entry_price", 0),
            sl_price=rec.get("sl_price", 0),
            tp_price=rec.get("tp_price", 0),
            quantity=rec.get("quantity", 0),
            leverage=rec.get("leverage", 1),
            dry_run=rec.get("dry_run", True),
            outcome=outcome,
            exit_price=exit_price,
            exit_ts=exit_ts,
            hold_min=round(hold_min, 1),
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct,
            margin_usd=margin_usd,
        )
        summary.trades.append(asdict(result))

        # 통계 집계
        if outcome == "WIN":
            summary.wins += 1
        elif outcome == "LOSS":
            summary.losses += 1
        elif outcome == "TIMEOUT":
            summary.timeouts += 1

        running_pnl  += pnl_usd
        peak_pnl      = max(peak_pnl, running_pnl)
        drawdown      = peak_pnl - running_pnl
        max_drawdown  = max(max_drawdown, drawdown)
        total_hold_min += hold_min
        evaluated += 1

    # ── 요약 ────────────────────────────────────
    summary.total        = len(records) + summary.invalids
    evaluated_total      = summary.wins + summary.losses + summary.timeouts
    summary.win_rate     = round(summary.wins / evaluated_total * 100, 1) if evaluated_total > 0 else 0.0
    summary.total_pnl    = round(sum(t.get("pnl_usd", 0) for t in summary.trades), 4)
    summary.avg_pnl      = round(summary.total_pnl / evaluated_total, 4) if evaluated_total > 0 else 0.0
    summary.max_drawdown = round(max_drawdown, 4)
    summary.avg_hold_min = round(total_hold_min / evaluated, 1) if evaluated > 0 else 0.0

    return summary


# ══════════════════════════════════════════════
# CLI 실행 (직접 호출 시)
# ══════════════════════════════════════════════

def _print_summary(s: BacktestSummary):
    sep = "=" * 60
    print(sep)
    print("  백테스트 결과 요약")
    print(sep)
    print(f"  전체 진입 건수  : {s.total}")
    print(f"  ✅ 승 (WIN)     : {s.wins}")
    print(f"  ❌ 패 (LOSS)    : {s.losses}")
    print(f"  ⏱  타임아웃     : {s.timeouts}")
    print(f"  ⚠  미체결/무효  : {s.invalids}")
    print(f"  승률            : {s.win_rate}%")
    print(f"  총 손익         : ${s.total_pnl:+,.4f} USDT")
    print(f"  평균 손익       : ${s.avg_pnl:+,.4f} USDT")
    print(f"  최대 낙폭       : ${s.max_drawdown:,.4f} USDT")
    print(f"  평균 보유 시간  : {s.avg_hold_min}분")
    print(sep)
    print()

    if not s.trades:
        print("  (진입 트레이드 없음)")
        return

    print(f"  {'시각':<26} {'심볼':<10} {'방향':<16} {'결과':<8} {'진입':>10} {'손절':>10} {'목표':>10} {'청산':>10} {'손익':>10} {'보유':>6}")
    print("-" * 120)
    for t in s.trades:
        direction = "LONG" if "LONG" in t.get("action","") else "SHORT"
        print(
            f"  {t.get('ts','')[:19]:<26} "
            f"{t.get('symbol',''):<10} "
            f"{direction:<16} "
            f"{t.get('outcome',''):<8} "
            f"${t.get('entry_price',0):>9,.1f} "
            f"${t.get('sl_price',0):>9,.1f} "
            f"${t.get('tp_price',0):>9,.1f} "
            f"${t.get('exit_price',0):>9,.1f} "
            f"${t.get('pnl_usd',0):>+9,.2f} "
            f"{t.get('hold_min',0):>5.0f}분"
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Crypto 자동매매 백테스터")
    parser.add_argument("--limit",    type=int, default=200, help="최신 N개 진입만 분석 (0=전체)")
    parser.add_argument("--dry-only", action="store_true",   help="드라이런 레코드만 대상")
    args = parser.parse_args()

    summary = run_backtest(limit=args.limit, dry_run_only=args.dry_only)
    _print_summary(summary)
