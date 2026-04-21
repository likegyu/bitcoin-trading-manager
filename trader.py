# =============================================
# Binance Futures 주문 실행 모듈
# =============================================
# 역할:
#   - 선물 주문 생성 / 취소 / 조회
#   - 레버리지 설정
#   - 포지션 조회
#   - HMAC 서명 (account_context.py 와 동일 방식)
#
# 주의:
#   - 이 모듈은 실제 자금을 움직입니다.
#   - DRY_RUN=True 로 실행하면 주문을 전송하지 않고 로그만 남깁니다.
# =============================================
from __future__ import annotations

import hashlib
import hmac
import logging
import time
from typing import Optional
from urllib.parse import urlencode

import requests

import config as _cfg
from http_client import _session as _http  # 프록시 환경변수 무시 세션

logger = logging.getLogger(__name__)

FUTURES_BASE = "https://fapi.binance.com"
RECV_WINDOW  = 5000

# ── Dry-run 전역 스위치 ───────────────────────────────
# config.py 의 AUTO_TRADE_DRY_RUN 을 따름
def _dry_run() -> bool:
    return getattr(_cfg, "AUTO_TRADE_DRY_RUN", True)


# ── 인증 헬퍼 ─────────────────────────────────────────

def _headers() -> dict:
    key = _cfg.BINANCE_API_KEY
    if not key:
        raise RuntimeError("BINANCE_API_KEY 가 비어 있습니다.")
    return {"X-MBX-APIKEY": key}


def _sign(params: dict) -> str:
    secret = _cfg.BINANCE_SECRET_KEY
    if not secret:
        raise RuntimeError("BINANCE_SECRET_KEY 가 비어 있습니다.")
    qs = urlencode(params)
    return hmac.new(secret.encode(), qs.encode(), hashlib.sha256).hexdigest()


def _signed_params(params: dict) -> dict:
    params = {**params, "timestamp": int(time.time() * 1000), "recvWindow": RECV_WINDOW}
    params["signature"] = _sign(params)
    return params


def _raise_for_binance(resp: requests.Response):
    """Binance API 오류 응답을 읽기 쉬운 예외로 변환."""
    if resp.ok:
        return
    try:
        body = resp.json()
        code = body.get("code", "")
        msg  = body.get("msg", resp.text)
    except Exception:
        code, msg = "", resp.text
    raise RuntimeError(f"Binance API 오류 [{resp.status_code}] code={code}: {msg}")


# ── 레버리지 설정 ─────────────────────────────────────

def set_leverage(symbol: str, leverage: int) -> dict:
    """심볼 레버리지 설정. 이미 같은 레버리지면 Binance 가 그냥 OK."""
    if _dry_run():
        logger.info("[DRY-RUN] set_leverage %s x%d", symbol, leverage)
        return {"symbol": symbol, "leverage": leverage, "dry_run": True}
    params = _signed_params({"symbol": symbol, "leverage": leverage})
    resp = _http.post(
        f"{FUTURES_BASE}/fapi/v1/leverage",
        params=params,
        headers=_headers(),
        timeout=10,
    )
    _raise_for_binance(resp)
    return resp.json()


# ── 포지션 조회 ───────────────────────────────────────

def get_position(symbol: str) -> Optional[dict]:
    """
    현재 오픈 포지션을 반환한다.
    포지션이 없으면 None, 있으면 positionAmt / entryPrice / leverage 등 포함.
    """
    params = _signed_params({"symbol": symbol})
    resp = _http.get(
        f"{FUTURES_BASE}/fapi/v2/positionRisk",
        params=params,
        headers=_headers(),
        timeout=10,
    )
    _raise_for_binance(resp)
    positions = resp.json()
    for p in positions:
        if p.get("symbol") == symbol and float(p.get("positionAmt", 0)) != 0:
            return p
    return None


def get_account_balance(asset: str = "USDT") -> float:
    """선물 계좌 가용 잔고 (availableBalance) 반환. 실패 시 0.0."""
    params = _signed_params({})
    resp = _http.get(
        f"{FUTURES_BASE}/fapi/v2/balance",
        params=params,
        headers=_headers(),
        timeout=10,
    )
    _raise_for_binance(resp)
    for item in resp.json():
        if item.get("asset") == asset:
            return float(item.get("availableBalance", 0))
    return 0.0


# ── 주문 생성 ─────────────────────────────────────────

def place_market_order(
    symbol: str,
    side: str,          # "BUY" | "SELL"
    quantity: float,    # 코인 수량 (소수점 허용)
    reduce_only: bool = False,
) -> dict:
    """
    시장가 주문 전송.

    Parameters
    ----------
    symbol      : "BTCUSDC" 등
    side        : "BUY" (롱 진입 / 숏 청산) | "SELL" (숏 진입 / 롱 청산)
    quantity    : 주문 수량 (코인 단위)
    reduce_only : True 면 청산(reduce-only) 주문
    """
    if _dry_run():
        logger.info("[DRY-RUN] place_market_order %s %s qty=%.6f reduce_only=%s",
                    symbol, side, quantity, reduce_only)
        return {
            "symbol": symbol, "side": side, "quantity": quantity,
            "reduce_only": reduce_only, "dry_run": True,
            "orderId": -1, "status": "DRY_RUN",
        }

    params: dict = {
        "symbol":   symbol,
        "side":     side,
        "type":     "MARKET",
        "quantity": f"{quantity:.6f}",
    }
    if reduce_only:
        params["reduceOnly"] = "true"

    signed = _signed_params(params)
    resp = _http.post(
        f"{FUTURES_BASE}/fapi/v1/order",
        params=signed,
        headers=_headers(),
        timeout=10,
    )
    _raise_for_binance(resp)
    result = resp.json()
    logger.info("주문 체결: %s", result)
    return result


def place_stop_order(
    symbol: str,
    side: str,
    quantity: float,
    stop_price: float,
    order_type: str = "STOP_MARKET",   # "STOP_MARKET" | "TAKE_PROFIT_MARKET"
) -> dict:
    """
    손절 / 익절 스탑 주문 전송.

    Parameters
    ----------
    side        : 포지션 청산 방향 (롱 포지션이면 "SELL", 숏 포지션이면 "BUY")
    stop_price  : 스탑 트리거 가격
    order_type  : "STOP_MARKET" (손절) | "TAKE_PROFIT_MARKET" (익절)
    """
    if _dry_run():
        logger.info("[DRY-RUN] place_stop_order %s %s qty=%.6f stop_price=%.2f type=%s",
                    symbol, side, quantity, stop_price, order_type)
        return {
            "symbol": symbol, "side": side, "quantity": quantity,
            "stopPrice": stop_price, "type": order_type,
            "dry_run": True, "orderId": -1, "status": "DRY_RUN",
        }

    params = {
        "symbol":     symbol,
        "side":       side,
        "type":       order_type,
        "stopPrice":  f"{stop_price:.2f}",
        "quantity":   f"{quantity:.6f}",
        "reduceOnly": "true",
        "workingType": "MARK_PRICE",
    }
    signed = _signed_params(params)
    resp = _http.post(
        f"{FUTURES_BASE}/fapi/v1/order",
        params=signed,
        headers=_headers(),
        timeout=10,
    )
    _raise_for_binance(resp)
    result = resp.json()
    logger.info("스탑 주문 등록: %s", result)
    return result


def cancel_all_open_orders(symbol: str) -> dict:
    """심볼의 모든 미체결 주문 취소."""
    if _dry_run():
        logger.info("[DRY-RUN] cancel_all_open_orders %s", symbol)
        return {"symbol": symbol, "dry_run": True}
    params = _signed_params({"symbol": symbol})
    resp = _http.delete(
        f"{FUTURES_BASE}/fapi/v1/allOpenOrders",
        params=params,
        headers=_headers(),
        timeout=10,
    )
    _raise_for_binance(resp)
    return resp.json()


def close_position(symbol: str) -> Optional[dict]:
    """
    현재 오픈 포지션 전량 시장가 청산.
    포지션 없으면 None 반환.
    """
    pos = get_position(symbol)
    if pos is None:
        logger.info("청산 요청 — %s 오픈 포지션 없음", symbol)
        return None

    amt = float(pos["positionAmt"])
    side = "SELL" if amt > 0 else "BUY"
    qty  = abs(amt)
    logger.info("포지션 청산: %s side=%s qty=%.6f", symbol, side, qty)
    return place_market_order(symbol, side, qty, reduce_only=True)


# ── 수량 정밀도 헬퍼 ──────────────────────────────────

_step_size_cache: dict[str, float] = {}

def get_lot_step_size(symbol: str) -> float:
    """
    심볼의 최소 주문 단위(stepSize)를 반환.
    캐시 사용 (프로세스 재시작 전까지 유지).
    """
    if symbol in _step_size_cache:
        return _step_size_cache[symbol]
    try:
        resp = _http.get(
            f"{FUTURES_BASE}/fapi/v1/exchangeInfo", timeout=10
        )
        resp.raise_for_status()
        for s in resp.json().get("symbols", []):
            if s["symbol"] == symbol:
                for f in s.get("filters", []):
                    if f["filterType"] == "LOT_SIZE":
                        step = float(f["stepSize"])
                        _step_size_cache[symbol] = step
                        return step
    except Exception as exc:
        logger.warning("stepSize 조회 실패 (%s) — 기본값 0.001 사용: %s", symbol, exc)
    return 0.001


def round_qty(qty: float, step: float) -> float:
    """stepSize 에 맞춰 수량을 내림 처리."""
    if step <= 0:
        return qty
    precision = max(0, round(-1 * (len(str(step).rstrip("0")) - 2)))
    factor = 1.0 / step
    return round(int(qty * factor) / factor, precision)
