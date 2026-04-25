# =============================================
# 내 계좌 상태 수집 (Binance Futures)
# 잔고 / 오픈 포지션 / 오늘 실현 손익
# =============================================
from __future__ import annotations

import hmac
import hashlib
import time as _time
from threading import Lock
import requests
from datetime import timezone
from typing import Optional
from urllib.parse import urlsplit
import config as _cfg
from http_client import _session as _http  # 프록시 환경변수 무시 세션
from account_history import attach_account_context_summary
from time_utils import start_of_kst_day

TRACKED_COLLATERAL_ASSETS = ("USDT", "USDC")
INCOME_CACHE_TTL_SECS = 8.0
_INCOME_CACHE_LOCK = Lock()
_INCOME_CACHE: dict[tuple[str, int], dict] = {}


def _safe_error_message(exc: Exception) -> str:
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        status = exc.response.status_code
        response_url = exc.response.url or ""
        try:
            body = exc.response.json()
        except ValueError:
            body = {}

        api_code = body.get("code")
        api_msg = body.get("msg")
        path = urlsplit(response_url).path or response_url

        if status == 401:
            if api_code == -2015:
                return (
                    "Binance 인증 실패 (401 / -2015: API 키, IP 화이트리스트, "
                    "또는 선물 권한 문제)"
                )
            return f"Binance 인증 실패 ({status}: {path})"

        if api_code == -1021:
            return "Binance 시간 오차 오류 (-1021: 서버 시간과 로컬 시간 차이)"
        if api_code == -1022:
            return "Binance 서명 오류 (-1022: API secret 또는 서명 문자열 불일치)"
        if api_msg:
            return f"Binance API 오류 ({status} / {api_code}: {api_msg})"
        return f"Binance HTTP 오류 ({status}: {path})"

    msg = str(exc).strip()
    return msg if msg else exc.__class__.__name__


def _api_key_headers() -> dict:
    if not _cfg.BINANCE_API_KEY:
        raise RuntimeError("BINANCE_API_KEY가 비어 있습니다.")
    return {"X-MBX-APIKEY": _cfg.BINANCE_API_KEY}


def open_user_data_stream() -> str:
    """USDⓈ-M Futures user data stream listenKey 생성/연장."""
    try:
        r = _http.post(
            f"{_cfg.BINANCE_FUTURES_URL}/fapi/v1/listenKey",
            headers=_api_key_headers(),
            timeout=8,
        )
        r.raise_for_status()
        return str(r.json()["listenKey"])
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 401:
            raise RuntimeError(
                "Binance user data stream 인증 실패 (401). "
                "선물 API 키 권한, IP 제한, 또는 저장된 키 갱신 여부를 확인하세요."
            ) from exc
        raise


def keepalive_user_data_stream() -> str:
    """활성 user data stream listenKey TTL 연장."""
    r = _http.put(
        f"{_cfg.BINANCE_FUTURES_URL}/fapi/v1/listenKey",
        headers=_api_key_headers(),
        timeout=8,
    )
    r.raise_for_status()
    return str(r.json().get("listenKey") or "")


def close_user_data_stream() -> None:
    """활성 user data stream 종료."""
    r = _http.delete(
        f"{_cfg.BINANCE_FUTURES_URL}/fapi/v1/listenKey",
        headers=_api_key_headers(),
        timeout=8,
    )
    r.raise_for_status()


def _signed_get(endpoint: str, params: dict) -> dict:
    """Binance HMAC-SHA256 서명 GET — market_context.py와 동일한 패턴 사용"""
    if not _cfg.BINANCE_API_KEY or not _cfg.BINANCE_SECRET_KEY:
        raise RuntimeError("BINANCE_API_KEY 또는 BINANCE_SECRET_KEY가 비어 있습니다.")

    params["timestamp"] = int(_time.time() * 1000)
    query = "&".join(f"{k}={v}" for k, v in params.items())
    sig = hmac.new(
        _cfg.BINANCE_SECRET_KEY.encode(), query.encode(), hashlib.sha256
    ).hexdigest()
    r = _http.get(
        f"{_cfg.BINANCE_FUTURES_URL}{endpoint}",
        params={**params, "signature": sig},
        headers={"X-MBX-APIKEY": _cfg.BINANCE_API_KEY},
        timeout=8,
    )
    r.raise_for_status()
    return r.json()


def _fetch_balance(ctx: dict) -> None:
    """
    잔고 수집 — 두 엔드포인트 순차 시도로 안정성 확보.
    1순위: /fapi/v2/balance  (자산별 조회, 선물계좌 잔고에 가장 정확)
    2순위: /fapi/v2/account  (계좌 전체 요약)
    """
    # ── 1순위: fapi/v2/balance ─────────────────
    try:
        assets = _signed_get("/fapi/v2/balance", {})
        tracked_assets = []
        for asset in assets:
            name = str(asset.get("asset") or "").upper()
            if name not in TRACKED_COLLATERAL_ASSETS:
                continue

            balance = float(asset.get("balance") or 0)
            available = float(asset.get("availableBalance") or 0)
            upnl = float(asset.get("crossUnPnl") or 0)
            margin = float(asset.get("crossWalletBalance") or asset.get("balance") or 0)
            if max(abs(balance), abs(available), abs(upnl), abs(margin)) <= 1e-9:
                continue

            tracked_assets.append({
                "asset": name,
                "wallet_balance": balance,
                "available_balance": available,
                "unrealized_pnl": upnl,
                "margin_balance": margin,
            })

        if tracked_assets:
            ctx["balance_assets"] = tracked_assets
            ctx["wallet_balance"] = sum(a["wallet_balance"] for a in tracked_assets)
            ctx["available_balance"] = sum(a["available_balance"] for a in tracked_assets)
            ctx["unrealized_pnl"] = sum(a["unrealized_pnl"] for a in tracked_assets)
            ctx["margin_balance"] = sum(a["margin_balance"] for a in tracked_assets)
            ctx["balance_error"] = None
            return
    except Exception as exc:
        first_error = exc
    else:
        first_error = RuntimeError("USDT/USDC 담보 자산을 찾지 못했습니다.")

    # ── 2순위: fapi/v2/account (폴백) ──────────
    try:
        data = _signed_get("/fapi/v2/account", {})
        ctx["wallet_balance"]    = float(data["totalWalletBalance"])
        ctx["unrealized_pnl"]    = float(data["totalUnrealizedProfit"])
        ctx["available_balance"] = float(data["availableBalance"])
        ctx["margin_balance"]    = float(data["totalMarginBalance"])
        ctx["balance_assets"]    = None
        ctx["balance_error"]     = None
    except Exception as exc:
        ctx["wallet_balance"]    = None
        ctx["unrealized_pnl"]    = None
        ctx["available_balance"] = None
        ctx["margin_balance"]    = None
        ctx["balance_assets"]    = None
        ctx["balance_error"]     = (
            f"1차: {_safe_error_message(first_error)} | "
            f"2차: {_safe_error_message(exc)}"
        )


def _account_equity(ctx: dict) -> float | None:
    wallet = ctx.get("wallet_balance")
    upnl = ctx.get("unrealized_pnl")
    try:
        if wallet is not None and upnl is not None:
            return float(wallet) + float(upnl)
        if ctx.get("margin_balance") is not None:
            return float(ctx["margin_balance"])
    except (TypeError, ValueError):
        return None
    return None


def _income_cache_key(symbol: Optional[str], start_ms: int) -> tuple[str, int]:
    return (str(symbol or "").upper(), int(start_ms))


_INCOME_PAGE_LIMIT = 1000  # Binance 단일 요청 최대값


def _fetch_income_all(income_type: str, base_params: dict) -> list:
    """1000건 초과 거래를 페이지네이션으로 전량 수집."""
    records: list = []
    params = {**base_params, "incomeType": income_type, "limit": _INCOME_PAGE_LIMIT}
    while True:
        page = _signed_get("/fapi/v1/income", params)
        if not page:
            break
        records.extend(page)
        if len(page) < _INCOME_PAGE_LIMIT:
            break
        # 마지막 항목의 time + 1ms 를 startTime으로 설정해 다음 페이지 조회
        last_time = int(page[-1].get("time") or 0)
        params = {**params, "startTime": last_time + 1}
    return records


def _fetch_income_summary(symbol: Optional[str], start_ms: int) -> dict:
    cache_key = _income_cache_key(symbol, start_ms)
    now_mono = _time.monotonic()

    with _INCOME_CACHE_LOCK:
        cached = _INCOME_CACHE.get(cache_key)
        if cached and cached.get("expires_at", 0.0) > now_mono:
            return dict(cached["value"])

    base_params: dict = {"startTime": start_ms}
    if symbol:
        base_params["symbol"] = symbol

    pnl_records        = _fetch_income_all("REALIZED_PNL", base_params)
    fee_records        = _fetch_income_all("FUNDING_FEE",  base_params)
    commission_records = _fetch_income_all("COMMISSION",   base_params)

    trade_keys = {
        (item.get("symbol"), item.get("time"), item.get("tranId"))
        for item in [*pnl_records, *commission_records]
    }
    summary = {
        "realized":    sum(float(item["income"]) for item in pnl_records),
        "funding":     sum(float(item["income"]) for item in fee_records),
        "commission":  sum(float(item["income"]) for item in commission_records),
        "trade_count": len(trade_keys),
    }

    with _INCOME_CACHE_LOCK:
        expired_keys = [
            key for key, value in _INCOME_CACHE.items()
            if value.get("expires_at", 0.0) <= now_mono
        ]
        for key in expired_keys:
            _INCOME_CACHE.pop(key, None)
        _INCOME_CACHE[cache_key] = {
            "expires_at": now_mono + INCOME_CACHE_TTL_SECS,
            "value": summary,
        }

    return dict(summary)


def fetch_account_context(symbol: Optional[str] = None) -> dict:
    """
    Binance Futures API로 계좌 현황을 수집.
    개별 요청 실패 시 None으로 채워 분석 전체를 블로킹하지 않음.
    """
    ctx: dict = {}

    # ── 잔고 ──────────────────────────────────
    _fetch_balance(ctx)
    ctx["account_equity"] = _account_equity(ctx)

    # ── 오픈 포지션 ───────────────────────────
    try:
        params = {"symbol": symbol} if symbol else {}
        positions = _signed_get("/fapi/v2/positionRisk", params)
        open_pos = [p for p in positions if abs(float(p["positionAmt"])) > 0]
        ctx["open_positions"] = []
        for p in open_pos:
            amt      = float(p["positionAmt"])
            entry    = float(p["entryPrice"])
            lev      = int(p["leverage"])
            upnl     = float(p["unRealizedProfit"])
            notional = abs(amt) * entry
            margin   = notional / lev if lev > 0 else 0
            pnl_pct  = (upnl / margin * 100) if margin > 0 else 0
            ctx["open_positions"].append({
                "symbol":             p["symbol"],
                "margin_asset":       p.get("marginAsset"),
                "side":               "롱" if amt > 0 else "숏",
                "size":               abs(amt),
                "entry_price":        entry,
                "mark_price":         float(p["markPrice"]),
                "unrealized_pnl":     upnl,
                "unrealized_pnl_pct": pnl_pct,
                "leverage":           lev,
                "liquidation_price":  float(p["liquidationPrice"]),
                "margin_type":        p["marginType"],
                "notional":           notional,
            })
        ctx["open_positions"].sort(key=lambda p: p["notional"], reverse=True)
        ctx["open_position_count"] = len(ctx["open_positions"])
        ctx["open_position_notional"] = sum(p["notional"] for p in ctx["open_positions"])
        ctx["open_position_upnl"] = sum(p["unrealized_pnl"] for p in ctx["open_positions"])
        leverages = [p["leverage"] for p in ctx["open_positions"] if p.get("leverage") is not None]
        if leverages:
            lev_min = min(leverages)
            lev_max = max(leverages)
            total_notional = ctx["open_position_notional"] or 0
            if total_notional > 0:
                weighted = sum(p["notional"] * p["leverage"] for p in ctx["open_positions"]) / total_notional
            else:
                weighted = sum(leverages) / len(leverages)

            ctx["effective_leverage"] = weighted if lev_min != lev_max else float(lev_min)
            ctx["leverage_min"] = lev_min
            ctx["leverage_max"] = lev_max
            ctx["leverage_weighted"] = weighted
            if lev_min == lev_max:
                ctx["leverage_display"] = f"{lev_min}x (실제 포지션 기준)"
                ctx["leverage_mode"] = "single"
            else:
                ctx["leverage_display"] = (
                    f"혼합 {lev_min}x~{lev_max}x "
                    f"(가중평균 {weighted:.1f}x)"
                )
                ctx["leverage_mode"] = "mixed"
        else:
            ctx["effective_leverage"] = None
            ctx["leverage_min"] = None
            ctx["leverage_max"] = None
            ctx["leverage_weighted"] = None
            ctx["leverage_display"] = f"오픈 포지션 없음 (기본 {_cfg.DEFAULT_LEVERAGE}x)"
            ctx["leverage_mode"] = "default"
        ctx["position_error"] = None
    except Exception as exc:
        ctx["open_positions"] = None
        ctx["open_position_count"] = None
        ctx["open_position_notional"] = None
        ctx["open_position_upnl"] = None
        ctx["effective_leverage"] = None
        ctx["leverage_min"] = None
        ctx["leverage_max"] = None
        ctx["leverage_weighted"] = None
        ctx["leverage_display"] = "조회 실패"
        ctx["leverage_mode"] = "error"
        ctx["position_error"] = _safe_error_message(exc)

    # ── 미체결 주문 (TP / SL / 지정가) ─────────────────
    try:
        raw_orders = _signed_get("/fapi/v1/openOrders", {"timestamp": int(_time.time() * 1000)})
        open_orders = []
        for o in (raw_orders if isinstance(raw_orders, list) else []):
            stop_price = float(o.get("stopPrice", 0) or 0)
            limit_price = float(o.get("price", 0) or 0)
            price = stop_price if stop_price > 0 else limit_price
            if price <= 0:
                continue
            open_orders.append({
                "symbol":      o.get("symbol", ""),
                "order_id":    o.get("orderId"),
                "type":        o.get("type", ""),
                "side":        o.get("side", ""),
                "price":       price,
                "qty":         float(o.get("origQty", 0) or 0),
                "reduce_only": bool(o.get("reduceOnly", False)),
            })

        # 포지션에 TP / SL 매칭
        # TP: TAKE_PROFIT_MARKET, TAKE_PROFIT, 또는 reduce-only LIMIT (롱이면 높은 가격, 숏이면 낮은 가격)
        TP_TYPES = {"TAKE_PROFIT_MARKET", "TAKE_PROFIT"}
        SL_TYPES = {"STOP_MARKET", "STOP"}
        matched_order_ids: set = set()
        if ctx.get("open_positions"):
            for pos in ctx["open_positions"]:
                sym = pos["symbol"]
                entry = pos["entry_price"]
                close_side = "SELL" if pos["side"] == "롱" else "BUY"
                sym_orders = [o for o in open_orders if o["symbol"] == sym
                              and o["reduce_only"] and o["side"] == close_side]

                # 표준 TP/SL 오더 타입 먼저 매칭
                tp_order = next((o for o in sym_orders if o["type"] in TP_TYPES), None)
                sl_order = next((o for o in sym_orders if o["type"] in SL_TYPES), None)

                # reduce-only LIMIT 오더 → 진입가보다 유리한 방향이면 TP, 불리한 방향이면 SL
                if tp_order is None or sl_order is None:
                    for o in sym_orders:
                        if o["type"] != "LIMIT":
                            continue
                        p = o["price"]
                        is_tp_side = (close_side == "SELL" and p > entry) or \
                                     (close_side == "BUY"  and p < entry)
                        if is_tp_side and tp_order is None:
                            tp_order = o
                        elif not is_tp_side and sl_order is None:
                            sl_order = o

                pos["tp_price"] = tp_order["price"] if tp_order else None
                pos["sl_price"] = sl_order["price"] if sl_order else None
                for o in (tp_order, sl_order):
                    if o:
                        matched_order_ids.add(o["order_id"])

        # TP/SL로 이미 매칭된 것 제외한 나머지 오더 전체 노출 (LIMIT, 미체결 진입 등)
        ctx["open_orders"] = [o for o in open_orders if o["order_id"] not in matched_order_ids]
        ctx["order_error"] = None
    except Exception as exc:
        ctx["open_orders"] = []
        ctx["order_error"] = _safe_error_message(exc)

    # ── 오늘 손익: 실현 손익 + 펀딩비 (KST 00:00 기준) ──
    try:
        today_start_ms = int(start_of_kst_day().astimezone(timezone.utc).timestamp() * 1000)
        income_summary = _fetch_income_summary(symbol, today_start_ms)
        today_realized = income_summary["realized"]
        today_funding = income_summary["funding"]
        today_commission = income_summary["commission"]
        ctx["today_trade_count"] = income_summary["trade_count"]

        ctx["today_realized_pnl"] = today_realized
        ctx["today_funding_fee"]  = today_funding
        ctx["today_commission_fee"] = today_commission
        ctx["today_cash_pnl"] = today_realized + today_funding + today_commission
        ctx["today_eval_pnl"] = None
        ctx["today_total_pnl"] = ctx["today_cash_pnl"]
        ctx["today_total_mode"] = "cash"
        ctx["today_total_label"] = "금일 현금손익"
        ctx["day_start_equity"] = None
        ctx["day_anchor_source"] = "cash"
        ctx["carryover_positions"] = []
        ctx["pnl_error"]          = None
    except Exception as exc:
        ctx["today_realized_pnl"] = None
        ctx["today_funding_fee"]  = None
        ctx["today_commission_fee"] = None
        ctx["today_cash_pnl"] = None
        ctx["today_eval_pnl"] = None
        ctx["today_total_pnl"]    = None
        ctx["today_total_mode"] = None
        ctx["today_total_label"] = None
        ctx["day_start_equity"] = None
        ctx["day_anchor_source"] = None
        ctx["carryover_positions"] = []
        ctx["today_trade_count"]  = None
        ctx["pnl_error"]          = _safe_error_message(exc)

    # ── 사용자 설정 ───────────────────────────
    ctx["configured_leverage"]  = _cfg.DEFAULT_LEVERAGE

    # ── UI / 보고용 요약 필드 ──────────────────
    wallet = ctx.get("wallet_balance")
    equity = ctx.get("account_equity")
    total_pnl = ctx.get("today_total_pnl")
    ctx["today_pnl_pct"] = None

    if equity is not None and total_pnl is not None:
        # wallet_balance 기준: unrealized PnL을 분모에서 제외해 수익률 오차 방지
        # equity(=wallet+unrealized)를 쓰면 오픈 포지션 규모만큼 분모가 부풀려짐
        wallet = ctx.get("wallet_balance")
        cash_pnl = ctx.get("today_cash_pnl") or total_pnl
        if wallet is not None:
            start_balance = wallet - cash_pnl
        else:
            start_balance = equity - total_pnl
        if start_balance <= 0:
            start_balance = equity - total_pnl  # fallback
        if ctx.get("day_start_equity") is None and start_balance > 0:
            ctx["day_start_equity"] = start_balance
        today_pct = (total_pnl / start_balance * 100) if start_balance > 0 else 0
        ctx["today_pnl_pct"] = today_pct

    attach_account_context_summary(ctx)
    return ctx


def format_account_context(ctx: dict) -> str:
    """현재 스냅샷 + 최근 계좌 운영 맥락을 함께 출력 — 판단은 Claude에게 위임"""
    lines = ["[계좌 / 리스크 제약]"]

    # ── 잔고 ──────────────────────────────────
    wallet = ctx.get("wallet_balance")
    balance_assets = ctx.get("balance_assets") or []
    if wallet is not None:
        if balance_assets:
            assets_str = " / ".join(
                f"{a['asset']} ${a['wallet_balance']:,.2f}"
                for a in balance_assets
            )
            lines.append(f"  담보 자산 잔고:  {assets_str}")
            lines.append(f"  추적 자산 합계:  ${wallet:,.2f} (USDT+USDC)")
        else:
            lines.append(f"  계좌 지갑 잔고:  ${wallet:,.2f}")
        if ctx.get("margin_balance") is not None:
            lines.append(f"  마진 잔고:       ${ctx['margin_balance']:,.2f}")
        if ctx.get("unrealized_pnl") is not None:
            lines.append(f"  계좌 미실현:     ${ctx['unrealized_pnl']:+,.2f}")
        if ctx.get("available_balance") is not None:
            lines.append(f"  사용 가능 잔고:  ${ctx['available_balance']:,.2f}")
    else:
        err = ctx.get("balance_error") or "API 키 권한 또는 네트워크 확인 필요"
        lines.append(f"  잔고 조회 실패 — {err}")

    # ── 일일 손익 & 목표/한도 ─────────────────
    total_pnl = ctx.get("today_total_pnl")
    total_label = ctx.get("today_total_label") or "오늘 손익"
    total_mode = ctx.get("today_total_mode") or "cash"
    cash_pnl = ctx.get("today_cash_pnl")
    realized  = ctx.get("today_realized_pnl")
    funding   = ctx.get("today_funding_fee")
    commission = ctx.get("today_commission_fee")
    anchor_source = ctx.get("day_anchor_source") or ""
    lev_display = ctx.get("leverage_display")
    start_equity = ctx.get("day_start_equity")
    current_equity = ctx.get("account_equity")

    if total_pnl is not None and current_equity is not None:
        if start_equity is not None:
            start_balance = start_equity
        else:
            # wallet_balance 기준 역산: unrealized PnL을 분모에서 제외
            _wallet = ctx.get("wallet_balance")
            _cash = cash_pnl if cash_pnl is not None else total_pnl
            start_balance = (_wallet - _cash) if _wallet is not None else (current_equity - total_pnl)
            if start_balance <= 0:
                start_balance = current_equity - total_pnl
        today_pct = (total_pnl / start_balance * 100) if start_balance > 0 else 0

        lines.append(f"  {total_label}(KST): ${total_pnl:+,.2f} ({today_pct:+.2f}%)")
        detail_lines: list[str] = []
        if total_mode == "evaluation" and start_equity is not None:
            detail_lines.append(
                f"자정 기준 평가: ${start_equity:,.2f} → 현재 ${current_equity:,.2f}"
            )
        if total_mode == "evaluation" and anchor_source == "prev_close":
            detail_lines.append("평가 기준:  전일 마지막 표본으로 보정")
        if total_mode == "cash" and anchor_source == "cash_fallback":
            detail_lines.append("평가 기준:  자정 인근 표본 대기 중")
        if cash_pnl is not None and total_mode == "evaluation":
            detail_lines.append(f"현금손익:  ${cash_pnl:+,.2f}")
        if realized is not None:
            detail_lines.append(f"실현 손익:  ${realized:+,.2f}")
        if funding is not None:
            detail_lines.append(f"펀딩비:     ${funding:+,.2f}")
        if commission is not None:
            detail_lines.append(f"거래 수수료: ${commission:+,.2f}")
        if ctx.get("open_position_upnl") is not None:
            detail_lines.append(f"현재 미실현: ${ctx['open_position_upnl']:+,.2f}")
        for idx, detail in enumerate(detail_lines):
            branch = "└" if idx == len(detail_lines) - 1 else "├"
            lines.append(f"    {branch} {detail}")
        if ctx.get("today_trade_count") is not None:
            lines.append(f"  오늘 거래 기록:  {ctx['today_trade_count']}건")
    else:
        err = ctx.get("pnl_error") or "income 조회 실패"
        lines.append(f"  오늘 손익 조회 실패 — {err}")

    if lev_display:
        lines.append(f"  레버리지 상태:   {lev_display}")
    else:
        lines.append(f"  레버리지 상태:   N/A")

    summary = ctx.get("context_summary") or {}
    summary_sections = [section for section in (summary.get("sections") or []) if section]
    summary_lines = [line for line in (summary.get("lines") or []) if line]
    if summary_sections:
        lines.append("  최근 계좌 운영 맥락:")
        for section in summary_sections:
            label = section.get("label") or "계좌 맥락"
            lines.append(f"    [{label}]")
            for line in section.get("lines") or []:
                lines.append(f"      {line}")
    elif summary_lines:
        lines.append("  최근 계좌 운영 맥락:")
        for line in summary_lines:
            lines.append(f"    {line}")

    # ── 오픈 포지션 ───────────────────────────
    positions = ctx.get("open_positions")
    if positions is None:
        err = ctx.get("position_error") or "positionRisk 조회 실패"
        lines.append(f"  포지션 조회 실패 — {err}")
    elif not positions:
        lines.append("  현재 오픈 포지션: 없음")
    else:
        count = ctx.get("open_position_count", len(positions))
        total_notional = ctx.get("open_position_notional")
        total_upnl = ctx.get("open_position_upnl")
        summary = f"  오픈 포지션: {count}개"
        if total_notional is not None:
            summary += f" / 총 명목 ${total_notional:,.2f}"
        if total_upnl is not None:
            summary += f" / 총 미실현 ${total_upnl:+,.2f}"
        lines.append(summary)
        preview = positions[:4]
        for p in preview:
            margin_asset = p.get("margin_asset") or "N/A"
            lines.append(
                f"    [{p['symbol']} {p['side']} / 담보 {margin_asset}]  "
                f"수량 {p['size']}  진입 ${p['entry_price']:,.2f}  "
                f"현재가 ${p['mark_price']:,.2f}  "
                f"미실현 ${p['unrealized_pnl']:+,.2f} ({p['unrealized_pnl_pct']:+.2f}%)  "
                f"레버리지 {p['leverage']}x  청산가 ${p['liquidation_price']:,.2f}"
            )
        remaining = len(positions) - len(preview)
        if remaining > 0:
            lines.append(f"    외 {remaining}개 포지션은 노이즈 방지를 위해 생략")

    return "\n".join(lines)
