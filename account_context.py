# =============================================
# 내 계좌 상태 수집 (Binance Futures)
# 잔고 / 오픈 포지션 / 오늘 실현 손익
# =============================================
import hmac
import hashlib
import time as _time
import requests
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlsplit
import config as _cfg

TRACKED_COLLATERAL_ASSETS = ("USDT", "USDC")


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
        r = requests.post(
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
    r = requests.put(
        f"{_cfg.BINANCE_FUTURES_URL}/fapi/v1/listenKey",
        headers=_api_key_headers(),
        timeout=8,
    )
    r.raise_for_status()
    return str(r.json().get("listenKey") or "")


def close_user_data_stream() -> None:
    """활성 user data stream 종료."""
    r = requests.delete(
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
    r = requests.get(
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


def fetch_account_context(symbol: Optional[str] = None) -> dict:
    """
    Binance Futures API로 계좌 현황을 수집.
    개별 요청 실패 시 None으로 채워 분석 전체를 블로킹하지 않음.
    """
    ctx: dict = {}

    # ── 잔고 ──────────────────────────────────
    _fetch_balance(ctx)

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

    # ── 오늘 손익: 실현 손익 + 펀딩비 (UTC 00:00 기준) ──
    try:
        now_utc = datetime.now(timezone.utc)
        today_start_ms = int(
            now_utc.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000
        )
        base_params = {"startTime": today_start_ms, "limit": 1000}
        if symbol:
            base_params["symbol"] = symbol

        # 실현 손익
        pnl_records = _signed_get(
            "/fapi/v1/income", {**base_params, "incomeType": "REALIZED_PNL"}
        )
        today_realized = sum(float(i["income"]) for i in pnl_records)
        # 거래 기록 수: 멀티 심볼 기준으로도 충돌이 적도록 symbol/time/tranId 조합 사용
        trade_keys = {
            (i.get("symbol"), i.get("time"), i.get("tranId"))
            for i in pnl_records
        }
        ctx["today_trade_count"] = len(trade_keys)

        # 펀딩비 (수취 +, 지불 -)
        fee_records = _signed_get(
            "/fapi/v1/income", {**base_params, "incomeType": "FUNDING_FEE"}
        )
        today_funding = sum(float(i["income"]) for i in fee_records)

        ctx["today_realized_pnl"] = today_realized
        ctx["today_funding_fee"]  = today_funding
        ctx["today_total_pnl"]    = today_realized + today_funding
        ctx["pnl_error"]          = None
    except Exception as exc:
        ctx["today_realized_pnl"] = None
        ctx["today_funding_fee"]  = None
        ctx["today_total_pnl"]    = None
        ctx["today_trade_count"]  = None
        ctx["pnl_error"]          = _safe_error_message(exc)

    # ── 사용자 설정 ───────────────────────────
    ctx["daily_target_pct"]     = _cfg.DAILY_TARGET_PCT
    ctx["daily_loss_limit_pct"] = _cfg.DAILY_LOSS_LIMIT_PCT
    ctx["configured_leverage"]  = _cfg.DEFAULT_LEVERAGE

    # ── UI / 보고용 요약 필드 ──────────────────
    wallet = ctx.get("wallet_balance")
    total_pnl = ctx.get("today_total_pnl")
    target = ctx.get("daily_target_pct")
    loss_lim = ctx.get("daily_loss_limit_pct")
    ctx["today_pnl_pct"] = None
    ctx["remaining_to_target_pct"] = None
    ctx["risk_status"] = None

    if wallet is not None and total_pnl is not None:
        start_balance = wallet - total_pnl
        today_pct = (total_pnl / start_balance * 100) if start_balance > 0 else 0
        ctx["today_pnl_pct"] = today_pct
        if target is not None:
            ctx["remaining_to_target_pct"] = target - today_pct
        if target is not None and today_pct >= target:
            ctx["risk_status"] = "target_hit"
        elif loss_lim is not None and today_pct <= loss_lim:
            ctx["risk_status"] = "loss_limit_hit"
        else:
            ctx["risk_status"] = "active"

    return ctx


def format_account_context(ctx: dict) -> str:
    """수집된 계좌 정보를 원시값 그대로 출력 — 판단은 Claude에게 위임"""
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
    realized  = ctx.get("today_realized_pnl")
    funding   = ctx.get("today_funding_fee")
    target    = ctx.get("daily_target_pct", 0.7)
    loss_lim  = ctx.get("daily_loss_limit_pct", -2.0)
    lev_display = ctx.get("leverage_display")

    if total_pnl is not None and wallet:
        # 오늘 시작 잔고 기준으로 수익률 계산 (현재 잔고 - 오늘 손익)
        start_balance = wallet - total_pnl
        today_pct = (total_pnl / start_balance * 100) if start_balance > 0 else 0
        remaining = target - today_pct
        status    = ""
        if today_pct >= target:
            status = "  ⚠️ 일일 목표 달성 — 신규 진입 자제 권장"
        elif today_pct <= loss_lim:
            status = "  🚫 일일 손실 한도 도달 — 추가 거래 중단 권장"

        lines.append(f"  오늘 총 손익:    ${total_pnl:+,.2f} ({today_pct:+.2f}%)")
        if realized is not None and funding is not None:
            lines.append(f"    ├ 실현 손익:  ${realized:+,.2f}")
            lines.append(f"    └ 펀딩비:     ${funding:+,.2f}")
        lines.append(
            f"  일일 목표:       +{target:.1f}%  |  손실 한도: {loss_lim:.1f}%  |  잔여: {remaining:+.2f}%"
        )
        if status:
            lines.append(status)
        if ctx.get("today_trade_count") is not None:
            lines.append(f"  오늘 거래 기록:  {ctx['today_trade_count']}건")
    else:
        err = ctx.get("pnl_error") or "income 조회 실패"
        lines.append(f"  오늘 손익 조회 실패 — {err}")
        lines.append(f"  일일 목표: +{target:.1f}%  |  손실 한도: {loss_lim:.1f}%")

    if lev_display:
        lines.append(f"  레버리지 상태:   {lev_display}")
    else:
        lines.append(f"  레버리지 상태:   N/A")

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
