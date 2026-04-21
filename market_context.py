# =============================================
# 시장 심리 & 파생상품 데이터 수집
# ─ 펀딩비 히스토리 / 테이커 매수매도 비율
# ─ 오픈 인터레스트 / 톱 트레이더 포지션 비율
# ─ 오더북 깊이 / 강제청산 합계
# ─ Deribit DVOL / 25-delta 스큐
# =============================================
import hmac
import hashlib
import time as _time
import requests
from datetime import datetime, timezone
from config import BINANCE_FUTURES_URL, BINANCE_API_KEY, BINANCE_SECRET_KEY, DEFAULT_SYMBOL
from time_utils import format_kst
from http_client import _session as _http  # 프록시 환경변수 무시 세션


def _signed_get(url: str, params: dict) -> requests.Response:
    """Binance SIGNED 엔드포인트 호출 (HMAC SHA256)"""
    params["timestamp"] = int(_time.time() * 1000)
    query = "&".join(f"{k}={v}" for k, v in params.items())
    sig = hmac.new(
        BINANCE_SECRET_KEY.encode(), query.encode(), hashlib.sha256
    ).hexdigest()
    return _http.get(
        url,
        params={**params, "signature": sig},
        headers={"X-MBX-APIKEY": BINANCE_API_KEY},
        timeout=6,
    )


# ══════════════════════════════════════════════
# Binance Futures
# ══════════════════════════════════════════════

def _fetch_liquidation_events(symbol: str, ctx: dict) -> None:
    """
    Binance forceOrders API로 최근 4시간 강제청산 합계 수집.
    side=SELL → 롱청산 / side=BUY → 숏청산(스퀴즈). 총계만 집계 (개별 건 불필요).
    """
    try:
        start_ms = int((_time.time() - 4 * 3600) * 1000)
        r = _signed_get(
            f"{BINANCE_FUTURES_URL}/fapi/v1/forceOrders",
            {"symbol": symbol, "autoCloseType": "LIQUIDATION",
             "startTime": start_ms, "limit": 100},
        )
        r.raise_for_status()
        orders = r.json()

        long_usd = short_usd = 0.0
        for o in orders:
            qty   = float(o.get("executedQty") or o.get("origQty") or 0)
            price = float(o.get("avgPrice") or o.get("price") or 0)
            usd   = qty * price
            if o["side"] == "SELL":
                long_usd += usd
            else:
                short_usd += usd

        ctx["liquidation_events"] = bool(orders)   # 이벤트 유무 (출력용 flag)
        ctx["liq_long_usd"]       = long_usd
        ctx["liq_short_usd"]      = short_usd

    except Exception:
        ctx["liquidation_events"] = None
        ctx["liq_long_usd"]       = None
        ctx["liq_short_usd"]      = None


def _fetch_binance(symbol: str, ctx: dict) -> None:
    """펀딩비·OI·롱숏비율·테이커비율 일괄 수집 (ctx에 직접 저장)"""

    # ── 펀딩비 현재 + 마크가격 ──────────────────
    try:
        r = _http.get(
            f"{BINANCE_FUTURES_URL}/fapi/v1/premiumIndex",
            params={"symbol": symbol}, timeout=5,
        )
        r.raise_for_status()
        d = r.json()
        ctx["funding_rate"]    = float(d["lastFundingRate"]) * 100
        ctx["next_funding_ms"] = int(d["nextFundingTime"])
        ctx["mark_price"]      = float(d["markPrice"])
        ctx["index_price"]     = float(d["indexPrice"])
    except Exception:
        ctx["funding_rate"] = ctx["next_funding_ms"] = None
        ctx["mark_price"]   = ctx["index_price"]     = None

    # ── 펀딩비 히스토리 (최근 8회 ≈ 24시간) ────
    try:
        r = _http.get(
            f"{BINANCE_FUTURES_URL}/fapi/v1/fundingRate",
            params={"symbol": symbol, "limit": 8}, timeout=5,
        )
        r.raise_for_status()
        ctx["funding_history"] = [
            float(h["fundingRate"]) * 100 for h in r.json()
        ]
    except Exception:
        ctx["funding_history"] = None

    # ── 오픈 인터레스트 (현재) ──────────────────
    try:
        r = _http.get(
            f"{BINANCE_FUTURES_URL}/fapi/v1/openInterest",
            params={"symbol": symbol}, timeout=5,
        )
        r.raise_for_status()
        ctx["open_interest"] = float(r.json()["openInterest"])
    except Exception:
        ctx["open_interest"] = None

    # ── OI 24h 변화율 ───────────────────────────
    try:
        r = _http.get(
            f"{BINANCE_FUTURES_URL}/futures/data/openInterestHist",
            params={"symbol": symbol, "period": "1h", "limit": 25}, timeout=5,
        )
        r.raise_for_status()
        hist = r.json()
        if len(hist) >= 2:
            oi_now = float(hist[-1]["sumOpenInterest"])
            oi_24h = float(hist[0]["sumOpenInterest"])
            ctx["oi_change_24h_pct"] = (oi_now - oi_24h) / oi_24h * 100 if oi_24h else None
        else:
            ctx["oi_change_24h_pct"] = None
    except Exception:
        ctx["oi_change_24h_pct"] = None

    # ── 테이커 매수/매도 비율 (최근 24h 히스토리 + CVD) ──
    # 4h → 24h로 확장: CVD 방향 편향 계산을 위해 더 긴 구간 필요
    try:
        r = _http.get(
            f"{BINANCE_FUTURES_URL}/futures/data/takerlongshortRatio",
            params={"symbol": symbol, "period": "1h", "limit": 24}, timeout=5,
        )
        r.raise_for_status()
        data = r.json()
        if data:
            taker_hist = [
                {"ratio": float(d["buySellRatio"]),
                 "buy":   float(d["buyVol"]),
                 "sell":  float(d["sellVol"])}
                for d in data
            ]
            ctx["taker_history"] = taker_hist

            # ── CVD (Cumulative Volume Delta) ────────
            # CVD = 누적(매수 테이커량 - 매도 테이커량)
            # 양수 편향: 매수 우세 / 음수 편향: 매도 우세
            cvd_series = []
            cumulative = 0.0
            for h in taker_hist:
                delta = h["buy"] - h["sell"]
                cumulative += delta
                cvd_series.append(round(cumulative, 2))

            ctx["cvd_series"]  = cvd_series          # 24h 시계열 (오래된→최근)
            ctx["cvd_current"] = cvd_series[-1]       # 최종 누적값
            ctx["cvd_4h"]      = sum(                 # 최근 4h 델타 (단기 모멘텀)
                h["buy"] - h["sell"] for h in taker_hist[-4:]
            )
        else:
            ctx["taker_history"] = ctx["cvd_series"] = ctx["cvd_current"] = ctx["cvd_4h"] = None
    except Exception:
        ctx["taker_history"] = ctx["cvd_series"] = ctx["cvd_current"] = ctx["cvd_4h"] = None


# ══════════════════════════════════════════════
# 톱 트레이더 롱/숏 비율
# ══════════════════════════════════════════════

def _fetch_top_trader_ratios(symbol: str, ctx: dict) -> None:
    """
    톱 트레이더 포지션 비율 수집.
    ⚠️ API 명명 주의: topLongShortPositionRatio 엔드포인트는 포지션 규모 비율을 반환하지만
    응답 필드명이 longAccount/shortAccount로 표기됨 (Binance 불일치 명명).
    계좌 수 기반 비율(topLongShortAccountRatio)과 혼동 주의 — 실제값 검증 권장.
    """
    try:
        r = _http.get(
            f"{BINANCE_FUTURES_URL}/futures/data/topLongShortPositionRatio",
            params={"symbol": symbol, "period": "1h", "limit": 1}, timeout=5,
        )
        r.raise_for_status()
        data = r.json()
        if data:
            ctx["top_pos_long"]  = float(data[0]["longAccount"])  * 100
            ctx["top_pos_short"] = float(data[0]["shortAccount"]) * 100
        else:
            ctx["top_pos_long"] = ctx["top_pos_short"] = None
    except Exception:
        ctx["top_pos_long"] = ctx["top_pos_short"] = None






# ══════════════════════════════════════════════
# Deribit 옵션 시장
# ══════════════════════════════════════════════

def _fetch_deribit(btc_price: float, ctx: dict) -> None:
    """DVOL 지수 + 25-delta 스큐 근사 (7~45일 만기 기준)"""

    # ── DVOL ────────────────────────────────────
    try:
        r = _http.get(
            "https://www.deribit.com/api/v2/public/get_index_price",
            params={"index_name": "btcdvol_usdc"}, timeout=8,
        )
        r.raise_for_status()
        ctx["dvol"] = round(r.json()["result"]["index_price"], 1)
    except Exception:
        ctx["dvol"] = None

    # ── 25-delta 스큐 ────────────────────────────
    try:
        r = _http.get(
            "https://www.deribit.com/api/v2/public/get_book_summary_by_currency",
            params={"currency": "BTC", "kind": "option"}, timeout=12,
        )
        r.raise_for_status()
        options = r.json()["result"]

        now = datetime.now(timezone.utc)

        # 만기 파싱 + 7~45일 필터
        parsed = []
        for o in options:
            iv = o.get("mark_iv")
            if not iv or iv <= 0:
                continue
            parts = o["instrument_name"].split("-")
            if len(parts) != 4:
                continue
            try:
                exp = datetime.strptime(parts[1], "%d%b%y").replace(
                    tzinfo=timezone.utc, hour=8
                )
            except ValueError:
                continue
            days = (exp - now).days
            if not (7 <= days <= 45):
                continue
            parsed.append({
                "expiry": exp,
                "days":   days,
                "strike": float(parts[2]),
                "type":   parts[3],   # C / P
                "iv":     float(iv),
                "underlying": float(o.get("underlying_price") or btc_price),
            })

        if not parsed:
            ctx["skew_25d"] = ctx["skew_expiry"] = ctx["skew_call_iv"] = ctx["skew_put_iv"] = None
            return

        # 가장 가까운 만기 선택
        front_days = min(o["days"] for o in parsed)
        front = [o for o in parsed if o["days"] == front_days]
        spot  = front[0]["underlying"]

        # 25-delta 근사: IV 수준에 따라 OTM 거리 동적 계산
        # delta ≈ N(ln(K/S) / (σ√T)) 역산 → σ√T * 0.674 ≈ 25d 이격
        dvol_frac = (ctx.get("dvol") or 60) / 100
        t_sqrt    = (front_days / 365) ** 0.5
        offset    = dvol_frac * t_sqrt * 0.674   # ln(K/S) ≈ ±offset

        import math
        call_target = spot * math.exp(offset)
        put_target  = spot * math.exp(-offset)

        calls = sorted(
            [o for o in front if o["type"] == "C"],
            key=lambda x: abs(x["strike"] - call_target),
        )
        puts = sorted(
            [o for o in front if o["type"] == "P"],
            key=lambda x: abs(x["strike"] - put_target),
        )

        if calls and puts:
            ctx["skew_25d"]      = round(puts[0]["iv"] - calls[0]["iv"], 1)
            ctx["skew_expiry"]   = front[0]["expiry"].strftime("%d%b%y")
            ctx["skew_call_iv"]  = round(calls[0]["iv"], 1)
            ctx["skew_put_iv"]   = round(puts[0]["iv"], 1)
            ctx["skew_days_left"] = front_days   # 만기까지 잔여 일수 (오차 판단용)
        else:
            ctx["skew_25d"] = ctx["skew_expiry"] = ctx["skew_call_iv"] = ctx["skew_put_iv"] = ctx["skew_days_left"] = None

    except Exception:
        ctx["skew_25d"] = ctx["skew_expiry"] = ctx["skew_call_iv"] = ctx["skew_put_iv"] = None


# ══════════════════════════════════════════════
# 오더북 불균형 (Bid-Ask Imbalance)
# ══════════════════════════════════════════════

def _fetch_orderbook_imbalance(symbol: str, ctx: dict) -> None:
    """
    Binance Futures depth API로 상위 20레벨 호가 잔량 비율 계산.

    imbalance = 총매수잔량 / (총매수잔량 + 총매도잔량)
      > 0.60 : 매수 우세 (bid-heavy)
      < 0.40 : 매도 우세 (ask-heavy)
      0.40~0.60 : 균형

    ※ 스냅샷 시점 기준 — 대형 지정가 취소 시 즉시 변동. 단기 확인용.
    """
    try:
        r = _http.get(
            f"{BINANCE_FUTURES_URL}/fapi/v1/depth",
            params={"symbol": symbol, "limit": 20},
            timeout=5,
        )
        r.raise_for_status()
        data = r.json()

        bid_qty = sum(float(b[1]) for b in data.get("bids", []))
        ask_qty = sum(float(a[1]) for a in data.get("asks", []))
        total   = bid_qty + ask_qty

        if total > 0:
            ctx["ob_bid_qty"]    = round(bid_qty, 2)
            ctx["ob_ask_qty"]    = round(ask_qty, 2)
            ctx["ob_imbalance"]  = round(bid_qty / total, 4)   # 0.0 ~ 1.0
        else:
            ctx["ob_bid_qty"] = ctx["ob_ask_qty"] = ctx["ob_imbalance"] = None

    except Exception:
        ctx["ob_bid_qty"] = ctx["ob_ask_qty"] = ctx["ob_imbalance"] = None


# ══════════════════════════════════════════════
# Bybit OI (멀티 거래소 OI 합산)
# ══════════════════════════════════════════════

def _fetch_bybit_oi(ctx: dict, symbol: str = "BTCUSDT") -> None:
    """
    Bybit v5 API로 BTC 선물 OI(계약수) 수집 후 BTC 단위로 변환.
    Bybit BTCUSDT 선물 1계약 = 1 USDT (inverse 아님, linear).
    OI(USDT) / mark_price ≈ OI(BTC).

    Binance OI와 합산해 시장 전체 레버리지 규모를 파악.
    """
    try:
        r = _http.get(
            "https://api.bybit.com/v5/market/open-interest",
            params={
                "category":     "linear",
                "symbol":       symbol,
                "intervalTime": "1h",
                "limit":        1,
            },
            timeout=6,
        )
        r.raise_for_status()
        result = r.json().get("result", {})
        items  = result.get("list", [])
        if not items:
            ctx["bybit_oi"] = ctx["combined_oi"] = None
            return

        oi_usdt    = float(items[0]["openInterest"])
        mark_price = ctx.get("mark_price") or ctx.get("index_price") or 80000.0
        bybit_oi_btc = oi_usdt / mark_price

        ctx["bybit_oi"] = round(bybit_oi_btc, 1)

        # 합산 OI
        binance_oi = ctx.get("open_interest")
        if binance_oi is not None:
            ctx["combined_oi"] = round(binance_oi + bybit_oi_btc, 1)
        else:
            ctx["combined_oi"] = None

    except Exception:
        ctx["bybit_oi"] = ctx["combined_oi"] = None


# ══════════════════════════════════════════════
# Fear & Greed Index (Alternative.me)
# ══════════════════════════════════════════════

def _fetch_fear_greed(ctx: dict) -> None:
    """
    Alternative.me Crypto Fear & Greed Index 수집.
    0~24: 극도의 공포 / 25~49: 공포 / 50~74: 탐욕 / 75~100: 극도의 탐욕
    키 없음, 무료.
    """
    try:
        r = _http.get(
            "https://api.alternative.me/fng/?limit=3&format=json",
            timeout=8,
        )
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            ctx["fear_greed"] = ctx["fear_greed_label"] = None
            ctx["fear_greed_history"] = None
            return

        # 최신값
        ctx["fear_greed"]       = int(data[0]["value"])
        ctx["fear_greed_label"] = data[0]["value_classification"]

        # 최근 3일 히스토리 (오래된→최신 순)
        history = []
        for d in reversed(data):
            history.append({
                "value": int(d["value"]),
                "label": d["value_classification"],
            })
        ctx["fear_greed_history"] = history

    except Exception:
        ctx["fear_greed"] = ctx["fear_greed_label"] = None
        ctx["fear_greed_history"] = None


# ══════════════════════════════════════════════
# 공개 인터페이스
# ══════════════════════════════════════════════

def fetch_market_context(symbol: str = DEFAULT_SYMBOL) -> dict:
    """
    파생상품 및 심리 지표를 한 번에 수집.
    개별 요청 실패 시 None으로 채워 분석 전체를 블로킹하지 않음.
    """
    ctx: dict = {}

    _fetch_binance(symbol, ctx)
    _fetch_top_trader_ratios(symbol, ctx)
    _fetch_liquidation_events(symbol, ctx)
    _fetch_orderbook_imbalance(symbol, ctx)
    _fetch_bybit_oi(ctx, symbol)
    _fetch_fear_greed(ctx)

    btc_price = ctx.get("mark_price") or ctx.get("index_price") or 80000.0
    _fetch_deribit(btc_price, ctx)

    return ctx


def format_market_context(ctx: dict) -> str:
    """수집된 시장 맥락 데이터를 원시값 그대로 출력 — 해석은 Claude에게 위임"""
    lines = ["[시장 심리 & 파생상품 데이터]"]

    # ── 펀딩비 현재 + 히스토리 ──────────────────
    if ctx.get("funding_rate") is not None:
        lines.append(f"  펀딩비 (현재): {ctx['funding_rate']:+.4f}%")

        if ctx.get("next_funding_ms"):
            dt = datetime.fromtimestamp(ctx["next_funding_ms"] / 1000, tz=timezone.utc)
            lines.append(f"  다음 펀딩: {format_kst(dt, '%H:%M')} KST")

        if ctx.get("mark_price") and ctx.get("index_price"):
            basis = ctx["mark_price"] - ctx["index_price"]
            lines.append(
                f"  마크가격: ${ctx['mark_price']:,.2f} / "
                f"인덱스: ${ctx['index_price']:,.2f} / "
                f"베이시스: {basis:+.2f}"
            )

    if ctx.get("funding_history"):
        history_str = " → ".join(f"{v:+.4f}%" for v in ctx["funding_history"])
        lines.append(f"  펀딩비 최근 8회(24h): {history_str}")

    # ── 테이커 매수/매도 비율 + CVD (24h 히스토리) ──
    taker_hist = ctx.get("taker_history")
    if taker_hist:
        # 비율 히스토리는 최근 6h만 표시 (24h 전체는 장황)
        recent6 = taker_hist[-6:] if len(taker_hist) >= 6 else taker_hist
        ratio_str = " → ".join(f"{d['ratio']:.3f}" for d in recent6)
        last = taker_hist[-1]
        lines.append(
            f"  테이커 매수/매도 비율(최근 6h): {ratio_str}  "
            f"(최근: 매수 {last['buy']:,.1f} BTC / 매도 {last['sell']:,.1f} BTC)"
        )

    # ── CVD (Cumulative Volume Delta, 24h 기준) ──
    if ctx.get("cvd_current") is not None:
        cvd_24h  = ctx["cvd_current"]
        cvd_4h   = ctx.get("cvd_4h", 0) or 0
        cvd_dir  = "매수 우세" if cvd_24h > 0 else "매도 우세"
        cvd4_dir = "매수 강화" if cvd_4h > 0 else "매도 강화"
        # 시계열 6개 샘플(오래된→최근) — 방향 전환 시점 파악용
        series = ctx.get("cvd_series") or []
        # 6h 간격으로 샘플링 (24개 중 4개 + 최신)
        sample_idx = [0, 6, 12, 18, -1]
        samples = [series[i] for i in sample_idx if abs(i) < len(series)]
        sample_str = " → ".join(f"{v:+,.0f}" for v in samples)
        lines.append(
            f"  CVD 24h 누적 델타: {cvd_24h:+,.1f} BTC ({cvd_dir}) / "
            f"최근 4h 델타: {cvd_4h:+,.1f} BTC ({cvd4_dir})"
        )
        lines.append(
            f"  CVD 추이(24h 샘플): {sample_str}  "
            f"※ 테이커 거래량 기반 — 방향 편향 참고, 절대값보다 추세 변화에 주목"
        )

    # ── 오더북 불균형 ────────────────────────────
    if ctx.get("ob_imbalance") is not None:
        imb   = ctx["ob_imbalance"]
        bid_q = ctx.get("ob_bid_qty", 0)
        ask_q = ctx.get("ob_ask_qty", 0)
        if imb > 0.60:
            imb_label = "매수 우세(bid-heavy)"
        elif imb < 0.40:
            imb_label = "매도 우세(ask-heavy)"
        else:
            imb_label = "균형"
        lines.append(
            f"  오더북 불균형(top20): {imb:.3f} ({imb_label})  "
            f"매수잔량 {bid_q:,.1f} BTC / 매도잔량 {ask_q:,.1f} BTC"
            f"  ※ 스냅샷 — 대형 호가 취소 시 즉시 변동, 단기 방향 편향 참고"
        )

    # ── 오픈 인터레스트 ──────────────────────────
    if ctx.get("open_interest") is not None:
        lines.append(f"  오픈 인터레스트 Binance: {ctx['open_interest']:,.1f} BTC")
        if ctx.get("bybit_oi") is not None:
            lines.append(f"  오픈 인터레스트 Bybit:   {ctx['bybit_oi']:,.1f} BTC")
        if ctx.get("combined_oi") is not None:
            lines.append(
                f"  오픈 인터레스트 합산(Binance+Bybit): {ctx['combined_oi']:,.1f} BTC"
                f"  ※ 주요 2개 거래소 기준 — 시장 전체 레버리지 규모 근사"
            )
        if ctx.get("oi_change_24h_pct") is not None:
            lines.append(f"  Binance OI 24h 변화: {ctx['oi_change_24h_pct']:+.2f}%")

    # ── 톱 트레이더 롱/숏 비율 ──
    # ※ Binance API 필드명(longAccount)이 포지션·계좌 엔드포인트 양쪽에 혼용됨 — 해석 시 주의
    if ctx.get("top_pos_long") is not None:
        lines.append(
            f"  톱 트레이더 포지션: 롱 {ctx['top_pos_long']:.1f}% / 숏 {ctx['top_pos_short']:.1f}%"
            f"  ※ API 필드명 longAccount — 포지션 규모 비율 추정, 계좌 수 기반과 혼용 명명"
        )
        lines.append(
            f"  ⚠️ 톱 트레이더 데이터 주의: Binance 집계 지연·샘플링 불투명 — 단기 예측력 낮음, 타 지표와 동등 신뢰도 부여하지 마세요"
        )

    # ── 강제청산 이벤트 (최근 4시간) ────────────
    # ※ 개별 건 명세 제거 — 특정 시각 청산을 현재 시그널과 과도하게 연결하는 해석 방지
    events = ctx.get("liquidation_events")
    if events is not None:
        long_usd  = ctx.get("liq_long_usd",  0) or 0
        short_usd = ctx.get("liq_short_usd", 0) or 0
        if long_usd > 0 or short_usd > 0:
            lines.append(
                f"  강제청산 합계(4h): "
                f"롱청산 ${long_usd/1e6:.2f}M / 숏청산 ${short_usd/1e6:.2f}M"
            )
        else:
            lines.append("  강제청산: 최근 4시간 없음")

    # ── Fear & Greed Index ───────────────────────
    if ctx.get("fear_greed") is not None:
        fg_val   = ctx["fear_greed"]
        fg_label = ctx.get("fear_greed_label", "")
        hist     = ctx.get("fear_greed_history")
        hist_str = ""
        if hist and len(hist) >= 2:
            hist_str = " → ".join(f"{h['value']}({h['label']})" for h in hist)
            hist_str = f"  최근 3일: {hist_str}"
        lines.append(
            f"  Fear & Greed Index: {fg_val} ({fg_label})"
            f"  ※ 0=극도공포 25=공포 50=탐욕 75=극도탐욕 — 극단값에서 역발상 신호 참고"
        )
        if hist_str:
            lines.append(f" {hist_str}")

    # ── Deribit 옵션 ─────────────────────────────
    if ctx.get("dvol") is not None:
        lines.append(
            f"  DVOL (내재변동성 지수): {ctx['dvol']:.1f}"
            f"  ※ 실현변동성(RV)은 각 TF 지표 섹션 참조 — RV>DVOL: IV 할인(변동성 확대 예상) / RV<DVOL: IV 프리미엄(불확실성 과대반영)"
        )

    if ctx.get("skew_25d") is not None:
        skew      = ctx["skew_25d"]
        days_left = ctx.get("skew_days_left")
        days_str  = f"잔여 {days_left}일" if days_left is not None else ""
        # 14일 미만 만기: BS 역산 오차가 급격히 커져 부호 반전 가능
        short_warn = "  ⚠️ 단기 만기 — BS 오차 특히 큼, 부호조차 신뢰도 낮음" if (days_left is not None and days_left < 14) else ""
        lines.append(
            f"  25d 스큐 ({ctx['skew_expiry']}, {days_str}): {skew:+.1f} "
            f"(콜IV {ctx['skew_call_iv']:.1f} / 풋IV {ctx['skew_put_iv']:.1f})"
            f"  ※ BS 역산 근사값·근월물 단일 만기 스냅샷(기간 구조 미반영) — 부호·크기 수준만 참고{short_warn}"
        )

    return "\n".join(lines)
