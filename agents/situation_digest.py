# =============================================
# Situation Digest — BM25 매칭을 위한 정규화된 범주형 태그
# =============================================
# 문제:
#   FinancialSituationMemory 에 원본 context_blob (10KB+) 을 그대로 넣으면
#   "$65,234" 같은 절대 수치가 토큰이 되어 비슷한 구조끼리 매칭되지 않는다.
#   (BM25 는 어휘 일치 기반 — 수치가 달라지면 유사도 0)
#
# 해결:
#   지표/파생/거시/계좌 상태를 카테고리로 bin 해서 파이프 구분 문자열로 배출.
#   예) "view_1h:과매수 | macd_1h:상방약화 | ma_1h:정배열 | funding:양수_과열 | oi_24h:증가 ..."
#   같은 '구조적 상황' 에서는 동일/유사 태그가 반복되므로 BM25 매칭이 의미를 갖는다.
#
# 철학:
#   - 정확한 수치는 의미 없다. '범주' 와 '방향' 만 남긴다.
#   - 필드가 누락되면 그 태그는 조용히 스킵. 잘못된 기본값은 쓰지 않는다.
# =============================================
from __future__ import annotations

from typing import Any, Optional


# ── RSI bin ────────────────────────────────────
def _rsi_bin(rsi: Optional[float]) -> Optional[str]:
    if rsi is None:
        return None
    if rsi < 20:
        return "과매도_극단"
    if rsi < 30:
        return "과매도"
    if rsi < 45:
        return "약세중립"
    if rsi <= 55:
        return "중립"
    if rsi <= 70:
        return "강세중립"
    if rsi <= 80:
        return "과매수"
    return "과매수_극단"


# ── MACD Hist 방향성 bin ────────────────────────
def _macd_bin(hist: Optional[float], prev_hist: Optional[float]) -> Optional[str]:
    if hist is None:
        return None
    # 부호 변환은 가장 중요한 신호
    if prev_hist is not None:
        if prev_hist <= 0 and hist > 0:
            return "상방전환"
        if prev_hist >= 0 and hist < 0:
            return "하방전환"
    if hist > 0:
        # 약화/강화
        if prev_hist is not None and hist < prev_hist:
            return "상방약화"
        return "상방강화"
    if hist < 0:
        if prev_hist is not None and hist > prev_hist:
            return "하방약화"
        return "하방강화"
    return "중립"


# ── MA 정렬 bin (ema_9 / sma_50 / sma_200) ─────
def _ma_bin(ema9: Optional[float], sma50: Optional[float], sma200: Optional[float]) -> Optional[str]:
    if None in (ema9, sma50, sma200):
        return None
    if ema9 > sma50 > sma200:
        return "정배열"
    if ema9 < sma50 < sma200:
        return "역배열"
    if ema9 > sma200 and sma50 > sma200:
        return "상위혼조"
    if ema9 < sma200 and sma50 < sma200:
        return "하위혼조"
    return "혼조"


# ── 펀딩 bin ────────────────────────────────────
def _funding_bin(rate_pct: Optional[float]) -> Optional[str]:
    # market_context 에서 percent 단위로 들어옴 (예: 0.01 = 0.01%)
    if rate_pct is None:
        return None
    if rate_pct >= 0.05:
        return "양수_과열"
    if rate_pct >= 0.01:
        return "양수"
    if rate_pct > -0.01:
        return "정상"
    if rate_pct > -0.05:
        return "음수"
    return "음수_과열"


# ── OI 24h 변화 bin ─────────────────────────────
def _oi_bin(pct: Optional[float]) -> Optional[str]:
    if pct is None:
        return None
    if pct >= 10:
        return "급증"
    if pct >= 3:
        return "증가"
    if pct > -3:
        return "보합"
    if pct > -10:
        return "감소"
    return "급감"


# ── 25d 스큐 bin ────────────────────────────────
def _skew_bin(skew: Optional[float]) -> Optional[str]:
    # put_iv - call_iv. 양수 = 풋 프리미엄 (약세 편향), 음수 = 콜 프리미엄 (강세 편향)
    if skew is None:
        return None
    if skew >= 3:
        return "풋우세_강함"
    if skew >= 1:
        return "풋우세"
    if skew > -1:
        return "중립"
    if skew > -3:
        return "콜우세"
    return "콜우세_강함"


# ── 계좌 모드 bin (손익 부호 + 운영 맥락) ────────
def _account_bin(account_ctx: Optional[dict]) -> Optional[str]:
    if not account_ctx:
        return None
    upnl = account_ctx.get("unrealized_pnl")
    wallet = account_ctx.get("wallet_balance")
    if wallet is None or wallet == 0 or upnl is None:
        return "평탄"
    ratio = (upnl / wallet) * 100.0
    if ratio >= 2:
        return "수익보호"
    if ratio >= 0.3:
        return "수익중"
    if ratio > -0.3:
        return "평탄"
    if ratio > -2:
        return "손실중"
    return "손실복구_모드"


def _safe_get(row: Any, key: str) -> Optional[float]:
    try:
        v = row[key]
    except (KeyError, IndexError, TypeError):
        return None
    # pandas NaN 검사
    try:
        import math
        if isinstance(v, float) and math.isnan(v):
            return None
    except Exception:
        pass
    return float(v) if v is not None else None


def _tf_tags(tf: str, df) -> list[str]:
    """단일 타임프레임의 지표를 태그 리스트로 변환."""
    if df is None or len(df) == 0:
        return []
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else None

    tags: list[str] = []

    rsi_b = _rsi_bin(_safe_get(last, "rsi"))
    if rsi_b:
        tags.append(f"rsi_{tf}:{rsi_b}")

    macd_b = _macd_bin(
        _safe_get(last, "macd_hist"),
        _safe_get(prev, "macd_hist") if prev is not None else None,
    )
    if macd_b:
        tags.append(f"macd_{tf}:{macd_b}")

    ma_b = _ma_bin(
        _safe_get(last, "ema_9"),
        _safe_get(last, "sma_50"),
        _safe_get(last, "sma_200"),
    )
    if ma_b:
        tags.append(f"ma_{tf}:{ma_b}")

    # SMA200 대비 위치 (추세 기둥)
    close = _safe_get(last, "close")
    sma200 = _safe_get(last, "sma_200")
    if close is not None and sma200 is not None:
        diff_pct = (close - sma200) / sma200 * 100 if sma200 else 0
        if abs(diff_pct) < 0.5:
            pos = "터치"
        elif diff_pct > 0:
            pos = "상위_강" if diff_pct > 3 else "상위"
        else:
            pos = "하위_강" if diff_pct < -3 else "하위"
        tags.append(f"trend_{tf}:{pos}")

    return tags


def summarize_situation_tags(
    multi_tf_data: dict,
    macro_snapshot: Optional[dict] = None,
    market_ctx: Optional[dict] = None,
    account_ctx: Optional[dict] = None,
) -> str:
    """
    멀티 TF + 거시 + 파생 + 계좌 상태를 정규화 태그 문자열로 변환.

    반환 형식: "rsi_1h:과매수 | macd_1h:상방약화 | funding:양수_과열 | ..."
    BM25 매칭을 위해 사용되므로 수치는 모두 카테고리로 bin.
    """
    tags: list[str] = []

    # 1) 타임프레임 — 4h/1h/15m 중심 (5m/1d 는 노이즈/과적합 방지)
    for tf in ("4h", "1h", "15m"):
        if tf in (multi_tf_data or {}):
            tags.extend(_tf_tags(tf, multi_tf_data[tf]))

    # 2) 파생상품
    if market_ctx:
        fb = _funding_bin(market_ctx.get("funding_rate"))
        if fb:
            tags.append(f"funding:{fb}")
        ob = _oi_bin(market_ctx.get("oi_change_24h_pct"))
        if ob:
            tags.append(f"oi_24h:{ob}")
        sb = _skew_bin(market_ctx.get("skew_25d"))
        if sb:
            tags.append(f"skew:{sb}")

    # 3) 거시 — regime 필드만 사용 (수치는 노이즈)
    if macro_snapshot:
        for key in ("DFII10", "DGS2", "DTWEXBGS", "STABLE_MCAP", "USDT_DOM", "BTC_DOM"):
            entry = macro_snapshot.get(key)
            if not isinstance(entry, dict):
                continue
            regime = entry.get("regime")
            if regime:
                # regime 값은 한글 라벨 (상승/하락/보합/등) — 그대로 사용
                tags.append(f"macro_{key.lower()}:{regime}")

    # 4) 계좌 모드
    ab = _account_bin(account_ctx)
    if ab:
        tags.append(f"account:{ab}")

    return " | ".join(tags)
