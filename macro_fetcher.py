# =============================================
# 거시경제 지표 수집 모듈
# ─ FRED : DFII10 (10Y 실질금리), DGS2 (2Y 국채), DTWEXBGS (달러 인덱스)
# ─ DefiLlama : 전체 스테이블코인 시총, USDT 도미넌스
# =============================================
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
from config import FRED_API_KEY
from macro_history import attach_macro_history_summary

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# ── FRED 시리즈 정의 ──────────────────────────────────────────
_FRED_SERIES = {
    "DFII10":   {"label": "10Y 실질금리",  "unit": "%",  "fmt": "+.2f"},
    "DGS2":     {"label": "2Y 국채금리",   "unit": "%",  "fmt": "+.2f"},
    "DTWEXBGS": {"label": "달러 인덱스",   "unit": "",   "fmt": ".2f"},
}


def _fetch_fred(series_id: str, days: int = 60) -> Optional[pd.Series]:
    """
    FRED API로 시계열을 수집해 pd.Series(index=datetime)로 반환.
    API 키 없거나 실패 시 None.
    """
    if not FRED_API_KEY:
        return None
    try:
        start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        r = requests.get(
            FRED_BASE,
            params={
                "series_id":        series_id,
                "api_key":          FRED_API_KEY,
                "file_type":        "json",
                "sort_order":       "asc",
                "observation_start": start,
            },
            timeout=10,
        )
        r.raise_for_status()
        obs = r.json().get("observations", [])
        if not obs:
            return None

        s = pd.Series(
            {o["date"]: float(o["value"])
             for o in obs if o["value"] not in (".", "")},
            dtype=float,
        )
        s.index = pd.to_datetime(s.index)
        return s.sort_index()
    except Exception:
        return None


def _compute_stats(s: Optional[pd.Series], change_threshold: float = 0.05) -> dict:
    """
    pd.Series로부터 최신값 · 변화율 · 최근 시계열 요약을 계산.
    """
    empty = {
        "value": None,
        "latest_date": None,
        "change5d": None,
        "change20d": None,
        "zscore20": None,
        "regime": None,
        "trend20": None,
        "recent_flow": None,
        "recent_points": None,
    }
    if s is None or len(s) < 2:
        return empty

    latest = float(s.iloc[-1])
    latest_date = s.index[-1].strftime("%Y-%m-%d")

    # 5일 변화 (영업일 기준: -6번째 인덱스)
    if len(s) >= 6:
        change5d = latest - float(s.iloc[-6])
    elif len(s) >= 2:
        change5d = latest - float(s.iloc[0])
    else:
        change5d = None

    # 20일 변화
    if len(s) >= 21:
        change20d = latest - float(s.iloc[-21])
    elif len(s) >= 2:
        change20d = latest - float(s.iloc[0])
    else:
        change20d = None

    # 20일 z-score
    if len(s) >= 20:
        window = s.iloc[-20:].astype(float)
        mu, sigma = window.mean(), window.std(ddof=1)
        zscore20 = float((latest - mu) / sigma) if sigma > 0 else 0.0
    else:
        zscore20 = None

    def _classify_change(change: Optional[float], threshold: float) -> Optional[str]:
        if change is None:
            return None
        if change > threshold:
            return "상승"
        if change < -threshold:
            return "하락"
        return "중립"

    # 레짐 플래그 (5일 변화 기준)
    regime = _classify_change(change5d, change_threshold)

    # 20일 추세는 최근 20개 구간의 앞/뒤 평균 차이로 판단
    trend20 = None
    if len(s) >= 8:
        window20 = s.iloc[-20:].astype(float) if len(s) >= 20 else s.astype(float)
        pivot = max(len(window20) // 2, 1)
        first_half = window20.iloc[:pivot]
        second_half = window20.iloc[pivot:]
        span = float(window20.max() - window20.min())
        trend_delta = float(second_half.mean() - first_half.mean()) if len(second_half) else 0.0
        if span <= 1e-9:
            trend20 = "중립"
        elif abs(trend_delta) / span >= 0.2:
            trend20 = "상승" if trend_delta > 0 else "하락"
        else:
            trend20 = "중립"

    # 최근 흐름은 마지막 4개 관측치의 기울기 변화를 압축
    recent_flow = None
    if len(s) >= 4:
        last4 = s.iloc[-4:].astype(float)
        diffs = np.diff(last4.values)
        prev_move = float(diffs[-2])
        last_move = float(diffs[-1])

        if prev_move <= 0 < last_move:
            recent_flow = "반등 시도"
        elif prev_move >= 0 > last_move:
            recent_flow = "하락 전환 시도"
        elif last_move > 0 and prev_move > 0:
            recent_flow = "상승 가속" if abs(last_move) > abs(prev_move) * 1.2 else "상승 지속"
        elif last_move < 0 and prev_move < 0:
            recent_flow = "하락 가속" if abs(last_move) > abs(prev_move) * 1.2 else "하락 지속"
        else:
            recent_flow = "중립"

    recent_points = None
    if len(s) >= 3:
        recent_points = [
            {
                "date": idx.strftime("%m-%d"),
                "value": float(value),
            }
            for idx, value in s.iloc[-3:].items()
        ]

    return {
        "value":         latest,
        "latest_date":   latest_date,
        "change5d":      change5d,
        "change20d":     change20d,
        "zscore20":      zscore20,
        "regime":        regime,
        "trend20":       trend20,
        "recent_flow":   recent_flow,
        "recent_points": recent_points,
    }


# ── CoinGecko BTC 도미넌스 ───────────────────────────────────

def _fetch_btc_dominance() -> Optional[float]:
    """
    CoinGecko 무료 글로벌 엔드포인트에서 BTC 도미넌스(%) 수집.
    키 불필요.
    """
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/global",
            timeout=10,
        )
        r.raise_for_status()
        return float(r.json()["data"]["market_cap_percentage"]["btc"])
    except Exception:
        return None


# ── DefiLlama 스테이블코인 ────────────────────────────────────

def _fetch_stablecoins() -> dict:
    """
    DefiLlama에서 전체 스테이블코인 시총 + USDT 도미넌스 수집.
    키 불필요. 실패 시 None으로 채움.
    """
    out = {"stable_total_b": None, "usdt_b": None, "usdt_dom": None}
    try:
        r = requests.get(
            "https://stablecoins.llama.fi/stablecoins?includePrices=true",
            timeout=12,
        )
        r.raise_for_status()
        coins = r.json().get("peggedAssets", [])

        total = 0.0
        usdt  = 0.0
        for c in coins:
            val = (c.get("circulating") or {}).get("peggedUSD") or 0
            total += float(val)
            if c.get("symbol", "").upper() == "USDT":
                usdt += float(val)

        out["stable_total_b"] = total / 1e9
        out["usdt_b"]         = usdt  / 1e9
        out["usdt_dom"]       = (usdt / total * 100) if total > 0 else None
    except Exception:
        pass
    return out


# ── 공개 인터페이스 ───────────────────────────────────────────

def fetch_macro_context() -> dict:
    """
    거시 지표 전체를 한 번에 수집.
    반환 dict 구조:
      {
        "DFII10":      {label, unit, fmt, value, change5d, zscore20, regime},
        "DGS2":        {...},
        "DTWEXBGS":    {...},
        "STABLE_MCAP": {label, unit, fmt, value, change5d=None, zscore20=None, regime=None},
        "USDT_DOM":    {...},
        "BTC_DOM":     {...},
      }
    """
    result: dict = {}

    # FRED
    for sid, meta in _FRED_SERIES.items():
        s     = _fetch_fred(sid)
        stats = _compute_stats(s)
        result[sid] = {**meta, **stats}

    # DefiLlama 스테이블코인
    sc = _fetch_stablecoins()

    result["STABLE_MCAP"] = {
        "label":    "스테이블코인 시총",
        "unit":     "B",
        "fmt":      ".1f",
        "value":    sc["stable_total_b"],
        "change5d": None,
        "zscore20": None,
        "regime":   None,
    }
    result["USDT_DOM"] = {
        "label":    "USDT 도미넌스",
        "unit":     "%",
        "fmt":      ".1f",
        "value":    sc["usdt_dom"],
        "change5d": None,
        "zscore20": None,
        "regime":   None,
    }

    # CoinGecko BTC 도미넌스
    result["BTC_DOM"] = {
        "label":    "BTC 도미넌스",
        "unit":     "%",
        "fmt":      ".1f",
        "value":    _fetch_btc_dominance(),
        "change5d": None,
        "zscore20": None,
        "regime":   None,
    }

    attach_macro_history_summary(result)
    return result


def format_macro_context(macro: dict) -> str:
    """
    Claude 프롬프트 삽입용 거시 지표 텍스트.
    원시값 + 변화율 + 최근 시계열 요약을 포함하며, 해석은 Claude에 위임.
    """
    lines = [
        "[거시경제 지표]",
        "  ※ FRED 항목은 최신값뿐 아니라 24h/72h/7d/5일/20일 변화, 최근 흐름, 최근 3개 관측치를 함께 제공합니다.",
    ]
    for key, d in macro.items():
        if str(key).startswith("_"):
            continue
        v = d.get("value")
        if v is None:
            if not FRED_API_KEY and key in ("DFII10", "DGS2", "DTWEXBGS"):
                lines.append(f"  {d['label']} ({key}): FRED API 키 없음 — 미제공")
            else:
                lines.append(f"  {d['label']} ({key}): 데이터 없음")
            continue

        unit = d.get("unit", "")
        fmt  = d.get("fmt", ".2f")
        val_str = f"{v:{fmt}}{unit}"

        extras = []
        latest_date = d.get("latest_date")
        c5  = d.get("change5d")
        c20 = d.get("change20d")
        z20 = d.get("zscore20")
        reg = d.get("regime")
        trend20 = d.get("trend20")
        recent_flow = d.get("recent_flow")
        recent_points = d.get("recent_points")
        if latest_date is not None:
            extras.append(f"기준일 {latest_date}")

        if d.get("change24h") is not None:
            suffix = "B" if key == "STABLE_MCAP" else "%p" if unit == "%" else ""
            extras.append(f"24h 변화 {d['change24h']:+.2f}{suffix}")
        if d.get("change72h") is not None:
            suffix = "B" if key == "STABLE_MCAP" else "%p" if unit == "%" else ""
            extras.append(f"72h 변화 {d['change72h']:+.2f}{suffix}")
        if d.get("change7d") is not None:
            suffix = "B" if key == "STABLE_MCAP" else "%p" if unit == "%" else ""
            extras.append(f"7d 변화 {d['change7d']:+.2f}{suffix}")
        if c5  is not None:
            extras.append(f"5일 변화 {c5:+.3f}")
        if c20 is not None:
            extras.append(f"20일 변화 {c20:+.3f}")
        if z20 is not None:
            extras.append(f"z-score {z20:+.2f}")
        if reg is not None:
            extras.append(f"레짐 {reg}")
        if trend20 is not None:
            extras.append(f"20일 추세 {trend20}")
        if d.get("trend7d") is not None:
            extras.append(f"7d 추세 {d['trend7d']}")
        if recent_flow is not None:
            extras.append(f"최근 흐름 {recent_flow}")
        if recent_points:
            point_str = " → ".join(f"{p['date']}:{p['value']:{fmt}}{unit}" for p in recent_points)
            extras.append(f"최근 3개 {point_str}")

        extra_str = f"  ({', '.join(extras)})" if extras else ""
        lines.append(f"  {d['label']} ({key}): {val_str}{extra_str}")

    history_summary = macro.get("_history_summary") or {}
    sections = history_summary.get("sections") or []
    if sections:
        lines.append("  [서버 저장 기반 거시 추이]")
        for section in sections:
            label = section.get("label") or "거시 추이"
            lines.append(f"    [{label}] {section.get('meta', '')}".rstrip())
            for line in section.get("lines") or []:
                if line.startswith("관찰 구간:"):
                    continue
                lines.append(f"      {line}")

    if not FRED_API_KEY:
        lines.append("  ※ FRED API 키 미설정 — 금리·달러 데이터 비활성")

    lines.append(
        "  ※ 해석 참고: "
        "실질금리·달러 상승은 BTC 등 위험자산에 부정적 압력. "
        "스테이블코인 시총↑+USDT 도미넌스↓+BTC 도미넌스↑ = 알트→BTC 순환(위험선호 유지). "
        "스테이블코인 시총 정체+USDT 도미넌스↑+BTC 도미넌스↑ = 방어적 포지셔닝. "
        "스테이블코인 시총↑+USDT 도미넌스↓+BTC 도미넌스↓ = 알트 시즌 가능성. "
        "단기 트레이딩 시그널보다 중기 레짐 필터로 활용하고, "
        "기술적 지표와 파생상품 데이터로 최종 시그널을 판단하세요."
    )

    return "\n".join(lines)
