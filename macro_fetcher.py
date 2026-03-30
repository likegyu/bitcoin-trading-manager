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
    pd.Series로부터 최신값 · 5일 변화 · 20일 z-score · 레짐 플래그 계산.
    """
    empty = {"value": None, "change5d": None, "zscore20": None, "regime": None}
    if s is None or len(s) < 2:
        return empty

    latest = float(s.iloc[-1])

    # 5일 변화 (영업일 기준: -6번째 인덱스)
    if len(s) >= 6:
        change5d = latest - float(s.iloc[-6])
    elif len(s) >= 2:
        change5d = latest - float(s.iloc[0])
    else:
        change5d = None

    # 20일 z-score
    if len(s) >= 20:
        window = s.iloc[-20:].astype(float)
        mu, sigma = window.mean(), window.std(ddof=1)
        zscore20 = float((latest - mu) / sigma) if sigma > 0 else 0.0
    else:
        zscore20 = None

    # 레짐 플래그 (5일 변화 기준)
    if change5d is not None:
        if change5d > change_threshold:
            regime = "상승"
        elif change5d < -change_threshold:
            regime = "하락"
        else:
            regime = "중립"
    else:
        regime = None

    return {
        "value":    latest,
        "change5d": change5d,
        "zscore20": zscore20,
        "regime":   regime,
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

    return result


def format_macro_context(macro: dict) -> str:
    """
    Claude 프롬프트 삽입용 거시 지표 텍스트.
    원시값 + 5일 변화 + z-score를 포함하며, 해석은 Claude에 위임.
    """
    lines = ["[거시경제 지표]"]
    for key, d in macro.items():
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
        c5  = d.get("change5d")
        z20 = d.get("zscore20")
        reg = d.get("regime")
        if c5  is not None: extras.append(f"5일 변화 {c5:+.3f}")
        if z20 is not None: extras.append(f"z-score {z20:+.2f}")
        if reg is not None: extras.append(f"레짐 {reg}")

        extra_str = f"  ({', '.join(extras)})" if extras else ""
        lines.append(f"  {d['label']} ({key}): {val_str}{extra_str}")

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
