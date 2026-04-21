# =============================================
# 거시경제 지표 수집 모듈
# ─ FRED : DFEDTARU (기준금리 상단), DFII10 (10Y 실질금리),
#           DGS2 (2Y 국채시장금리), DTWEXBGS (달러 인덱스)
# ─ DefiLlama : 전체 스테이블코인 시총, USDT 도미넌스
# =============================================
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
from config import FRED_API_KEY
from macro_history import attach_macro_history_summary
from http_client import _session as _http  # 프록시 환경변수 무시 세션

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# ── FRED 시리즈 정의 ──────────────────────────────────────────
# DFEDTARU : Fed Funds Target Rate Upper Bound (실제 FOMC 기준금리 상단, 일별)
#   → 25bp 단위로 계단식 변화. 시장금리(DGS2)와 달리 FOMC 결정일에만 변동.
# DGS2     : 2Y Treasury Yield (시장 기대 반영 — 기준금리 예상치에 선행)
# DFII10   : 10Y TIPS Yield (실질금리)
# DTWEXBGS : Broad Dollar Index
_FRED_SERIES = {
    "DFEDTARU": {"label": "기준금리(상단)",  "unit": "%",  "fmt": ".2f"},
    "DFII10":   {"label": "10Y 실질금리",   "unit": "%",  "fmt": "+.2f"},
    "DGS2":     {"label": "2Y 국채금리",    "unit": "%",  "fmt": "+.2f"},
    "DTWEXBGS": {"label": "달러 인덱스",    "unit": "",   "fmt": ".2f"},
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
        r = _http.get(
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


# ── 전통 시장 지표 (yfinance) ─────────────────────────────────

def _fetch_traditional_markets() -> dict:
    """
    yfinance로 전통 시장 지표 수집:
      - ^GSPC  : S&P 500
      - ^IXIC  : 나스닥 컴포지트
      - ^VIX   : CBOE 변동성 지수
      - GC=F   : 금 선물 (USD/oz)

    반환 구조:
      {
        "SPX":  {"price": float, "chg_pct": float, "prev_close": float},
        "NDX":  {...},
        "VIX":  {"price": float, "chg_pct": float},
        "GOLD": {"price": float, "chg_pct": float},
        "error": str | None,
      }
    """
    result: dict = {
        "SPX": None, "NDX": None, "VIX": None, "GOLD": None, "error": None,
    }
    try:
        import yfinance as yf  # 런타임 임포트 — 설치 안 됐을 때 전체 모듈 블로킹 방지

        tickers_map = {
            "^GSPC": "SPX",
            "^IXIC": "NDX",
            "^VIX":  "VIX",
            "GC=F":  "GOLD",
        }

        # period="5d"로 충분 — 하루 종가 2개(전일·최신)만 필요
        data = yf.download(
            list(tickers_map.keys()),
            period="5d",
            interval="1d",
            progress=False,
            auto_adjust=True,
            threads=False,
        )

        # MultiIndex DataFrame — data.get() 은 MultiIndex에서 None 반환하므로 직접 접근
        try:
            close_df = data["Close"]
        except KeyError:
            result["error"] = "yfinance Close 컬럼 없음"
            return result
        if close_df is None or (hasattr(close_df, "empty") and close_df.empty):
            result["error"] = "yfinance 데이터 없음"
            return result

        for ticker, key in tickers_map.items():
            try:
                series = close_df[ticker].dropna()
                if len(series) < 2:
                    continue
                price      = float(series.iloc[-1])
                prev_close = float(series.iloc[-2])
                chg_pct    = (price - prev_close) / prev_close * 100
                result[key] = {
                    "price":      round(price, 2),
                    "prev_close": round(prev_close, 2),
                    "chg_pct":    round(chg_pct, 2),
                }
            except Exception:
                continue

    except ImportError:
        result["error"] = "yfinance 미설치 — pip install yfinance"
    except Exception as e:
        result["error"] = str(e)[:120]

    return result


# ── CoinGecko BTC 도미넌스 ───────────────────────────────────

def _fetch_btc_dominance() -> Optional[float]:
    """
    CoinGecko 무료 글로벌 엔드포인트에서 BTC 도미넌스(%) 수집.
    키 불필요.
    """
    try:
        r = _http.get(
            "https://api.coingecko.com/api/v3/global",
            timeout=10,
        )
        r.raise_for_status()
        return float(r.json()["data"]["market_cap_percentage"]["btc"])
    except Exception:
        return None


# ── ETH/BTC 비율 ─────────────────────────────────────────────

def _fetch_eth_btc_ratio() -> dict:
    """
    CoinGecko에서 ETH·BTC USD 가격을 수집해 ETH/BTC 비율과 24h 변화율을 계산.
    키 불필요.

    반환:
      {
        "eth_usd": float,
        "btc_usd": float,
        "eth_btc": float,          # ETH/BTC 비율
        "eth_chg_24h": float,      # ETH USD 24h 변화율(%)
        "btc_chg_24h": float,      # BTC USD 24h 변화율(%)
        "ratio_chg_24h": float,    # ETH/BTC 비율 24h 변화율(%)
        "error": str | None,
      }
    """
    result: dict = {k: None for k in ("eth_usd","btc_usd","eth_btc","eth_chg_24h","btc_chg_24h","ratio_chg_24h","error")}
    try:
        r = _http.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={
                "ids":            "bitcoin,ethereum",
                "vs_currencies":  "usd",
                "include_24hr_change": "true",
            },
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()

        btc_usd      = float(data["bitcoin"]["usd"])
        eth_usd      = float(data["ethereum"]["usd"])
        btc_chg      = float(data["bitcoin"].get("usd_24h_change", 0) or 0)
        eth_chg      = float(data["ethereum"].get("usd_24h_change", 0) or 0)
        eth_btc      = eth_usd / btc_usd if btc_usd > 0 else None

        # ETH/BTC 비율의 24h 변화율 = (1 + eth_chg/100) / (1 + btc_chg/100) - 1
        if btc_chg is not None and eth_chg is not None:
            ratio_chg = ((1 + eth_chg / 100) / (1 + btc_chg / 100) - 1) * 100
        else:
            ratio_chg = None

        result.update({
            "eth_usd":       round(eth_usd, 2),
            "btc_usd":       round(btc_usd, 2),
            "eth_btc":       round(eth_btc, 6) if eth_btc else None,
            "eth_chg_24h":   round(eth_chg, 2),
            "btc_chg_24h":   round(btc_chg, 2),
            "ratio_chg_24h": round(ratio_chg, 2) if ratio_chg is not None else None,
            "error":         None,
        })
    except Exception as e:
        result["error"] = str(e)[:80]
    return result


# ── DefiLlama 스테이블코인 ────────────────────────────────────

def _fetch_stablecoins() -> dict:
    """
    DefiLlama에서 전체 스테이블코인 시총 + USDT 도미넌스 수집.
    키 불필요. 실패 시 None으로 채움.
    """
    out = {"stable_total_b": None, "usdt_b": None, "usdt_dom": None}
    try:
        r = _http.get(
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

    # FRED — DFEDTARU 는 step-function(계단식) 시계열이므로 days=120으로 길게 조회
    for sid, meta in _FRED_SERIES.items():
        days = 120 if sid == "DFEDTARU" else 60
        s     = _fetch_fred(sid, days=days)
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

    # ETH/BTC 비율
    result["_eth_btc"] = _fetch_eth_btc_ratio()

    # 전통 시장 (yfinance) — 별도 키로 저장 (format_macro_context에서 별도 섹션 출력)
    result["_trad_markets"] = _fetch_traditional_markets()

    attach_macro_history_summary(result)
    return result


def _staleness_days(latest_date_str: Optional[str]) -> Optional[int]:
    """FRED 기준일로부터 오늘까지 경과 일수. None이면 날짜 없음."""
    if not latest_date_str:
        return None
    try:
        from datetime import date as _date
        ld = _date.fromisoformat(latest_date_str)
        return (_date.today() - ld).days
    except Exception:
        return None


# FRED 시리즈별 허용 지연 일수 (이 값 초과 시 경고)
# DTWEXBGS는 H.10 릴리즈 기준 최대 7-8영업일 지연 정상
_STALE_THRESHOLD_DAYS: dict[str, int] = {
    "DFEDTARU": 4,   # FOMC 이후 익일 반영
    "DFII10":   4,   # 영업일 기준 T+1 (주말 포함 시 3일)
    "DGS2":     4,   # 영업일 기준 T+1
    "DTWEXBGS": 7,   # H.10 릴리즈 5-7일 지연 정상; 7일 초과 시 경고
}


def format_macro_context(macro: dict) -> str:
    """
    Claude 프롬프트 삽입용 거시 지표 텍스트.
    원시값 + 변화율 + 최근 시계열 요약을 포함하며, 해석은 Claude에 위임.
    """
    lines = [
        "[거시경제 지표]",
        "  ※ FRED 항목은 최신값뿐 아니라 24h/72h/7d/5일/20일 변화, 최근 흐름, 최근 3개 관측치를 함께 제공합니다.",
        "  ※ DTWEXBGS(달러 인덱스)는 H.10 릴리즈 기준 최대 7-8영업일 지연이 정상입니다.",
    ]
    for key, d in macro.items():
        if str(key).startswith("_"):
            continue
        v = d.get("value")
        if v is None:
            if not FRED_API_KEY and key in ("DFEDTARU", "DFII10", "DGS2", "DTWEXBGS"):
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

        # 데이터 지연 경고
        stale_days = _staleness_days(latest_date)
        stale_threshold = _STALE_THRESHOLD_DAYS.get(key, 4)
        stale_flag = ""
        if stale_days is not None and stale_days > stale_threshold:
            stale_flag = f" ⚠️{stale_days}일지연"

        if latest_date is not None:
            extras.append(f"기준일 {latest_date}{stale_flag}")

        if d.get("change24h") is not None:
            suffix = "B" if key == "STABLE_MCAP" else "%p" if unit == "%" else ""
            extras.append(f"24h 변화 {d['change24h']:+.2f}{suffix}")
        if d.get("change72h") is not None:
            suffix = "B" if key == "STABLE_MCAP" else "%p" if unit == "%" else ""
            extras.append(f"72h 변화 {d['change72h']:+.2f}{suffix}")
        if d.get("change7d") is not None:
            suffix = "B" if key == "STABLE_MCAP" else "%p" if unit == "%" else ""
            ch7 = d["change7d"]
            # history 기반 7d 변화는 서버 구동 중 누적된 스냅샷으로 계산됨.
            # 스냅샷이 적으면 0처럼 보일 수 있어 FRED 5일 변화를 함께 참고해야 함.
            samples = d.get("change7d_samples")
            if samples is not None and samples < 3:
                extras.append(f"7d 변화 {ch7:+.2f}{suffix}(스냅샷{samples}개-FRED5일변화참고)")
            else:
                extras.append(f"7d 변화 {ch7:+.2f}{suffix}")
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

    # ── ETH/BTC 비율 섹션 ────────────────────────────────────────
    eb = macro.get("_eth_btc") or {}
    if eb.get("eth_btc") is not None:
        ratio     = eb["eth_btc"]
        ratio_chg = eb.get("ratio_chg_24h")
        eth_chg   = eb.get("eth_chg_24h")
        btc_chg   = eb.get("btc_chg_24h")
        chg_str   = f"  24h 비율변화 {ratio_chg:+.2f}%" if ratio_chg is not None else ""
        price_str = ""
        if eb.get("eth_usd") and eb.get("btc_usd"):
            price_str = (
                f"  (ETH ${eb['eth_usd']:,.0f} {eth_chg:+.1f}% / "
                f"BTC ${eb['btc_usd']:,.0f} {btc_chg:+.1f}%)"
            )
        lines.append(
            f"  ETH/BTC 비율: {ratio:.6f}{chg_str}{price_str}"
        )
        lines.append(
            "  ※ 해석: 비율↑ = ETH 상대 강세(알트 시즌 가능성) / 비율↓ = BTC 단독 강세 또는 리스크오프. "
            "BTC 도미넌스와 함께 순환 방향 확인."
        )
    elif eb.get("error"):
        lines.append(f"  ETH/BTC 비율: 수집 실패 — {eb['error']}")

    # ── 전통 시장 섹션 ───────────────────────────────────────────
    trad = macro.get("_trad_markets") or {}
    trad_error = trad.get("error")
    trad_labels = {
        "SPX":  ("S&P 500",  ""),
        "NDX":  ("나스닥",   ""),
        "VIX":  ("VIX",      "  ※ 20 미만=저변동 / 20~30=경계 / 30+=공포"),
        "GOLD": ("금(USD/oz)", ""),
    }
    trad_lines = []
    for key, (label, note) in trad_labels.items():
        d = trad.get(key)
        if d:
            arrow = "▲" if d["chg_pct"] >= 0 else "▼"
            trad_lines.append(
                f"  {label}: ${d['price']:,.2f}  {arrow}{d['chg_pct']:+.2f}%{note}"
            )
    if trad_lines:
        lines.append("[전통 시장]  ※ 전일 종가 기준 — 미국 장 마감 이후 갱신")
        lines.extend(trad_lines)
        lines.append(
            "  ※ 해석 참고: SPX/NDX↓+BTC↓ = 리스크오프 동조. "
            "VIX↑(30+) = 공포 구간. "
            "Gold↑+BTC↑ = 인플레이션 헤지 동조 또는 달러 약세 테마. "
            "SPX↑+BTC↓ = BTC 개별 약세(알트·레버리지 청산 등)."
        )
    elif trad_error:
        lines.append(f"[전통 시장]: 수집 실패 — {trad_error}")

    if not FRED_API_KEY:
        lines.append("  ※ FRED API 키 미설정 — 금리·달러 데이터 비활성")

    lines.append(
        "  ※ 해석 참고: "
        "기준금리(DFEDTARU)는 FOMC 결정일에만 25bp 단위로 변동 — 2Y 국채금리(DGS2)와 달리 일별 등락 없음. "
        "DGS2가 DFEDTARU보다 낮으면 시장이 금리 인하를 선반영. "
        "실질금리·달러 상승은 BTC 등 위험자산에 부정적 압력. "
        "스테이블코인 시총↑+USDT 도미넌스↓+BTC 도미넌스↑ = 알트→BTC 순환(위험선호 유지). "
        "스테이블코인 시총 정체+USDT 도미넌스↑+BTC 도미넌스↑ = 방어적 포지셔닝. "
        "스테이블코인 시총↑+USDT 도미넌스↓+BTC 도미넌스↓ = 알트 시즌 가능성. "
        "단기 트레이딩 시그널보다 중기 레짐 필터로 활용하고, "
        "기술적 지표와 파생상품 데이터로 최종 시그널을 판단하세요."
    )

    return "\n".join(lines)
