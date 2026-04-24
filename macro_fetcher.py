# =============================================
# 거시경제 지표 수집 모듈
# ─ yfinance  : ^TNX (10Y 금리), ^FVX (5Y 금리), DX-Y.NYB (달러 인덱스),
#                ^GSPC, ^IXIC, ^VIX, GC=F, HYG, LQD, IBIT
# ─ DefiLlama : 전체 스테이블코인 시총, USDT 도미넌스
# ─ CoinGecko : BTC 도미넌스, ETH/BTC 비율
#
# 과거 FRED 의존성은 제거됨 — FRED 는 영업일 기준 T+1 지연이라
# 서버 스냅샷(30분 주기)과 동기화할 때 24h/72h 변화가 0 으로 찍히는
# 문제가 있어, 동일 지표군을 yfinance 실시간 데이터로 대체함.
# =============================================
import pandas as pd
import numpy as np
from typing import Optional
from macro_history import attach_macro_history_summary
from http_client import _session as _http  # 프록시 환경변수 무시 세션


# ── 시계열 통계 공통 헬퍼 ─────────────────────────────────────

def _compute_stats(s: Optional[pd.Series], change_threshold: float = 0.05) -> dict:
    """
    pd.Series로부터 최신값 · 변화율 · 최근 시계열 요약을 계산.
    입력 시리즈가 비어있거나 길이가 짧으면 None 필드로 채운 구조를 반환.
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
    try:
        latest_date = s.index[-1].strftime("%Y-%m-%d")
    except Exception:
        latest_date = None

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
        try:
            recent_points = [
                {
                    "date": idx.strftime("%m-%d"),
                    "value": float(value),
                }
                for idx, value in s.iloc[-3:].items()
            ]
        except Exception:
            recent_points = None

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


# ── yfinance helpers ─────────────────────────────────────────

def _extract_close(data) -> Optional[pd.Series]:
    """yfinance.download 결과에서 Close 시리즈를 안전하게 추출."""
    try:
        if data is None or (hasattr(data, "empty") and data.empty):
            return None
        close = data["Close"].dropna()
        if isinstance(close, pd.DataFrame):
            if close.shape[1] == 0:
                return None
            close = close.iloc[:, 0].dropna()
        return close if len(close) > 0 else None
    except Exception:
        return None


def _yf_close_series(ticker: str, period: str = "60d") -> Optional[pd.Series]:
    """yfinance로 단일 티커 종가 시계열 조회."""
    try:
        import yfinance as yf
        data = yf.download(
            ticker, period=period, interval="1d",
            progress=False, auto_adjust=True, threads=False,
        )
        return _extract_close(data)
    except Exception:
        return None


# ── 시장 금리·달러 인덱스 (yfinance 실시간) ───────────────────

def _normalize_yield_series(series: pd.Series) -> pd.Series:
    """
    Yahoo yield tickers have appeared in both percent (4.3) and percent*10
    (43.0) formats across environments. Normalize to percent points.
    """
    s = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if s.empty:
        return s
    recent = s.tail(min(len(s), 10)).median()
    if pd.notna(recent) and recent > 20:
        return s * 0.1
    return s


def _fetch_market_rates() -> dict:
    """
    실시간 시장금리·달러 지수 수집.

    티커 규칙:
      ^TNX  — 10Y Treasury Yield. 환경에 따라 4.25 또는 42.5 형식으로 와서 보정.
      ^FVX  — 5Y  Treasury Yield. 동일 규칙.
      DX-Y.NYB — ICE 달러 인덱스 (실제 값).

    각 키별 pd.Series(종가)를 반환. 실패 시 None.
    """
    out: dict[str, Optional[pd.Series]] = {"TNX_10Y": None, "FVX_5Y": None, "DXY": None}

    for ticker, key, is_yield in (
        ("^TNX",     "TNX_10Y", True),
        ("^FVX",     "FVX_5Y",  True),
        ("DX-Y.NYB", "DXY",     False),
    ):
        s = _yf_close_series(ticker, period="90d")
        if s is not None and len(s) > 0:
            out[key] = _normalize_yield_series(s) if is_yield else s.astype(float)
    return out


# ── 신용 스프레드 프록시: HYG/LQD ─────────────────────────────

def _fetch_credit_spread() -> Optional[pd.Series]:
    """
    HYG (하이일드 채권 ETF) / LQD (투자등급 채권 ETF) 비율.

    비율↓ = 하이일드 스프레드 확대 = 리스크오프 선행 신호.
    BTC 와 상관이 높고(특히 하락장 방향), 전통 시장 스트레스 → 코인 디레버리지를
    먼저 신호한다.
    """
    try:
        import yfinance as yf
        data = yf.download(
            ["HYG", "LQD"], period="90d", interval="1d",
            progress=False, auto_adjust=True, threads=False,
        )
        close_df = data["Close"].dropna() if data is not None else None
        if close_df is None or close_df.empty:
            return None
        # MultiIndex 케이스 방어
        if isinstance(close_df, pd.Series):
            return None
        if "HYG" not in close_df.columns or "LQD" not in close_df.columns:
            return None
        ratio = (close_df["HYG"] / close_df["LQD"]).dropna()
        return ratio if len(ratio) > 0 else None
    except Exception:
        return None


# ── 현물 BTC ETF: IBIT ───────────────────────────────────────

def _fetch_btc_etf() -> dict:
    """
    IBIT (BlackRock iShares Bitcoin Trust) 시계열.
    기관 수급 프록시 — 거래량/20MA 비율로 참여도 증감을 추정.

    반환:
      {
        "close_series": pd.Series | None,
        "vol_latest":  float | None,
        "vol_20ma":    float | None,
        "vol_ratio":   float | None,  # latest / 20MA
      }
    """
    out: dict = {
        "close_series": None, "vol_latest": None, "vol_20ma": None, "vol_ratio": None,
    }
    try:
        import yfinance as yf
        data = yf.download(
            "IBIT", period="60d", interval="1d",
            progress=False, auto_adjust=True, threads=False,
        )
        if data is None or (hasattr(data, "empty") and data.empty):
            return out

        out["close_series"] = _extract_close(data)

        try:
            vol = data["Volume"].dropna()
            if isinstance(vol, pd.DataFrame):
                vol = vol.iloc[:, 0].dropna()
            if len(vol) >= 20:
                vol_20ma   = float(vol.iloc[-20:].mean())
                vol_latest = float(vol.iloc[-1])
                out["vol_latest"] = vol_latest
                out["vol_20ma"]   = vol_20ma
                out["vol_ratio"]  = (vol_latest / vol_20ma) if vol_20ma > 0 else None
        except Exception:
            pass
    except Exception:
        pass
    return out


# ── 전통 시장 지표 (yfinance) ─────────────────────────────────

def _fetch_traditional_markets() -> dict:
    """
    yfinance로 전통 시장 지표 수집:
      - ^GSPC  : S&P 500
      - ^IXIC  : 나스닥 컴포지트
      - ^VIX   : CBOE 변동성 지수
      - GC=F   : 금 선물 (USD/oz)
    """
    result: dict = {
        "SPX": None, "NDX": None, "VIX": None, "GOLD": None, "error": None,
    }
    try:
        import yfinance as yf

        tickers_map = {
            "^GSPC": "SPX",
            "^IXIC": "NDX",
            "^VIX":  "VIX",
            "GC=F":  "GOLD",
        }

        data = yf.download(
            list(tickers_map.keys()),
            period="5d",
            interval="1d",
            progress=False,
            auto_adjust=True,
            threads=False,
        )

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
    """CoinGecko 무료 글로벌 엔드포인트에서 BTC 도미넌스(%) 수집."""
    try:
        r = _http.get("https://api.coingecko.com/api/v3/global", timeout=10)
        r.raise_for_status()
        return float(r.json()["data"]["market_cap_percentage"]["btc"])
    except Exception:
        return None


# ── ETH/BTC 비율 ─────────────────────────────────────────────

def _fetch_eth_btc_ratio() -> dict:
    """CoinGecko simple/price — ETH/BTC 비율 및 24h 변화."""
    result: dict = {k: None for k in (
        "eth_usd", "btc_usd", "eth_btc", "eth_chg_24h",
        "btc_chg_24h", "ratio_chg_24h", "error",
    )}
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

        btc_usd = float(data["bitcoin"]["usd"])
        eth_usd = float(data["ethereum"]["usd"])
        btc_chg = float(data["bitcoin"].get("usd_24h_change", 0) or 0)
        eth_chg = float(data["ethereum"].get("usd_24h_change", 0) or 0)
        eth_btc = eth_usd / btc_usd if btc_usd > 0 else None

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
    """DefiLlama 전체 스테이블코인 시총 + USDT 도미넌스."""
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

# 지표 메타 (키 순서 = 프롬프트 출력 순서)
_METRIC_META = {
    "TNX_10Y":     {"label": "10Y 국채금리",    "unit": "%",  "fmt": ".2f", "threshold": 0.03},
    "FVX_5Y":      {"label": "5Y 국채금리",     "unit": "%",  "fmt": ".2f", "threshold": 0.03},
    "DXY":         {"label": "달러 인덱스",     "unit": "",   "fmt": ".2f", "threshold": 0.20},
    "STABLE_MCAP": {"label": "스테이블코인 시총", "unit": "B",  "fmt": ".1f", "threshold": 0.20},
    "USDT_DOM":    {"label": "USDT 도미넌스",   "unit": "%",  "fmt": ".1f", "threshold": 0.10},
    "BTC_DOM":     {"label": "BTC 도미넌스",    "unit": "%",  "fmt": ".1f", "threshold": 0.10},
    "HYG_LQD":     {"label": "HYG/LQD 비율",    "unit": "",   "fmt": ".4f", "threshold": 0.002},
    "IBIT_PX":     {"label": "IBIT 가격",       "unit": "$",  "fmt": ".2f", "threshold": 0.50},
}


def _empty_stat_fields() -> dict:
    return {
        "latest_date": None,
        "change5d": None, "change20d": None, "zscore20": None,
        "regime": None, "trend20": None,
        "recent_flow": None, "recent_points": None,
    }


def fetch_macro_context() -> dict:
    """
    거시 지표 전체 수집.

    반환 dict 구조 (모든 값은 선택적):
      {
        "TNX_10Y":     {label, unit, fmt, value, change5d, change20d, zscore20, regime, trend20, recent_flow, recent_points, latest_date},
        "FVX_5Y":      {...},
        "DXY":         {...},
        "STABLE_MCAP": {label, unit, fmt, value, ...},
        "USDT_DOM":    {...},
        "BTC_DOM":     {...},
        "HYG_LQD":     {...},
        "IBIT_PX":     {..., vol_ratio, vol_latest, vol_20ma},
        "_eth_btc":    {...},
        "_trad_markets": {...},
        "_history_summary": {...},  # macro_history.attach_macro_history_summary가 주입
      }
    """
    result: dict = {}

    # ── 시장 금리·달러 (yfinance, 실시간) ────────────────────
    rates = _fetch_market_rates()
    for key in ("TNX_10Y", "FVX_5Y", "DXY"):
        meta = _METRIC_META[key]
        stats = _compute_stats(rates.get(key), change_threshold=meta["threshold"])
        result[key] = {
            "label": meta["label"], "unit": meta["unit"], "fmt": meta["fmt"],
            **stats,
        }

    # ── DefiLlama 스테이블코인 ──────────────────────────────
    sc = _fetch_stablecoins()
    result["STABLE_MCAP"] = {
        "label": _METRIC_META["STABLE_MCAP"]["label"],
        "unit":  _METRIC_META["STABLE_MCAP"]["unit"],
        "fmt":   _METRIC_META["STABLE_MCAP"]["fmt"],
        "value": sc["stable_total_b"],
        **_empty_stat_fields(),
    }
    result["USDT_DOM"] = {
        "label": _METRIC_META["USDT_DOM"]["label"],
        "unit":  _METRIC_META["USDT_DOM"]["unit"],
        "fmt":   _METRIC_META["USDT_DOM"]["fmt"],
        "value": sc["usdt_dom"],
        **_empty_stat_fields(),
    }

    # ── CoinGecko BTC 도미넌스 ──────────────────────────────
    result["BTC_DOM"] = {
        "label": _METRIC_META["BTC_DOM"]["label"],
        "unit":  _METRIC_META["BTC_DOM"]["unit"],
        "fmt":   _METRIC_META["BTC_DOM"]["fmt"],
        "value": _fetch_btc_dominance(),
        **_empty_stat_fields(),
    }

    # ── HYG/LQD 신용 스프레드 프록시 ─────────────────────────
    hyg_lqd_series = _fetch_credit_spread()
    hyg_meta  = _METRIC_META["HYG_LQD"]
    hyg_stats = _compute_stats(hyg_lqd_series, change_threshold=hyg_meta["threshold"])
    result["HYG_LQD"] = {
        "label": hyg_meta["label"], "unit": hyg_meta["unit"], "fmt": hyg_meta["fmt"],
        **hyg_stats,
    }

    # ── IBIT (Spot BTC ETF) ─────────────────────────────────
    ibit = _fetch_btc_etf()
    ibit_meta  = _METRIC_META["IBIT_PX"]
    ibit_stats = _compute_stats(ibit.get("close_series"), change_threshold=ibit_meta["threshold"])
    result["IBIT_PX"] = {
        "label": ibit_meta["label"], "unit": ibit_meta["unit"], "fmt": ibit_meta["fmt"],
        **ibit_stats,
        "vol_latest": ibit.get("vol_latest"),
        "vol_20ma":   ibit.get("vol_20ma"),
        "vol_ratio":  ibit.get("vol_ratio"),
    }

    # ── ETH/BTC 비율 ────────────────────────────────────────
    result["_eth_btc"] = _fetch_eth_btc_ratio()

    # ── 전통 시장 (yfinance) ────────────────────────────────
    result["_trad_markets"] = _fetch_traditional_markets()

    attach_macro_history_summary(result)
    return result


# ── 포매팅 (Claude 프롬프트 삽입용) ───────────────────────────

def _change_suffix(key: str, unit: str) -> str:
    """히스토리 기반 24h/72h/7d 변화량 값 뒤에 붙일 단위."""
    if key == "STABLE_MCAP":
        return "B"
    if unit == "%":
        return "%p"
    # $, "" 등은 접미사 없음 (숫자 자체로 명확)
    return ""


def _change_format(key: str, fmt: str) -> str:
    """
    24h/72h/7d 변화량을 위한 포맷 스펙.
    HYG/LQD 처럼 소수점 이하가 중요한 비율은 지표 fmt(.4f)를 재활용하고,
    그 외는 .2f 로 간결하게 표시.
    """
    if key == "HYG_LQD":
        return "+.4f"
    return "+.2f"


def format_macro_context(macro: dict) -> str:
    """
    Claude 프롬프트 삽입용 거시 지표 텍스트.
    원시값 + 변화율 + 최근 시계열 요약을 포함하며, 해석은 Claude에 위임.
    """
    lines = [
        "[거시경제 지표]",
        "  ※ 금리·달러 인덱스·채권 ETF·IBIT 는 yfinance 실시간 종가 기준 시계열입니다.",
        "  ※ 24h/72h/7d 변화는 서버 스냅샷 기반(30분 주기 누적) — 스냅샷 부족 시 5일 변화를 함께 참고.",
    ]

    for key, d in macro.items():
        if str(key).startswith("_"):
            continue
        v = d.get("value")
        unit = d.get("unit", "")
        fmt  = d.get("fmt", ".2f")

        if v is None:
            lines.append(f"  {d.get('label', key)} ({key}): 데이터 없음")
            continue

        val_str = f"{v:{fmt}}{unit}" if unit != "$" else f"${v:{fmt}}"

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

        suffix = _change_suffix(key, unit)
        cfmt   = _change_format(key, fmt)

        if d.get("change24h") is not None:
            extras.append(f"24h 변화 {d['change24h']:{cfmt}}{suffix}")
        if d.get("change72h") is not None:
            extras.append(f"72h 변화 {d['change72h']:{cfmt}}{suffix}")
        if d.get("change7d") is not None:
            ch7 = d["change7d"]
            samples = d.get("change7d_samples")
            if samples is not None and samples < 3:
                extras.append(f"7d 변화 {ch7:{cfmt}}{suffix}(스냅샷{samples}개-5일변화참고)")
            else:
                extras.append(f"7d 변화 {ch7:{cfmt}}{suffix}")
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
            point_str = " → ".join(
                f"{p['date']}:{p['value']:{fmt}}{unit if unit != '$' else ''}"
                for p in recent_points
            )
            extras.append(f"최근 3개 {point_str}")

        # IBIT 전용 보조 필드: 거래량/20MA
        if key == "IBIT_PX" and d.get("vol_ratio") is not None:
            extras.append(f"거래량/20MA {d['vol_ratio']:.2f}x")

        extra_str = f"  ({', '.join(extras)})" if extras else ""
        lines.append(f"  {d['label']} ({key}): {val_str}{extra_str}")

    # ── 서버 저장 기반 거시 추이 ─────────────────────────────
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

    # ── ETH/BTC 비율 섹션 ────────────────────────────────────
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
        lines.append(f"  ETH/BTC 비율: {ratio:.6f}{chg_str}{price_str}")
        lines.append(
            "  ※ 해석: 비율↑ = ETH 상대 강세(알트 시즌 가능성) / 비율↓ = BTC 단독 강세 또는 리스크오프. "
            "BTC 도미넌스와 함께 순환 방향 확인."
        )
    elif eb.get("error"):
        lines.append(f"  ETH/BTC 비율: 수집 실패 — {eb['error']}")

    # ── 전통 시장 섹션 ──────────────────────────────────────
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

    # ── 종합 해석 ────────────────────────────────────────────
    lines.append(
        "  ※ 해석 참고: "
        "10Y/5Y 금리·DXY↑ = 실질 할인율 상승·달러 강세 → BTC 등 위험자산 압박. "
        "HYG/LQD↓ = 신용 스프레드 확대 → 리스크오프 선행 신호(BTC 조정 위험). "
        "IBIT 거래량/20MA 1.5x 이상 + 가격↑ = 기관 수급 가속(현물 ETF 유입). "
        "스테이블코인 시총↑+USDT 도미넌스↓+BTC 도미넌스↑ = 알트→BTC 순환(위험선호 유지). "
        "스테이블코인 시총 정체+USDT 도미넌스↑+BTC 도미넌스↑ = 방어적 포지셔닝. "
        "스테이블코인 시총↑+USDT 도미넌스↓+BTC 도미넌스↓ = 알트 시즌 가능성. "
        "단기 트레이딩 시그널보다 중기 레짐 필터로 활용하고, "
        "기술적 지표와 파생상품 데이터로 최종 시그널을 판단하세요."
    )

    return "\n".join(lines)
