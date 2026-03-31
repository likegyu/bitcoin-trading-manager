# =============================================
# 기술적 보조지표 계산 모듈
# RSI · MACD · 볼린저밴드 · 이동평균 · 거래량MA · 피보나치
# =============================================
from __future__ import annotations

import pandas as pd
import numpy as np
from time_utils import format_kst

FIB_SWING_WINDOWS = {"1d": 5, "4h": 5, "1h": 5, "15m": 5, "5m": 3}


# ── RSI ──────────────────────────────────────
def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta = df["close"].diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


# ── MACD ─────────────────────────────────────
def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, sig: int = 9) -> pd.DataFrame:
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    df["macd"]        = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=sig, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]
    return df


# ── 볼린저밴드 ────────────────────────────────
def add_bollinger(df: pd.DataFrame, period: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    mid  = df["close"].rolling(period).mean()
    std  = df["close"].rolling(period).std()
    df["bb_upper"] = mid + n_std * std
    df["bb_mid"]   = mid
    df["bb_lower"] = mid - n_std * std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / mid
    denom = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    df["bb_pct"]   = (df["close"] - df["bb_lower"]) / denom
    return df


# ── 이동평균선 ────────────────────────────────
def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    # SMA20 제거 — bb_mid로 BB 내부에서 계산, summarize_indicators 출력 없는 연산 낭비
    # EMA21 제거 — EMA9·SMA50·SMA200으로 충분, 출력 없는 연산 낭비
    for p in [50, 200]:
        df[f"sma_{p}"] = df["close"].rolling(p).mean()
    df["ema_9"] = df["close"].ewm(span=9, adjust=False).mean()
    return df


# ── 거래량 MA ─────────────────────────────────
def add_volume_ma(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    df["volume_ma"] = df["volume"].rolling(period).mean()
    return df


# ── 스토캐스틱 ────────────────────────────────
def add_stochastic(df: pd.DataFrame, k: int = 14, d: int = 3) -> pd.DataFrame:
    low_min  = df["low"].rolling(k).min()
    high_max = df["high"].rolling(k).max()
    denom = (high_max - low_min).replace(0, np.nan)
    df["stoch_k"] = 100 * (df["close"] - low_min) / denom
    df["stoch_d"] = df["stoch_k"].rolling(d).mean()
    return df


# ── ATR (Average True Range) ─────────────────
def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    hl  = df["high"] - df["low"]
    hc  = (df["high"] - df["close"].shift()).abs()
    lc  = (df["low"]  - df["close"].shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr"] = tr.ewm(com=period - 1, adjust=False).mean()
    return df


def fib_window_for_tf(tf: str) -> int:
    return FIB_SWING_WINDOWS.get(tf, 5)


# ── 스윙 포인트 기반 피보나치 (Claude 프롬프트용) ──
def fibonacci_swing_levels(
    df: pd.DataFrame,
    window: int = 5,
    lookback: int = 100,
    min_move_pct: float = 0.008,
) -> dict | None:
    """
    지그재그(Zigzag) 알고리즘 기반 피보나치 되돌림 레벨 계산.

    기존 방식의 문제:
      스윙 고점/저점을 각각 독립적으로 뽑아 '가장 최근 것'끼리 페어링하면
      두 극점이 서로 연속된 단일 다리(leg)가 아닐 수 있음.
      예: 저점(4주 전) → 고점(2주 전) 페어인데 현재가는 그 범위 아래.

    지그재그 방식:
      1. 모든 스윙 고점/저점을 시간순으로 합침
      2. 같은 타입이 연속되면 더 극단값으로 교체 (H는 더 높게, L은 더 낮게)
      3. 결과: H-L-H-L … 교차 배열 (지그재그)
      4. 마지막 두 극점 = 현재 진행 중인 마지막 다리

    Parameters
    ----------
    window       : 스윙 확인 기준 (좌우 N캔들 내 극값이어야 스윙으로 인정)
    lookback     : 탐색 범위 (캔들 수)
    min_move_pct : 최소 이동폭 비율 — 이보다 작은 다리는 노이즈로 제외

    Returns
    -------
    dict  : levels, swing_high, swing_high_ago, swing_low, swing_low_ago
    None  : 유효한 다리를 찾지 못한 경우
    """
    data = df.iloc[-(lookback + window):].reset_index(drop=True)
    n    = len(data)

    # ── 스윙 고점 / 저점 탐지 ─────────────────────
    raw: list[tuple[int, float, str]] = []  # (index, price, 'H'|'L')

    for i in range(window, n - window):
        hi_win = data.iloc[i - window : i + window + 1]["high"]
        lo_win = data.iloc[i - window : i + window + 1]["low"]
        is_high = int(hi_win.idxmax()) == i
        is_low = int(lo_win.idxmin()) == i

        # 하나의 봉이 동시에 로컬 고점/저점이면 outside bar 성격이 강해
        # 단일 leg 기준 피보나치 앵커로는 모호하므로 제외한다.
        if is_high and is_low:
            continue
        if is_high:
            raw.append((i, float(data.iloc[i]["high"]), "H"))
        if is_low:
            raw.append((i, float(data.iloc[i]["low"]),  "L"))

    if len(raw) < 2:
        return None

    raw.sort(key=lambda x: x[0])

    # ── 지그재그 구성 ──────────────────────────────
    # 같은 타입이 연속되면 더 극단값으로 교체, 다른 타입이면 추가
    zz: list[tuple[int, float, str]] = [raw[0]]
    for p in raw[1:]:
        if p[2] == zz[-1][2]:
            if p[2] == "H" and p[1] > zz[-1][1]:
                zz[-1] = p
            elif p[2] == "L" and p[1] < zz[-1][1]:
                zz[-1] = p
        else:
            zz.append(p)

    if len(zz) < 2:
        return None

    # ── 마지막 다리(leg) 추출 ─────────────────────
    p2 = zz[-1]   # 더 최근 극점
    p1 = zz[-2]   # 그 직전 극점 (반드시 다른 타입)

    # 최소 이동폭 검증 (노이즈 다리 제외)
    move_size = abs(p2[1] - p1[1])
    mid_price = (p2[1] + p1[1]) / 2
    if mid_price > 0 and move_size / mid_price < min_move_pct:
        return None

    # ── 방향별 고점/저점 결정 ─────────────────────
    if p1[2] == "L":  # L → H : 상승 다리 → 상승분의 되돌림 레벨
        direction = "up"
        sw_low,  sw_low_ago  = p1[1], n - 1 - p1[0]
        sw_high, sw_high_ago = p2[1], n - 1 - p2[0]
        leg_start, leg_start_ago, leg_start_type = p1[1], n - 1 - p1[0], p1[2]
        leg_end, leg_end_ago, leg_end_type = p2[1], n - 1 - p2[0], p2[2]
    else:             # H → L : 하락 다리 → 하락분의 되돌림 레벨
        direction = "down"
        sw_high, sw_high_ago = p1[1], n - 1 - p1[0]
        sw_low,  sw_low_ago  = p2[1], n - 1 - p2[0]
        leg_start, leg_start_ago, leg_start_type = p1[1], n - 1 - p1[0], p1[2]
        leg_end, leg_end_ago, leg_end_type = p2[1], n - 1 - p2[0], p2[2]

    diff   = sw_high - sw_low
    # 차트 우측 라벨은 핵심 3개만 사용하고, 패널 표는 5개를 보여준다.
    core_ratios = [0.382, 0.5, 0.618]
    display_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
    if direction == "up":
        levels = {f"{r:.3f}": sw_high - r * diff for r in core_ratios}
        display_levels = {f"{r:.3f}": sw_high - r * diff for r in display_ratios}
        anchors = {"0.000": sw_high, "1.000": sw_low}
    else:
        levels = {f"{r:.3f}": sw_low + r * diff for r in core_ratios}
        display_levels = {f"{r:.3f}": sw_low + r * diff for r in display_ratios}
        anchors = {"0.000": sw_low, "1.000": sw_high}

    return {
        "levels":         levels,
        "display_levels": display_levels,
        "anchors":        anchors,
        "direction":      direction,
        "swing_high":     sw_high,
        "swing_high_ago": sw_high_ago,
        "swing_low":      sw_low,
        "swing_low_ago":  sw_low_ago,
        "leg_start":      leg_start,
        "leg_start_ago":  leg_start_ago,
        "leg_start_type": leg_start_type,
        "leg_end":        leg_end,
        "leg_end_ago":    leg_end_ago,
        "leg_end_type":   leg_end_type,
    }


# ── 모든 지표 한번에 ──────────────────────────
def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger(df)
    df = add_moving_averages(df)
    df = add_volume_ma(df)
    # add_stochastic 제거 — summarize_indicators에서 사용하지 않음 (연산 낭비)
    df = add_atr(df)
    return df




# ── 지표 요약 텍스트 (Claude 프롬프트용) ──────
def summarize_indicators(tf: str, df: pd.DataFrame) -> str:
    last  = df.iloc[-1]
    price = last["close"]

    # ── 100캔들 고점·저점 (지지·저항 원시값) ──
    # ※ 50캔들 레벨 제거 — 100캔들 범위의 서브셋으로 독립 정보 없음 (레벨 과밀 방지)
    high100 = df["high"].iloc[-100:].max()
    low100  = df["low"].iloc[-100:].min()

    # ── TF별 OHLC 캔들 수 (핵심 최근봉만 — 원시 숫자 과부하 방지) ──
    # 1d: 15→10 — RSI/MACD Hist 시계열(7봉)과 비대칭 해소, 캔들 패턴 과해석 방지
    _candle_n = {"1d": 10, "4h": 12, "1h": 10, "15m": 10, "5m": 6}
    n_candles = _candle_n.get(tf, 10)
    recent_n  = df.iloc[-n_candles:]

    # ── OHLC 테이블 ──
    _time_tfs = ("5m", "15m", "1h", "4h")   # 시간 표시 포함 TF
    ohlc_rows = []
    n_recent = len(recent_n)
    for i, (idx, row) in enumerate(recent_n.iterrows()):
        o, h, l, c, v = row["open"], row["high"], row["low"], row["close"], row["volume"]
        rng   = h - l if h != l else 1e-9
        upper = (h - max(o, c)) / rng * 100
        lower = (min(o, c) - l) / rng * 100
        body  = abs(c - o) / rng * 100
        try:
            lbl = format_kst(idx, "%m/%d %H:%M") if tf in _time_tfs else format_kst(idx, "%m/%d")
        except Exception:
            lbl = str(idx)[:16]
        incomplete = "  ⚠️미완성봉" if i == n_recent - 1 else ""
        # Vol: 절대값만 있으면 cross-TF 앵커링 위험 — MA 대비 비율을 인라인으로 함께 표시
        vol_ma_val = row.get("volume_ma", np.nan)
        if not np.isnan(vol_ma_val) and vol_ma_val > 0:
            vol_str = f"Vol:{v:,.0f}(MA비{v/vol_ma_val*100:.0f}%)"
        else:
            vol_str = f"Vol:{v:,.0f}"
        ohlc_rows.append(
            f"  {lbl}  O:{o:,.0f} H:{h:,.0f} L:{l:,.0f} C:{c:,.0f}  "
            f"몸통:{body:.0f}% 위꼬리:{upper:.0f}% 아래꼬리:{lower:.0f}%  {vol_str}{incomplete}"
        )

    # ── 지표 시계열: 추세 흐름 파악에 충분한 범위로 축소 ──
    # ※ 1d는 추세 방향 파악용 7봉 / 나머지는 10봉으로 통일
    # (20봉 나열은 LLM이 토큰 단위 처리 시 미세 수치 차이 과해석 위험)
    _series_n    = {"1d": 7, "4h": 10, "1h": 10, "15m": 8, "5m": 5}
    _series_lbl  = {
        "1d":  "7봉 ≈ 7일",
        "4h":  "10봉 ≈ 40시간",
        "1h":  "10봉 ≈ 10시간",
        "15m": "10봉 ≈ 2.5시간",
        "5m":  "10봉 ≈ 50분",
    }
    n_series    = _series_n.get(tf, 20)
    series_lbl  = _series_lbl.get(tf, f"{n_series}봉")
    # ※ Stoch K 시계열 제거 — RSI와 동류 오실레이터로 상관도 높음 (독립 근거 중복 방지)
    # RSI: 소수점 1자리 → 정수 (미세 차이에서 패턴 과해석 방지)
    rsi_series  = " → ".join(f"{v:.0f}" for v in df["rsi"].iloc[-n_series:])
    hist_series = " → ".join(f"{v:+.1f}" for v in df["macd_hist"].iloc[-n_series:])

    # ── 지표 현재값 계산 ──
    # [가격 위치 요약] 제거 — tf_alignment_summary가 이미 SMA200 방향을 커버;
    # 내부 ▲/▼·BB%B·RSI 사전 라벨은 anchoring만 추가. 원시값으로 직접 전달.
    hist_now  = last["macd_hist"]
    hist_prev = df.iloc[-2]["macd_hist"] if len(df) >= 2 else hist_now

    h100_gap = (high100 - price) / price * 100
    l100_gap = (price - low100)  / price * 100

    # 5m 섹션: 진입 타이밍 참고 전용 역할 명시
    tf_header = (
        f"=== [{tf}] ===  ※ 진입 타이밍 참고 전용 — 추세 판단 근거로 사용하지 마세요"
        if tf == "5m" else f"=== [{tf}] ==="
    )

    lines = [
        tf_header,
        f"현재가: ${price:,.2f}",
        f"",
        f"[최근 {n_candles}캔들 OHLC]  ← 마지막 행은 ⚠️미완성봉",
    ] + ohlc_rows + [
        f"",
        f"[지표 현재값]",
        f"MACD: {last['macd']:.2f} / Signal: {last['macd_signal']:.2f} / Hist: {hist_now:+.2f} (직전: {hist_prev:+.2f})",
        f"볼린저밴드: 상단 ${last['bb_upper']:,.2f} / 하단 ${last['bb_lower']:,.2f}  BB%B: {last['bb_pct']:.3f}",
        # bb_mid(SMA20) 제거 — EMA9·SMA50·SMA200이 MA 맥락 제공, 중복 정보
        f"SMA200: ${last['sma_200']:,.2f}",
        f"SMA50: ${last['sma_50']:,.2f}",
        f"EMA9: ${last['ema_9']:,.2f}",
        f"ATR(14): ${last['atr']:.2f}",
        f"거래량: {last['volume']:,.0f} / 거래량MA20: {last['volume_ma']:,.0f}",
    ]

    if tf != "5m":
        lines += [
            f"100캔들 키레벨: 고점까지 +{h100_gap:.1f}% (${high100:,.0f}) / 저점까지 -{l100_gap:.1f}% (${low100:,.0f})",
            f"",
            f"[지표 추이 — {series_lbl}]  ※ RSI·MACD Hist는 독립 오실레이터 아님 — 수렴 시 과신 주의",
            f"RSI:       {rsi_series}",
            f"MACD Hist: {hist_series}",
        ]
    else:
        lines += [
            f"",
            f"[5m 참고 메모]",
            f"최근 5m는 진입 타이밍 힌트용. 방향 결론보다 과열·속도 변화 확인에만 제한적으로 사용.",
        ]

    return "\n".join(lines)
