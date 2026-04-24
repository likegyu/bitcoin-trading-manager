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


# ── VWAP (Volume Weighted Average Price) ─────
def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    앵커드 VWAP: 각 UTC 날짜(자정)를 기준점으로 삼아 일중 누적 VWAP 계산.
    인덱스가 DatetimeTZ이면 date()로 날짜 분리, 아닌 경우 전체를 하나로 처리.

    - vwap      : 당일 앵커 기준 VWAP
    - vwap_dev  : 현재가의 VWAP 대비 괴리율(%) — 양수=VWAP 위, 음수=VWAP 아래
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3

    try:
        # DatetimeIndex인 경우 UTC 날짜별 그루핑
        if isinstance(df.index, pd.DatetimeIndex):
            dates = df.index.normalize()  # 일 단위 절사 (시간 0으로)
        else:
            # 타임스탬프 컬럼이 없으면 전체를 하나의 세션으로 처리
            dates = pd.Series(["session"] * len(df), index=df.index)

        vwap_vals = np.empty(len(df))
        vwap_vals[:] = np.nan

        for date_key in pd.unique(dates):
            mask = dates == date_key
            tp_day  = typical[mask]
            vol_day = df["volume"][mask]
            cum_tp_vol = (tp_day * vol_day).cumsum()
            cum_vol    = vol_day.cumsum().replace(0, np.nan)
            vwap_day   = cum_tp_vol / cum_vol
            idx_pos    = np.where(mask)[0]
            vwap_vals[idx_pos] = vwap_day.values

        df["vwap"] = vwap_vals
        df["vwap_dev"] = (df["close"] - df["vwap"]) / df["vwap"] * 100
    except Exception:
        df["vwap"]     = np.nan
        df["vwap_dev"] = np.nan

    return df


# ── Supertrend ────────────────────────────────
def add_supertrend(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """
    Supertrend 지표 (ATR 기반 추세 추종).

    - supertrend        : Supertrend 밴드 레벨
    - supertrend_dir    : 1=상승(매수), -1=하락(매도)
    - supertrend_signal : 방향 변화 시 'BUY'/'SELL', 그 외 None

    ATR이 이미 계산되어 있어야 함 (add_atr 선행 필요).
    """
    if "atr" not in df.columns:
        df = add_atr(df, period)

    hl2    = (df["high"] + df["low"]) / 2
    upper  = hl2 + multiplier * df["atr"]
    lower  = hl2 - multiplier * df["atr"]

    n   = len(df)
    st  = np.full(n, np.nan)
    dir_ = np.full(n, 1, dtype=int)   # 1=상승, -1=하락

    # 첫 유효 ATR 위치 탐색
    first_valid = df["atr"].first_valid_index()
    if first_valid is None:
        df["supertrend"]        = np.nan
        df["supertrend_dir"]    = np.nan
        df["supertrend_signal"] = None
        return df

    start = df.index.get_loc(first_valid)

    # 첫 봉 초기화
    close_arr  = df["close"].values
    upper_arr  = upper.values
    lower_arr  = lower.values

    final_upper = upper_arr.copy()
    final_lower = lower_arr.copy()

    for i in range(start + 1, n):
        # Final Upper Band
        if upper_arr[i] < final_upper[i - 1] or close_arr[i - 1] > final_upper[i - 1]:
            final_upper[i] = upper_arr[i]
        else:
            final_upper[i] = final_upper[i - 1]

        # Final Lower Band
        if lower_arr[i] > final_lower[i - 1] or close_arr[i - 1] < final_lower[i - 1]:
            final_lower[i] = lower_arr[i]
        else:
            final_lower[i] = final_lower[i - 1]

        # Direction
        if dir_[i - 1] == -1 and close_arr[i] > final_upper[i]:
            dir_[i] = 1
        elif dir_[i - 1] == 1 and close_arr[i] < final_lower[i]:
            dir_[i] = -1
        else:
            dir_[i] = dir_[i - 1]

        # Supertrend value
        st[i] = final_lower[i] if dir_[i] == 1 else final_upper[i]

    df["supertrend"]     = st
    df["supertrend_dir"] = dir_.astype(float)
    df["supertrend_dir"] = df["supertrend_dir"].where(df.index >= df.index[start])

    # 신호: 방향 전환 시점만 표시
    dir_series = pd.Series(dir_, index=df.index)
    signal = pd.Series([None] * n, index=df.index, dtype=object)
    signal[(dir_series == 1) & (dir_series.shift(1) == -1)] = "BUY"
    signal[(dir_series == -1) & (dir_series.shift(1) == 1)] = "SELL"
    df["supertrend_signal"] = signal

    return df


# ── 실현변동성 (Realized Volatility) ─────────
# TF별 캔들 수 → 연환산 계수
# 크립토 시장은 24/7 거래 → 전통 주식의 영업일 252 가 아니라 365 사용.
# (과거 252 를 쓰면 DVOL 내재변동성 대비 약 √(252/365) ≈ 83%, 즉 RV 가
#  17% 저평가되어 IV 프리미엄 판정이 편향됨.)
_RV_ANNUALIZE = {
    "1d":  365,
    "4h":  365 * 6,
    "1h":  365 * 24,
    "15m": 365 * 96,
    "5m":  365 * 288,
}

def add_realized_vol(df: pd.DataFrame, tf: str = "1d", period: int = 20) -> pd.DataFrame:
    """
    로그 수익률 표준편차 기반 연환산 실현변동성(Realized Volatility).

    rv_{period}d : period 봉 기준 연환산 RV (%)
    rv_7d        : 7봉 기준 단기 RV (%)  — DVOL과 비교하기 위한 단기 창

    DVOL(내재변동성)과 비교:
      RV > DVOL : IV 할인 → 옵션 매수 유리 / 변동성 확대 예상
      RV < DVOL : IV 프리미엄 → 방향 불확실성 높게 pricing 중
    """
    ann_factor = _RV_ANNUALIZE.get(tf, 365)
    log_ret    = np.log(df["close"] / df["close"].shift(1))

    rv_long = log_ret.rolling(period).std() * np.sqrt(ann_factor) * 100
    rv_short = log_ret.rolling(7).std()     * np.sqrt(ann_factor) * 100

    df[f"rv_{period}"] = rv_long.round(2)
    df["rv_7"]         = rv_short.round(2)
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
def add_all_indicators(df: pd.DataFrame, tf: str = "1d") -> pd.DataFrame:
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger(df)
    df = add_moving_averages(df)
    df = add_volume_ma(df)
    # add_stochastic 제거 — summarize_indicators에서 사용하지 않음 (연산 낭비)
    df = add_atr(df)
    df = add_vwap(df)
    df = add_supertrend(df)   # ATR 이후에 호출 (내부에서 ATR 재사용)
    df = add_realized_vol(df, tf=tf)
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

    # ── 실현변동성 표시값 계산 ──
    rv20 = last.get("rv_20", np.nan)
    rv7  = last.get("rv_7",  np.nan)
    if not np.isnan(rv20) and not np.isnan(rv7):
        rv_str = f"RV20: {rv20:.1f}%  RV7: {rv7:.1f}%  (연환산 — DVOL과 비교 시 IV 프리미엄/디스카운트 판단)"
    elif not np.isnan(rv20):
        rv_str = f"RV20: {rv20:.1f}%  (연환산)"
    else:
        rv_str = "N/A"

    # ── VWAP 표시값 계산 ──
    vwap_val = last.get("vwap", np.nan)
    vwap_dev = last.get("vwap_dev", np.nan)
    if not np.isnan(vwap_val) and not np.isnan(vwap_dev):
        vwap_pos = "위" if vwap_dev >= 0 else "아래"
        vwap_str = f"${vwap_val:,.2f} (현재가 VWAP {vwap_pos} {abs(vwap_dev):.2f}%)"
    else:
        vwap_str = "N/A"

    # ── Supertrend 표시값 계산 ──
    st_val = last.get("supertrend", np.nan)
    st_dir = last.get("supertrend_dir", np.nan)
    st_signal = last.get("supertrend_signal", None)
    if not np.isnan(st_val) and not np.isnan(st_dir):
        st_dir_str = "상승(매수)" if st_dir == 1 else "하락(매도)"
        st_signal_str = f" ⚡신호: {st_signal}" if st_signal else ""
        supertrend_str = f"${st_val:,.2f}  방향: {st_dir_str}{st_signal_str}"
    else:
        supertrend_str = "N/A"

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
        f"VWAP(일중): {vwap_str}",
        f"Supertrend(10,3): {supertrend_str}",
        f"실현변동성: {rv_str}",
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
