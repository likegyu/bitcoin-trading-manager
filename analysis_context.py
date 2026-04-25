from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import pandas as pd


_REQUIRED_TF_COLUMNS = (
    "open", "high", "low", "close", "volume",
    "rsi", "macd_hist", "bb_pct", "sma_200", "sma_50", "ema_9",
    "volume_ma", "atr",
)

_MACRO_KEYS = ("TNX_10Y", "FVX_5Y", "DXY", "STABLE_MCAP", "USDT_DOM", "BTC_DOM", "HYG_LQD", "IBIT_PX")
_MARKET_KEYS = ("funding_rate", "open_interest", "combined_oi", "cvd_4h", "ob_imbalance")
_ACCOUNT_KEYS = ("wallet_balance", "account_equity", "open_position_count", "today_cash_pnl")


def _as_float(value: Any) -> float | None:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num):
        return None
    return num


def _fmt_num(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def _latest_label(index_value: Any) -> str:
    try:
        if hasattr(index_value, "tz_convert"):
            return index_value.tz_convert("Asia/Seoul").strftime("%m-%d %H:%M KST")
        if isinstance(index_value, datetime):
            dt = index_value
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone().strftime("%m-%d %H:%M")
    except Exception:
        pass
    return str(index_value)[:19]


def _zone_rsi(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value >= 70:
        return "overbought"
    if value <= 30:
        return "oversold"
    if value >= 55:
        return "bullish"
    if value <= 45:
        return "bearish"
    return "neutral"


def _macro_stale_days(latest_date: str | None, now: datetime | None = None) -> int | None:
    if not latest_date:
        return None
    try:
        dt = datetime.strptime(latest_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except Exception:
        return None
    now_dt = now or datetime.now(timezone.utc)
    return max(0, int((now_dt - dt).total_seconds() // 86400))


def _tf_feature(tf: str, df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {"tf": tf, "status": "missing", "warnings": ["캔들 데이터 없음"]}

    missing_cols = [col for col in _REQUIRED_TF_COLUMNS if col not in df.columns]
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last

    close = _as_float(last.get("close"))
    high = _as_float(last.get("high"))
    low = _as_float(last.get("low"))
    sma200 = _as_float(last.get("sma_200"))
    sma50 = _as_float(last.get("sma_50"))
    ema9 = _as_float(last.get("ema_9"))
    rsi = _as_float(last.get("rsi"))
    macd_hist = _as_float(last.get("macd_hist"))
    macd_prev = _as_float(prev.get("macd_hist"))
    bb_pct = _as_float(last.get("bb_pct"))
    atr = _as_float(last.get("atr"))
    volume = _as_float(last.get("volume"))
    volume_ma = _as_float(last.get("volume_ma"))
    vwap_dev = _as_float(last.get("vwap_dev"))
    rv20 = _as_float(last.get("rv_20"))
    rv7 = _as_float(last.get("rv_7"))

    trend_score = 0
    if close is not None and sma200 is not None:
        trend_score += 1 if close > sma200 else -1
    if ema9 is not None and sma50 is not None:
        trend_score += 1 if ema9 > sma50 else -1
    if sma50 is not None and sma200 is not None:
        trend_score += 1 if sma50 > sma200 else -1

    if trend_score >= 2:
        trend_bias = "bullish"
    elif trend_score <= -2:
        trend_bias = "bearish"
    else:
        trend_bias = "mixed"

    lookback = df.iloc[-21:-1] if len(df) >= 21 else df.iloc[:-1]
    prior_high = _as_float(lookback["high"].max()) if not lookback.empty and "high" in lookback else None
    prior_low = _as_float(lookback["low"].min()) if not lookback.empty and "low" in lookback else None
    structure = "inside_range"
    if close is not None and prior_high is not None and close > prior_high:
        structure = "breakout"
    elif close is not None and prior_low is not None and close < prior_low:
        structure = "breakdown"

    volume_ratio = None
    if volume is not None and volume_ma and volume_ma > 0:
        volume_ratio = volume / volume_ma

    atr_pct = None
    if atr is not None and close and close > 0:
        atr_pct = atr / close * 100

    distance_sma200 = None
    if close is not None and sma200 and sma200 > 0:
        distance_sma200 = (close - sma200) / sma200 * 100

    macd_delta = None
    if macd_hist is not None and macd_prev is not None:
        macd_delta = macd_hist - macd_prev

    warnings: list[str] = []
    if missing_cols:
        warnings.append("누락 컬럼: " + ", ".join(missing_cols[:5]))
    if len(df) < 60:
        warnings.append(f"표본 부족: {len(df)}캔들")
    if close is None:
        warnings.append("현재가 없음")

    return {
        "tf": tf,
        "status": "ok" if not warnings else "caution",
        "latest": _latest_label(df.index[-1]),
        "rows": int(len(df)),
        "close": close,
        "trend_bias": trend_bias,
        "trend_score": trend_score,
        "structure": structure,
        "rsi": rsi,
        "rsi_zone": _zone_rsi(rsi),
        "macd_hist": macd_hist,
        "macd_delta": macd_delta,
        "bb_pct": bb_pct,
        "volume_ratio": volume_ratio,
        "atr_pct": atr_pct,
        "vwap_dev": vwap_dev,
        "rv20": rv20,
        "rv7": rv7,
        "distance_sma200": distance_sma200,
        "prior_high": prior_high,
        "prior_low": prior_low,
        "warnings": warnings,
    }


def build_derived_features(multi_tf_data: dict) -> dict:
    tf_order = ("1d", "4h", "1h", "15m", "5m")
    features = [_tf_feature(tf, multi_tf_data.get(tf)) for tf in tf_order if tf in multi_tf_data]

    higher = [f for f in features if f.get("tf") in ("1d", "4h", "1h")]
    lower = [f for f in features if f.get("tf") in ("15m", "5m")]
    bullish_higher = sum(1 for f in higher if f.get("trend_bias") == "bullish")
    bearish_higher = sum(1 for f in higher if f.get("trend_bias") == "bearish")
    bullish_lower = sum(1 for f in lower if f.get("trend_bias") == "bullish")
    bearish_lower = sum(1 for f in lower if f.get("trend_bias") == "bearish")

    if bullish_higher > bearish_higher:
        higher_bias = "bullish"
    elif bearish_higher > bullish_higher:
        higher_bias = "bearish"
    else:
        higher_bias = "mixed"

    if bullish_lower > bearish_lower:
        lower_bias = "bullish"
    elif bearish_lower > bullish_lower:
        lower_bias = "bearish"
    else:
        lower_bias = "mixed"

    conflicts = []
    if higher_bias != "mixed" and lower_bias != "mixed" and higher_bias != lower_bias:
        conflicts.append(f"상위 TF({higher_bias})와 단기 TF({lower_bias}) 방향 충돌")

    incomplete_warning = "마지막 캔들은 형성 중이므로 트리거 확정에 사용 금지"

    return {
        "timeframes": features,
        "higher_tf_bias": higher_bias,
        "lower_tf_bias": lower_bias,
        "conflicts": conflicts,
        "warnings": [incomplete_warning],
    }


def build_data_quality_report(
    multi_tf_data: dict,
    macro_snapshot: dict | None,
    market_ctx: dict | None,
    account_ctx: dict | None,
    now: datetime | None = None,
) -> dict:
    now_dt = now or datetime.now(timezone.utc)
    cautions: list[str] = []
    trusted: list[str] = []
    no_use: list[str] = []

    tf_rows = {}
    for tf, df in (multi_tf_data or {}).items():
        rows = len(df) if df is not None else 0
        tf_rows[tf] = rows
        if rows >= 100:
            trusted.append(f"{tf} 기술 지표 표본 충분({rows}캔들)")
        elif rows > 0:
            cautions.append(f"{tf} 기술 지표 표본 부족({rows}캔들)")
        else:
            no_use.append(f"{tf} 캔들 데이터 없음")

    macro = macro_snapshot or {}
    for key in _MACRO_KEYS:
        item = macro.get(key) if isinstance(macro, dict) else None
        if not isinstance(item, dict) or item.get("value") is None:
            no_use.append(f"{key} 값 없음")
            continue
        stale_days = _macro_stale_days(item.get("latest_date"), now_dt)
        samples = item.get("change24h_samples")
        if stale_days is not None and stale_days > 3:
            cautions.append(f"{key} 기준일 {stale_days}일 전")
        if samples is not None and samples < 2:
            cautions.append(f"{key} 24h 변화 표본 부족({samples})")
        trusted.append(f"{key} 값 사용 가능")

    market = market_ctx or {}
    for key in _MARKET_KEYS:
        if market.get(key) is None:
            no_use.append(f"{key} 시장심리 데이터 없음")
        else:
            trusted.append(f"{key} 시장심리 사용 가능")

    account = account_ctx or {}
    for key in _ACCOUNT_KEYS:
        if account.get(key) is None:
            cautions.append(f"{key} 계좌 정보 없음")

    quality_score = 100
    quality_score -= min(45, len(no_use) * 5)
    quality_score -= min(25, len(cautions) * 3)
    quality_score = max(0, quality_score)

    if quality_score >= 80:
        grade = "high"
    elif quality_score >= 55:
        grade = "medium"
    else:
        grade = "low"

    return {
        "grade": grade,
        "score": quality_score,
        "trusted": trusted[:12],
        "cautions": cautions[:12],
        "no_use": no_use[:12],
        "tf_rows": tf_rows,
    }


def build_data_auditor_report(quality: dict, derived: dict) -> dict:
    conflicts = list(derived.get("conflicts") or [])
    warnings = list(derived.get("warnings") or [])
    cautions = list(quality.get("cautions") or [])
    no_use = list(quality.get("no_use") or [])

    auditor_warnings = []
    auditor_warnings.extend(conflicts[:3])
    auditor_warnings.extend(cautions[:4])
    auditor_warnings.extend(no_use[:4])
    auditor_warnings.extend(warnings[:2])

    use_first = list(quality.get("trusted") or [])[:6]
    avoid = no_use[:6]

    return {
        "summary": f"데이터 품질 {quality.get('grade')}({quality.get('score')}/100). 표본 부족·결측 항목은 확신도에서 감점.",
        "use_first": use_first,
        "warnings": auditor_warnings,
        "do_not_overweight": avoid,
    }


def format_quality_block(quality: dict) -> str:
    lines = [
        "<data_quality>",
        f"grade: {quality.get('grade')} ({quality.get('score')}/100)",
    ]
    if quality.get("trusted"):
        lines.append("trusted:")
        lines.extend(f"- {item}" for item in quality["trusted"])
    if quality.get("cautions"):
        lines.append("cautions:")
        lines.extend(f"- {item}" for item in quality["cautions"])
    if quality.get("no_use"):
        lines.append("do_not_use_as_evidence:")
        lines.extend(f"- {item}" for item in quality["no_use"])
    lines.append("</data_quality>")
    return "\n".join(lines)


def format_derived_block(derived: dict) -> str:
    lines = [
        "<derived_features>",
        f"higher_tf_bias: {derived.get('higher_tf_bias')}",
        f"lower_tf_bias: {derived.get('lower_tf_bias')}",
    ]
    conflicts = derived.get("conflicts") or []
    if conflicts:
        lines.append("signal_conflicts:")
        lines.extend(f"- {item}" for item in conflicts)
    for f in derived.get("timeframes") or []:
        lines.append(
            "- {tf}: trend={trend} score={score} structure={structure} "
            "rsi={rsi}({rsi_zone}) macd_delta={macd_delta} "
            "bb_pct={bb_pct} vol/ma={vol_ratio} atr_pct={atr_pct} "
            "sma200_gap={sma_gap}%".format(
                tf=f.get("tf"),
                trend=f.get("trend_bias"),
                score=f.get("trend_score"),
                structure=f.get("structure"),
                rsi=_fmt_num(f.get("rsi"), 1),
                rsi_zone=f.get("rsi_zone"),
                macd_delta=_fmt_num(f.get("macd_delta"), 2),
                bb_pct=_fmt_num(f.get("bb_pct"), 3),
                vol_ratio=_fmt_num(f.get("volume_ratio"), 2),
                atr_pct=_fmt_num(f.get("atr_pct"), 2),
                sma_gap=_fmt_num(f.get("distance_sma200"), 2),
            )
        )
        for warning in f.get("warnings") or []:
            lines.append(f"  warning: {warning}")
    lines.append("</derived_features>")
    return "\n".join(lines)


def format_auditor_block(auditor: dict) -> str:
    lines = ["<data_auditor>", f"summary: {auditor.get('summary')}"]
    if auditor.get("use_first"):
        lines.append("use_first:")
        lines.extend(f"- {item}" for item in auditor["use_first"])
    if auditor.get("warnings"):
        lines.append("warnings:")
        lines.extend(f"- {item}" for item in auditor["warnings"])
    if auditor.get("do_not_overweight"):
        lines.append("do_not_overweight:")
        lines.extend(f"- {item}" for item in auditor["do_not_overweight"])
    lines.append("</data_auditor>")
    return "\n".join(lines)


def build_analysis_context(
    multi_tf_data: dict,
    macro_snapshot: dict | None,
    market_ctx: dict | None,
    account_ctx: dict | None,
) -> dict:
    derived = build_derived_features(multi_tf_data)
    quality = build_data_quality_report(multi_tf_data, macro_snapshot, market_ctx, account_ctx)
    auditor = build_data_auditor_report(quality, derived)
    text = "\n\n".join((
        format_quality_block(quality),
        format_auditor_block(auditor),
        format_derived_block(derived),
    ))
    return {
        "text": text,
        "quality": quality,
        "auditor": auditor,
        "derived": derived,
    }
