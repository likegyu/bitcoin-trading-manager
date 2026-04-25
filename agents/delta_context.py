# =============================================
# Delta Context — 직전 분석 대비 변화 블록
# =============================================
# 목적:
#   매 호출이 정적 스냅샷이라 "RSI 70이 방금 도달인지 2시간째 머무는지"를
#   LLM이 시계열 비교로 직접 추정해야 했음. 토큰 효율 떨어지고 휴리스틱이 일관되지 않음.
#
# 해결:
#   analyst 메모리의 가장 최근 기록(situation_tags + meta)에서 핵심값을 뽑아
#   현재값과의 차이를 한 블록으로 미리 계산해 LLM 입력에 주입.
#
# 철학:
#   - 절대 수치보다 '변화 자체'와 '임계 돌파 여부' 가 단기 분석에 결정적
#   - 직전 기록이 없으면 빈 문자열 — 첫 호출은 그냥 패스
#   - 너무 오래된 기록(예: 6시간 이상 전)은 비교 의미 없음 — stale 라벨 부여
# =============================================
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

# 직전 기록 유효 기간 — 이보다 오래되면 비교 의미 약함
DELTA_FRESH_WINDOW_SEC = 90 * 60   # 90분
DELTA_STALE_WINDOW_SEC = 6 * 3600  # 6시간 — 이걸 넘으면 비교 자체를 포기


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def _last_close(multi_tf_data: dict, tf_priority=("5m", "15m", "1h", "4h", "1d")) -> Optional[float]:
    for tf in tf_priority:
        df = multi_tf_data.get(tf) if multi_tf_data else None
        if df is None or len(df) == 0:
            continue
        try:
            return float(df.iloc[-1]["close"])
        except Exception:
            continue
    return None


def _tag_dict_from_string(tag_str: str) -> dict:
    """'rsi_1h:과매수 | macd_1h:상방약화 | ...' → {'rsi_1h': '과매수', ...}"""
    if not tag_str:
        return {}
    out: dict[str, str] = {}
    for chunk in tag_str.split("|"):
        chunk = chunk.strip()
        if not chunk or ":" not in chunk:
            continue
        k, v = chunk.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def _format_pct_change(prev: float, curr: float) -> str:
    if prev == 0:
        return "n/a"
    pct = (curr - prev) / prev * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.2f}%"


def _abs_change(prev: float, curr: float, digits: int = 2) -> str:
    diff = curr - prev
    sign = "+" if diff >= 0 else ""
    fmt = f"{{:{sign}.{digits}f}}"
    return fmt.format(diff)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


# ── 핵심 빌더 ────────────────────────────────────────
def build_delta_block(
    multi_tf_data: dict,
    market_ctx: Optional[dict],
    macro_snapshot: Optional[dict],
    current_situation_tags: str,
    memory_obj: Any,  # FinancialSituationMemory or None
) -> str:
    """
    직전 analyst 기록과 비교해 '무엇이 변했는가' 한 블록을 만든다.

    반환:
      - 직전 기록 없거나 stale 이면 빈 문자열
      - 정상이면 [직전 대비 변화 — N분 전] 블록 텍스트
    """
    if memory_obj is None:
        return ""

    try:
        records = memory_obj.list_records() if hasattr(memory_obj, "list_records") else []
    except Exception:
        return ""

    if not records:
        return ""

    prev = records[-1]
    prev_ts = _parse_iso(getattr(prev, "timestamp", "") or "")
    if prev_ts is None:
        return ""

    elapsed = (_now_utc() - prev_ts).total_seconds()
    if elapsed > DELTA_STALE_WINDOW_SEC:
        # 너무 오래된 기록 — 비교 무의미
        return ""

    elapsed_min = int(elapsed / 60)
    is_stale = elapsed > DELTA_FRESH_WINDOW_SEC

    prev_meta = getattr(prev, "meta", {}) or {}
    prev_price = _to_float(prev_meta.get("price_at_analysis"))
    prev_signal = prev_meta.get("signal")
    prev_confidence = prev_meta.get("confidence")
    prev_view = None
    aj = prev_meta.get("analysis_json")
    if isinstance(aj, dict):
        prev_view = aj.get("view") or None
        prev_regime = aj.get("regime")
    else:
        prev_regime = None

    prev_tags_str = prev_meta.get("situation_tags") or ""
    prev_tags = _tag_dict_from_string(prev_tags_str)
    curr_tags = _tag_dict_from_string(current_situation_tags or "")

    # ── 가격 변화 ───────────────────────────────
    curr_price = _last_close(multi_tf_data)
    price_line = ""
    if prev_price and curr_price:
        price_line = (
            f"가격: ${prev_price:,.0f} → ${curr_price:,.0f} "
            f"({_format_pct_change(prev_price, curr_price)})"
        )

    # ── 카테고리 태그 변화 ─────────────────────
    # 우선순위 순서로 표시 — 1h/4h 추세, RSI, MACD, 펀딩, OI, 스큐, 계좌
    priority_keys = (
        "trend_4h", "trend_1h", "trend_15m",
        "ma_4h", "ma_1h",
        "rsi_4h", "rsi_1h", "rsi_15m",
        "macd_4h", "macd_1h", "macd_15m",
        "funding", "oi_24h", "skew",
        "account",
    )
    diff_lines: list[str] = []
    for k in priority_keys:
        prev_v = prev_tags.get(k)
        curr_v = curr_tags.get(k)
        if not curr_v and not prev_v:
            continue
        if prev_v == curr_v:
            continue
        if prev_v is None:
            diff_lines.append(f"  {k}: (직전 미기록) → {curr_v}")
        elif curr_v is None:
            diff_lines.append(f"  {k}: {prev_v} → (현재 미기록)")
        else:
            diff_lines.append(f"  {k}: {prev_v} → {curr_v}")

    # ── 거시 변화 (수치는 안 보고 regime 만) ───
    if macro_snapshot and isinstance(macro_snapshot, dict):
        for key in ("DXY", "TNX_10Y", "STABLE_MCAP"):
            entry = macro_snapshot.get(key)
            if not isinstance(entry, dict):
                continue
            curr_regime = entry.get("regime")
            prev_macro = prev_tags.get(f"macro_{key.lower()}")
            if curr_regime and prev_macro and prev_macro != curr_regime:
                diff_lines.append(f"  macro_{key.lower()}: {prev_macro} → {curr_regime}")

    # ── 직전 view/관점 변화 ─────────────────────
    view_line = ""
    if prev_view:
        view_line = f"직전 view: {prev_view}"
        if prev_confidence is not None:
            view_line += f" (확신도 {prev_confidence})"
        if prev_regime:
            view_line += f" · regime: {prev_regime}"

    # ── 블록 어셈블 ────────────────────────────
    if not (price_line or diff_lines or view_line):
        # 기록은 있으나 모든 핵심 필드가 비어있음 → 출력할 게 없음
        return ""

    header = f"[직전 대비 변화 — {elapsed_min}분 전 분석 기준]"
    if is_stale:
        header += "  ⚠️ stale (60분+ 경과 — 변화 정도 약함)"

    lines = [header]
    if price_line:
        lines.append(f"  {price_line}")
    if view_line:
        lines.append(f"  {view_line}")
    if diff_lines:
        lines.append("  카테고리 태그 변화:")
        lines.extend(diff_lines)
    else:
        lines.append("  카테고리 태그 변화: (구조적 변화 없음 — 같은 상태 지속)")

    lines.append(
        "  ※ 이 변화 자체를 새 신호로 해석할지 잡음으로 무시할지는 가격 구조와 함께 판단."
    )
    return "\n".join(lines)
