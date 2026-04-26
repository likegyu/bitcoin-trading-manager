# =============================================
# Consistency Check — 분석 결과 자체 검증
# =============================================
# 목적:
#   <analysis_json> 의 view·confidence·trade(entry/stop/target) 와
#   본문 한국어 리포트가 모순되는 경우를 잡아낸다.
#   (예: view=상방 우위인데 trade.stop > trade.entry, JSON 신호와 본문 결론이 정반대)
#
# 검증 두 단계:
#   1) 결정적 검증 (deterministic) — JSON 자체의 부호·범위·관계 점검 (LLM 호출 없음)
#   2) 의미 검증 (semantic) — 짧은 Haiku 콜로 JSON ↔ 본문 일치 확인 (선택)
#
# 결과:
#   {"ok": bool, "issues": [...], "level": "info|warn|error"}
#
# Settings:
#   CONSISTENCY_LLM_ENABLED=0|1   기본 0 (비용 보수적)
#   CONSISTENCY_LLM_MODEL         기본 claude-haiku-4-5-20251001
# =============================================
from __future__ import annotations

import os
import re
import time
from typing import Any, Optional

import anthropic

from config import CLAUDE_API_KEY


CONSISTENCY_LLM_ENABLED = (
    os.getenv("CONSISTENCY_LLM_ENABLED", "0").lower() not in ("0", "false", "no", "")
)
CONSISTENCY_LLM_MODEL = os.getenv(
    "CONSISTENCY_LLM_MODEL", "claude-haiku-4-5-20251001"
)
CONSISTENCY_LLM_MAX_TOKENS = int(os.getenv("CONSISTENCY_LLM_MAX_TOKENS", "300"))
CONFIDENCE_MIN = 1
CONFIDENCE_MAX = 100
CONFIDENCE_BREAKDOWN_BOUNDS = {
    "price_structure": (0, 30),
    "momentum": (0, 20),
    "derivatives": (0, 20),
    "macro": (0, 15),
    "account_risk_fit": (0, 15),
    "data_quality_penalty": (-15, 0),
    "counter_scenario_penalty": (-10, 0),
}


# ── 결정적 검증 ──────────────────────────────────────
def _clamp_confidence(value: int) -> int:
    return max(CONFIDENCE_MIN, min(CONFIDENCE_MAX, int(value)))


def _clamp_int(value: Any, min_value: int, max_value: int) -> Optional[int]:
    try:
        num = int(value)
    except (TypeError, ValueError):
        return None
    return max(min_value, min(max_value, num))


def _check_levels_geometry(view: str, trade: dict) -> list[str]:
    """view 와 trade entry/stop/target 의 부호 관계 검증."""
    issues: list[str] = []
    entry = _f(trade.get("entry"))
    stop = _f(trade.get("stop"))
    target = _f(trade.get("target"))
    if entry is None or stop is None:
        return issues   # 데이터 부족 시 통과 (parse 단계 책임)

    if "상방" in view or view == "매수":
        if not (stop < entry):
            issues.append(f"상방 view 인데 stop({stop}) ≥ entry({entry})")
        if target is not None and not (target > entry):
            issues.append(f"상방 view 인데 target({target}) ≤ entry({entry})")
    elif "하방" in view or view == "매도":
        if not (stop > entry):
            issues.append(f"하방 view 인데 stop({stop}) ≤ entry({entry})")
        if target is not None and not (target < entry):
            issues.append(f"하방 view 인데 target({target}) ≥ entry({entry})")

    # 손익비 sanity (>= 0.5:1) — 너무 작으면 의심
    if entry and stop and target:
        risk = abs(entry - stop)
        reward = abs(target - entry)
        if risk > 0 and reward / risk < 0.5:
            issues.append(f"손익비 {reward/risk:.2f}:1 — 0.5 미만, target 재검토 필요")

    return issues


def _check_confidence_breakdown(analysis_json: dict) -> list[str]:
    """confidence 가 confidence_breakdown 합과 일치하는지 검증."""
    issues: list[str] = []
    cb = analysis_json.get("confidence_breakdown")
    confidence = analysis_json.get("confidence")
    if not isinstance(cb, dict) or confidence is None:
        return issues

    expected_keys = (
        "price_structure", "momentum", "derivatives", "macro",
        "account_risk_fit", "data_quality_penalty", "counter_scenario_penalty",
    )
    normalized_total = 0
    missing = []
    for k in expected_keys:
        v = cb.get(k)
        if v is None:
            missing.append(k)
            continue
        min_value, max_value = CONFIDENCE_BREAKDOWN_BOUNDS[k]
        normalized_value = _clamp_int(v, min_value, max_value)
        if normalized_value is None:
            issues.append(f"confidence_breakdown.{k} 가 정수 아님: {v!r}")
            return issues
        if normalized_value != v:
            issues.append(
                f"confidence_breakdown.{k}={v!r} 가 허용 범위 {min_value}~{max_value} 밖"
            )
        normalized_total += normalized_value
    if missing:
        issues.append(f"confidence_breakdown 누락: {', '.join(missing)}")
        return issues

    try:
        confidence_int = int(confidence)
    except (TypeError, ValueError):
        return issues

    if not (CONFIDENCE_MIN <= confidence_int <= CONFIDENCE_MAX):
        issues.append(f"confidence({confidence_int}) 가 {CONFIDENCE_MIN}~{CONFIDENCE_MAX} 범위 밖")
        return issues

    expected_confidence = _clamp_confidence(normalized_total)
    diff = abs(confidence_int - expected_confidence)
    if diff > 2:   # ±2 점 허용 (반올림 차)
        issues.append(
            f"confidence({confidence_int}) ≠ breakdown 합 clamp({normalized_total} → {expected_confidence})  "
            f"차이 {diff}점 — 모델이 합산을 안 맞춤"
        )
    return issues


def _check_view_signal_alignment(analysis_json: dict, report_text: str) -> list[str]:
    """본문 리포트의 '관점:' 줄이 JSON view 와 일치하는지."""
    issues: list[str] = []
    view = (analysis_json.get("view") or "").strip()
    if not view:
        return issues
    m = re.search(r"📊\s*관점\s*[:：]\s*([^\n]+)", report_text or "")
    if not m:
        return issues
    body_view = m.group(1).strip()
    # 정확 일치 또는 매수/매도 ↔ 상방/하방 매핑
    map_pairs = {"매수": "상방", "매도": "하방"}
    if view != body_view:
        # 부분 정합 허용 — 둘 다 '상방'을 포함하면 OK
        for cls in ("상방", "하방", "중립"):
            if cls in view and cls in body_view:
                return issues
        # 매수/매도 ↔ 상방/하방
        for k, v in map_pairs.items():
            if (k in view and v in body_view) or (v in view and k in body_view):
                return issues
        issues.append(f"JSON view='{view}' ↔ 본문 관점='{body_view}' 불일치")
    return issues


def _check_regime_validity(analysis_json: dict) -> list[str]:
    """regime 이 6개 허용값 중 하나인지."""
    issues: list[str] = []
    regime = (analysis_json.get("regime") or "").strip()
    if not regime:
        return issues
    allowed = ("상승 추세", "하락 추세", "박스", "변동성 확장", "변동성 축소", "이벤트 대기 중")
    if regime not in allowed:
        # 부분 매칭 시도
        if not any(a in regime for a in allowed):
            issues.append(f"regime='{regime}' 이 허용값 6개에 없음")
    return issues


def _f(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", "").replace("$", ""))
    except Exception:
        return None


# ── LLM 의미 검증 (선택) ─────────────────────────────
_LLM_SYSTEM = """당신은 BTC 분석 리포트의 'Consistency Auditor' 입니다.
주어진 JSON 과 본문 리포트가 서로 모순되는지 빠르게 점검합니다.

검사 항목:
1. JSON view 와 본문 '📊 관점' 이 같은 방향인가
2. 본문 '🤖 매매 파라미터' 의 진입/손절/목표가 JSON trade 와 부호·관계가 맞는가
3. JSON regime 과 본문 '🧭 시장 레짐' 이 같은 카테고리인가
4. 본문에 'JSON 과 다른 결론' 으로 읽힐 만한 모순 문장이 있는가

출력 형식 (반드시 준수):
status: ok | warn | error
issues:
- [모순 1줄]
- [모순 1줄]
(모순 없으면 'issues:' 다음 줄에 '없음' 만)

마크다운(**, ##, ---), HTML 금지. 일반 텍스트.
"""

_LLM_USER_TEMPLATE = """[JSON]
{json_repr}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[본문 리포트]
{report_text}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
위 두 출력의 일관성을 점검하세요."""


def _llm_consistency_check(
    analysis_json: dict, report_text: str
) -> Optional[dict]:
    """짧은 Haiku 콜로 JSON ↔ 본문 의미 일치 검증."""
    if not CONSISTENCY_LLM_ENABLED or not CLAUDE_API_KEY:
        return None
    try:
        import json as _json
        json_repr = _json.dumps(analysis_json, ensure_ascii=False, indent=2)[:2000]
    except Exception:
        return None
    report_snippet = (report_text or "")[:2500]
    user = _LLM_USER_TEMPLATE.format(json_repr=json_repr, report_text=report_snippet)
    try:
        client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
        msg = client.messages.create(
            model=CONSISTENCY_LLM_MODEL,
            max_tokens=CONSISTENCY_LLM_MAX_TOKENS,
            system=_LLM_SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
        text = next((b.text for b in msg.content if b.type == "text"), "").strip()
    except Exception as exc:
        return {"status": "error", "issues": [f"LLM 검증 호출 실패 — {exc}"]}

    return _parse_llm_check(text)


def _parse_llm_check(text: str) -> dict:
    status = "ok"
    issues: list[str] = []
    in_issues = False
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"^status\s*[:：]\s*(\w+)", line, re.IGNORECASE)
        if m:
            status = m.group(1).lower()
            continue
        if line.lower().startswith("issues"):
            in_issues = True
            continue
        if in_issues:
            if line in ("없음", "none", "-"):
                continue
            stripped = re.sub(r"^[-•]\s*", "", line).strip()
            if stripped:
                issues.append(stripped)
    if not issues:
        status = "ok" if status not in ("error", "warn") else status
    return {"status": status, "issues": issues}


# ── 외부 API ────────────────────────────────────────
def check_consistency(
    analysis_json: dict,
    report_text: str,
    use_llm: Optional[bool] = None,
) -> dict:
    """
    분석 결과의 자체 일관성을 검증.

    Returns
    -------
    {
      "ok": bool,
      "level": "info" | "warn" | "error",
      "issues": [str, ...],      # 결정적 검증 + (선택) LLM 검증 통합
      "llm": dict | None,        # LLM 검증 결과 원본 (활성화 시)
    }
    """
    issues: list[str] = []
    if not isinstance(analysis_json, dict):
        return {"ok": False, "level": "error", "issues": ["analysis_json 이 dict 아님"], "llm": None}

    trade = analysis_json.get("trade") if isinstance(analysis_json.get("trade"), dict) else {}
    view = (analysis_json.get("view") or "").strip()

    issues.extend(_check_levels_geometry(view, trade))
    issues.extend(_check_confidence_breakdown(analysis_json))
    issues.extend(_check_view_signal_alignment(analysis_json, report_text))
    issues.extend(_check_regime_validity(analysis_json))

    # LLM 검증 (선택)
    llm_result = None
    if (use_llm if use_llm is not None else CONSISTENCY_LLM_ENABLED):
        llm_result = _llm_consistency_check(analysis_json, report_text)
        if isinstance(llm_result, dict):
            for it in llm_result.get("issues") or []:
                issues.append(f"[LLM] {it}")

    if not issues:
        level = "info"
    else:
        # geometry 위반은 error, 나머지는 warn
        level = "error" if any("stop" in i or "target" in i for i in issues) else "warn"

    return {
        "ok": not issues,
        "level": level,
        "issues": issues,
        "llm": llm_result,
    }


def format_consistency_block(check: dict) -> str:
    """결과를 사람이 읽을 수 있는 한 블록으로."""
    if not isinstance(check, dict):
        return ""
    if check.get("ok"):
        return "[Consistency Check] ✅ JSON ↔ 본문 일관성 OK"
    level = check.get("level", "warn")
    icon = "❌" if level == "error" else "⚠️"
    lines = [f"[Consistency Check] {icon} 일관성 문제 감지 (level={level})"]
    for i in (check.get("issues") or []):
        lines.append(f"  - {i}")
    return "\n".join(lines)
