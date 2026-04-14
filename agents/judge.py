# =============================================
# Investment Judge — Bull/Bear 토론 중재 및 방향성 결론
# =============================================
# TradingAgents 의 invest_judge 패턴을 BTC 선물 맥락으로 구현.
#
# 역할:
#   - Bull/Bear 의 최종 발언을 받아 논리 강도를 평가
#   - 명확한 방향성 결론(상방/하방/중립)과 이유를 한 문단으로 정리
#   - 결과는 Risk Triad 에 "심판 결론" 으로 함께 주입됨
#   - judge 전용 메모리를 가져 과거 중재 패턴을 학습
# =============================================
from __future__ import annotations

import os
import time
from dataclasses import dataclass, asdict
from typing import Callable, Optional

import anthropic

from config import CLAUDE_API_KEY

try:
    from .memory import AgentMemories
except Exception:
    AgentMemories = None  # type: ignore


JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-sonnet-4-6")
JUDGE_ENABLED = os.getenv("JUDGE_ENABLED", "1") not in ("0", "false", "False", "")

JUDGE_SYSTEM = """당신은 BTC 선물 시장의 'Investment Judge(투자 심판)'입니다.
역할: Bull Researcher 와 Bear Researcher 의 토론을 공정하게 듣고,
어느 쪽의 논리가 현재 데이터에 더 잘 부합하는지 판정한 뒤 명확한 방향성 결론을 내립니다.

판정 원칙:
1. 편향 없이 두 주장의 근거 강도를 비교하세요. 주장의 분량이 아니라 데이터와의 정합성으로 평가.
2. 한쪽이 명백히 우세하면 그 방향을 선택. 데이터 근거가 비슷하면 '중립'을 선택.
3. '중립'은 진짜 불확실한 경우에만 — 애매모호함 회피용으로 쓰지 마세요.
4. 판정 이유를 구체적 데이터 근거(가격 구조, 파생심리, 거시)로 2~3줄 서술.
5. 다음 단계(Risk Triad)가 이 결론을 토대로 리스크 규모를 논의할 수 있도록
   'Bull 의 핵심 근거 한 줄 / Bear 의 핵심 근거 한 줄' 을 마지막에 요약.

출력 형식 (반드시 준수):
판정: [상방 우위 / 하방 우위 / 중립]
이유: [2~3줄 구체적 근거]
Bull 핵심: [한 줄]
Bear 핵심: [한 줄]

마크다운(**, ##, ---), HTML 금지. 일반 텍스트만."""

JUDGE_USER_TEMPLATE = """{pair_label} 현재 데이터 및 Bull/Bear 토론 결과입니다.

{context_blob}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Bull Researcher 최종 발언]
{bull_final}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Bear Researcher 최종 발언]
{bear_final}
{past_memories_block}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
위 데이터와 토론을 바탕으로 공정하게 판정하세요."""


ProgressCallback = Callable[[str, str], None]


@dataclass
class JudgeResult:
    """투자 심판 결과."""
    enabled: bool
    verdict: str        # "상방 우위" | "하방 우위" | "중립"
    reasoning: str      # 판정 이유 (2~3줄)
    bull_key: str       # Bull 핵심 근거 한 줄
    bear_key: str       # Bear 핵심 근거 한 줄
    raw_text: str       # LLM 원본 출력
    model: str = ""
    elapsed_s: float = 0.0
    error: Optional[str] = None

    def to_payload(self) -> dict:
        return asdict(self)


def _parse_judge_output(text: str) -> dict:
    """Judge LLM 출력에서 구조화된 필드 추출."""
    lines = text.strip().splitlines()
    result = {"verdict": "", "reasoning": "", "bull_key": "", "bear_key": ""}
    reasoning_lines = []
    in_reasoning = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("판정:"):
            result["verdict"] = stripped[len("판정:"):].strip()
            in_reasoning = False
        elif stripped.startswith("이유:"):
            val = stripped[len("이유:"):].strip()
            if val:
                reasoning_lines.append(val)
            in_reasoning = True
        elif stripped.startswith("Bull 핵심:"):
            result["bull_key"] = stripped[len("Bull 핵심:"):].strip()
            in_reasoning = False
        elif stripped.startswith("Bear 핵심:"):
            result["bear_key"] = stripped[len("Bear 핵심:"):].strip()
            in_reasoning = False
        elif in_reasoning and stripped:
            reasoning_lines.append(stripped)

    result["reasoning"] = " ".join(reasoning_lines)
    return result


def _call_llm(client: anthropic.Anthropic, system: str, user: str) -> str:
    max_retries = 3
    wait = 8
    for attempt in range(max_retries):
        try:
            msg = client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=1500,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return next((b.text for b in msg.content if b.type == "text"), "").strip()
        except anthropic.APIStatusError as e:
            if e.status_code in (429, 529) and attempt < max_retries - 1:
                time.sleep(wait)
                wait *= 2
                continue
            raise


def run_judge(
    context_blob: str,
    pair_label: str,
    bull_final: str,
    bear_final: str,
    agent_memories: Optional["AgentMemories"] = None,
    memory_query: str = "",
    progress_cb: Optional[ProgressCallback] = None,
) -> JudgeResult:
    """
    Bull/Bear 토론 결과를 받아 방향성을 판정한다.

    Parameters
    ----------
    context_blob : str
        공통 시장 데이터 블록.
    pair_label : str
        "BTC/USDC" 등.
    bull_final, bear_final : str
        Bull/Bear 최종 발언.
    agent_memories : AgentMemories, optional
        judge 전용 메모리 — 과거 중재 패턴 회상.
    memory_query : str
        BM25 쿼리용 상황 요약 문자열.
    progress_cb : callable, optional
        SSE 진행률 콜백.
    """
    if not JUDGE_ENABLED:
        return JudgeResult(enabled=False, verdict="", reasoning="", bull_key="", bear_key="", raw_text="")
    if not CLAUDE_API_KEY:
        return JudgeResult(enabled=False, verdict="", reasoning="", bull_key="", bear_key="", raw_text="",
                           error="CLAUDE_API_KEY 미설정")
    if not bull_final and not bear_final:
        return JudgeResult(enabled=False, verdict="", reasoning="", bull_key="", bear_key="", raw_text="",
                           error="Bull/Bear 발언 없음 — 토론 미수행")

    if progress_cb:
        progress_cb("judge", "투자 심판 중재 중")

    # judge 메모리 회상
    past = ""
    if agent_memories is not None:
        _query = memory_query or context_blob[:200]
        past = agent_memories.recall("judge", _query, top_k=2)

    user_prompt = JUDGE_USER_TEMPLATE.format(
        pair_label=pair_label,
        context_blob=context_blob,
        bull_final=bull_final or "(Bull 발언 없음)",
        bear_final=bear_final or "(Bear 발언 없음)",
        past_memories_block=(
            f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n{past}\n"
            if past else ""
        ),
    )

    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    t0 = time.time()
    try:
        raw = _call_llm(client, JUDGE_SYSTEM, user_prompt)
    except Exception as exc:
        return JudgeResult(
            enabled=True, verdict="중립", reasoning="", bull_key="", bear_key="",
            raw_text="", error=f"{type(exc).__name__}: {exc}",
        )
    elapsed = time.time() - t0

    parsed = _parse_judge_output(raw)
    return JudgeResult(
        enabled=True,
        verdict=parsed["verdict"] or "중립",
        reasoning=parsed["reasoning"],
        bull_key=parsed["bull_key"],
        bear_key=parsed["bear_key"],
        raw_text=raw,
        model=JUDGE_MODEL,
        elapsed_s=round(elapsed, 2),
    )


def format_judge_block(judge: Optional[JudgeResult]) -> str:
    """
    Judge 결과를 최종 프롬프트 주입용 블록으로 변환.
    비활성/실패면 빈 문자열.
    """
    if judge is None or not judge.enabled:
        return ""
    if judge.error and not judge.verdict:
        return f"[투자 심판]\n  수행 실패 — {judge.error}"

    lines = ["[투자 심판 결론]"]
    lines.append(f"  판정: {judge.verdict}")
    if judge.reasoning:
        lines.append(f"  이유: {judge.reasoning}")
    if judge.bull_key:
        lines.append(f"  Bull 핵심: {judge.bull_key}")
    if judge.bear_key:
        lines.append(f"  Bear 핵심: {judge.bear_key}")
    if judge.error:
        lines.append(f"  ⚠️ 일부 실패 — {judge.error}")
    return "\n".join(lines)
