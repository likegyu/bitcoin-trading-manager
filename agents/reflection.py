# =============================================
# Reflection Agent — 사후 결과로 메모리 업데이트
# =============================================
# 원본: TradingAgents/tradingagents/agents/utils/agent_utils.py (Reflector 부분)
# 적용:
#   - "이번 판단 이후 실제 가격이 어떻게 움직였나" 를 간단한 변화율로 계측
#   - Claude 가 '무엇을 잘했고/놓쳤고/다음에는 어떻게' 를 1~2문단 리플렉션으로 작성
#   - FinancialSituationMemory.add_situation(...) 에 outcome 으로 기록
#
# 호출 방식:
#   - server.py 에 /reflect 엔드포인트를 추가하거나
#   - schedule 스킬로 매 N시간 자동 실행
#   - 혹은 분석 후 일정 시간이 지난 시점에 수동 트리거
# =============================================
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import anthropic

from config import CLAUDE_API_KEY
from .memory import FinancialSituationMemory, get_memory


REFLECTION_MODEL = os.getenv("REFLECTION_MODEL", "claude-haiku-4-5")

REFLECTION_SYSTEM = """당신은 BTC 선물 애널리스트의 'Reflection Coach' 입니다.
지난 판단과 그 이후 실제 시장 움직임을 대조해 교훈을 뽑아내는 역할입니다.

원칙:
1. 결과론적 비난 금지. 그때 가용했던 정보 기준으로 '놓친 단서'와 '과대평가한 근거'를 각각 짚으세요.
2. '무엇이 맞았는가' 와 '무엇이 틀렸는가' 를 분리해서 서술. 방향성만 맞았다고 성공으로 포장하지 말 것.
3. 다음번 유사 구조에서 적용할 수 있는 '구체적 체크리스트 1~2줄' 로 끝내세요.

출력 규칙:
- 마크다운(**,##,---), HTML 금지. 일반 텍스트 + 최소 이모지.
- 300~500자. 장황함 금지.
"""

REFLECTION_USER_TEMPLATE = """[과거 판단 시점] {past_ts}
[판단 당시 상황 요약]
{past_situation}

[판단 당시 조언]
{past_advice}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[사후 가격 변화]
기준가: ${price_then:,.2f}
현재가: ${price_now:,.2f}
변화율: {pct_change:+.2f}% ({direction})
경과 시간: {elapsed_label}

위 정보를 바탕으로 리플렉션을 작성하세요. 마지막 줄은 반드시
'다음 체크리스트:' 로 시작하는 1~2줄 요약으로 마치세요.
"""


@dataclass
class ReflectionResult:
    """개별 기록 하나에 대한 리플렉션 결과."""
    timestamp: str          # 대상 MemoryRecord 의 timestamp
    price_then: float
    price_now: float
    pct_change: float
    reflection_text: str
    updated: bool           # 메모리에 outcome 업데이트 성공 여부
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "price_then": self.price_then,
            "price_now": self.price_now,
            "pct_change": round(self.pct_change, 4),
            "reflection_text": self.reflection_text,
            "updated": self.updated,
            "error": self.error,
        }


def _call_llm(client: anthropic.Anthropic, system: str, user: str) -> str:
    max_retries = 3
    wait = 8
    for attempt in range(max_retries):
        try:
            msg = client.messages.create(
                model=REFLECTION_MODEL,
                max_tokens=1200,
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


def _elapsed_label(seconds: float) -> str:
    if seconds < 3600:
        return f"{seconds / 60:.0f}분"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}시간"
    return f"{seconds / 86400:.1f}일"


def reflect_on_record(
    record_ts: str,
    situation: str,
    advice: str,
    price_then: float,
    price_now: float,
    elapsed_seconds: float,
    memory: Optional[FinancialSituationMemory] = None,
    memory_name: str = "analyst",
) -> ReflectionResult:
    """
    단일 과거 기록에 대해 리플렉션을 실행하고 메모리의 outcome 필드를 업데이트한다.
    """
    if memory is None:
        memory = get_memory(memory_name)

    if not CLAUDE_API_KEY:
        return ReflectionResult(
            timestamp=record_ts,
            price_then=price_then,
            price_now=price_now,
            pct_change=0.0,
            reflection_text="",
            updated=False,
            error="CLAUDE_API_KEY 미설정",
        )

    # 가격 변화 계산
    pct = 0.0
    if price_then and price_then != 0:
        pct = (price_now - price_then) / price_then * 100.0
    direction = "상승" if pct > 0 else ("하락" if pct < 0 else "정체")

    prompt = REFLECTION_USER_TEMPLATE.format(
        past_ts=record_ts,
        past_situation=situation.strip() or "(기록 없음)",
        past_advice=advice.strip() or "(기록 없음)",
        price_then=price_then,
        price_now=price_now,
        pct_change=pct,
        direction=direction,
        elapsed_label=_elapsed_label(elapsed_seconds),
    )

    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    try:
        reflection_text = _call_llm(client, REFLECTION_SYSTEM, prompt)
    except Exception as exc:
        return ReflectionResult(
            timestamp=record_ts,
            price_then=price_then,
            price_now=price_now,
            pct_change=pct,
            reflection_text="",
            updated=False,
            error=f"{type(exc).__name__}: {exc}",
        )

    # 메모리에 outcome 기록 (가격 변화 요약 + 리플렉션 텍스트)
    outcome_block = (
        f"[사후 {_elapsed_label(elapsed_seconds)}] "
        f"${price_then:,.2f} → ${price_now:,.2f} ({pct:+.2f}%, {direction})\n"
        f"{reflection_text}"
    )
    updated = memory.update_outcome(record_ts, outcome_block)

    return ReflectionResult(
        timestamp=record_ts,
        price_then=price_then,
        price_now=price_now,
        pct_change=pct,
        reflection_text=reflection_text,
        updated=updated,
    )
