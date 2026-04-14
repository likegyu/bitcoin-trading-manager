# =============================================
# Bull ↔ Bear Debate Runner
# =============================================
# 동기 함수로 작성 (server.py 가 이미 ThreadPoolExecutor 로 블로킹 호출을 감싸는
# 구조이므로, 동기 구현이 가장 단순하고 에러 경로 관리가 쉬움).
# =============================================
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

import anthropic

from config import CLAUDE_API_KEY, CLAUDE_MODEL
from .prompts import (
    BULL_SYSTEM,
    BEAR_SYSTEM,
    BULL_USER_TEMPLATE,
    BEAR_USER_TEMPLATE,
    opponent_block,
)


# ── 설정 ──────────────────────────────────────────
# 토론 에이전트는 "quick" 모델 사용 (비용/지연 최적화).
# 기본값: Haiku 4.5. 필요 시 env 로 오버라이드.
DEBATE_MODEL = os.getenv("DEBATE_MODEL", "claude-haiku-4-5")

# 한 라운드 = Bull 1회 + Bear 1회.
# max_rounds=1 → 총 2회 LLM 호출 (가장 가벼운 조합).
# max_rounds=2 → Bull→Bear→Bull반박→Bear반박, 총 4회.
DEBATE_MAX_ROUNDS = int(os.getenv("DEBATE_MAX_ROUNDS", "1"))

# 토론 자체를 끄고 싶을 때: DEBATE_ENABLED=0
DEBATE_ENABLED = os.getenv("DEBATE_ENABLED", "1") not in ("0", "false", "False", "")


@dataclass
class DebateTurn:
    """한 에이전트가 한 번 발언한 기록."""
    side: str          # "bull" | "bear"
    round_index: int   # 0부터 시작
    content: str
    model: str
    elapsed_s: float


@dataclass
class DebateResult:
    """토론 전체 결과."""
    enabled: bool
    rounds: int
    turns: list[DebateTurn] = field(default_factory=list)
    final_bull: str = ""
    final_bear: str = ""
    error: Optional[str] = None

    def to_payload(self) -> dict:
        """SSE/JSON 직렬화용. DebateTurn 도 dict 로 변환."""
        return {
            "enabled": self.enabled,
            "rounds": self.rounds,
            "turns": [asdict(t) for t in self.turns],
            "final_bull": self.final_bull,
            "final_bear": self.final_bear,
            "error": self.error,
        }


ProgressCallback = Callable[[str, str], None]
# progress_cb(phase, detail) — 예: ("bull_round_1", "Bull 라운드 1/1 시작")


def _call_llm(client: anthropic.Anthropic, system: str, user: str) -> str:
    """단일 에이전트 호출. 429/529 백오프 포함."""
    max_retries = 3
    wait = 8
    for attempt in range(max_retries):
        try:
            msg = client.messages.create(
                model=DEBATE_MODEL,
                max_tokens=1500,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            # Haiku 4.5 는 thinking 미지원 → 그대로 text 블록만.
            text = next((b.text for b in msg.content if b.type == "text"), "")
            return text.strip()
        except anthropic.APIStatusError as e:
            if e.status_code in (429, 529) and attempt < max_retries - 1:
                time.sleep(wait)
                wait *= 2
                continue
            raise


def run_bull_bear_debate(
    context_blob: str,
    pair_label: str,
    max_rounds: Optional[int] = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> DebateResult:
    """
    Bull ↔ Bear 사전 토론을 실행한다.

    Parameters
    ----------
    context_blob : str
        Bull/Bear 가 공통으로 보는 데이터 블록 (analyzer._build_context_blob 의 출력).
    pair_label : str
        "BTC/USDC" 같은 표시용 심볼.
    max_rounds : int, optional
        None 이면 env DEBATE_MAX_ROUNDS 를 사용.
    progress_cb : callable, optional
        (phase, detail) 형태 콜백. SSE 진행률 표시에 사용.

    Returns
    -------
    DebateResult
    """
    rounds = max_rounds if max_rounds is not None else DEBATE_MAX_ROUNDS

    # 토론 비활성 또는 키 없음 → 빈 결과 반환 (analyzer 는 정상 진행)
    if not DEBATE_ENABLED:
        return DebateResult(enabled=False, rounds=0)
    if not CLAUDE_API_KEY:
        return DebateResult(enabled=False, rounds=0, error="CLAUDE_API_KEY 미설정")

    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    result = DebateResult(enabled=True, rounds=rounds)

    last_bull: str = ""
    last_bear: str = ""

    try:
        for r in range(rounds):
            # ── Bull 발언 ───────────────────────
            if progress_cb:
                progress_cb("bull", f"Bull 라운드 {r + 1}/{rounds} 분석 중")

            bull_user = BULL_USER_TEMPLATE.format(
                pair_label=pair_label,
                context_blob=context_blob,
                opponent_block=opponent_block("bull", last_bear),
                rebuttal_instruction=(
                    "Bear 의 반박 포인트를 하나씩 짚어 재반박하세요."
                    if last_bear
                    else "Bear 가 제기할 가장 강한 반박을 선제적으로 무력화하세요."
                ),
            )
            t0 = time.time()
            bull_reply = _call_llm(client, BULL_SYSTEM, bull_user)
            elapsed = time.time() - t0
            result.turns.append(DebateTurn(
                side="bull",
                round_index=r,
                content=bull_reply,
                model=DEBATE_MODEL,
                elapsed_s=round(elapsed, 2),
            ))
            last_bull = bull_reply

            # ── Bear 발언 ───────────────────────
            if progress_cb:
                progress_cb("bear", f"Bear 라운드 {r + 1}/{rounds} 반박 중")

            bear_user = BEAR_USER_TEMPLATE.format(
                pair_label=pair_label,
                context_blob=context_blob,
                opponent_block=opponent_block("bear", last_bull),
                rebuttal_instruction=(
                    "Bull 의 근거 하나하나를 구체적으로 반박하세요."
                ),
            )
            t0 = time.time()
            bear_reply = _call_llm(client, BEAR_SYSTEM, bear_user)
            elapsed = time.time() - t0
            result.turns.append(DebateTurn(
                side="bear",
                round_index=r,
                content=bear_reply,
                model=DEBATE_MODEL,
                elapsed_s=round(elapsed, 2),
            ))
            last_bear = bear_reply

        result.final_bull = last_bull
        result.final_bear = last_bear
        return result

    except Exception as exc:
        # 토론이 실패해도 기존 단일 콜 분석은 계속 가능해야 한다.
        result.error = f"{type(exc).__name__}: {exc}"
        # 지금까지 진행된 턴은 유지하고, 마지막 발언을 final 로 반영.
        result.final_bull = last_bull
        result.final_bear = last_bear
        return result


def format_debate_block(result: DebateResult) -> str:
    """
    토론 결과를 최종 analyze_with_claude() 프롬프트에 주입할 텍스트 블록으로 변환.
    토론이 비활성/실패면 빈 문자열 반환 → 기존 프롬프트 그대로 동작.
    """
    if not result.enabled:
        return ""
    if result.error and not result.turns:
        return f"[사전 토론]\n  수행 실패 — {result.error}"
    if not result.turns:
        return ""

    lines = ["[사전 토론 — Bull vs Bear]"]
    lines.append(
        "  ⚠️ 아래는 두 리서처의 주장이 서로 반대임을 전제로 작성된 것입니다. "
        "둘 다 데이터에 근거하되 방향은 편향되어 있습니다. 당신(최종 애널리스트)은 "
        "양측의 주장 강도를 비교·판정하고 정합성이 더 높은 쪽을 선택하세요.")

    for t in result.turns:
        header = f"\n[라운드 {t.round_index + 1} · {'🐂 Bull' if t.side == 'bull' else '🐻 Bear'}]"
        lines.append(header)
        # 각 라인 앞에 들여쓰기 2칸 — 기존 build_prompt 의 데이터 블록과 시각적 일관성.
        for raw in t.content.splitlines():
            lines.append(f"  {raw}" if raw else "")

    if result.error:
        lines.append(f"\n  ⚠️ 토론 일부 실패 — {result.error}")

    return "\n".join(lines)
