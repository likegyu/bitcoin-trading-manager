# =============================================
# Multi-Agent Orchestrator (Phase 2)
# =============================================
# 기능:
#   Bull/Bear Debate → Risk Triad → (선택) Memory 회상 →
#   최종 analyze_with_claude() 에 주입할 텍스트 블록 조립까지 담당.
#
# analyzer.run_full_analysis() 는 이 모듈의 run_pipeline() 을 호출해
# debate_block 을 얻고, 그걸 build_prompt 에 넘긴다.
# =============================================
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Optional

from .debate import (
    run_bull_bear_debate,
    format_debate_block,
    DebateResult,
)
from .risk_triad import (
    run_risk_triad,
    format_risk_block,
    RiskTriadResult,
)

# Phase 3 메모리 — 임포트는 lazy 하게 (BM25 의존성 미설치시에도 동작하도록)
try:
    from .memory import FinancialSituationMemory, format_memory_block
    _MEMORY_AVAILABLE = True
except Exception:
    FinancialSituationMemory = None  # type: ignore
    format_memory_block = None       # type: ignore
    _MEMORY_AVAILABLE = False


ProgressCallback = Callable[[str, str], None]


# 파이프라인 단계 ON/OFF (환경변수로 제어)
RISK_TRIAD_IN_PIPELINE = (
    os.getenv("RISK_ENABLED", "1") not in ("0", "false", "False", "")
)
MEMORY_IN_PIPELINE = (
    os.getenv("MEMORY_ENABLED", "1") not in ("0", "false", "False", "")
)


@dataclass
class PipelineResult:
    """run_pipeline 의 결과 — analyzer 가 최종 빌드에 사용."""
    debate: DebateResult
    risk: Optional[RiskTriadResult] = None
    # 메모리 회상 결과 (원시 dict 리스트)
    memories: list[dict] = field(default_factory=list)
    # build_prompt 에 주입할 통합 debate_block 문자열
    combined_block: str = ""

    def to_payload(self) -> dict:
        """SSE/JSON 직렬화용."""
        return {
            "debate": self.debate.to_payload() if self.debate else None,
            "risk": self.risk.to_payload() if self.risk else None,
            "memories": list(self.memories),
        }


def _merge_blocks(*blocks: str) -> str:
    """비어있지 않은 블록만 구분선으로 이어 붙인다."""
    sep = "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    non_empty = [b for b in blocks if b and b.strip()]
    if not non_empty:
        return ""
    return sep.lstrip("\n").join(non_empty)


def run_pipeline(
    context_blob: str,
    pair_label: str,
    memory: Optional["FinancialSituationMemory"] = None,
    current_situation: Optional[str] = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> PipelineResult:
    """
    전체 에이전트 파이프라인 실행.

    Parameters
    ----------
    context_blob : str
        모든 에이전트가 공통으로 보는 시장 데이터 블록.
    pair_label : str
        "BTC/USDC" 등.
    memory : FinancialSituationMemory, optional
        Phase 3 메모리 객체. 없으면 메모리 단계 건너뜀.
    current_situation : str, optional
        메모리 쿼리용 요약 텍스트. 보통 context_blob 일부/전체.
    progress_cb : callable, optional
        (phase, detail) 콜백.

    Returns
    -------
    PipelineResult
    """
    # 1) Bull/Bear 토론
    debate = run_bull_bear_debate(
        context_blob=context_blob,
        pair_label=pair_label,
        progress_cb=progress_cb,
    )

    # 2) Risk Triad (Bull/Bear 결과를 입력으로)
    risk: Optional[RiskTriadResult] = None
    if RISK_TRIAD_IN_PIPELINE:
        risk = run_risk_triad(
            context_blob=context_blob,
            pair_label=pair_label,
            bull_final=debate.final_bull,
            bear_final=debate.final_bear,
            progress_cb=progress_cb,
        )

    # 3) 메모리 회상 (과거 유사 상황 top-K)
    memories: list[dict] = []
    memory_block = ""
    if (
        MEMORY_IN_PIPELINE
        and _MEMORY_AVAILABLE
        and memory is not None
        and current_situation
    ):
        if progress_cb:
            progress_cb("memory", "과거 유사 상황 검색 중")
        try:
            memories = memory.get_memories(current_situation, top_k=3)
            if format_memory_block is not None:
                memory_block = format_memory_block(memories)
        except Exception as exc:
            memory_block = f"[과거 유사 상황]\n  회상 실패 — {type(exc).__name__}: {exc}"

    # 4) 최종 주입용 통합 블록
    debate_block = format_debate_block(debate)
    risk_block = format_risk_block(risk) if risk is not None else ""
    combined = _merge_blocks(debate_block, risk_block, memory_block)

    return PipelineResult(
        debate=debate,
        risk=risk,
        memories=memories,
        combined_block=combined,
    )
