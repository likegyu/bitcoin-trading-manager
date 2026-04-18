# =============================================
# Multi-Agent Orchestrator (Phase 2+)
# =============================================
# 파이프라인 단계:
#   1) Bull/Bear Debate  (debate.py)
#   2) Investment Judge  (judge.py)
#   3) Risk Triad        (risk_triad.py)
#   4) Memory Recall     (memory.py — FinancialSituationMemory)
#   5) 최종 통합 블록 조립 → analyzer.analyze_with_claude() 에 주입
#
# 에이전트별 메모리 쓰기:
#   - Bull/Bear debate 발언 → bull/bear 역할 메모리에 기록
#   - Judge 판정          → judge 역할 메모리에 기록
#   - Risk Triad 발언     → aggressive/conservative/neutral 역할 메모리에 기록
#   (analyzer.run_full_analysis 에서 analyst 메모리 기록은 그대로 유지)
# =============================================
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

_logger = logging.getLogger(__name__)

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

# Judge — 선택적 임포트 (환경변수 JUDGE_ENABLED=0 시에도 모듈은 로드)
try:
    from .judge import run_judge, format_judge_block, JudgeResult
    _JUDGE_AVAILABLE = True
except Exception as _judge_exc:
    run_judge = None           # type: ignore
    format_judge_block = None  # type: ignore
    JudgeResult = None         # type: ignore
    _JUDGE_AVAILABLE = False
    _logger.warning("agents.judge 로드 실패 — %s: %s", type(_judge_exc).__name__, _judge_exc)

# AgentMemories — 선택적 임포트 (rank_bm25 미설치 시에도 동작)
try:
    from .memory import FinancialSituationMemory, AgentMemories, get_agent_memories, format_memory_block
    _MEMORY_AVAILABLE = True
except Exception as _mem_exc:
    FinancialSituationMemory = None  # type: ignore
    AgentMemories = None             # type: ignore
    get_agent_memories = None        # type: ignore
    format_memory_block = None       # type: ignore
    _MEMORY_AVAILABLE = False
    _logger.warning(
        "agents.memory 로드 실패 — %s: %s (rank_bm25 미설치 여부 확인)",
        type(_mem_exc).__name__, _mem_exc,
    )


ProgressCallback = Callable[[str, str], None]


# 파이프라인 단계 ON/OFF (환경변수로 제어)
RISK_TRIAD_IN_PIPELINE = (
    os.getenv("RISK_ENABLED", "1") not in ("0", "false", "False", "")
)
MEMORY_IN_PIPELINE = (
    os.getenv("MEMORY_ENABLED", "1") not in ("0", "false", "False", "")
)
# 에이전트 메모리 쓰기 여부 (분석 환경에서 기록 방지용)
AGENT_MEMORY_WRITE_ENABLED = (
    os.getenv("AGENT_MEMORY_WRITE_ENABLED", "1") not in ("0", "false", "False", "")
)


@dataclass
class PipelineResult:
    """run_pipeline 의 결과 — analyzer 가 최종 빌드에 사용."""
    debate: DebateResult
    judge: Optional["JudgeResult"] = None
    risk: Optional[RiskTriadResult] = None
    # 메모리 회상 결과 (원시 dict 리스트) — analyst 메모리
    memories: list[dict] = field(default_factory=list)
    # build_prompt 에 주입할 통합 debate_block 문자열
    combined_block: str = ""

    def to_payload(self) -> dict:
        """SSE/JSON 직렬화용."""
        judge_payload = None
        if self.judge is not None:
            try:
                judge_payload = self.judge.to_payload()
            except Exception:
                pass
        return {
            "debate":   self.debate.to_payload() if self.debate else None,
            "judge":    judge_payload,
            "risk":     self.risk.to_payload() if self.risk else None,
            "memories": list(self.memories),
        }


def _merge_blocks(*blocks: str) -> str:
    """비어있지 않은 블록만 구분선으로 이어 붙인다."""
    sep = "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    non_empty = [b for b in blocks if b and b.strip()]
    if not non_empty:
        return ""
    return sep.lstrip("\n").join(non_empty)


def _write_agent_memory(
    agent_memories: "AgentMemories",
    role: str,
    situation: str,
    advice: str,
    meta: dict,
) -> None:
    """에이전트 메모리에 안전하게 기록. 실패해도 파이프라인은 계속 진행."""
    if not AGENT_MEMORY_WRITE_ENABLED:
        return
    try:
        mem = agent_memories.get(role)
        stored = mem.add_situation(
            situation=situation,
            advice=advice,
            outcome="",
            meta=meta,
        )
        if stored is None:
            _logger.debug("agent_memory[%s]: 최근 기록과 유사 — dedup skip", role)
    except Exception as exc:
        _logger.warning("agent_memory[%s] 쓰기 실패 — %s", role, exc)


def run_pipeline(
    context_blob: str,
    pair_label: str,
    memory: Optional["FinancialSituationMemory"] = None,
    current_situation: Optional[str] = None,
    progress_cb: Optional[ProgressCallback] = None,
    agent_memories: Optional["AgentMemories"] = None,
    price_at_analysis: Optional[float] = None,
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
        Phase 3 analyst 메모리. 없으면 메모리 단계 건너뜀.
    current_situation : str, optional
        메모리 쿼리용 요약 텍스트 (situation_tags 또는 context_blob 일부).
    progress_cb : callable, optional
        (phase, detail) 콜백.
    agent_memories : AgentMemories, optional
        역할별 메모리. 제공 시 Bull/Bear/Judge/Risk 에이전트에 과거 기억 주입 + 사후 기록.

    Returns
    -------
    PipelineResult
    """
    _query = current_situation or context_blob[:200]

    # ── 1) Bull/Bear 토론 ─────────────────────────────
    debate = run_bull_bear_debate(
        context_blob=context_blob,
        pair_label=pair_label,
        progress_cb=progress_cb,
        agent_memories=agent_memories,
        memory_query=_query,
    )

    # 1-a) Debate 부분 실패 감지
    if debate.enabled and debate.error:
        missing = []
        if not debate.final_bull:
            missing.append("Bull")
        if not debate.final_bear:
            missing.append("Bear")
        warn_detail = (
            f"토론 부분 실패 ({debate.error})"
            + (f" — {'/'.join(missing)} 발언 누락, 하위 단계 입력 비대칭" if missing else "")
        )
        _logger.warning("pipeline: %s", warn_detail)
        if progress_cb:
            progress_cb("debate_warn", warn_detail)

    # 1-b) Bull/Bear 역할 메모리 기록
    _price_meta = {"price_at_analysis": price_at_analysis} if price_at_analysis else {}
    if agent_memories is not None and debate.enabled:
        if debate.final_bull:
            _write_agent_memory(
                agent_memories, "bull",
                situation=_query,
                advice=debate.final_bull,
                meta={"pair": pair_label, "round": debate.rounds, **_price_meta},
            )
        if debate.final_bear:
            _write_agent_memory(
                agent_memories, "bear",
                situation=_query,
                advice=debate.final_bear,
                meta={"pair": pair_label, "round": debate.rounds, **_price_meta},
            )

    # ── 2) 투자 심판(Judge) ────────────────────────────
    judge: Optional["JudgeResult"] = None
    if _JUDGE_AVAILABLE and run_judge is not None:
        if progress_cb:
            progress_cb("judge", "투자 심판 중재 중")
        judge = run_judge(
            context_blob=context_blob,
            pair_label=pair_label,
            bull_final=debate.final_bull,
            bear_final=debate.final_bear,
            agent_memories=agent_memories,
            memory_query=_query,
            progress_cb=None,  # progress_cb 은 이미 위에서 emit 함
        )
        # Judge 역할 메모리 기록
        if (
            agent_memories is not None
            and judge is not None
            and judge.enabled
            and judge.verdict
        ):
            _write_agent_memory(
                agent_memories, "judge",
                situation=_query,
                advice=f"판정: {judge.verdict}\n이유: {judge.reasoning}",
                meta={
                    "pair": pair_label,
                    "verdict": judge.verdict,
                    "bull_key": judge.bull_key,
                    "bear_key": judge.bear_key,
                    **_price_meta,
                },
            )

    # Judge 블록 (Risk Triad 에 주입할 텍스트)
    _judge_block_text = ""
    if format_judge_block is not None and judge is not None:
        _judge_block_text = format_judge_block(judge)

    # ── 3) Risk Triad ─────────────────────────────────
    risk: Optional[RiskTriadResult] = None
    if RISK_TRIAD_IN_PIPELINE:
        risk = run_risk_triad(
            context_blob=context_blob,
            pair_label=pair_label,
            bull_final=debate.final_bull,
            bear_final=debate.final_bear,
            progress_cb=progress_cb,
            agent_memories=agent_memories,
            memory_query=_query,
            judge_block=_judge_block_text,
        )
        # Risk 역할 메모리 기록
        if agent_memories is not None and risk is not None and risk.enabled:
            for role_key, final_text in (
                ("aggressive",  risk.final_aggressive),
                ("conservative", risk.final_conservative),
                ("neutral",      risk.final_neutral),
            ):
                if final_text:
                    _write_agent_memory(
                        agent_memories, role_key,
                        situation=_query,
                        advice=final_text,
                        meta={"pair": pair_label, "rounds": risk.rounds, **_price_meta},
                    )

    # ── 4) Analyst 메모리 회상 ─────────────────────────
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

    # ── 5) 최종 주입용 통합 블록 ──────────────────────
    debate_block = format_debate_block(debate)
    risk_block   = format_risk_block(risk) if risk is not None else ""
    judge_block  = _judge_block_text       # 이미 만들어 둔 것 재활용
    combined = _merge_blocks(debate_block, judge_block, risk_block, memory_block)

    return PipelineResult(
        debate=debate,
        judge=judge,
        risk=risk,
        memories=memories,
        combined_block=combined,
    )
