# =============================================
# Multi-Agent Debate & Memory Layer
# =============================================
# TradingAgents(https://github.com/TauricResearch/TradingAgents) 의
# 애널리스트 팀 패턴을 BTC 선물 맥락으로 포팅한 레이어.
#
# 구조:
#   - Phase 1: Bull / Bear Researcher (debate.py, prompts.py)
#   - Phase 2: Risk Triad + Pipeline (risk_triad.py, risk_prompts.py, pipeline.py)
#   - Phase 3: BM25 Memory + Reflection (memory.py, reflection.py)
#
# 철학:
#   - LangGraph/LangChain 의존성 없이 순수 anthropic SDK + asyncio로 동작
#   - 토론/리스크/리플렉션은 "quick" 모델(Haiku)로 저비용 운영
#   - 최종 종합은 analyzer.analyze_with_claude() 가 담당 (deep 모델)
# =============================================

import logging as _logging

_log = _logging.getLogger(__name__)

from .debate import run_bull_bear_debate, format_debate_block, DebateResult
from .risk_triad import run_risk_triad, format_risk_block, RiskTriadResult
from .pipeline import run_pipeline, PipelineResult

# 메모리/리플렉션은 rank_bm25 의존성이 있을 수 있어 선택적 임포트.
try:
    from .memory import (
        FinancialSituationMemory,
        format_memory_block,
        get_memory,
    )
    _MEMORY_OK = True
except Exception as _memory_exc:
    FinancialSituationMemory = None   # type: ignore
    format_memory_block = None        # type: ignore
    get_memory = None                 # type: ignore
    _MEMORY_OK = False
    _log.warning(
        "agents.memory 로드 실패 — %s: %s (rank_bm25 미설치 여부 확인)",
        type(_memory_exc).__name__, _memory_exc,
    )

try:
    from .reflection import reflect_on_record, ReflectionResult
    _REFLECTION_OK = True
except Exception as _reflect_exc:
    reflect_on_record = None          # type: ignore
    ReflectionResult = None           # type: ignore
    _REFLECTION_OK = False
    _log.warning(
        "agents.reflection 로드 실패 — %s: %s",
        type(_reflect_exc).__name__, _reflect_exc,
    )

__all__ = [
    # Phase 1
    "run_bull_bear_debate",
    "format_debate_block",
    "DebateResult",
    # Phase 2
    "run_risk_triad",
    "format_risk_block",
    "RiskTriadResult",
    "run_pipeline",
    "PipelineResult",
    # Phase 3
    "FinancialSituationMemory",
    "format_memory_block",
    "get_memory",
    "reflect_on_record",
    "ReflectionResult",
]
