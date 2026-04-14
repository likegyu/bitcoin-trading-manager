# =============================================
# Financial Situation Memory (BM25 기반)
# =============================================
# 원본: TradingAgents/tradingagents/agents/utils/memory.py
# 적용:
#   - OpenAI embeddings 의존 제거 → BM25 (rank_bm25) 로 대체
#     * 순수 로컬/오프라인, 외부 API 호출 없음
#     * BTC 분석 맥락에서 "지지 이탈 + 펀딩 과열" 류 키워드 매칭에 충분
#   - JSONL 파일 기반 영속화 (/data/memory/*.jsonl)
#   - 분석 단계마다 get_memories(current_situation, top_k) 로 회상
#   - 분석 사후 add_situation(situation, advice, outcome) 로 경험 축적 (reflection)
# =============================================
from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except Exception:
    BM25Okapi = None  # type: ignore
    _BM25_AVAILABLE = False


# ── 저장 위치 ──────────────────────────────────────
# crypto_analyzer/ 와 같은 레벨에 data/memory/ 생성.
# 이 모듈 파일 기준 부모의 부모.
_THIS_DIR = Path(__file__).resolve().parent         # .../crypto_analyzer/agents
_PROJECT_DIR = _THIS_DIR.parent                     # .../crypto_analyzer
DEFAULT_MEMORY_DIR = Path(
    os.getenv("MEMORY_DIR", str(_PROJECT_DIR / "data" / "memory"))
)


# ── 토크나이저 ────────────────────────────────────
# 한국어 + 영문 + 숫자 혼재 → 공백/특수문자로 단순 분리 후 소문자화.
# 한글 형태소 분석기를 쓰면 더 좋지만 의존성 무거워져서 생략.
_TOKEN_RE = re.compile(r"[A-Za-z가-힣0-9%]+")


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class MemoryRecord:
    """한 건의 경험 — 상황 + 당시 조언 + (선택) 결과."""
    timestamp: str       # ISO8601
    situation: str       # 당시 시장 상황 요약 (쿼리 대상)
    advice: str          # 당시 애널리스트 조언/판단
    outcome: str = ""    # 사후 결과 (reflection 단계에서 채움)
    meta: dict = None    # type: ignore # 자유 메타 (심볼·PnL 등)

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.meta is None:
            d["meta"] = {}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryRecord":
        return cls(
            timestamp=d.get("timestamp", ""),
            situation=d.get("situation", ""),
            advice=d.get("advice", ""),
            outcome=d.get("outcome", ""),
            meta=d.get("meta") or {},
        )


class FinancialSituationMemory:
    """
    시장 상황-조언 페어를 저장하고 BM25 로 유사 상황을 회상한다.

    Usage
    -----
    mem = FinancialSituationMemory("bull_bear")
    mem.add_situation(situation_text, advice_text, outcome="")
    hits = mem.get_memories(current_situation_text, top_k=3)
      # hits: [{"record": MemoryRecord.to_dict(), "score": float}, ...]
    """

    def __init__(
        self,
        name: str,
        memory_dir: Optional[Path] = None,
    ):
        self.name = name
        self.memory_dir = Path(memory_dir) if memory_dir else DEFAULT_MEMORY_DIR
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.memory_dir / f"{name}.jsonl"

        self._lock = threading.Lock()
        self._records: list[MemoryRecord] = []
        self._load()

    # ── 영속화 ────────────────────────────────────
    def _load(self) -> None:
        self._records = []
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self._records.append(MemoryRecord.from_dict(obj))
        except OSError:
            self._records = []

    def _append_to_disk(self, record: MemoryRecord) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def _rewrite_disk(self) -> None:
        tmp = self.path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for r in self._records:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
        tmp.replace(self.path)

    # ── 내부 dedup 헬퍼 ────────────────────────────
    @staticmethod
    def _jaccard(a: list[str], b: list[str]) -> float:
        if not a or not b:
            return 0.0
        sa, sb = set(a), set(b)
        inter = len(sa & sb)
        union = len(sa | sb)
        return inter / union if union else 0.0

    # ── API ───────────────────────────────────────
    def add_situation(
        self,
        situation: str,
        advice: str,
        outcome: str = "",
        meta: Optional[dict] = None,
        dedup_threshold: float = 0.70,
        dedup_window: int = 2,
    ) -> Optional[MemoryRecord]:
        """
        이번 판단을 메모리에 추가한다.

        dedup:
          최근 `dedup_window` 개 기록의 situation 토큰과 Jaccard >= threshold 이면
          '실질적으로 같은 구조' 로 보고 저장을 건너뛴다 (무한 누적 방지).
          건너뛴 경우 반환값은 None.

          BTC 분석처럼 지표가 비슷한 상황이 반복될 때 0.92는 너무 엄격해서
          대부분이 dedup 처리됨 → 0.70 / window=2 로 완화.
        """
        new_tokens = _tokenize(situation)
        with self._lock:
            # 최근 N개와 비교
            recent = self._records[-dedup_window:] if dedup_window > 0 else []
            for r in recent:
                sim = self._jaccard(new_tokens, _tokenize(r.situation))
                if sim >= dedup_threshold:
                    return None

            rec = MemoryRecord(
                timestamp=datetime.utcnow().isoformat(timespec="seconds") + "Z",
                situation=situation.strip(),
                advice=advice.strip(),
                outcome=outcome.strip(),
                meta=meta or {},
            )
            self._records.append(rec)
            self._append_to_disk(rec)
        return rec

    def update_outcome(self, timestamp: str, outcome: str) -> bool:
        """
        특정 timestamp 의 기록에 사후 결과를 덧붙인다.
        Reflection 에서 사용.
        """
        with self._lock:
            for r in self._records:
                if r.timestamp == timestamp:
                    r.outcome = (r.outcome + "\n" + outcome).strip() if r.outcome else outcome
                    self._rewrite_disk()
                    return True
        return False

    def get_memories(self, query: str, top_k: int = 3) -> list[dict]:
        """
        query 와 가장 유사한 상황 top_k 개를 반환.
        반환: [{"record": {...}, "score": float}, ...]
        """
        with self._lock:
            records = list(self._records)

        if not records:
            return []
        if not _BM25_AVAILABLE:
            # rank_bm25 미설치 → 최근 K개 fallback
            recent = records[-top_k:]
            return [{"record": r.to_dict(), "score": 0.0} for r in recent[::-1]]

        corpus_tokens = [_tokenize(r.situation) for r in records]
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        try:
            bm25 = BM25Okapi(corpus_tokens)
            scores = bm25.get_scores(query_tokens)
        except Exception:
            return []

        # 점수 내림차순 정렬 후 상위 K개
        ranked = sorted(
            zip(records, scores),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        # 모두 0점이면 (매칭 없음) 빈 리스트
        ranked = [(r, s) for r, s in ranked if s > 0]
        if not ranked:
            return []

        return [{"record": r.to_dict(), "score": float(s)} for r, s in ranked]

    def __len__(self) -> int:
        return len(self._records)

    def size(self) -> int:
        return len(self._records)

    # ── 공개 열람 API (server.py 등 외부에서 _records 직접 접근 대신 사용) ──
    def list_records(self) -> list[MemoryRecord]:
        """전체 기록의 shallow copy 를 반환."""
        with self._lock:
            return list(self._records)

    def list_pending_reflections(
        self,
        min_age_seconds: float = 1800.0,
        limit: int = 5,
    ) -> list[MemoryRecord]:
        """
        Reflection 대상 — outcome 이 비어 있고 충분히 시간이 지난 기록만.
        오래된 것 우선 (FIFO), 최대 `limit` 개.
        """
        import datetime as _dt
        now = _dt.datetime.now(_dt.timezone.utc)
        out: list[MemoryRecord] = []

        with self._lock:
            for rec in self._records:
                if rec.outcome:
                    continue
                try:
                    ts = _dt.datetime.fromisoformat(rec.timestamp.replace("Z", "+00:00"))
                except Exception:
                    continue
                elapsed = (now - ts).total_seconds()
                if elapsed < min_age_seconds:
                    continue
                out.append(rec)
                if len(out) >= limit:
                    break
        return out


# ── 프롬프트 주입용 포매터 ──────────────────────────
def format_memory_block(memories: list[dict]) -> str:
    """
    get_memories() 결과를 최종 프롬프트 주입용 [과거 유사 상황] 블록으로 변환.
    비어 있으면 빈 문자열 반환.
    """
    if not memories:
        return ""

    lines = ["[과거 유사 상황 — BM25 회상]"]
    lines.append(
        "  ⚠️ 아래는 비슷한 지표 조합에서 과거에 어떤 조언을 했고 결과가 어땠는지의 기록입니다. "
        "단순히 답습하지 말고, 현재 조건과 차이를 대조하면서 교훈만 추출하세요."
    )

    for i, item in enumerate(memories, start=1):
        rec = item.get("record", {}) if isinstance(item, dict) else {}
        ts = rec.get("timestamp", "?")
        score = item.get("score", 0.0)
        situation = (rec.get("situation") or "").strip()
        advice = (rec.get("advice") or "").strip()
        outcome = (rec.get("outcome") or "").strip()

        lines.append(f"\n— 사례 {i} · {ts} · 유사도 {score:.2f} —")
        if situation:
            # 상황 요약 — 너무 길면 잘라낸다 (프롬프트 비용 관리)
            snippet = situation if len(situation) <= 600 else situation[:600] + " …"
            lines.append(f"  상황: {snippet}")
        if advice:
            snippet = advice if len(advice) <= 400 else advice[:400] + " …"
            lines.append(f"  당시 조언: {snippet}")
        if outcome:
            snippet = outcome if len(outcome) <= 300 else outcome[:300] + " …"
            lines.append(f"  실제 결과: {snippet}")
        else:
            lines.append("  실제 결과: (아직 리플렉션 미기록)")

    return "\n".join(lines)


# ── 편의 팩토리 ────────────────────────────────────
_MEMORIES: dict[str, FinancialSituationMemory] = {}
_FACTORY_LOCK = threading.Lock()


def get_memory(name: str = "analyst") -> FinancialSituationMemory:
    """모듈 전역 싱글턴 — 동일 name 은 같은 인스턴스를 공유."""
    with _FACTORY_LOCK:
        mem = _MEMORIES.get(name)
        if mem is None:
            mem = FinancialSituationMemory(name=name)
            _MEMORIES[name] = mem
        return mem


# ══════════════════════════════════════════════════════
# AgentMemories — 역할별 독립 메모리 집합
# ══════════════════════════════════════════════════════
# TradingAgents 의 bull_memory / bear_memory / trader_memory 패턴을
# BTC 선물 에이전트 구조에 맞게 확장.
# 각 역할이 자신의 과거 판단 이력을 분리해 학습하므로,
# Bull 의 상승 편향 실수는 Bull 메모리에, Bear 의 하락 과신은 Bear 메모리에 쌓인다.

AGENT_ROLES = ("bull", "bear", "judge", "aggressive", "conservative", "neutral", "analyst")

_AGENT_MEMORIES_INSTANCE: Optional["AgentMemories"] = None
_AGENT_MEMORIES_LOCK = threading.Lock()


class AgentMemories:
    """
    역할별 FinancialSituationMemory 를 한 곳에서 관리하는 컨테이너.

    사용 예:
        am = get_agent_memories()
        past = am.recall("bull", situation_query)   # 프롬프트 삽입용 문자열
        am.get("bull").add_situation(...)           # 직접 저장
    """

    def __init__(self, memory_dir: Optional[Path] = None):
        _dir = Path(memory_dir) if memory_dir else DEFAULT_MEMORY_DIR
        self._stores: dict[str, FinancialSituationMemory] = {
            role: FinancialSituationMemory(role, _dir)
            for role in AGENT_ROLES
        }

    def get(self, role: str) -> FinancialSituationMemory:
        """역할별 메모리 인스턴스 반환. 알 수 없는 role 도 자동 생성."""
        if role not in self._stores:
            self._stores[role] = FinancialSituationMemory(role)
        return self._stores[role]

    def recall(self, role: str, situation: str, top_k: int = 2) -> str:
        """
        역할별 과거 기억을 회상해 프롬프트 삽입용 텍스트로 반환.
        기억이 없거나 BM25 매칭 실패 시 빈 문자열.
        """
        mems = self.get(role).get_memories(situation, top_k=top_k)
        if not mems:
            return ""
        lines = ["[과거 유사 상황에서의 내 판단 이력]"]
        for i, item in enumerate(mems, 1):
            rec = item.get("record", {}) if isinstance(item, dict) else {}
            score = item.get("score", 0.0)
            advice = (rec.get("advice") or "").strip()
            outcome = (rec.get("outcome") or "").strip()
            ts = rec.get("timestamp", "?")
            # 너무 길면 잘라서 프롬프트 토큰 절약
            advice_snippet = advice[:300] + " …" if len(advice) > 300 else advice
            outcome_snippet = outcome[:200] + " …" if len(outcome) > 200 else outcome
            lines.append(f"\n— 사례 {i} · {ts} · 유사도 {score:.2f} —")
            if advice_snippet:
                lines.append(f"  당시 주장: {advice_snippet}")
            if outcome_snippet:
                lines.append(f"  이후 결과: {outcome_snippet}")
            else:
                lines.append("  이후 결과: (미기록 — reflection 대기)")
        return "\n".join(lines)

    def all_roles(self) -> list[str]:
        return list(self._stores.keys())


def get_agent_memories() -> "AgentMemories":
    """프로세스 전역 싱글턴 AgentMemories."""
    global _AGENT_MEMORIES_INSTANCE
    with _AGENT_MEMORIES_LOCK:
        if _AGENT_MEMORIES_INSTANCE is None:
            _AGENT_MEMORIES_INSTANCE = AgentMemories()
        return _AGENT_MEMORIES_INSTANCE
