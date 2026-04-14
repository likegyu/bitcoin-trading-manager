# =============================================
# Signal Processing — 최종 분석 텍스트에서 구조화된 시그널 추출
# =============================================
# 역할:
#   - 최종 애널리스트 리포트 텍스트를 파싱해 TradingSignal 을 생성
#   - 투자 심판(Judge) 결론을 선택적으로 반영해 시그널 신뢰도를 조정
#   - BUY / SELL / HOLD 외 OVERWEIGHT / UNDERWEIGHT 중간 강도도 지원
#
# 사용:
#   from agents.signal_processing import extract_trading_signal, TradingSignal
# =============================================
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ── 시그널 강도 정의 ─────────────────────────────────
# 강도 순서: STRONG_BUY > BUY > OVERWEIGHT > HOLD > UNDERWEIGHT > SELL > STRONG_SELL
SIGNAL_STRENGTH: dict[str, int] = {
    "STRONG_BUY":    3,
    "BUY":           2,
    "OVERWEIGHT":    1,
    "HOLD":          0,
    "UNDERWEIGHT":  -1,
    "SELL":         -2,
    "STRONG_SELL":  -3,
}

# 한국어 관점 → 영문 시그널 매핑
_VIEW_MAP: dict[str, str] = {
    "상방 우위": "BUY",
    "하방 우위": "SELL",
    "중립":       "HOLD",
    # 하위 호환 — 구 포맷 시그널 직접 입력 시
    "매수":       "BUY",
    "매도":       "SELL",
    "홀드":       "HOLD",
}

# Judge 판정 → 방향 편향 매핑
_JUDGE_BIAS: dict[str, int] = {
    "상방 우위": +1,
    "하방 우위": -1,
    "중립":       0,
}


@dataclass
class TradingSignal:
    """구조화된 트레이딩 시그널."""
    # 핵심 시그널
    signal_en:    str   # BUY / SELL / HOLD / OVERWEIGHT / UNDERWEIGHT / STRONG_BUY / STRONG_SELL
    signal_kr:    str   # 매수 / 매도 / 홀드 / 비중확대 / 비중축소
    strength:     int   # SIGNAL_STRENGTH 에 따른 정수 (-3 ~ +3)

    # 신뢰도 & 레짐
    confidence:   int   # 0-100 (파싱값)
    regime:       str   # 시장 레짐 문자열 (없으면 "")

    # Judge 연계
    judge_verdict:      str = ""    # 투자 심판 판정
    judge_aligned:      bool = True # 시그널 방향과 심판 판정이 일치하는지

    # 원시 파싱값
    raw_view:     str = ""  # 관점 원문

    # 추가 컨텍스트
    notes:        list[str] = field(default_factory=list)

    @property
    def is_bullish(self) -> bool:
        return self.strength > 0

    @property
    def is_bearish(self) -> bool:
        return self.strength < 0

    @property
    def is_neutral(self) -> bool:
        return self.strength == 0

    def to_dict(self) -> dict:
        return {
            "signal_en":       self.signal_en,
            "signal_kr":       self.signal_kr,
            "strength":        self.strength,
            "confidence":      self.confidence,
            "regime":          self.regime,
            "judge_verdict":   self.judge_verdict,
            "judge_aligned":   self.judge_aligned,
            "raw_view":        self.raw_view,
            "notes":           list(self.notes),
        }


# ── 내부 파싱 헬퍼 ────────────────────────────────────

def _parse_view(text: str) -> str:
    """관점/시그널 행에서 뷰 텍스트 추출."""
    m = re.search(
        r'(?:관점|시그널)[^:：\n]*[:：]?\s*(상방 우위|하방 우위|중립|매수|매도|홀드)',
        text,
    )
    if m:
        return m.group(1)
    # 앞 300자에서 첫 번째 키워드 위치 기반 폴백
    front = text[:300]
    candidates = ["상방 우위", "하방 우위", "중립", "매수", "매도", "홀드"]
    positions = {kw: front.find(kw) for kw in candidates if kw in front}
    if positions:
        return min(positions, key=positions.get)
    return "중립"


def _parse_confidence(text: str) -> int:
    m = re.search(r'(?:확신도|신뢰도)\D*?(\d{1,3})', text)
    if m:
        return min(int(m.group(1)), 100)
    return 50


def _parse_regime(text: str) -> str:
    m = re.search(r'시장\s*레짐\s*[:：]\s*(.+?)(?:\n|$)', text)
    if m:
        return m.group(1).strip()
    return ""


def _view_to_signal_en(view: str) -> str:
    return _VIEW_MAP.get(view, "HOLD")


def _signal_en_to_kr(signal_en: str) -> str:
    _map = {
        "STRONG_BUY":   "강력 매수",
        "BUY":          "매수",
        "OVERWEIGHT":   "비중확대",
        "HOLD":         "홀드",
        "UNDERWEIGHT":  "비중축소",
        "SELL":         "매도",
        "STRONG_SELL":  "강력 매도",
    }
    return _map.get(signal_en, "홀드")


# ── 공개 API ──────────────────────────────────────────

def extract_trading_signal(
    raw_text: str,
    judge_result=None,  # Optional[JudgeResult] — 순환 임포트 방지를 위해 Any
) -> TradingSignal:
    """
    최종 애널리스트 리포트 텍스트에서 TradingSignal 을 추출한다.

    Parameters
    ----------
    raw_text : str
        analyze_with_claude() 의 raw_text (애널리스트 전체 응답).
    judge_result : JudgeResult, optional
        투자 심판 결론. 제공 시 시그널 강도/정렬 여부를 추가 기록.
    """
    raw_view  = _parse_view(raw_text)
    confidence = _parse_confidence(raw_text)
    regime     = _parse_regime(raw_text)
    signal_en  = _view_to_signal_en(raw_view)
    strength   = SIGNAL_STRENGTH.get(signal_en, 0)

    notes: list[str] = []

    # ── Judge 연계 ───────────────────────────────────
    judge_verdict = ""
    judge_aligned = True
    if judge_result is not None and getattr(judge_result, "enabled", False):
        judge_verdict = getattr(judge_result, "verdict", "")
        judge_bias = _JUDGE_BIAS.get(judge_verdict, 0)

        if judge_bias != 0 and strength != 0:
            # 방향이 일치하는지 확인
            judge_aligned = (judge_bias > 0) == (strength > 0)
            if not judge_aligned:
                notes.append(
                    f"⚠️ 애널리스트({raw_view})와 심판({judge_verdict}) 방향 불일치 — 확신도 신중하게 해석"
                )
            else:
                notes.append(
                    f"✅ 심판 판정({judge_verdict})이 애널리스트 관점과 일치"
                )
        elif judge_bias == 0 and strength != 0:
            notes.append(f"심판 판정 중립 — 시장 방향 불확실성 고려")
        elif judge_bias != 0 and strength == 0:
            notes.append(f"심판 판정 {judge_verdict} — 홀드 구간에서 방향성 힌트")

    signal_kr = _signal_en_to_kr(signal_en)

    return TradingSignal(
        signal_en=signal_en,
        signal_kr=signal_kr,
        strength=strength,
        confidence=confidence,
        regime=regime,
        judge_verdict=judge_verdict,
        judge_aligned=judge_aligned,
        raw_view=raw_view,
        notes=notes,
    )
