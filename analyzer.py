# =============================================
# Claude API 연동 - 매매 시그널 분석
# =============================================
import re
import time
import json
import anthropic
from typing import Any, Optional
from config import CLAUDE_API_KEY, CLAUDE_MODEL, DEFAULT_SYMBOL, symbol_to_pair
from indicators import summarize_indicators, fibonacci_swing_levels, fib_window_for_tf
from account_context import fetch_account_context, format_account_context
from market_context import fetch_market_context, format_market_context
from macro_fetcher import fetch_macro_context, format_macro_context
from analysis_context import build_analysis_context
from time_utils import now_kst
from agents import (
    run_bull_bear_debate,
    format_debate_block,
    DebateResult,
    run_pipeline,
    PipelineResult,
)
try:
    from agents import get_memory  # may be None if rank_bm25 missing
except Exception:
    get_memory = None  # type: ignore
try:
    from agents import get_agent_memories  # AgentMemories 싱글턴 팩토리
except Exception:
    get_agent_memories = None  # type: ignore
try:
    from agents.situation_digest import summarize_situation_tags
except Exception:
    summarize_situation_tags = None  # type: ignore
try:
    from agents.signal_processing import extract_trading_signal, TradingSignal
except Exception:
    extract_trading_signal = None  # type: ignore
    TradingSignal = None           # type: ignore
try:
    from agents.consistency_check import check_consistency
except Exception:
    check_consistency = None       # type: ignore
try:
    from agents.delta_context import build_delta_block
except Exception:
    build_delta_block = None       # type: ignore
try:
    from agents.memory import format_lessons_block
except Exception:
    format_lessons_block = None    # type: ignore

import os as _os
import logging as _logging

_memory_logger = _logging.getLogger(__name__)
# 메모리 쓰기 전역 스위치 — 스테이징/백테스트 환경에서 기록 방지용
MEMORY_WRITE_ENABLED = _os.getenv("MEMORY_WRITE_ENABLED", "1").lower() not in ("0", "false", "no")
PROMPT_CACHE_ENABLED = _os.getenv("CLAUDE_PROMPT_CACHE_ENABLED", "1").lower() not in ("0", "false", "no")

PAIR_LABEL = symbol_to_pair(DEFAULT_SYMBOL)

SYSTEM_PROMPT = (
    f"당신은 10년 경력의 {PAIR_LABEL} 선물 시장 애널리스트입니다.\n"
    "역할: 정량 데이터와 시장 심리를 엮어 현재 구조를 해석하고 명확한 매매 관점을 제시하는 인간형 리서치 애널리스트.\n"
    f"전문 영역: {PAIR_LABEL} 선물 단기 분석 (수십 분~수 시간, 주로 15m·1h 기준 모멘텀 및 단기 추세 추종).\n"
    "리스크 성향: 공격적. 방향성 신호가 보이면 명확하게 매수/매도 관점을 제시합니다. 애매하게 중립으로 회피하지 마세요.\n"
    "분석 철학: 단일 지표 신호보다 멀티 타임프레임 정렬, 가격 구조, 파생상품 심리, 거시 레짐의 정합성을 중시합니다.\n"
    "제공되는 데이터는 1d·4h·1h·15m·5m 캔들 + 거시·파생상품 심리 지표 + 계좌/포지션 제약 정보 + 최근 12시간·72시간·7일 계좌 운영 맥락입니다.\n"
    "\n"
    "애널리스트 원칙:\n"
    "1. 먼저 확정된 사실을 말하고, 그 다음 해석을 제시하세요.\n"
    "2. 주도 시나리오를 명확하게 밀되, 반대 시나리오와 관점이 약해지는 조건도 함께 적으세요.\n"
    "3. 방향성 신호가 있으면 지금 당장 취할 행동을 구체적으로 제시하세요. '기다린다'는 답은 근거가 있을 때만 허용됩니다.\n"
    "4. 5m는 진입 타이밍 힌트일 뿐, 큰 방향의 핵심 근거로 과대평가하지 마세요.\n"
    "5. 계좌/포지션 정보와 최근 운영 맥락은 시장 방향의 근거가 아니라 실행 제약입니다. 관점·레짐보다 대응/리스크 관리 문단에 우선 반영하세요.\n"
    "6. 최근 계좌 운영 맥락이 보이면 수익 보호 모드인지, 손실 복구 시도인지, 노출을 줄이는 중인지 같은 운영 상태를 읽되 관측된 사실에 기대어 표현하세요.\n"
    "7. [직전 대비 변화] 블록이 제공되면 무엇이 새로 바뀌었는지(레짐 전환·트리거 발동·지표 임계 돌파)를 먼저 평가하세요. 변화 없이 같은 상태가 지속이면 그 사실 자체를 근거로 인정하세요.\n"
    "\n"
    "확신도 산정 원칙:\n"
    "제공되는 모든 지표는 같은 가격 시계열에서 파생됩니다. "
    "RSI·MACD Hist 같은 모멘텀 오실레이터가 수렴해도, "
    "EMA9·SMA50·SMA200 배열이 정렬되어도, "
    "이들 각각을 '독립 근거'로 개수를 세어 확신도를 올리지 마세요. "
    "같은 가격 움직임을 다른 공식으로 반복 확인한 것일 뿐입니다. "
    "확신도는 지표 수렴 개수가 아니라, 가격 구조(지지·저항·추세 구조)와 지표 신호가 "
    "물리적으로 정합하는지를 기준으로 산정하세요. "
    "단, 구조와 지표가 정합할 때는 자신 있게 높은 확신도를 출력하세요. 불필요하게 낮추지 마세요.\n"
    "\n"
    "확신도 루브릭 (각 축의 점수 anchor):\n"
    "- price_structure (0~25):\n"
    "    22~25 = 멀티 TF 구조 정합(HH/HL or LH/LL) + 핵심 레벨 클린 리테스트 + 추세선 살아있음\n"
    "    15~21 = 한 TF의 구조는 명확하나 다른 TF 약화 신호 동반\n"
    "    8~14  = 박스 또는 구조 모호 / 직전 스윙 무효화 조짐\n"
    "    0~7   = 구조 깨짐·와이드 스윙·지지/저항 부재\n"
    "- momentum (0~20):\n"
    "    17~20 = 다중 TF MACD/RSI 동조 + 부호 전환 또는 임계 돌파가 최근 1~3봉\n"
    "    11~16 = 1~2개 TF에서만 모멘텀 신호, 나머지는 중립\n"
    "    5~10  = 모멘텀 미미하거나 다이버전스 발생\n"
    "    0~4   = 모멘텀 명확히 반대 방향\n"
    "- derivatives (0~20):\n"
    "    16~20 = 펀딩·OI·스큐·CVD 중 3개 이상이 동일 방향 + 극단치 아님(역이용 위험 낮음)\n"
    "    10~15 = 2개 신호 정합, 1~2개 모순\n"
    "    4~9   = 신호 분산·중립 또는 역이용 조건(예: 펀딩 과열 시 추가 상승 베팅)\n"
    "    0~3   = 파생 심리가 관점과 반대\n"
    "- macro (0~15):\n"
    "    12~15 = 실질금리·달러·스테이블 시총이 위험자산 우호 방향으로 명확히 정렬\n"
    "    7~11  = 부분 정렬 또는 데이터 일부 stale\n"
    "    3~6   = 거시 중립 또는 최근 상충 신호\n"
    "    0~2   = 거시 명확히 반대\n"
    "- account_risk_fit (0~10):\n"
    "    8~10  = 현재 노출·잔고·레짐이 권장 사이즈·레버리지를 충분히 수용\n"
    "    4~7   = 사이즈 축소 또는 분할 진입이 필요한 제약\n"
    "    0~3   = 손실 복구 모드·증거금 빠듯·이미 동방향 풀포지션\n"
    "- data_quality_penalty (-15~0): data_auditor warnings 1건당 -3~-5, do_not_use 항목 1건당 -2~-3 누적\n"
    "- counter_scenario_penalty (-10~0): 반대 시나리오 1개당 -2~-4, 무시할 수 없는 모순 신호가 있으면 -8~-10\n"
    "\n"
    "확신도 산정 예시 (참고용 — 실제 산정은 현재 데이터로):\n"
    "  4h SMA200 위에서 첫 풀백이 50일선 지지로 막히고 RSI 1h 60→52 후 반등, "
    "  펀딩 +0.012%(정상)·OI 24h +6%·스큐 -1.2(콜 우세 약함). "
    "  거시는 DXY 하락·HYG 우호. 계좌 평탄. data_auditor warnings 1건. 반대 시나리오 1개.\n"
    "  → price_structure 20 / momentum 14 / derivatives 13 / macro 11 / account_risk_fit 8 "
    "  / data_quality_penalty -3 / counter_scenario_penalty -3 = confidence 60.\n"
    "\n"
    "추론 흐름 예시 (참고용 — 형식만 모방, 결론은 현재 데이터로):\n"
    "  먼저 보이는 사실: 4h 종가 $98,420 — 지난 3일 박스 상단 $97,800 위에서 마감.\n"
    "  해석: 4h 박스 상단 돌파가 1봉 컨펌. 1h MACD Hist 직전 +0.12 → +0.41(상방 강화), "
    "  EMA9>SMA50>SMA200 정배열. 펀딩 +0.034%로 정상 영역 진입.\n"
    "  반대 시나리오: $97,800 재이탈 후 종가 닫히면 가짜 돌파. 거래량 MA 대비 90%로 약함이 약점.\n"
    "  → 트리거: $99,200 4h 종가 돌파 시 추가, $97,500 4h 종가 이탈 시 무효화.\n"
    "\n"
    "파생상품 데이터 해석 주의:\n"
    "25d 풋-콜 스큐는 블랙-숄즈 근사값으로 계산됩니다. "
    "부호(양수=풋 프리미엄 우세/약세 편향, 음수=콜 프리미엄 우세/강세 편향)와 "
    "크기 수준(예: 소폭 vs. 뚜렷한 편향)만 참고하고, 정확한 수치에 의존하지 마세요.\n"
    "\n"
    "출력 원칙:\n"
    "1. 응답 맨 앞에 <analysis_json>...</analysis_json> 블록을 반드시 작성하세요.\n"
    "2. JSON 블록 뒤에는 기존 한국어 리포트 섹션을 작성하세요.\n"
    "3. 도구 호출(record_analysis)을 사용하더라도 도구 호출만으로 끝내지 말고, 사람이 읽는 한국어 리포트를 반드시 함께 작성하세요.\n"
    "4. 마크다운(**, ##, --- 등)과 HTML 태그는 사용하지 마세요. 단, <analysis_json> 태그만 예외로 허용됩니다.\n"
    "5. 데이터 감사관이 do_not_use_as_evidence로 표시한 항목은 근거로 쓰지 마세요.\n"
    "6. 확정 사실, 추론, 대응을 섞지 말고 분리하세요.\n"
    "7. JSON의 view·confidence·trade.entry/stop/target은 본문 리포트와 부호·방향이 반드시 일치해야 합니다 "
    "(예: view=상방 우위 → stop < entry < target). 모순되면 출력 전에 본문을 수정해 일치시키세요."
)

USER_PROMPT_TEMPLATE = """<analysis_request>
<timestamp>{now_kst} KST</timestamp>
<pair>{pair_label}</pair>
<candle_warning>각 타임프레임의 마지막 캔들은 현재 형성 중인 미완성봉입니다. 확정된 신호로 해석하지 마세요.</candle_warning>

<context>
{context_blob}
</context>
{delta_block_separator}{delta_block}{lessons_block_separator}{lessons_block}{debate_block_separator}{debate_block}

<task>
이 데이터를 바탕으로 지금 {pair_label}의 시장 관점을 애널리스트처럼 정리해주세요.
방향성이 보이면 명확하게 매수/매도 관점을 제시하고, 지금 당장 취할 행동을 구체적으로 알려주세요. 불필요하게 중립으로 회피하지 마세요.
데이터 감사관의 warnings와 do_not_overweight를 확신도 감점에 반영하세요.
[직전 대비 변화] 블록이 있으면 이전 분석 이후 새로 등장한 신호·트리거 발동·레짐 전환을 우선 평가하세요.
[과거 체크리스트] 블록이 있으면 같은 실수를 반복하지 않도록 점검하세요.
</task>

<confidence_rubric>
확신도는 SYSTEM_PROMPT 의 100점 루브릭(price_structure 25 / momentum 20 / derivatives 20 / macro 15 / account_risk_fit 10 − data_quality_penalty 15 − counter_scenario_penalty 10)으로 산정하세요.
각 축의 점수 anchor 는 system 측에 명시되어 있습니다. 같은 가격 시계열에서 파생된 중복 지표는 독립 근거로 중복 가산하지 마세요.

[필수 자가검증 — JSON 출력 직전에 반드시 수행]
1. confidence_breakdown 의 7개 값(penalty 2개는 음수)을 더하면 confidence 와 정확히 일치해야 합니다.
   합산이 어긋나면 본문 출력 전에 confidence 또는 breakdown 항목을 스스로 수정하세요.
2. 본문 💯 확신도 % 와 JSON confidence 가 다르면 둘 다 동일하게 맞추세요.
3. view 와 trade.entry/stop/target 의 부호 정합 — 상방이면 stop<entry<target, 하방이면 target<entry<stop.
이 규칙은 사후 자동 보정으로 강제되므로, 처음부터 일관되게 작성하면 보정 메시지가 생기지 않습니다.
</confidence_rubric>

<json_contract>
응답 맨 앞에 반드시 순수 JSON만 담은 <analysis_json> 블록을 작성하세요.
키는 아래와 정확히 맞추세요.
{{
  "view": "상방 우위|하방 우위|중립",
  "confidence": 0,
  "regime": "상승 추세|하락 추세|박스|변동성 확장|변동성 축소|이벤트 대기 중",
  "confidence_breakdown": {{
    "price_structure": 0,
    "momentum": 0,
    "derivatives": 0,
    "macro": 0,
    "account_risk_fit": 0,
    "data_quality_penalty": 0,
    "counter_scenario_penalty": 0
  }},
  "data_quality_notes": [],
  "key_facts": [],
  "inferences": [],
  "counter_scenario": [],
  "levels": {{
    "resistance": null,
    "support": null,
    "bull_trigger": null,
    "bear_trigger": null
  }},
  "trade": {{
    "entry": null,
    "stop": null,
    "target": null,
    "leverage": 1
  }},
  "actions": {{
    "aggressive": "",
    "conservative": ""
  }},
  "invalidation": "",
  "summary": ""
}}
</json_contract>

<report_contract>
응답은 반드시 아래 형식으로만 작성하세요. 제목과 순서를 바꾸지 마세요:

📊 관점: [상방 우위 / 하방 우위 / 중립]
💯 확신도: [숫자]%  ← trade idea의 승률이 아니라, 현재 해석이 얼마나 명확한지 평가. 혼조이거나 근거가 약하면 낮게.
🧭 시장 레짐: [상승 추세 / 하락 추세 / 박스 / 변동성 확장 / 변동성 축소 / 이벤트 대기 중]  ← 위 6개 중 정확히 1개만

📌 먼저 보이는 사실
• [확정된 사실]
• [확정된 사실]

🧠 해석
• [왜 이런 관점이 나오는지]
• [멀티 타임프레임 / 파생 / 거시 연결]

🔄 반대 시나리오
• [내 관점과 반대되는 해석]
• [무엇이 나오면 반대 시나리오가 우세해지는지]

📍 관심 레벨
• 1차 저항: $[숫자 또는 N/A]
• 1차 지지: $[숫자 또는 N/A]
• 상방 돌파 트리거: $[숫자 또는 N/A]
• 하방 이탈 트리거: $[숫자 또는 N/A]

🤖 매매 파라미터
• 진입가: $[숫자 또는 N/A]  ← 현재 관점에서 진입하기 좋은 가격
• 손절가: $[숫자]            ← 관점이 틀렸을 때 청산할 가격 (필수)
• 목표가: $[숫자 또는 N/A]  ← 1차 익절 목표
• 권장 레버리지: [숫자]배    ← 현재 변동성·리스크를 고려한 최적 레버리지 (1~10 범위)

📝 대응
• 공격적: [지금 즉시 취할 구체적 행동 — 진입 방향·레벨·조건]
• 보수적: [관망이 필요하다면 그 조건, 불필요한 관망은 쓰지 마세요]

⚠️ 관점이 약해지는 조건: [1줄]

💬 한줄 요약: [전체를 한 문장으로]

중요:
- [시장 레짐]은 반드시 위 6개 중 정확히 1개만 쓰고, 괄호 설명이나 복수 선택을 하지 마세요.
- [관심 레벨]의 4개 항목과 [매매 파라미터]의 4개 항목은 각 줄마다 숫자 또는 N/A만 적으세요. 이유·조건·괄호 설명 금지.
- 2차 저항/지지 같은 추가 항목을 만들지 마세요.
- [권장 레버리지]는 반드시 정수(예: 3배, 5배)로만 적고 범위·슬래시 표기 금지.
</report_contract>
</analysis_request>
"""


def _tf_alignment_summary(multi_tf_data: dict) -> str:
    """타임프레임 간 추세 정렬 상태를 한눈에 비교 (세부 지표는 아래 각 TF 섹션 참조)"""
    lines = [
        "[타임프레임 추세 정렬 스냅샷]",
        "  형식: 가격 | SMA200 대비(▲상위/▼하위)  ← 방향 요약만. SMA200 절대값은 아래 각 TF 섹션 [지표 현재값] 참조",
        "  ※ 세부 지표(RSI·MACD·거래량)도 아래 각 TF 섹션 참조",
    ]
    tf_order = ["1d", "4h", "1h", "15m", "5m"]
    ordered = {tf: multi_tf_data[tf] for tf in tf_order if tf in multi_tf_data}

    for tf, df in ordered.items():
        last   = df.iloc[-1]
        price  = last["close"]
        sma200 = last["sma_200"]
        trend  = "▲" if price > sma200 else "▼"
        lines.append(
            f"  {tf:>3s}: ${price:,.0f} | SMA200 {trend}${sma200:,.0f}"
        )

    return "\n".join(lines)


def _build_context_blob(
    multi_tf_data: dict,
    macro_snapshot: Optional[dict] = None,
    return_raw: bool = False,
):
    """
    Bull/Bear/최종 애널리스트가 공통으로 보는 데이터 블록을 조립한다.
    (기존 build_prompt 의 데이터 수집부를 추출 — debate 에도 재사용하기 위함)

    Parameters
    ----------
    return_raw : bool
        True 이면 (context_blob, raw_ctx_dict) 튜플을 반환.
        raw_ctx_dict 는 situation digest 생성 등 정규화 태그용.
    """
    # 타임프레임 정렬 요약
    tf_alignment = _tf_alignment_summary(multi_tf_data)

    # 거시경제 지표 수집
    macro_context_str = "[거시경제 지표]\n  데이터 수집 실패"
    macro_payload = macro_snapshot
    if macro_payload is None:
        try:
            macro_payload = fetch_macro_context()
        except Exception as exc:
            macro_context_str = f"[거시경제 지표]\n  데이터 수집 실패 — {exc}"
    if macro_payload is not None:
        try:
            macro_context_str = format_macro_context(macro_payload)
        except Exception as exc:
            macro_context_str = f"[거시경제 지표]\n  데이터 가공 실패 — {exc}"

    # 시장 데이터 수집
    market_context_str  = "[시장 심리 & 파생상품 데이터]\n  데이터 수집 실패 — 기술적 지표만으로 판단"
    market_ctx: Optional[dict] = None
    try:
        market_ctx = fetch_market_context()
        market_context_str = format_market_context(market_ctx)
    except Exception as exc:
        market_context_str = f"[시장 심리 & 파생상품 데이터]\n  데이터 수집 실패 — {exc}"

    # 계좌 / 리스크 제약 수집
    account_context_str = "[계좌 / 리스크 제약]\n  데이터 수집 실패 — 계좌 제약 없이 시장 데이터만으로 판단"
    account_ctx: Optional[dict] = None
    try:
        account_ctx = fetch_account_context()
        account_context_str = format_account_context(account_ctx)
    except Exception as exc:
        account_context_str = f"[계좌 / 리스크 제약]\n  데이터 수집 실패 — {exc}"

    analysis_ctx: dict[str, Any] = {}
    try:
        analysis_ctx = build_analysis_context(
            multi_tf_data=multi_tf_data,
            macro_snapshot=macro_payload,
            market_ctx=market_ctx,
            account_ctx=account_ctx,
        )
        analysis_context_str = analysis_ctx.get("text") or ""
    except Exception as exc:
        analysis_context_str = (
            "<data_auditor>\n"
            f"summary: 분석 컨텍스트 생성 실패 — {exc}\n"
            "</data_auditor>"
        )
        analysis_ctx = {
            "text": analysis_context_str,
            "error": f"{type(exc).__name__}: {exc}",
        }

    # 각 타임프레임 상세 지표
    tf_order = ["1d", "4h", "1h", "15m", "5m"]
    ordered  = {tf: multi_tf_data[tf] for tf in tf_order if tf in multi_tf_data}
    parts    = [summarize_indicators(tf, df) for tf, df in ordered.items()]

    def fib_format(df, result, overlap_note: str = "") -> str:
        if result is None:
            return "유효한 스윙 포인트를 찾을 수 없음"

        current_price = df.iloc[-1]["close"]
        sw_low  = result["swing_low"]
        sw_high = result["swing_high"]
        if result["direction"] == "up":
            header = (
                f"상승 스윙: 저점 ${sw_low:,.0f} ({result['swing_low_ago']}봉 전) → "
                f"고점 ${sw_high:,.0f} ({result['swing_high_ago']}봉 전)"
            )
        else:
            header = (
                f"하락 스윙: 고점 ${sw_high:,.0f} ({result['swing_high_ago']}봉 전) → "
                f"저점 ${sw_low:,.0f} ({result['swing_low_ago']}봉 전)"
            )
        if overlap_note:
            header += f"\n  {overlap_note}"

        # 현재가가 피보 범위 밖에 있으면 레벨이 지지/저항으로 기능하지 않음
        if current_price < sw_low:
            return (
                f"{header}\n"
                f"  ⚠️ 현재가(${current_price:,.0f})가 스윙저점 아래 — 해당 레벨 유효하지 않음"
            )
        if current_price > sw_high:
            return (
                f"{header}\n"
                f"  ⚠️ 현재가(${current_price:,.0f})가 스윙고점 위 — 해당 레벨 유효하지 않음"
            )

        levels_str = "  ".join(f"Fib{r}=${p:,.0f}" for r, p in result["levels"].items())
        return f"{header}\n  {levels_str}"

    # 스윙 계산 (1h·4h 각각 한 번만 호출)
    _res_1h = fibonacci_swing_levels(multi_tf_data["1h"], window=fib_window_for_tf("1h")) if "1h" in multi_tf_data else None
    _res_4h = fibonacci_swing_levels(multi_tf_data["4h"], window=fib_window_for_tf("4h")) if "4h" in multi_tf_data else None

    # 1h·4h 스윙 구간 중복 탐지 — 두 TF가 동일 스윙을 잡으면 독립 확인 아님
    def _pct_close(a, b, tol=1.0):
        return abs(a - b) / max(abs(b), 1e-9) * 100 < tol

    _overlap_note = ""
    if _res_1h and _res_4h:
        if (_pct_close(_res_1h["swing_low"],  _res_4h["swing_low"])
                and _pct_close(_res_1h["swing_high"], _res_4h["swing_high"])):
            _overlap_note = "※ 1h·4h 동일 스윙 구간 탐지 — 두 레벨은 독립 확인 아님"

    fib_1h = fib_format(multi_tf_data["1h"], _res_1h, _overlap_note) if "1h" in multi_tf_data else "N/A"
    fib_4h = fib_format(multi_tf_data["4h"], _res_4h, _overlap_note) if "4h" in multi_tf_data else "N/A"

    # 최종 블록 조립 (기존 USER_PROMPT_TEMPLATE 의 내부 데이터 섹션과 동일 순서)
    indicators_summary = "\n\n".join(parts)
    context_blob = (
        f"{analysis_context_str}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{tf_alignment}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{macro_context_str}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{market_context_str}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{account_context_str}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{indicators_summary}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"[피보나치 레벨]\n"
        f"1h 기준: {fib_1h}\n"
        f"4h 기준: {fib_4h}"
    )
    if return_raw:
        return context_blob, {
            "macro": macro_payload,
            "market": market_ctx,
            "account": account_ctx,
            "analysis_context": analysis_ctx,
        }
    return context_blob


def build_prompt(
    multi_tf_data: dict,
    macro_snapshot: Optional[dict] = None,
    debate_block: str = "",
    delta_block: str = "",
    lessons_block: str = "",
) -> str:
    """
    최종 애널리스트 Claude 호출용 user prompt 를 조립한다.

    Parameters
    ----------
    multi_tf_data : dict
        {tf: DataFrame} 형태의 멀티 TF 인디케이터 데이터.
    macro_snapshot : dict, optional
        미리 수집된 거시 스냅샷. None 이면 fetch_macro_context() 수행.
    debate_block : str, optional
        agents.format_debate_block() 의 출력. 빈 문자열이면 토론 섹션 생략.
    delta_block : str, optional
        agents.delta_context.build_delta_block() 의 출력 — 직전 분석 대비 변화.
    lessons_block : str, optional
        agents.memory.format_lessons_block() 의 출력 — 과거 reflection 체크리스트.
    """
    context_blob = _build_context_blob(multi_tf_data, macro_snapshot)
    now_kst_label = now_kst().strftime("%Y-%m-%d %H:%M")

    def _sep(block: str) -> str:
        return "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n" if block else ""

    return USER_PROMPT_TEMPLATE.format(
        now_kst=now_kst_label,
        pair_label=PAIR_LABEL,
        context_blob=context_blob,
        delta_block_separator=_sep(delta_block),
        delta_block=delta_block,
        lessons_block_separator=_sep(lessons_block),
        lessons_block=lessons_block,
        debate_block_separator=_sep(debate_block),
        debate_block=debate_block,
    )


VIEW_TO_SIGNAL = {
    "상방 우위": "매수",
    "하방 우위": "매도",
    "중립": "홀드",
}

CONFIDENCE_BREAKDOWN_KEYS = (
    "price_structure", "momentum", "derivatives", "macro",
    "account_risk_fit", "data_quality_penalty", "counter_scenario_penalty",
)


def _strip_markdown_text(text: Any) -> str:
    """파싱 대상 라벨 주변의 가벼운 마크다운 문법을 제거한다."""
    cleaned = str(text or "")
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"(?m)^\s*(?:[-+*]\s+|>\s*|#{1,6}\s*)", "", cleaned)
    cleaned = cleaned.replace("`", "").replace("*", "")
    return cleaned


def _normalize_view_label(value: Any) -> str:
    """상방/하방/중립 계열 표현을 리포트 표준 관점으로 정규화."""
    cleaned = re.sub(r"\s+", " ", _strip_markdown_text(value)).strip()
    candidates: list[tuple[int, str]] = []
    for keyword, view in (
        ("상방", "상방 우위"),
        ("매수", "상방 우위"),
        ("하방", "하방 우위"),
        ("매도", "하방 우위"),
        ("중립", "중립"),
        ("홀드", "중립"),
    ):
        pos = cleaned.find(keyword)
        if pos >= 0:
            candidates.append((pos, view))
    if candidates:
        return min(candidates, key=lambda item: item[0])[1]
    return cleaned


REPORT_SECTION_LABELS = {
    "view": "관점",
    "regime": "시장 레짐",
    "facts": "먼저 보이는 사실",
    "interpretation": "해석",
    "counter_scenario": "반대 시나리오",
    "response": "대응",
    "invalidation": "관점이 약해지는 조건",
    "summary": "한줄 요약",
}


def parse_report_sections(text: str) -> dict:
    """애널리스트 리포트의 핵심 섹션을 구조적으로 파싱."""
    sections = {
        "view": None,
        "regime": None,
        "facts": [],
        "interpretation": [],
        "counter_scenario": [],
        "response": [],
        "invalidation": None,
        "summary": None,
    }

    current_block = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        m = re.match(r'📊\s*관점\s*[:：]\s*(.+)$', line)
        if m:
            sections["view"] = m.group(1).strip()
            current_block = None
            continue

        if re.match(r'💯\s*(?:확신도|신뢰도)\s*[:：]', line):
            current_block = None
            continue

        m = re.match(r'🧭\s*시장\s*레짐\s*[:：]\s*(.+)$', line)
        if m:
            sections["regime"] = m.group(1).strip()
            current_block = None
            continue

        if re.match(r'📌\s*먼저\s*보이는\s*사실', line):
            current_block = "facts"
            continue

        if re.match(r'🧠\s*해석', line):
            current_block = "interpretation"
            continue

        if re.match(r'🔄\s*반대\s*시나리오', line):
            current_block = "counter_scenario"
            continue

        if re.match(r'📍\s*관심\s*레벨', line):
            current_block = None
            continue

        if re.match(r'📝\s*대응', line):
            current_block = "response"
            continue

        m = re.match(r'⚠️\s*관점이\s*약해지는\s*조건\s*[:：]\s*(.+)$', line)
        if m:
            sections["invalidation"] = m.group(1).strip()
            current_block = None
            continue

        m = re.match(r'💬\s*한줄\s*요약\s*[:：]\s*(.+)$', line)
        if m:
            sections["summary"] = m.group(1).strip()
            current_block = None
            continue

        if current_block in ("facts", "interpretation", "counter_scenario", "response"):
            item = re.sub(r'^[•\-]\s*', '', line).strip()
            if item:
                sections[current_block].append(item)

    required_keys = (
        "view",
        "regime",
        "facts",
        "interpretation",
        "counter_scenario",
        "response",
        "invalidation",
        "summary",
    )
    missing_sections = []
    for key in required_keys:
        value = sections[key]
        if isinstance(value, list):
            if not value:
                missing_sections.append(REPORT_SECTION_LABELS[key])
        elif not value:
            missing_sections.append(REPORT_SECTION_LABELS[key])

    return {
        "sections": sections,
        "missing_sections": missing_sections,
        "format_ok": not missing_sections,
    }


def parse_signal(text: str) -> tuple[str, int]:
    # ── 관점/시그널 파싱: 새 포맷(관점) 우선, 구 포맷(시그널) 폴백 ──
    cleaned_text = _strip_markdown_text(text)
    signal = "홀드"
    sig_match = re.search(
        r'(?:관점|시그널)[^:：\n]*[:：]?\s*(상방(?:\s*우위)?|하방(?:\s*우위)?|중립|매수|매도|홀드)',
        cleaned_text,
    )
    if sig_match:
        raw_signal = _normalize_view_label(sig_match.group(1))
        signal = VIEW_TO_SIGNAL.get(raw_signal, raw_signal)
    else:
        front = cleaned_text[:300]
        keys = ("상방 우위", "하방 우위", "상방", "하방", "중립", "매수", "매도", "홀드")
        positions = {kw: front.find(kw) for kw in keys if kw in front}
        if positions:
            raw_signal = _normalize_view_label(min(positions, key=positions.get))
            signal = VIEW_TO_SIGNAL.get(raw_signal, raw_signal)

    # ── 확신도/신뢰도 파싱 ──
    # [버그 수정] 기존 [^:\n] 패턴은 콜론을 제외해 "신뢰도: 72%" 형식에서 매칭 실패
    # \D*? 로 변경 — 숫자가 아닌 모든 문자(콜론·공백 포함)를 lazily 건너뜀
    confidence = 50
    conf_match = re.search(r'(?:확신도|신뢰도)\D*?(\d{1,3})', cleaned_text)
    if conf_match:
        confidence = min(int(conf_match.group(1)), 100)

    return signal, confidence


def parse_leverage(text: str) -> Optional[int]:
    """
    Claude 분석 텍스트에서 권장 레버리지를 파싱.
    '권장 레버리지' 필드 우선, 없으면 자유 텍스트에서 탐색.
    반환: 1~10 범위 정수 or None
    """
    # 구조화 필드 우선 (매매 파라미터 섹션)
    m = re.search(r'권장\s*레버리지\s*[:：]\s*(\d+)\s*배', text)
    if m:
        val = int(m.group(1))
        if 1 <= val <= 10:
            return val

    # 자유 텍스트 폴백 패턴들
    patterns = [
        r'레버리지\s*[:：]\s*(\d+)\s*배',
        r'(\d+)\s*배\s*레버리지',
        r'leverage\s*[:：]?\s*(\d+)',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if 1 <= val <= 10:
                return val
    return None


def parse_trade_levels(text: str) -> dict:
    """관심 레벨 파싱. 구 포맷(진입/손절/목표/손익비)도 폴백 지원."""
    def _price_from_line(label: str):
        m = re.search(rf'^\s*[•\-]?\s*{label}\s*[:：]\s*(.+)$', text, re.MULTILINE)
        if not m:
            return None

        value_text = m.group(1).strip()
        if re.match(r'^N/?A\b', value_text, re.IGNORECASE):
            return None

        dollar_match = re.search(r'\$([\d,]+(?:\.\d+)?)', value_text)
        if dollar_match:
            val = dollar_match.group(1).replace(',', '').strip()
            try:
                return float(val)
            except ValueError:
                return None

        numeric_only_match = re.fullmatch(r'([\d,]+(?:\.\d+)?)', value_text)
        if not numeric_only_match:
            return None

        val = numeric_only_match.group(1).replace(',', '').strip()
        try:
            return float(val)
        except ValueError:
            return None

    resistance   = _price_from_line(r'1차\s*저항')
    support      = _price_from_line(r'1차\s*지지')
    bull_trigger = _price_from_line(r'상방\s*돌파\s*트리거')
    bear_trigger = _price_from_line(r'하방\s*이탈\s*트리거')

    entry  = _price_from_line(r'진입가')
    stop   = _price_from_line(r'손절가')
    target = _price_from_line(r'목표가')

    rr = None
    rr_m = re.search(r'손익비\s*[:：]\s*([\d.]+)\s*[:：]\s*1', text)
    if rr_m:
        try:
            rr = float(rr_m.group(1))
        except ValueError:
            pass

    return {
        "resistance": resistance if resistance is not None else target,
        "support": support if support is not None else stop,
        "bull_trigger": bull_trigger if bull_trigger is not None else entry,
        "bear_trigger": bear_trigger,
        "entry": entry,
        "stop": stop,
        "target": target,
        "rr": rr,
    }


def _strip_analysis_json_block(text: str) -> str:
    """사용자에게 보여줄 리포트에서 기계 판독용 JSON 블록을 제거."""
    if not text:
        return ""
    cleaned = re.sub(
        r'\s*<analysis_json>\s*(?:```(?:json)?\s*)?\{.*?\}\s*(?:```\s*)?</analysis_json>\s*',
        "",
        text,
        count=1,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return cleaned.strip()


def _extract_analysis_json(text: str) -> dict:
    """Claude 응답의 <analysis_json> 블록을 dict로 파싱. 실패하면 빈 dict."""
    if not text:
        return {}
    m = re.search(
        r'<analysis_json>\s*(?:```(?:json)?\s*)?(\{.*?\})\s*(?:```\s*)?</analysis_json>',
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not m:
        return {}

    payload = m.group(1).strip()
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        # 일부 모델이 마지막 쉼표를 남기는 경우만 가볍게 보정한다.
        payload = re.sub(r',\s*([}\]])', r'\1', payload)
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return {}
    return parsed if isinstance(parsed, dict) else {}


def _price_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        num = float(value)
        return num if num > 0 else None
    text = str(value).strip()
    if not text or re.match(r'^(?:n/?a|null|none|-)$', text, re.IGNORECASE):
        return None
    m = re.search(r'[-+]?\d[\d,]*(?:\.\d+)?', text)
    if not m:
        return None
    try:
        num = float(m.group(0).replace(",", ""))
    except ValueError:
        return None
    return num if num > 0 else None


def _int_or_none(value: Any, min_value: int, max_value: int) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        num = int(round(float(value)))
    else:
        m = re.search(r'-?\d+', str(value))
        if not m:
            return None
        num = int(m.group(0))
    if min_value <= num <= max_value:
        return num
    return None


def _signal_from_structured(analysis_json: dict) -> Optional[str]:
    view = _normalize_view_label(analysis_json.get("view") or "")
    for raw_view, signal in VIEW_TO_SIGNAL.items():
        if raw_view in view:
            return signal
    return None


def _levels_from_structured(analysis_json: dict) -> dict:
    levels = analysis_json.get("levels") if isinstance(analysis_json, dict) else None
    trade = analysis_json.get("trade") if isinstance(analysis_json, dict) else None
    levels = levels if isinstance(levels, dict) else {}
    trade = trade if isinstance(trade, dict) else {}

    return {
        "resistance": _price_or_none(levels.get("resistance")),
        "support": _price_or_none(levels.get("support")),
        "bull_trigger": _price_or_none(levels.get("bull_trigger")),
        "bear_trigger": _price_or_none(levels.get("bear_trigger")),
        "entry": _price_or_none(trade.get("entry")),
        "stop": _price_or_none(trade.get("stop")),
        "target": _price_or_none(trade.get("target")),
    }


def _confidence_from_breakdown(analysis_json: dict) -> Optional[int]:
    cb = analysis_json.get("confidence_breakdown") if isinstance(analysis_json, dict) else None
    if not isinstance(cb, dict):
        return None

    total = 0
    for key in CONFIDENCE_BREAKDOWN_KEYS:
        if key not in cb:
            return None
        try:
            total += int(cb[key])
        except (TypeError, ValueError):
            return None
    return max(0, min(100, total))


def _normalize_analysis_json(analysis_json: dict) -> tuple[dict, list[str]]:
    if not isinstance(analysis_json, dict):
        return {}, []

    normalized = dict(analysis_json)
    adjustments: list[str] = []

    breakdown_confidence = _confidence_from_breakdown(normalized)
    stated_confidence = _int_or_none(normalized.get("confidence"), 0, 100)
    if breakdown_confidence is not None and (
        stated_confidence is None or abs(stated_confidence - breakdown_confidence) > 2
    ):
        normalized["confidence"] = breakdown_confidence
        before = "N/A" if stated_confidence is None else str(stated_confidence)
        adjustments.append(
            f"confidence {before} -> {breakdown_confidence} "
            "because confidence_breakdown sum is authoritative"
        )

    return normalized, adjustments


def _format_report_price(value: Any) -> str:
    price = _price_or_none(value)
    if price is None:
        return "N/A"
    if abs(price - round(price)) < 0.005:
        return f"${price:,.0f}"
    return f"${price:,.2f}"


def _clean_report_text(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    return text if text else fallback


def _clean_report_items(value: Any, fallback: list[str], limit: int = 3) -> list[str]:
    if isinstance(value, list):
        items = [str(v).strip() for v in value if str(v or "").strip()]
    else:
        items = []
    return (items or fallback)[:limit]


def _render_report_from_structured(analysis_json: dict) -> str:
    """
    Tool use 응답이 구조화 JSON만 남기고 본문 리포트를 생략할 때,
    UI와 파서가 기대하는 표준 리포트를 서버에서 결정적으로 복원한다.
    """
    if not isinstance(analysis_json, dict) or not analysis_json:
        return ""

    view = _normalize_view_label(analysis_json.get("view") or "중립")
    if view not in VIEW_TO_SIGNAL:
        view = "중립"

    confidence = _confidence_from_breakdown(analysis_json)
    if confidence is None:
        confidence = _int_or_none(analysis_json.get("confidence"), 0, 100)
    if confidence is None:
        confidence = 50

    allowed_regimes = (
        "상승 추세", "하락 추세", "박스",
        "변동성 확장", "변동성 축소", "이벤트 대기 중",
    )
    regime_raw = str(analysis_json.get("regime") or "").strip()
    regime = regime_raw if regime_raw in allowed_regimes else "박스"

    levels = analysis_json.get("levels") if isinstance(analysis_json.get("levels"), dict) else {}
    trade = analysis_json.get("trade") if isinstance(analysis_json.get("trade"), dict) else {}
    actions = analysis_json.get("actions") if isinstance(analysis_json.get("actions"), dict) else {}

    leverage = _int_or_none(trade.get("leverage"), 1, 10) or 1

    facts = _clean_report_items(
        analysis_json.get("key_facts"),
        ["구조화 분석 결과는 있으나 본문 리포트가 누락되어 핵심 사실을 JSON 기준으로 복원했습니다."],
    )
    inferences = _clean_report_items(
        analysis_json.get("inferences"),
        ["현재 데이터의 방향성과 리스크 요인을 종합해 위 관점을 도출했습니다."],
    )
    counters = _clean_report_items(
        analysis_json.get("counter_scenario"),
        ["주요 트리거가 반대로 작동하면 현재 관점의 우위가 약해집니다."],
    )

    aggressive = _clean_report_text(
        actions.get("aggressive"),
        "현재 관점의 트리거와 손절 기준을 함께 두고 제한된 사이즈로 대응합니다.",
    )
    conservative = _clean_report_text(
        actions.get("conservative"),
        "트리거 확인 또는 되돌림 확인 전까지 사이즈를 줄이고 관망합니다.",
    )
    invalidation = _clean_report_text(
        analysis_json.get("invalidation"),
        "핵심 지지/저항 트리거가 반대로 확정되면 관점을 재검토합니다.",
    )
    summary = _clean_report_text(
        analysis_json.get("summary"),
        f"{view} 관점이나 트리거 확인과 리스크 관리가 우선입니다.",
    )

    lines = [
        f"📊 관점: {view}",
        f"💯 확신도: {confidence}%",
        f"🧭 시장 레짐: {regime}",
        "",
        "📌 먼저 보이는 사실",
    ]
    lines.extend(f"• {item}" for item in facts)
    lines.extend([
        "",
        "🧠 해석",
    ])
    lines.extend(f"• {item}" for item in inferences)
    lines.extend([
        "",
        "🔄 반대 시나리오",
    ])
    lines.extend(f"• {item}" for item in counters)
    lines.extend([
        "",
        "📍 관심 레벨",
        f"• 1차 저항: {_format_report_price(levels.get('resistance'))}",
        f"• 1차 지지: {_format_report_price(levels.get('support'))}",
        f"• 상방 돌파 트리거: {_format_report_price(levels.get('bull_trigger'))}",
        f"• 하방 이탈 트리거: {_format_report_price(levels.get('bear_trigger'))}",
        "",
        "🤖 매매 파라미터",
        f"• 진입가: {_format_report_price(trade.get('entry'))}",
        f"• 손절가: {_format_report_price(trade.get('stop'))}",
        f"• 목표가: {_format_report_price(trade.get('target'))}",
        f"• 권장 레버리지: {leverage}배",
        "",
        "📝 대응",
        f"• 공격적: {aggressive}",
        f"• 보수적: {conservative}",
        "",
        f"⚠️ 관점이 약해지는 조건: {invalidation}",
        "",
        f"💬 한줄 요약: {summary}",
    ])
    return "\n".join(lines)


def _system_prompt_param():
    """Anthropic prompt caching: 정적 system prompt를 cache breakpoint로 표시."""
    if not PROMPT_CACHE_ENABLED:
        return SYSTEM_PROMPT
    return [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def _analysis_tool_schema() -> dict:
    """
    ANALYST_USE_TOOL_SCHEMA=1 일 때 Claude 에 전달할 tool 정의.
    스키마 강제로 JSON parse 실패가 구조적으로 사라진다.
    """
    return {
        "name": "record_analysis",
        "description": (
            "현재 시장 분석의 구조화된 결과를 기록한다. "
            "관점·확신도·매매 파라미터 모두 본문 한국어 리포트와 부호·방향이 일치해야 한다."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "view": {
                    "type": "string",
                    "enum": ["상방 우위", "하방 우위", "중립"],
                },
                "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
                "regime": {
                    "type": "string",
                    "enum": [
                        "상승 추세", "하락 추세", "박스",
                        "변동성 확장", "변동성 축소", "이벤트 대기 중",
                    ],
                },
                "confidence_breakdown": {
                    "type": "object",
                    "properties": {
                        "price_structure": {"type": "integer", "minimum": 0, "maximum": 25},
                        "momentum": {"type": "integer", "minimum": 0, "maximum": 20},
                        "derivatives": {"type": "integer", "minimum": 0, "maximum": 20},
                        "macro": {"type": "integer", "minimum": 0, "maximum": 15},
                        "account_risk_fit": {"type": "integer", "minimum": 0, "maximum": 10},
                        "data_quality_penalty": {"type": "integer", "minimum": -15, "maximum": 0},
                        "counter_scenario_penalty": {"type": "integer", "minimum": -10, "maximum": 0},
                    },
                    "required": [
                        "price_structure", "momentum", "derivatives", "macro",
                        "account_risk_fit", "data_quality_penalty", "counter_scenario_penalty",
                    ],
                    "additionalProperties": False,
                },
                "data_quality_notes": {"type": "array", "items": {"type": "string"}},
                "key_facts": {"type": "array", "items": {"type": "string"}},
                "inferences": {"type": "array", "items": {"type": "string"}},
                "counter_scenario": {"type": "array", "items": {"type": "string"}},
                "levels": {
                    "type": "object",
                    "properties": {
                        "resistance":   {"type": ["number", "null"]},
                        "support":      {"type": ["number", "null"]},
                        "bull_trigger": {"type": ["number", "null"]},
                        "bear_trigger": {"type": ["number", "null"]},
                    },
                    "required": ["resistance", "support", "bull_trigger", "bear_trigger"],
                    "additionalProperties": False,
                },
                "trade": {
                    "type": "object",
                    "properties": {
                        "entry":    {"type": ["number", "null"]},
                        "stop":     {"type": ["number", "null"]},
                        "target":   {"type": ["number", "null"]},
                        "leverage": {"type": "integer", "minimum": 1, "maximum": 10},
                    },
                    "required": ["entry", "stop", "target", "leverage"],
                    "additionalProperties": False,
                },
                "actions": {
                    "type": "object",
                    "properties": {
                        "aggressive":   {"type": "string"},
                        "conservative": {"type": "string"},
                    },
                    "required": ["aggressive", "conservative"],
                    "additionalProperties": False,
                },
                "invalidation": {"type": "string"},
                "summary":      {"type": "string"},
            },
            "required": [
                "view", "confidence", "regime",
                "confidence_breakdown",
                "data_quality_notes", "key_facts", "inferences", "counter_scenario",
                "levels", "trade", "actions", "invalidation", "summary",
            ],
            "additionalProperties": False,
        },
    }


def analyze_with_claude(
    multi_tf_data: dict,
    macro_snapshot: Optional[dict] = None,
    debate: Optional[DebateResult] = None,
    pipeline: Optional[PipelineResult] = None,
    delta_block: str = "",
    lessons_block: str = "",
) -> dict:
    """
    최종 애널리스트 Claude 호출.

    토론/리스크/메모리 컨텍스트 주입 우선순위:
      1) pipeline (Phase 2+3 통합 블록, combined_block)
      2) debate   (Phase 1 단독 블록, 하위 호환)
      3) 없음

    delta_block / lessons_block 은 run_full_analysis 가 만들어 전달.
    """
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    if pipeline is not None and pipeline.combined_block:
        debate_block = pipeline.combined_block
    elif debate is not None:
        debate_block = format_debate_block(debate)
    else:
        debate_block = ""

    prompt = build_prompt(
        multi_tf_data,
        macro_snapshot=macro_snapshot,
        debate_block=debate_block,
        delta_block=delta_block,
        lessons_block=lessons_block,
    )
    # 실제 출력 구조: 약 10개 섹션 × 3~5줄 ≈ 600~1000 tokens.
    # 12000은 과도하며 디버깅용 대형 마진. ANALYST_MAX_TOKENS으로 조절 가능.
    # 기본값 4000: 충분한 여유 + 비용·속도 개선.
    _analyst_max_tokens = int(_os.getenv("ANALYST_MAX_TOKENS", "4000"))
    request_kwargs = {
        "model": CLAUDE_MODEL,
        "max_tokens": _analyst_max_tokens,
        "system": _system_prompt_param(),
        "messages": [{"role": "user", "content": prompt}],
    }

    # ── Tool Use opt-in (ANALYST_USE_TOOL_SCHEMA=1) ──
    # JSON 스키마를 tool 로 강제하면 regex parse 실패가 구조적으로 사라짐.
    # 본문 한국어 리포트는 동일 응답의 text 블록에서 함께 받음.
    _use_tool = _os.getenv("ANALYST_USE_TOOL_SCHEMA", "1").lower() not in ("0", "false", "no", "")
    if _use_tool:
        request_kwargs["tools"] = [_analysis_tool_schema()]
        # tool_choice 강제하지 않음. 모델이 본문을 생략하는 경우는
        # 아래에서 tool_json 기반 표준 리포트로 결정적으로 복원한다.

    # thinking 완전 비활성화 — 최종 분석은 이미 debate/judge/risk 블록이 reasoning을 제공하므로
    # adaptive thinking은 수만 토큰을 소모해 비용을 크게 높임. 구조화 출력에는 불필요.

    # 529/429 과부하 대비 지수 백오프 재시도 (최대 4회: 10s → 20s → 40s → 80s)
    max_retries = 4
    wait = 10
    message = None
    for attempt in range(max_retries):
        try:
            message = client.messages.create(**request_kwargs)
            break

        except anthropic.APIStatusError as e:
            if e.status_code == 400 and request_kwargs.get("system") != SYSTEM_PROMPT:
                # 모델/SDK 조합이 cache_control system block을 받지 못하면
                # 동일 프롬프트를 일반 system 문자열로 한 번 더 시도한다.
                request_kwargs["system"] = SYSTEM_PROMPT
                continue
            if e.status_code in (429, 529) and attempt < max_retries - 1:
                time.sleep(wait)
                wait *= 2  # 10 → 20 → 40 → 80초
            else:
                raise

    # 응답 타입 방어 검사 — SDK 버전이나 API 오류로 인해 예상 외 타입이 반환될 수 있음
    if message is None:
        raise RuntimeError("Anthropic API 응답 없음 (모든 재시도 소진)")
    if not hasattr(message, "content") or not isinstance(message.content, list):
        raise RuntimeError(
            f"Anthropic API 응답 형식 오류 — 타입: {type(message).__name__}, "
            f"content: {getattr(message, 'content', '(없음)')!r:.200}"
        )

    # 응답 블록에서 텍스트와 tool_use 추출 (thinking 블록 제외)
    raw_text = next(
        (b.text for b in message.content if getattr(b, "type", None) == "text"), ""
    )
    tool_json = None
    for b in message.content:
        if getattr(b, "type", None) == "tool_use" and getattr(b, "name", None) == "record_analysis":
            inp = getattr(b, "input", None)
            if isinstance(inp, dict):
                tool_json = inp
                break

    # Tool Use 가 활성화·성공이면 tool_json 우선, 아니면 regex 추출 폴백
    analysis_json = tool_json if tool_json is not None else _extract_analysis_json(raw_text)
    analysis_adjustments: list[str] = []
    if isinstance(analysis_json, dict) and analysis_json:
        analysis_json, analysis_adjustments = _normalize_analysis_json(analysis_json)
    report_text = _strip_analysis_json_block(raw_text) or raw_text
    report_meta = parse_report_sections(report_text)
    report_generated_from_json = False

    if isinstance(analysis_json, dict) and analysis_json and not report_meta["format_ok"]:
        rendered_report = _render_report_from_structured(analysis_json)
        rendered_meta = parse_report_sections(rendered_report)
        if rendered_report and rendered_meta["format_ok"]:
            report_text = rendered_report
            report_meta = rendered_meta
            report_generated_from_json = True

    signal, confidence = parse_signal(report_text)
    trade_levels = parse_trade_levels(report_text)
    claude_leverage = parse_leverage(report_text)

    structured_signal = _signal_from_structured(analysis_json)
    structured_confidence = _int_or_none(analysis_json.get("confidence"), 0, 100)
    structured_leverage = None
    if isinstance(analysis_json.get("trade"), dict):
        structured_leverage = _int_or_none(analysis_json["trade"].get("leverage"), 1, 10)

    if structured_signal:
        signal = structured_signal
    if structured_confidence is not None:
        confidence = structured_confidence
    if structured_leverage is not None:
        claude_leverage = structured_leverage

    structured_levels = _levels_from_structured(analysis_json)
    for key, value in structured_levels.items():
        if value is not None:
            trade_levels[key] = value

    # Judge 결과 추출 (signal processing 에 활용)
    judge_result = (
        pipeline.judge if pipeline is not None else None
    )

    # Trading Signal 구조화 (extract_trading_signal 이 가용할 때만)
    trading_signal_dict = None
    if extract_trading_signal is not None:
        try:
            canonical_view = {
                "매수": "상방 우위",
                "매도": "하방 우위",
                "홀드": "중립",
            }.get(signal, "중립")
            signal_source_text = (
                f"📊 관점: {canonical_view}\n"
                f"💯 확신도: {confidence}%\n"
                f"{report_text}"
            )
            ts = extract_trading_signal(signal_source_text, judge_result=judge_result)
            trading_signal_dict = ts.to_dict()
        except Exception as _ts_exc:
            _memory_logger.warning("extract_trading_signal 실패 — %s", _ts_exc)

    # ── 사후 일관성 검증 (deterministic — 항상 실행 / LLM 검증은 env 로 opt-in) ──
    consistency = None
    if check_consistency is not None and isinstance(analysis_json, dict) and analysis_json:
        try:
            consistency = check_consistency(analysis_json, report_text)
            if consistency and not consistency.get("ok"):
                _memory_logger.warning(
                    "Consistency check 실패(level=%s) — issues: %s",
                    consistency.get("level"),
                    consistency.get("issues"),
                )
        except Exception as _cc_exc:
            _memory_logger.warning("check_consistency 실패 — %s", _cc_exc)

    return {
        "signal":       signal,
        "confidence":   confidence,
        "raw_text":     report_text,
        "raw_response":  raw_text,
        "analysis_json": analysis_json,
        "trade_levels": trade_levels,
        "prompt_used":  prompt,
        "report_sections": report_meta["sections"],
        "report_format_ok": report_meta["format_ok"],
        "report_missing_sections": report_meta["missing_sections"],
        "report_generated_from_json": report_generated_from_json,
        "structured_output_used": tool_json is not None,
        "analysis_adjustments": analysis_adjustments,
        "trading_signal": trading_signal_dict,
        "claude_leverage": claude_leverage,
        "consistency":  consistency,
        "debate":       (
            pipeline.debate.to_payload() if pipeline is not None and pipeline.debate
            else (debate.to_payload() if debate is not None else None)
        ),
        "judge":        (
            pipeline.judge.to_payload() if pipeline is not None and pipeline.judge is not None
            else None
        ),
        "risk":         (
            pipeline.risk.to_payload() if pipeline is not None and pipeline.risk
            else None
        ),
        "memories":     (
            list(pipeline.memories) if pipeline is not None else []
        ),
    }


def run_full_analysis(
    multi_tf_data: dict,
    macro_snapshot: Optional[dict] = None,
    progress_cb=None,
) -> dict:
    """
    Bull/Bear 토론 + Risk Triad + 메모리 회상 + 최종 애널리스트 호출까지
    묶은 편의 함수. server.py 의 _run_job 에서 ThreadPoolExecutor 로 호출.

    Parameters
    ----------
    multi_tf_data : dict
        멀티 TF 캔들/지표 DataFrame.
    macro_snapshot : dict, optional
        이미 수집된 거시 스냅샷.
    progress_cb : callable, optional
        (phase, detail) -> None. 단계별 진행률 보고.
    """
    # 1) 공통 데이터 블록 + 원본 ctx 조립 (모든 에이전트가 이것을 본다)
    context_blob, raw_ctx = _build_context_blob(
        multi_tf_data, macro_snapshot, return_raw=True
    )

    # 1-a) BM25 매칭용 '정규화 태그' 생성 — 원본 blob 대신 이걸로 저장/검색
    situation_tags = ""
    if summarize_situation_tags is not None:
        try:
            situation_tags = summarize_situation_tags(
                multi_tf_data=multi_tf_data,
                macro_snapshot=raw_ctx.get("macro"),
                market_ctx=raw_ctx.get("market"),
                account_ctx=raw_ctx.get("account"),
            )
        except Exception as exc:
            _memory_logger.warning("situation_tags 생성 실패 — %s", exc)
            situation_tags = ""

    # 1-b) 분석 시점 현재가 추출 (reflection baseline)
    price_at_analysis: Optional[float] = None
    try:
        # 가장 짧은 TF 의 마지막 봉 close 를 분석 시점 현재가로 사용
        for tf in ("5m", "15m", "1h", "4h", "1d"):
            if tf in multi_tf_data and len(multi_tf_data[tf]) > 0:
                price_at_analysis = float(multi_tf_data[tf].iloc[-1]["close"])
                break
    except Exception as exc:
        _memory_logger.warning("price_at_analysis 추출 실패 — %s", exc)

    # 2) 메모리 객체 준비 (rank_bm25 미설치 시 None)
    memory_obj = None
    if get_memory is not None:
        try:
            memory_obj = get_memory("analyst")
        except Exception:
            memory_obj = None

    # 역할별 에이전트 메모리 (AgentMemories 싱글턴)
    agent_memories_obj = None
    if get_agent_memories is not None:
        try:
            agent_memories_obj = get_agent_memories()
        except Exception as exc:
            _memory_logger.warning("get_agent_memories 실패 — %s", exc)

    # 쿼리로는 정규화 태그를 쓰고, 태그가 없으면 blob 앞 200자만 사용
    # (context_blob 전체는 수천 토큰이어서 BM25 잡음이 심해짐)
    memory_query = situation_tags or context_blob[:200]

    # 3) 파이프라인 실행: Bull/Bear → Judge → Risk Triad → Memory
    pipeline = run_pipeline(
        context_blob=context_blob,
        pair_label=PAIR_LABEL,
        memory=memory_obj,
        current_situation=memory_query,
        progress_cb=progress_cb,
        agent_memories=agent_memories_obj,
        price_at_analysis=price_at_analysis,   # ← reflection baseline
    )

    # 3-a) 직전 분석 대비 변화 블록 (Δ context)
    delta_block_str = ""
    if build_delta_block is not None and memory_obj is not None:
        try:
            delta_block_str = build_delta_block(
                multi_tf_data=multi_tf_data,
                market_ctx=raw_ctx.get("market"),
                macro_snapshot=raw_ctx.get("macro"),
                current_situation_tags=situation_tags,
                memory_obj=memory_obj,
            )
        except Exception as exc:
            _memory_logger.warning("build_delta_block 실패 — %s", exc)

    # 3-b) 최근 reflection 체크리스트 surfacing
    lessons_block_str = ""
    if (
        format_lessons_block is not None
        and memory_obj is not None
        and hasattr(memory_obj, "extract_recent_checklists")
    ):
        try:
            checklists = memory_obj.extract_recent_checklists(max_records=12, max_lines=5)
            lessons_block_str = format_lessons_block(checklists)
        except Exception as exc:
            _memory_logger.warning("extract_recent_checklists 실패 — %s", exc)

    # 4) 최종 애널리스트 호출
    if progress_cb:
        progress_cb("final", "최종 애널리스트 종합 중")

    result = analyze_with_claude(
        multi_tf_data,
        macro_snapshot=macro_snapshot,
        pipeline=pipeline,
        delta_block=delta_block_str,
        lessons_block=lessons_block_str,
    )

    # 5) 메모리에 이번 상황-조언 페어 기록 (reflection 을 위한 씨앗)
    if memory_obj is not None and MEMORY_WRITE_ENABLED:
        try:
            situation_for_memory = situation_tags if situation_tags else context_blob
            # judge 판정도 메타에 기록
            judge_meta = {}
            if pipeline is not None and pipeline.judge is not None and pipeline.judge.enabled:
                judge_meta = {
                    "judge_verdict": pipeline.judge.verdict,
                    "judge_bull_key": pipeline.judge.bull_key,
                    "judge_bear_key": pipeline.judge.bear_key,
                    "judge_rubric_scores": getattr(pipeline.judge, "rubric_scores", {}),
                }
            analysis_ctx_meta = raw_ctx.get("analysis_context") or {}
            derived_meta = analysis_ctx_meta.get("derived") or {}
            derived_summary = {
                "higher_tf_bias": derived_meta.get("higher_tf_bias"),
                "lower_tf_bias": derived_meta.get("lower_tf_bias"),
                "conflicts": derived_meta.get("conflicts") or [],
                "timeframes": [
                    {
                        "tf": f.get("tf"),
                        "trend_bias": f.get("trend_bias"),
                        "structure": f.get("structure"),
                        "rsi_zone": f.get("rsi_zone"),
                        "atr_pct": f.get("atr_pct"),
                    }
                    for f in (derived_meta.get("timeframes") or [])
                    if isinstance(f, dict)
                ],
            }
            stored = memory_obj.add_situation(
                situation=situation_for_memory,
                advice=result.get("raw_text", ""),
                outcome="",
                meta={
                    "signal": result.get("signal"),
                    "confidence": result.get("confidence"),
                    "analysis_json": result.get("analysis_json") or {},
                    "confidence_breakdown": (
                        (result.get("analysis_json") or {}).get("confidence_breakdown") or {}
                    ),
                    "data_quality": analysis_ctx_meta.get("quality") or {},
                    "data_auditor": analysis_ctx_meta.get("auditor") or {},
                    "derived_features_summary": derived_summary,
                    "trade_levels": result.get("trade_levels"),
                    "trading_signal": result.get("trading_signal"),
                    "pair": PAIR_LABEL,
                    "price_at_analysis": price_at_analysis,
                    "situation_tags": situation_tags,
                    **judge_meta,
                },
            )
            if stored is None:
                _memory_logger.info("memory.add_situation: 최근 기록과 유사 — dedup skip")
        except Exception as exc:
            # 메모리 쓰기 실패는 조용히 무시 (분석 결과는 이미 나왔다)
            _memory_logger.warning("memory.add_situation 실패 — %s", exc)

    return result
