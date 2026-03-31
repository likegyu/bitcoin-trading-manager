# =============================================
# Claude API 연동 - 매매 시그널 분석
# =============================================
import re
import time
import anthropic
from datetime import datetime, timezone
from typing import Optional
from config import CLAUDE_API_KEY, CLAUDE_MODEL, DEFAULT_SYMBOL, symbol_to_pair
from indicators import summarize_indicators, fibonacci_swing_levels, fib_window_for_tf
from account_context import fetch_account_context, format_account_context
from market_context import fetch_market_context, format_market_context
from macro_fetcher import fetch_macro_context, format_macro_context

PAIR_LABEL = symbol_to_pair(DEFAULT_SYMBOL)

SYSTEM_PROMPT = (
    f"당신은 10년 경력의 {PAIR_LABEL} 선물 시장 애널리스트입니다.\n"
    "역할: 자동매매 엔진이 아니라, 정량 데이터와 시장 심리를 엮어 현재 구조를 해석하는 인간형 리서치 애널리스트.\n"
    f"전문 영역: {PAIR_LABEL} 선물 단기 분석 (수십 분~수 시간, 주로 15m·1h 기준 모멘텀 및 단기 추세 추종).\n"
    "리스크 성향: 중립적. 명확한 방향성이 없으면 중립 의견을 유지하고, 억지로 매매 아이디어를 만들지 않습니다.\n"
    "분석 철학: 단일 지표 신호보다 멀티 타임프레임 정렬, 가격 구조, 파생상품 심리, 거시 레짐의 정합성을 중시합니다.\n"
    "제공되는 데이터는 1d·4h·1h·15m·5m 캔들 + 거시·파생상품 심리 지표 + 계좌/포지션 제약 정보 + 최근 12시간·72시간·7일 계좌 운영 맥락입니다.\n"
    "\n"
    "애널리스트 원칙:\n"
    "1. 먼저 확정된 사실을 말하고, 그 다음 해석을 제시하세요.\n"
    "2. 한쪽 주장만 밀지 말고 반대 시나리오와 관점이 약해지는 조건을 반드시 적으세요.\n"
    "3. 지금 당장 거래를 강권하지 말고, 무엇을 기다려야 하는지도 함께 설명하세요.\n"
    "4. 5m는 진입 타이밍 힌트일 뿐, 큰 방향의 핵심 근거로 과대평가하지 마세요.\n"
    "5. 계좌/포지션 정보와 최근 운영 맥락은 시장 방향의 근거가 아니라 실행 제약입니다. 관점·레짐보다 대응/리스크 관리 문단에 우선 반영하세요.\n"
    "6. 최근 계좌 운영 맥락이 보이면 수익 보호 모드인지, 손실 복구 시도인지, 노출을 줄이는 중인지 같은 운영 상태를 읽되 단정하지 말고 관측된 사실에 기대어 표현하세요.\n"
    "\n"
    "확신도 산정 원칙:\n"
    "제공되는 모든 지표는 같은 가격 시계열에서 파생됩니다. "
    "RSI·MACD Hist 같은 모멘텀 오실레이터가 수렴해도, "
    "EMA9·SMA50·SMA200 배열이 정렬되어도, "
    "이들 각각을 '독립 근거'로 개수를 세어 확신도를 올리지 마세요. "
    "같은 가격 움직임을 다른 공식으로 반복 확인한 것일 뿐입니다. "
    "확신도는 지표 수렴 개수가 아니라, 가격 구조(지지·저항·추세 구조)와 지표 신호가 "
    "물리적으로 정합하는지를 기준으로 산정하세요. "
    "근거가 충분하지 않으면 낮은 확신도를 솔직하게 출력하는 것이 과신보다 정직한 판단입니다.\n"
    "\n"
    "파생상품 데이터 해석 주의:\n"
    "25d 풋-콜 스큐는 블랙-숄즈 근사값으로 계산됩니다. "
    "부호(양수=풋 프리미엄 우세/약세 편향, 음수=콜 프리미엄 우세/강세 편향)와 "
    "크기 수준(예: 소폭 vs. 뚜렷한 편향)만 참고하고, 정확한 수치에 의존하지 마세요.\n"
    "\n"
    "마크다운(**, ##, --- 등)과 HTML 태그는 절대 사용하지 마세요. 이모지와 일반 텍스트만 사용하세요."
)

USER_PROMPT_TEMPLATE = """분석 기준 시각: {now_utc} (UTC)
⚠️ 각 타임프레임의 마지막 캔들은 현재 형성 중인 미완성봉입니다. 확정된 신호로 해석하지 마세요.

다음은 {pair_label}의 현재 멀티 타임프레임 기술적 분석 데이터입니다.

{tf_alignment}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{macro_context}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{market_context}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{account_context}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{indicators_summary}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[피보나치 레벨]
1h 기준: {fib_1h}
4h 기준: {fib_4h}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
이 데이터를 바탕으로 지금 {pair_label}의 시장 관점을 애널리스트처럼 정리해주세요.
핵심은 지금 당장 매매를 강권하는 것이 아니라, 현재 구조를 해석하고 무엇을 기다려야 하는지 설명하는 것입니다.

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

📝 대응
• 공격적: [지금 할 행동 또는 관망 이유]
• 보수적: [기다릴 조건]

⚠️ 관점이 약해지는 조건: [1줄]

💬 한줄 요약: [전체를 한 문장으로]

중요:
- [시장 레짐]은 반드시 위 6개 중 정확히 1개만 쓰고, 괄호 설명이나 복수 선택을 하지 마세요.
- [관심 레벨]의 4개 항목은 각 줄마다 가격 숫자 또는 N/A만 적으세요. 이유, 조건, 괄호 설명을 붙이지 마세요.
- 2차 저항/지지 같은 추가 항목을 만들지 마세요.
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


def build_prompt(multi_tf_data: dict, macro_snapshot: Optional[dict] = None) -> str:
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
    try:
        market_context_str = format_market_context(fetch_market_context())
    except Exception as exc:
        market_context_str = f"[시장 심리 & 파생상품 데이터]\n  데이터 수집 실패 — {exc}"

    # 계좌 / 리스크 제약 수집
    account_context_str = "[계좌 / 리스크 제약]\n  데이터 수집 실패 — 계좌 제약 없이 시장 데이터만으로 판단"
    try:
        account_context_str = format_account_context(fetch_account_context())
    except Exception as exc:
        account_context_str = f"[계좌 / 리스크 제약]\n  데이터 수집 실패 — {exc}"

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

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")

    return USER_PROMPT_TEMPLATE.format(
        now_utc=now_utc,
        pair_label=PAIR_LABEL,
        tf_alignment=tf_alignment,
        macro_context=macro_context_str,
        market_context=market_context_str,
        account_context=account_context_str,
        indicators_summary="\n\n".join(parts),
        fib_1h=fib_1h,
        fib_4h=fib_4h,
    )


VIEW_TO_SIGNAL = {
    "상방 우위": "매수",
    "하방 우위": "매도",
    "중립": "홀드",
}

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
    signal = "홀드"
    sig_match = re.search(
        r'(?:관점|시그널)[^:：\n]*[:：]?\s*(상방 우위|하방 우위|중립|매수|매도|홀드)',
        text,
    )
    if sig_match:
        raw_signal = sig_match.group(1)
        signal = VIEW_TO_SIGNAL.get(raw_signal, raw_signal)
    else:
        front = text[:300]
        keys = ("상방 우위", "하방 우위", "중립", "매수", "매도", "홀드")
        positions = {kw: front.find(kw) for kw in keys if kw in front}
        if positions:
            raw_signal = min(positions, key=positions.get)
            signal = VIEW_TO_SIGNAL.get(raw_signal, raw_signal)

    # ── 확신도/신뢰도 파싱 ──
    # [버그 수정] 기존 [^:\n] 패턴은 콜론을 제외해 "신뢰도: 72%" 형식에서 매칭 실패
    # \D*? 로 변경 — 숫자가 아닌 모든 문자(콜론·공백 포함)를 lazily 건너뜀
    confidence = 50
    conf_match = re.search(r'(?:확신도|신뢰도)\D*?(\d{1,3})', text)
    if conf_match:
        confidence = min(int(conf_match.group(1)), 100)

    return signal, confidence


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


def analyze_with_claude(multi_tf_data: dict, macro_snapshot: Optional[dict] = None) -> dict:
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    prompt = build_prompt(multi_tf_data, macro_snapshot=macro_snapshot)
    request_kwargs = {
        "model": CLAUDE_MODEL,
        "max_tokens": 16000,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
    }

    # Claude 4.6 계열은 adaptive thinking을, 그 외 thinking 지원 모델은 수동 예산 방식을 사용한다.
    if (
        CLAUDE_MODEL.startswith("claude-opus-4-6")
        or CLAUDE_MODEL.startswith("claude-sonnet-4-6")
    ):
        request_kwargs["thinking"] = {"type": "adaptive"}
    elif (
        CLAUDE_MODEL.startswith("claude-haiku-4-5")
        or CLAUDE_MODEL.startswith("claude-sonnet-4-5")
        or CLAUDE_MODEL.startswith("claude-opus-4-5")
        or CLAUDE_MODEL.startswith("claude-opus-4-1")
        or CLAUDE_MODEL.startswith("claude-opus-4")
        or CLAUDE_MODEL.startswith("claude-sonnet-4")
        or CLAUDE_MODEL.startswith("claude-3-7-sonnet")
    ):
        request_kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": 4096,
        }

    # 529 과부하 대비 지수 백오프 재시도 (최대 4회: 10s → 20s → 40s → 80s)
    max_retries = 4
    wait = 10
    for attempt in range(max_retries):
        try:
            message = client.messages.create(**request_kwargs)
            break

        except anthropic.APIStatusError as e:
            if e.status_code == 529 and attempt < max_retries - 1:
                time.sleep(wait)
                wait *= 2  # 10 → 20 → 40 → 80초
            else:
                raise

    # 응답 블록에서 텍스트만 추출 (thinking 블록 제외)
    raw_text = next(
        (b.text for b in message.content if b.type == "text"), ""
    )
    signal, confidence = parse_signal(raw_text)
    trade_levels = parse_trade_levels(raw_text)
    report_meta = parse_report_sections(raw_text)

    return {
        "signal":       signal,
        "confidence":   confidence,
        "raw_text":     raw_text,
        "trade_levels": trade_levels,
        "prompt_used":  prompt,
        "report_sections": report_meta["sections"],
        "report_format_ok": report_meta["format_ok"],
        "report_missing_sections": report_meta["missing_sections"],
    }


def chat_with_claude(messages: list, context: str) -> str:
    """
    분석 결과를 컨텍스트로 유지하며 멀티턴 채팅.

    Parameters
    ----------
    messages : [{"role": "user"|"assistant", "content": str}, ...]
    context  : 원본 분석 데이터 + Claude 응답 (세션 컨텍스트)
    """
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    system = (
        f"당신은 10년 경력의 {PAIR_LABEL} 시장 애널리스트입니다.\n"
        f"아래는 방금 수행한 {PAIR_LABEL} 분석의 원본 데이터와 애널리스트 리포트입니다.\n"
        "사용자의 추가 질문에 이 컨텍스트를 바탕으로 간결하고 명확하게 답하세요.\n"
        "사실과 해석을 구분하고, 필요하면 반대 시나리오와 관점 무효화 조건도 함께 설명하세요.\n"
        "마크다운(**,##,---), HTML 태그는 절대 사용하지 마세요. 이모지와 일반 텍스트만 사용하세요.\n\n"
        f"[분석 컨텍스트]\n{context}"
    )

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=3000,
        system=system,
        messages=messages,
    )
    return response.content[0].text
