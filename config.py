# =============================================
# Crypto Trading Signal Analyzer - Config
# =============================================
# ⚠️ 보안 주의: API 키는 .env 파일에서 관리합니다.
#    .gitignore에 .env를 반드시 추가하세요!

import os
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드 (실행 위치와 무관하게 현재 파일 기준으로 탐색)
_BASE_DIR = Path(__file__).resolve().parent
load_dotenv(_BASE_DIR / ".env")
load_dotenv()

CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "")
FRED_API_KEY   = os.getenv("FRED_API_KEY", "")
CLAUDE_MODEL   = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

BINANCE_BASE_URL    = "https://api.binance.com"
BINANCE_FUTURES_URL = "https://fapi.binance.com"
BINANCE_API_KEY     = os.getenv("BINANCE_API_KEY", "")
BINANCE_SECRET_KEY  = os.getenv("BINANCE_SECRET_KEY", "")
DEFAULT_SYMBOL      = os.getenv("DEFAULT_SYMBOL", "BTCUSDC").upper()


def symbol_to_pair(symbol: str) -> str:
    symbol = (symbol or "").upper()
    quote_candidates = ("USDC", "USDT", "FDUSD", "BUSD", "TUSD", "USD", "BTC", "ETH", "BNB")
    for quote in quote_candidates:
        if symbol.endswith(quote) and len(symbol) > len(quote):
            return f"{symbol[:-len(quote)]}/{quote}"
    return symbol

# ── 매매 설정 (복리 전략 파라미터) ──────────────
DEFAULT_LEVERAGE      = 3       # 희망 레버리지 배수

# ── 자동매매 설정 ──────────────────────────────
# 모든 값은 .env 에서 환경변수로 오버라이드 가능
AUTO_TRADE_ENABLED        = os.getenv("AUTO_TRADE_ENABLED", "0") not in ("0", "false", "no")
AUTO_TRADE_DRY_RUN        = os.getenv("AUTO_TRADE_DRY_RUN", "1") not in ("0", "false", "no")
AUTO_TRADE_MIN_CONFIDENCE = int(os.getenv("AUTO_TRADE_MIN_CONFIDENCE", "65"))   # 최소 확신도
AUTO_TRADE_MIN_STRENGTH   = int(os.getenv("AUTO_TRADE_MIN_STRENGTH",   "2"))    # 최소 강도 (BUY=2, STRONG_BUY=3)
AUTO_TRADE_RISK_PCT       = float(os.getenv("AUTO_TRADE_RISK_PCT",     "0.02")) # 진입당 리스크 비율 (2%)
AUTO_TRADE_SL_ATR_MULT    = float(os.getenv("AUTO_TRADE_SL_ATR_MULT",  "1.5")) # 손절 = ATR × 배수
AUTO_TRADE_TP_RR          = float(os.getenv("AUTO_TRADE_TP_RR",        "2.0")) # 익절 = 손절폭 × R:R
AUTO_TRADE_MAX_LEVERAGE   = int(os.getenv("AUTO_TRADE_MAX_LEVERAGE",   "5"))    # 레버리지 상한
AUTO_TRADE_COOLDOWN_MIN   = int(os.getenv("AUTO_TRADE_COOLDOWN_MIN",   "30"))   # 연속 거래 쿨다운(분)
AUTO_TRADE_FLIP_GUARD     = os.getenv("AUTO_TRADE_FLIP_GUARD", "1") not in ("0", "false", "no")  # 즉시 반전 방지

# 분석할 시간봉 목록
TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]

# 각 시간봉별 로드할 캔들 수
CANDLE_LIMIT = 200

# 자동 갱신 기본 간격 (초)
AUTO_REFRESH_INTERVAL = 60

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
