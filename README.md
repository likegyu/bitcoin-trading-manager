# ₿ BTC Signal Analyzer

AI 기반 비트코인 트레이딩 신호 분석 대시보드.
Binance 실시간 선물 데이터 + Claude AI 분석 + 거시경제 지표를 하나의 웹 인터페이스로 제공합니다.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green?logo=fastapi)
![Claude](https://img.shields.io/badge/Claude-Anthropic-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## 주요 기능

### 📈 실시간 시장 데이터
- Binance 선물 WebSocket 연결 (aggTrade + kline 스트림)
- 다중 타임프레임 지원: 15m · 1h · 4h · 1d
- ECharts 기반 캔들스틱 차트 (볼린저 밴드, EMA, SMA 오버레이)
- RSI · MACD · 거래량 서브차트 실시간 업데이트

### 🤖 Claude AI 분석
- 멀티 타임프레임 기술적 지표를 종합한 Claude 분석 리포트
- 롱/숏/관망 신호 + 신뢰도 점수 출력
- 진입가 · 목표가 · 손절가 자동 산출
- AI와 후속 질문 채팅 가능 (분석 컨텍스트 공유)

### 🌍 거시경제 지표
매크로 환경이 BTC에 미치는 영향을 실시간 카드로 표시합니다.

| 지표 | 소스 | 설명 |
|------|------|------|
| 10Y 실질금리 (DFII10) | FRED | 금리 상승 → BTC 부정적 |
| 2Y 국채금리 (DGS2) | FRED | 단기 금리 환경 |
| 달러 인덱스 (DTWEXBGS) | FRED | 달러 강세 → BTC 부정적 |
| 스테이블코인 시총 | DefiLlama | 유동성 공급 지표 |
| USDT 도미넌스 | DefiLlama | 리스크 온/오프 심리 |
| BTC 도미넌스 | CoinGecko | 알트코인 자금 유입 여부 |

각 지표는 **5일 변화량**, **20일 Z-스코어**, **레짐(상승/하락/횡보)** 을 함께 표시합니다.

### 💼 바이낸스 계좌 연동 (선택)
- 선물 계좌 잔고 · 포지션 실시간 조회
- 진입가 대비 현재 손익 표시

### 🔑 웹 기반 API 키 설정
- 최초 실행 시 키 입력 모달 자동 표시
- Claude API 키 (필수) / Binance API (선택) / FRED API (선택)
- 입력값은 로컬 `.env` 파일에만 저장 — 서버 재시작 없이 반영

---

## 시작하기

### 1. 저장소 클론

```bash
git clone https://github.com/likegyu/bitcoin-trading-manager.git
cd bitcoin-trading-manager/crypto_analyzer
```

### 2. API 키 준비

| 키 | 필수 여부 | 발급 주소 |
|----|----------|----------|
| Claude API Key | ✅ 필수 | [console.anthropic.com](https://console.anthropic.com) |
| Binance API Key + Secret | ✅ 필수 | [binance.com → API 관리](https://www.binance.com/ko/my/settings/api-management) |
| FRED API Key | ✅ 필수 | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) (무료) |

> **Binance API 권한 설정:** 읽기 전용 + **선물 거래(Enable Futures)** 권한이 필요합니다. 출금(Withdrawal) 권한은 절대 부여하지 마세요.

### 3. 서버 실행

```bash
./run.sh
```

`run.sh`가 패키지 설치 → `.env` 확인 → 서버 기동을 자동으로 처리합니다.
브라우저에서 **http://localhost:8000** 접속.

### 4. API 키 입력 (선택적)

`.env` 파일이 없거나 Claude 키가 없으면 시작 시 설정 모달이 자동으로 뜹니다.
직접 입력하거나 `.env.example`을 복사해 사용할 수 있습니다.

```bash
cp .env.example .env
# 열어서 값 입력
```

---

## 프로젝트 구조

```
crypto_analyzer/
├── server.py            # FastAPI 메인 서버 (SSE 스트리밍, API 라우트)
├── analyzer.py          # Claude 프롬프트 빌드 & 응답 파싱
├── macro_fetcher.py     # 거시경제 지표 수집 (FRED / DefiLlama / CoinGecko)
├── market_context.py    # 시장 컨텍스트 문자열 생성
├── indicators.py        # 기술적 지표 계산 (RSI, MACD, BB, 피보나치 등)
├── data_fetcher.py      # Binance REST API OHLCV 수집
├── account_context.py   # Binance 계좌 WebSocket 스트림
├── config.py            # 환경 변수 로드
├── static/
│   └── index.html       # 단일 파일 프론트엔드 (ECharts + Vanilla JS)
├── run.sh               # 설치 & 실행 스크립트
├── requirements.txt     # Python 의존성
├── .env.example         # API 키 템플릿
└── .gitignore
```

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| 백엔드 | FastAPI · Uvicorn · Python 3.9+ |
| 실시간 통신 | WebSocket (Binance) · SSE (Server-Sent Events) |
| AI | Anthropic Claude API |
| 데이터 | Binance Futures REST/WS · FRED API · DefiLlama · CoinGecko |
| 차트 | Apache ECharts 5 |
| 프론트엔드 | Vanilla JS · HTML/CSS (단일 파일, 빌드 도구 없음) |

---

## 지표 설명

### 기술적 지표
- **RSI(14)**: 과매수(>70) / 과매도(<30) 구분
- **MACD(12,26,9)**: 모멘텀 방향 및 히스토그램 추이
- **볼린저 밴드(20,2)**: 밴드 %B로 가격 위치 정규화
- **SMA 50 / 200**: 중장기 추세 판단
- **EMA 9**: 단기 모멘텀
- **ATR(14)**: 변동성 측정, 손절 거리 산정
- **피보나치 스윙**: 스윙 고저 기반 되돌림 레벨 자동 계산

### 거시경제 지표 해석
- **금리 상승 + 달러 강세** → 위험자산 매도 압력
- **스테이블코인 시총 증가 + USDT 도미넌스 하락** → 크립토 매수세 유입
- **BTC 도미넌스 상승** → 알트코인보다 BTC 선호 (리스크 오프 내 상대 강세)

---

## 주의사항

- **투자 조언이 아닙니다.** 본 프로젝트는 데이터 분석 도구이며, 매매 결과에 대한 책임은 사용자에게 있습니다.
- API 키는 절대 외부에 공유하지 마세요. `.env` 파일은 `.gitignore`에 포함되어 있습니다.
- Binance API 키 생성 시 **읽기 전용 + 선물 거래(Enable Futures) 권한만** 부여하세요. 출금(Withdrawal) 권한은 절대 추가하지 마세요.

---

## License

MIT
