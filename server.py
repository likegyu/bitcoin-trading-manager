#!/usr/bin/env python3
from __future__ import annotations

# =============================================
# BTC Signal Analyzer — FastAPI backend
# =============================================
import asyncio
import copy
import concurrent.futures
import contextlib
import functools
import json
import math
import os
import time
import uuid

import pandas as pd
import websockets
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

import config as runtime_config
from config import CANDLE_LIMIT, DEFAULT_SYMBOL, TIMEFRAMES, symbol_to_pair
from account_context import (
    close_user_data_stream,
    fetch_account_context,
    keepalive_user_data_stream,
    open_user_data_stream,
)
from data_fetcher import fetch_ohlcv, fetch_current_price
from indicators import add_all_indicators, fibonacci_swing_levels, fib_window_for_tf
from analyzer import analyze_with_claude, run_full_analysis
from macro_fetcher import fetch_macro_context
from time_utils import format_kst, now_kst

# ── 자동매매 (선택적 임포트 — 모듈 없어도 서버 동작) ──
try:
    import auto_trader as _auto_trader
    _AUTO_TRADER_AVAILABLE = True
except Exception as _at_exc:
    _auto_trader = None  # type: ignore
    _AUTO_TRADER_AVAILABLE = False
    import logging as _at_log
    _at_log.getLogger(__name__).warning(
        "auto_trader 로드 실패 — %s: %s", type(_at_exc).__name__, _at_exc
    )

# ── Reflection / Memory (optional: rank_bm25 미설치 시 None) ──
try:
    from agents import get_memory as _get_memory
    from agents import get_agent_memories as _get_agent_memories
    from agents import reflect_on_record as _reflect_on_record
    from agents import reflect_for_role as _reflect_for_role
    _REFLECTION_ENABLED = (
        _get_memory is not None
        and _reflect_on_record is not None
        and _reflect_for_role is not None
    )
except Exception as _reflect_exc:  # pragma: no cover
    _get_memory = None             # type: ignore
    _get_agent_memories = None     # type: ignore
    _reflect_on_record = None      # type: ignore
    _reflect_for_role = None       # type: ignore
    _REFLECTION_ENABLED = False
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "Reflection/Memory 비활성 — %s: %s",
        type(_reflect_exc).__name__, _reflect_exc,
    )

# 에이전트 메모리 역할 목록 (analyst 는 별도 memory 이름 사용)
_AGENT_ROLES_FOR_REFLECT = ("bull", "bear", "judge", "aggressive", "conservative", "neutral")

# ── 색상 팔레트 ────────────────────────────────
C = {
    "bg":      "#FAF8F5",
    "surface": "#FFFFFF",
    "border":  "#E8DDD4",
    "txt":     "#1C1917",
    "dim":     "#78716C",
    "muted":   "#A8978A",
    "orange":  "#EA580C",
    "amber":   "#D97706",
    "bull":    "#16A34A",
    "bear":    "#DC2626",
    "sky":     "#0284C7",
    "violet":  "#7C3AED",
    "teal":    "#0D9488",
    "plot_bg": "#FDFBF8",
    "grid":    "#EDE8E0",
}
FIB_COLORS = {
    "0.000": "#DC2626", "0.236": "#EA580C", "0.382": "#D97706",
    "0.500": "#78716C", "0.618": "#16A34A", "0.786": "#0284C7",
    "1.000": "#16A34A",
}

app = FastAPI()
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
CHART_WINDOW = 120
BASE_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
ACCOUNT_STREAM_WS_BASE = "wss://fstream.binance.com/ws"
ACCOUNT_STREAM_KEEPALIVE_SECS = 50 * 60
ACCOUNT_REFRESH_DEBOUNCE_SECS = 0.35
ACCOUNT_REFRESH_MIN_INTERVAL_SECS = 1.5
MARKET_STREAM_WS_BASE = "wss://fstream.binance.com/stream"
MACRO_REFRESH_SECS = 60 * 60
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LATEST_ANALYSIS_PATH   = os.path.join(BASE_DIR, "data", "latest_analysis.json")
ANALYSIS_HISTORY_PATH  = os.path.join(BASE_DIR, "data", "analysis_history.jsonl")
ANALYSIS_HISTORY_MAX   = 500   # JSONL 최대 보관 건수


# ══════════════════════════════════════════════
# ECharts용 차트 데이터 직렬화
# ══════════════════════════════════════════════
def _series_values(values) -> list:
    return [_safe(v) for v in values]


def _series_labels(index) -> list[str]:
    labels = []
    for ts in index:
        try:
            labels.append(format_kst(ts, "%Y-%m-%d %H:%M"))
        except Exception:
            labels.append(str(ts)[:16])
    return labels


def _now_label() -> str:
    return now_kst().strftime("%H:%M:%S")


def _now_iso() -> str:
    return now_kst().isoformat(timespec="seconds")


def _load_latest_analysis() -> dict | None:
    try:
        with open(LATEST_ANALYSIS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except FileNotFoundError:
        return None
    except Exception as exc:
        print(f"[analysis-cache] load failed: {exc}")
        return None


def _persist_latest_analysis(payload: dict):
    try:
        os.makedirs(os.path.dirname(LATEST_ANALYSIS_PATH), exist_ok=True)
        tmp_path = f"{LATEST_ANALYSIS_PATH}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp_path, LATEST_ANALYSIS_PATH)
    except Exception as exc:
        print(f"[analysis-cache] persist failed: {exc}")


def _persist_analysis_history(payload: dict):
    """분석 결과 핵심 필드를 JSONL 히스토리 파일에 누적 저장."""
    try:
        os.makedirs(os.path.dirname(ANALYSIS_HISTORY_PATH), exist_ok=True)
        sections = payload.get("report_sections") or {}
        entry = {
            "timestamp":   _now_iso(),
            "signal":      payload.get("signal"),
            "confidence":  payload.get("confidence"),
            "price":       payload.get("price"),
            "pair_label":  payload.get("pair_label"),
            "regime":      sections.get("regime"),
            "summary":     sections.get("summary"),
            "trade_levels": payload.get("trade_levels"),
        }
        # 추가
        with open(ANALYSIS_HISTORY_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # 최대 건수 초과 시 앞쪽 줄 잘라내기
        with open(ANALYSIS_HISTORY_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) > ANALYSIS_HISTORY_MAX:
            with open(ANALYSIS_HISTORY_PATH, "w", encoding="utf-8") as f:
                f.writelines(lines[-ANALYSIS_HISTORY_MAX:])
    except Exception as exc:
        print(f"[analysis-history] persist failed: {exc}")


def build_chart_payload(df, fib: dict | None) -> dict:
    plot_df = df.tail(CHART_WINDOW).copy()
    labels = _series_labels(plot_df.index)

    candles = []
    volume_colors = []
    macd_hist_colors = []
    for row in plot_df.itertuples():
        open_v = _safe(row.open)
        close_v = _safe(row.close)
        low_v = _safe(row.low)
        high_v = _safe(row.high)
        hist_v = _safe(row.macd_hist)

        candles.append([open_v, close_v, low_v, high_v])

        is_bull = (
            open_v is not None and close_v is not None and close_v >= open_v
        )
        volume_colors.append(
            "rgba(22,163,74,.45)" if is_bull else "rgba(220,38,38,.45)"
        )
        macd_hist_colors.append(
            "rgba(22,163,74,.65)"
            if hist_v is not None and hist_v >= 0
            else "rgba(220,38,38,.65)"
        )

    active_levels = fib["levels"] if fib and fib.get("is_active") else {}
    fib_lines = [
        {
            "ratio": ratio,
            "price": round(float(price), 2),
            "color": FIB_COLORS.get(ratio, C["dim"]),
        }
        for ratio, price in active_levels.items()
    ]

    return {
        "labels": labels,
        "candles": candles,
        "bb_upper": _series_values(plot_df["bb_upper"]),
        "bb_lower": _series_values(plot_df["bb_lower"]),
        "sma_50": _series_values(plot_df["sma_50"]),
        "sma_200": _series_values(plot_df["sma_200"]),
        "ema_9": _series_values(plot_df["ema_9"]),
        "rsi": _series_values(plot_df["rsi"]),
        "macd": _series_values(plot_df["macd"]),
        "macd_signal": _series_values(plot_df["macd_signal"]),
        "macd_hist": _series_values(plot_df["macd_hist"]),
        "macd_hist_colors": macd_hist_colors,
        "volume": _series_values(plot_df["volume"]),
        "volume_ma": _series_values(plot_df["volume_ma"]),
        "volume_colors": volume_colors,
        "fib": fib,
        "fib_lines": fib_lines,
    }


# ══════════════════════════════════════════════
# 헬퍼
# ══════════════════════════════════════════════
def _safe(v):
    """NaN/None → None (JSON-safe float)."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return None


def _fetch_all(symbol: str) -> dict:
    result = {}
    for tf in TIMEFRAMES:
        df = fetch_ohlcv(symbol, tf)
        df = add_all_indicators(df)
        result[tf] = df
    return result


def _upsert_ohlcv(df, timestamp, row: dict) -> pd.DataFrame:
    if df is None or df.empty:
        base_df = pd.DataFrame(columns=BASE_OHLCV_COLUMNS)
    else:
        base_df = df[BASE_OHLCV_COLUMNS].copy()

    base_df = base_df.astype(float, copy=False)
    base_df.loc[timestamp, BASE_OHLCV_COLUMNS] = [
        row["open"], row["high"], row["low"], row["close"], row["volume"]
    ]
    base_df.sort_index(inplace=True)
    base_df = base_df[~base_df.index.duplicated(keep="last")]
    return base_df.tail(CANDLE_LIMIT)


def _serialize_swing_fib(tf: str, df) -> dict | None:
    result = fibonacci_swing_levels(df, window=fib_window_for_tf(tf))
    if result is None:
        return None

    current_price = _safe(df.iloc[-1]["close"]) if len(df.index) else None
    swing_low = round(float(result["swing_low"]), 2)
    swing_high = round(float(result["swing_high"]), 2)
    is_active = (
        current_price is not None and swing_low <= current_price <= swing_high
    )
    invalid_reason = None
    if current_price is not None:
        if current_price < swing_low:
            invalid_reason = "below_swing_low"
        elif current_price > swing_high:
            invalid_reason = "above_swing_high"

    return {
        "source_tf": tf,
        "direction": result["direction"],
        "direction_label": "상승 스윙" if result["direction"] == "up" else "하락 스윙",
        "swing_low": swing_low,
        "swing_low_ago": int(result["swing_low_ago"]),
        "swing_high": swing_high,
        "swing_high_ago": int(result["swing_high_ago"]),
        "leg_start": round(float(result["leg_start"]), 2),
        "leg_start_ago": int(result["leg_start_ago"]),
        "leg_start_type": result["leg_start_type"],
        "leg_end": round(float(result["leg_end"]), 2),
        "leg_end_ago": int(result["leg_end_ago"]),
        "leg_end_type": result["leg_end_type"],
        "current_price": current_price,
        "is_active": is_active,
        "invalid_reason": invalid_reason,
        "anchors": {
            ratio: round(float(level), 2)
            for ratio, level in result.get("anchors", {}).items()
        },
        "levels": {
            ratio: round(float(level), 2)
            for ratio, level in result["levels"].items()
        },
        "display_levels": {
            ratio: round(float(level), 2)
            for ratio, level in result.get("display_levels", result["levels"]).items()
        },
    }


def _build_swing_fibs(tf_data: dict, target_tfs: list[str]) -> dict:
    fibs = {}
    for tf in target_tfs:
        if tf in tf_data:
            fibs[tf] = _serialize_swing_fib(tf, tf_data[tf])
    return fibs


def _market_overview(tf_data: dict, price: float, last_update: str | None) -> dict:
    h1 = tf_data.get("1h")
    if h1 is None:
        h1 = tf_data.get(TIMEFRAMES[0]) if TIMEFRAMES else None

    if h1 is not None and not h1.empty:
        last = h1.iloc[-1]
        close_v = _safe(last["close"]) or 1.0
        indicators = {
            "rsi":       _safe(last["rsi"]),
            "macd_hist": _safe(last["macd_hist"]),
            "bb_pct":    _safe(last["bb_pct"]),
            "vol_ratio": round(_safe(last["volume"]) / max(_safe(last["volume_ma"]) or 1, 1) * 100, 1),
            "atr":       _safe(last["atr"]),
            "close":     _safe(last["close"]),
            "atr_pct":   round((_safe(last["atr"]) or 0) / close_v * 100, 2),
        }
    else:
        # tf_data 아직 없음 — WebSocket 스트림이 채울 때까지 빈 indicators
        indicators = {
            "rsi": None, "macd_hist": None, "bb_pct": None,
            "vol_ratio": None, "atr": None, "close": None, "atr_pct": None,
        }

    overview = {
        "symbol":      DEFAULT_SYMBOL,
        "pair_label":  symbol_to_pair(DEFAULT_SYMBOL),
        "price":       price,
        "last_update": last_update or _now_label(),
        "indicators":  indicators,
    }
    return overview


def _build_account_payload() -> dict:
    ctx = fetch_account_context()
    return {
        "updated_at": _now_label(),
        **ctx,
    }


def build_market_payload(
    tf_data: dict,
    price: float,
    last_update: str | None = None,
    chart_tfs: list[str] | tuple[str, ...] | set[str] | None = None,
    include_overview: bool = True,
) -> dict:
    target_tfs = TIMEFRAMES if chart_tfs is None else [tf for tf in TIMEFRAMES if tf in chart_tfs]
    fib_targets = TIMEFRAMES if include_overview else target_tfs
    fibs = _build_swing_fibs(tf_data, fib_targets)

    if include_overview:
        payload = _market_overview(tf_data, price, last_update)
        payload["fibs"] = fibs
    else:
        payload = {
            "symbol":      DEFAULT_SYMBOL,
            "pair_label":  symbol_to_pair(DEFAULT_SYMBOL),
            "price":       price,
            "last_update": last_update or _now_label(),
        }

    charts = {}
    for tf in target_tfs:
        if tf in tf_data:
            charts[tf] = build_chart_payload(tf_data[tf], fibs.get(tf))
    payload["charts"] = charts
    return payload


def _build_payload(tf_data: dict, price: float, analysis: dict) -> dict:
    payload = build_market_payload(tf_data, price, include_overview=True)
    return {
        **payload,
        "analysis_time": _now_label(),
        "signal":       analysis["signal"],
        "confidence":   analysis["confidence"],
        "raw_text":     analysis["raw_text"],
        "trade_levels": analysis["trade_levels"],
        "report_sections": analysis.get("report_sections", {}),
        "report_format_ok": analysis.get("report_format_ok", False),
        "report_missing_sections": analysis.get("report_missing_sections", []),
        # 구조화 트레이딩 시그널 (signal_processing.TradingSignal.to_dict())
        "trading_signal": analysis.get("trading_signal"),
        "claude_leverage": analysis.get("claude_leverage"),
        # Bull/Bear 사전 토론 결과
        "debate": analysis.get("debate"),
        # 투자 심판 결론
        "judge": analysis.get("judge"),
        # Risk Triad (Aggressive / Conservative / Neutral) 토론 결과
        "risk": analysis.get("risk"),
        # 과거 유사 상황 (BM25 회상 결과)
        "memories": analysis.get("memories", []),
        # account 는 account-stream 에서 실시간으로 별도 관리 — 여기서 포함하지 않음
    }


class MarketStreamManager:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self._tf_data: dict = {}
        self._price: float | None = None
        self._last_update: str | None = None
        self._ready = asyncio.Event()
        self._lock = asyncio.Lock()
        self._listeners: set[asyncio.Queue] = set()
        self._runner_task: asyncio.Task | None = None
        self._market_flush_task: asyncio.Task | None = None
        self._price_flush_task: asyncio.Task | None = None
        self._dirty_tfs: set[str] = set()
        self._needs_full_refresh = False
        self._stopped = False
        self._price_tick_task: asyncio.Task | None = None
        self._trade_count: int = 0          # aggTrade 수신 카운터 (디버그용)
        self._kline_count: int = 0          # kline 수신 카운터 (디버그용)
        self._ws_connect_count: int = 0     # WebSocket 재연결 횟수
        self._pending_kline: dict[str, dict] = {}    # 확정 kline 처리 큐
        self._kline_running: dict[str, bool] = {}    # 타임프레임별 계산 진행 플래그
        self._forming_kline: dict[str, dict] = {}    # 미확정(진행 중) 캔들 최신 OHLCV
        self._forming_refresh_task: asyncio.Task | None = None

    async def start(self):
        if self._runner_task and not self._runner_task.done():
            return
        self._stopped = False
        self._runner_task = asyncio.create_task(self._run_forever(), name="binance-market-stream")
        self._price_tick_task = asyncio.create_task(self._periodic_price_tick(), name="price-tick-1s")
        self._forming_refresh_task = asyncio.create_task(self._forming_indicator_refresh(), name="forming-refresh-2s")

    async def stop(self):
        self._stopped = True
        tasks = [self._runner_task, self._market_flush_task, self._price_flush_task,
                 self._price_tick_task, self._forming_refresh_task]
        for task in tasks:
            if task and not task.done():
                task.cancel()
        for task in tasks:
            if task:
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    async def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=16)
        async with self._lock:
            self._listeners.add(q)
        return q

    async def unsubscribe(self, q: asyncio.Queue):
        async with self._lock:
            self._listeners.discard(q)

    async def wait_until_ready(self):
        await self._ready.wait()

    def is_ready(self) -> bool:
        return self._ready.is_set()

    async def get_full_snapshot(self) -> dict:
        await self.wait_until_ready()
        async with self._lock:
            return build_market_payload(
                self._tf_data,
                self._price,
                self._last_update,
                chart_tfs=TIMEFRAMES,
                include_overview=True,
            )

    async def get_analysis_inputs(self) -> tuple[dict, float | None]:
        """분석용 데이터 반환. 미확정 캔들 OHLCV를 반영해 지표를 재계산한다."""
        await self.wait_until_ready()
        async with self._lock:
            raw_snapshot = {tf: df.copy(deep=True) for tf, df in self._tf_data.items()}
            forming = dict(self._forming_kline)  # 미확정 캔들 최신본
            price = self._price

        def _recompute_all(snapshot: dict, forming_klines: dict) -> dict:
            result = {}
            for tf, df in snapshot.items():
                try:
                    # 미확정 캔들이 있으면 OHLCV를 먼저 upsert
                    k = forming_klines.get(tf)
                    if k:
                        ts = pd.to_datetime(int(k["t"]), unit="ms")
                        row = {
                            "open": float(k["o"]), "high": float(k["h"]),
                            "low":  float(k["l"]), "close": float(k["c"]),
                            "volume": float(k["v"]),
                        }
                        df = _upsert_ohlcv(df, ts, row)
                    result[tf] = add_all_indicators(df)
                except Exception:
                    result[tf] = snapshot[tf]  # 실패 시 원본 유지
            return result

        tf_data = await asyncio.to_thread(_recompute_all, raw_snapshot, forming)
        return tf_data, price

    async def _run_forever(self):
        while not self._stopped:
            if not self._ready.is_set():
                try:
                    await self._bootstrap()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    print(f"[market-stream] bootstrap failed: {exc} — 5초 후 재시도")
                    await asyncio.sleep(5)
                    continue

            try:
                await self._consume_stream()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"[market-stream] websocket disconnected: {exc}")
                await asyncio.sleep(3)

    async def _bootstrap(self):
        """REST API로 초기 데이터 로드. 실패해도 WebSocket 기동은 계속한다."""
        # ── OHLCV 히스토리 fetch (실패 시 빈 dict로 폴백) ──
        try:
            tf_data = await asyncio.wait_for(
                asyncio.to_thread(_fetch_all, self.symbol),
                timeout=30,
            )
            print(f"[market-stream] bootstrap OHLCV 완료: {list(tf_data.keys())}")
        except asyncio.TimeoutError:
            print("[market-stream] bootstrap OHLCV timeout — 빈 데이터로 계속")
            tf_data = {}
        except Exception as exc:
            print(f"[market-stream] bootstrap OHLCV 실패: {exc} — 빈 데이터로 계속")
            tf_data = {}

        # ── 현재가 fetch (실패 허용) ──
        try:
            price = await asyncio.wait_for(
                asyncio.to_thread(fetch_current_price, self.symbol),
                timeout=8,
            )
            print(f"[market-stream] bootstrap 현재가: {price}")
        except Exception as exc:
            print(f"[market-stream] bootstrap 현재가 실패: {exc}")
            price = None

        async with self._lock:
            self._tf_data = tf_data
            self._price = price
            self._last_update = _now_label()

        # OHLCV 없어도 ready 설정 — WebSocket이 실시간으로 채움
        self._ready.set()
        print(f"[market-stream] ready! tf_data={list(tf_data.keys())} price={price}")

    def _stream_url(self) -> str:
        symbol = self.symbol.lower()
        streams = [f"{symbol}@aggTrade"] + [f"{symbol}@kline_{tf}" for tf in TIMEFRAMES]
        joined = "/".join(streams)
        return f"{MARKET_STREAM_WS_BASE}?streams={joined}"

    async def _consume_stream(self):
        self._ws_connect_count += 1
        print(f"[market-stream] WebSocket 연결 시도 #{self._ws_connect_count}: {self._stream_url()}")
        async with websockets.connect(
            self._stream_url(),
            ping_interval=None,   # Binance는 WS 프로토콜 ping frame 미지원 → None 필수
            ping_timeout=None,
            close_timeout=5,
            max_size=2**20,
        ) as ws:
            print(f"[market-stream] WebSocket 연결 성공 #{self._ws_connect_count}")
            async for raw_msg in ws:
                payload = json.loads(raw_msg)
                data = payload.get("data", payload)
                event_type = data.get("e")

                if event_type == "aggTrade":
                    self._trade_count += 1
                    await self._handle_trade(data)
                elif event_type == "kline":
                    self._kline_count += 1
                    await self._enqueue_kline(data.get("k", {}))

    async def _handle_trade(self, data: dict):
        price = _safe(data.get("p"))
        if price is None:
            return

        async with self._lock:
            self._price = price
            self._last_update = _now_label()

        self._schedule_price_flush()

    async def _enqueue_kline(self, kline: dict):
        """
        확정 캔들(x=True)만 지표 재계산 큐에 넣는다.

        진행 중 캔들(x=False)은 무시 — 이유:
          1) RSI/MACD/BB 등 지표는 과거 확정 캔들 기준으로 계산되어
             현재 캔들이 진행 중일 때 재계산해도 의미있는 변화가 없다.
          2) 현재 캔들의 실시간 가격(close/high/low)은
             프론트엔드 patchLastCandlePrice가 aggTrade 가격으로 직접 처리한다.
          3) 초당 4회 지표 계산 → 이벤트 루프 과부하 → 실시간 가격 지연의 근본 원인.

        미확정 캔들은 OHLCV만 tf_data에 업데이트 (지표 계산 없이 빠르게).
        → 분석 실행 시 최신 OHLCV가 포함된 상태로 지표를 재계산해서 Claude에 전달.
        확정 빈도: 15m=15분, 1h=1시간, 4h=4시간, 1d=1일 → 하루 총 수십 회.
        """
        tf = kline.get("i")
        if tf not in TIMEFRAMES:
            return

        if not kline.get("x", False):
            # 미확정 캔들: _forming_kline에만 보관 — tf_data는 건드리지 않음
            # tf_data를 수정하면 지표 컬럼이 제거돼 KeyError 발생
            # 분석 시 get_analysis_inputs에서 _forming_kline을 합산해 처리
            self._forming_kline[tf] = kline
            return

        # 확정 캔들: 계산 중에 또 확정이 오면 최신 것으로 덮어씀
        self._pending_kline[tf] = kline
        if not self._kline_running.get(tf, False):
            self._kline_running[tf] = True
            asyncio.create_task(self._process_kline(tf), name=f"kline-{tf}")

    async def _process_kline(self, tf: str):
        """확정 캔들에 대해 지표를 재계산하고 브로드캐스트. pending 소진까지 반복."""
        try:
            while True:
                kline = self._pending_kline.pop(tf, None)
                if kline is None:
                    break

                timestamp = pd.to_datetime(int(kline["t"]), unit="ms")
                row = {
                    "open":   float(kline["o"]),
                    "high":   float(kline["h"]),
                    "low":    float(kline["l"]),
                    "close":  float(kline["c"]),
                    "volume": float(kline["v"]),
                }

                async with self._lock:
                    current = self._tf_data.get(tf)

                def _compute(cur=current, ts=timestamp, r=row):
                    up = _upsert_ohlcv(cur, ts, r)
                    return add_all_indicators(up)

                updated = await asyncio.to_thread(_compute)

                async with self._lock:
                    self._tf_data[tf] = updated
                    self._last_update = _now_label()
                    self._dirty_tfs.add(tf)
                    if tf == "1h":
                        self._needs_full_refresh = True

                self._schedule_market_flush()
        finally:
            self._kline_running[tf] = False
            if self._pending_kline.get(tf):
                self._kline_running[tf] = True
                asyncio.create_task(self._process_kline(tf), name=f"kline-{tf}")

    def _schedule_price_flush(self):
        if self._price_flush_task and not self._price_flush_task.done():
            return
        self._price_flush_task = asyncio.create_task(self._flush_price_update())

    def _schedule_market_flush(self):
        if self._market_flush_task and not self._market_flush_task.done():
            return
        self._market_flush_task = asyncio.create_task(self._flush_market_update())

    async def _periodic_price_tick(self):
        """1초 주기 가격 + 진행 중 캔들 OHLCV 브로드캐스트.

        price: 실시간 aggTrade 가격
        forming: 타임프레임별 현재 캔들 OHLCV (Binance kline 기준)
                 → 프론트가 마지막 캔들을 실시간으로 갱신하는 데 사용
        지표(RSI·MACD 등)는 캔들 확정 시에만 재계산 — 여기서는 포함하지 않음
        """
        while not self._stopped:
            await asyncio.sleep(1.0)
            if self._stopped:
                break
            async with self._lock:
                price = self._price
                last_update = self._last_update
                forming_raw = dict(self._forming_kline)
            if price is None:
                continue

            # 진행 중 캔들 OHLCV 직렬화 (계산 없음 — 단순 숫자 변환)
            forming = {}
            for tf, k in forming_raw.items():
                try:
                    forming[tf] = {
                        "open":   float(k["o"]),
                        "high":   float(k["h"]),
                        "low":    float(k["l"]),
                        "close":  float(k["c"]),
                        "volume": float(k["v"]),
                    }
                except Exception:
                    pass

            await self._broadcast({"type": "price", "data": {
                "price": price,
                "last_update": last_update,
                "forming": forming,   # tf → {open, high, low, close, volume}
            }})

    async def _forming_indicator_refresh(self):
        """2초마다 진행 중 캔들 포함 지표를 재계산해 브로드캐스트.

        확정 캔들 처리(_process_kline)와 독립적으로 동작.
        tf_data는 수정하지 않고 스냅샷 + forming_kline으로 임시 계산 후 브로드캐스트.
        이벤트 루프는 비블로킹 — 계산은 asyncio.to_thread 스레드풀에서 실행.
        """
        while not self._stopped:
            await asyncio.sleep(2.0)
            if self._stopped or not self._ready.is_set():
                continue

            async with self._lock:
                if not self._forming_kline:
                    continue  # forming kline 아직 없음 (서버 막 시작)
                forming_raw = dict(self._forming_kline)
                tf_snap = {tf: df.copy(deep=True) for tf, df in self._tf_data.items()}
                price = self._price
                last_update = self._last_update

            def _compute(snap, forming):
                result = {}
                for tf, df in snap.items():
                    k = forming.get(tf)
                    if k:
                        ts = pd.to_datetime(int(k["t"]), unit="ms")
                        row = {
                            "open":   float(k["o"]), "high": float(k["h"]),
                            "low":    float(k["l"]), "close": float(k["c"]),
                            "volume": float(k["v"]),
                        }
                        df = _upsert_ohlcv(df, ts, row)
                    try:
                        result[tf] = add_all_indicators(df)
                    except Exception:
                        result[tf] = df
                return result

            try:
                updated = await asyncio.to_thread(_compute, tf_snap, forming_raw)
                payload = await asyncio.to_thread(
                    build_market_payload,
                    updated, price, last_update,
                    chart_tfs=TIMEFRAMES,
                    include_overview=True,
                )
                await self._broadcast({"type": "market", "data": payload})
            except Exception as exc:
                print(f"[forming-refresh] 오류: {exc}")

    async def _flush_price_update(self):
        """aggTrade 수신 후 100ms 디바운스 — 최신 가격을 빠르게 브로드캐스트."""
        await asyncio.sleep(0.10)
        async with self._lock:
            price = self._price
            last_update = self._last_update
        if price is None:
            return
        await self._broadcast({"type": "price", "data": {
            "price": price,
            "last_update": last_update,
        }})

    async def _flush_market_update(self):
        await asyncio.sleep(0.35)
        # 락 안에서는 데이터만 복사 — CPU 연산은 락 밖에서 스레드로 실행
        async with self._lock:
            dirty_tfs = set(self._dirty_tfs)
            full_refresh = self._needs_full_refresh
            self._dirty_tfs.clear()
            self._needs_full_refresh = False

            if not dirty_tfs and not full_refresh:
                return

            tf_data_snapshot = dict(self._tf_data)
            price_snapshot = self._price
            last_update_snapshot = self._last_update

        # build_market_payload 는 CPU 집약적 — 스레드풀에서 실행하여 이벤트 루프 비블로킹
        payload = await asyncio.to_thread(
            build_market_payload,
            tf_data_snapshot,
            price_snapshot,
            last_update_snapshot,
            chart_tfs=TIMEFRAMES if full_refresh else dirty_tfs,
            include_overview=full_refresh,
        )

        await self._broadcast({"type": "market", "data": payload})

    async def _broadcast(self, message: dict):
        async with self._lock:
            listeners = list(self._listeners)

        for listener in listeners:
            if listener.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    listener.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                listener.put_nowait(message)


_market_stream = MarketStreamManager(DEFAULT_SYMBOL)


ACCOUNT_PERIODIC_REFRESH_SECS = 60  # 주기적 REST 폴링 간격 (초)


class AccountStreamManager:
    def __init__(self):
        self._payload: dict | None = None
        self._ready = asyncio.Event()
        self._lock = asyncio.Lock()
        self._listeners: set[asyncio.Queue] = set()
        self._runner_task: asyncio.Task | None = None
        self._refresh_task: asyncio.Task | None = None
        self._periodic_task: asyncio.Task | None = None
        self._pending_refresh = False
        self._last_refresh_started_at = 0.0
        self._stopped = False

    async def start(self):
        if self._runner_task and not self._runner_task.done():
            return
        self._stopped = False
        self._runner_task = asyncio.create_task(self._run_forever(), name="binance-account-stream")
        self._periodic_task = asyncio.create_task(self._periodic_refresh(), name="binance-account-periodic")

    async def stop(self):
        self._stopped = True
        tasks = [self._runner_task, self._refresh_task, self._periodic_task]
        for task in tasks:
            if task and not task.done():
                task.cancel()
        for task in tasks:
            if task:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        with contextlib.suppress(Exception):
            await asyncio.to_thread(close_user_data_stream)

    async def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=8)
        async with self._lock:
            self._listeners.add(q)
        return q

    async def unsubscribe(self, q: asyncio.Queue):
        async with self._lock:
            self._listeners.discard(q)

    async def wait_until_ready(self):
        await self._ready.wait()

    def is_ready(self) -> bool:
        return self._ready.is_set()

    async def get_snapshot(self) -> dict:
        await self.wait_until_ready()
        async with self._lock:
            return copy.deepcopy(self._payload or {})

    async def _run_forever(self):
        while not self._stopped:
            if not self._ready.is_set():
                try:
                    await self._refresh_payload(delay=0, broadcast=False)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    print(f"[account-stream] bootstrap failed: {exc}")
                    await asyncio.sleep(5)
                    continue

            if not runtime_config.BINANCE_API_KEY:
                await asyncio.sleep(30)
                continue

            try:
                await self._consume_stream()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                msg = str(exc)
                if "인증 실패 (401)" in msg:
                    print(f"[account-stream] auth failed, using REST snapshot fallback: {exc}")
                    with contextlib.suppress(Exception):
                        await self._refresh_payload(delay=0, broadcast=True)
                    await asyncio.sleep(10)
                else:
                    print(f"[account-stream] websocket disconnected: {exc}")
                    # 재연결 전 REST 스냅샷 갱신 — 끊긴 동안의 변경사항 반영
                    with contextlib.suppress(Exception):
                        await self._refresh_payload(delay=0, broadcast=True)
                    await asyncio.sleep(3)

    async def _periodic_refresh(self):
        """WebSocket 이벤트 여부와 무관하게 주기적으로 REST 스냅샷을 갱신."""
        while not self._stopped:
            await asyncio.sleep(ACCOUNT_PERIODIC_REFRESH_SECS)
            if self._stopped:
                break
            if not runtime_config.BINANCE_API_KEY:
                continue
            with contextlib.suppress(Exception):
                await self._refresh_payload(delay=0, broadcast=True)

    async def _consume_stream(self):
        listen_key = await asyncio.to_thread(open_user_data_stream)
        ws_url = f"{ACCOUNT_STREAM_WS_BASE}/{listen_key}"

        try:
            async with websockets.connect(
                ws_url,
                ping_interval=None,
                ping_timeout=None,
                close_timeout=5,
                max_size=2**20,
            ) as ws:
                while not self._stopped:
                    try:
                        raw_msg = await asyncio.wait_for(
                            ws.recv(),
                            timeout=ACCOUNT_STREAM_KEEPALIVE_SECS,
                        )
                    except asyncio.TimeoutError:
                        await asyncio.to_thread(keepalive_user_data_stream)
                        continue

                    payload = json.loads(raw_msg)
                    event_type = payload.get("e")
                    if event_type in {
                        "ACCOUNT_UPDATE",
                        "ACCOUNT_CONFIG_UPDATE",
                        "MARGIN_CALL",
                        "ORDER_TRADE_UPDATE",
                        "TRADE_LITE",
                    }:
                        self._schedule_refresh()
                    elif event_type == "listenKeyExpired":
                        raise RuntimeError("listenKey expired")
        finally:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(close_user_data_stream)

    def _schedule_refresh(self):
        if self._refresh_task and not self._refresh_task.done():
            self._pending_refresh = True
            return
        self._refresh_task = asyncio.create_task(self._refresh_payload())

    async def _refresh_payload(
        self,
        delay: float = ACCOUNT_REFRESH_DEBOUNCE_SECS,
        broadcast: bool = True,
    ):
        loop = asyncio.get_running_loop()
        min_delay = max(
            0.0,
            ACCOUNT_REFRESH_MIN_INTERVAL_SECS - (loop.time() - self._last_refresh_started_at),
        )
        wait_for = max(delay, min_delay)
        if wait_for > 0:
            await asyncio.sleep(wait_for)

        self._last_refresh_started_at = loop.time()
        try:
            payload = await asyncio.to_thread(_build_account_payload)
            async with self._lock:
                self._payload = payload
                self._ready.set()
            if broadcast:
                await self._broadcast({"type": "account", "data": payload})
        finally:
            pending = self._pending_refresh
            self._pending_refresh = False
            if pending and not self._stopped:
                self._refresh_task = asyncio.create_task(
                    self._refresh_payload(delay=ACCOUNT_REFRESH_DEBOUNCE_SECS)
                )
            else:
                self._refresh_task = None

    async def _broadcast(self, message: dict):
        async with self._lock:
            listeners = list(self._listeners)

        for listener in listeners:
            if listener.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    listener.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                listener.put_nowait(message)


_account_stream = AccountStreamManager()


class MacroSnapshotManager:
    def __init__(self):
        self._payload: dict | None = None
        self._ready = asyncio.Event()
        self._lock = asyncio.Lock()
        self._runner_task: asyncio.Task | None = None
        self._stopped = False

    async def start(self):
        if self._runner_task and not self._runner_task.done():
            return
        self._stopped = False
        self._runner_task = asyncio.create_task(self._run_forever(), name="macro-snapshot-refresh")

    async def stop(self):
        self._stopped = True
        task = self._runner_task
        if task and not task.done():
            task.cancel()
        if task:
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def wait_until_ready(self):
        await self._ready.wait()

    def is_ready(self) -> bool:
        return self._ready.is_set()

    async def get_snapshot(self) -> dict:
        await self.wait_until_ready()
        async with self._lock:
            return copy.deepcopy(self._payload or {})

    async def _run_forever(self):
        while not self._stopped:
            sleep_for = MACRO_REFRESH_SECS
            try:
                payload = await asyncio.to_thread(fetch_macro_context)
                async with self._lock:
                    self._payload = payload
                    self._ready.set()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"[macro-refresh] fetch failed: {exc}")
                sleep_for = 120
            await asyncio.sleep(sleep_for)


_macro_snapshot = MacroSnapshotManager()


# 수동 분석 버튼 쿨다운: 10분
MANUAL_COOLDOWN_SECS = 60 * 60
MANUAL_COOLDOWN_STATE_PATH = os.path.join(BASE_DIR, "data", "manual_cooldown.json")


def _load_manual_cooldown_time() -> float:
    """서버 재시작 후 수동 클릭 시각을 복원. 없으면 0.0 반환."""
    try:
        with open(MANUAL_COOLDOWN_STATE_PATH, "r", encoding="utf-8") as f:
            return float(json.load(f).get("last_manual_started_at", 0.0))
    except Exception:
        return 0.0


def _save_manual_cooldown_time(ts: float) -> None:
    try:
        os.makedirs(os.path.dirname(MANUAL_COOLDOWN_STATE_PATH), exist_ok=True)
        with open(MANUAL_COOLDOWN_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump({"last_manual_started_at": ts}, f)
    except Exception as exc:
        import logging as _log
        _log.getLogger(__name__).warning("manual cooldown 저장 실패 — %s", exc)


class AnalysisManager:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._job: dict | None = None
        self._task: asyncio.Task | None = None
        self._latest_result: dict | None = _load_latest_analysis()
        # deepcopy 방지용 JSON 캐시 — 시작 시 디스크 로드 결과도 즉시 직렬화
        _lr = self._latest_result
        self._latest_result_json: str | None = (
            json.dumps(_lr, ensure_ascii=False) if _lr is not None else None
        )
        self._last_manual_started_at: float = _load_manual_cooldown_time()

    async def stop(self):
        task = None
        async with self._lock:
            task = self._task
        if task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def start_job(self, bypass_cooldown: bool = False) -> tuple[dict, bool]:
        async with self._lock:
            if self._job and self._job["status"] in {"pending", "running"}:
                return self._serialize_job(self._job), False

            # 쿨다운 체크 (스케줄러는 bypass)
            if not bypass_cooldown:
                remaining = max(0.0, MANUAL_COOLDOWN_SECS - (time.time() - self._last_manual_started_at))
                if remaining > 0:
                    raise HTTPException(
                        status_code=429,
                        detail={
                            "reason": "cooldown",
                            "cooldown_remaining_secs": int(remaining),
                        },
                    )
                self._last_manual_started_at = time.time()
                _save_manual_cooldown_time(self._last_manual_started_at)

            started_at = _now_iso()
            job = {
                "id": uuid.uuid4().hex,
                "status": "running",
                "step": 0,
                # phase/phase_detail: step 2 내부의 세부 진행 상태
                # (예: "bull", "bear", "final"). UI 진행 표시용.
                "phase": None,
                "phase_detail": None,
                "error": None,
                "result": None,
                "started_at": started_at,
                "updated_at": started_at,
                "completed_at": None,
            }
            self._job = job
            self._task = asyncio.create_task(
                self._run_job(job["id"]),
                name=f"analysis-job-{job['id'][:8]}",
            )
            return self._serialize_job(job), True

    async def get_status(self, include_latest: bool = False) -> dict:
        async with self._lock:
            response = {
                "job": self._serialize_job(self._job),
                "cooldown_remaining_secs": int(max(0.0, MANUAL_COOLDOWN_SECS - (time.time() - self._last_manual_started_at))),
            }
            if (
                self._job
                and self._job["status"] == "completed"
                and self._job.get("result") is not None
            ):
                # deepcopy 대신 JSON 역직렬화 — 더 빠름
                result_json = self._job.get("_result_json")
                if result_json is None:
                    result_json = json.dumps(self._job["result"], ensure_ascii=False)
                    self._job["_result_json"] = result_json
                response["result"] = json.loads(result_json)
            elif include_latest and self._latest_result_json is not None:
                response["latest_result"] = json.loads(self._latest_result_json)
            return response

    def _serialize_job(self, job: dict | None) -> dict | None:
        if not job:
            return None
        return {
            "id": job["id"],
            "status": job["status"],
            "step": job["step"],
            "phase": job.get("phase"),
            "phase_detail": job.get("phase_detail"),
            "error": job["error"],
            "started_at": job["started_at"],
            "updated_at": job["updated_at"],
            "completed_at": job["completed_at"],
        }

    async def _set_step(self, job_id: str, step: int):
        async with self._lock:
            if not self._job or self._job["id"] != job_id:
                return
            self._job["status"] = "running"
            self._job["step"] = step
            self._job["updated_at"] = _now_iso()

    async def _set_phase(self, job_id: str, phase: str, detail: str):
        """step 2 내부의 세부 진행 상태(phase)를 갱신한다.
        토론 진행(Bull/Bear) 중 UI 프로그레스 표시를 위한 용도."""
        async with self._lock:
            if not self._job or self._job["id"] != job_id:
                return
            self._job["phase"] = phase
            self._job["phase_detail"] = detail
            self._job["updated_at"] = _now_iso()

    async def _complete(self, job_id: str, payload: dict):
        completed_at = _now_iso()
        async with self._lock:
            if not self._job or self._job["id"] != job_id:
                return
            self._job["status"] = "completed"
            self._job["step"] = 3
            self._job["error"] = None
            self._job["result"] = copy.deepcopy(payload)
            self._job["updated_at"] = completed_at
            self._job["completed_at"] = completed_at
            self._latest_result = copy.deepcopy(payload)
            self._latest_result_json = json.dumps(self._latest_result, ensure_ascii=False)

        # ── 자동매매 훅 ─────────────────────────────────
        if _AUTO_TRADER_AVAILABLE and _auto_trader is not None:
            loop = asyncio.get_event_loop()
            try:
                trade_rec = await loop.run_in_executor(
                    None, _auto_trader.execute_trade, copy.deepcopy(payload)
                )
                import logging as _atlog
                _atlog.getLogger(__name__).info(
                    "[AutoTrader] action=%s reason=%s",
                    trade_rec.action, trade_rec.reason,
                )
            except Exception as _ate:
                import logging as _atlog
                _atlog.getLogger(__name__).error("[AutoTrader] 실행 오류: %s", _ate)

        # ── 서버 사이드 증분 백테스트 (백그라운드) ────────────
        asyncio.create_task(_run_backtest_background())

    async def _fail(self, job_id: str, message: str, status: str = "error"):
        failed_at = _now_iso()
        async with self._lock:
            if not self._job or self._job["id"] != job_id:
                return
            self._job["status"] = status
            self._job["error"] = message
            self._job["updated_at"] = failed_at
            self._job["completed_at"] = failed_at

    async def _run_job(self, job_id: str):
        loop = asyncio.get_event_loop()
        try:
            await self._set_step(job_id, 1)
            if _market_stream.is_ready():
                tf_data, price = await _market_stream.get_analysis_inputs()
            else:
                tf_data = await loop.run_in_executor(_executor, _fetch_all, DEFAULT_SYMBOL)
                try:
                    price = await loop.run_in_executor(_executor, fetch_current_price, DEFAULT_SYMBOL)
                except Exception:
                    price = None

            if _macro_snapshot.is_ready():
                macro_snapshot = await _macro_snapshot.get_snapshot()
            else:
                macro_snapshot = await loop.run_in_executor(_executor, fetch_macro_context)

            await self._set_step(job_id, 2)

            # 토론 진행률 콜백 — 워커 스레드에서 호출되므로 이벤트 루프로 넘긴다.
            def _progress_cb(phase: str, detail: str):
                try:
                    asyncio.run_coroutine_threadsafe(
                        self._set_phase(job_id, phase, detail), loop
                    )
                except Exception:
                    # 진행률 실패가 분석 흐름을 끊어서는 안 된다.
                    pass

            analysis = await loop.run_in_executor(
                _executor,
                functools.partial(
                    run_full_analysis,
                    tf_data,
                    macro_snapshot,
                    progress_cb=_progress_cb,
                ),
            )

            await self._set_step(job_id, 3)
            payload = await loop.run_in_executor(
                _executor, _build_payload, tf_data, price, analysis
            )
            await self._complete(job_id, payload)
            await asyncio.to_thread(_persist_latest_analysis, payload)
            await asyncio.to_thread(_persist_analysis_history, payload)
        except asyncio.CancelledError:
            await self._fail(job_id, "분석 작업이 취소되었습니다.", status="cancelled")
            raise
        except Exception as exc:
            await self._fail(job_id, str(exc))


_analysis_manager = AnalysisManager()


# ══════════════════════════════════════════════
# 서버 사이드 자동 분석 스케줄러
# ══════════════════════════════════════════════
import datetime as _dt

SCHEDULE_STATE_PATH = os.path.join(BASE_DIR, "data", "schedule_state.json")

class ScheduleManager:
    """서버에서 30분마다 분석을 실행하는 백그라운드 스케줄러."""

    INTERVAL_MIN = 240  # 고정 4시간

    def __init__(self):
        self._enabled: bool = False
        self._next_run_at: str | None = None
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._load_state()

    def _load_state(self):
        """서버 재시작 후에도 설정을 복원한다."""
        try:
            if os.path.exists(SCHEDULE_STATE_PATH):
                with open(SCHEDULE_STATE_PATH, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                self._enabled = bool(saved.get("enabled", False))
        except Exception:
            pass

    def _save_state(self):
        try:
            os.makedirs(os.path.dirname(SCHEDULE_STATE_PATH), exist_ok=True)
            with open(SCHEDULE_STATE_PATH, "w", encoding="utf-8") as f:
                json.dump({"enabled": self._enabled, "interval_min": self.INTERVAL_MIN}, f)
        except Exception:
            pass

    def status(self) -> dict:
        return {
            "enabled":      self._enabled,
            "interval_min": self.INTERVAL_MIN,
            "next_run_at":  self._next_run_at,
        }

    async def set_schedule(self, enabled: bool, interval_min: int = INTERVAL_MIN):
        async with self._lock:
            self._enabled = enabled
            self._save_state()
            await self._restart_task()

    async def start(self):
        """앱 시작 시 호출. 저장된 설정이 enabled면 태스크 재개."""
        if self._enabled:
            await self._restart_task()

    async def stop(self):
        if self._task and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _restart_task(self):
        await self.stop()
        if self._enabled:
            self._task = asyncio.create_task(
                self._loop(), name="schedule-auto-analyze"
            )

    async def _loop(self):
        while True:
            wait_sec = self.INTERVAL_MIN * 60
            next_dt  = _dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(seconds=wait_sec)
            self._next_run_at = next_dt.isoformat()
            try:
                await asyncio.sleep(wait_sec)
            except asyncio.CancelledError:
                self._next_run_at = None
                raise
            # 분석 실행 (이미 진행 중이면 skip, 스케줄러는 쿨다운 우회)
            _, started = await _analysis_manager.start_job(bypass_cooldown=True)
            if not started:
                print("[scheduler] analysis already running – skipped")


_schedule_manager = ScheduleManager()


# ══════════════════════════════════════════════
# 서버 사이드 백테스트 백그라운드 태스크
# ══════════════════════════════════════════════
import logging as _bt_log
_bt_logger = _bt_log.getLogger("backtest_bg")

async def _run_backtest_background():
    """분석 완료 후 백그라운드에서 증분 백테스트 실행 후 캐시 저장."""
    try:
        import backtester as _bt
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _bt.run_backtest_incremental, True)
        _bt_logger.info("[Backtest] 증분 백테스트 완료 → 캐시 갱신")
    except Exception as exc:
        _bt_logger.warning("[Backtest] 백그라운드 실행 실패: %s", exc)


async def _position_watcher_loop():
    """
    매 분 열린 드라이런 포지션의 SL/TP 도달 여부를 확인.
    포지션이 없으면 아무것도 안 함 (Binance 호출 0회).
    포지션이 있을 때만 1분봉 최신 1~5개만 조회.
    """
    try:
        import backtester as _bt
    except ImportError:
        _bt_logger.warning("[Watcher] backtester 임포트 실패 — 감시 루프 종료")
        return

    _bt_logger.info("[Watcher] 포지션 감시 루프 시작")
    while True:
        # 다음 분 00초에 맞춰 실행 (예: :01초에 시작하면 다음 :00초까지 대기)
        now_sec = time.time()
        next_minute = (int(now_sec // 60) + 1) * 60 + 2  # +2초: 캔들 확정 대기
        await asyncio.sleep(next_minute - now_sec)

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, _bt.tick_open_position)
            if result:
                _bt_logger.info(
                    "[Watcher] 포지션 확정: %s %s  pnl=%+.2f USD",
                    result.get("symbol"), result.get("outcome"), result.get("pnl_usd", 0)
                )
        except Exception as exc:
            _bt_logger.warning("[Watcher] tick 오류: %s", exc)


# ══════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════
async def _reflection_loop():
    """6시간마다 자동으로 /api/reflect 를 내부 호출해 리플렉션을 수행한다."""
    import logging as _rlog
    _rlog = _rlog.getLogger("reflection-loop")
    INTERVAL = 6 * 60 * 60  # 6시간
    await asyncio.sleep(60)  # 서버 시작 후 1분 대기 (스트림 안정화)
    while True:
        if _REFLECTION_ENABLED:
            try:
                _rlog.info("[reflection-loop] 자동 리플렉션 시작")
                result = await reflect_endpoint()
                _rlog.info(
                    "[reflection-loop] 완료 — processed=%s skipped=%s",
                    result.get("processed", 0),
                    result.get("skipped_no_baseline", 0),
                )
            except Exception as _exc:
                _rlog.warning("[reflection-loop] 실패 — %s", _exc)
        await asyncio.sleep(INTERVAL)


@app.on_event("startup")
async def on_startup():
    await _market_stream.start()
    await _account_stream.start()
    await _macro_snapshot.start()
    await _schedule_manager.start()
    asyncio.create_task(_position_watcher_loop(), name="position-watcher")
    asyncio.create_task(_reflection_loop(), name="reflection-loop")


@app.on_event("shutdown")
async def on_shutdown():
    await _market_stream.stop()
    await _account_stream.stop()
    await _macro_snapshot.stop()
    await _analysis_manager.stop()
    await _schedule_manager.stop()


@app.get("/api/market-stream")
async def market_stream():
    async def generate():
        queue = await _market_stream.subscribe()
        try:
            # ready 대기 중에도 keepalive를 5초마다 보내
            # → 브라우저가 연결이 끊겼다고 오해해서 재연결을 반복하는 것을 방지
            while not _market_stream.is_ready():
                try:
                    await asyncio.wait_for(
                        asyncio.shield(_market_stream._ready.wait()), timeout=5
                    )
                except asyncio.TimeoutError:
                    yield ": waiting\n\n"

            snapshot = await _market_stream.get_full_snapshot()
            yield f"data: {json.dumps({'type':'snapshot','data':snapshot})}\n\n"

            while True:
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=15)
                    yield f"data: {json.dumps(message)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            await _market_stream.unsubscribe(queue)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/account-stream")
async def account_stream():
    async def generate():
        queue = await _account_stream.subscribe()
        try:
            await _account_stream.wait_until_ready()
            snapshot = await _account_stream.get_snapshot()
            yield f"data: {json.dumps({'type':'snapshot','data':snapshot})}\n\n"

            while True:
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=20)
                    yield f"data: {json.dumps(message)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            await _account_stream.unsubscribe(queue)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/connections")
async def connections():
    """현재 market-stream 에 연결된 SSE 클라이언트 수 반환."""
    async with _market_stream._lock:
        count = len(_market_stream._listeners)
    return {"count": count}


@app.get("/api/debug/price")
async def debug_price():
    """WebSocket 연결 상태 및 현재 가격 진단용 엔드포인트."""
    import datetime as _dt
    async with _market_stream._lock:
        price = _market_stream._price
        last_update = _market_stream._last_update
        listeners = len(_market_stream._listeners)
        ready = _market_stream._ready.is_set()
        runner_alive = (
            _market_stream._runner_task is not None
            and not _market_stream._runner_task.done()
        )
        tick_alive = (
            _market_stream._price_tick_task is not None
            and not _market_stream._price_tick_task.done()
        )
    return {
        "price": price,
        "last_update": last_update,
        "server_time": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "stream_ready": ready,
        "runner_task_alive": runner_alive,
        "price_tick_task_alive": tick_alive,
        "sse_listeners": listeners,
        # aggTrade/kline 수신 카운터 — 0이면 WebSocket이 데이터를 못 받는 것
        "trade_events_received": _market_stream._trade_count,
        "kline_events_received": _market_stream._kline_count,
        "ws_reconnect_count": _market_stream._ws_connect_count,
    }


@app.get("/api/schedule")
async def schedule_get():
    """현재 자동분석 스케줄 상태 반환."""
    return _schedule_manager.status()


class ScheduleSetRequest(BaseModel):
    enabled: bool


@app.post("/api/schedule")
async def schedule_set(body: ScheduleSetRequest):
    """자동분석 스케줄 설정. enabled=true/false 로 30분 타이머를 제어."""
    await _schedule_manager.set_schedule(body.enabled)
    return _schedule_manager.status()


@app.post("/api/analyze")
async def analyze_start():
    job, started = await _analysis_manager.start_job()
    return {"job": job, "started": started}


@app.get("/api/analyze")
async def analyze_status(include_latest: bool = False):
    return await _analysis_manager.get_status(include_latest=include_latest)



@app.post("/api/reflect")
async def reflect_endpoint():
    """
    메모리에 누적된 과거 기록들 중 outcome 이 비어 있는 것들을
    '현재가 vs 기록 당시 가격' 변화와 함께 역할별 Reflection 에이전트에 돌려
    교훈을 기록한다.

    처리 대상:
      - analyst 메모리 (종합 리포트)
      - bull/bear/judge/aggressive/conservative/neutral 에이전트 메모리

    사용 예:
      - 프론트엔드 '리플렉션' 버튼
      - schedule 스킬로 6시간마다 자동 호출

    가격 베이스라인 우선순위 (analyst):
      1) meta["price_at_analysis"]  ← 분석 시점 현재가 (정확)
      2) meta["trade_levels"]["entry"]  ← 진입가가 있으면 그것
      3) skip (잘못된 피드백 방지)
    에이전트 역할 메모리: meta["price_at_analysis"] 또는 skip
    """
    if not _REFLECTION_ENABLED:
        return {
            "ok": False,
            "error": "rank_bm25 또는 관련 모듈이 설치되지 않아 reflection 을 사용할 수 없습니다.",
        }

    loop = asyncio.get_event_loop()
    import datetime as _dt

    # 현재가 수집 (단일 호출)
    try:
        price_now = await loop.run_in_executor(_executor, fetch_current_price, DEFAULT_SYMBOL)
    except Exception as exc:
        return {"ok": False, "error": f"가격 수집 실패 — {exc}"}

    now_utc = _dt.datetime.now(_dt.timezone.utc)
    all_results = []
    total_skipped = 0

    # ── Analyst 메모리 리플렉션 ─────────────────────────
    analyst_memory = _get_memory("analyst")
    pending = analyst_memory.list_pending_reflections(min_age_seconds=300.0, limit=5)

    for rec in pending:
        try:
            ts = _dt.datetime.fromisoformat(rec.timestamp.replace("Z", "+00:00"))
        except Exception:
            continue
        elapsed = (now_utc - ts).total_seconds()

        meta = rec.meta or {}
        price_then = meta.get("price_at_analysis")
        if not isinstance(price_then, (int, float)) or price_then <= 0:
            trade_levels = meta.get("trade_levels") or {}
            entry = trade_levels.get("entry")
            if isinstance(entry, (int, float)) and entry > 0:
                price_then = float(entry)
            else:
                total_skipped += 1
                continue
        price_then = float(price_then)

        res = await loop.run_in_executor(
            _executor,
            lambda r=rec, pt=price_then, el=elapsed: _reflect_for_role(
                role="analyst",
                record_ts=r.timestamp,
                situation=r.situation,
                advice=r.advice,
                price_then=pt,
                price_now=price_now,
                elapsed_seconds=el,
                memory=analyst_memory,
            ),
        )
        all_results.append(res.to_dict())

    # ── 에이전트 역할 메모리 리플렉션 ─────────────────────
    if _get_agent_memories is not None and _reflect_for_role is not None:
        try:
            agent_mems = _get_agent_memories()
            for role in _AGENT_ROLES_FOR_REFLECT:
                role_mem = agent_mems.get(role)
                role_pending = role_mem.list_pending_reflections(min_age_seconds=300.0, limit=3)
                for rec in role_pending:
                    try:
                        ts = _dt.datetime.fromisoformat(rec.timestamp.replace("Z", "+00:00"))
                    except Exception:
                        continue
                    elapsed = (now_utc - ts).total_seconds()
                    meta = rec.meta or {}
                    price_then = meta.get("price_at_analysis")
                    if not isinstance(price_then, (int, float)) or price_then <= 0:
                        total_skipped += 1
                        continue
                    price_then = float(price_then)

                    res = await loop.run_in_executor(
                        _executor,
                        lambda r=rec, pt=price_then, el=elapsed, rl=role, rm=role_mem: _reflect_for_role(
                            role=rl,
                            record_ts=r.timestamp,
                            situation=r.situation,
                            advice=r.advice,
                            price_then=pt,
                            price_now=price_now,
                            elapsed_seconds=el,
                            memory=rm,
                        ),
                    )
                    all_results.append(res.to_dict())
        except Exception as exc:
            import logging as _logging
            _logging.getLogger(__name__).warning("에이전트 역할 reflection 실패 — %s", exc)

    # 전체 기록 수 (outcome 유무 무관)
    total_records = analyst_memory.size()
    # 아직 리플렉션이 안 된 기록 수 (outcome 비어있는 것)
    still_pending = len(analyst_memory.list_pending_reflections(min_age_seconds=0, limit=9999))

    return {
        "ok": True,
        "price_now": price_now,
        "processed": len(all_results),
        "skipped_no_baseline": total_skipped,
        "memory_size": total_records,          # 프론트엔드 호환 필드
        "analyst_memory_size": total_records,  # 하위 호환
        "pending_count": still_pending,
        "results": all_results,
    }


@app.get("/api/macro")
async def macro_endpoint():
    """
    거시경제 지표 JSON 반환.
    FRED (1시간 캐시) + DefiLlama + CoinGecko.
    """
    if _macro_snapshot.is_ready():
        data = await _macro_snapshot.get_snapshot()
    else:
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(_executor, fetch_macro_context)
    # JSON 직렬화 가능하도록 None 포함 dict 그대로 반환
    return data


@app.get("/api/account")
async def account_endpoint():
    if _account_stream.is_ready():
        data = await _account_stream.get_snapshot()
    else:
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(_executor, _build_account_payload)
    return data


@app.get("/api/setup/status")
async def setup_status():
    """
    어떤 API 키가 빠져 있는지 반환.
    값이 채워져 있는지만 확인 — 실제 유효성 검증은 하지 않음.
    """
    import config as _cfg
    return {
        "claude":   bool(_cfg.CLAUDE_API_KEY),
        "binance":  bool(_cfg.BINANCE_API_KEY and _cfg.BINANCE_SECRET_KEY),
        "fred":     bool(_cfg.FRED_API_KEY),
    }


class SetupSaveRequest(BaseModel):
    claude_api_key:     str = ""
    binance_api_key:    str = ""
    binance_secret_key: str = ""
    fred_api_key:       str = ""


@app.post("/api/setup/save")
async def setup_save(body: SetupSaveRequest):
    """
    입력받은 키를 .env 파일에 저장하고 config 모듈을 재로드.
    로컬 실행 전용.
    """
    import config as _cfg
    from pathlib import Path

    env_path = Path(__file__).resolve().parent / ".env"

    # 기존 .env 읽기 (없으면 빈 dict)
    existing: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                existing[k.strip()] = v.strip()

    # 새 값 병합 (빈 문자열이면 기존 값 유지)
    updates = {
        "CLAUDE_API_KEY":     body.claude_api_key,
        "BINANCE_API_KEY":    body.binance_api_key,
        "BINANCE_SECRET_KEY": body.binance_secret_key,
        "FRED_API_KEY":       body.fred_api_key,
    }
    for k, v in updates.items():
        if v:
            existing[k] = v

    # .env 재작성
    lines = ["# API 키 — 절대 외부에 공유하지 마세요!\n"]
    for k, v in existing.items():
        lines.append(f"{k}={v}\n")
    env_path.write_text("".join(lines), encoding="utf-8")

    # os.environ + config 모듈 인메모리 갱신 (서버 재시작 없이 반영)
    for k, v in existing.items():
        os.environ[k] = v
    import importlib
    importlib.reload(_cfg)

    # 계좌 스트림은 API 키를 사용하므로 저장 직후 재시작해 새 값을 즉시 반영
    await _account_stream.stop()
    await _account_stream.start()

    return {"ok": True}


@app.get("/api/analysis-history")
async def analysis_history_endpoint(limit: int = 100):
    """저장된 분석 히스토리 반환 (최신순)."""
    entries: list = []
    try:
        if os.path.exists(ANALYSIS_HISTORY_PATH):
            with open(ANALYSIS_HISTORY_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except Exception:
                            pass
    except Exception as exc:
        print(f"[analysis-history] read failed: {exc}")
    return {"entries": list(reversed(entries[-limit:]))}


@app.get("/api/performance")
async def performance_endpoint(days: int = 30):
    """account_history.jsonl 기반 성과 지표 반환."""
    from datetime import datetime, timezone, timedelta
    history_path = os.path.join(BASE_DIR, "data", "account_history.jsonl")
    entries: list = []
    try:
        if os.path.exists(history_path):
            with open(history_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except Exception:
                            pass
    except Exception as exc:
        return {"error": str(exc), "snapshots": [], "daily": []}

    # days 범위 필터
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).timestamp()
    snapshots = [e for e in entries if (e.get("observed_ts") or 0) >= cutoff]

    # 일별 집계 (KST 기준 날짜로 그룹)
    from collections import defaultdict
    from time_utils import format_kst
    daily_map: dict = defaultdict(list)
    for snap in snapshots:
        try:
            dt = datetime.fromtimestamp(snap["observed_ts"], tz=timezone.utc)
            day_key = format_kst(dt, "%Y-%m-%d")
            daily_map[day_key].append(snap)
        except Exception:
            pass

    daily = []
    for day_key in sorted(daily_map.keys()):
        day_snaps = daily_map[day_key]
        # 당일 마지막 스냅샷 기준
        last = day_snaps[-1]
        first = day_snaps[0]
        pnl_vals = [s.get("today_total_pnl") for s in day_snaps if s.get("today_total_pnl") is not None]
        eq_vals  = [s.get("account_equity")  for s in day_snaps if s.get("account_equity")  is not None]
        daily.append({
            "date":           day_key,
            "pnl":            last.get("today_total_pnl"),
            "pnl_pct":        last.get("today_pnl_pct"),
            "equity_start":   first.get("account_equity"),
            "equity_end":     last.get("account_equity"),
            "equity_high":    max(eq_vals)  if eq_vals  else None,
            "equity_low":     min(eq_vals)  if eq_vals  else None,
            "pnl_high":       max(pnl_vals) if pnl_vals else None,
            "pnl_low":        min(pnl_vals) if pnl_vals else None,
            "risk_status":    last.get("risk_status"),
            "snap_count":     len(day_snaps),
        })

    return {"snapshots": snapshots, "daily": daily}


@app.get("/")
async def root():
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(html_path, encoding="utf-8") as f:
        return HTMLResponse(f.read())


# ══════════════════════════════════════════════
# 자동매매 API
# ══════════════════════════════════════════════

class AutoTraderConfig(BaseModel):
    enabled: bool
    dry_run: bool = True


@app.get("/api/autotrader")
async def autotrader_status():
    """자동매매 현재 설정 + 최근 10건 매매 기록."""
    if not _AUTO_TRADER_AVAILABLE or _auto_trader is None:
        raise HTTPException(status_code=503, detail="auto_trader 모듈을 로드할 수 없습니다.")
    return _auto_trader.get_status()


@app.post("/api/autotrader")
async def autotrader_config(body: AutoTraderConfig):
    """자동매매 ON/OFF 및 드라이런 설정."""
    if not _AUTO_TRADER_AVAILABLE or _auto_trader is None:
        raise HTTPException(status_code=503, detail="auto_trader 모듈을 로드할 수 없습니다.")
    result = _auto_trader.set_config(enabled=body.enabled, dry_run=body.dry_run)
    return result


@app.get("/api/autotrader/trades")
async def autotrader_trades(limit: int = 50):
    """매매 로그 조회 (최근 N건)."""
    if not _AUTO_TRADER_AVAILABLE or _auto_trader is None:
        raise HTTPException(status_code=503, detail="auto_trader 모듈을 로드할 수 없습니다.")
    return {"trades": _auto_trader.load_trade_log(limit=limit)}


@app.post("/api/autotrader/close")
async def autotrader_close():
    """현재 포지션 즉시 청산 (긴급 청산용)."""
    if not _AUTO_TRADER_AVAILABLE or _auto_trader is None:
        raise HTTPException(status_code=503, detail="auto_trader 모듈을 로드할 수 없습니다.")
    import trader as _trader_mod
    symbol = DEFAULT_SYMBOL
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, _trader_mod.close_position, symbol)
        return {"result": result or "포지션 없음"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/autotrader/balance")
async def autotrader_balance():
    """드라이런 복리 가상 잔고 조회."""
    try:
        import backtester as _bt
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return _bt.load_dry_balance()


@app.post("/api/autotrader/balance/reset")
async def autotrader_balance_reset(initial: float = 10000):
    """드라이런 가상 잔고 초기화 (initial: 초기 잔고 금액)."""
    try:
        import backtester as _bt
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    _bt.reset_dry_balance(initial)
    return {"ok": True, "balance": initial}


@app.get("/api/autotrader/summary")
async def autotrader_summary():
    """
    캔들 조회 없이 trade_log 만으로 즉시 반환하는 경량 통계.
    Bot Status 위젯에서 15초마다 폴링용.
    """
    try:
        import backtester as _bt
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, _bt.get_today_summary)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/autotrader/live-position")
async def autotrader_live_position():
    """
    현재 열려있는 드라이런 포지션 조회.
    포지션 없으면 {"position": null} 반환.
    """
    try:
        import backtester as _bt
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=f"backtester 모듈 로드 실패: {exc}")

    loop = asyncio.get_event_loop()
    try:
        pos = await loop.run_in_executor(None, _bt.get_live_position)
        return {"position": pos}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/autotrader/backtest")
async def autotrader_backtest(limit: int = 200, dry_only: bool = False, force: bool = False):
    """
    백테스트 결과 조회.
    - 기본: 서버가 미리 저장해둔 캐시를 즉시 반환 (캔들 조회 없음)
    - force=true: 강제로 전체 재실행 (느림, 수동 디버그용)
    """
    try:
        import backtester as _bt
        from dataclasses import asdict
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=f"backtester 모듈 로드 실패: {exc}")

    loop = asyncio.get_event_loop()
    try:
        if force:
            summary = await loop.run_in_executor(
                None, lambda: _bt.run_backtest(limit=limit, dry_run_only=dry_only)
            )
        else:
            # 캐시 우선 반환 → 없으면 증분 실행
            cached = await loop.run_in_executor(None, _bt.get_cached_backtest)
            if cached is not None:
                summary = cached
            else:
                summary = await loop.run_in_executor(
                    None, lambda: _bt.run_backtest_incremental(dry_run_only=True)
                )
        return asdict(summary)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
