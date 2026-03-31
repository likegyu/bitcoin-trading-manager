#!/usr/bin/env python3
from __future__ import annotations

# =============================================
# BTC Signal Analyzer — FastAPI backend
# =============================================
import asyncio
import copy
import concurrent.futures
import contextlib
import json
import math
import os
from datetime import datetime

import pandas as pd
import websockets
from fastapi import FastAPI
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
from analyzer import analyze_with_claude, chat_with_claude
from macro_fetcher import fetch_macro_context

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
MARKET_STREAM_WS_BASE = "wss://fstream.binance.com/stream"
MACRO_REFRESH_SECS = 60 * 60


# ══════════════════════════════════════════════
# ECharts용 차트 데이터 직렬화
# ══════════════════════════════════════════════
def _series_values(values) -> list:
    return [_safe(v) for v in values]


def _series_labels(index) -> list[str]:
    labels = []
    for ts in index:
        try:
            labels.append(ts.strftime("%Y-%m-%d %H:%M"))
        except Exception:
            labels.append(str(ts)[:16])
    return labels


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
    h1   = tf_data.get("1h")
    if h1 is None:
        h1 = tf_data.get(TIMEFRAMES[0])
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

    overview = {
        "symbol":      DEFAULT_SYMBOL,
        "pair_label":  symbol_to_pair(DEFAULT_SYMBOL),
        "price":       price,
        "last_update": last_update or datetime.now().strftime("%H:%M:%S"),
        "indicators":  indicators,
    }
    return overview


def _build_account_payload() -> dict:
    ctx = fetch_account_context()
    return {
        "updated_at": datetime.now().strftime("%H:%M:%S"),
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
            "last_update": last_update or datetime.now().strftime("%H:%M:%S"),
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
        "analysis_time": datetime.now().strftime("%H:%M:%S"),
        "signal":       analysis["signal"],
        "confidence":   analysis["confidence"],
        "raw_text":     analysis["raw_text"],
        "trade_levels": analysis["trade_levels"],
        "report_sections": analysis.get("report_sections", {}),
        "report_format_ok": analysis.get("report_format_ok", False),
        "report_missing_sections": analysis.get("report_missing_sections", []),
        "account": _build_account_payload(),
        "chat_context": f"{analysis.get('prompt_used', '')}\n\n[Claude 분석 결과]\n{analysis['raw_text']}",
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

    async def start(self):
        if self._runner_task and not self._runner_task.done():
            return
        self._stopped = False
        self._runner_task = asyncio.create_task(self._run_forever(), name="binance-market-stream")

    async def stop(self):
        self._stopped = True
        tasks = [self._runner_task, self._market_flush_task, self._price_flush_task]
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
        await self.wait_until_ready()
        async with self._lock:
            tf_data = {tf: df.copy(deep=True) for tf, df in self._tf_data.items()}
            price = self._price
        return tf_data, price

    async def _run_forever(self):
        while not self._stopped:
            if not self._ready.is_set():
                try:
                    await self._bootstrap()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    print(f"[market-stream] bootstrap failed: {exc}")
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
        tf_data = await asyncio.to_thread(_fetch_all, self.symbol)
        try:
            price = await asyncio.to_thread(fetch_current_price, self.symbol)
        except Exception:
            price = None

        async with self._lock:
            self._tf_data = tf_data
            self._price = price
            self._last_update = datetime.now().strftime("%H:%M:%S")

        self._ready.set()

    def _stream_url(self) -> str:
        symbol = self.symbol.lower()
        streams = [f"{symbol}@aggTrade"] + [f"{symbol}@kline_{tf}" for tf in TIMEFRAMES]
        joined = "/".join(streams)
        return f"{MARKET_STREAM_WS_BASE}?streams={joined}"

    async def _consume_stream(self):
        async with websockets.connect(
            self._stream_url(),
            ping_interval=None,
            ping_timeout=None,
            close_timeout=5,
            max_size=2**20,
        ) as ws:
            async for raw_msg in ws:
                payload = json.loads(raw_msg)
                data = payload.get("data", payload)
                event_type = data.get("e")

                if event_type == "aggTrade":
                    await self._handle_trade(data)
                elif event_type == "kline":
                    await self._handle_kline(data.get("k", {}))

    async def _handle_trade(self, data: dict):
        price = _safe(data.get("p"))
        if price is None:
            return

        async with self._lock:
            self._price = price
            self._last_update = datetime.now().strftime("%H:%M:%S")

        self._schedule_price_flush()

    async def _handle_kline(self, kline: dict):
        tf = kline.get("i")
        if tf not in TIMEFRAMES:
            return

        timestamp = pd.to_datetime(int(kline["t"]), unit="ms")
        row = {
            "open": float(kline["o"]),
            "high": float(kline["h"]),
            "low": float(kline["l"]),
            "close": float(kline["c"]),
            "volume": float(kline["v"]),
        }

        async with self._lock:
            current = self._tf_data.get(tf)
            updated = _upsert_ohlcv(current, timestamp, row)
            updated = add_all_indicators(updated)
            self._tf_data[tf] = updated
            self._price = row["close"]
            self._last_update = datetime.now().strftime("%H:%M:%S")
            self._dirty_tfs.add(tf)
            if tf == "1h":
                self._needs_full_refresh = True

        self._schedule_market_flush()

    def _schedule_price_flush(self):
        if self._price_flush_task and not self._price_flush_task.done():
            return
        self._price_flush_task = asyncio.create_task(self._flush_price_update())

    def _schedule_market_flush(self):
        if self._market_flush_task and not self._market_flush_task.done():
            return
        self._market_flush_task = asyncio.create_task(self._flush_market_update())

    async def _flush_price_update(self):
        await asyncio.sleep(0.25)
        async with self._lock:
            payload = {
                "price": self._price,
                "last_update": self._last_update,
            }
        await self._broadcast({"type": "price", "data": payload})

    async def _flush_market_update(self):
        await asyncio.sleep(0.35)
        async with self._lock:
            dirty_tfs = set(self._dirty_tfs)
            full_refresh = self._needs_full_refresh
            self._dirty_tfs.clear()
            self._needs_full_refresh = False

            if not dirty_tfs and not full_refresh:
                return

            payload = build_market_payload(
                self._tf_data,
                self._price,
                self._last_update,
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


class AccountStreamManager:
    def __init__(self):
        self._payload: dict | None = None
        self._ready = asyncio.Event()
        self._lock = asyncio.Lock()
        self._listeners: set[asyncio.Queue] = set()
        self._runner_task: asyncio.Task | None = None
        self._refresh_task: asyncio.Task | None = None
        self._stopped = False

    async def start(self):
        if self._runner_task and not self._runner_task.done():
            return
        self._stopped = False
        self._runner_task = asyncio.create_task(self._run_forever(), name="binance-account-stream")

    async def stop(self):
        self._stopped = True
        tasks = [self._runner_task, self._refresh_task]
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
                    await asyncio.sleep(3)

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
            return
        self._refresh_task = asyncio.create_task(self._refresh_payload())

    async def _refresh_payload(self, delay: float = 0.35, broadcast: bool = True):
        if delay > 0:
            await asyncio.sleep(delay)
        payload = await asyncio.to_thread(_build_account_payload)
        async with self._lock:
            self._payload = payload
            self._ready.set()
        if broadcast:
            await self._broadcast({"type": "account", "data": payload})

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


# ══════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════
@app.on_event("startup")
async def on_startup():
    await _market_stream.start()
    await _account_stream.start()
    await _macro_snapshot.start()


@app.on_event("shutdown")
async def on_shutdown():
    await _market_stream.stop()
    await _account_stream.stop()
    await _macro_snapshot.stop()


@app.get("/api/market-stream")
async def market_stream():
    async def generate():
        queue = await _market_stream.subscribe()
        try:
            await _market_stream.wait_until_ready()
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


@app.get("/api/analyze")
async def analyze_stream():
    async def generate():
        loop = asyncio.get_event_loop()
        try:
            yield f"data: {json.dumps({'type':'progress','step':0})}\n\n"
            yield f"data: {json.dumps({'type':'progress','step':1})}\n\n"
            if _market_stream.is_ready():
                tf_data, price = await _market_stream.get_analysis_inputs()
            else:
                tf_data = await loop.run_in_executor(_executor, _fetch_all, DEFAULT_SYMBOL)
                try:
                    price = await loop.run_in_executor(_executor, fetch_current_price, DEFAULT_SYMBOL)
                except Exception:
                    price = None

            yield f"data: {json.dumps({'type':'progress','step':2})}\n\n"
            analysis = await loop.run_in_executor(_executor, analyze_with_claude, tf_data)

            yield f"data: {json.dumps({'type':'progress','step':3})}\n\n"
            payload = await loop.run_in_executor(
                _executor, _build_payload, tf_data, price, analysis
            )

            yield f"data: {json.dumps({'type':'result','data':payload})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type':'error','message':str(exc)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class ChatRequest(BaseModel):
    messages: list
    context:  str


@app.post("/api/chat")
async def chat_endpoint(body: ChatRequest):
    loop = asyncio.get_event_loop()
    reply = await loop.run_in_executor(
        _executor,
        lambda: chat_with_claude(body.messages, body.context),
    )
    return {"reply": reply}


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


@app.get("/")
async def root():
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(html_path, encoding="utf-8") as f:
        return HTMLResponse(f.read())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
