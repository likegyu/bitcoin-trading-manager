from __future__ import annotations

import json
import math
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any

LOOKBACK_HOURS = 12
RETENTION_HOURS = 72
MAX_ENTRIES = 1500
MIN_RECORD_INTERVAL_SECS = 60
FORCE_RECORD_INTERVAL_SECS = 15 * 60
COMPACT_EVERY_WRITES = 25

_BASE_DIR = Path(__file__).resolve().parent
_HISTORY_FILE = _BASE_DIR / "data" / "account_history.jsonl"


def _as_float(value: Any) -> float | None:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num):
        return None
    return num


def _fmt_money(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"${value:,.2f}"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:+.2f}%"


def _fmt_duration(minutes: int) -> str:
    if minutes <= 0:
        return "0분"
    hours, mins = divmod(minutes, 60)
    if hours and mins:
        return f"{hours}시간 {mins}분"
    if hours:
        return f"{hours}시간"
    return f"{mins}분"


def _delta_text(start: float | None, end: float | None) -> str:
    if start is None or end is None:
        return "N/A"
    delta = end - start
    if abs(delta) >= 1000:
        return f"{delta:+,.0f}"
    return f"{delta:+,.2f}"


def _status_label(status: str | None) -> str:
    mapping = {
        "active": "정상 운용",
        "target_hit": "일일 목표 도달",
        "loss_limit_hit": "일일 손실 한도 도달",
    }
    return mapping.get(status or "", "정보 없음")


def _bias_from_notional(long_notional: float | None, short_notional: float | None) -> str:
    long_v = long_notional or 0.0
    short_v = short_notional or 0.0
    total = long_v + short_v
    if total < 1:
        return "flat"
    dominance = abs(long_v - short_v) / total
    if dominance < 0.2:
        return "balanced"
    return "long" if long_v > short_v else "short"


def _bias_label(bias: str | None) -> str:
    mapping = {
        "long": "롱 우세",
        "short": "숏 우세",
        "balanced": "양방향 혼합",
        "flat": "무포지션",
    }
    return mapping.get(bias or "", "정보 없음")


def _first_value(entries: list[dict], field: str) -> float | None:
    for entry in entries:
        value = _as_float(entry.get(field))
        if value is not None:
            return value
    return None


def _last_value(entries: list[dict], field: str) -> float | None:
    for entry in reversed(entries):
        value = _as_float(entry.get(field))
        if value is not None:
            return value
    return None


def _series_range(entries: list[dict], field: str) -> tuple[float | None, float | None]:
    values = [_as_float(entry.get(field)) for entry in entries]
    valid = [value for value in values if value is not None]
    if not valid:
        return None, None
    return min(valid), max(valid)


def _position_symbol_list(positions: list[dict]) -> list[str]:
    symbols: list[str] = []
    for position in positions:
        symbol = str(position.get("symbol") or "").upper()
        if symbol and symbol not in symbols:
            symbols.append(symbol)
    return symbols


def _top_positions(positions: list[dict], limit: int = 3) -> list[str]:
    tops: list[str] = []
    for position in positions[:limit]:
        symbol = str(position.get("symbol") or "N/A").upper()
        side = str(position.get("side") or "포지션")
        notional = _as_float(position.get("notional"))
        if notional is None:
            tops.append(f"{symbol} {side}")
        else:
            tops.append(f"{symbol} {side} {_fmt_money(notional)}")
    return tops


def _position_signature(positions: list[dict]) -> str:
    keys: list[str] = []
    for position in positions:
        symbol = str(position.get("symbol") or "N/A").upper()
        side = str(position.get("side") or "N/A")
        size = _as_float(position.get("size")) or 0.0
        leverage = _as_float(position.get("leverage")) or 0.0
        keys.append(f"{symbol}:{side}:{size:.8f}:{leverage:.2f}")
    return "|".join(sorted(keys))


def _snapshot_from_context(ctx: dict, observed_at: datetime | None = None) -> dict:
    observed = observed_at or datetime.now(timezone.utc)
    positions = ctx.get("open_positions")
    position_list = positions if isinstance(positions, list) else []

    long_notional = 0.0
    short_notional = 0.0
    for position in position_list:
        notional = _as_float(position.get("notional")) or 0.0
        side = str(position.get("side") or "")
        if side == "숏":
            short_notional += notional
        else:
            long_notional += notional

    snapshot = {
        "observed_at": observed.isoformat(),
        "observed_label": observed.strftime("%m-%d %H:%M"),
        "observed_ts": observed.timestamp(),
        "wallet_balance": _as_float(ctx.get("wallet_balance")),
        "available_balance": _as_float(ctx.get("available_balance")),
        "margin_balance": _as_float(ctx.get("margin_balance")),
        "today_total_pnl": _as_float(ctx.get("today_total_pnl")),
        "today_pnl_pct": _as_float(ctx.get("today_pnl_pct")),
        "open_position_count": (
            int(ctx["open_position_count"])
            if ctx.get("open_position_count") is not None
            else None
        ),
        "open_position_notional": _as_float(ctx.get("open_position_notional")),
        "open_position_upnl": _as_float(ctx.get("open_position_upnl")),
        "effective_leverage": _as_float(ctx.get("effective_leverage")),
        "risk_status": ctx.get("risk_status"),
        "long_notional": round(long_notional, 2),
        "short_notional": round(short_notional, 2),
        "exposure_bias": _bias_from_notional(long_notional, short_notional),
        "position_symbols": _position_symbol_list(position_list),
        "top_positions": _top_positions(position_list),
        "position_signature": _position_signature(position_list),
    }
    return snapshot


def _is_observable(snapshot: dict) -> bool:
    for field in (
        "wallet_balance",
        "today_total_pnl",
        "open_position_count",
        "open_position_notional",
        "open_position_upnl",
        "effective_leverage",
    ):
        if snapshot.get(field) is not None:
            return True
    return bool(snapshot.get("position_signature"))


def _value_changed(prev: dict, curr: dict, field: str, threshold: float) -> bool:
    prev_value = _as_float(prev.get(field))
    curr_value = _as_float(curr.get(field))
    if prev_value is None and curr_value is None:
        return False
    if prev_value is None or curr_value is None:
        return True
    return abs(curr_value - prev_value) >= threshold


def _describe_transition(prev: dict, curr: dict) -> list[str]:
    events: list[str] = []

    prev_symbols = set(prev.get("position_symbols") or [])
    curr_symbols = set(curr.get("position_symbols") or [])
    added = sorted(curr_symbols - prev_symbols)
    removed = sorted(prev_symbols - curr_symbols)
    if added:
        events.append(f"신규 심볼 진입: {', '.join(added[:2])}")
    if removed:
        events.append(f"심볼 정리: {', '.join(removed[:2])}")

    prev_count = prev.get("open_position_count")
    curr_count = curr.get("open_position_count")
    if prev_count is not None and curr_count is not None and prev_count != curr_count:
        events.append(f"오픈 포지션 {prev_count}개 → {curr_count}개")

    prev_bias = prev.get("exposure_bias")
    curr_bias = curr.get("exposure_bias")
    if prev_bias and curr_bias and prev_bias != curr_bias:
        events.append(f"순노출 {_bias_label(prev_bias)} → {_bias_label(curr_bias)}")

    prev_notional = _as_float(prev.get("open_position_notional"))
    curr_notional = _as_float(curr.get("open_position_notional"))
    if prev_notional is not None and curr_notional is not None and prev_notional > 0:
        ratio = (curr_notional - prev_notional) / prev_notional * 100
        if abs(ratio) >= 25:
            direction = "확대" if ratio > 0 else "축소"
            events.append(f"총 명목 {direction} ({ratio:+.0f}%)")

    prev_lev = _as_float(prev.get("effective_leverage"))
    curr_lev = _as_float(curr.get("effective_leverage"))
    if prev_lev is not None and curr_lev is not None and abs(curr_lev - prev_lev) >= 0.75:
        events.append(f"실효 레버리지 {prev_lev:.1f}x → {curr_lev:.1f}x")

    prev_status = prev.get("risk_status")
    curr_status = curr.get("risk_status")
    if prev_status and curr_status and prev_status != curr_status:
        events.append(f"리스크 상태 {_status_label(prev_status)} → {_status_label(curr_status)}")

    return events


def _recent_unique(events: list[str], limit: int = 3) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for event in reversed(events):
        if not event or event in seen:
            continue
        result.append(event)
        seen.add(event)
        if len(result) >= limit:
            break
    return list(reversed(result))


class AccountContextTimeline:
    def __init__(self, history_file: Path):
        self._history_file = history_file
        self._lock = Lock()
        self._entries: deque[dict] = deque(maxlen=MAX_ENTRIES)
        self._loaded = False
        self._writes_since_compact = 0

    def observe(self, ctx: dict) -> dict:
        snapshot = _snapshot_from_context(ctx)
        current = snapshot if _is_observable(snapshot) else None

        with self._lock:
            self._ensure_loaded_locked()

            if current is not None:
                prev = self._entries[-1] if self._entries else None
                if self._should_store(prev, current):
                    self._store_locked(current)

            summary = self._build_summary_locked(current)

        ctx["context_summary"] = summary
        return summary

    def _ensure_loaded_locked(self) -> None:
        if self._loaded:
            return

        loaded: deque[dict] = deque(maxlen=MAX_ENTRIES)
        if self._history_file.exists():
            try:
                with self._history_file.open("r", encoding="utf-8") as handle:
                    for raw in handle:
                        raw = raw.strip()
                        if not raw:
                            continue
                        try:
                            loaded.append(json.loads(raw))
                        except json.JSONDecodeError:
                            continue
            except OSError:
                loaded.clear()

        self._entries = loaded
        if self._entries:
            pruned = self._prune_locked(datetime.now(timezone.utc).timestamp())
            if pruned:
                self._rewrite_locked()

        self._loaded = True

    def _should_store(self, prev: dict | None, curr: dict) -> bool:
        if prev is None:
            return True

        elapsed = curr["observed_ts"] - (prev.get("observed_ts") or 0)
        if curr.get("position_signature") != prev.get("position_signature"):
            return True
        if curr.get("risk_status") != prev.get("risk_status"):
            return True
        if curr.get("exposure_bias") != prev.get("exposure_bias"):
            return True
        if elapsed >= FORCE_RECORD_INTERVAL_SECS:
            return True
        if elapsed < MIN_RECORD_INTERVAL_SECS:
            return False

        thresholds = {
            "wallet_balance": 10.0,
            "today_total_pnl": 15.0,
            "open_position_notional": 75.0,
            "open_position_upnl": 25.0,
            "effective_leverage": 0.35,
            "long_notional": 75.0,
            "short_notional": 75.0,
        }
        for field, threshold in thresholds.items():
            if _value_changed(prev, curr, field, threshold):
                return True
        if prev.get("open_position_count") != curr.get("open_position_count"):
            return True
        return False

    def _store_locked(self, snapshot: dict) -> None:
        self._entries.append(snapshot)
        pruned = self._prune_locked(snapshot["observed_ts"])
        self._writes_since_compact += 1

        if pruned or self._writes_since_compact >= COMPACT_EVERY_WRITES:
            self._rewrite_locked()
            self._writes_since_compact = 0
            return

        try:
            self._history_file.parent.mkdir(parents=True, exist_ok=True)
            with self._history_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def _prune_locked(self, now_ts: float) -> bool:
        changed = False
        cutoff_ts = now_ts - RETENTION_HOURS * 3600
        while self._entries and (_as_float(self._entries[0].get("observed_ts")) or 0) < cutoff_ts:
            self._entries.popleft()
            changed = True
        while len(self._entries) > MAX_ENTRIES:
            self._entries.popleft()
            changed = True
        return changed

    def _rewrite_locked(self) -> None:
        try:
            self._history_file.parent.mkdir(parents=True, exist_ok=True)
            with self._history_file.open("w", encoding="utf-8") as handle:
                for entry in self._entries:
                    handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def _build_summary_locked(self, current: dict | None) -> dict:
        timeline = list(self._entries)
        if current is not None and (
            not timeline or current.get("observed_ts") != timeline[-1].get("observed_ts")
        ):
            timeline.append(current)

        if not timeline:
            return {
                "available": False,
                "window_hours": LOOKBACK_HOURS,
                "samples": 0,
                "recent_events": [],
                "top_positions": [],
                "lines": [],
            }

        latest = timeline[-1]
        cutoff = datetime.fromtimestamp(latest["observed_ts"], tz=timezone.utc) - timedelta(
            hours=LOOKBACK_HOURS
        )
        window = [
            entry
            for entry in timeline
            if datetime.fromtimestamp(entry["observed_ts"], tz=timezone.utc) >= cutoff
        ]
        if not window:
            window = [latest]

        observed_minutes = max(
            0,
            int(round((window[-1]["observed_ts"] - window[0]["observed_ts"]) / 60)),
        )

        wallet_start = _first_value(window, "wallet_balance")
        wallet_end = _last_value(window, "wallet_balance")
        wallet_min, wallet_max = _series_range(window, "wallet_balance")

        pnl_start = _first_value(window, "today_total_pnl")
        pnl_end = _last_value(window, "today_total_pnl")
        pnl_min, pnl_max = _series_range(window, "today_total_pnl")

        count_start = _first_value(window, "open_position_count")
        count_end = _last_value(window, "open_position_count")
        count_min, count_max = _series_range(window, "open_position_count")

        notional_start = _first_value(window, "open_position_notional")
        notional_end = _last_value(window, "open_position_notional")
        notional_min, notional_max = _series_range(window, "open_position_notional")

        lev_start = _first_value(window, "effective_leverage")
        lev_end = _last_value(window, "effective_leverage")
        lev_min, lev_max = _series_range(window, "effective_leverage")

        long_end = _as_float(latest.get("long_notional"))
        short_end = _as_float(latest.get("short_notional"))
        bias = latest.get("exposure_bias")
        top_positions = list(latest.get("top_positions") or [])

        transitions: list[str] = []
        recent_window = window[-8:]
        for prev, curr in zip(recent_window, recent_window[1:]):
            transitions.extend(_describe_transition(prev, curr))
        recent_events = _recent_unique(transitions, limit=3)

        lines: list[str] = []
        lines.append(
            "관찰 구간: "
            f"최근 {LOOKBACK_HOURS}시간 중 실관찰 {_fmt_duration(observed_minutes)} / "
            f"표본 {len(window)}개 / "
            f"{window[0]['observed_label']}~{window[-1]['observed_label']} UTC"
        )

        if wallet_start is not None and wallet_end is not None:
            lines.append(
                "담보 추이: "
                f"{_fmt_money(wallet_start)} → {_fmt_money(wallet_end)} "
                f"({_delta_text(wallet_start, wallet_end)}) / "
                f"구간 {_fmt_money(wallet_min)}~{_fmt_money(wallet_max)}"
            )

        if pnl_start is not None and pnl_end is not None:
            lines.append(
                "오늘 손익 추이: "
                f"{_fmt_money(pnl_start)} → {_fmt_money(pnl_end)} / "
                f"구간 {_fmt_money(pnl_min)}~{_fmt_money(pnl_max)} / "
                f"현재 {_fmt_pct(_as_float(latest.get('today_pnl_pct')))}"
            )

        if count_start is not None or notional_start is not None or lev_start is not None:
            parts: list[str] = []
            if count_start is not None and count_end is not None:
                parts.append(f"오픈 {int(count_start)}개 → {int(count_end)}개")
                if count_min is not None and count_max is not None and count_min != count_max:
                    parts.append(f"범위 {int(count_min)}~{int(count_max)}개")
            if notional_start is not None and notional_end is not None:
                parts.append(f"총 명목 {_fmt_money(notional_start)} → {_fmt_money(notional_end)}")
                if notional_max is not None:
                    parts.append(f"최대 {_fmt_money(notional_max)}")
            if lev_end is not None:
                if lev_min is not None and lev_max is not None and abs(lev_max - lev_min) >= 0.15:
                    parts.append(f"실효 레버리지 {lev_min:.1f}x~{lev_max:.1f}x")
                else:
                    parts.append(f"실효 레버리지 {lev_end:.1f}x")
            if parts:
                lines.append("포지션 추이: " + " / ".join(parts))

        exposure_line = (
            "현재 노출: "
            f"롱 {_fmt_money(long_end)} / 숏 {_fmt_money(short_end)} / {_bias_label(bias)}"
        )
        if top_positions:
            exposure_line += f" / 상위 {', '.join(top_positions[:2])}"
        lines.append(exposure_line)

        if recent_events:
            lines.append("최근 변화: " + " · ".join(recent_events))

        return {
            "available": True,
            "window_hours": LOOKBACK_HOURS,
            "samples": len(window),
            "observed_minutes": observed_minutes,
            "observed_from": window[0]["observed_label"],
            "observed_to": window[-1]["observed_label"],
            "wallet_start": wallet_start,
            "wallet_end": wallet_end,
            "wallet_min": wallet_min,
            "wallet_max": wallet_max,
            "today_total_pnl_start": pnl_start,
            "today_total_pnl_end": pnl_end,
            "today_total_pnl_min": pnl_min,
            "today_total_pnl_max": pnl_max,
            "open_position_count_start": int(count_start) if count_start is not None else None,
            "open_position_count_end": int(count_end) if count_end is not None else None,
            "open_position_notional_start": notional_start,
            "open_position_notional_end": notional_end,
            "open_position_notional_max": notional_max,
            "effective_leverage_start": lev_start,
            "effective_leverage_end": lev_end,
            "effective_leverage_min": lev_min,
            "effective_leverage_max": lev_max,
            "long_notional": long_end,
            "short_notional": short_end,
            "exposure_bias": bias,
            "risk_status": latest.get("risk_status"),
            "recent_events": recent_events,
            "top_positions": top_positions,
            "lines": lines,
        }


_TIMELINE = AccountContextTimeline(_HISTORY_FILE)


def attach_account_context_summary(ctx: dict) -> dict:
    _TIMELINE.observe(ctx)
    return ctx
