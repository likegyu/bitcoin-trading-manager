from __future__ import annotations

import json
import math
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any

WINDOW_CONFIGS = (
    {"key": "day", "hours": 24, "label": "24시간 거시 추이"},
    {"key": "swing", "hours": 72, "label": "72시간 거시 추이"},
    {"key": "week", "hours": 168, "label": "7일 거시 추이"},
)

METRIC_CONFIG = {
    "DFII10": {"label": "10Y 실질금리", "threshold": 0.03},
    "DGS2": {"label": "2Y 국채금리", "threshold": 0.03},
    "DTWEXBGS": {"label": "달러 인덱스", "threshold": 0.15},
    "STABLE_MCAP": {"label": "스테이블 시총", "threshold": 0.20},
    "USDT_DOM": {"label": "USDT 도미넌스", "threshold": 0.10},
    "BTC_DOM": {"label": "BTC 도미넌스", "threshold": 0.10},
}

RETENTION_HOURS = 240
MAX_ENTRIES = 8000
MIN_RECORD_INTERVAL_SECS = 30 * 60
FORCE_RECORD_INTERVAL_SECS = 2 * 60 * 60
COMPACT_EVERY_WRITES = 25

_BASE_DIR = Path(__file__).resolve().parent
_HISTORY_FILE = _BASE_DIR / "data" / "macro_history.jsonl"


def _as_float(value: Any) -> float | None:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num):
        return None
    return num


def _fmt_duration(minutes: int) -> str:
    if minutes <= 0:
        return "0분"
    hours, mins = divmod(minutes, 60)
    if hours and mins:
        return f"{hours}시간 {mins}분"
    if hours:
        return f"{hours}시간"
    return f"{mins}분"


def _metric_label(metric: str) -> str:
    return METRIC_CONFIG.get(metric, {}).get("label", metric)


def _metric_threshold(metric: str) -> float:
    return float(METRIC_CONFIG.get(metric, {}).get("threshold", 0.1))


def _format_delta(metric: str, value: float | None) -> str:
    if value is None:
        return "N/A"
    if metric in {"DFII10", "DGS2", "USDT_DOM", "BTC_DOM"}:
        return f"{value:+.2f}%p"
    if metric == "STABLE_MCAP":
        return f"{value:+.2f}B"
    return f"{value:+.2f}"


def _classify_delta(metric: str, value: float | None) -> str | None:
    if value is None:
        return None
    threshold = _metric_threshold(metric)
    if value > threshold:
        return "상승"
    if value < -threshold:
        return "하락"
    return "중립"


def _snapshot_from_macro(macro: dict, observed_at: datetime | None = None) -> dict:
    observed = observed_at or datetime.now(timezone.utc)
    snapshot = {
        "observed_at": observed.isoformat(),
        "observed_label": observed.strftime("%m-%d %H:%M"),
        "observed_ts": observed.timestamp(),
    }
    for metric in METRIC_CONFIG:
        data = macro.get(metric) or {}
        snapshot[metric] = _as_float(data.get("value"))
    return snapshot


def _has_observable_metric(snapshot: dict) -> bool:
    return any(snapshot.get(metric) is not None for metric in METRIC_CONFIG)


def _value_changed(prev: dict, curr: dict, metric: str) -> bool:
    prev_value = _as_float(prev.get(metric))
    curr_value = _as_float(curr.get(metric))
    if prev_value is None and curr_value is None:
        return False
    if prev_value is None or curr_value is None:
        return True
    return abs(curr_value - prev_value) >= _metric_threshold(metric)


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


def _metric_window(window: list[dict], metric: str) -> dict:
    points = []
    for entry in window:
        value = _as_float(entry.get(metric))
        if value is None:
            continue
        ts = _as_float(entry.get("observed_ts"))
        if ts is None:
            continue
        points.append((ts, value))

    if not points:
        return {
            "samples": 0,
            "start": None,
            "end": None,
            "change": None,
            "min": None,
            "max": None,
            "trend": None,
        }

    start_ts, start_value = points[0]
    end_ts, end_value = points[-1]
    values = [value for _, value in points]
    change = None
    if len(points) >= 2 and end_ts > start_ts:
        change = end_value - start_value

    return {
        "samples": len(points),
        "start": start_value,
        "end": end_value,
        "change": change,
        "min": min(values),
        "max": max(values),
        "trend": _classify_delta(metric, change),
    }


def _section_lines(metrics: dict[str, dict], include_rates: bool = True) -> tuple[list[str], list[str]]:
    lines: list[str] = []
    highlights: list[str] = []

    if include_rates:
        rate_parts = []
        for metric in ("DFII10", "DGS2", "DTWEXBGS"):
            change = metrics[metric]["change"]
            if change is None:
                continue
            rate_parts.append(f"{_metric_label(metric)} {_format_delta(metric, change)}")
        if rate_parts:
            lines.append("금리·달러: " + " / ".join(rate_parts))
            highlights.extend(rate_parts[:2])

    crypto_parts = []
    for metric in ("STABLE_MCAP", "USDT_DOM", "BTC_DOM"):
        change = metrics[metric]["change"]
        if change is None:
            continue
        crypto_parts.append(f"{_metric_label(metric)} {_format_delta(metric, change)}")
    if crypto_parts:
        lines.append("크립토 자금지표: " + " / ".join(crypto_parts))
        highlights.extend(crypto_parts)

    return lines, highlights


def _build_section(window: list[dict], config: dict) -> dict:
    observed_minutes = max(
        0,
        int(round((window[-1]["observed_ts"] - window[0]["observed_ts"]) / 60)),
    )
    metrics = {metric: _metric_window(window, metric) for metric in METRIC_CONFIG}
    meta = (
        f"실관찰 {_fmt_duration(observed_minutes)} · 표본 {len(window)}개 · "
        f"{window[0]['observed_label']}~{window[-1]['observed_label']} UTC"
    )

    lines: list[str] = [f"관찰 구간: {meta}"]
    delta_lines, highlights = _section_lines(metrics, include_rates=True)
    lines.extend(delta_lines)

    if len(lines) == 1:
        lines.append("표본 부족 — 서버 가동 후 누적 중")

    return {
        "key": config["key"],
        "label": config["label"],
        "window_hours": config["hours"],
        "meta": meta,
        "samples": len(window),
        "observed_minutes": observed_minutes,
        "observed_from": window[0]["observed_label"],
        "observed_to": window[-1]["observed_label"],
        "metrics": metrics,
        "lines": lines,
        "highlights": _recent_unique(highlights, limit=4),
    }


class MacroHistoryTimeline:
    def __init__(self, history_file: Path):
        self._history_file = history_file
        self._lock = Lock()
        self._entries: deque[dict] = deque(maxlen=MAX_ENTRIES)
        self._loaded = False
        self._writes_since_compact = 0

    def observe(self, macro: dict) -> dict:
        snapshot = _snapshot_from_macro(macro)
        current = snapshot if _has_observable_metric(snapshot) else None

        with self._lock:
            self._ensure_loaded_locked()
            if current is not None:
                prev = self._entries[-1] if self._entries else None
                if self._should_store(prev, current):
                    self._store_locked(current)
            summary = self._build_summary_locked(current)

        self._attach_to_macro(macro, summary)
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
        if elapsed >= FORCE_RECORD_INTERVAL_SECS:
            return True
        if elapsed < MIN_RECORD_INTERVAL_SECS:
            return False

        for metric in METRIC_CONFIG:
            if _value_changed(prev, curr, metric):
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
            return {"available": False, "sections": [], "windows": {}}

        latest = timeline[-1]
        latest_dt = datetime.fromtimestamp(latest["observed_ts"], tz=timezone.utc)

        sections: list[dict] = []
        windows: dict[str, dict] = {}
        for config in WINDOW_CONFIGS:
            cutoff = latest_dt - timedelta(hours=config["hours"])
            window = [
                entry
                for entry in timeline
                if datetime.fromtimestamp(entry["observed_ts"], tz=timezone.utc) >= cutoff
            ]
            if not window:
                window = [latest]

            section = _build_section(window, config)
            sections.append(section)
            windows[config["key"]] = section

        return {
            "available": True,
            "sections": sections,
            "windows": windows,
        }

    def _attach_to_macro(self, macro: dict, summary: dict) -> None:
        windows = summary.get("windows") or {}
        window_field_map = {
            "day": "change24h",
            "swing": "change72h",
            "week": "change7d",
        }

        for metric in METRIC_CONFIG:
            data = macro.get(metric)
            if not isinstance(data, dict):
                continue

            for window_key, field_name in window_field_map.items():
                window = windows.get(window_key) or {}
                metric_stats = (window.get("metrics") or {}).get(metric) or {}
                data[field_name] = metric_stats.get("change")
                data[f"{field_name}_samples"] = metric_stats.get("samples")

            week_stats = ((windows.get("week") or {}).get("metrics") or {}).get(metric) or {}
            data["trend7d"] = week_stats.get("trend")

        macro["_history_summary"] = summary


_TIMELINE = MacroHistoryTimeline(_HISTORY_FILE)


def attach_macro_history_summary(macro: dict) -> dict:
    _TIMELINE.observe(macro)
    return macro
