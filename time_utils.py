from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")


def now_kst() -> datetime:
    return datetime.now(KST)


def to_kst(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(KST)


def start_of_kst_day(dt: datetime | None = None) -> datetime:
    base = to_kst(dt) if dt is not None else now_kst()
    return base.replace(hour=0, minute=0, second=0, microsecond=0)


def format_kst(dt: datetime, fmt: str = "%Y-%m-%d %H:%M") -> str:
    return to_kst(dt).strftime(fmt)
