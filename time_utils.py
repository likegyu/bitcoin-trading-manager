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


def format_kst(dt: datetime, fmt: str = "%Y-%m-%d %H:%M") -> str:
    return to_kst(dt).strftime(fmt)
