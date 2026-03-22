"""Manifest-based resolution of physical table objects for logical date slices."""

from __future__ import annotations

import json
import re
from calendar import monthrange
from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


class Tier(str, Enum):
    DAILY = "daily"
    MONTHLY = "monthly"
    YEARLY = "yearly"


class ResolutionPolicy(str, Enum):
    DAILY_PREFER = "daily_prefer"
    MONTHLY_PREFER = "monthly_prefer"
    YEARLY_PREFER = "yearly_prefer"
    COST_OPTIMIZED = "cost_optimized"


@dataclass(frozen=True)
class ManifestEntry:
    table: str
    key: str
    tier: Tier
    start_date: date
    end_date: date
    row_count: Optional[int] = None
    checksum: Optional[str] = None

    def covers(self, value: date) -> bool:
        return self.start_date <= value <= self.end_date


_SUFFIX_DATE_DAILY = re.compile(r".*_(\d{4})_(\d{2})_(\d{2})\.[^.]+$")
_SUFFIX_DATE_MONTHLY = re.compile(r".*_(\d{4})_(\d{2})\.[^.]+$")
_SUFFIX_DATE_YEARLY = re.compile(r".*_(\d{4})\.[^.]+$")


def infer_entry_from_key(table: str, key: str) -> Optional[ManifestEntry]:
    """Infer manifest metadata from filename suffix convention."""
    key = key.strip()
    if not key:
        return None

    m_daily = _SUFFIX_DATE_DAILY.match(key)
    if m_daily:
        y, mo, d = (int(m_daily.group(1)), int(m_daily.group(2)), int(m_daily.group(3)))
        dt = date(y, mo, d)
        return ManifestEntry(table=table, key=key, tier=Tier.DAILY, start_date=dt, end_date=dt)

    m_month = _SUFFIX_DATE_MONTHLY.match(key)
    if m_month:
        y, mo = (int(m_month.group(1)), int(m_month.group(2)))
        start = date(y, mo, 1)
        end = date(y, mo, monthrange(y, mo)[1])
        return ManifestEntry(table=table, key=key, tier=Tier.MONTHLY, start_date=start, end_date=end)

    m_year = _SUFFIX_DATE_YEARLY.match(key)
    if m_year:
        y = int(m_year.group(1))
        return ManifestEntry(
            table=table,
            key=key,
            tier=Tier.YEARLY,
            start_date=date(y, 1, 1),
            end_date=date(y, 12, 31),
        )

    return None


def _tier_priority(policy: ResolutionPolicy, tier: Tier) -> int:
    priorities = {
        ResolutionPolicy.DAILY_PREFER: {Tier.DAILY: 0, Tier.MONTHLY: 1, Tier.YEARLY: 2},
        ResolutionPolicy.MONTHLY_PREFER: {Tier.MONTHLY: 0, Tier.DAILY: 1, Tier.YEARLY: 2},
        ResolutionPolicy.YEARLY_PREFER: {Tier.YEARLY: 0, Tier.MONTHLY: 1, Tier.DAILY: 2},
        ResolutionPolicy.COST_OPTIMIZED: {Tier.YEARLY: 0, Tier.MONTHLY: 1, Tier.DAILY: 2},
    }
    return priorities[policy][tier]


def resolve_for_date(
    entries: Sequence[ManifestEntry],
    value: date,
    policy: ResolutionPolicy = ResolutionPolicy.DAILY_PREFER,
) -> Optional[ManifestEntry]:
    """Resolve best covering object for a target date."""
    candidates = [e for e in entries if e.covers(value)]
    if not candidates:
        return None

    def sort_key(entry: ManifestEntry) -> tuple[int, int, int]:
        span_days = (entry.end_date - entry.start_date).days
        distance_from_slice = abs((value - entry.start_date).days)
        return (_tier_priority(policy, entry.tier), span_days, distance_from_slice)

    return sorted(candidates, key=sort_key)[0]


class JsonManifestStore:
    """JSON-backed manifest store for local development."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load_entries(self, table: str) -> List[ManifestEntry]:
        if not self.path.exists():
            return []
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        raw = payload.get("entries", [])
        result: List[ManifestEntry] = []
        for row in raw:
            if row.get("table") != table:
                continue
            result.append(
                ManifestEntry(
                    table=row["table"],
                    key=row["key"],
                    tier=Tier(row["tier"]),
                    start_date=date.fromisoformat(row["start_date"]),
                    end_date=date.fromisoformat(row["end_date"]),
                    row_count=row.get("row_count"),
                    checksum=row.get("checksum"),
                )
            )
        return result

    def resolve_for_date(
        self,
        table: str,
        value: date,
        policy: ResolutionPolicy = ResolutionPolicy.DAILY_PREFER,
        discovered_entries: Optional[Iterable[ManifestEntry]] = None,
    ) -> Optional[ManifestEntry]:
        entries = self.load_entries(table)
        if discovered_entries:
            entries.extend(list(discovered_entries))
        return resolve_for_date(entries, value, policy=policy)
