"""Incremental execution work-unit definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class WorkUnit:
    """A concrete recompute instruction emitted by the planner."""

    node_name: str
    grain_type: str
    keys: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    source_change_ids: List[str] = field(default_factory=list)
