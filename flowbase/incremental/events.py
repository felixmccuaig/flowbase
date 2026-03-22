"""Incremental change event contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ChangeEvent:
    """A normalized change event used by the incremental planner."""

    source_name: str
    change_type: str
    primary_key: Dict[str, Any] = field(default_factory=dict)
    entity_keys: Dict[str, Any] = field(default_factory=dict)
    partition_keys: Dict[str, Any] = field(default_factory=dict)
    event_time: Optional[str] = None
    observed_at: Optional[str] = None
    version: Optional[str] = None
    idempotency_key: Optional[str] = None
    payload_ref: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChangeEvent":
        source_name = data.get("source_name") or data.get("source")
        if not source_name:
            raise ValueError("ChangeEvent requires 'source_name' or 'source'")
        return cls(
            source_name=str(source_name),
            change_type=str(data.get("change_type", "upsert")),
            primary_key=dict(data.get("primary_key", {})),
            entity_keys=dict(data.get("entity_keys", {})),
            partition_keys=dict(data.get("partition_keys", {})),
            event_time=data.get("event_time"),
            observed_at=data.get("observed_at"),
            version=data.get("version"),
            idempotency_key=data.get("idempotency_key"),
            payload_ref=data.get("payload_ref"),
        )
