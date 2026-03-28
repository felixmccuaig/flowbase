"""Generic entity-state and replay interfaces for incremental pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, Sequence


@dataclass
class StateEvent:
    """A generic ordered event used to update entity state."""

    entity_key: str
    event_time: datetime
    payload: dict[str, Any] = field(default_factory=dict)
    event_id: str | None = None


@dataclass
class StateCheckpoint:
    """A persisted snapshot of entity state as of a point in time."""

    entity_key: str
    as_of_time: datetime
    state: dict[str, Any] = field(default_factory=dict)
    checkpoint_id: str | None = None


@dataclass
class StateUpdateResult:
    """Result of applying a sequence of events to entity state."""

    entity_key: str
    latest_state: dict[str, Any] = field(default_factory=dict)
    history_rows: list[dict[str, Any]] = field(default_factory=list)
    checkpoints: list[StateCheckpoint] = field(default_factory=list)


class EntityStateBackend(Protocol):
    """Backend interface for latest-state and checkpoint persistence."""

    def load_latest_state(self, entity_key: str) -> StateCheckpoint | None:
        """Load the latest persisted state for an entity."""

    def load_checkpoint_before(self, entity_key: str, as_of_time: datetime) -> StateCheckpoint | None:
        """Load the nearest checkpoint before the given time."""

    def persist_update(self, update: StateUpdateResult) -> None:
        """Persist latest state, history rows, and checkpoints from an update."""


class EntityStateProcessor(Protocol):
    """Pure state-transition logic over ordered events."""

    def apply_events(
        self,
        *,
        entity_key: str,
        checkpoint: StateCheckpoint | None,
        events: Sequence[StateEvent],
    ) -> StateUpdateResult:
        """Apply ordered events to state and return updated results."""
