"""Generic entity-state interfaces for incremental feature pipelines."""

from .interfaces import (
    EntityStateBackend,
    EntityStateProcessor,
    StateCheckpoint,
    StateEvent,
    StateUpdateResult,
)

__all__ = [
    "EntityStateBackend",
    "EntityStateProcessor",
    "StateCheckpoint",
    "StateEvent",
    "StateUpdateResult",
]
