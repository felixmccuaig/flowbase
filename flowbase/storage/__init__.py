"""Storage abstraction layer for local and cloud storage."""

from flowbase.storage.base import StorageBackend
from flowbase.storage.local.filesystem import LocalFileSystem

__all__ = ["StorageBackend", "LocalFileSystem"]
