"""Base storage backend interface."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, BinaryIO, List, Optional, Union


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def read(self, path: str) -> bytes:
        """Read file contents as bytes."""
        pass

    @abstractmethod
    def write(self, path: str, data: Union[bytes, BinaryIO]) -> None:
        """Write data to a file."""
        pass

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if a file exists."""
        pass

    @abstractmethod
    def list(self, path: str, pattern: Optional[str] = None) -> List[str]:
        """List files in a directory."""
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        """Delete a file."""
        pass

    @abstractmethod
    def get_uri(self, path: str) -> str:
        """Get the full URI for a path."""
        pass
