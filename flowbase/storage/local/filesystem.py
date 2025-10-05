"""Local filesystem storage implementation."""

import shutil
from pathlib import Path
from typing import BinaryIO, List, Optional, Union

from flowbase.storage.base import StorageBackend


class LocalFileSystem(StorageBackend):
    """Local filesystem storage backend."""

    def __init__(self, base_path: Optional[Union[str, Path]] = None):
        """
        Initialize local filesystem storage.

        Args:
            base_path: Base directory for storage. Defaults to ./data
        """
        self.base_path = Path(base_path) if base_path else Path.cwd() / "data"
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, path: str) -> Path:
        """Resolve a path relative to base_path."""
        full_path = self.base_path / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        return full_path

    def read(self, path: str) -> bytes:
        """Read file contents as bytes."""
        return self._resolve_path(path).read_bytes()

    def write(self, path: str, data: Union[bytes, BinaryIO]) -> None:
        """Write data to a file."""
        full_path = self._resolve_path(path)
        if isinstance(data, bytes):
            full_path.write_bytes(data)
        else:
            with open(full_path, "wb") as f:
                shutil.copyfileobj(data, f)

    def exists(self, path: str) -> bool:
        """Check if a file exists."""
        return self._resolve_path(path).exists()

    def list(self, path: str, pattern: Optional[str] = None) -> List[str]:
        """List files in a directory."""
        dir_path = self._resolve_path(path)
        if not dir_path.exists():
            return []

        if pattern:
            files = dir_path.glob(pattern)
        else:
            files = dir_path.iterdir()

        return [str(f.relative_to(self.base_path)) for f in files if f.is_file()]

    def delete(self, path: str) -> None:
        """Delete a file."""
        full_path = self._resolve_path(path)
        if full_path.exists():
            full_path.unlink()

    def get_uri(self, path: str) -> str:
        """Get the full URI for a path."""
        return f"file://{self._resolve_path(path).absolute()}"
