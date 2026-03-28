"""Sink implementations for Flowbase observability events."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from decimal import Decimal
from pathlib import Path
from typing import Any

try:
    import boto3
except ModuleNotFoundError:  # pragma: no cover
    boto3 = None


class ObservabilitySink(ABC):
    """Abstract sink for observability events."""

    @abstractmethod
    def emit(self, event: dict[str, Any]) -> None:
        """Persist an observability event."""


class NullSink(ObservabilitySink):
    """No-op sink used when observability is disabled."""

    def emit(self, event: dict[str, Any]) -> None:
        return None


class FileSink(ObservabilitySink):
    """Append observability events to a JSONL file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def emit(self, event: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, default=str))
            handle.write("\n")


class DynamoDBSink(ObservabilitySink):
    """Write observability events to DynamoDB."""

    def __init__(self, *, table_name: str, region: str | None = None):
        if boto3 is None:  # pragma: no cover
            raise RuntimeError("boto3 is required for DynamoDB observability")
        kwargs: dict[str, Any] = {}
        if region:
            kwargs["region_name"] = region
        self.table = boto3.resource("dynamodb", **kwargs).Table(table_name)

    def emit(self, event: dict[str, Any]) -> None:
        self.table.put_item(Item=_to_dynamodb_item(event))


def _to_dynamodb_item(value: Any) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, dict):
        return {str(key): _to_dynamodb_item(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_dynamodb_item(item) for item in value]
    return str(value)
