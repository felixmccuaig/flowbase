"""Incremental materializer implementations."""

from flowbase.incremental.materializers.base import IncrementalMaterializer, MaterializationResult
from flowbase.incremental.materializers.parquet import (
    KeyUpsertMaterializer,
    PartitionReplaceMaterializer,
)

__all__ = [
    "IncrementalMaterializer",
    "MaterializationResult",
    "KeyUpsertMaterializer",
    "PartitionReplaceMaterializer",
]
