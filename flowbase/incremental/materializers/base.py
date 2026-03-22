"""Base contracts for incremental materializers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd


@dataclass
class MaterializationResult:
    """Result of an incremental materialization."""

    strategy: str
    output_path: str
    rows_written: int
    partitions_touched: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class IncrementalMaterializer:
    """Abstract incremental materializer."""

    strategy_name: str = "unknown"

    def execute(self, df: pd.DataFrame, **kwargs: Any) -> MaterializationResult:
        raise NotImplementedError
