"""Factory for creating query engines with configuration."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

from flowbase.query.engines.duckdb_engine import DuckDBEngine


def create_query_engine(
    engine_type: str = "duckdb",
    database: Optional[Union[str, Path]] = None,
    read_only: bool = False,
    config: Optional[Dict[str, Any]] = None,
) -> DuckDBEngine:
    """
    Create a query engine with the specified configuration.

    Args:
        engine_type: Type of query engine ('duckdb')
        database: Path to database file (None for in-memory)
        read_only: Open database in read-only mode
        config: Query engine configuration dict

    Returns:
        Configured QueryEngine instance

    Raises:
        ValueError: If engine_type is not supported
    """
    if engine_type.lower() == "duckdb":
        return DuckDBEngine(database=database, read_only=read_only, config=config)
    else:
        raise ValueError(f"Unsupported query engine type: {engine_type}")
