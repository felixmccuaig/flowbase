"""SQL query engine abstraction."""

from flowbase.query.base import QueryEngine
from flowbase.query.engines.duckdb_engine import DuckDBEngine

__all__ = ["QueryEngine", "DuckDBEngine"]
