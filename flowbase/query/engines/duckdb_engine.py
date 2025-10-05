"""DuckDB query engine implementation."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import duckdb
import pandas as pd

from flowbase.query.base import QueryEngine


class DuckDBEngine(QueryEngine):
    """DuckDB-based query engine for local and S3 data."""

    def __init__(self, database: Optional[Union[str, Path]] = None, read_only: bool = False):
        """
        Initialize DuckDB query engine.

        Args:
            database: Path to DuckDB database file. Use None for in-memory.
            read_only: Open database in read-only mode
        """
        self.database = str(database) if database else ":memory:"
        self.read_only = read_only
        self.conn = duckdb.connect(self.database, read_only=read_only)

        # Configure for better performance
        try:
            import os
            # Use all available cores, but ensure at least 1
            threads = max(1, os.cpu_count() or 1)
            self.conn.execute(f"SET threads TO {threads}")
            self.conn.execute("SET memory_limit = '4GB'")
        except Exception:
            # If configuration fails, continue with defaults
            pass

    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Execute a SQL query and return results as a DataFrame."""
        if params:
            # DuckDB uses $param_name syntax for parameters
            formatted_params = {f"${k}": v for k, v in params.items()}
            result = self.conn.execute(query, formatted_params)
        else:
            result = self.conn.execute(query)

        return result.df()

    def execute_many(self, queries: List[str]) -> List[pd.DataFrame]:
        """Execute multiple SQL queries."""
        results = []
        for query in queries:
            results.append(self.execute(query))
        return results

    def create_table(
        self, table_name: str, df: pd.DataFrame, if_exists: str = "replace"
    ) -> None:
        """Create a table from a DataFrame."""
        if if_exists == "fail" and self._table_exists(table_name):
            raise ValueError(f"Table {table_name} already exists")
        elif if_exists == "replace":
            self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")

        self.conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")

    def register_file(
        self, name: str, file_path: str, file_format: str = "parquet"
    ) -> None:
        """Register a file as a table/view."""
        if file_format.lower() == "parquet":
            self.conn.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM '{file_path}'")
        elif file_format.lower() == "csv":
            self.conn.execute(
                f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_csv_auto('{file_path}')"
            )
        elif file_format.lower() == "json":
            self.conn.execute(
                f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_json_auto('{file_path}')"
            )
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

    def register_dataframe(self, name: str, df: pd.DataFrame) -> None:
        """Register a pandas DataFrame as a view."""
        self.conn.register(name, df)

    def list_tables(self) -> List[str]:
        """List all tables and views in the database."""
        result = self.conn.execute("SHOW TABLES").df()
        return result["name"].tolist() if not result.empty else []

    def _table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        return table_name in self.list_tables()

    def close(self) -> None:
        """Close the connection to DuckDB."""
        self.conn.close()

    def __enter__(self) -> "DuckDBEngine":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
