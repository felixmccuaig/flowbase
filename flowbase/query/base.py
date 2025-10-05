"""Base query engine interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd


class QueryEngine(ABC):
    """Abstract base class for SQL query engines."""

    @abstractmethod
    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Execute a SQL query and return results as a DataFrame.

        Args:
            query: SQL query string
            params: Optional query parameters

        Returns:
            Query results as a pandas DataFrame
        """
        pass

    @abstractmethod
    def execute_many(self, queries: List[str]) -> List[pd.DataFrame]:
        """
        Execute multiple SQL queries.

        Args:
            queries: List of SQL query strings

        Returns:
            List of DataFrames, one per query
        """
        pass

    @abstractmethod
    def create_table(
        self, table_name: str, df: pd.DataFrame, if_exists: str = "replace"
    ) -> None:
        """
        Create a table from a DataFrame.

        Args:
            table_name: Name of the table to create
            df: DataFrame to insert
            if_exists: Action if table exists ('fail', 'replace', 'append')
        """
        pass

    @abstractmethod
    def register_file(
        self, name: str, file_path: str, file_format: str = "parquet"
    ) -> None:
        """
        Register a file as a table/view.

        Args:
            name: Table name to use in queries
            file_path: Path to the data file
            file_format: File format (parquet, csv, json, etc.)
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the connection to the query engine."""
        pass
