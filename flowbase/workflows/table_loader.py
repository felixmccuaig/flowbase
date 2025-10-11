"""Table loader for DuckDB - loads tables from S3 or local filesystem."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Dict, Any

from flowbase.core.config.schemas import SourceType

if TYPE_CHECKING:
    from flowbase.query.engines.duckdb_engine import DuckDBEngine
    from flowbase.workflows.dependency_resolver import TableDependency


class TableLoader:
    """Loads tables into DuckDB from various sources."""

    def __init__(self, engine: DuckDBEngine, project_config: Optional[Dict[str, Any]] = None):
        self.engine = engine
        self.logger = logging.getLogger(__name__)
        self.project_config = project_config or {}

        # Check if S3 sync is enabled from project config
        self.s3_enabled = self.project_config.get('sync_artifacts', False)
        storage_config = self.project_config.get('storage', {})
        self.s3_bucket = storage_config.get('bucket') if isinstance(storage_config, dict) else None
        self.s3_prefix = storage_config.get('prefix', '') if isinstance(storage_config, dict) else ''

        if self.s3_enabled and self.s3_bucket:
            self.logger.info(f"S3 sync enabled for tables: s3://{self.s3_bucket}/{self.s3_prefix}")

    def load_table(self, table: TableDependency) -> None:
        """
        Load a table into DuckDB.

        Args:
            table: TableDependency containing config
        """
        self.logger.info(f"Loading table: {table.name}")

        config = table.config

        # Priority: 1) Explicit S3 source in table config, 2) Project-level S3 sync, 3) Local
        if config.source and config.source.type == SourceType.S3:
            # Explicit S3 source in table config
            self._load_from_s3(table)
        elif self.s3_enabled and self.s3_bucket:
            # Project-level S3 sync enabled - auto-construct S3 path from storage.base_path
            self._load_from_s3_auto(table)
        else:
            # Local source
            self._load_from_local(table)

    def _load_from_s3_auto(self, table: TableDependency) -> None:
        """Load table from S3 using auto-constructed path from project config."""
        config = table.config
        storage_config = config.storage_config

        if not storage_config:
            self.logger.error(f"No storage config found for table {table.name}")
            return

        # Auto-construct S3 path: s3://{bucket}/{prefix}/{storage.base_path}
        base_path = storage_config.base_path
        s3_prefix = f"{self.s3_prefix}/{base_path}".strip("/")

        self.logger.info(f"Loading table {table.name} from S3: s3://{self.s3_bucket}/{s3_prefix}")

        # Install and load httpfs extension for S3 support
        try:
            self.engine.execute("INSTALL httpfs;")
            self.engine.execute("LOAD httpfs;")
        except Exception as e:
            self.logger.debug(f"httpfs extension already installed: {e}")

        file_format = storage_config.file_format.value

        # Construct S3 path pattern
        s3_pattern = f"s3://{self.s3_bucket}/{s3_prefix}/*.{file_format}"

        # Create view from S3 files
        self.logger.info(f"Creating view '{table.name}' from {s3_pattern}")

        if file_format == 'parquet':
            read_func = f"read_parquet('{s3_pattern}')"
        elif file_format == 'csv':
            read_func = f"read_csv_auto('{s3_pattern}')"
        else:
            self.logger.error(f"Unsupported file format: {file_format}")
            return

        create_view_sql = f"CREATE OR REPLACE VIEW {table.name} AS SELECT * FROM {read_func}"

        try:
            self.engine.execute(create_view_sql)
            self.logger.info(f"Successfully loaded table {table.name} from S3")
        except Exception as e:
            self.logger.error(f"Failed to load table {table.name} from S3: {e}")
            # Try local fallback
            self.logger.info(f"Attempting local fallback for table {table.name}")
            self._load_from_local(table)

    def _load_from_s3(self, table: TableDependency) -> None:
        """Load table from S3 using explicit S3 source config."""
        config = table.config
        bucket = config.source.bucket
        prefix = config.source.prefix

        self.logger.info(f"Loading table {table.name} from S3: s3://{bucket}/{prefix}")

        # Install and load httpfs extension for S3 support
        try:
            self.engine.execute("INSTALL httpfs;")
            self.engine.execute("LOAD httpfs;")
        except Exception as e:
            self.logger.debug(f"httpfs extension already installed: {e}")

        # Get file format
        storage_config = config.storage_config
        if not storage_config:
            self.logger.error(f"No storage config found for table {table.name}")
            return

        file_format = storage_config.file_format.value

        # Construct S3 path pattern
        s3_pattern = f"s3://{bucket}/{prefix}*.{file_format}"

        # Create view from S3 files
        self.logger.info(f"Creating view '{table.name}' from {s3_pattern}")

        if file_format == 'parquet':
            read_func = f"read_parquet('{s3_pattern}')"
        elif file_format == 'csv':
            read_func = f"read_csv_auto('{s3_pattern}')"
        else:
            self.logger.error(f"Unsupported file format: {file_format}")
            return

        create_view_sql = f"CREATE OR REPLACE VIEW {table.name} AS SELECT * FROM {read_func}"

        try:
            self.engine.execute(create_view_sql)
            self.logger.info(f"Successfully loaded table {table.name} from S3")
        except Exception as e:
            self.logger.error(f"Failed to load table {table.name} from S3: {e}")
            # Try local fallback
            self.logger.info(f"Attempting local fallback for table {table.name}")
            self._load_from_local(table)

    def _load_from_local(self, table: TableDependency) -> None:
        """Load table from local filesystem."""
        config = table.config
        storage_config = config.storage_config

        if not storage_config:
            self.logger.error(f"No storage config found for table {table.name}")
            return

        base_path = Path(storage_config.base_path)

        if not base_path.exists():
            self.logger.warning(f"Table path does not exist: {base_path}")
            # Create empty view as placeholder
            self.logger.info(f"Skipping table {table.name} - no data available")
            return

        self.logger.info(f"Loading table {table.name} from local: {base_path}")

        file_format = storage_config.file_format.value

        # Construct file pattern
        if file_format == 'parquet':
            pattern = str(base_path / "*.parquet")
            read_function = f"read_parquet('{pattern}')"
        elif file_format == 'csv':
            pattern = str(base_path / "*.csv")
            read_function = f"read_csv_auto('{pattern}')"
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        # Create view from local files
        self.logger.info(f"Creating view '{table.name}' from {pattern}")
        create_view_sql = f"CREATE OR REPLACE VIEW {table.name} AS SELECT * FROM {read_function}"

        try:
            self.engine.execute(create_view_sql)

            # Get row count for logging
            count_result = self.engine.execute(f"SELECT COUNT(*) as cnt FROM {table.name}")
            row_count = count_result['cnt'].iloc[0] if not count_result.empty else 0
            self.logger.info(f"Successfully loaded table {table.name} with {row_count:,} rows")

        except Exception as e:
            self.logger.error(f"Failed to load table {table.name} from local: {e}")
            raise


__all__ = ['TableLoader']
