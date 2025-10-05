"""Pipeline execution engine."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from flowbase.core.config.models import PipelineConfig, DataSourceConfig, FeatureConfig
from flowbase.query.base import QueryEngine
from flowbase.query.engines.duckdb_engine import DuckDBEngine
from flowbase.storage.base import StorageBackend
from flowbase.storage.local.filesystem import LocalFileSystem

logger = logging.getLogger(__name__)


class PipelineExecutor:
    """Executes data pipelines based on configuration."""

    def __init__(
        self,
        storage: Optional[StorageBackend] = None,
        query_engine: Optional[QueryEngine] = None,
    ):
        """
        Initialize pipeline executor.

        Args:
            storage: Storage backend (defaults to LocalFileSystem)
            query_engine: Query engine (defaults to DuckDBEngine)
        """
        self.storage = storage or LocalFileSystem()
        self.query_engine = query_engine or DuckDBEngine()

    def execute(self, config: PipelineConfig) -> Dict[str, pd.DataFrame]:
        """
        Execute a complete pipeline.

        Args:
            config: Pipeline configuration

        Returns:
            Dictionary mapping feature set names to resulting DataFrames
        """
        logger.info(f"Starting pipeline: {config.name}")
        start_time = datetime.now()

        # Load data sources
        logger.info("Loading data sources...")
        self._load_sources(config.sources)

        # Execute feature engineering
        logger.info("Executing feature engineering...")
        results = {}
        for feature in config.features:
            logger.info(f"Processing feature set: {feature.name}")
            df = self._execute_feature(feature)
            results[feature.name] = df

            # Materialize if requested
            if feature.materialize and config.output_path:
                self._save_feature_set(feature.name, df, config.output_path)

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Pipeline completed in {elapsed:.2f}s")

        return results

    def _load_sources(self, sources: list[DataSourceConfig]) -> None:
        """Load data sources into the query engine."""
        for source in sources:
            logger.info(f"Loading source: {source.name}")

            if source.type == "file":
                if not source.path or not source.format:
                    raise ValueError(f"File source {source.name} requires path and format")

                # Register file with query engine
                self.query_engine.register_file(
                    name=source.name, file_path=source.path, file_format=source.format.value
                )

            elif source.type == "sql":
                if not source.query:
                    raise ValueError(f"SQL source {source.name} requires a query")

                # Execute query and register result
                df = self.query_engine.execute(source.query)
                self.query_engine.register_dataframe(source.name, df)

            else:
                raise ValueError(f"Unsupported source type: {source.type}")

    def _execute_feature(self, feature: FeatureConfig) -> pd.DataFrame:
        """Execute a feature engineering query."""
        return self.query_engine.execute(feature.sql)

    def _save_feature_set(self, name: str, df: pd.DataFrame, output_path: str) -> None:
        """Save a feature set to storage."""
        path = Path(output_path) / f"{name}.parquet"
        logger.info(f"Saving feature set to: {path}")

        # Save to parquet
        buffer = df.to_parquet()
        self.storage.write(str(path), buffer)

    def close(self) -> None:
        """Clean up resources."""
        self.query_engine.close()
