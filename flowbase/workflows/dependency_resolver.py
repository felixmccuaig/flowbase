"""Dependency resolver for workflows - resolves data dependencies from features -> datasets -> tables."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

from flowbase.core.config.schemas import DatasetConfig, FeatureConfig, TableConfig


@dataclass
class TableDependency:
    """A table that needs to be loaded into DuckDB."""

    name: str
    config: TableConfig
    config_path: str


@dataclass
class DatasetDependency:
    """A dataset configuration."""

    name: str
    config: DatasetConfig
    config_path: str
    depends_on_tables: List[TableDependency]
    depends_on_datasets: List[str]  # Names of other datasets


@dataclass
class FeatureDependency:
    """A feature set configuration."""

    name: str
    config: FeatureConfig
    config_path: str
    dataset: Optional[DatasetDependency]


class DependencyResolver:
    """Resolves the dependency chain from features -> datasets -> tables."""

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.logger = logging.getLogger(__name__)

        # Cache to avoid re-parsing same configs
        self._table_cache: Dict[str, TableDependency] = {}
        self._dataset_cache: Dict[str, DatasetDependency] = {}
        self._feature_cache: Dict[str, FeatureDependency] = {}

    def resolve_feature_dependencies(
        self,
        feature_config_path: str
    ) -> FeatureDependency:
        """
        Resolve all dependencies for a feature config.

        Returns:
            FeatureDependency with full dependency tree
        """
        config_path = self._resolve_path(feature_config_path)

        # Check cache
        cache_key = str(config_path)
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]

        self.logger.info(f"Resolving dependencies for feature config: {config_path}")

        # Load feature config using schema
        feature_config = FeatureConfig.load(config_path)

        # Get source dataset if specified
        dataset_dep = None
        if feature_config.source and feature_config.source.dataset_config:
            dataset_config_path = feature_config.source.dataset_config
            dataset_dep = self.resolve_dataset_dependencies(dataset_config_path)

        feature_dep = FeatureDependency(
            name=feature_config.name,
            config=feature_config,
            config_path=str(config_path),
            dataset=dataset_dep
        )

        # Cache it
        self._feature_cache[cache_key] = feature_dep

        return feature_dep

    def resolve_dataset_dependencies(
        self,
        dataset_config_path: str
    ) -> DatasetDependency:
        """
        Resolve all dependencies for a dataset config.

        Returns:
            DatasetDependency with full dependency tree
        """
        config_path = self._resolve_path(dataset_config_path)

        # Check cache
        cache_key = str(config_path)
        if cache_key in self._dataset_cache:
            return self._dataset_cache[cache_key]

        self.logger.info(f"Resolving dependencies for dataset config: {config_path}")

        # Load dataset config using schema
        dataset_config = DatasetConfig.load(config_path)

        tables: List[TableDependency] = []
        dependent_datasets: List[str] = []

        # Check if this is a merged dataset (has multiple sources)
        if dataset_config.sources:
            # Merged dataset - resolve each source dataset
            for source_ref in dataset_config.sources:
                dependent_datasets.append(source_ref.name)

                # Recursively resolve source dataset
                source_dep = self.resolve_dataset_dependencies(source_ref.dataset_config)
                tables.extend(source_dep.depends_on_tables)
        else:
            # Single source dataset - resolve its table
            if dataset_config.source and dataset_config.source.table_config:
                table_config_path = dataset_config.source.table_config
                table_dep = self.resolve_table_config(table_config_path)
                tables.append(table_dep)

        dataset_dep = DatasetDependency(
            name=dataset_config.name,
            config=dataset_config,
            config_path=str(config_path),
            depends_on_tables=tables,
            depends_on_datasets=dependent_datasets
        )

        # Cache it
        self._dataset_cache[cache_key] = dataset_dep

        return dataset_dep

    def resolve_table_config(
        self,
        table_config_path: str
    ) -> TableDependency:
        """
        Resolve table configuration.

        Returns:
            TableDependency with parsed config
        """
        config_path = self._resolve_path(table_config_path)

        # Check cache
        cache_key = str(config_path)
        if cache_key in self._table_cache:
            return self._table_cache[cache_key]

        self.logger.info(f"Resolving table config: {config_path}")

        # Load table config using schema
        table_config = TableConfig.load(config_path)

        table_dep = TableDependency(
            name=table_config.name,
            config=table_config,
            config_path=str(config_path)
        )

        # Cache it
        self._table_cache[cache_key] = table_dep

        return table_dep

    def get_all_table_dependencies(
        self,
        feature_config_path: str
    ) -> List[TableDependency]:
        """
        Get all table dependencies for a feature config (flattened list).

        Returns:
            List of unique TableDependency objects
        """
        feature_dep = self.resolve_feature_dependencies(feature_config_path)

        tables: List[TableDependency] = []
        seen_tables: Set[str] = set()

        if feature_dep.dataset:
            for table in feature_dep.dataset.depends_on_tables:
                if table.name not in seen_tables:
                    tables.append(table)
                    seen_tables.add(table.name)

        return tables

    def _resolve_path(self, relative_path: str) -> Path:
        """Resolve a relative path from project root."""
        path = Path(relative_path)
        if path.is_absolute():
            return path

        # Try relative to project root
        candidate = self.project_root / relative_path
        if candidate.exists():
            return candidate

        # Try relative to cwd
        candidate = Path.cwd() / relative_path
        if candidate.exists():
            return candidate

        # Return as-is and let it fail later if doesn't exist
        return self.project_root / relative_path


__all__ = ['DependencyResolver', 'TableDependency', 'DatasetDependency', 'FeatureDependency']
