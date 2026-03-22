"""Unified configuration schemas for all Flowbase config files."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class SourceType(str, Enum):
    """Data source types."""
    S3 = "s3"
    LOCAL = "local"
    FILE = "file"


class FileFormat(str, Enum):
    """Supported file formats."""
    PARQUET = "parquet"
    CSV = "csv"
    JSON = "json"


@dataclass
class GrainConfig:
    """Logical recompute grain for a dataset, feature set, or table."""

    type: str = "partition"  # partition | key | entity | range
    primary_key: List[str] = field(default_factory=list)
    partition_by: List[str] = field(default_factory=list)
    entity_keys: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GrainConfig:
        return cls(
            type=str(data.get("type", "partition")),
            primary_key=[str(v) for v in data.get("primary_key", [])],
            partition_by=[str(v) for v in data.get("partition_by", [])],
            entity_keys=[str(v) for v in data.get("entity_keys", [])],
        )


@dataclass
class WatermarkConfig:
    """Late-arrival and event-time policy for incremental planning."""

    mode: str = "event_time"
    allowed_lateness: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WatermarkConfig:
        return cls(
            mode=str(data.get("mode", "event_time")),
            allowed_lateness=data.get("allowed_lateness"),
        )


@dataclass
class ChangePropagationRule:
    """How source changes map into downstream recompute keys."""

    source: str
    match_on: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ChangePropagationRule:
        return cls(
            source=str(data["source"]),
            match_on=[str(v) for v in data.get("match_on", [])],
        )


@dataclass
class IncrementalConfig:
    """Incremental materialization metadata."""

    strategy: str = "full_refresh"
    change_propagation: List[ChangePropagationRule] = field(default_factory=list)
    watermark: Optional[WatermarkConfig] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> IncrementalConfig:
        propagation_data = data.get("change_propagation", {})
        from_sources = []
        if isinstance(propagation_data, dict):
            from_sources = propagation_data.get("from_sources", [])
        elif isinstance(propagation_data, list):
            from_sources = propagation_data

        watermark_data = data.get("watermark")
        return cls(
            strategy=str(data.get("strategy", "full_refresh")),
            change_propagation=[
                ChangePropagationRule.from_dict(item)
                for item in from_sources
                if isinstance(item, dict) and item.get("source")
            ],
            watermark=WatermarkConfig.from_dict(watermark_data) if watermark_data else None,
        )


# ============================================================================
# Table Configuration
# ============================================================================

@dataclass
class TableSource:
    """Source configuration for a table."""
    type: SourceType
    bucket: Optional[str] = None
    prefix: Optional[str] = None
    path: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TableSource:
        source_type = SourceType(data.get('type', 'local'))
        return cls(
            type=source_type,
            bucket=data.get('bucket'),
            prefix=data.get('prefix'),
            path=data.get('path')
        )


@dataclass
class TableStorage:
    """Storage configuration for a table."""
    base_path: str
    file_format: FileFormat
    partition_by: Optional[str] = None
    partition_format: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TableStorage:
        return cls(
            base_path=data.get('base_path', 'data/tables'),
            file_format=FileFormat(data.get('file_format', data.get('format', 'parquet'))),
            partition_by=data.get('partition_by'),
            partition_format=data.get('partition_format')
        )


@dataclass
class TablePartitioning:
    """Partitioning configuration for a table."""
    pattern: Optional[str] = None
    date_format: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TablePartitioning:
        return cls(
            pattern=data.get('pattern'),
            date_format=data.get('date_format')
        )


@dataclass
class TableConfig:
    """Complete table configuration."""
    name: str
    description: Optional[str] = None
    source: Optional[TableSource] = None
    storage: Optional[TableStorage] = None
    destination: Optional[TableStorage] = None  # Alternative name for storage
    storage_profiles: Dict[str, TableStorage] = field(default_factory=dict)
    partitioning: Optional[TablePartitioning] = None
    grain: Optional[GrainConfig] = None
    incremental: Optional[IncrementalConfig] = None

    @property
    def storage_config(self) -> Optional[TableStorage]:
        """Get storage config (supports both 'storage' and 'destination' keys)."""
        return self.storage or self.destination

    def get_storage_config(self, profile: Optional[str] = None) -> Optional[TableStorage]:
        """Get storage config for a named profile, falling back to default storage."""
        if profile:
            prof = self.storage_profiles.get(profile)
            if prof:
                return prof
        return self.storage_config

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TableConfig:
        """Parse table config from dictionary."""
        # Handle wrapped format: { table: { name: ..., ... } }
        if 'table' in data:
            data = data['table']

        source_data = data.get('source')
        source = TableSource.from_dict(source_data) if source_data else None

        storage_data = data.get('storage')
        storage = TableStorage.from_dict(storage_data) if storage_data else None

        destination_data = data.get('destination')
        destination = TableStorage.from_dict(destination_data) if destination_data else None

        storage_profiles_data = data.get('storage_profiles', {})
        storage_profiles: Dict[str, TableStorage] = {}
        if isinstance(storage_profiles_data, dict):
            for profile_name, profile_cfg in storage_profiles_data.items():
                if isinstance(profile_cfg, dict):
                    storage_profiles[profile_name] = TableStorage.from_dict(profile_cfg)

        partitioning_data = data.get('partitioning')
        partitioning = TablePartitioning.from_dict(partitioning_data) if partitioning_data else None
        grain_data = data.get("grain")
        grain = GrainConfig.from_dict(grain_data) if grain_data else None
        incremental_data = data.get("incremental")
        incremental = IncrementalConfig.from_dict(incremental_data) if incremental_data else None

        return cls(
            name=data['name'],
            description=data.get('description'),
            source=source,
            storage=storage,
            destination=destination,
            storage_profiles=storage_profiles,
            partitioning=partitioning,
            grain=grain,
            incremental=incremental,
        )

    @classmethod
    def load(cls, path: str | Path) -> TableConfig:
        """Load table config from YAML file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


# ============================================================================
# Dataset Configuration
# ============================================================================

@dataclass
class DatasetSource:
    """Source for a dataset."""
    table: Optional[str] = None
    table_config: Optional[str] = None
    dataset_config: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DatasetSource:
        return cls(
            table=data.get('table'),
            table_config=data.get('table_config'),
            dataset_config=data.get('dataset_config')
        )


@dataclass
class DatasetSourceRef:
    """Reference to another dataset as a source for merging."""
    name: str
    dataset_config: str
    alias: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DatasetSourceRef:
        return cls(
            name=data['name'],
            dataset_config=data['dataset_config'],
            alias=data.get('alias')
        )


@dataclass
class DatasetColumn:
    """Column definition for a dataset."""
    name: str
    type: str
    source: Optional[str] = None  # For merged datasets
    expression: Optional[str] = None
    required: bool = False
    validate: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DatasetColumn:
        return cls(
            name=data['name'],
            type=data['type'],
            source=data.get('source'),
            expression=data.get('expression'),
            required=data.get('required', False),
            validate=data.get('validate')
        )


@dataclass
class DatasetFilter:
    """Filter condition for a dataset."""
    column: str
    operator: str
    value: Any

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DatasetFilter:
        return cls(
            column=data['column'],
            operator=data['operator'],
            value=data.get('value')
        )


@dataclass
class DatasetJoin:
    """Join configuration for merged datasets."""
    type: str  # left, right, inner, outer
    conditions: List[Dict[str, str]]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DatasetJoin:
        return cls(
            type=data.get('type', 'inner'),
            conditions=data.get('conditions', [])
        )


@dataclass
class DatasetConfig:
    """Complete dataset configuration."""
    name: str
    description: Optional[str] = None
    source: Optional[DatasetSource] = None
    sources: List[DatasetSourceRef] = field(default_factory=list)  # For merged datasets
    join: Optional[DatasetJoin] = None
    columns: List[DatasetColumn] = field(default_factory=list)
    filters: List[DatasetFilter] = field(default_factory=list)
    grain: Optional[GrainConfig] = None
    incremental: Optional[IncrementalConfig] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DatasetConfig:
        """Parse dataset config from dictionary."""
        # Handle wrapped format: { dataset: { name: ..., ... } }
        if 'dataset' in data:
            data = data['dataset']

        source_data = data.get('source')
        source = DatasetSource.from_dict(source_data) if source_data else None

        sources_data = data.get('sources', [])
        sources = [DatasetSourceRef.from_dict(s) for s in sources_data]

        join_data = data.get('join')
        join = DatasetJoin.from_dict(join_data) if join_data else None

        columns = [DatasetColumn.from_dict(c) for c in data.get('columns', [])]
        filters = [DatasetFilter.from_dict(f) for f in data.get('filters', [])]
        grain_data = data.get("grain")
        incremental_data = data.get("incremental")

        return cls(
            name=data['name'],
            description=data.get('description'),
            source=source,
            sources=sources,
            join=join,
            columns=columns,
            filters=filters,
            grain=GrainConfig.from_dict(grain_data) if grain_data else None,
            incremental=IncrementalConfig.from_dict(incremental_data) if incremental_data else None,
        )

    @classmethod
    def load(cls, path: str | Path) -> DatasetConfig:
        """Load dataset config from YAML file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


# ============================================================================
# Feature Configuration
# ============================================================================

@dataclass
class FeatureSource:
    """Source for features."""
    dataset_config: Optional[str] = None
    table: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FeatureSource:
        return cls(
            dataset_config=data.get('dataset_config'),
            table=data.get('table')
        )


@dataclass
class FeatureDefinition:
    """Single feature definition (supports both expression-based and declarative)."""
    name: str
    expression: Optional[str] = None
    description: Optional[str] = None

    # Declarative feature engineering fields
    type: Optional[str] = None  # count, average, sum, max, min, stddev, etc.
    windows: Optional[List[str]] = None  # all_time, last_5, last_3, etc.
    partition_by: Optional[List[str]] = None  # Additional partition columns
    filter: Optional[str] = None  # SQL filter expression
    value_column: Optional[str] = None  # For diff_from_last
    value_expression: Optional[str] = None  # For contextual features
    lagged: Optional[bool] = None  # For lagged features
    pass_num: Optional[int] = None  # For multi-pass feature compilation (renamed from 'pass' to avoid Python keyword)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FeatureDefinition:
        return cls(
            name=data['name'],
            expression=data.get('expression'),
            description=data.get('description'),
            type=data.get('type'),
            windows=data.get('windows'),
            partition_by=data.get('partition_by'),
            filter=data.get('filter'),
            value_column=data.get('value_column'),
            value_expression=data.get('value_expression'),
            lagged=data.get('lagged'),
            pass_num=data.get('pass')  # Read from 'pass' in YAML, store as 'pass_num'
        )


@dataclass
class FeatureConfig:
    """Complete feature set configuration."""
    name: str
    description: Optional[str] = None
    source: Optional[FeatureSource] = None
    features: List[FeatureDefinition] = field(default_factory=list)
    # Declarative feature engineering config
    entity_id_column: Optional[str] = None
    time_column: Optional[str] = None
    grain: Optional[GrainConfig] = None
    incremental: Optional[IncrementalConfig] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FeatureConfig:
        """Parse feature config from dictionary."""
        # Handle wrapped format: { features: { name: ..., ... } }
        if 'features' in data and isinstance(data['features'], dict):
            data = data['features']

        source_data = data.get('source')
        source = FeatureSource.from_dict(source_data) if source_data else None

        features = [FeatureDefinition.from_dict(f) for f in data.get('features', [])]
        grain_data = data.get("grain")
        incremental_data = data.get("incremental")

        return cls(
            name=data['name'],
            description=data.get('description'),
            source=source,
            features=features,
            entity_id_column=data.get('entity_id_column'),
            time_column=data.get('time_column'),
            grain=GrainConfig.from_dict(grain_data) if grain_data else None,
            incremental=IncrementalConfig.from_dict(incremental_data) if incremental_data else None,
        )

    @classmethod
    def load(cls, path: str | Path) -> FeatureConfig:
        """Load feature config from YAML file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


__all__ = [
    'SourceType',
    'FileFormat',
    'GrainConfig',
    'WatermarkConfig',
    'ChangePropagationRule',
    'IncrementalConfig',
    'TableSource',
    'TableStorage',
    'TablePartitioning',
    'TableConfig',
    'DatasetSource',
    'DatasetSourceRef',
    'DatasetColumn',
    'DatasetFilter',
    'DatasetJoin',
    'DatasetConfig',
    'FeatureSource',
    'FeatureDefinition',
    'FeatureConfig',
]
