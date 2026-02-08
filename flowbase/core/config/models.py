"""Configuration data models using Pydantic."""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, model_validator


class StorageType(str, Enum):
    """Storage backend types."""

    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE = "azure"


class QueryEngineType(str, Enum):
    """Query engine types."""

    DUCKDB = "duckdb"
    TRINO = "trino"


class FileFormat(str, Enum):
    """Supported file formats."""

    PARQUET = "parquet"
    CSV = "csv"
    JSON = "json"


class DataSourceConfig(BaseModel):
    """Configuration for a data source."""

    name: str = Field(..., description="Name to reference this data source")
    type: str = Field(..., description="Data source type (file, sql, stream)")
    path: Optional[str] = Field(None, description="File path or URI")
    format: Optional[FileFormat] = Field(None, description="File format")
    query: Optional[str] = Field(None, description="SQL query for data loading")
    connection: Optional[Dict[str, Any]] = Field(None, description="Connection parameters")


class ScheduleConfig(BaseModel):
    """Cron schedule configuration."""

    cron: str = Field(..., description="Cron expression")
    timezone: str = Field(default="UTC", description="Timezone for schedule")


class FeatureConfig(BaseModel):
    """Feature engineering configuration."""

    name: str = Field(..., description="Feature set name")
    sql: str = Field(..., description="SQL query to generate features")
    sources: List[str] = Field(..., description="Data sources required")
    output_table: Optional[str] = Field(None, description="Output table name")
    materialize: bool = Field(default=True, description="Materialize results")


class PipelineConfig(BaseModel):
    """Data pipeline configuration."""

    name: str = Field(..., description="Pipeline name")
    description: Optional[str] = Field(None, description="Pipeline description")
    sources: List[DataSourceConfig] = Field(..., description="Input data sources")
    features: List[FeatureConfig] = Field(..., description="Feature definitions")
    schedule: Optional[ScheduleConfig] = Field(None, description="Execution schedule")
    output_path: Optional[str] = Field(None, description="Output path for results")


class ModelConfig(BaseModel):
    """Model training configuration."""

    name: str = Field(..., description="Model name")
    type: str = Field(..., description="Model type (sklearn, xgboost, lightgbm, etc)")
    class_name: str = Field(..., description="Model class name")
    hyperparameters: Dict[str, Any] = Field(
        default_factory=dict, description="Model hyperparameters"
    )
    features: List[str] = Field(..., description="Feature columns to use")
    target: str = Field(..., description="Target column")
    group_column: Optional[str] = Field(
        default=None, description="Optional group column for race-level/ranking training"
    )
    group_objective: Optional[str] = Field(
        default=None, description="Optional ranking objective (e.g., rank:softmax)"
    )
    cv_folds: int = Field(default=5, description="Cross-validation folds")
    leakage_columns: Optional[List[str]] = Field(
        default=None,
        description="Columns excluded from training due to target/temporal leakage. "
                    "Included in test output for post-hoc evaluation."
    )

    @model_validator(mode='after')
    def check_leakage_columns(self):
        if self.leakage_columns:
            leaked = set(self.features) & set(self.leakage_columns)
            if leaked:
                raise ValueError(
                    f"Features contain leakage columns: {sorted(leaked)}. "
                    "These columns must not be used as training features."
                )
        return self


class ExperimentConfig(BaseModel):
    """Experiment configuration for training multiple models."""

    name: str = Field(..., description="Experiment name")
    description: Optional[str] = Field(None, description="Experiment description")
    dataset: str = Field(..., description="Dataset table or path")
    models: List[ModelConfig] = Field(..., description="Models to train")
    metrics: List[str] = Field(
        default=["accuracy", "f1", "roc_auc"], description="Metrics to track"
    )
    test_size: float = Field(default=0.2, description="Test set size")
    random_state: int = Field(default=42, description="Random seed")


class FlowbaseConfig(BaseModel):
    """Root Flowbase project configuration."""

    project_name: str = Field(..., description="Project name")
    version: str = Field(default="1.0.0", description="Project version")
    storage: StorageType = Field(default=StorageType.LOCAL, description="Storage backend")
    storage_config: Dict[str, Any] = Field(
        default_factory=dict, description="Storage backend configuration"
    )
    query_engine: QueryEngineType = Field(
        default=QueryEngineType.DUCKDB, description="Query engine"
    )
    query_engine_config: Dict[str, Any] = Field(
        default_factory=dict, description="Query engine configuration"
    )
    base_path: Optional[str] = Field(None, description="Base path for data and artifacts")
    pipelines: List[PipelineConfig] = Field(default_factory=list, description="Pipelines")
    experiments: List[ExperimentConfig] = Field(default_factory=list, description="Experiments")
