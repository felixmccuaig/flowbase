"""Configuration file loader."""

from pathlib import Path
from typing import TYPE_CHECKING, Union

import yaml
from pydantic import ValidationError

from flowbase.core.config.models import FlowbaseConfig, PipelineConfig, ExperimentConfig

if TYPE_CHECKING:
    from flowbase.query.base import QueryEngine


def load_config(config_path: Union[str, Path]) -> FlowbaseConfig:
    """
    Load a Flowbase configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Parsed FlowbaseConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValidationError: If config is invalid
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        config_data = yaml.safe_load(f)

    try:
        return FlowbaseConfig(**config_data)
    except ValidationError as e:
        raise ValueError(f"Invalid configuration: {e}")


def load_pipeline_config(config_path: Union[str, Path]) -> PipelineConfig:
    """Load a pipeline configuration from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Pipeline config file not found: {config_path}")

    with open(path, "r") as f:
        config_data = yaml.safe_load(f)

    return PipelineConfig(**config_data)


def load_experiment_config(config_path: Union[str, Path]) -> ExperimentConfig:
    """Load an experiment configuration from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Experiment config file not found: {config_path}")

    with open(path, "r") as f:
        config_data = yaml.safe_load(f)

    return ExperimentConfig(**config_data)


def get_query_engine_from_config(config: FlowbaseConfig) -> "QueryEngine":
    """
    Create a query engine from a FlowbaseConfig.

    Args:
        config: FlowbaseConfig instance

    Returns:
        Configured QueryEngine instance
    """
    from flowbase.query.factory import create_query_engine

    return create_query_engine(
        engine_type=config.query_engine.value,
        config=config.query_engine_config if config.query_engine_config else None,
    )
