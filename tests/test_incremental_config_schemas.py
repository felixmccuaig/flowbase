from pathlib import Path

from flowbase.core.config.schemas import DatasetConfig, FeatureConfig, TableConfig


def test_table_config_parses_grain_and_incremental_metadata(tmp_path: Path) -> None:
    config_path = tmp_path / "table.yaml"
    config_path.write_text(
        """
table:
  name: runner_event_core
  storage:
    base_path: data/marts
    file_format: parquet
  grain:
    type: key
    primary_key: [race_date, race_number, dog_key]
    partition_by: [race_date]
    entity_keys: [dog_key]
  incremental:
    strategy: key_upsert
    change_propagation:
      from_sources:
        - source: grv_signals
          match_on: [race_date, race_number, dog_key]
    watermark:
      mode: event_time
      allowed_lateness: 6h
""",
        encoding="utf-8",
    )

    config = TableConfig.load(config_path)

    assert config.grain is not None
    assert config.grain.type == "key"
    assert config.grain.primary_key == ["race_date", "race_number", "dog_key"]
    assert config.grain.partition_by == ["race_date"]
    assert config.incremental is not None
    assert config.incremental.strategy == "key_upsert"
    assert len(config.incremental.change_propagation) == 1
    assert config.incremental.change_propagation[0].source == "grv_signals"
    assert config.incremental.watermark is not None
    assert config.incremental.watermark.allowed_lateness == "6h"


def test_dataset_and_feature_configs_parse_incremental_blocks() -> None:
    dataset = DatasetConfig.from_dict(
        {
            "dataset": {
                "name": "runner_event_dataset",
                "grain": {
                    "type": "partition",
                    "partition_by": ["race_date"],
                },
                "incremental": {
                    "strategy": "partition_replace",
                },
            }
        }
    )
    feature = FeatureConfig.from_dict(
        {
            "features": {
                "name": "runner_features",
                "grain": {
                    "type": "entity",
                    "entity_keys": ["dog_key"],
                },
                "incremental": {
                    "strategy": "entity_state_update",
                },
                "features": [],
            }
        }
    )

    assert dataset.grain is not None
    assert dataset.grain.partition_by == ["race_date"]
    assert dataset.incremental is not None
    assert dataset.incremental.strategy == "partition_replace"

    assert feature.grain is not None
    assert feature.grain.type == "entity"
    assert feature.grain.entity_keys == ["dog_key"]
    assert feature.incremental is not None
    assert feature.incremental.strategy == "entity_state_update"
