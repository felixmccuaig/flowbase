from pathlib import Path

from flowbase.core.config.schemas import DatasetConfig, FeatureConfig, TableConfig


def test_table_config_parses_grain_and_incremental_metadata(tmp_path: Path) -> None:
    config_path = tmp_path / "table.yaml"
    config_path.write_text(
        """
table:
  name: entity_event_core
  storage:
    base_path: data/marts
    file_format: parquet
  grain:
    type: key
    primary_key: [partition_date, group_id, entity_id]
    partition_by: [partition_date]
    entity_keys: [entity_id]
  incremental:
    strategy: key_upsert
    change_propagation:
      from_sources:
        - source: source_records
          match_on: [partition_date, group_id, entity_id]
    watermark:
      mode: event_time
      allowed_lateness: 6h
""",
        encoding="utf-8",
    )

    config = TableConfig.load(config_path)

    assert config.grain is not None
    assert config.grain.type == "key"
    assert config.grain.primary_key == ["partition_date", "group_id", "entity_id"]
    assert config.grain.partition_by == ["partition_date"]
    assert config.incremental is not None
    assert config.incremental.strategy == "key_upsert"
    assert len(config.incremental.change_propagation) == 1
    assert config.incremental.change_propagation[0].source == "source_records"
    assert config.incremental.watermark is not None
    assert config.incremental.watermark.allowed_lateness == "6h"


def test_dataset_and_feature_configs_parse_incremental_blocks() -> None:
    dataset = DatasetConfig.from_dict(
        {
            "dataset": {
                "name": "entity_event_dataset",
                "grain": {
                    "type": "partition",
                    "partition_by": ["partition_date"],
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
                "name": "entity_features",
                "grain": {
                    "type": "entity",
                    "entity_keys": ["entity_id"],
                },
                "incremental": {
                    "strategy": "entity_state_update",
                },
                "features": [],
            }
        }
    )

    assert dataset.grain is not None
    assert dataset.grain.partition_by == ["partition_date"]
    assert dataset.incremental is not None
    assert dataset.incremental.strategy == "partition_replace"

    assert feature.grain is not None
    assert feature.grain.type == "entity"
    assert feature.grain.entity_keys == ["entity_id"]
    assert feature.incremental is not None
    assert feature.incremental.strategy == "entity_state_update"
