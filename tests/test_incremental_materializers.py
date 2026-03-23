from pathlib import Path

import pandas as pd

from flowbase.incremental.materializers import KeyUpsertMaterializer, PartitionReplaceMaterializer


def test_partition_replace_materializer_rewrites_only_touched_partitions(tmp_path: Path) -> None:
    root = tmp_path / "partitioned"
    materializer = PartitionReplaceMaterializer()

    initial = pd.DataFrame(
        [
            {"id": 1, "bucket": "A", "value": 10},
            {"id": 2, "bucket": "B", "value": 20},
        ]
    )
    materializer.execute(initial, output_path=str(root), partition_column="bucket", sort_by=["id"])

    update = pd.DataFrame(
        [
            {"id": 1, "bucket": "A", "value": 99},
            {"id": 3, "bucket": "A", "value": 30},
        ]
    )
    result = materializer.execute(update, output_path=str(root), partition_column="bucket", sort_by=["id"])

    assert sorted(result.partitions_touched) == ["A"]
    df_a = pd.read_parquet(root / "bucket=A" / "data.parquet")
    df_b = pd.read_parquet(root / "bucket=B" / "data.parquet")
    assert df_a["id"].tolist() == [1, 3]
    assert df_a["value"].tolist() == [99, 30]
    assert df_b["id"].tolist() == [2]
    assert df_b["value"].tolist() == [20]


def test_key_upsert_materializer_updates_partitioned_rows_by_primary_key(tmp_path: Path) -> None:
    root = tmp_path / "upsert_partitioned"
    materializer = KeyUpsertMaterializer()

    seed = pd.DataFrame(
        [
            {"partition_date": "2026-01-01", "entity_id": "entity-1", "value": 10},
            {"partition_date": "2026-01-01", "entity_id": "entity-2", "value": 20},
            {"partition_date": "2026-01-02", "entity_id": "entity-3", "value": 30},
        ]
    )
    materializer.execute(
        seed,
        output_path=str(root),
        primary_key=["partition_date", "entity_id"],
        partition_column="partition_date",
        sort_by=["entity_id"],
    )

    incoming = pd.DataFrame(
        [
            {"partition_date": "2026-01-01", "entity_id": "entity-2", "value": 200},
            {"partition_date": "2026-01-01", "entity_id": "entity-4", "value": 40},
        ]
    )
    result = materializer.execute(
        incoming,
        output_path=str(root),
        primary_key=["partition_date", "entity_id"],
        partition_column="partition_date",
        sort_by=["entity_id"],
    )

    assert result.partitions_touched == ["2026-01-01"]
    df_day1 = pd.read_parquet(root / "partition_date=2026-01-01" / "data.parquet")
    df_day2 = pd.read_parquet(root / "partition_date=2026-01-02" / "data.parquet")
    assert df_day1["entity_id"].tolist() == ["entity-1", "entity-2", "entity-4"]
    assert df_day1["value"].tolist() == [10, 200, 40]
    assert df_day2["entity_id"].tolist() == ["entity-3"]


def test_key_upsert_materializer_updates_single_parquet_file(tmp_path: Path) -> None:
    target = tmp_path / "latest_state.parquet"
    materializer = KeyUpsertMaterializer()

    seed = pd.DataFrame(
        [
            {"entity_id": "entity-1", "state": "ok"},
            {"entity_id": "entity-2", "state": "watch"},
        ]
    )
    materializer.execute(seed, output_path=str(target), primary_key=["entity_id"], sort_by=["entity_id"])

    incoming = pd.DataFrame(
        [
            {"entity_id": "entity-2", "state": "bad"},
            {"entity_id": "entity-3", "state": "ok"},
        ]
    )
    result = materializer.execute(
        incoming,
        output_path=str(target),
        primary_key=["entity_id"],
        sort_by=["entity_id"],
    )

    final = pd.read_parquet(target)
    assert result.rows_written == 3
    assert final["entity_id"].tolist() == ["entity-1", "entity-2", "entity-3"]
    assert final["state"].tolist() == ["ok", "bad", "ok"]
