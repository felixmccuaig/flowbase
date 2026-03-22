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
            {"race_date": "2026-01-01", "dog_key": "dog-1", "value": 10},
            {"race_date": "2026-01-01", "dog_key": "dog-2", "value": 20},
            {"race_date": "2026-01-02", "dog_key": "dog-3", "value": 30},
        ]
    )
    materializer.execute(
        seed,
        output_path=str(root),
        primary_key=["race_date", "dog_key"],
        partition_column="race_date",
        sort_by=["dog_key"],
    )

    incoming = pd.DataFrame(
        [
            {"race_date": "2026-01-01", "dog_key": "dog-2", "value": 200},
            {"race_date": "2026-01-01", "dog_key": "dog-4", "value": 40},
        ]
    )
    result = materializer.execute(
        incoming,
        output_path=str(root),
        primary_key=["race_date", "dog_key"],
        partition_column="race_date",
        sort_by=["dog_key"],
    )

    assert result.partitions_touched == ["2026-01-01"]
    df_day1 = pd.read_parquet(root / "race_date=2026-01-01" / "data.parquet")
    df_day2 = pd.read_parquet(root / "race_date=2026-01-02" / "data.parquet")
    assert df_day1["dog_key"].tolist() == ["dog-1", "dog-2", "dog-4"]
    assert df_day1["value"].tolist() == [10, 200, 40]
    assert df_day2["dog_key"].tolist() == ["dog-3"]


def test_key_upsert_materializer_updates_single_parquet_file(tmp_path: Path) -> None:
    target = tmp_path / "latest_state.parquet"
    materializer = KeyUpsertMaterializer()

    seed = pd.DataFrame(
        [
            {"dog_key": "dog-1", "state": "ok"},
            {"dog_key": "dog-2", "state": "watch"},
        ]
    )
    materializer.execute(seed, output_path=str(target), primary_key=["dog_key"], sort_by=["dog_key"])

    incoming = pd.DataFrame(
        [
            {"dog_key": "dog-2", "state": "bad"},
            {"dog_key": "dog-3", "state": "ok"},
        ]
    )
    result = materializer.execute(
        incoming,
        output_path=str(target),
        primary_key=["dog_key"],
        sort_by=["dog_key"],
    )

    final = pd.read_parquet(target)
    assert result.rows_written == 3
    assert final["dog_key"].tolist() == ["dog-1", "dog-2", "dog-3"]
    assert final["state"].tolist() == ["ok", "bad", "ok"]
