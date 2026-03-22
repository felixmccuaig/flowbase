from pathlib import Path

import duckdb
import pandas as pd

from flowbase.transforms.runner import TransformRunner


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_transform_partitioned_output_replace_partition(tmp_path: Path) -> None:
    source_path = tmp_path / "input.parquet"
    output_root = tmp_path / "serving"
    _write_parquet(
        source_path,
        [
            {"id": 1, "value": 10},
            {"id": 2, "value": 20},
            {"id": 3, "value": 30},
            {"id": 4, "value": 40},
        ],
    )

    runner = TransformRunner()
    cfg = {
        "name": "partitioned_replace",
        "sources": {"src": {"path": str(source_path), "format": "parquet"}},
        "models": [
            {
                "name": "m",
                "materialized": "table",
                "sql": "SELECT * FROM src",
                "output": {
                    "path": str(output_root),
                    "format": "parquet",
                    "partition_column_name": "bucket",
                    "partition_by_expression": "id % 2",
                    "sort_by": ["id"],
                    "write_mode": "replace_partition",
                },
            }
        ],
    }
    result1 = runner.run(config_dict=cfg, project_root=str(tmp_path))
    out1 = result1["outputs"][0]
    assert out1["partitioned"] is True
    assert out1["partition_count"] == 2
    assert (output_root / "bucket=0" / "data.parquet").exists()
    assert (output_root / "bucket=1" / "data.parquet").exists()

    # Re-run with only one partition present; touched partition should still be a single file.
    _write_parquet(source_path, [{"id": 1, "value": 99}, {"id": 3, "value": 88}])
    runner.run(config_dict=cfg, project_root=str(tmp_path))
    bucket_one_files = list((output_root / "bucket=1").glob("*.parquet"))
    assert len(bucket_one_files) == 1

    df = duckdb.sql(
        f"SELECT id FROM read_parquet('{(output_root / 'bucket=1' / 'data.parquet')}')"
    ).df()
    assert df["id"].tolist() == [1, 3]


def test_transform_partitioned_output_append_mode(tmp_path: Path) -> None:
    source_path = tmp_path / "input.parquet"
    output_root = tmp_path / "serving"
    _write_parquet(source_path, [{"id": 1, "value": 10}, {"id": 3, "value": 30}])

    runner = TransformRunner()
    cfg = {
        "name": "partitioned_append",
        "sources": {"src": {"path": str(source_path), "format": "parquet"}},
        "models": [
            {
                "name": "m",
                "materialized": "table",
                "sql": "SELECT * FROM src",
                "output": {
                    "path": str(output_root),
                    "format": "parquet",
                    "partition_column_name": "bucket",
                    "partition_by_expression": "id % 2",
                    "write_mode": "append",
                },
            }
        ],
    }

    runner.run(config_dict=cfg, project_root=str(tmp_path))
    runner.run(config_dict=cfg, project_root=str(tmp_path))
    bucket_one_files = list((output_root / "bucket=1").glob("*.parquet"))
    assert len(bucket_one_files) >= 2
