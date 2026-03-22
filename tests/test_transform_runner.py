from pathlib import Path

import pandas as pd
import pytest

from flowbase.transforms.runner import TransformRunner


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_transform_validations_support_failing_rows_and_metric(tmp_path: Path) -> None:
    source_path = tmp_path / "input.parquet"
    _write_parquet(
        source_path,
        [
            {"id": 1, "value": 10, "dog_name": "Alpha"},
            {"id": 2, "value": 20, "dog_name": "Beta"},
            {"id": 3, "value": 30, "dog_name": None},
        ],
    )

    runner = TransformRunner()
    result = runner.run(
        config_dict={
            "name": "validation_pass_case",
            "sources": {"src": {"path": str(source_path), "format": "parquet"}},
            "models": [
                {"name": "stg_src", "materialized": "view", "sql": "SELECT * FROM src"}
            ],
            "validations": [
                {
                    "name": "no_negative_values",
                    "mode": "failing_rows",
                    "sql": "SELECT * FROM stg_src WHERE value < 0",
                    "severity": "error",
                },
                {
                    "name": "minimum_row_count",
                    "mode": "metric",
                    "sql": "SELECT COUNT(*) FROM stg_src",
                    "operator": "gte",
                    "threshold": 3,
                    "severity": "error",
                },
                {
                    "name": "warn_if_missing_dog_name",
                    "mode": "failing_rows",
                    "sql": "SELECT * FROM stg_src WHERE dog_name IS NULL",
                    "severity": "warn",
                },
            ],
        },
        project_root=str(tmp_path),
    )

    assert result["type"] == "transform"
    assert len(result["models"]) == 1
    assert len(result["validations"]) == 3

    checks = {v["name"]: v for v in result["validations"]}
    assert checks["no_negative_values"]["passed"] is True
    assert checks["minimum_row_count"]["passed"] is True
    assert checks["warn_if_missing_dog_name"]["passed"] is False
    assert checks["warn_if_missing_dog_name"]["severity"] == "warn"


def test_transform_validation_error_severity_fails_run(tmp_path: Path) -> None:
    source_path = tmp_path / "input.parquet"
    _write_parquet(source_path, [{"id": 1}, {"id": 2}])

    runner = TransformRunner()
    with pytest.raises(RuntimeError, match="Transform execution failed"):
        runner.run(
            config_dict={
                "name": "validation_fail_case",
                "sources": {"src": {"path": str(source_path), "format": "parquet"}},
                "models": [
                    {"name": "stg_src", "materialized": "view", "sql": "SELECT * FROM src"}
                ],
                "validations": [
                    {
                        "name": "id_must_be_gt_10",
                        "mode": "failing_rows",
                        "sql": "SELECT * FROM stg_src WHERE id <= 10",
                        "severity": "error",
                    }
                ],
            },
            project_root=str(tmp_path),
        )


def test_transform_missing_required_source_has_detailed_error_context(tmp_path: Path) -> None:
    missing_file = tmp_path / "does_not_exist.parquet"
    runner = TransformRunner()

    with pytest.raises(RuntimeError) as exc_info:
        runner.run(
            config_dict={
                "name": "missing_required_source_case",
                "sources": {"src": {"path": str(missing_file), "format": "parquet", "required": True}},
                "models": [{"name": "stg_src", "materialized": "view", "sql": "SELECT * FROM src"}],
            },
            project_root=str(tmp_path),
        )

    message = str(exc_info.value)
    assert "Transform execution failed" in message
    assert "missing_required_source_case" in message
    assert "Required source 'src' not found" in message
    assert "configured_path=" in message
    assert "resolved_local_path=" in message
