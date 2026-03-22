import json
from pathlib import Path

from flowbase.workflows.runner import WorkflowRunner


def test_incremental_planner_matches_fixture_work_units(tmp_path: Path) -> None:
    fixtures_dir = Path(__file__).parent / "fixtures" / "incremental"
    workflow_root = fixtures_dir.parent

    runner = WorkflowRunner(base_dir=str(workflow_root), logs_dir=str(tmp_path / "logs"))
    result = runner.run(
        "fixture_incremental_demo",
        config_path=str(fixtures_dir / "simple_workflow.yaml"),
        dry_run=True,
        changes=[
            {
                "source_name": "grv_signals",
                "change_type": "upsert",
                "primary_key": {
                    "race_date": "2026-01-01",
                    "race_number": 1,
                    "dog_key": "dog-1",
                },
                "entity_keys": {"dog_key": "dog-1"},
                "partition_keys": {"race_date": "2026-01-01"},
            }
        ],
    )

    observed = [
        {
            "node_name": unit["node_name"],
            "grain_type": unit["grain_type"],
            "keys": unit["keys"],
            "reason": unit["reason"],
        }
        for unit in sorted(
            result["incremental_plan"]["work_units"],
            key=lambda item: item["node_name"],
        )
    ]

    expected = json.loads(
        (fixtures_dir / "simple_expected_work_units.json").read_text(encoding="utf-8")
    )
    expected = sorted(expected, key=lambda item: item["node_name"])

    assert observed == expected
