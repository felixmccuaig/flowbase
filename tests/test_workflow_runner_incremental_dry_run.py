from pathlib import Path

from flowbase.workflows.runner import WorkflowRunner


def test_workflow_runner_dry_run_returns_incremental_plan(tmp_path: Path) -> None:
    workflow_dir = tmp_path / "incremental_demo"
    workflow_dir.mkdir(parents=True)
    (workflow_dir / "workflow.yaml").write_text(
        """
name: incremental_demo
params: {}
tasks:
  - name: normalize_signals
    type: custom
    config: dummy.yaml
    incremental:
      strategy: key_upsert
      sources: [grv_signals]
    grain:
      type: key
      primary_key: [race_date, race_number, dog_key]
  - name: update_dog_state
    type: custom
    config: dummy.yaml
    depends_on: [normalize_signals]
    incremental:
      strategy: entity_state_update
      change_propagation:
        - upstream: normalize_signals
          propagation_mode: entity
          key_mapping:
            dog_key: dog_key
    grain:
      type: entity
      entity_keys: [dog_key]
""",
        encoding="utf-8",
    )

    runner = WorkflowRunner(base_dir=str(tmp_path), logs_dir=str(tmp_path / "logs"))
    result = runner.run(
        "incremental_demo",
        dry_run=True,
        changes=[
            {
                "source_name": "grv_signals",
                "change_type": "upsert",
                "primary_key": {"race_date": "2026-01-01", "race_number": 1, "dog_key": "dog-1"},
                "entity_keys": {"dog_key": "dog-1"},
                "partition_keys": {"race_date": "2026-01-01"},
            }
        ],
    )

    assert result["incremental_plan"] is not None
    assert len(result["incremental_plan"]["changes"]) == 1
    work_units = {unit["node_name"]: unit for unit in result["incremental_plan"]["work_units"]}
    assert work_units["normalize_signals"]["keys"] == {
        "race_date": "2026-01-01",
        "race_number": 1,
        "dog_key": "dog-1",
    }
    assert work_units["update_dog_state"]["keys"] == {"dog_key": "dog-1"}
