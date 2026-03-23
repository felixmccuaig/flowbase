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
  - name: normalize_source_records
    type: custom
    config: dummy.yaml
    incremental:
      strategy: key_upsert
      sources: [source_records]
    grain:
      type: key
      primary_key: [partition_date, group_id, entity_id]
  - name: update_entity_state
    type: custom
    config: dummy.yaml
    depends_on: [normalize_source_records]
    incremental:
      strategy: entity_state_update
      change_propagation:
        - upstream: normalize_source_records
          propagation_mode: entity
          key_mapping:
            entity_id: entity_id
    grain:
      type: entity
      entity_keys: [entity_id]
""",
        encoding="utf-8",
    )

    runner = WorkflowRunner(base_dir=str(tmp_path), logs_dir=str(tmp_path / "logs"))
    result = runner.run(
        "incremental_demo",
        dry_run=True,
        changes=[
            {
                "source_name": "source_records",
                "change_type": "upsert",
                "primary_key": {"partition_date": "2026-01-01", "group_id": 1, "entity_id": "entity-1"},
                "entity_keys": {"entity_id": "entity-1"},
                "partition_keys": {"partition_date": "2026-01-01"},
            }
        ],
    )

    assert result["incremental_plan"] is not None
    assert len(result["incremental_plan"]["changes"]) == 1
    work_units = {unit["node_name"]: unit for unit in result["incremental_plan"]["work_units"]}
    assert work_units["normalize_source_records"]["keys"] == {
        "partition_date": "2026-01-01",
        "group_id": 1,
        "entity_id": "entity-1",
    }
    assert work_units["update_entity_state"]["keys"] == {"entity_id": "entity-1"}
