from flowbase.incremental import ChangeEvent, DependencyGraph, DependencyRule, IncrementalNodeSpec, IncrementalPlanner


def test_incremental_planner_propagates_key_change_to_entity_state() -> None:
    graph = DependencyGraph(
        nodes={
            "normalize_source_records": IncrementalNodeSpec(
                name="normalize_source_records",
                grain_type="key",
                primary_key=["partition_date", "group_id", "entity_id"],
                source_names=["source_records"],
                incremental_strategy="key_upsert",
            ),
            "update_entity_state": IncrementalNodeSpec(
                name="update_entity_state",
                grain_type="entity",
                entity_keys=["entity_id"],
                depends_on=["normalize_source_records"],
                dependency_rules=[
                    DependencyRule(
                        upstream="normalize_source_records",
                        propagation_mode="entity",
                        key_mapping={"entity_id": "entity_id"},
                    )
                ],
                incremental_strategy="entity_state_update",
            ),
        }
    )
    changes = [
        ChangeEvent(
            source_name="source_records",
            change_type="upsert",
            primary_key={"partition_date": "2026-01-01", "group_id": 1, "entity_id": "entity-1"},
            entity_keys={"entity_id": "entity-1"},
            partition_keys={"partition_date": "2026-01-01"},
        )
    ]

    plan = IncrementalPlanner().plan(changes, graph)
    by_name = {unit.node_name: unit for unit in plan}

    assert set(by_name) == {"normalize_source_records", "update_entity_state"}
    assert by_name["normalize_source_records"].grain_type == "key"
    assert by_name["normalize_source_records"].keys == {
        "partition_date": "2026-01-01",
        "group_id": 1,
        "entity_id": "entity-1",
    }
    assert by_name["update_entity_state"].grain_type == "entity"
    assert by_name["update_entity_state"].keys == {"entity_id": "entity-1"}


def test_incremental_planner_respects_partition_grain() -> None:
    graph = DependencyGraph(
        nodes={
            "entity_partition_refresh": IncrementalNodeSpec(
                name="entity_partition_refresh",
                grain_type="partition",
                partition_by=["partition_date"],
                source_names=["source_records"],
                incremental_strategy="partition_replace",
            )
        }
    )
    changes = [
        ChangeEvent(
            source_name="source_records",
            change_type="upsert",
            primary_key={"entity_id": "entity-2"},
            partition_keys={"partition_date": "2026-02-01"},
        )
    ]

    plan = IncrementalPlanner().plan(changes, graph)

    assert len(plan) == 1
    assert plan[0].grain_type == "partition"
    assert plan[0].keys == {"partition_date": "2026-02-01"}
