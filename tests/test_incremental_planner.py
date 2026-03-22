from flowbase.incremental import ChangeEvent, DependencyGraph, DependencyRule, IncrementalNodeSpec, IncrementalPlanner


def test_incremental_planner_propagates_key_change_to_entity_state() -> None:
    graph = DependencyGraph(
        nodes={
            "normalize_signals": IncrementalNodeSpec(
                name="normalize_signals",
                grain_type="key",
                primary_key=["race_date", "race_number", "dog_key"],
                source_names=["grv_signals"],
                incremental_strategy="key_upsert",
            ),
            "update_dog_state": IncrementalNodeSpec(
                name="update_dog_state",
                grain_type="entity",
                entity_keys=["dog_key"],
                depends_on=["normalize_signals"],
                dependency_rules=[
                    DependencyRule(
                        upstream="normalize_signals",
                        propagation_mode="entity",
                        key_mapping={"dog_key": "dog_key"},
                    )
                ],
                incremental_strategy="entity_state_update",
            ),
        }
    )
    changes = [
        ChangeEvent(
            source_name="grv_signals",
            change_type="upsert",
            primary_key={"race_date": "2026-01-01", "race_number": 1, "dog_key": "dog-1"},
            entity_keys={"dog_key": "dog-1"},
            partition_keys={"race_date": "2026-01-01"},
        )
    ]

    plan = IncrementalPlanner().plan(changes, graph)
    by_name = {unit.node_name: unit for unit in plan}

    assert set(by_name) == {"normalize_signals", "update_dog_state"}
    assert by_name["normalize_signals"].grain_type == "key"
    assert by_name["normalize_signals"].keys == {
        "race_date": "2026-01-01",
        "race_number": 1,
        "dog_key": "dog-1",
    }
    assert by_name["update_dog_state"].grain_type == "entity"
    assert by_name["update_dog_state"].keys == {"dog_key": "dog-1"}


def test_incremental_planner_respects_partition_grain() -> None:
    graph = DependencyGraph(
        nodes={
            "runner_partition_refresh": IncrementalNodeSpec(
                name="runner_partition_refresh",
                grain_type="partition",
                partition_by=["race_date"],
                source_names=["grv_signals"],
                incremental_strategy="partition_replace",
            )
        }
    )
    changes = [
        ChangeEvent(
            source_name="grv_signals",
            change_type="upsert",
            primary_key={"dog_key": "dog-2"},
            partition_keys={"race_date": "2026-02-01"},
        )
    ]

    plan = IncrementalPlanner().plan(changes, graph)

    assert len(plan) == 1
    assert plan[0].grain_type == "partition"
    assert plan[0].keys == {"race_date": "2026-02-01"}
