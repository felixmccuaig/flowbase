"""Minimal incremental planner for dry-run propagation."""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List

from flowbase.incremental.events import ChangeEvent
from flowbase.incremental.graph import DependencyGraph, DependencyRule, IncrementalNodeSpec
from flowbase.incremental.work_units import WorkUnit


class IncrementalPlanner:
    """Expands source changes into downstream work units."""

    def plan(self, changes: List[ChangeEvent], graph: DependencyGraph) -> List[WorkUnit]:
        if not changes:
            return []

        by_name: Dict[str, ChangeEvent] = {
            self._change_id(change): change
            for change in changes
        }
        downstream = graph.downstream_map()
        queue = deque()
        planned: Dict[tuple[str, str, tuple[tuple[str, str], ...]], WorkUnit] = {}

        for change in changes:
            for node in graph.nodes.values():
                if change.source_name in node.source_names or change.source_name == node.name:
                    unit = self._make_work_unit(node, change, reason=f"source:{change.source_name}")
                    key = self._unit_key(unit)
                    planned[key] = unit
                    queue.append(node.name)

        visited_edges: set[tuple[str, str]] = set()
        while queue:
            current = queue.popleft()
            for downstream_node in downstream.get(current, []):
                edge = (current, downstream_node.name)
                if edge in visited_edges:
                    continue
                visited_edges.add(edge)

                for base_unit in [u for u in planned.values() if u.node_name == current]:
                    propagated = self._propagate_unit(
                        base_unit=base_unit,
                        upstream=current,
                        node=downstream_node,
                    )
                    key = self._unit_key(propagated)
                    if key not in planned:
                        planned[key] = propagated
                        queue.append(downstream_node.name)

        return list(planned.values())

    def _propagate_unit(self, base_unit: WorkUnit, upstream: str, node: IncrementalNodeSpec) -> WorkUnit:
        rule = next(
            (r for r in node.dependency_rules if r.upstream == upstream),
            DependencyRule(upstream=upstream, propagation_mode=node.grain_type),
        )
        keys = dict(base_unit.keys)
        if rule.key_mapping:
            remapped = {}
            for src_key, value in keys.items():
                remapped[rule.key_mapping.get(src_key, src_key)] = value
            keys = remapped
        if (rule.propagation_mode or node.grain_type) == "entity" and node.entity_keys:
            keys = {k: keys.get(k) for k in node.entity_keys}
        elif (rule.propagation_mode or node.grain_type) == "key" and node.primary_key:
            keys = {k: keys.get(k) for k in node.primary_key}
        elif (rule.propagation_mode or node.grain_type) == "partition" and node.partition_by:
            keys = {k: keys.get(k) for k in node.partition_by}
        return WorkUnit(
            node_name=node.name,
            grain_type=rule.propagation_mode or node.grain_type,
            keys=keys,
            reason=f"downstream_of:{upstream}",
            source_change_ids=list(base_unit.source_change_ids),
        )

    def _make_work_unit(self, node: IncrementalNodeSpec, change: ChangeEvent, reason: str) -> WorkUnit:
        keys = self._keys_for_grain(node, change)
        return WorkUnit(
            node_name=node.name,
            grain_type=node.grain_type,
            keys=keys,
            reason=reason,
            source_change_ids=[self._change_id(change)],
        )

    def _keys_for_grain(self, node: IncrementalNodeSpec, change: ChangeEvent) -> Dict[str, object]:
        if node.grain_type == "entity":
            if node.entity_keys:
                return {k: change.entity_keys.get(k) for k in node.entity_keys}
            return dict(change.entity_keys)
        if node.grain_type == "key":
            if node.primary_key:
                return {
                    k: change.primary_key.get(k) or change.partition_keys.get(k) or change.entity_keys.get(k)
                    for k in node.primary_key
                }
            return dict(change.primary_key)
        if node.grain_type == "partition":
            if node.partition_by:
                return {k: change.partition_keys.get(k) or change.primary_key.get(k) for k in node.partition_by}
            return dict(change.partition_keys)
        keys = {}
        keys.update(change.partition_keys)
        keys.update(change.primary_key)
        keys.update(change.entity_keys)
        return keys

    def _change_id(self, change: ChangeEvent) -> str:
        if change.idempotency_key:
            return change.idempotency_key
        return f"{change.source_name}:{change.change_type}:{sorted(change.primary_key.items())}"

    def _unit_key(self, unit: WorkUnit) -> tuple[str, str, tuple[tuple[str, str], ...]]:
        items = tuple(sorted((str(k), "" if v is None else str(v)) for k, v in unit.keys.items()))
        return (unit.node_name, unit.grain_type, items)


def build_graph_from_workflow_tasks(tasks: Iterable[object]) -> DependencyGraph:
    """Build a planner graph from WorkflowTask-like objects."""
    nodes: Dict[str, IncrementalNodeSpec] = {}
    for task in tasks:
        grain = getattr(task, "grain", None) or {}
        incremental = getattr(task, "incremental", None) or {}
        raw_rules = incremental.get("change_propagation", []) if isinstance(incremental, dict) else []
        if isinstance(raw_rules, dict):
            raw_rules = raw_rules.get("from_sources", [])
        rules = [
            DependencyRule(
                upstream=str(rule["upstream"]),
                propagation_mode=str(rule.get("propagation_mode", grain.get("type", "partition"))),
                key_mapping={str(k): str(v) for k, v in dict(rule.get("key_mapping", {})).items()},
            )
            for rule in raw_rules
            if isinstance(rule, dict) and rule.get("upstream")
        ]
        source_names = [str(v) for v in incremental.get("sources", [])] if isinstance(incremental, dict) else []
        nodes[task.name] = IncrementalNodeSpec(
            name=task.name,
            grain_type=str(grain.get("type", "partition")),
            primary_key=[str(v) for v in grain.get("primary_key", [])],
            partition_by=[str(v) for v in grain.get("partition_by", [])],
            entity_keys=[str(v) for v in grain.get("entity_keys", [])],
            depends_on=list(getattr(task, "depends_on", [])),
            source_names=source_names,
            dependency_rules=rules,
            incremental_strategy=str(incremental.get("strategy", "full_refresh")) if isinstance(incremental, dict) else "full_refresh",
        )
    return DependencyGraph(nodes=nodes)
