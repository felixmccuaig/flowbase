"""Incremental dependency graph primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class DependencyRule:
    """A directed dependency edge with propagation metadata."""

    upstream: str
    propagation_mode: str = "partition"  # partition | key | entity
    key_mapping: Dict[str, str] = field(default_factory=dict)


@dataclass
class IncrementalNodeSpec:
    """A planner-visible node with grain and incremental metadata."""

    name: str
    grain_type: str = "partition"
    primary_key: List[str] = field(default_factory=list)
    partition_by: List[str] = field(default_factory=list)
    entity_keys: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)
    source_names: List[str] = field(default_factory=list)
    dependency_rules: List[DependencyRule] = field(default_factory=list)
    incremental_strategy: str = "full_refresh"


@dataclass
class DependencyGraph:
    """Planner input graph."""

    nodes: Dict[str, IncrementalNodeSpec]

    def downstream_map(self) -> Dict[str, List[IncrementalNodeSpec]]:
        mapping: Dict[str, List[IncrementalNodeSpec]] = {}
        for node in self.nodes.values():
            for dep in node.depends_on:
                mapping.setdefault(dep, []).append(node)
        return mapping
