"""Event recorder for optional Flowbase observability."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from .sinks import DynamoDBSink, FileSink, NullSink, ObservabilitySink


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class ObservabilityRecorder:
    """Records workflow, task, and quality events to a configured sink."""

    def __init__(self, sink: ObservabilitySink, *, project_name: str = "unknown-project"):
        self.sink = sink
        self.project_name = project_name

    def new_run_id(self, workflow_name: str) -> str:
        return f"{workflow_name}__{uuid4().hex[:12]}"

    def emit(
        self,
        *,
        event_type: str,
        workflow_name: str,
        run_id: str,
        status: str,
        task_name: str | None = None,
        phase: str | None = None,
        timestamp: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event = {
            "pk": f"{self.project_name}#{workflow_name}",
            "sk": f"{timestamp or utc_now_iso()}#{run_id}#{event_type}#{task_name or '_'}",
            "project_name": self.project_name,
            "workflow_name": workflow_name,
            "run_id": run_id,
            "event_type": event_type,
            "phase": phase or event_type,
            "task_name": task_name,
            "status": status,
            "timestamp": timestamp or utc_now_iso(),
            "details": details or {},
        }
        self.sink.emit(event)
        return event


def build_observability_recorder(project_config: dict[str, Any] | None) -> ObservabilityRecorder:
    config = project_config if isinstance(project_config, dict) else {}
    observability = config.get("observability", {})
    enabled = bool(observability.get("enabled", False))
    project_name = str(config.get("project_name", "unknown-project"))

    if not enabled:
        return ObservabilityRecorder(NullSink(), project_name=project_name)

    sink_name = str(observability.get("sink", "null")).strip().lower()
    if sink_name == "file":
        path = observability.get("path", "logs/observability/events.jsonl")
        sink: ObservabilitySink = FileSink(path)
    elif sink_name == "dynamodb":
        dynamodb_cfg = observability.get("dynamodb", {})
        sink = DynamoDBSink(
            table_name=str(dynamodb_cfg["table_name"]),
            region=dynamodb_cfg.get("region"),
        )
    else:
        sink = NullSink()

    return ObservabilityRecorder(sink, project_name=project_name)
