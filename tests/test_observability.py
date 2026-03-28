import json
from pathlib import Path

from flowbase.observability import build_observability_recorder
from flowbase.workflows.runner import WorkflowRunner


def test_build_observability_recorder_defaults_to_null_sink() -> None:
    recorder = build_observability_recorder(None)
    event = recorder.emit(
        event_type="run_started",
        workflow_name="example",
        run_id="run-1",
        status="running",
    )
    assert event["workflow_name"] == "example"


def test_workflow_runner_emits_file_observability_events(tmp_path: Path) -> None:
    project_root = tmp_path / "example-project"
    (project_root / "workflows" / "simple").mkdir(parents=True)
    (project_root / "configs").mkdir(parents=True)
    (project_root / "data").mkdir(parents=True)

    events_path = project_root / "logs" / "observability" / "events.jsonl"
    (project_root / "flowbase.yaml").write_text(
        f"""
project_name: example-project
observability:
  enabled: true
  sink: file
  path: "{events_path}"
""",
        encoding="utf-8",
    )
    (project_root / "workflows" / "simple" / "workflow.yaml").write_text(
        """
name: simple
params: {}
tasks:
  - name: noop
    type: custom
    config: configs/noop.yaml
""",
        encoding="utf-8",
    )
    (project_root / "configs" / "noop.yaml").write_text(
        """
name: noop
command:
  - "/usr/bin/true"
""",
        encoding="utf-8",
    )

    runner = WorkflowRunner(base_dir=str(project_root / "workflows"), logs_dir=str(project_root / "logs" / "workflows"))
    result = runner.run("simple")

    assert result["success"] is True
    lines = events_path.read_text(encoding="utf-8").strip().splitlines()
    events = [json.loads(line) for line in lines]
    event_types = [event["event_type"] for event in events]
    assert event_types == ["run_started", "task_started", "task_finished", "run_finished"]
    assert events[1]["task_name"] == "noop"
    assert events[-1]["status"] == "success"
