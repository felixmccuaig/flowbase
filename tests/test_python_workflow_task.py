import json
from pathlib import Path

from flowbase.workflows.runner import WorkflowRunner


def test_workflow_runner_executes_python_task_from_callable_spec(tmp_path: Path) -> None:
    project_root = tmp_path / "example-project"
    (project_root / "workflows" / "pyflow").mkdir(parents=True)
    (project_root / "data").mkdir(parents=True)
    provider_path = project_root / "python_tasks.py"
    provider_path.write_text(
        """
def emit_summary(date, workflow_task_name, project_root):
    return {
        "handled_date": date,
        "task_name": workflow_task_name,
        "project_root": str(project_root),
    }
""",
        encoding="utf-8",
    )
    (project_root / "workflows" / "pyflow" / "workflow.yaml").write_text(
        f"""
name: pyflow
params:
  date: "2026-03-28"
tasks:
  - name: emit_summary
    type: python
    config: "{provider_path}:emit_summary"
    params:
      date: "{{{{ date }}}}"
""",
        encoding="utf-8",
    )

    runner = WorkflowRunner(
        base_dir=str(project_root / "workflows"),
        logs_dir=str(tmp_path / "logs"),
    )
    result = runner.run("pyflow")

    assert result["success"] is True
    task_output = result["results"][0]["output"]
    assert task_output["type"] == "python"
    assert task_output["handled_date"] == "2026-03-28"
    assert task_output["task_name"] == "emit_summary"
    assert task_output["project_root"] == str(project_root)


def test_workflow_runner_executes_python_task_from_yaml_config(tmp_path: Path) -> None:
    project_root = tmp_path / "example-project"
    (project_root / "workflows" / "pyflow").mkdir(parents=True)
    (project_root / "configs").mkdir(parents=True)
    (project_root / "data").mkdir(parents=True)
    provider_path = project_root / "python_tasks.py"
    provider_path.write_text(
        """
def emit_payload(value, logger=None):
    if logger:
        logger.info("python task executed")
    return {"value": value}
""",
        encoding="utf-8",
    )
    (project_root / "configs" / "python_task.yaml").write_text(
        f"""
callable: "{provider_path}:emit_payload"
params:
  value: "ok"
""",
        encoding="utf-8",
    )
    (project_root / "workflows" / "pyflow" / "workflow.yaml").write_text(
        """
name: pyflow
tasks:
  - name: emit_payload
    type: python
    config: configs/python_task.yaml
""",
        encoding="utf-8",
    )

    runner = WorkflowRunner(
        base_dir=str(project_root / "workflows"),
        logs_dir=str(tmp_path / "logs"),
    )
    result = runner.run("pyflow")

    assert result["success"] is True
    task_output = result["results"][0]["output"]
    assert json.loads(json.dumps(task_output))["value"] == "ok"
