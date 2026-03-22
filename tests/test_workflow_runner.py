from pathlib import Path

from flowbase.workflows.runner import WorkflowRunner


def test_find_project_root_prefers_flowbase_yaml_over_nested_models(tmp_path: Path) -> None:
    project_root = tmp_path / "flowbase-greyhounds"
    nested = project_root / "transforms" / "greyhounds" / "configs"

    nested.mkdir(parents=True)
    (project_root / "workflows").mkdir(parents=True)
    (project_root / "data").mkdir(parents=True)
    (project_root / "transforms" / "greyhounds" / "models").mkdir(parents=True)
    (project_root / "flowbase.yaml").write_text("name: test\n", encoding="utf-8")

    runner = WorkflowRunner(
        base_dir=str(project_root / "workflows"),
        logs_dir=str(tmp_path / "logs"),
    )
    resolved = runner._find_project_root(nested)

    assert resolved == project_root
