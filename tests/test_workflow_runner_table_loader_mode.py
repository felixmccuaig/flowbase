from flowbase.workflows.runner import WorkflowRunner


def test_table_loader_prefer_s3_defaults_by_intent() -> None:
    runner = WorkflowRunner()

    assert runner._resolve_table_loader_prefer_s3("daily") is True
    assert runner._resolve_table_loader_prefer_s3("backfill") is True
    assert runner._resolve_table_loader_prefer_s3("features") is False
    assert runner._resolve_table_loader_prefer_s3("inference") is False


def test_table_loader_prefer_s3_respects_overrides() -> None:
    runner = WorkflowRunner()
    runner.project_config = {
        "table_loader_prefer_s3": {
            "default": False,
            "features": True,
            "inference": "false",
            "backfill": "true",
        }
    }

    assert runner._resolve_table_loader_prefer_s3("features") is True
    assert runner._resolve_table_loader_prefer_s3("inference") is False
    assert runner._resolve_table_loader_prefer_s3("backfill") is True
    assert runner._resolve_table_loader_prefer_s3("daily") is False
