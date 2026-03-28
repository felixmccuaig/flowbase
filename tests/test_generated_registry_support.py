from pathlib import Path

from flowbase.scrapers.runner import ScraperRunner
from flowbase.workflows.runner import WorkflowRunner


def test_workflow_runner_loads_generated_workflow_from_provider(tmp_path: Path) -> None:
    project_root = tmp_path / "example-project"
    (project_root / "workflows").mkdir(parents=True)
    (project_root / "data").mkdir(parents=True)
    provider_path = project_root / "registry_provider.py"
    provider_path.write_text(
        """
def get_generated_workflow(name):
    if name != "generated_ingest":
        return None
    return {
        "name": "generated_ingest",
        "params": {"date": "{{ today }}"},
        "tasks": [
            {
                "name": "scrape_source",
                "type": "scraper",
                "config": "registry:generated_source",
                "params": {"date": "{{ date }}"},
            }
        ],
    }

def get_generated_scraper(name):
    if name != "generated_source":
        return None
    return {
        "name": "generated_source",
        "scraper": {"function": "scrapers/example.py:fetch_data_for_day", "parameters": {}},
        "output": {"table": "example", "table_config": "tables/example.yaml"},
    }
""",
        encoding="utf-8",
    )
    (project_root / "flowbase.yaml").write_text(
        f"""
project_name: example
generated_workflows_provider: "{provider_path}:get_generated_workflow"
generated_scrapers_provider: "{provider_path}:get_generated_scraper"
""",
        encoding="utf-8",
    )

    runner = WorkflowRunner(base_dir=str(project_root / "workflows"), logs_dir=str(tmp_path / "logs"))
    config, config_file = runner.load_config("generated_ingest")

    assert config["name"] == "generated_ingest"
    assert config["tasks"][0]["config"] == "registry:generated_source"
    assert ".generated_workflows" in str(config_file)


def test_scraper_runner_run_from_spec_uses_in_memory_config(tmp_path: Path, monkeypatch) -> None:
    runner = ScraperRunner(metadata_db=str(tmp_path / "meta.db"), temp_dir=str(tmp_path / "temp"))

    spec = {
        "name": "generated_source",
        "scraper": {"function": "dummy.py:fetch_data_for_day", "parameters": {"flag": True}},
        "output": {"table": "example", "table_config": "tables/example.yaml"},
    }

    calls = {}

    def fake_load_scraper_function(_path):
        def fetch_data_for_day(*, date, flag):
            import pandas as pd
            calls["flag"] = flag
            return pd.DataFrame([{"value": 1}])
        return fetch_data_for_day

    def fake_ingest(**kwargs):
        calls["ingest"] = kwargs
        return {"destination": "data/tables/example/example_2026_01_20.parquet"}

    monkeypatch.setattr(runner, "load_scraper_function", fake_load_scraper_function)
    monkeypatch.setattr(runner.table_manager, "ingest", fake_ingest)

    result = runner.run_from_spec(spec, date="2026-01-20")

    assert result["rows"] == 1
    assert result["destination"] == "data/tables/example/example_2026_01_20.parquet"
    assert calls["flag"] is True
    assert calls["ingest"]["table_name"] == "example"
