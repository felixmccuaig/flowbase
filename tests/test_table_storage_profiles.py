from flowbase.core.config.schemas import FileFormat, TableConfig
from flowbase.query.engines.duckdb_engine import DuckDBEngine
from flowbase.workflows.table_loader import TableLoader


def _table_config_with_profiles() -> TableConfig:
    return TableConfig.from_dict(
        {
            "name": "bsp",
            "storage": {"base_path": "data/tables/bsp", "format": "parquet"},
            "storage_profiles": {
                "raw_append": {"base_path": "data/tables/bsp", "format": "parquet"},
                "rollup_cold": {"base_path": "data/tables_rollup/bsp", "format": "parquet"},
                "serving_entity": {"base_path": "data/serving/bsp", "format": "parquet"},
            },
        }
    )


def test_table_config_parses_storage_profiles() -> None:
    cfg = _table_config_with_profiles()
    assert cfg.storage_config is not None
    assert cfg.storage_config.base_path == "data/tables/bsp"
    assert cfg.get_storage_config("serving_entity") is not None
    assert cfg.get_storage_config("serving_entity").base_path == "data/serving/bsp"
    assert cfg.get_storage_config("rollup_cold").file_format == FileFormat.PARQUET


def test_table_loader_selects_profile_by_run_intent() -> None:
    cfg = _table_config_with_profiles()
    engine = DuckDBEngine()
    try:
        loader_features = TableLoader(engine, run_intent="features")
        selected_features, profile_features = loader_features._select_storage_config(cfg)
        assert selected_features is not None
        assert selected_features.base_path == "data/serving/bsp"
        assert profile_features == "serving_entity"

        loader_backfill = TableLoader(engine, run_intent="backfill")
        selected_backfill, profile_backfill = loader_backfill._select_storage_config(cfg)
        assert selected_backfill is not None
        assert selected_backfill.base_path == "data/tables_rollup/bsp"
        assert profile_backfill == "rollup_cold"

        loader_daily = TableLoader(engine, run_intent="daily")
        selected_daily, profile_daily = loader_daily._select_storage_config(cfg)
        assert selected_daily is not None
        assert selected_daily.base_path == "data/tables/bsp"
        assert profile_daily == "raw_append"
    finally:
        engine.close()

