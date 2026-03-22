"""Transform runner for building curated datasets and marts via SQL."""

from __future__ import annotations

import re
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from flowbase.query.engines.duckdb_engine import DuckDBEngine


_TEMPLATE_PATTERN = re.compile(r"\{\{\s*(\w+)\s*\}\}")
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class TransformRunner:
    """Runs declarative SQL transform configs."""

    def __init__(
        self,
        query_engine_config: Optional[Dict[str, Any]] = None,
        project_config: Optional[Dict[str, Any]] = None,
    ):
        self.query_engine_config = query_engine_config or {}
        self.project_config = project_config or {}
        storage = self.project_config.get("storage", {})
        if not isinstance(storage, dict):
            storage = {}
        self.s3_bucket = storage.get("bucket")
        self.s3_prefix = storage.get("prefix", "")
        self._s3_ready = False

    def run(
        self,
        config_path: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        project_root: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a transform config."""
        if config_dict:
            config = config_dict
            config_file = Path(".")
        elif config_path:
            config_file = Path(config_path)
            with config_file.open("r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        else:
            raise ValueError("Either config_path or config_dict must be provided")

        runtime_params = dict(params or {})
        if "date" in runtime_params:
            date_value = str(runtime_params["date"])
            runtime_params.setdefault("date_underscore", date_value.replace("-", "_"))
            runtime_params.setdefault("date_compact", date_value.replace("-", ""))

        resolved = self._substitute_templates(config, runtime_params)

        base_dir = Path(project_root) if project_root else config_file.parent
        engine_cfg = resolved.get("duckdb", {}) or self.query_engine_config
        database = engine_cfg.get("database")
        engine = DuckDBEngine(database=database, config=engine_cfg)
        outputs: List[Dict[str, Any]] = []
        model_results: List[Dict[str, Any]] = []

        try:
            self._register_sources(engine, resolved.get("sources", []), base_dir)

            models = resolved.get("models", [])
            if not isinstance(models, list) or not models:
                raise ValueError("Transform config must define a non-empty 'models' list")

            for model in models:
                model_result = self._materialize_model(engine, model, base_dir)
                model_results.append(model_result)
                if model_result.get("output"):
                    outputs.append(model_result["output"])
        finally:
            engine.close()

        return {
            "type": "transform",
            "name": resolved.get("name"),
            "models": model_results,
            "outputs": outputs,
        }

    def _register_sources(self, engine: DuckDBEngine, sources_cfg: Any, base_dir: Path) -> None:
        sources = self._normalize_sources(sources_cfg)
        for source in sources:
            name = source["name"]
            self._validate_identifier(name)
            source_format = str(source.get("format", "parquet")).lower()
            required = bool(source.get("required", True))
            source_path = str(source.get("path", "")).strip()
            if not source_path:
                raise ValueError(f"Source '{name}' missing required 'path'")

            resolved_path = self._resolve_data_path(source_path, base_dir)

            if source_path.startswith("s3://"):
                self._register_source_view(engine, name, source_format, source_path)
                continue

            local_matches = sorted(glob(resolved_path)) if self._is_glob_pattern(resolved_path) else []
            if self._is_glob_pattern(resolved_path):
                has_data = len(local_matches) > 0
            else:
                has_data = Path(resolved_path).exists()

            if has_data:
                if self._is_glob_pattern(resolved_path):
                    pattern = str(Path(resolved_path).resolve())
                    self._register_source_view(engine, name, source_format, pattern)
                else:
                    abs_path = str(Path(resolved_path).resolve())
                    self._register_source_view(engine, name, source_format, abs_path)
                continue

            # Local file missing: if project S3 storage is configured, try same path on S3.
            s3_fallback_path = self._build_s3_fallback_path(source_path)
            s3_fallback_error = None
            if s3_fallback_path:
                try:
                    self._ensure_s3_ready(engine)
                    self._register_source_view(engine, name, source_format, s3_fallback_path)
                    continue
                except Exception as exc:
                    s3_fallback_error = str(exc)

            if required:
                message = f"Required source '{name}' not found: {source_path}"
                if s3_fallback_path:
                    message += (
                        f" (S3 fallback failed: {s3_fallback_path}"
                        + (f" [{s3_fallback_error}]" if s3_fallback_error else "")
                        + ")"
                    )
                raise FileNotFoundError(message)

            empty_schema = source.get("empty_schema", [])
            if not empty_schema:
                raise ValueError(
                    f"Optional source '{name}' not found and no 'empty_schema' provided: {source_path}"
                )
            self._create_empty_view(engine, name, empty_schema)

    def _materialize_model(
        self,
        engine: DuckDBEngine,
        model_cfg: Dict[str, Any],
        base_dir: Path,
    ) -> Dict[str, Any]:
        if not isinstance(model_cfg, dict):
            raise ValueError("Each model entry must be an object")

        name = str(model_cfg.get("name", "")).strip()
        if not name:
            raise ValueError("Model is missing required 'name'")
        self._validate_identifier(name)

        materialized = str(model_cfg.get("materialized", "view")).lower()
        if materialized not in {"view", "table"}:
            raise ValueError(f"Unsupported materialized type for model '{name}': {materialized}")

        sql = self._load_model_sql(model_cfg, base_dir)
        keyword = "VIEW" if materialized == "view" else "TABLE"
        engine.conn.execute(f"CREATE OR REPLACE {keyword} {name} AS {sql}")

        row_count_row = engine.conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()
        row_count = int(row_count_row[0]) if row_count_row else 0

        model_result: Dict[str, Any] = {
            "name": name,
            "materialized": materialized,
            "row_count": row_count,
        }

        output_cfg = model_cfg.get("output")
        if output_cfg:
            output = self._write_output(engine, name, output_cfg, base_dir, row_count)
            model_result["output"] = output

        return model_result

    def _write_output(
        self,
        engine: DuckDBEngine,
        model_name: str,
        output_cfg: Any,
        base_dir: Path,
        row_count: int,
    ) -> Dict[str, Any]:
        if isinstance(output_cfg, str):
            output_cfg = {"path": output_cfg}
        if not isinstance(output_cfg, dict):
            raise ValueError(f"Invalid output config for model '{model_name}'")

        output_path_value = str(output_cfg.get("path", "")).strip()
        if not output_path_value:
            raise ValueError(f"Output path missing for model '{model_name}'")

        output_format = str(output_cfg.get("format", "parquet")).lower()
        if output_format not in {"parquet", "csv"}:
            raise ValueError(f"Unsupported output format for model '{model_name}': {output_format}")

        output_path = Path(self._resolve_data_path(output_path_value, base_dir)).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        escaped_path = self._escape_sql_string(str(output_path))
        if output_format == "parquet":
            engine.conn.execute(f"COPY (SELECT * FROM {model_name}) TO '{escaped_path}' (FORMAT PARQUET)")
        else:
            engine.conn.execute(
                f"COPY (SELECT * FROM {model_name}) TO '{escaped_path}' (FORMAT CSV, HEADER TRUE)"
            )

        return {
            "model": model_name,
            "path": str(output_path),
            "format": output_format,
            "row_count": row_count,
        }

    def _load_model_sql(self, model_cfg: Dict[str, Any], base_dir: Path) -> str:
        inline_sql = model_cfg.get("sql")
        sql_path = model_cfg.get("sql_file")

        if inline_sql and sql_path:
            raise ValueError("Use only one of 'sql' (inline) or 'sql_file' for a model")

        if sql_path:
            path = Path(self._resolve_data_path(str(sql_path), base_dir))
            with path.open("r", encoding="utf-8") as f:
                return f.read().strip()

        if inline_sql:
            sql_str = str(inline_sql).strip()
            if not sql_str:
                raise ValueError("Inline SQL cannot be empty")
            return sql_str

        raise ValueError("Model must define either 'sql' or 'sql_file'")

    def _register_source_view(
        self,
        engine: DuckDBEngine,
        view_name: str,
        source_format: str,
        source_path: str,
    ) -> None:
        escaped_path = self._escape_sql_string(source_path)
        if source_format == "parquet":
            engine.conn.execute(
                f"CREATE OR REPLACE VIEW {view_name} AS "
                f"SELECT * FROM read_parquet('{escaped_path}', union_by_name=true)"
            )
        elif source_format == "csv":
            engine.conn.execute(
                f"CREATE OR REPLACE VIEW {view_name} AS "
                f"SELECT * FROM read_csv_auto('{escaped_path}')"
            )
        elif source_format == "json":
            engine.conn.execute(
                f"CREATE OR REPLACE VIEW {view_name} AS "
                f"SELECT * FROM read_json_auto('{escaped_path}')"
            )
        else:
            raise ValueError(f"Unsupported source format: {source_format}")

    def _create_empty_view(self, engine: DuckDBEngine, view_name: str, schema_cfg: Any) -> None:
        if not isinstance(schema_cfg, list) or not schema_cfg:
            raise ValueError(f"empty_schema for '{view_name}' must be a non-empty list")

        columns = []
        for column in schema_cfg:
            if not isinstance(column, dict):
                raise ValueError(f"empty_schema entry for '{view_name}' must be an object")
            col_name = str(column.get("name", "")).strip()
            col_type = str(column.get("type", "")).strip()
            if not col_name or not col_type:
                raise ValueError(f"empty_schema entry for '{view_name}' must include name and type")
            self._validate_identifier(col_name)
            columns.append(f"CAST(NULL AS {col_type}) AS {col_name}")

        select_clause = ", ".join(columns)
        engine.conn.execute(
            f"CREATE OR REPLACE VIEW {view_name} AS SELECT {select_clause} WHERE FALSE"
        )

    def _ensure_s3_ready(self, engine: DuckDBEngine) -> None:
        if self._s3_ready:
            return

        try:
            engine.conn.execute("INSTALL httpfs")
        except Exception:
            pass
        engine.conn.execute("LOAD httpfs")

        import os

        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        if region:
            escaped = self._escape_sql_string(region)
            engine.conn.execute(f"SET s3_region='{escaped}'")

        self._s3_ready = True

    def _build_s3_fallback_path(self, source_path: str) -> Optional[str]:
        if source_path.startswith("s3://"):
            return source_path
        if not self.s3_bucket:
            return None
        if Path(source_path).is_absolute():
            return None

        key = "/".join(
            segment for segment in [str(self.s3_prefix).strip("/"), source_path.strip("/")] if segment
        )
        return f"s3://{self.s3_bucket}/{key}"

    def _normalize_sources(self, sources_cfg: Any) -> List[Dict[str, Any]]:
        if isinstance(sources_cfg, dict):
            normalized = []
            for name, value in sources_cfg.items():
                if isinstance(value, str):
                    normalized.append({"name": name, "path": value})
                elif isinstance(value, dict):
                    source = dict(value)
                    source.setdefault("name", name)
                    normalized.append(source)
                else:
                    raise ValueError(f"Invalid source config for '{name}'")
            return normalized

        if isinstance(sources_cfg, list):
            normalized = []
            for value in sources_cfg:
                if not isinstance(value, dict):
                    raise ValueError("Each source entry must be an object")
                if not value.get("name"):
                    raise ValueError("Each source entry must include 'name'")
                normalized.append(value)
            return normalized

        if not sources_cfg:
            return []
        raise ValueError("Sources must be a mapping or a list")

    def _substitute_templates(self, obj: Any, params: Dict[str, Any]) -> Any:
        if isinstance(obj, str):
            return _TEMPLATE_PATTERN.sub(
                lambda match: str(params.get(match.group(1), match.group(0))),
                obj,
            )
        if isinstance(obj, list):
            return [self._substitute_templates(item, params) for item in obj]
        if isinstance(obj, dict):
            return {key: self._substitute_templates(value, params) for key, value in obj.items()}
        return obj

    def _resolve_data_path(self, value: str, base_dir: Path) -> str:
        if value.startswith("s3://"):
            return value
        path = Path(value)
        if path.is_absolute():
            return str(path)
        return str((base_dir / path).resolve())

    def _validate_identifier(self, value: str) -> None:
        if not _IDENTIFIER_PATTERN.match(value):
            raise ValueError(f"Invalid SQL identifier: {value}")

    def _escape_sql_string(self, value: str) -> str:
        return value.replace("'", "''")

    def _is_glob_pattern(self, value: str) -> bool:
        return any(token in value for token in ("*", "?", "[", "]"))
