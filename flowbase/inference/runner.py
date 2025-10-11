"""Inference runner for executing model predictions from config."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

from flowbase.models.trainer import ModelTrainer
from flowbase.tables.manager import TableManager


@dataclass(frozen=True)
class FilterClause:
    """Structured representation of a filter defined in config."""

    name: str
    normalized_name: str
    column: str
    operator: str = "="
    value_type: str = "string"
    required: bool = False
    aliases: Tuple[str, ...] = ()
    include_if_missing: bool = False
    default: Optional[Any] = None


class InferenceRunner:
    """Runs configured inference jobs for trained models."""

    DEFAULT_CONFIG_FILENAMES = ("config.yaml", "inference.yaml")

    def __init__(self, base_dir: str = "inference", metadata_db: Optional[str] = None):
        self.base_dir = Path(base_dir)
        self.metadata_db = metadata_db or "data/tables/.metadata.db"
        self._table_manager: Optional[TableManager] = None

    @property
    def table_manager(self) -> TableManager:
        if self._table_manager is None:
            self._table_manager = TableManager(metadata_db=self.metadata_db)
        return self._table_manager

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_config(
        self,
        model_name: str,
        config_path: Optional[str] = None
    ) -> Tuple[Dict[str, Any], Path]:
        """Load an inference config for the requested model."""
        if config_path:
            candidate = Path(config_path)
        else:
            candidate = self._discover_default_config(model_name)

        if not candidate.exists():
            raise FileNotFoundError(f"Inference config not found: {candidate}")

        with candidate.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}

        return config, candidate

    def run(
        self,
        model_name: str,
        params: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
        skip_outputs: bool = False,
    ) -> Dict[str, Any]:
        """Execute inference for a model based on its config."""
        raw_params = params or {}
        config, config_file = self.load_config(model_name, config_path=config_path)
        config_dir = config_file.parent

        # Model is just a string reference to the model name
        target_model = config.get("model", model_name)
        if not isinstance(target_model, str):
            raise ValueError("Config 'model' must be a string (model name)")

        models_dir = config.get("models_dir", "data/models")
        trainer = ModelTrainer(models_dir=models_dir)

        # Auto-resolve feature path: model config → feature_set → data/features/{feature_set}.parquet
        feature_path = self._resolve_feature_path_from_model(target_model, models_dir, config_dir)

        # Get select columns and filters from top level config
        select_columns = config.get("select_columns")
        select_clause = self._coerce_select_columns(select_columns)

        filters = self._parse_filter_clauses(config.get("filters", []))
        normalized_params = self._normalize_params(raw_params)

        where_template = config.get("where_template")
        template_required = config.get("where_template_required", [])
        manual_where = normalized_params.get("where") or normalized_params.get("where_clause")

        where_clause, used_params = self._build_where_clause(
            filters=filters,
            params=normalized_params,
            where_template=where_template,
            template_required=template_required,
            additional_where=manual_where,
        )

        # Allow empty where clause for "predict all" scenarios
        if not where_clause:
            where_clause = "1=1"  # SQL that always evaluates to true

        results = trainer.predict_from_query(
            model_name=target_model,
            feature_path=feature_path,
            where_clause=where_clause,
            select_columns=select_clause,
        )

        dataframe = self._results_to_dataframe(
            results=results,
            model_name=target_model,
            all_params=normalized_params,
            used_params=used_params,
            config=config,
        )

        outputs: Dict[str, Any] = {}
        if not skip_outputs and config.get("output"):
            outputs = self._write_outputs(
                df=dataframe,
                output_cfg=config["output"],
                config_dir=config_dir,
                params=normalized_params,
            )

        return {
            "model": target_model,
            "config_path": str(config_file),
            "feature_path": feature_path,
            "where_clause": where_clause,
            "results": results,
            "dataframe": dataframe,
            "outputs": outputs,
            "params": normalized_params,
        }

    def list_configs(self) -> List[str]:
        """List available inference configs."""
        if not self.base_dir.exists():
            return []

        configs: List[str] = []
        for directory in sorted(self.base_dir.iterdir()):
            if not directory.is_dir():
                continue
            for filename in self.DEFAULT_CONFIG_FILENAMES:
                candidate = directory / filename
                if candidate.exists():
                    configs.append(str(candidate))
                    break
        return configs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_feature_path_from_model(
        self,
        model_name: str,
        models_dir: str,
        config_dir: Path
    ) -> str:
        """
        Resolve feature path from model config.
        Flow: model.yaml → feature_set → data/features/{feature_set}.parquet
        """
        # Find model config file
        models_base = config_dir
        # Walk up to find models directory
        while models_base != models_base.parent:
            model_config_path = models_base / "models" / f"{model_name}.yaml"
            if model_config_path.exists():
                break
            models_base = models_base.parent
        else:
            # Try models_dir as absolute or relative
            models_path = Path(models_dir)
            if not models_path.is_absolute():
                models_path = config_dir / models_path
            model_config_path = models_path / f"{model_name}.yaml"

        if not model_config_path.exists():
            raise FileNotFoundError(
                f"Model config not found: {model_config_path}. "
                f"Searched from {config_dir} up to project root."
            )

        # Load model config to get feature_set
        with model_config_path.open("r", encoding="utf-8") as f:
            model_config = yaml.safe_load(f) or {}

        feature_set = model_config.get("feature_set")
        if not feature_set:
            raise ValueError(
                f"Model config {model_config_path} must specify 'feature_set'"
            )

        # Resolve to materialized features: data/features/{feature_set}.parquet
        # Walk up to find data/features directory
        features_base = model_config_path.parent
        while features_base != features_base.parent:
            feature_path = features_base.parent / "data" / "features" / f"{feature_set}.parquet"
            if feature_path.exists():
                return str(feature_path)
            features_base = features_base.parent

        # If not found, return expected path and let it fail later with clear error
        expected_path = config_dir / "data" / "features" / f"{feature_set}.parquet"
        return str(expected_path)

    def _discover_default_config(self, model_name: str) -> Path:
        base = self.base_dir / model_name
        for filename in self.DEFAULT_CONFIG_FILENAMES:
            candidate = base / filename
            if candidate.exists():
                return candidate
        # Fall back to first filename even if it doesn't exist so caller raises useful error
        return base / self.DEFAULT_CONFIG_FILENAMES[0]

    def _coerce_select_columns(self, select_columns: Any) -> Optional[str]:
        if select_columns is None:
            return None
        if isinstance(select_columns, str):
            return select_columns
        if isinstance(select_columns, (list, tuple)):
            return ", ".join(str(col) for col in select_columns)
        raise ValueError("select_columns must be string or list")

    def _normalize_param_name(self, name: str) -> str:
        return name.replace("-", "_").strip().lower()

    def _normalize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {}
        for key, value in params.items():
            normalized[self._normalize_param_name(key)] = value
        return normalized

    def _parse_filter_clauses(self, filters_cfg: Any) -> List[FilterClause]:
        clauses: List[FilterClause] = []
        for entry in filters_cfg or []:
            if not isinstance(entry, dict):
                raise ValueError("Each filter must be a mapping")

            raw_name = entry.get("param") or entry.get("name")
            column = entry.get("column")
            if not raw_name or not column:
                raise ValueError("Filter requires 'param' (or 'name') and 'column'")

            normalized = self._normalize_param_name(str(raw_name))
            aliases = tuple(
                self._normalize_param_name(alias)
                for alias in entry.get("aliases", [])
            )

            clause = FilterClause(
                name=str(raw_name),
                normalized_name=normalized,
                column=str(column),
                operator=str(entry.get("operator", "=")).strip() or "=",
                value_type=str(
                    entry.get("type")
                    or entry.get("value_type")
                    or "string"
                ).lower(),
                required=bool(entry.get("required", False)),
                aliases=aliases,
                include_if_missing=bool(entry.get("include_if_missing", False)),
                default=entry.get("default"),
            )
            clauses.append(clause)
        return clauses

    def _get_param_value(self, clause: FilterClause, params: Dict[str, Any]) -> Optional[Any]:
        for key in (clause.normalized_name, *clause.aliases):
            if key in params:
                return params[key]
        return None

    def _coerce_value(self, value: Any, value_type: str) -> Any:
        if isinstance(value, (list, tuple)):
            return [self._coerce_value(v, value_type) for v in value]

        if value is None:
            return None

        if value_type in {"int", "integer"}:
            return int(value)
        if value_type in {"float", "double", "number"}:
            return float(value)
        if value_type in {"bool", "boolean"}:
            if isinstance(value, bool):
                return value
            return str(value).lower() in {"1", "true", "yes", "y"}
        if value_type == "json":
            if isinstance(value, str):
                return json.loads(value)
            return value
        if value_type in {"date", "datetime", "timestamp"}:
            return pd.to_datetime(value)
        if value_type == "raw":
            return value
        return str(value)

    def _format_value(self, value: Any, value_type: str) -> str:
        if value is None:
            return "NULL"

        if value_type in {"int", "integer"}:
            return str(int(value))
        if value_type in {"float", "double", "number"}:
            return str(float(value))
        if value_type in {"bool", "boolean"}:
            return "TRUE" if bool(value) else "FALSE"
        if value_type == "raw":
            return str(value)
        if value_type == "json":
            text = json.dumps(value)
            return f"'{text.replace("'", "''")}'"
        if value_type == "date":
            dt = pd.to_datetime(value)
            return f"'{dt.strftime('%Y-%m-%d')}'"
        if value_type in {"datetime", "timestamp"}:
            dt = pd.to_datetime(value)
            return f"'{dt.strftime('%Y-%m-%d %H:%M:%S')}'"

        text = str(value)
        return f"'{text.replace("'", "''")}'"

    def _render_clause(self, clause: FilterClause, value: Any) -> str:
        operator = clause.operator.upper()
        column = clause.column

        if operator in {"IS NULL", "IS NOT NULL"}:
            return f"{column} {operator}"

        if value is None:
            if operator in {"=", "=="}:
                return f"{column} IS NULL"
            if operator in {"!=", "<>"}:
                return f"{column} IS NOT NULL"

        if operator in {"IN", "NOT IN"}:
            values = value if isinstance(value, (list, tuple)) else [value]
            formatted = [self._format_value(v, clause.value_type) for v in values]
            if not formatted:
                raise ValueError(f"No values provided for {column} {operator}")
            return f"{column} {operator} ({', '.join(formatted)})"

        if operator == "BETWEEN":
            if isinstance(value, (list, tuple)) and len(value) == 2:
                low, high = value
            else:
                raise ValueError(f"BETWEEN requires two values for {column}")
            return (
                f"{column} BETWEEN {self._format_value(low, clause.value_type)}"
                f" AND {self._format_value(high, clause.value_type)}"
            )

        formatted = self._format_value(value, clause.value_type)
        return f"{column} {operator} {formatted}"

    def _render_template_clause(
        self,
        template: Optional[str],
        params: Dict[str, Any],
        required: Any,
    ) -> Optional[str]:
        if not template:
            return None

        required_params = [self._normalize_param_name(p) for p in required or []]
        missing = [p for p in required_params if p not in params]
        if missing:
            raise ValueError(
                f"Missing required parameter(s) for where_template: {', '.join(missing)}"
            )

        try:
            return template.format(**params)
        except KeyError as exc:
            missing_key = str(exc).strip("'")
            raise ValueError(
                f"Missing parameter '{missing_key}' for where_template"
            ) from exc

    def _build_where_clause(
        self,
        filters: List[FilterClause],
        params: Dict[str, Any],
        where_template: Optional[str] = None,
        template_required: Any = None,
        additional_where: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        clauses: List[str] = []
        used_params: Dict[str, Any] = {}

        for clause in filters:
            raw_value = self._get_param_value(clause, params)

            if raw_value is None:
                if clause.default is not None:
                    raw_value = clause.default
                elif clause.include_if_missing and clause.operator.upper() in {"IS NULL", "IS NOT NULL"}:
                    clauses.append(f"{clause.column} {clause.operator.upper()}")
                    continue
                elif clause.required:
                    raise ValueError(f"Missing required parameter: {clause.name}")
                else:
                    continue

            coerced = self._coerce_value(raw_value, clause.value_type)
            clause_sql = self._render_clause(clause, coerced)
            clauses.append(clause_sql)
            used_params[clause.normalized_name] = coerced

        template_clause = self._render_template_clause(where_template, params, template_required)
        if template_clause:
            clauses.append(template_clause)

        if additional_where:
            clauses.append(str(additional_where))

        return " AND ".join(clause for clause in clauses if clause), used_params

    def _probability_column_name(self, label: Any) -> str:
        text = str(label)
        sanitized = "".join(ch if ch.isalnum() else "_" for ch in text)
        sanitized = "_".join(filter(None, sanitized.split("_"))).lower() or "class"
        return f"prob_{sanitized}"

    def _results_to_dataframe(
        self,
        results: List[Dict[str, Any]],
        model_name: str,
        all_params: Dict[str, Any],
        used_params: Dict[str, Any],
        config: Dict[str, Any],
    ) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for item in results:
            row: Dict[str, Any] = {
                "model": item.get("model", model_name),
                "prediction": item.get("prediction"),
            }

            identifiers = item.get("identifiers") or {}
            row.update(identifiers)

            probabilities = item.get("probabilities")
            if probabilities:
                row["probabilities_json"] = json.dumps(probabilities)
                for label, value in probabilities.items():
                    row[self._probability_column_name(label)] = value

            rows.append(row)

        df = pd.DataFrame(rows)

        options = config.get("options", {})
        include_timestamp = options.get("include_timestamp", True)
        param_scope = options.get("parameter_scope", "used")

        if include_timestamp:
            df["inference_run_at"] = datetime.utcnow().isoformat()

        params_to_include = used_params if param_scope == "used" else all_params
        for key, value in params_to_include.items():
            column_name = f"param_{key}"
            if isinstance(value, list):
                df[column_name] = json.dumps(value)
            elif isinstance(value, datetime):
                df[column_name] = value.isoformat()
            else:
                df[column_name] = value

        if "model" not in df.columns:
            df["model"] = model_name

        return df

    def _resolve_path(self, path_value: str, reference_dir: Path) -> str:
        path = Path(path_value)
        if path.is_absolute():
            return str(path)
        # Preserve glob patterns without resolving them
        has_wildcard = any(ch in path_value for ch in "*?[]")
        resolved = reference_dir / path
        if has_wildcard:
            return str(resolved.as_posix())
        return str(resolved.resolve())

    def _determine_ingest_date(
        self,
        config: Dict[str, Any],
        params: Dict[str, Any],
        df: pd.DataFrame,
    ) -> str:
        if not config:
            return datetime.utcnow().strftime("%Y-%m-%d")

        mode = str(config.get("mode", "today")).lower()

        if mode == "today":
            return datetime.utcnow().strftime("%Y-%m-%d")
        if mode == "literal":
            value = config.get("value")
            if not value:
                raise ValueError("output.table.date.value must be provided for literal mode")
            return str(value)
        if mode == "parameter":
            param_name = self._normalize_param_name(config.get("param", ""))
            if not param_name or param_name not in params:
                raise ValueError("output.table.date.param must reference a provided parameter")
            value = params[param_name]
            return self._coerce_date_value(value, config.get("format"))
        if mode == "column":
            column = config.get("column")
            if not column or column not in df.columns:
                raise ValueError("output.table.date.column must exist in prediction results")
            series = df[column].dropna()
            if series.empty:
                raise ValueError("output.table.date.column produced no values")
            return self._coerce_date_value(series.iloc[0], config.get("format"))

        raise ValueError(f"Unknown output.table.date.mode: {mode}")

    def _coerce_date_value(self, value: Any, date_format: Optional[str]) -> str:
        if isinstance(value, datetime):
            dt = value
        elif date_format:
            dt = datetime.strptime(str(value), date_format)
        else:
            dt = pd.to_datetime(value)
            if pd.isna(dt):
                raise ValueError(f"Could not parse date value: {value}")
        return dt.strftime("%Y-%m-%d")

    def _write_outputs(
        self,
        df: pd.DataFrame,
        output_cfg: Dict[str, Any],
        config_dir: Path,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        if df.empty and output_cfg.get("skip_if_empty", True):
            return info

        file_cfg = output_cfg.get("file")
        if file_cfg:
            destination = file_cfg.get("path")
            directory = file_cfg.get("directory")
            if not destination and not directory:
                raise ValueError("output.file requires 'path' or 'directory'")

            fmt = file_cfg.get("format", "parquet").lower()
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

            if directory and not destination:
                resolved_dir = Path(self._resolve_path(directory, config_dir))
                resolved_dir.mkdir(parents=True, exist_ok=True)
                filename = file_cfg.get("filename") or f"predictions_{timestamp}.{fmt}"
                destination_path = resolved_dir / filename
            else:
                destination_path = Path(self._resolve_path(destination, config_dir))
                destination_path.parent.mkdir(parents=True, exist_ok=True)

            if fmt == "parquet":
                df.to_parquet(destination_path, index=False)
            elif fmt == "csv":
                mode = file_cfg.get("mode", "overwrite")
                header = True
                if destination_path.exists() and mode == "append":
                    header = False
                df.to_csv(destination_path, index=False, mode="a" if mode == "append" else "w", header=header)
            else:
                raise ValueError(f"Unsupported output.file.format: {fmt}")

            info["file"] = str(destination_path)

        table_cfg = output_cfg.get("table")
        if table_cfg:
            table_name = table_cfg.get("name")
            table_config = table_cfg.get("config") or table_cfg.get("table_config")
            if not table_name or not table_config:
                raise ValueError("output.table requires 'name' and 'config' (or table_config)")

            resolved_table_config = self._resolve_path(table_config, config_dir)
            dataset_config = table_cfg.get("dataset_config")
            resolved_dataset = (
                self._resolve_path(dataset_config, config_dir)
                if dataset_config
                else None
            )

            ingest_date = self._determine_ingest_date(table_cfg.get("date", {}), params, df)

            temp_dir = Path("data/inference/.temp")
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_file = temp_dir / f"{table_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.parquet"
            df.to_parquet(temp_file, index=False)

            ingestion_result = self.table_manager.ingest(
                table_name=table_name,
                config_path=resolved_table_config,
                source_file=str(temp_file),
                date=ingest_date,
                dataset_config_path=resolved_dataset,
            )
            info["table"] = ingestion_result

            temp_file.unlink(missing_ok=True)

        return info


__all__ = ["InferenceRunner", "FilterClause"]
