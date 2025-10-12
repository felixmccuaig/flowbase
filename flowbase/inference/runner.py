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

    def __init__(self, base_dir: str = "inference", metadata_db: Optional[str] = None, data_root: Optional[str] = None,
                 s3_bucket: Optional[str] = None, s3_prefix: Optional[str] = ""):
        self.base_dir = Path(base_dir)
        self.metadata_db = metadata_db or "data/tables/.metadata.db"
        self._table_manager: Optional[TableManager] = None
        self.data_root = Path(data_root) if data_root else None
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix

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
        trainer = ModelTrainer(
            models_dir=models_dir,
            data_root=str(self.data_root) if self.data_root else None,
            s3_bucket=self.s3_bucket,
            s3_prefix=self.s3_prefix
        )

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

        # Apply post-processing transformations if configured
        if config.get("post_processing"):
            dataframe = self._apply_post_processing(dataframe, config["post_processing"])

        # Run validation checks if configured
        if config.get("validation"):
            self._validate_predictions(dataframe, config["validation"])

        outputs: Dict[str, Any] = {}
        if not skip_outputs and config.get("output"):
            job_name = config.get("name", model_name)
            outputs = self._write_outputs(
                df=dataframe,
                output_cfg=config["output"],
                config_dir=config_dir,
                params=normalized_params,
                job_name=job_name,
                feature_path=feature_path,
                where_clause=where_clause,
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
        # Start from config_dir and walk up to find project root with models/ directory
        search_dir = config_dir.resolve()
        model_config_path = None

        # Walk up directory tree looking for models/{model_name}.yaml
        while search_dir != search_dir.parent:
            candidate = search_dir / "models" / f"{model_name}.yaml"
            if candidate.exists():
                model_config_path = candidate
                break
            search_dir = search_dir.parent

        if not model_config_path:
            raise FileNotFoundError(
                f"Model config not found: models/{model_name}.yaml. "
                f"Searched from {config_dir} up to root."
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
        # Use the same project root where we found the model config
        project_root = model_config_path.parent.parent  # models/ -> project_root

        # If data_root is set (Lambda), use it; otherwise use project_root
        if self.data_root:
            feature_path = self.data_root / "data" / "features" / f"{feature_set}.parquet"
        else:
            feature_path = project_root / "data" / "features" / f"{feature_set}.parquet"

        if not feature_path.exists():
            raise FileNotFoundError(
                f"Feature file not found: {feature_path}. "
                f"Expected materialized features at data/features/{feature_set}.parquet"
            )

        return str(feature_path)

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

    def _apply_post_processing(
        self,
        df: pd.DataFrame,
        post_processing_cfg: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        """
        Apply SQL-based post-processing transformations to predictions.

        Args:
            df: DataFrame with predictions
            post_processing_cfg: List of transformation configs with 'name' and 'expression'

        Returns:
            DataFrame with new columns added
        """
        if df.empty:
            return df

        # Use DuckDB to apply SQL transformations
        try:
            import duckdb
        except ImportError:
            raise ImportError("DuckDB is required for post-processing transformations")

        conn = duckdb.connect(":memory:")
        conn.register("predictions", df)

        # First, create a CTE that casts prediction to DOUBLE if it exists
        # This ensures all post-processing expressions can use numeric operations
        if "prediction" in df.columns:
            cast_columns = []
            for col in df.columns:
                if col == "prediction":
                    cast_columns.append("CAST(prediction AS DOUBLE) AS prediction")
                else:
                    cast_columns.append(col)
            base_sql = f"SELECT {', '.join(cast_columns)} FROM predictions"
        else:
            base_sql = "SELECT * FROM predictions"

        # Build SQL with all transformations
        select_parts = ["*"]  # Start with all columns from CTE

        for transform in post_processing_cfg:
            name = transform.get("name")
            expression = transform.get("expression")

            if not name or not expression:
                raise ValueError("Each post_processing entry requires 'name' and 'expression'")

            # Clean up expression (remove leading/trailing whitespace and newlines)
            expression = expression.strip()

            # Add the transformation as a new column
            select_parts.append(f"({expression}) AS {name}")

        sql = f"WITH base AS ({base_sql}) SELECT {', '.join(select_parts)} FROM base"

        try:
            result_df = conn.execute(sql).fetchdf()
            conn.close()
            return result_df
        except Exception as e:
            conn.close()
            raise ValueError(f"Failed to apply post-processing: {e}") from e

    def _validate_predictions(
        self,
        df: pd.DataFrame,
        validation_cfg: List[Dict[str, Any]],
    ) -> None:
        """
        Run validation checks on predictions.

        Args:
            df: DataFrame with predictions
            validation_cfg: List of validation rule configs

        Raises:
            ValueError: If validation fails with error_on_fail=True
        """
        if df.empty:
            return

        for rule in validation_cfg:
            rule_type = rule.get("type")

            if rule_type == "group_count":
                self._validate_group_count(df, rule)
            elif rule_type == "group_sum":
                self._validate_group_sum(df, rule)
            elif rule_type == "column_range":
                self._validate_column_range(df, rule)
            elif rule_type == "not_null":
                self._validate_not_null(df, rule)
            elif rule_type == "required_columns":
                self._validate_required_columns(df, rule)
            else:
                raise ValueError(f"Unknown validation type: {rule_type}")

    def _validate_group_count(self, df: pd.DataFrame, rule: Dict[str, Any]) -> None:
        """Validate the count of rows per group."""
        group_by = rule.get("group_by")
        min_count = rule.get("min")
        max_count = rule.get("max")
        error_on_fail = rule.get("error_on_fail", True)
        message = rule.get("message", "Group count validation failed")

        if not group_by:
            raise ValueError("group_count validation requires 'group_by'")

        counts = df.groupby(group_by).size()

        failed_groups = []
        for group_value, count in counts.items():
            if (min_count is not None and count < min_count) or \
               (max_count is not None and count > max_count):
                failed_groups.append((group_value, count))

        if failed_groups:
            for group_value, count in failed_groups:
                formatted_message = message.format(group_value=group_value, count=count)
                if error_on_fail:
                    raise ValueError(formatted_message)
                else:
                    print(f"WARNING: {formatted_message}")

    def _validate_group_sum(self, df: pd.DataFrame, rule: Dict[str, Any]) -> None:
        """Validate the sum of a column per group."""
        group_by = rule.get("group_by")
        column = rule.get("column")
        expected = rule.get("expected")
        tolerance = rule.get("tolerance", 0.0)
        error_on_fail = rule.get("error_on_fail", True)
        warning_on_fail = rule.get("warning_on_fail", False)
        message = rule.get("message", "Group sum validation failed")

        if not group_by or not column:
            raise ValueError("group_sum validation requires 'group_by' and 'column'")

        if expected is None:
            raise ValueError("group_sum validation requires 'expected' value")

        sums = df.groupby(group_by)[column].sum()

        failed_groups = []
        for group_value, sum_value in sums.items():
            if abs(sum_value - expected) > tolerance:
                failed_groups.append((group_value, sum_value))

        if failed_groups:
            for group_value, sum_value in failed_groups:
                formatted_message = message.format(group_value=group_value, sum=sum_value)
                if error_on_fail:
                    raise ValueError(formatted_message)
                elif warning_on_fail:
                    print(f"WARNING: {formatted_message}")

    def _validate_column_range(self, df: pd.DataFrame, rule: Dict[str, Any]) -> None:
        """Validate that column values are within a specified range."""
        column = rule.get("column")
        min_value = rule.get("min")
        max_value = rule.get("max")
        error_on_fail = rule.get("error_on_fail", True)
        message = rule.get("message", "Column range validation failed")

        if not column:
            raise ValueError("column_range validation requires 'column'")

        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in dataframe")

        out_of_range = df[
            (df[column] < min_value) if min_value is not None else False |
            (df[column] > max_value) if max_value is not None else False
        ]

        if not out_of_range.empty:
            for _, row in out_of_range.iterrows():
                # Try to format message with row data
                try:
                    formatted_message = message.format(**row.to_dict(), value=row[column])
                except KeyError:
                    formatted_message = f"{message} (column={column}, value={row[column]})"

                if error_on_fail:
                    raise ValueError(formatted_message)
                else:
                    print(f"WARNING: {formatted_message}")

    def _validate_not_null(self, df: pd.DataFrame, rule: Dict[str, Any]) -> None:
        """Validate that columns do not contain null values."""
        columns = rule.get("columns", [])
        error_on_fail = rule.get("error_on_fail", True)
        message = rule.get("message", "Null values found")

        if not columns:
            raise ValueError("not_null validation requires 'columns'")

        for column in columns:
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in dataframe")

            null_count = df[column].isna().sum()
            if null_count > 0:
                formatted_message = f"{message} (column={column}, null_count={null_count})"
                if error_on_fail:
                    raise ValueError(formatted_message)
                else:
                    print(f"WARNING: {formatted_message}")

    def _validate_required_columns(self, df: pd.DataFrame, rule: Dict[str, Any]) -> None:
        """Validate that all required columns are present."""
        columns = rule.get("columns", [])
        error_on_fail = rule.get("error_on_fail", True)

        if not columns:
            raise ValueError("required_columns validation requires 'columns'")

        missing = [col for col in columns if col not in df.columns]

        if missing:
            message = f"Missing required columns: {', '.join(missing)}"
            if error_on_fail:
                raise ValueError(message)
            else:
                print(f"WARNING: {message}")

    def _write_outputs(
        self,
        df: pd.DataFrame,
        output_cfg: Dict[str, Any],
        config_dir: Path,
        params: Dict[str, Any],
        job_name: Optional[str] = None,
        feature_path: Optional[str] = None,
        where_clause: Optional[str] = None,
    ) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        if df.empty and output_cfg.get("skip_if_empty", True):
            return info

        file_cfg = output_cfg.get("file")
        if file_cfg:
            destination = file_cfg.get("path")
            directory = file_cfg.get("directory")

            # Auto-derive directory from job name if not specified
            if not destination and not directory:
                if not job_name:
                    raise ValueError("output.file requires 'path' or 'directory', or job_name for auto-resolution")
                # Find project root by walking up from config_dir to find data/ or models/
                search_dir = config_dir.resolve()
                project_root = None
                while search_dir != search_dir.parent:
                    if (search_dir / "data").exists() or (search_dir / "models").exists():
                        project_root = search_dir
                        break
                    search_dir = search_dir.parent

                if not project_root:
                    project_root = config_dir.resolve()

                # Use data_root if specified (for Lambda /tmp support)
                if self.data_root:
                    directory = str(self.data_root / "data" / "predictions" / job_name)
                else:
                    directory = str(project_root / "data" / "predictions" / job_name)

            fmt = file_cfg.get("format", "parquet").lower()
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

            if directory and not destination:
                # Resolve directory (either from config or auto-derived)
                if directory.startswith("/"):
                    # Absolute path
                    resolved_dir = Path(directory)
                else:
                    # Relative path from config_dir
                    resolved_dir = Path(self._resolve_path(directory, config_dir))

                resolved_dir.mkdir(parents=True, exist_ok=True)
                filename = file_cfg.get("filename") or f"predictions_{timestamp}.{fmt}"
                # Replace template placeholders in filename with parameter values
                try:
                    filename = filename.format(**params)
                except KeyError:
                    # If placeholder not found in params, leave it as-is (don't fail)
                    pass
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

            # Use data_root if specified (for Lambda /tmp support)
            if self.data_root:
                temp_dir = self.data_root / "data" / "inference" / ".temp"
            else:
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

        # Save feature snapshot if enabled
        save_features = output_cfg.get("save_features_snapshot", False)
        if save_features and feature_path and where_clause and self.s3_bucket:
            try:
                from flowbase.query.engines.duckdb_engine import DuckDBEngine

                # Query the features that were used for these predictions
                sql = f"""
                SELECT *
                FROM read_parquet('{feature_path}')
                WHERE {where_clause}
                """

                engine = DuckDBEngine()
                features_df = engine.execute(sql)
                engine.close()

                if not features_df.empty:
                    # Determine where to save the feature snapshot
                    if file_cfg:
                        # Save in the same directory as the prediction file
                        if directory:
                            if directory.startswith("/"):
                                features_dir = Path(directory)
                            else:
                                features_dir = Path(self._resolve_path(directory, config_dir))
                        else:
                            # Get directory from the saved prediction file path
                            prediction_path = Path(info.get("file", ""))
                            features_dir = prediction_path.parent

                        features_dir.mkdir(parents=True, exist_ok=True)

                        # Generate filename: features_{date}.parquet or features_{timestamp}.parquet
                        filename_template = file_cfg.get("filename", "")
                        if "{date}" in filename_template or "predictions_" in filename_template:
                            # Extract the date/timestamp from the predictions filename
                            features_filename = filename_template.replace("predictions_", "features_")
                            try:
                                features_filename = features_filename.format(**params)
                            except KeyError:
                                pass
                        else:
                            # Fallback to timestamp
                            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                            features_filename = f"features_{timestamp}.parquet"

                        features_path_local = features_dir / features_filename
                        features_df.to_parquet(features_path_local, index=False)

                        # Upload to S3 if sync is enabled
                        if self.s3_sync:
                            # Construct S3 key based on local path structure
                            import os
                            data_root_env = os.environ.get('FLOWBASE_DATA_ROOT')
                            if data_root_env and features_path_local.is_relative_to(Path(data_root_env)):
                                relative_path = features_path_local.relative_to(Path(data_root_env))
                            else:
                                # Try to find project root
                                search_dir = config_dir.resolve()
                                project_root = None
                                while search_dir != search_dir.parent:
                                    if (search_dir / "data").exists() or (search_dir / "models").exists():
                                        project_root = search_dir
                                        break
                                    search_dir = search_dir.parent

                                if project_root and features_path_local.is_relative_to(project_root):
                                    relative_path = features_path_local.relative_to(project_root)
                                else:
                                    # Fallback: use the filename with a standard prefix
                                    relative_path = Path("data/features") / features_filename

                            s3_key = str(relative_path)
                            if self.s3_sync.upload_file(features_path_local, s3_key):
                                s3_url = f"s3://{self.s3_sync.bucket}/{self.s3_sync.prefix}/{s3_key}".replace("//", "/")
                                info["features_snapshot"] = str(features_path_local)
                                info["features_snapshot_s3"] = s3_url
            except Exception as e:
                # Log error but don't fail the entire inference job
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to save feature snapshot: {e}")

        return info


__all__ = ["InferenceRunner", "FilterClause"]
