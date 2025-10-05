"""Compile dataset configs with type casting and cleaning."""

from typing import Dict, List, Any
import yaml


class DatasetCompiler:
    """Compiles dataset configs into SQL views with proper types."""

    def __init__(self, source_table: str = "raw_data"):
        self.source_table = source_table

    def compile(self, config: Dict[str, Any]) -> str:
        """
        Compile a dataset config to SQL.

        Args:
            config: Dataset configuration

        Returns:
            Compiled SQL query that creates a clean, typed view
        """
        select_parts = []

        # Get columns with type casting
        columns = config.get("columns", [])

        for col in columns:
            name = col.get("name")
            source_col = col.get("source", name)  # Source column name (if different)
            dtype = col.get("type")
            transform = col.get("transform")
            default = col.get("default")

            # Build column expression
            col_expr = source_col

            # Apply transformation
            if transform:
                if transform == "trim":
                    col_expr = f"TRIM({col_expr})"
                elif transform == "upper":
                    col_expr = f"UPPER({col_expr})"
                elif transform == "lower":
                    col_expr = f"LOWER({col_expr})"

            # Apply type casting
            if dtype:
                cast_expr = self._get_cast_expression(col_expr, dtype, default)
                select_parts.append(f"    {cast_expr} AS {name}")
            else:
                select_parts.append(f"    {col_expr} AS {name}")

        # Build WHERE clause for filters
        filters = config.get("filters", [])
        where_conditions = []

        for f in filters:
            condition = self._compile_filter(f)
            if condition:
                where_conditions.append(condition)

        # Assemble SQL
        sql_parts = ["SELECT"]
        sql_parts.append(",\n".join(select_parts))
        sql_parts.append(f"FROM {self.source_table}")

        if where_conditions:
            sql_parts.append("WHERE " + " AND ".join(where_conditions))

        # Add ordering if specified
        if config.get("order_by"):
            order_cols = ", ".join(config["order_by"])
            sql_parts.append(f"ORDER BY {order_cols}")

        return "\n".join(sql_parts)

    def _get_cast_expression(self, col_expr: str, dtype: str, default: Any = None) -> str:
        """Generate type casting expression with optional default."""
        dtype_upper = dtype.upper()

        if default is not None:
            # Use COALESCE for default values
            if dtype_upper in ["INTEGER", "INT", "BIGINT"]:
                return f"COALESCE(TRY_CAST({col_expr} AS {dtype_upper}), {default})"
            elif dtype_upper in ["DOUBLE", "FLOAT", "DECIMAL"]:
                return f"COALESCE(TRY_CAST({col_expr} AS DOUBLE), {default})"
            elif dtype_upper in ["BOOLEAN", "BOOL"]:
                return f"COALESCE(TRY_CAST({col_expr} AS BOOLEAN), {str(default).lower()})"
            elif dtype_upper == "DATE":
                return f"COALESCE(TRY_CAST({col_expr} AS DATE), DATE '{default}')"
            elif dtype_upper == "TIMESTAMP":
                return f"COALESCE(TRY_CAST({col_expr} AS TIMESTAMP), TIMESTAMP '{default}')"
            else:  # VARCHAR, TEXT
                return f"COALESCE(CAST({col_expr} AS VARCHAR), '{default}')"
        else:
            # Simple cast without default
            if dtype_upper in ["INTEGER", "INT", "BIGINT", "DOUBLE", "FLOAT", "DECIMAL"]:
                return f"TRY_CAST({col_expr} AS {dtype_upper})"
            elif dtype_upper in ["BOOLEAN", "BOOL"]:
                return f"TRY_CAST({col_expr} AS BOOLEAN)"
            elif dtype_upper == "DATE":
                return f"TRY_CAST({col_expr} AS DATE)"
            elif dtype_upper == "TIMESTAMP":
                return f"TRY_CAST({col_expr} AS TIMESTAMP)"
            else:  # VARCHAR, TEXT
                return f"CAST({col_expr} AS VARCHAR)"

    def _compile_filter(self, filter_def: Dict[str, Any]) -> str:
        """Compile a filter condition."""
        column = filter_def.get("column")
        operator = filter_def.get("operator", "=")
        value = filter_def.get("value")
        allow_null = filter_def.get("allow_null", False)

        condition = None

        if operator == "is_not_null":
            condition = f"{column} IS NOT NULL"
        elif operator == "is_null":
            condition = f"{column} IS NULL"
        elif operator == "not_empty":
            condition = f"{column} IS NOT NULL AND {column} != ''"
        elif operator == "=":
            if isinstance(value, str):
                condition = f"{column} = '{value}'"
            else:
                condition = f"{column} = {value}"
        elif operator in [">", "<", ">=", "<=", "!="]:
            if isinstance(value, str):
                condition = f"{column} {operator} '{value}'"
            else:
                condition = f"{column} {operator} {value}"
        elif operator == "in":
            values_str = ", ".join([f"'{v}'" if isinstance(v, str) else str(v) for v in value])
            condition = f"{column} IN ({values_str})"
        elif operator == "like":
            condition = f"{column} LIKE '{value}'"

        # Wrap with OR IS NULL if allow_null is True
        if condition and allow_null:
            condition = f"({condition} OR {column} IS NULL)"

        return condition


def load_dataset_config(yaml_path: str) -> Dict[str, Any]:
    """Load dataset config from YAML file."""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def save_dataset_config(config: Dict[str, Any], yaml_path: str) -> None:
    """Save dataset config to YAML file."""
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
