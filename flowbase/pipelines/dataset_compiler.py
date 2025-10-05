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
        # Check if this is a merged dataset (has multiple sources)
        sources = config.get("sources")
        if sources:
            return self._compile_merged(config)
        else:
            return self._compile_single(config)

    def _compile_single(self, config: Dict[str, Any]) -> str:
        """Compile a single-source dataset."""
        select_parts = []

        # Get columns with type casting
        columns = config.get("columns", [])

        for col in columns:
            name = col.get("name")
            source_col = col.get("source", name)  # Source column name (if different)
            expression = col.get("expression")  # Custom SQL expression
            dtype = col.get("type")
            transform = col.get("transform")
            default = col.get("default")

            # Build column expression
            if expression:
                col_expr = expression
            else:
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

        # Check if source specifies a table config
        source = config.get("source", {})
        if isinstance(source, dict) and source.get("table"):
            # Will be registered by CLI
            sql_parts.append(f"FROM {source['table']}")
        else:
            sql_parts.append(f"FROM {self.source_table}")

        if where_conditions:
            sql_parts.append("WHERE " + " AND ".join(where_conditions))

        # Add ordering if specified
        if config.get("order_by"):
            order_cols = ", ".join(config["order_by"])
            sql_parts.append(f"ORDER BY {order_cols}")

        return "\n".join(sql_parts)

    def _compile_merged(self, config: Dict[str, Any]) -> str:
        """Compile a merged dataset with joins."""
        sources = config.get("sources", [])
        join_config = config.get("join", {})
        columns = config.get("columns", [])
        filters = config.get("filters", [])

        # Build SELECT clause
        select_parts = []
        for col in columns:
            name = col.get("name")
            source_alias = col.get("source")
            expression = col.get("expression")
            dtype = col.get("type")
            default = col.get("default")

            # Build column expression
            if expression:
                # Expression can reference source columns via alias
                col_expr = expression
                if source_alias:
                    col_expr = f"{source_alias}.{expression}"
            elif source_alias:
                col_expr = f"{source_alias}.{name}"
            else:
                col_expr = name

            # Apply type casting
            if dtype:
                cast_expr = self._get_cast_expression(col_expr, dtype, default)
                select_parts.append(f"    {cast_expr} AS {name}")
            else:
                select_parts.append(f"    {col_expr} AS {name}")

        # Build FROM clause with joins
        from_parts = []
        first_source = sources[0]
        from_clause = f"FROM {first_source['name']} AS {first_source['alias']}"

        # Add subsequent joins
        join_clauses = []
        for source in sources[1:]:
            join_type = join_config.get("type", "left").upper()

            # Build ON conditions
            on_conditions = []
            for on_clause in join_config.get("conditions", []):
                left_expr = on_clause.get("left")
                right_expr = on_clause.get("right")
                on_conditions.append(f"{left_expr} = {right_expr}")

            on_clause_str = f"ON {' AND '.join(on_conditions)}" if on_conditions else ""
            join_clauses.append(f"{join_type} JOIN {source['name']} AS {source['alias']} {on_clause_str}")

        from_parts.append(from_clause)
        from_parts.extend(join_clauses)

        # Build WHERE clause
        where_conditions = []
        for f in filters:
            condition = self._compile_filter(f)
            if condition:
                where_conditions.append(condition)

        # Assemble SQL
        sql_parts = ["SELECT"]
        sql_parts.append(",\n".join(select_parts))
        sql_parts.extend(from_parts)

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
