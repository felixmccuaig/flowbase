"""Compile declarative feature configs to SQL."""

from typing import Dict, List, Any, Optional
from datetime import datetime
import yaml


class FeatureCompiler:
    """Compiles YAML feature configs into executable SQL."""

    def __init__(self, source_table: str = "raw_data"):
        self.source_table = source_table

    def compile(self, config: Dict[str, Any]) -> str:
        """
        Compile a feature config to SQL.

        Args:
            config: Feature set configuration

        Returns:
            Compiled SQL query
        """
        parts = []

        # Start WITH clauses for filters and quality checks
        with_clauses = []

        # Apply source filters
        if config.get("source_filters"):
            filtered_table = self._compile_filters(config["source_filters"])
            with_clauses.append(f"filtered_data AS (\n{filtered_table}\n)")
            base_table = "filtered_data"
        else:
            base_table = self.source_table

        # Apply data quality rules
        if config.get("quality_rules"):
            quality_table = self._compile_quality_rules(base_table, config["quality_rules"])
            with_clauses.append(f"quality_checked AS (\n{quality_table}\n)")
            base_table = "quality_checked"

        # Build main SELECT with features
        select_parts = []

        # Include base columns if specified
        if config.get("include_source_columns", True):
            select_parts.append("    *")

        # Add computed features
        if config.get("features"):
            for feature in config["features"]:
                feature_sql = self._compile_feature(feature)
                select_parts.append(f"    {feature_sql}")

        # Add window features
        if config.get("window_features"):
            for window_feat in config["window_features"]:
                window_sql = self._compile_window_feature(window_feat)
                select_parts.append(f"    {window_sql}")

        # Add aggregations
        if config.get("aggregations"):
            agg_table = self._compile_aggregations(base_table, config["aggregations"])
            with_clauses.append(f"aggregated AS (\n{agg_table}\n)")
            # Join aggregations back
            base_table = "aggregated"

        # Combine all parts
        sql_parts = []

        if with_clauses:
            sql_parts.append("WITH " + ",\n\n".join(with_clauses))

        sql_parts.append("SELECT")
        sql_parts.append(",\n".join(select_parts))
        sql_parts.append(f"FROM {base_table}")

        # Add final filters
        if config.get("final_filters"):
            conditions = [self._compile_condition(c) for c in config["final_filters"]]
            sql_parts.append("WHERE " + " AND ".join(conditions))

        # Add ordering
        if config.get("order_by"):
            order_cols = ", ".join(config["order_by"])
            sql_parts.append(f"ORDER BY {order_cols}")

        # Add limit
        if config.get("limit"):
            sql_parts.append(f"LIMIT {config['limit']}")

        return "\n".join(sql_parts)

    def _compile_filters(self, filters: List[Dict[str, Any]]) -> str:
        """Compile source filters."""
        conditions = [self._compile_condition(f) for f in filters]
        where_clause = " AND ".join(conditions)

        return f"""    SELECT *
    FROM {self.source_table}
    WHERE {where_clause}"""

    def _compile_quality_rules(self, base_table: str, rules: List[Dict[str, Any]]) -> str:
        """Compile data quality rules."""
        conditions = []

        for rule in rules:
            rule_type = rule.get("type")
            column = rule.get("column")

            if rule_type == "not_null":
                conditions.append(f"{column} IS NOT NULL")

            elif rule_type == "not_empty":
                conditions.append(f"{column} IS NOT NULL AND {column} != ''")

            elif rule_type == "range":
                min_val = rule.get("min")
                max_val = rule.get("max")
                cast_type = rule.get("cast", None)  # Optional: specify type to cast to

                # Try to cast the column if needed for numeric comparison
                col_expr = f"TRY_CAST({column} AS DOUBLE)" if cast_type == "numeric" else column

                if min_val is not None:
                    conditions.append(f"{col_expr} >= {min_val}")
                if max_val is not None:
                    conditions.append(f"{col_expr} <= {max_val}")

            elif rule_type == "in_list":
                values = rule.get("values", [])
                values_str = ", ".join([f"'{v}'" if isinstance(v, str) else str(v) for v in values])
                conditions.append(f"{column} IN ({values_str})")

            elif rule_type == "regex":
                pattern = rule.get("pattern")
                conditions.append(f"regexp_matches({column}, '{pattern}')")

            elif rule_type == "unique":
                # For unique, we'd need window functions or distinct
                pass  # Handle separately

            elif rule_type == "custom":
                conditions.append(rule.get("condition"))

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        return f"""    SELECT *
    FROM {base_table}
    WHERE {where_clause}"""

    def _compile_condition(self, condition: Dict[str, Any]) -> str:
        """Compile a filter condition."""
        column = condition.get("column")
        operator = condition.get("operator", "=")
        value = condition.get("value")

        # Handle different operators
        if operator == "=":
            if isinstance(value, str):
                return f"{column} = '{value}'"
            return f"{column} = {value}"

        elif operator in [">", "<", ">=", "<=", "!="]:
            if isinstance(value, str):
                return f"{column} {operator} '{value}'"
            return f"{column} {operator} {value}"

        elif operator == "in":
            values_str = ", ".join([f"'{v}'" if isinstance(v, str) else str(v) for v in value])
            return f"{column} IN ({values_str})"

        elif operator == "not_in":
            values_str = ", ".join([f"'{v}'" if isinstance(v, str) else str(v) for v in value])
            return f"{column} NOT IN ({values_str})"

        elif operator == "like":
            return f"{column} LIKE '{value}'"

        elif operator == "between":
            return f"{column} BETWEEN {value[0]} AND {value[1]}"

        elif operator == "is_null":
            return f"{column} IS NULL"

        elif operator == "is_not_null":
            return f"{column} IS NOT NULL"

        return "1=1"

    def _compile_feature(self, feature: Dict[str, Any]) -> str:
        """Compile a single feature."""
        name = feature.get("name")
        expression = feature.get("expression")
        description = feature.get("description", "")

        # Handle different feature types
        if "expression" in feature:
            return f"{expression} AS {name}"

        elif "column" in feature:
            # Simple column rename or transformation
            column = feature.get("column")
            transform = feature.get("transform")

            if transform == "upper":
                return f"UPPER({column}) AS {name}"
            elif transform == "lower":
                return f"LOWER({column}) AS {name}"
            elif transform == "trim":
                return f"TRIM({column}) AS {name}"
            elif transform == "abs":
                return f"ABS({column}) AS {name}"
            elif transform == "round":
                decimals = feature.get("decimals", 2)
                return f"ROUND({column}, {decimals}) AS {name}"
            else:
                return f"{column} AS {name}"

        return f"NULL AS {name}"

    def _compile_window_feature(self, window_feat: Dict[str, Any]) -> str:
        """Compile window function features."""
        name = window_feat.get("name")
        function = window_feat.get("function")  # e.g., 'AVG', 'SUM', 'ROW_NUMBER'
        column = window_feat.get("column")
        partition_by = window_feat.get("partition_by", [])
        order_by = window_feat.get("order_by", [])

        window_clause = "OVER ("
        if partition_by:
            window_clause += "PARTITION BY " + ", ".join(partition_by)
        if order_by:
            if partition_by:
                window_clause += " "
            window_clause += "ORDER BY " + ", ".join(order_by)
        window_clause += ")"

        if function.upper() in ["ROW_NUMBER", "RANK", "DENSE_RANK", "PERCENT_RANK", "CUME_DIST", "NTILE"]:
            # These functions don't take a column argument
            return f"{function}() {window_clause} AS {name}"
        elif function.upper() == "COUNT" and not column:
            # COUNT without column means COUNT(*)
            return f"COUNT(*) {window_clause} AS {name}"
        else:
            return f"{function}({column}) {window_clause} AS {name}"

    def _compile_aggregations(self, base_table: str, aggregations: Dict[str, Any]) -> str:
        """Compile aggregation features."""
        group_by = aggregations.get("group_by", [])
        metrics = aggregations.get("metrics", [])

        select_parts = []

        # Add group by columns
        if group_by:
            select_parts.extend(group_by)

        # Add aggregation metrics
        for metric in metrics:
            name = metric.get("name")
            function = metric.get("function")  # e.g., 'AVG', 'SUM', 'COUNT'
            column = metric.get("column")

            if function.upper() == "COUNT":
                if column == "*":
                    select_parts.append(f"COUNT(*) AS {name}")
                else:
                    select_parts.append(f"COUNT({column}) AS {name}")
            else:
                select_parts.append(f"{function}({column}) AS {name}")

        group_clause = f"GROUP BY {', '.join(group_by)}" if group_by else ""

        return f"""    SELECT
        {', '.join(select_parts)}
    FROM {base_table}
    {group_clause}"""


def load_feature_config(yaml_path: str) -> Dict[str, Any]:
    """Load feature config from YAML file."""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def save_feature_config(config: Dict[str, Any], yaml_path: str) -> None:
    """Save feature config to YAML file."""
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
