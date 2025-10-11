"""Declarative feature engineering compiler for time-series entity data.

Compiles high-level feature specifications into SQL with proper window functions,
partitioning, and temporal ordering for features like win rates, averages, etc.
"""

from typing import Dict, List, Any, Optional, Union
import yaml


class DeclarativeFeatureCompiler:
    """
    Compiles declarative feature specs to SQL with automatic windowing.

    Designed for entity-based time-series data (e.g., greyhound races) where:
    - Each row is an event for an entity (e.g., a greyhound in a race)
    - Features look back at historical data for that entity
    - Time ordering is critical (no data leakage from future)
    """

    def __init__(
        self,
        entity_id_column: str,
        time_column: str,
        source_table: str = "source_data"
    ):
        """
        Initialize compiler.

        Args:
            entity_id_column: Column identifying the entity (e.g., user_id, customer_id)
            time_column: Column for temporal ordering (e.g., timestamp, created_at)
            source_table: Name of the source table/view
        """
        self.entity_id_column = entity_id_column
        self.time_column = time_column
        self.source_table = source_table

    def compile(self, config: Dict[str, Any]) -> str:
        """
        Compile feature config to SQL.

        Args:
            config: Feature configuration dict

        Returns:
            SQL query string
        """
        features = config.get("features", [])

        # Build SELECT statement
        select_parts = ["*"]  # Include all base columns

        for feature_spec in features:
            feature_sql = self._compile_feature(feature_spec)
            if feature_sql:
                select_parts.append(f"    {feature_sql}")

        # Build final SQL
        sql = f"SELECT\n{',\n'.join(select_parts)}\nFROM {self.source_table}"

        return sql

    def _compile_feature(self, spec: Dict[str, Any]) -> Optional[str]:
        """
        Compile a single feature specification.

        Args:
            spec: Feature specification dict

        Returns:
            SQL expression for the feature, or None if invalid
        """
        feature_type = spec.get("type")
        name = spec.get("name")

        if not name:
            raise ValueError("Feature must have a 'name'")

        # Handle different feature types
        if feature_type == "count":
            return self._compile_count_feature(spec)
        elif feature_type == "average":
            return self._compile_aggregate_feature(spec, "AVG")
        elif feature_type == "sum":
            return self._compile_aggregate_feature(spec, "SUM")
        elif feature_type == "max":
            return self._compile_aggregate_feature(spec, "MAX")
        elif feature_type == "min":
            return self._compile_aggregate_feature(spec, "MIN")
        elif feature_type == "stddev":
            return self._compile_aggregate_feature(spec, "STDDEV_SAMP")
        elif feature_type == "time_since_last_event":
            return self._compile_time_since_last_event(spec)
        elif feature_type == "is_first_time":
            return self._compile_is_first_time(spec)
        elif feature_type == "diff_from_last":
            return self._compile_diff_from_last(spec)
        elif feature_type == "contextual_delta_from_min":
            return self._compile_contextual_delta_from_min(spec)
        elif feature_type == "derived":
            return self._compile_derived_feature(spec)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")

    def _compile_count_feature(self, spec: Dict[str, Any]) -> str:
        """Compile COUNT feature (e.g., race_count)."""
        name = spec["name"]
        windows = spec.get("windows", ["all_time"])
        partition_by = spec.get("partition_by", [])
        filter_expr = spec.get("filter")

        # Build window specs
        result_exprs = []
        for window in windows:
            window_name = f"{name}_{window}" if len(windows) > 1 else name
            sql_expr = self._build_window_aggregate(
                "COUNT(*)",
                window,
                partition_by,
                filter_expr
            )
            result_exprs.append(f"{sql_expr} AS {window_name}")

        return ",\n    ".join(result_exprs)

    def _compile_aggregate_feature(
        self,
        spec: Dict[str, Any],
        agg_func: str
    ) -> str:
        """Compile aggregate feature (AVG, SUM, MAX, MIN, STDDEV)."""
        name = spec["name"]
        expression = spec.get("expression")

        if not expression:
            raise ValueError(f"Feature {name} requires 'expression'")

        windows = spec.get("windows", ["all_time"])
        partition_by = spec.get("partition_by", [])
        filter_expr = spec.get("filter")

        # Build window specs
        result_exprs = []
        for window in windows:
            window_name = f"{name}_{window}" if len(windows) > 1 else name
            sql_expr = self._build_window_aggregate(
                f"{agg_func}({expression})",
                window,
                partition_by,
                filter_expr
            )
            result_exprs.append(f"{sql_expr} AS {window_name}")

        return ",\n    ".join(result_exprs)

    def _compile_time_since_last_event(self, spec: Dict[str, Any]) -> str:
        """
        Compile time_since_last_event feature.

        Calculates days since the entity's previous event.
        """
        name = spec["name"]
        partition_by = spec.get("partition_by", [])

        # Get partition columns
        partition_cols = self._get_partition_columns(partition_by)
        partition_clause = f"PARTITION BY {', '.join(partition_cols)}" if partition_cols else ""

        # Calculate days since last event
        sql = f"""EXTRACT(EPOCH FROM (
            {self.time_column} - LAG({self.time_column}, 1) OVER (
                {partition_clause}
                ORDER BY {self.time_column}
            )
        )) / 86400.0"""

        return f"{sql} AS {name}"

    def _compile_is_first_time(self, spec: Dict[str, Any]) -> str:
        """
        Compile is_first_time feature.

        Checks if this is the first time the entity has encountered
        a particular partition (e.g., first race at track, first race at distance).
        """
        name = spec["name"]
        partition_by = spec.get("partition_by", [])

        if not partition_by:
            raise ValueError(f"Feature {name} requires 'partition_by'")

        # Get partition columns including entity
        partition_cols = self._get_partition_columns(partition_by)
        partition_clause = f"PARTITION BY {', '.join(partition_cols)}"

        # Check if this is row #1 for this partition
        sql = f"""CASE WHEN ROW_NUMBER() OVER (
            {partition_clause}
            ORDER BY {self.time_column}
        ) = 1 THEN 1 ELSE 0 END"""

        return f"{sql} AS {name}"

    def _compile_diff_from_last(self, spec: Dict[str, Any]) -> str:
        """
        Compile diff_from_last feature.

        Calculates difference from the previous value (e.g., weight change).
        """
        name = spec["name"]
        value_column = spec.get("value_column")
        partition_by = spec.get("partition_by", [])

        if not value_column:
            raise ValueError(f"Feature {name} requires 'value_column'")

        # Get partition columns
        partition_cols = self._get_partition_columns(partition_by)
        partition_clause = f"PARTITION BY {', '.join(partition_cols)}" if partition_cols else ""

        # Calculate difference from previous value
        sql = f"""{value_column} - LAG({value_column}, 1) OVER (
            {partition_clause}
            ORDER BY {self.time_column}
        )"""

        return f"{sql} AS {name}"

    def _compile_contextual_delta_from_min(self, spec: Dict[str, Any]) -> str:
        """
        Compile contextual_delta_from_min feature.

        For the entity's PREVIOUS race, calculates how much slower it was
        than the best time (BON - Best Of Night) for that context.
        """
        name = spec["name"]
        value_expression = spec.get("value_expression")
        partition_by = spec.get("partition_by", [])
        filter_expr = spec.get("filter")
        lagged = spec.get("lagged", False)

        if not value_expression:
            raise ValueError(f"Feature {name} requires 'value_expression'")

        if not partition_by:
            raise ValueError(f"Feature {name} requires 'partition_by' for context")

        # Build the contextual partition (e.g., same track, distance, date)
        # WITHOUT the entity_id, so we get the min across all competitors
        context_partition = ", ".join(partition_by)

        # Get the minimum (best) value within context
        min_expr = f"MIN({value_expression}) OVER (PARTITION BY {context_partition})"

        # Delta from min
        delta_expr = f"({value_expression}) - ({min_expr})"

        # Apply filter if specified
        if filter_expr:
            delta_expr = f"CASE WHEN ({filter_expr}) THEN {delta_expr} ELSE NULL END"

        # If lagged=true, get this value from the previous race
        if lagged:
            # Get partition columns for entity (to find their previous race)
            entity_partition = f"PARTITION BY {self.entity_id_column}"
            delta_expr = f"LAG({delta_expr}, 1) OVER ({entity_partition} ORDER BY {self.time_column})"

        return f"{delta_expr} AS {name}"

    def _compile_derived_feature(self, spec: Dict[str, Any]) -> str:
        """
        Compile derived feature (raw SQL expression).

        For features that don't fit the other types.
        """
        name = spec["name"]
        expression = spec.get("expression")

        if not expression:
            raise ValueError(f"Feature {name} requires 'expression'")

        return f"({expression}) AS {name}"

    def _build_window_aggregate(
        self,
        agg_expression: str,
        window: str,
        partition_by: List[str],
        filter_expr: Optional[str] = None
    ) -> str:
        """
        Build window aggregate SQL with proper partitioning and framing.

        Args:
            agg_expression: Aggregate function SQL (e.g., "AVG(speed)")
            window: Window type ("all_time", "last_5", "last_3", etc.)
            partition_by: Additional partition columns beyond entity_id
            filter_expr: Optional SQL filter expression

        Returns:
            Complete window function SQL
        """
        # Get partition columns (entity + additional)
        partition_cols = self._get_partition_columns(partition_by)
        partition_clause = f"PARTITION BY {', '.join(partition_cols)}"

        # Build window frame
        if window == "all_time":
            # All prior data (excluding current row to prevent leakage)
            frame = "ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING"
        elif window.startswith("last_"):
            # Last N rows (excluding current)
            n = int(window.split("_")[1])
            frame = f"ROWS BETWEEN {n} PRECEDING AND 1 PRECEDING"
        else:
            raise ValueError(f"Unknown window type: {window}")

        # Apply filter if specified
        if filter_expr:
            # Wrap the inner expression in CASE WHEN for filtering
            # Extract the function name and the argument
            import re
            match = re.match(r'(\w+)\((.*)\)$', agg_expression)
            if match:
                func_name = match.group(1)
                inner_expr = match.group(2)
                # Wrap inner expression with CASE WHEN filter
                agg_expression = f"{func_name}(CASE WHEN ({filter_expr}) THEN {inner_expr} ELSE NULL END)"

        # Build complete window function
        sql = f"{agg_expression} OVER ({partition_clause} ORDER BY {self.time_column} {frame})"

        return sql

    def _get_partition_columns(self, additional_partitions: List[str]) -> List[str]:
        """
        Get partition column list: entity_id + any additional columns.

        Args:
            additional_partitions: Additional columns to partition by

        Returns:
            List of partition columns
        """
        if not additional_partitions:
            return [self.entity_id_column]

        return [self.entity_id_column] + additional_partitions


def load_and_compile_features(
    config_path: str,
    source_table: str = "source_data"
) -> str:
    """
    Load feature config from YAML and compile to SQL.

    Reads entity_id_column and time_column from the config file.
    If not specified in config, defaults to 'entity_id' and 'timestamp'.

    Args:
        config_path: Path to feature config YAML
        source_table: Source table name

    Returns:
        Compiled SQL query
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract entity and time columns from config, with defaults
    entity_id_column = config.get('entity_id_column', 'entity_id')
    time_column = config.get('time_column', 'timestamp')

    compiler = DeclarativeFeatureCompiler(
        entity_id_column=entity_id_column,
        time_column=time_column,
        source_table=source_table
    )

    return compiler.compile(config)


__all__ = ["DeclarativeFeatureCompiler", "load_and_compile_features"]
