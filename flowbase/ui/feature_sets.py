"""Feature Sets UI - Declarative feature engineering with YAML."""

import streamlit as st
import pandas as pd
import yaml
from pathlib import Path
from typing import Optional, Dict, Any

from flowbase.query.engines.duckdb_engine import DuckDBEngine
from flowbase.pipelines.feature_compiler import FeatureCompiler, save_feature_config


st.set_page_config(
    page_title="Flowbase Feature Sets",
    page_icon="⚙️",
    layout="wide",
)


# Example feature config template
EXAMPLE_CONFIG = """name: greyhound_racing_features
description: Feature set for greyhound racing predictions
version: 1.0

# Filter source data for quality
source_filters:
  - column: meeting_date
    operator: ">="
    value: "2018-01-01"
  - column: distance
    operator: ">"
    value: 0

# Data quality rules (remove bad rows)
quality_rules:
  - type: not_null
    column: event_id
  - type: not_null
    column: competitor_name
  - type: range
    column: box
    min: 1
    max: 8
    cast: numeric
  - type: range
    column: starting_price
    min: 1.01
    max: 999
    cast: numeric
  - type: not_null
    column: finish_time

# Include all source columns
include_source_columns: true

# Computed features
features:
  - name: is_favorite
    expression: "CASE WHEN starting_price = MIN(starting_price) OVER (PARTITION BY event_id) THEN 1 ELSE 0 END"
    description: "Binary flag for race favorite"

  - name: odds_rank
    expression: "RANK() OVER (PARTITION BY event_id ORDER BY starting_price ASC)"
    description: "Rank by odds within each race"

  - name: distance_km
    expression: "distance / 1000.0"
    description: "Distance in kilometers"

  - name: speed_rating
    expression: "CAST(distance AS DOUBLE) / CAST(finish_time AS DOUBLE)"
    description: "Speed in meters per second"

  - name: won_race
    expression: "CASE WHEN finish_position = 1 THEN 1 ELSE 0 END"
    description: "Binary flag for winner"

  - name: placed_race
    expression: "CASE WHEN finish_position <= 3 THEN 1 ELSE 0 END"
    description: "Binary flag for placed (top 3)"

# Window features (rolling aggregations, ranks, etc)
window_features:
  - name: box_position_rank
    function: ROW_NUMBER
    partition_by: [event_id]
    order_by: [box]

  - name: avg_speed_by_meeting
    function: AVG
    column: speed_rating
    partition_by: [meeting_name]

# Final filters (after feature computation)
final_filters:
  - column: speed_rating
    operator: ">"
    value: 0
  - column: finish_time
    operator: is_not_null

# Ordering
order_by: [meeting_date DESC, event_id, box]

# Limit for preview (remove for full dataset)
limit: 10000
"""


@st.cache_resource
def get_query_engine() -> DuckDBEngine:
    """Get cached query engine."""
    return DuckDBEngine()


@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a file."""
    engine = get_query_engine()

    path = Path(file_path)
    if path.suffix == ".parquet":
        file_format = "parquet"
    elif path.suffix == ".csv":
        file_format = "csv"
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    engine.register_file("raw_data", file_path, file_format)
    return engine.execute("SELECT * FROM raw_data LIMIT 1000")


def main() -> None:
    """Main Streamlit app."""
    st.title("⚙️ Feature Sets - Declarative Feature Engineering")
    st.markdown("Define features using YAML configs that compile to SQL")

    # Sidebar - Dataset selection
    st.sidebar.header("📁 Source Dataset")

    data_dir = Path("data")
    if data_dir.exists():
        files = list(data_dir.glob("*.parquet")) + list(data_dir.glob("*.csv"))
        file_options = [str(f) for f in files]

        if file_options:
            selected_file = st.sidebar.selectbox("Select dataset", file_options)
        else:
            st.warning("No data files found in ./data/")
            return
    else:
        st.warning("Data directory not found")
        return

    # Load existing feature configs
    feature_config_dir = Path("configs/features")
    feature_config_dir.mkdir(parents=True, exist_ok=True)

    existing_configs = list(feature_config_dir.glob("*.yaml"))
    config_names = ["Create New"] + [f.stem for f in existing_configs]

    selected_config_name = st.sidebar.selectbox("Feature Config", config_names)

    # Main layout - Tabs for YAML editor, SQL preview, and execution
    tab1, tab2, tab3 = st.tabs(["📝 YAML Editor", "⚡ Generated SQL", "▶️ Execute & Preview"])

    # Tab 1: YAML Editor
    with tab1:
        st.subheader("📝 YAML Configuration")

        # Load or create config
        if selected_config_name == "Create New":
            config_yaml = st.text_area(
                "Feature Configuration (YAML)",
                value=EXAMPLE_CONFIG,
                height=600,
                help="Define your feature set using YAML"
            )
        else:
            config_path = feature_config_dir / f"{selected_config_name}.yaml"
            with open(config_path, 'r') as f:
                existing_yaml = f.read()

            config_yaml = st.text_area(
                "Feature Configuration (YAML)",
                value=existing_yaml,
                height=600,
                help="Edit your feature set configuration"
            )

        # Save button
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            config_name = st.text_input(
                "Config Name",
                value=selected_config_name if selected_config_name != "Create New" else "my_features"
            )

        with col_b:
            if st.button("💾 Save Config", type="primary"):
                try:
                    # Validate YAML
                    config_dict = yaml.safe_load(config_yaml)

                    # Save to file
                    save_path = feature_config_dir / f"{config_name}.yaml"
                    save_feature_config(config_dict, str(save_path))

                    st.success(f"✓ Saved to {save_path}")
                except Exception as e:
                    st.error(f"Error saving config: {e}")

        with col_c:
            if st.button("🔄 Reload"):
                st.rerun()

        # Show YAML documentation
        with st.expander("📚 YAML Configuration Guide"):
            st.markdown("""
### Feature Config Structure

```yaml
name: feature_set_name
description: What this feature set does
version: 1.0

# 1. Source Filters (filter raw data)
source_filters:
  - column: date_col
    operator: ">="  # =, >, <, >=, <=, !=, in, not_in, like, between
    value: "2020-01-01"

# 2. Data Quality Rules (remove bad rows)
quality_rules:
  - type: not_null  # or: not_empty, range, in_list, regex
    column: id_column
  - type: range
    column: age
    min: 0
    max: 120

# 3. Computed Features
features:
  - name: feature_name
    expression: "column1 * column2"
    description: "What this feature represents"

# 4. Window Features (rolling stats, ranks)
window_features:
  - name: rolling_avg
    function: AVG  # or: SUM, COUNT, MIN, MAX, ROW_NUMBER, RANK
    column: sales
    partition_by: [customer_id]
    order_by: [date]

# 5. Final Filters (after features computed)
final_filters:
  - column: computed_feature
    operator: ">"
    value: 0

# 6. Ordering
order_by: [date DESC, id]
```

### Supported Quality Rules
- **not_null**: Column must have a value
- **not_empty**: String column must not be empty
- **range**: Numeric column within min/max
- **in_list**: Column value in allowed list
- **regex**: String matches pattern

### Operators for Filters
`=`, `>`, `<`, `>=`, `<=`, `!=`, `in`, `not_in`, `like`, `between`, `is_null`, `is_not_null`
            """)

    # Tab 2: Generated SQL
    with tab2:
        st.subheader("⚡ Generated SQL")

        try:
            # Parse YAML
            config_dict = yaml.safe_load(config_yaml)

            # Compile to SQL
            compiler = FeatureCompiler(source_table="raw_data")
            compiled_sql = compiler.compile(config_dict)

            # Show compiled SQL
            st.code(compiled_sql, language="sql")

            # Copy button (optional - Streamlit doesn't have native copy but we can show the SQL)
            st.info("💡 This SQL is automatically generated from your YAML config")

        except yaml.YAMLError as e:
            st.error(f"YAML parsing error: {e}")
        except Exception as e:
            st.error(f"Compilation error: {e}")

    # Tab 3: Execute & Preview
    with tab3:
        st.subheader("▶️ Execute & Preview")

        try:
            # Parse YAML and compile
            config_dict = yaml.safe_load(config_yaml)
            compiler = FeatureCompiler(source_table="raw_data")
            compiled_sql = compiler.compile(config_dict)

            # Execute button
            if st.button("▶️ Execute Feature Pipeline", type="primary"):
                try:
                    with st.spinner("Executing feature pipeline..."):
                        # Load source data and register
                        engine = get_query_engine()

                        path = Path(selected_file)
                        file_format = "parquet" if path.suffix == ".parquet" else "csv"
                        engine.register_file("raw_data", selected_file, file_format)

                        # Execute compiled SQL
                        result_df = engine.execute(compiled_sql)

                        # Show results
                        st.success(f"✓ Generated {len(result_df):,} rows with {len(result_df.columns)} features")

                        # Metrics
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Rows", f"{len(result_df):,}")
                        col_b.metric("Columns", len(result_df.columns))
                        col_c.metric("Size", f"{result_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

                        # Preview data
                        st.subheader("Data Preview")
                        st.dataframe(result_df.head(100), use_container_width=True)

                        # Column info
                        st.subheader("Column Information")
                        col_info = pd.DataFrame({
                            "Column": result_df.columns,
                            "Type": result_df.dtypes.astype(str),
                            "Non-Null": result_df.count().values,
                            "Null %": ((1 - result_df.count() / len(result_df)) * 100).round(2).values,
                        })
                        st.dataframe(col_info, use_container_width=True)

                        # Feature statistics
                        numeric_cols = result_df.select_dtypes(include=["number"]).columns
                        if len(numeric_cols) > 0:
                            st.subheader("Numeric Feature Statistics")
                            st.dataframe(result_df[numeric_cols].describe(), use_container_width=True)

                        # Export options
                        st.subheader("💾 Export Feature Set")

                        col_a, col_b = st.columns(2)

                        with col_a:
                            export_name = st.text_input(
                                "Feature Set Name",
                                value=config_dict.get("name", "features")
                            )

                        with col_b:
                            if st.button("Save as Parquet"):
                                output_path = f"data/features/{export_name}.parquet"
                                Path("data/features").mkdir(parents=True, exist_ok=True)
                                result_df.to_parquet(output_path)
                                st.success(f"✓ Saved to {output_path}")

                                # Also save the SQL for reference
                                sql_path = f"data/features/{export_name}.sql"
                                Path(sql_path).write_text(compiled_sql)
                                st.info(f"SQL saved to {sql_path}")

                except Exception as e:
                    st.error(f"Execution error: {e}")
                    st.exception(e)

        except yaml.YAMLError as e:
            st.error(f"YAML parsing error: {e}")
        except Exception as e:
            st.error(f"Compilation error: {e}")

    # Source data preview
    st.divider()
    st.subheader("📊 Source Data Preview")

    with st.expander("View source dataset"):
        try:
            df = load_data(selected_file)
            st.dataframe(df, use_container_width=True)

            st.markdown(f"**Available columns:** {', '.join(df.columns.tolist())}")
        except Exception as e:
            st.error(f"Error loading source data: {e}")


if __name__ == "__main__":
    main()
