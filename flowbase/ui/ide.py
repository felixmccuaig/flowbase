"""IDE-style interface for Flowbase - Datasets and Features."""

import streamlit as st
import pandas as pd
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from streamlit_ace import st_ace

from flowbase.query.engines.duckdb_engine import DuckDBEngine
from flowbase.pipelines.dataset_compiler import DatasetCompiler, save_dataset_config
from flowbase.pipelines.feature_compiler import FeatureCompiler, save_feature_config


# Remove default padding for IDE-like feel
st.set_page_config(
    page_title="Flowbase IDE",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for IDE-like appearance
st.markdown("""
<style>
    /* Remove padding */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    /* Make tabs look better */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }

    .stTabs [data-baseweb="tab"] {
        padding-left: 20px;
        padding-right: 20px;
    }

    /* Code blocks */
    .stCodeBlock {
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)


DATASET_EXAMPLE = """name: greyhound_clean
description: Cleaned greyhound racing dataset with proper types
version: 1.0

# Define columns with proper types
columns:
  # IDs
  - name: selection_id
    type: INTEGER

  - name: event_id
    type: INTEGER

  - name: meeting_id
    type: INTEGER

  # Race info
  - name: event_name
    type: VARCHAR

  - name: event_number
    type: INTEGER

  - name: meeting_date
    type: DATE

  - name: meeting_name
    type: VARCHAR

  - name: distance
    type: INTEGER

  # Competitor info
  - name: competitor_name
    type: VARCHAR
    transform: trim

  - name: box
    type: INTEGER

  - name: finish_position
    type: INTEGER

  # Performance metrics
  - name: finish_time
    type: DOUBLE

  - name: starting_price
    type: DOUBLE
    default: 999.0

  - name: rated_price
    type: DOUBLE

  - name: margin
    type: DOUBLE

  - name: weight
    type: DOUBLE

  # Other fields (keep as VARCHAR)
  - name: status
    type: VARCHAR

  - name: trainer_id
    type: INTEGER

  - name: owner
    type: VARCHAR

# Filters for data quality
filters:
  - column: event_id
    operator: is_not_null

  - column: distance
    operator: ">"
    value: 0

  - column: meeting_date
    operator: ">="
    value: "2018-01-01"

# Order by date and event
order_by: [meeting_date DESC, event_id, box]
"""


FEATURE_EXAMPLE = """name: greyhound_features
description: Racing features for model training
version: 1.0
dataset: greyhound_clean  # Reference to dataset

# Simple computed features
features:
  - name: is_favorite
    expression: "starting_price = MIN(starting_price) OVER (PARTITION BY event_id)"
    description: "Is this the race favorite"

  - name: odds_rank
    expression: "RANK() OVER (PARTITION BY event_id ORDER BY starting_price)"
    description: "Rank within race by odds"

  - name: distance_km
    expression: "distance / 1000.0"
    description: "Distance in kilometers"

  - name: speed_mps
    expression: "distance / finish_time"
    description: "Speed in meters per second"

  - name: won_race
    expression: "finish_position = 1"
    description: "Did this competitor win"

  - name: placed
    expression: "finish_position <= 3"
    description: "Finished in top 3"

# Window features
window_features:
  - name: box_rank
    function: ROW_NUMBER
    partition_by: [event_id]
    order_by: [box]

  - name: avg_speed_by_meeting
    function: AVG
    column: speed_mps
    partition_by: [meeting_name]

# Final quality filters
filters:
  - column: finish_time
    operator: ">"
    value: 0

limit: 50000
"""


@st.cache_resource
def get_query_engine() -> DuckDBEngine:
    """Get cached query engine."""
    return DuckDBEngine()


def main() -> None:
    """Main IDE interface."""

    # Top bar
    col1, col2, col3 = st.columns([2, 6, 2])
    with col1:
        st.markdown("### 💻 Flowbase IDE")
    with col3:
        # Source data selector
        data_dir = Path("data")
        if data_dir.exists():
            files = list(data_dir.glob("*.parquet")) + list(data_dir.glob("*.csv"))
            if files:
                selected_file = st.selectbox("Source", [str(f) for f in files], label_visibility="collapsed")
            else:
                st.error("No data files")
                return
        else:
            st.error("No data directory")
            return

    # Main tabs
    tab1, tab2 = st.tabs(["📊 Datasets", "⚙️ Features"])

    # DATASETS TAB
    with tab1:
        dataset_tab(selected_file)

    # FEATURES TAB
    with tab2:
        features_tab(selected_file)


def dataset_tab(source_file: str) -> None:
    """Datasets management tab."""

    # Load existing configs
    config_dir = Path("configs/datasets")
    config_dir.mkdir(parents=True, exist_ok=True)

    existing_configs = list(config_dir.glob("*.yaml"))
    config_names = ["+ New Dataset"] + [f.stem for f in existing_configs]

    col1, col2 = st.columns([8, 4])

    with col1:
        selected_config = st.selectbox("Dataset", config_names, label_visibility="collapsed")

    with col2:
        config_name = st.text_input("Name",
            value=selected_config if selected_config != "+ New Dataset" else "my_dataset",
            label_visibility="collapsed")

    st.divider()

    # Editor and preview side by side
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("**YAML Configuration**")

        # Load config
        if selected_config == "+ New Dataset":
            config_yaml = DATASET_EXAMPLE
        else:
            config_path = config_dir / f"{selected_config}.yaml"
            with open(config_path, 'r') as f:
                config_yaml = f.read()

        # Syntax-highlighted editor
        edited_yaml = st_ace(
            value=config_yaml,
            language="yaml",
            theme="monokai",
            height=600,
            font_size=13,
            tab_size=2,
            show_gutter=True,
            show_print_margin=False,
            wrap=False,
            auto_update=True,
            key=f"dataset_editor_{selected_config}"
        )

        # Action buttons
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("💾 Save", key="save_dataset", use_container_width=True):
                try:
                    config_dict = yaml.safe_load(edited_yaml)
                    save_path = config_dir / f"{config_name}.yaml"
                    save_dataset_config(config_dict, str(save_path))
                    st.success(f"Saved to {save_path}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

        with col_b:
            if st.button("🔄 Compile SQL", key="compile_dataset", use_container_width=True):
                st.session_state['show_dataset_sql'] = True

        with col_c:
            if st.button("▶️ Execute", key="execute_dataset", type="primary", use_container_width=True):
                st.session_state['execute_dataset'] = True

    with col_right:
        st.markdown("**Output**")

        try:
            config_dict = yaml.safe_load(edited_yaml)
            compiler = DatasetCompiler(source_table="raw_data")
            compiled_sql = compiler.compile(config_dict)

            # Show SQL
            if st.session_state.get('show_dataset_sql'):
                st.code(compiled_sql, language="sql")

            # Execute
            if st.session_state.get('execute_dataset'):
                with st.spinner("Executing..."):
                    engine = get_query_engine()

                    # Register source
                    path = Path(source_file)
                    file_format = "parquet" if path.suffix == ".parquet" else "csv"
                    engine.register_file("raw_data", source_file, file_format)

                    # Execute
                    result_df = engine.execute(compiled_sql)

                    st.success(f"✓ {len(result_df):,} rows × {len(result_df.columns)} cols")

                    # Preview
                    st.dataframe(result_df.head(100), height=400, use_container_width=True)

                    # Save option
                    if st.button("💾 Save as Parquet", key="save_dataset_parquet"):
                        output_path = f"data/datasets/{config_name}.parquet"
                        Path("data/datasets").mkdir(parents=True, exist_ok=True)
                        result_df.to_parquet(output_path)
                        st.success(f"Saved to {output_path}")

                st.session_state['execute_dataset'] = False

        except yaml.YAMLError as e:
            st.error(f"YAML Error: {e}")
        except Exception as e:
            st.error(f"Error: {e}")


def features_tab(source_file: str) -> None:
    """Features management tab."""

    # Load existing configs
    config_dir = Path("configs/features")
    config_dir.mkdir(parents=True, exist_ok=True)

    existing_configs = list(config_dir.glob("*.yaml"))
    config_names = ["+ New Feature Set"] + [f.stem for f in existing_configs]

    col1, col2 = st.columns([8, 4])

    with col1:
        selected_config = st.selectbox("Feature Set", config_names, label_visibility="collapsed")

    with col2:
        config_name = st.text_input("Name",
            value=selected_config if selected_config != "+ New Feature Set" else "my_features",
            label_visibility="collapsed")

    st.divider()

    # Editor and preview side by side
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("**YAML Configuration**")

        # Load config
        if selected_config == "+ New Feature Set":
            config_yaml = FEATURE_EXAMPLE
        else:
            config_path = config_dir / f"{selected_config}.yaml"
            with open(config_path, 'r') as f:
                config_yaml = f.read()

        # Syntax-highlighted editor
        edited_yaml = st_ace(
            value=config_yaml,
            language="yaml",
            theme="monokai",
            height=600,
            font_size=13,
            tab_size=2,
            show_gutter=True,
            show_print_margin=False,
            wrap=False,
            auto_update=True,
            key=f"feature_editor_{selected_config}"
        )

        # Action buttons
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("💾 Save", key="save_features", use_container_width=True):
                try:
                    config_dict = yaml.safe_load(edited_yaml)
                    save_path = config_dir / f"{config_name}.yaml"
                    save_feature_config(config_dict, str(save_path))
                    st.success(f"Saved to {save_path}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

        with col_b:
            if st.button("🔄 Compile SQL", key="compile_features", use_container_width=True):
                st.session_state['show_feature_sql'] = True

        with col_c:
            if st.button("▶️ Execute", key="execute_features", type="primary", use_container_width=True):
                st.session_state['execute_features'] = True

    with col_right:
        st.markdown("**Output**")

        try:
            config_dict = yaml.safe_load(edited_yaml)

            # Determine source table
            dataset_ref = config_dict.get("dataset")
            if dataset_ref:
                # Use referenced dataset
                source_table = dataset_ref
            else:
                source_table = "raw_data"

            compiler = FeatureCompiler(source_table=source_table)
            compiled_sql = compiler.compile(config_dict)

            # Show SQL
            if st.session_state.get('show_feature_sql'):
                st.code(compiled_sql, language="sql")

            # Execute
            if st.session_state.get('execute_features'):
                with st.spinner("Executing..."):
                    engine = get_query_engine()

                    # Register source
                    path = Path(source_file)
                    file_format = "parquet" if path.suffix == ".parquet" else "csv"
                    engine.register_file("raw_data", source_file, file_format)

                    # If using dataset, load and register it
                    if dataset_ref:
                        dataset_path = f"data/datasets/{dataset_ref}.parquet"
                        if Path(dataset_path).exists():
                            engine.register_file(dataset_ref, dataset_path, "parquet")
                        else:
                            st.warning(f"Dataset {dataset_ref} not found. Compile it first in Datasets tab.")

                    # Execute
                    result_df = engine.execute(compiled_sql)

                    st.success(f"✓ {len(result_df):,} rows × {len(result_df.columns)} cols")

                    # Preview
                    st.dataframe(result_df.head(100), height=400, use_container_width=True)

                    # Save option
                    if st.button("💾 Save as Parquet", key="save_features_parquet"):
                        output_path = f"data/features/{config_name}.parquet"
                        Path("data/features").mkdir(parents=True, exist_ok=True)
                        result_df.to_parquet(output_path)
                        st.success(f"Saved to {output_path}")

                st.session_state['execute_features'] = False

        except yaml.YAMLError as e:
            st.error(f"YAML Error: {e}")
        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)


if __name__ == "__main__":
    main()
