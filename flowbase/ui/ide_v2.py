"""Flowbase IDE - Complete project navigation and management."""

import streamlit as st
import pandas as pd
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List
from streamlit_ace import st_ace

from flowbase.query.engines.duckdb_engine import DuckDBEngine
from flowbase.pipelines.dataset_compiler import DatasetCompiler, save_dataset_config
from flowbase.pipelines.feature_compiler import FeatureCompiler, save_feature_config


# IDE-style configuration
st.set_page_config(
    page_title="Flowbase IDE",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1e1e1e;
    }

    [data-testid="stSidebar"] .stMarkdown {
        color: #cccccc;
    }

    /* Make code blocks look better */
    .stCodeBlock {
        font-size: 13px;
    }

    /* Better button styling */
    .stButton button {
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'selected_item' not in st.session_state:
    st.session_state.selected_item = None
if 'item_type' not in st.session_state:
    st.session_state.item_type = None


@st.cache_resource
def get_query_engine() -> DuckDBEngine:
    """Get cached query engine."""
    return DuckDBEngine()


def scan_project() -> Dict[str, List[Path]]:
    """Scan project directory for all assets."""
    data_dir = Path("data")

    return {
        "sources": list(data_dir.glob("*.parquet")) + list(data_dir.glob("*.csv")) if data_dir.exists() else [],
        "datasets": list(Path("configs/datasets").glob("*.yaml")) if Path("configs/datasets").exists() else [],
        "features": list(Path("configs/features").glob("*.yaml")) if Path("configs/features").exists() else [],
        "models": list(Path("configs/models").glob("*.yaml")) if Path("configs/models").exists() else [],
        "evals": list(Path("configs/evals").glob("*.yaml")) if Path("configs/evals").exists() else [],
    }


def render_sidebar() -> None:
    """Render project navigation sidebar."""
    with st.sidebar:
        st.markdown("### 💻 Flowbase IDE")
        st.markdown("---")

        project = scan_project()

        # SOURCE FILES
        st.markdown("**📁 Source Files**")
        st.caption("Raw data files")
        for source in project["sources"]:
            if st.button(f"📄 {source.name}", key=f"source_{source.stem}", use_container_width=True):
                st.session_state.selected_item = str(source)
                st.session_state.item_type = "source"
                st.rerun()

        if st.button("➕ Add Source", key="add_source", use_container_width=True):
            st.session_state.selected_item = "new_source"
            st.session_state.item_type = "source"
            st.rerun()

        st.markdown("---")

        # DATASETS
        st.markdown("**📊 Datasets**")
        st.caption("Cleaned & typed data")
        for dataset in project["datasets"]:
            if st.button(f"📊 {dataset.stem}", key=f"dataset_{dataset.stem}", use_container_width=True):
                st.session_state.selected_item = str(dataset)
                st.session_state.item_type = "dataset"
                st.rerun()

        if st.button("➕ New Dataset", key="new_dataset", use_container_width=True):
            st.session_state.selected_item = "new_dataset"
            st.session_state.item_type = "dataset"
            st.rerun()

        st.markdown("---")

        # FEATURE SETS
        st.markdown("**⚙️ Feature Sets**")
        st.caption("Engineered features")
        for feature in project["features"]:
            if st.button(f"⚙️ {feature.stem}", key=f"feature_{feature.stem}", use_container_width=True):
                st.session_state.selected_item = str(feature)
                st.session_state.item_type = "feature"
                st.rerun()

        if st.button("➕ New Features", key="new_features", use_container_width=True):
            st.session_state.selected_item = "new_features"
            st.session_state.item_type = "feature"
            st.rerun()

        st.markdown("---")

        # MODELS
        st.markdown("**🤖 Models**")
        st.caption("Model configurations")
        for model in project["models"]:
            if st.button(f"🤖 {model.stem}", key=f"model_{model.stem}", use_container_width=True):
                st.session_state.selected_item = str(model)
                st.session_state.item_type = "model"
                st.rerun()

        if st.button("➕ New Model", key="new_model", use_container_width=True):
            st.session_state.selected_item = "new_model"
            st.session_state.item_type = "model"
            st.rerun()

        st.markdown("---")

        # EVALUATIONS
        st.markdown("**📈 Evaluations**")
        st.caption("Model evaluations")
        for eval_file in project["evals"]:
            if st.button(f"📈 {eval_file.stem}", key=f"eval_{eval_file.stem}", use_container_width=True):
                st.session_state.selected_item = str(eval_file)
                st.session_state.item_type = "eval"
                st.rerun()

        if st.button("➕ New Evaluation", key="new_eval", use_container_width=True):
            st.session_state.selected_item = "new_eval"
            st.session_state.item_type = "eval"
            st.rerun()


def main() -> None:
    """Main IDE interface."""

    render_sidebar()

    # Main content area
    item_type = st.session_state.get('item_type')
    selected_item = st.session_state.get('selected_item')

    if not item_type or not selected_item:
        # Welcome screen
        st.markdown("# 💻 Flowbase IDE")
        st.markdown("### Build ML pipelines declaratively")
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📁 Sources** → Raw data files (Parquet, CSV)")
            st.markdown("**📊 Datasets** → Cleaned, typed, merged data")
            st.markdown("**⚙️ Features** → Engineered features from datasets")
        with col2:
            st.markdown("**🤖 Models** → Model configurations")
            st.markdown("**📈 Evaluations** → Model performance tracking")

        st.info("👈 Select an item from the sidebar to get started")
        return

    # Render appropriate editor based on type
    if item_type == "source":
        render_source_viewer(selected_item)
    elif item_type == "dataset":
        render_dataset_editor(selected_item)
    elif item_type == "feature":
        render_feature_editor(selected_item)
    elif item_type == "model":
        render_model_editor(selected_item)
    elif item_type == "eval":
        render_eval_viewer(selected_item)


def render_source_viewer(source_path: str) -> None:
    """View source data files."""
    st.markdown(f"### 📄 {Path(source_path).name}")

    if source_path == "new_source":
        st.info("To add a new source file, copy your data files (Parquet or CSV) to the `data/` directory")
        return

    try:
        # Load and preview
        path = Path(source_path)
        if path.suffix == ".parquet":
            df = pd.read_parquet(source_path)
        else:
            df = pd.read_csv(source_path)

        # Stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{len(df):,}")
        col2.metric("Columns", len(df.columns))
        col3.metric("Size", f"{path.stat().st_size / 1024**2:.1f} MB")

        st.markdown("**Preview**")
        st.dataframe(df.head(100), use_container_width=True, height=400)

        st.markdown("**Schema**")
        schema_df = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.astype(str),
            "Non-Null": df.count().values,
            "Null %": ((1 - df.count() / len(df)) * 100).round(2).values,
        })
        st.dataframe(schema_df, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading source: {e}")


def render_dataset_editor(config_path: str) -> None:
    """Dataset configuration editor."""

    if config_path == "new_dataset":
        name = "new_dataset"
        yaml_content = get_dataset_template()
    else:
        path = Path(config_path)
        name = path.stem
        with open(path, 'r') as f:
            yaml_content = f.read()

    st.markdown(f"### 📊 Dataset: {name}")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("**Configuration**")

        edited_yaml = st_ace(
            value=yaml_content,
            language="yaml",
            theme="monokai",
            height=600,
            font_size=13,
            tab_size=2,
            show_gutter=True,
            show_print_margin=False,
            wrap=False,
            auto_update=True,
            key=f"dataset_editor_{name}"
        )

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            save_name = st.text_input("Name", value=name, label_visibility="collapsed")

        with col_b:
            if st.button("💾 Save", use_container_width=True):
                try:
                    config_dict = yaml.safe_load(edited_yaml)
                    save_path = Path("configs/datasets") / f"{save_name}.yaml"
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    save_dataset_config(config_dict, str(save_path))
                    st.success(f"Saved!")
                    st.session_state.selected_item = str(save_path)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

        with col_c:
            if st.button("▶️ Compile", type="primary", use_container_width=True):
                st.session_state[f'compile_dataset_{name}'] = True
                st.rerun()

    with col_right:
        st.markdown("**Output**")

        if st.session_state.get(f'compile_dataset_{name}'):
            try:
                config_dict = yaml.safe_load(edited_yaml)
                compiler = DatasetCompiler(source_table="raw_data")
                sql = compiler.compile(config_dict)

                st.code(sql, language="sql")

                if st.button("▶️ Execute", type="primary"):
                    with st.spinner("Executing..."):
                        engine = get_query_engine()

                        # Register source
                        sources = scan_project()["sources"]
                        if sources:
                            source = sources[0]  # Use first source for now
                            file_format = "parquet" if source.suffix == ".parquet" else "csv"
                            engine.register_file("raw_data", str(source), file_format)

                            result_df = engine.execute(sql)

                            st.success(f"✓ {len(result_df):,} rows × {len(result_df.columns)} cols")
                            st.dataframe(result_df.head(100), height=350, use_container_width=True)

                            # Save
                            output_path = f"data/datasets/{save_name}.parquet"
                            Path("data/datasets").mkdir(parents=True, exist_ok=True)
                            result_df.to_parquet(output_path)
                            st.info(f"Saved to {output_path}")
                        else:
                            st.warning("No source files found")

            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)


def render_feature_editor(config_path: str) -> None:
    """Feature set configuration editor."""

    if config_path == "new_features":
        name = "new_features"
        yaml_content = get_feature_template()
    else:
        path = Path(config_path)
        name = path.stem
        with open(path, 'r') as f:
            yaml_content = f.read()

    st.markdown(f"### ⚙️ Feature Set: {name}")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("**Configuration**")

        edited_yaml = st_ace(
            value=yaml_content,
            language="yaml",
            theme="monokai",
            height=600,
            font_size=13,
            tab_size=2,
            show_gutter=True,
            show_print_margin=False,
            wrap=False,
            auto_update=True,
            key=f"feature_editor_{name}"
        )

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            save_name = st.text_input("Name", value=name, label_visibility="collapsed")

        with col_b:
            if st.button("💾 Save", use_container_width=True):
                try:
                    config_dict = yaml.safe_load(edited_yaml)
                    save_path = Path("configs/features") / f"{save_name}.yaml"
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    save_feature_config(config_dict, str(save_path))
                    st.success(f"Saved!")
                    st.session_state.selected_item = str(save_path)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

        with col_c:
            if st.button("▶️ Compile", type="primary", use_container_width=True):
                st.session_state[f'compile_feature_{name}'] = True
                st.rerun()

    with col_right:
        st.markdown("**Output**")

        if st.session_state.get(f'compile_feature_{name}'):
            try:
                config_dict = yaml.safe_load(edited_yaml)
                dataset_ref = config_dict.get("dataset", "raw_data")

                compiler = FeatureCompiler(source_table=dataset_ref)
                sql = compiler.compile(config_dict)

                st.code(sql, language="sql")

                if st.button("▶️ Execute", type="primary"):
                    with st.spinner("Executing..."):
                        engine = get_query_engine()

                        # Register source or dataset
                        if dataset_ref == "raw_data":
                            sources = scan_project()["sources"]
                            if sources:
                                source = sources[0]
                                file_format = "parquet" if source.suffix == ".parquet" else "csv"
                                engine.register_file("raw_data", str(source), file_format)
                        else:
                            # Try to load dataset
                            dataset_path = f"data/datasets/{dataset_ref}.parquet"
                            if Path(dataset_path).exists():
                                engine.register_file(dataset_ref, dataset_path, "parquet")
                            else:
                                st.warning(f"Dataset {dataset_ref} not found. Compile it first.")
                                return

                        result_df = engine.execute(sql)

                        st.success(f"✓ {len(result_df):,} rows × {len(result_df.columns)} cols")
                        st.dataframe(result_df.head(100), height=350, use_container_width=True)

                        # Save
                        output_path = f"data/features/{save_name}.parquet"
                        Path("data/features").mkdir(parents=True, exist_ok=True)
                        result_df.to_parquet(output_path)
                        st.info(f"Saved to {output_path}")

            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)


def render_model_editor(config_path: str) -> None:
    """Model configuration editor."""

    if config_path == "new_model":
        name = "new_model"
        yaml_content = get_model_template()
    else:
        path = Path(config_path)
        name = path.stem
        with open(path, 'r') as f:
            yaml_content = f.read()

    st.markdown(f"### 🤖 Model: {name}")

    st.markdown("**Configuration**")

    edited_yaml = st_ace(
        value=yaml_content,
        language="yaml",
        theme="monokai",
        height=600,
        font_size=13,
        tab_size=2,
        show_gutter=True,
        show_print_margin=False,
        wrap=False,
        auto_update=True,
        key=f"model_editor_{name}"
    )

    col_a, col_b, col_c = st.columns([6, 2, 2])

    with col_a:
        save_name = st.text_input("Name", value=name, label_visibility="collapsed")

    with col_b:
        if st.button("💾 Save", use_container_width=True):
            try:
                config_dict = yaml.safe_load(edited_yaml)
                save_path = Path("configs/models") / f"{save_name}.yaml"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False)
                st.success(f"Saved!")
                st.session_state.selected_item = str(save_path)
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    with col_c:
        if st.button("🚀 Train", type="primary", use_container_width=True):
            st.info("Training will be implemented next!")


def render_eval_viewer(eval_path: str) -> None:
    """View evaluation results."""

    st.markdown(f"### 📈 Evaluation")

    if eval_path == "new_eval":
        st.info("Evaluations are automatically created when you train models")
    else:
        st.info("Evaluation viewer coming soon!")


def get_dataset_template() -> str:
    """Get dataset YAML template."""
    return """name: my_dataset
description: Cleaned dataset with proper types
version: 1.0

columns:
  - name: id
    type: INTEGER

  - name: name
    type: VARCHAR
    transform: trim

  - name: value
    type: DOUBLE
    default: 0.0

  - name: created_at
    type: TIMESTAMP

filters:
  - column: id
    operator: is_not_null

order_by: [created_at DESC]
"""


def get_feature_template() -> str:
    """Get feature YAML template."""
    return """name: my_features
description: Engineered features
version: 1.0
dataset: my_dataset  # Reference to dataset

features:
  - name: feature_1
    expression: "column1 * 2"
    description: "Doubled value"

window_features:
  - name: row_num
    function: ROW_NUMBER
    partition_by: [group_col]
    order_by: [date_col]

limit: 10000
"""


def get_model_template() -> str:
    """Get model YAML template."""
    return """name: my_model
description: Model configuration
version: 1.0

feature_set: my_features
target: target_column

model:
  type: sklearn
  class: ensemble.RandomForestClassifier
  hyperparameters:
    n_estimators: 100
    max_depth: 10
    random_state: 42

split:
  method: time_based
  time_column: date
  test_size: 0.2

metrics:
  - accuracy
  - f1
  - roc_auc
"""


if __name__ == "__main__":
    main()
