"""ML-focused UI for feature engineering and model training."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import pickle

from flowbase.query.engines.duckdb_engine import DuckDBEngine
from flowbase.experiments.tracker import ExperimentTracker
from flowbase.experiments.runner import ExperimentRunner
from flowbase.core.config.models import ExperimentConfig, ModelConfig


st.set_page_config(
    page_title="Flowbase ML Studio",
    page_icon="🤖",
    layout="wide",
)


# Initialize session state
if "feature_queries" not in st.session_state:
    st.session_state.feature_queries = []
if "train_test_config" not in st.session_state:
    st.session_state.train_test_config = {"test_size": 0.2, "random_state": 42}
if "models" not in st.session_state:
    st.session_state.models = []


@st.cache_resource
def get_query_engine() -> DuckDBEngine:
    """Get a cached query engine instance."""
    return DuckDBEngine()


@st.cache_resource
def get_tracker() -> ExperimentTracker:
    """Get experiment tracker."""
    return ExperimentTracker()


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
    return engine.execute("SELECT * FROM raw_data")


def apply_split(df: pd.DataFrame, split_config: Dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply train/test split based on configuration."""
    from sklearn.model_selection import train_test_split

    if split_config["method"] == "random":
        train_df, test_df = train_test_split(
            df,
            test_size=split_config["test_size"],
            random_state=split_config["random_state"]
        )

    elif split_config["method"] == "time_percentage":
        time_col = split_config["time_column"]
        df_sorted = df.sort_values(time_col)
        split_idx = int(len(df_sorted) * (1 - split_config["test_size"]))
        train_df = df_sorted.iloc[:split_idx]
        test_df = df_sorted.iloc[split_idx:]

    elif split_config["method"] == "time_cutoff":
        time_col = split_config["time_column"]
        cutoff = pd.to_datetime(split_config["cutoff_date"])

        df_temp = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_temp[time_col]):
            df_temp[time_col] = pd.to_datetime(df_temp[time_col])

        train_df = df_temp[df_temp[time_col] < cutoff]
        test_df = df_temp[df_temp[time_col] >= cutoff]

    elif split_config["method"] == "column":
        split_col = split_config["split_column"]
        test_values = split_config["test_values"]
        train_df = df[~df[split_col].isin(test_values)]
        test_df = df[df[split_col].isin(test_values)]

    else:
        raise ValueError(f"Unknown split method: {split_config['method']}")

    return train_df, test_df


def main() -> None:
    """Main Streamlit app."""
    st.title("🤖 Flowbase ML Studio")
    st.markdown("Build features, train models, and compare results")

    # Sidebar - Dataset selection
    st.sidebar.header("📁 Dataset")

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

    # Load data
    try:
        with st.spinner(f"Loading {selected_file}..."):
            df = load_data(selected_file)
        st.sidebar.success(f"✓ {len(df):,} rows, {len(df.columns)} cols")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔧 Feature Engineering",
        "🎯 Train Models",
        "📊 Experiments",
        "🏆 Compare Results"
    ])

    # Tab 1: Feature Engineering
    with tab1:
        feature_engineering_tab(df)

    # Tab 2: Train Models
    with tab2:
        train_models_tab(df)

    # Tab 3: Experiments
    with tab3:
        experiments_tab()

    # Tab 4: Compare Results
    with tab4:
        results_tab()


def feature_engineering_tab(df: pd.DataFrame) -> None:
    """Feature engineering interface."""
    st.header("🔧 Feature Engineering")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Raw Dataset Preview")
        st.dataframe(df.head(100), height=300)

        st.subheader("Available Columns")
        cols_df = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.astype(str),
            "Nulls": df.isnull().sum().values,
        })
        st.dataframe(cols_df, height=300)

    with col2:
        st.subheader("SQL Feature Engineering")
        st.markdown("Write SQL to create features (table: `raw_data`)")

        feature_name = st.text_input("Feature Set Name", value="feature_set_1")

        # SQL editor with example
        default_sql = """SELECT
    *,
    -- Add your feature engineering here
    column1 * 2 as feature_1,
    column2 + 100 as feature_2
FROM raw_data
LIMIT 1000"""

        feature_sql = st.text_area(
            "SQL Query",
            value=default_sql,
            height=200,
            help="Write SQL to engineer features from raw_data"
        )

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("▶️ Preview Features", type="primary"):
                try:
                    engine = get_query_engine()
                    result = engine.execute(feature_sql)
                    st.success(f"✓ Generated {len(result)} rows, {len(result.columns)} columns")
                    st.dataframe(result.head(50), height=300)

                    # Show statistics
                    st.subheader("Feature Statistics")
                    numeric_cols = result.select_dtypes(include=["number"]).columns
                    if len(numeric_cols) > 0:
                        st.dataframe(result[numeric_cols].describe())

                except Exception as e:
                    st.error(f"SQL Error: {e}")

        with col_b:
            if st.button("💾 Save Feature Set"):
                try:
                    engine = get_query_engine()
                    result = engine.execute(feature_sql)

                    # Save to parquet
                    output_path = f"data/features/{feature_name}.parquet"
                    Path("data/features").mkdir(parents=True, exist_ok=True)
                    result.to_parquet(output_path)

                    st.success(f"✓ Saved to {output_path}")

                    # Save SQL query for reference
                    sql_path = f"data/features/{feature_name}.sql"
                    Path(sql_path).write_text(feature_sql)

                except Exception as e:
                    st.error(f"Error saving: {e}")


def train_models_tab(df: pd.DataFrame) -> None:
    """Model training interface."""
    st.header("🎯 Train Models")

    # Load feature sets
    feature_dir = Path("data/features")
    if feature_dir.exists():
        feature_files = list(feature_dir.glob("*.parquet"))
        if feature_files:
            selected_features = st.selectbox(
                "Select Feature Set",
                ["Use raw data"] + [str(f) for f in feature_files]
            )

            if selected_features != "Use raw data":
                df = pd.read_parquet(selected_features)
                st.success(f"✓ Loaded features: {len(df)} rows, {len(df.columns)} columns")
        else:
            st.info("No feature sets found. Create one in Feature Engineering tab.")

    st.subheader("Dataset Configuration")

    col1, col2 = st.columns(2)

    with col1:
        target_col = st.selectbox("Target Column", df.columns)

    with col2:
        feature_cols = st.multiselect(
            "Feature Columns",
            [col for col in df.columns if col != target_col],
            default=[col for col in df.columns if col != target_col][:5]
        )

    st.subheader("Train/Test Split Strategy")

    split_strategy = st.radio(
        "Split Method",
        ["Random Split", "Time-based Split", "Custom Column Split"],
        help="Choose how to split your data into train and test sets"
    )

    if split_strategy == "Random Split":
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
        with col2:
            random_state = st.number_input("Random State", value=42)

        split_config = {
            "method": "random",
            "test_size": test_size,
            "random_state": random_state
        }

    elif split_strategy == "Time-based Split":
        # Find date/time columns
        datetime_cols = []
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                datetime_cols.append(col)
            elif df[col].dtype == 'object':
                # Try to parse as datetime
                try:
                    pd.to_datetime(df[col].head(100))
                    datetime_cols.append(col)
                except:
                    pass

        if not datetime_cols:
            st.warning("No date/time columns detected. Trying numeric columns that might represent dates...")
            datetime_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

        col1, col2 = st.columns(2)

        with col1:
            time_col = st.selectbox(
                "Time Column",
                datetime_cols if datetime_cols else df.columns,
                help="Column to use for time-based splitting"
            )

        with col2:
            split_method = st.selectbox(
                "Split Method",
                ["Percentage (most recent as test)", "Specific Date Cutoff"],
                help="How to determine the train/test cutoff"
            )

        if split_method == "Percentage (most recent as test)":
            test_size = st.slider("Test Size (most recent %)", 0.1, 0.5, 0.2)
            split_config = {
                "method": "time_percentage",
                "time_column": time_col,
                "test_size": test_size
            }
        else:
            # Show date range
            try:
                df_temp = df.copy()
                if not pd.api.types.is_datetime64_any_dtype(df_temp[time_col]):
                    df_temp[time_col] = pd.to_datetime(df_temp[time_col])

                min_date = df_temp[time_col].min()
                max_date = df_temp[time_col].max()

                st.info(f"Date range: {min_date} to {max_date}")

                cutoff_date = st.date_input(
                    "Train/Test Cutoff Date",
                    value=min_date + (max_date - min_date) * 0.8,
                    min_value=min_date.date() if hasattr(min_date, 'date') else min_date,
                    max_value=max_date.date() if hasattr(max_date, 'date') else max_date,
                    help="Data before this date = train, after = test"
                )

                split_config = {
                    "method": "time_cutoff",
                    "time_column": time_col,
                    "cutoff_date": str(cutoff_date)
                }
            except Exception as e:
                st.error(f"Error parsing dates: {e}")
                split_config = {"method": "random", "test_size": 0.2, "random_state": 42}

    else:  # Custom Column Split
        st.markdown("Use a column to determine train/test split (e.g., fold, split, partition)")

        col1, col2 = st.columns(2)

        with col1:
            split_col = st.selectbox(
                "Split Column",
                df.columns,
                help="Column containing train/test indicators"
            )

        with col2:
            unique_vals = df[split_col].unique()[:20]  # Show first 20 unique values
            test_values = st.multiselect(
                "Test Set Values",
                unique_vals,
                help="Rows with these values will be in test set"
            )

        split_config = {
            "method": "column",
            "split_column": split_col,
            "test_values": test_values
        }

    # Show split preview
    if st.checkbox("Preview Split"):
        try:
            train_df, test_df = apply_split(df, split_config)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Train Set Size", f"{len(train_df):,} rows")
            with col2:
                st.metric("Test Set Size", f"{len(test_df):,} rows")

            st.info(f"Split ratio: {len(train_df)/(len(train_df)+len(test_df))*100:.1f}% train, {len(test_df)/(len(train_df)+len(test_df))*100:.1f}% test")
        except Exception as e:
            st.error(f"Error previewing split: {e}")

    st.subheader("Model Configuration")

    # Model selection
    model_type = st.selectbox(
        "Model Type",
        ["sklearn", "xgboost", "lightgbm"]
    )

    if model_type == "sklearn":
        model_class = st.selectbox(
            "Algorithm",
            [
                "ensemble.RandomForestClassifier",
                "ensemble.RandomForestRegressor",
                "linear_model.LogisticRegression",
                "linear_model.LinearRegression",
                "ensemble.GradientBoostingClassifier",
                "svm.SVC",
            ]
        )
    elif model_type == "xgboost":
        model_class = st.selectbox("Algorithm", ["XGBClassifier", "XGBRegressor"])
    else:  # lightgbm
        model_class = st.selectbox("Algorithm", ["LGBMClassifier", "LGBMRegressor"])

    # Hyperparameters
    st.subheader("Hyperparameters")

    hyperparams = {}
    if "RandomForest" in model_class:
        col1, col2, col3 = st.columns(3)
        with col1:
            hyperparams["n_estimators"] = st.number_input("n_estimators", 10, 500, 100)
        with col2:
            hyperparams["max_depth"] = st.number_input("max_depth", 1, 50, 10)
        with col3:
            hyperparams["random_state"] = random_state

    elif "LogisticRegression" in model_class:
        col1, col2 = st.columns(2)
        with col1:
            hyperparams["max_iter"] = st.number_input("max_iter", 100, 5000, 1000)
        with col2:
            hyperparams["random_state"] = random_state

    elif "XGB" in model_class or "LGBM" in model_class:
        col1, col2, col3 = st.columns(3)
        with col1:
            hyperparams["n_estimators"] = st.number_input("n_estimators", 10, 500, 100)
        with col2:
            hyperparams["learning_rate"] = st.number_input("learning_rate", 0.01, 0.3, 0.1)
        with col3:
            hyperparams["max_depth"] = st.number_input("max_depth", 1, 20, 6)

    # Metrics
    st.subheader("Metrics")
    is_classification = "Classifier" in model_class or "Logistic" in model_class

    if is_classification:
        default_metrics = ["accuracy", "f1", "roc_auc"]
    else:
        default_metrics = ["mse", "r2"]

    metrics = st.multiselect("Evaluation Metrics", default_metrics, default=default_metrics)

    # Train button
    experiment_name = st.text_input("Experiment Name", value=f"exp_{model_class.split('.')[-1]}")

    if st.button("🚀 Train Model", type="primary"):
        if not feature_cols:
            st.error("Please select at least one feature column")
            return

        if target_col not in df.columns:
            st.error("Target column not found in dataset")
            return

        try:
            with st.spinner("Training model..."):
                # Apply custom split
                train_df, test_df = apply_split(df, split_config)

                st.info(f"Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")

                # Prepare train and test sets
                X_train = train_df[feature_cols]
                y_train = train_df[target_col]
                X_test = test_df[feature_cols]
                y_test = test_df[target_col]

                # Instantiate model
                import importlib
                if model_type == "sklearn":
                    module_name = f"sklearn.{model_class.rsplit('.', 1)[0]}"
                    class_name = model_class.rsplit(".", 1)[-1]
                elif model_type == "xgboost":
                    module_name = "xgboost"
                    class_name = model_class
                elif model_type == "lightgbm":
                    module_name = "lightgbm"
                    class_name = model_class

                module = importlib.import_module(module_name)
                model_cls = getattr(module, class_name)
                model = model_cls(**hyperparams)

                # Train
                model.fit(X_train, y_train)

                # Predict
                y_pred = model.predict(X_test)

                # Compute metrics
                from sklearn.metrics import (
                    accuracy_score, f1_score, roc_auc_score,
                    mean_squared_error, r2_score
                )

                results = {}
                for metric in metrics:
                    if metric == "accuracy":
                        results["accuracy"] = accuracy_score(y_test, y_pred)
                    elif metric == "f1":
                        results["f1"] = f1_score(y_test, y_pred, average="weighted")
                    elif metric == "roc_auc" and hasattr(model, "predict_proba"):
                        y_proba = model.predict_proba(X_test)
                        if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
                            results["roc_auc"] = roc_auc_score(y_test, y_proba, multi_class="ovr")
                    elif metric == "mse":
                        results["mse"] = mean_squared_error(y_test, y_pred)
                    elif metric == "r2":
                        results["r2"] = r2_score(y_test, y_pred)

                # Save to experiment tracker
                tracker = get_tracker()
                exp_id = tracker.create_experiment(experiment_name, f"Training {model_class} with {split_config['method']} split")
                run_id = tracker.create_run(
                    exp_id,
                    f"{model_class.split('.')[-1]}",
                    model_type,
                    model_class,
                    hyperparams,
                    feature_cols,
                    target_col
                )

                for metric_name, metric_value in results.items():
                    tracker.log_metric(run_id, metric_name, metric_value)

                # Save model
                model_path = f"data/models/{experiment_name}_{model_class.split('.')[-1]}.pkl"
                Path("data/models").mkdir(parents=True, exist_ok=True)
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)

                tracker.log_artifact(run_id, "model", model_path)
                tracker.complete_run(run_id)

                # Display results
                st.success("✓ Training completed!")

                st.subheader("Results")
                metrics_df = pd.DataFrame([results])
                st.dataframe(metrics_df)

                # Plot metrics
                fig = px.bar(
                    x=list(results.keys()),
                    y=list(results.values()),
                    title="Model Metrics",
                    labels={"x": "Metric", "y": "Value"}
                )
                st.plotly_chart(fig, use_container_width=True)

                st.info(f"Model saved to: {model_path}")

        except Exception as e:
            st.error(f"Training error: {e}")
            st.exception(e)


def experiments_tab() -> None:
    """View experiments and runs."""
    st.header("📊 Experiments")

    tracker = get_tracker()

    # List all experiments
    cursor = tracker.conn.execute("SELECT * FROM experiments ORDER BY created_at DESC")
    experiments = [dict(row) for row in cursor.fetchall()]

    if not experiments:
        st.info("No experiments yet. Train a model to create your first experiment!")
        return

    # Select experiment
    exp_names = [exp["name"] for exp in experiments]
    selected_exp = st.selectbox("Select Experiment", exp_names)

    # Get experiment details
    exp = next(e for e in experiments if e["name"] == selected_exp)

    st.subheader(f"Experiment: {exp['name']}")
    st.markdown(f"**Description:** {exp['description'] or 'N/A'}")
    st.markdown(f"**Created:** {exp['created_at']}")

    # Get runs for this experiment
    runs = tracker.get_experiment_runs(exp["id"])

    if runs:
        st.subheader("Runs")

        runs_data = []
        for run in runs:
            # Parse metrics from concatenated string
            metrics_dict = {}
            if run.get("metrics"):
                for pair in run["metrics"].split(","):
                    if ":" in pair:
                        k, v = pair.split(":")
                        metrics_dict[k] = float(v)

            runs_data.append({
                "Run": run["run_name"],
                "Model": run["model_class"],
                "Status": run["status"],
                "Created": run["created_at"],
                **metrics_dict
            })

        runs_df = pd.DataFrame(runs_data)
        st.dataframe(runs_df, use_container_width=True)


def results_tab() -> None:
    """Compare experiment results."""
    st.header("🏆 Compare Results")

    tracker = get_tracker()

    # Get all runs across experiments
    cursor = tracker.conn.execute("""
        SELECT
            e.name as experiment,
            r.run_name,
            r.model_class,
            r.status,
            m.metric_name,
            m.metric_value,
            r.created_at
        FROM experiments e
        JOIN runs r ON e.id = r.experiment_id
        LEFT JOIN metrics m ON r.id = m.run_id
        WHERE r.status = 'completed'
        ORDER BY r.created_at DESC
    """)

    results = [dict(row) for row in cursor.fetchall()]

    if not results:
        st.info("No completed runs yet. Train some models to see results!")
        return

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Pivot metrics
    if not df.empty:
        pivot_df = df.pivot_table(
            index=["experiment", "run_name", "model_class"],
            columns="metric_name",
            values="metric_value",
            aggfunc="first"
        ).reset_index()

        st.subheader("All Results")
        st.dataframe(pivot_df, use_container_width=True)

        # Visualizations
        st.subheader("Metric Comparison")

        if len(pivot_df.columns) > 3:
            metric_cols = [col for col in pivot_df.columns if col not in ["experiment", "run_name", "model_class"]]

            selected_metric = st.selectbox("Select Metric", metric_cols)

            fig = px.bar(
                pivot_df,
                x="run_name",
                y=selected_metric,
                color="model_class",
                title=f"{selected_metric} by Model",
                labels={"run_name": "Run", selected_metric: selected_metric.upper()}
            )
            st.plotly_chart(fig, use_container_width=True)

            # Best model
            best_idx = pivot_df[selected_metric].idxmax()
            best_model = pivot_df.iloc[best_idx]

            st.success(f"🏆 Best model for {selected_metric}: **{best_model['run_name']}** ({best_model['model_class']}) = {best_model[selected_metric]:.4f}")


if __name__ == "__main__":
    main()
