"""Data exploration UI using Streamlit."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import Optional

from flowbase.query.engines.duckdb_engine import DuckDBEngine


st.set_page_config(
    page_title="Flowbase Data Explorer",
    page_icon="📊",
    layout="wide",
)


@st.cache_resource
def get_query_engine() -> DuckDBEngine:
    """Get a cached query engine instance."""
    return DuckDBEngine()


@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a file."""
    engine = get_query_engine()

    # Determine file format
    path = Path(file_path)
    if path.suffix == ".parquet":
        file_format = "parquet"
    elif path.suffix == ".csv":
        file_format = "csv"
    elif path.suffix == ".json":
        file_format = "json"
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    # Register and query
    engine.register_file("dataset", file_path, file_format)
    return engine.execute("SELECT * FROM dataset")


@st.cache_data
def run_query(sql: str, _engine: DuckDBEngine) -> pd.DataFrame:
    """Run a SQL query."""
    return _engine.execute(sql)


def main() -> None:
    """Main Streamlit app."""
    st.title("📊 Flowbase Data Explorer")
    st.markdown("Explore your datasets and build SQL queries")

    # Sidebar - File selection
    st.sidebar.header("Dataset")

    # List available files
    data_dir = Path("data")
    if data_dir.exists():
        files = list(data_dir.glob("*.parquet")) + list(data_dir.glob("*.csv"))
        file_options = [str(f) for f in files]

        if file_options:
            selected_file = st.sidebar.selectbox("Select a dataset", file_options)
        else:
            st.warning("No data files found in ./data/")
            st.info("Add .parquet or .csv files to the data/ directory")
            return
    else:
        st.warning("Data directory not found")
        return

    # Load data
    try:
        with st.spinner(f"Loading {selected_file}..."):
            df = load_data(selected_file)

        st.sidebar.success(f"✓ Loaded {len(df):,} rows")

        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Overview", "🔍 SQL Query", "📊 Visualize", "📈 Statistics"])

        # Tab 1: Overview
        with tab1:
            st.subheader("Dataset Overview")

            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", f"{len(df):,}")
            col2.metric("Columns", len(df.columns))
            col3.metric("Size", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

            st.subheader("Column Types")
            dtype_df = pd.DataFrame({
                "Column": df.columns,
                "Type": df.dtypes.astype(str),
                "Non-Null": df.count().values,
                "Null %": ((1 - df.count() / len(df)) * 100).round(2).values,
            })
            st.dataframe(dtype_df, use_container_width=True)

            st.subheader("Sample Data")
            n_rows = st.slider("Number of rows to display", 5, 100, 10)
            st.dataframe(df.head(n_rows), use_container_width=True)

        # Tab 2: SQL Query
        with tab2:
            st.subheader("SQL Query Builder")
            st.markdown("Run custom SQL queries on your dataset (available as `dataset`)")

            # Example queries
            examples = {
                "Select all": "SELECT * FROM dataset LIMIT 100",
                "Count rows": "SELECT COUNT(*) as total_rows FROM dataset",
                "Column stats": f"SELECT {df.columns[0]}, COUNT(*) as count FROM dataset GROUP BY {df.columns[0]} ORDER BY count DESC LIMIT 10",
            }

            col1, col2 = st.columns([3, 1])
            with col1:
                selected_example = st.selectbox("Example queries", ["Custom"] + list(examples.keys()))

            if selected_example == "Custom":
                default_query = "SELECT * FROM dataset LIMIT 100"
            else:
                default_query = examples[selected_example]

            sql_query = st.text_area(
                "SQL Query",
                value=default_query,
                height=150,
                help="Use 'dataset' as the table name"
            )

            if st.button("Run Query", type="primary"):
                try:
                    engine = get_query_engine()
                    engine.register_file("dataset", selected_file, Path(selected_file).suffix[1:])

                    with st.spinner("Running query..."):
                        result = run_query(sql_query, engine)

                    st.success(f"✓ Query returned {len(result):,} rows")
                    st.dataframe(result, use_container_width=True)

                    # Download button
                    csv = result.to_csv(index=False)
                    st.download_button(
                        label="Download as CSV",
                        data=csv,
                        file_name="query_result.csv",
                        mime="text/csv"
                    )

                except Exception as e:
                    st.error(f"Query error: {str(e)}")

        # Tab 3: Visualizations
        with tab3:
            st.subheader("Data Visualization")

            # Select columns for visualization
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

            viz_type = st.selectbox(
                "Chart Type",
                ["Histogram", "Bar Chart", "Scatter Plot", "Box Plot", "Line Chart"]
            )

            if viz_type == "Histogram" and numeric_cols:
                col = st.selectbox("Column", numeric_cols)
                bins = st.slider("Number of bins", 10, 100, 30)

                fig = px.histogram(df, x=col, nbins=bins, title=f"Distribution of {col}")
                st.plotly_chart(fig, use_container_width=True)

            elif viz_type == "Bar Chart" and categorical_cols:
                col = st.selectbox("Column", categorical_cols)
                top_n = st.slider("Show top N categories", 5, 50, 10)

                value_counts = df[col].value_counts().head(top_n)
                fig = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f"Top {top_n} values in {col}",
                    labels={"x": col, "y": "Count"}
                )
                st.plotly_chart(fig, use_container_width=True)

            elif viz_type == "Scatter Plot" and len(numeric_cols) >= 2:
                col1, col2 = st.columns(2)
                x_col = col1.selectbox("X-axis", numeric_cols)
                y_col = col2.selectbox("Y-axis", [c for c in numeric_cols if c != x_col])

                color_col = st.selectbox("Color by (optional)", ["None"] + categorical_cols)
                color = None if color_col == "None" else color_col

                sample_size = min(len(df), 10000)
                df_sample = df.sample(n=sample_size) if len(df) > sample_size else df

                fig = px.scatter(
                    df_sample,
                    x=x_col,
                    y=y_col,
                    color=color,
                    title=f"{x_col} vs {y_col}"
                )
                st.plotly_chart(fig, use_container_width=True)

            elif viz_type == "Box Plot" and numeric_cols:
                col = st.selectbox("Column", numeric_cols)

                fig = px.box(df, y=col, title=f"Box Plot of {col}")
                st.plotly_chart(fig, use_container_width=True)

            elif viz_type == "Line Chart" and numeric_cols:
                y_col = st.selectbox("Y-axis", numeric_cols)

                # Sample if too large
                sample_size = min(len(df), 1000)
                df_sample = df.sample(n=sample_size).sort_index() if len(df) > sample_size else df

                fig = px.line(df_sample, y=y_col, title=f"Line Chart of {y_col}")
                st.plotly_chart(fig, use_container_width=True)

        # Tab 4: Statistics
        with tab4:
            st.subheader("Statistical Summary")

            # Numeric stats
            if numeric_cols:
                st.write("**Numeric Columns**")
                st.dataframe(df[numeric_cols].describe(), use_container_width=True)

            # Categorical stats
            if categorical_cols:
                st.write("**Categorical Columns**")
                cat_stats = pd.DataFrame({
                    "Column": categorical_cols,
                    "Unique Values": [df[col].nunique() for col in categorical_cols],
                    "Most Common": [df[col].mode()[0] if len(df[col].mode()) > 0 else None for col in categorical_cols],
                    "Frequency": [df[col].value_counts().iloc[0] if len(df[col]) > 0 else 0 for col in categorical_cols],
                })
                st.dataframe(cat_stats, use_container_width=True)

            # Correlation matrix for numeric columns
            if len(numeric_cols) > 1:
                st.subheader("Correlation Matrix")
                corr = df[numeric_cols].corr()

                fig = px.imshow(
                    corr,
                    text_auto=True,
                    aspect="auto",
                    title="Correlation Heatmap",
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1
                )
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()
