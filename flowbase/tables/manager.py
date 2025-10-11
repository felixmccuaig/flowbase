"""Table manager for ingestion and compaction operations."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import pandas as pd

from flowbase.tables.metadata import TableMetadata
from flowbase.pipelines.dataset_compiler import DatasetCompiler
from flowbase.query.engines.duckdb_engine import DuckDBEngine


class TableManager:
    """Manages table operations: ingestion, compaction, querying."""

    def __init__(self, metadata_db: str = "data/tables/.metadata.db", data_root: Optional[str] = None):
        """Initialize table manager.

        Args:
            metadata_db: Path to metadata database
            data_root: Optional root directory to prepend to table paths (for Lambda /tmp support)
        """
        self.metadata = TableMetadata(metadata_db)
        self.engine = DuckDBEngine()
        self.data_root = Path(data_root) if data_root else None

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load table configuration from YAML.

        Args:
            config_path: Path to table config file

        Returns:
            Configuration dictionary
        """
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def create_table(self, config_path: str):
        """Register a new table from config.

        Args:
            config_path: Path to table config file
        """
        config = self.load_config(config_path)
        table_name = config["name"]
        base_path = Path(config["storage"]["base_path"])

        # Prepend data_root if specified (for Lambda /tmp support)
        if self.data_root:
            base_path = self.data_root / base_path

        # Create base directory
        base_path.mkdir(parents=True, exist_ok=True)

        # Register in metadata
        self.metadata.register_table(table_name, config_path, base_path)

        return {"table_name": table_name, "base_path": base_path}

    def ingest(
        self,
        table_name: str,
        config_path: str,
        source_file: str,
        date: str,
        dataset_config_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Ingest data for a specific date.

        Args:
            table_name: Name of the table
            config_path: Path to table config
            source_file: Source data file (CSV or Parquet)
            date: Date for this data (YYYY-MM-DD)
            dataset_config_path: Optional dataset cleaning config

        Returns:
            Dictionary with ingestion results
        """
        config = self.load_config(config_path)

        # Build destination path
        base_path = Path(config["storage"]["base_path"])

        # Prepend data_root if specified (for Lambda /tmp support)
        if self.data_root:
            base_path = self.data_root / base_path

        pattern = config["partitioning"]["pattern"]

        # Format date according to date_format if specified
        date_format = config["partitioning"].get("date_format", "%Y-%m-%d")
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        formatted_date = date_obj.strftime(date_format)

        dest_file = base_path / pattern.format(date=formatted_date)
        dest_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Load source data
            if source_file.endswith('.csv'):
                df = pd.read_csv(source_file)
            elif source_file.endswith('.parquet'):
                df = pd.read_parquet(source_file)
            else:
                raise ValueError(f"Unsupported file format: {source_file}")

            # Apply dataset cleaning if config provided
            if dataset_config_path:
                compiler = DatasetCompiler()
                with open(dataset_config_path, 'r') as f:
                    dataset_config = yaml.safe_load(f)

                # Register source data in DuckDB
                self.engine.register_dataframe("raw_data", df)

                # Compile and execute cleaning SQL
                sql = compiler.compile(dataset_config)
                df = self.engine.execute(sql)

            # Write to destination
            output_format = config["storage"].get("format", "parquet")
            if output_format == "parquet":
                df.to_parquet(dest_file, index=False)
            elif output_format == "csv":
                df.to_csv(dest_file, index=False)

            # Log successful ingestion
            self.metadata.log_ingestion(
                table_name=table_name,
                date=date,
                source_file=source_file,
                destination_file=str(dest_file),
                rows_ingested=len(df),
                status="success"
            )

            return {
                "status": "success",
                "table": table_name,
                "date": date,
                "rows": len(df),
                "destination": str(dest_file)
            }

        except Exception as e:
            # Log failed ingestion
            self.metadata.log_ingestion(
                table_name=table_name,
                date=date,
                source_file=source_file,
                destination_file=str(dest_file),
                rows_ingested=0,
                status="failed",
                error_message=str(e)
            )
            raise

    def compact(
        self,
        table_name: str,
        config_path: str,
        period: str,
        delete_source: bool = False
    ) -> Dict[str, Any]:
        """Compact daily files into a monthly file.

        Args:
            table_name: Name of the table
            config_path: Path to table config
            period: Period to compact (YYYY-MM format)
            delete_source: Whether to delete source files after compaction

        Returns:
            Dictionary with compaction results
        """
        config = self.load_config(config_path)

        # Parse period (YYYY-MM)
        year, month = period.split("-")
        period_start = f"{year}-{month}-01"

        # Calculate period end (last day of month)
        if month == "12":
            next_month = f"{int(year)+1}-01-01"
        else:
            next_month = f"{year}-{int(month)+1:02d}-01"
        period_end_dt = datetime.strptime(next_month, "%Y-%m-%d") - timedelta(days=1)
        period_end = period_end_dt.strftime("%Y-%m-%d")

        # Get files to compact
        source_files = self.metadata.get_files_for_compaction(
            table_name, period_start, period_end
        )

        if not source_files:
            return {
                "status": "skipped",
                "reason": f"No files found for period {period}"
            }

        # Read and combine all files
        dfs = []
        for file_path in source_files:
            if file_path.endswith('.parquet'):
                dfs.append(pd.read_parquet(file_path))
            elif file_path.endswith('.csv'):
                dfs.append(pd.read_csv(file_path))

        combined_df = pd.concat(dfs, ignore_index=True)

        # Build output path
        base_path = Path(config["storage"]["base_path"])

        # Prepend data_root if specified (for Lambda /tmp support)
        if self.data_root:
            base_path = self.data_root / base_path

        compaction_config = config.get("compaction", {})
        strategy_config = compaction_config.get("strategy", {})
        output_pattern = strategy_config.get("output_pattern", "{year}-{month}.parquet")

        output_file = base_path / output_pattern.format(year=year, month=month)

        # Write compacted file
        output_format = config["storage"].get("format", "parquet")
        if output_format == "parquet":
            combined_df.to_parquet(output_file, index=False)
        elif output_format == "csv":
            combined_df.to_csv(output_file, index=False)

        # Log compaction
        self.metadata.log_compaction(
            table_name=table_name,
            period=period,
            period_start=period_start,
            period_end=period_end,
            source_files=source_files,
            output_file=str(output_file),
            rows_processed=len(combined_df)
        )

        # Delete source files if requested
        if delete_source:
            for file_path in source_files:
                Path(file_path).unlink()

        return {
            "status": "success",
            "table": table_name,
            "period": period,
            "source_files": len(source_files),
            "rows": len(combined_df),
            "output": str(output_file),
            "deleted_source": delete_source
        }

    def get_status(self, table_name: str) -> Dict[str, Any]:
        """Get status of a table.

        Args:
            table_name: Name of the table

        Returns:
            Status dictionary
        """
        return self.metadata.get_table_status(table_name)

    def query(self, table_name: str, config_path: str, sql: str) -> pd.DataFrame:
        """Query a table using DuckDB (reads all files).

        Args:
            table_name: Name of the table
            config_path: Path to table config
            sql: SQL query (use table name as table reference)

        Returns:
            Query results as DataFrame
        """
        config = self.load_config(config_path)
        base_path = Path(config["storage"]["base_path"])

        # Prepend data_root if specified (for Lambda /tmp support)
        if self.data_root:
            base_path = self.data_root / base_path

        file_format = config["storage"].get("format", "parquet")

        # Register all files as a view
        pattern = f"{base_path}/*.{file_format}"
        register_sql = f"CREATE OR REPLACE VIEW {table_name} AS SELECT * FROM '{pattern}'"
        self.engine.execute(register_sql)

        # Execute user query
        return self.engine.execute(sql)

    def close(self):
        """Close connections."""
        self.metadata.close()
        self.engine.close()
