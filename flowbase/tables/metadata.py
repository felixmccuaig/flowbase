"""Metadata tracking for table operations."""

import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import json


class TableMetadata:
    """Manages metadata for table ingestion and compaction."""

    def __init__(self, db_path: str = "data/tables/.metadata.db"):
        """Initialize metadata database.

        Args:
            db_path: Path to SQLite metadata database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        """Initialize database schema."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS ingestion_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_name TEXT NOT NULL,
                date DATE NOT NULL,
                source_file TEXT,
                destination_file TEXT NOT NULL,
                rows_ingested INTEGER,
                status TEXT NOT NULL,
                error_message TEXT,
                ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(table_name, date)
            );

            CREATE INDEX IF NOT EXISTS idx_ingestion_table_date
                ON ingestion_log(table_name, date);

            CREATE TABLE IF NOT EXISTS compaction_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_name TEXT NOT NULL,
                period TEXT NOT NULL,
                period_start DATE NOT NULL,
                period_end DATE NOT NULL,
                source_files TEXT NOT NULL,
                output_file TEXT NOT NULL,
                rows_processed INTEGER,
                compacted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(table_name, period)
            );

            CREATE INDEX IF NOT EXISTS idx_compaction_table_period
                ON compaction_log(table_name, period);

            CREATE TABLE IF NOT EXISTS table_registry (
                table_name TEXT PRIMARY KEY,
                config_path TEXT NOT NULL,
                base_path TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self.conn.commit()

    def register_table(self, table_name: str, config_path: str, base_path: str):
        """Register a new table in the metadata.

        Args:
            table_name: Name of the table
            config_path: Path to table config file
            base_path: Base path where table data is stored
        """
        self.conn.execute("""
            INSERT OR REPLACE INTO table_registry (table_name, config_path, base_path, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        """, (table_name, config_path, base_path))
        self.conn.commit()

    def log_ingestion(
        self,
        table_name: str,
        date: str,
        destination_file: str,
        rows_ingested: int,
        source_file: Optional[str] = None,
        status: str = "success",
        error_message: Optional[str] = None
    ):
        """Log an ingestion operation.

        Args:
            table_name: Name of the table
            date: Date of data (YYYY-MM-DD)
            destination_file: Path to output file
            rows_ingested: Number of rows ingested
            source_file: Source file path (optional)
            status: success or failed
            error_message: Error message if failed
        """
        self.conn.execute("""
            INSERT OR REPLACE INTO ingestion_log
            (table_name, date, source_file, destination_file, rows_ingested, status, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (table_name, date, source_file, destination_file, rows_ingested, status, error_message))
        self.conn.commit()

    def log_compaction(
        self,
        table_name: str,
        period: str,
        period_start: str,
        period_end: str,
        source_files: List[str],
        output_file: str,
        rows_processed: int
    ):
        """Log a compaction operation.

        Args:
            table_name: Name of the table
            period: Period identifier (e.g., "2024-01")
            period_start: Start date of period
            period_end: End date of period
            source_files: List of source files compacted
            output_file: Output compacted file
            rows_processed: Number of rows processed
        """
        source_files_json = json.dumps(source_files)
        self.conn.execute("""
            INSERT OR REPLACE INTO compaction_log
            (table_name, period, period_start, period_end, source_files, output_file, rows_processed)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (table_name, period, period_start, period_end, source_files_json, output_file, rows_processed))
        self.conn.commit()

    def get_ingested_dates(self, table_name: str) -> List[str]:
        """Get list of dates that have been ingested for a table.

        Args:
            table_name: Name of the table

        Returns:
            List of dates (YYYY-MM-DD) that have been successfully ingested
        """
        cursor = self.conn.execute("""
            SELECT date FROM ingestion_log
            WHERE table_name = ? AND status = 'success'
            ORDER BY date
        """, (table_name,))
        return [row["date"] for row in cursor.fetchall()]

    def get_compacted_periods(self, table_name: str) -> List[str]:
        """Get list of periods that have been compacted for a table.

        Args:
            table_name: Name of the table

        Returns:
            List of period identifiers (e.g., ["2024-01", "2024-02"])
        """
        cursor = self.conn.execute("""
            SELECT period FROM compaction_log
            WHERE table_name = ?
            ORDER BY period
        """, (table_name,))
        return [row["period"] for row in cursor.fetchall()]

    def get_table_status(self, table_name: str) -> Dict[str, Any]:
        """Get status summary for a table.

        Args:
            table_name: Name of the table

        Returns:
            Dictionary with table status information
        """
        # Get ingestion stats
        cursor = self.conn.execute("""
            SELECT
                COUNT(*) as total_ingestions,
                MIN(date) as first_date,
                MAX(date) as last_date,
                SUM(rows_ingested) as total_rows
            FROM ingestion_log
            WHERE table_name = ? AND status = 'success'
        """, (table_name,))
        ingestion_stats = dict(cursor.fetchone())

        # Get compaction stats
        cursor = self.conn.execute("""
            SELECT
                COUNT(*) as total_compactions,
                MIN(period_start) as first_compacted_date,
                MAX(period_end) as last_compacted_date
            FROM compaction_log
            WHERE table_name = ?
        """, (table_name,))
        compaction_stats = dict(cursor.fetchone())

        return {
            "table_name": table_name,
            "ingestion": ingestion_stats,
            "compaction": compaction_stats
        }

    def get_files_for_compaction(
        self,
        table_name: str,
        period_start: str,
        period_end: str
    ) -> List[str]:
        """Get list of ingested files for a date range that haven't been compacted.

        Args:
            table_name: Name of the table
            period_start: Start date (YYYY-MM-DD)
            period_end: End date (YYYY-MM-DD)

        Returns:
            List of file paths
        """
        cursor = self.conn.execute("""
            SELECT destination_file
            FROM ingestion_log
            WHERE table_name = ?
              AND date >= ?
              AND date <= ?
              AND status = 'success'
            ORDER BY date
        """, (table_name, period_start, period_end))
        return [row["destination_file"] for row in cursor.fetchall()]

    def close(self):
        """Close the database connection."""
        self.conn.close()
