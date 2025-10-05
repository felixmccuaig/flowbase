"""Experiment tracking using SQLite."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import sqlite3

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Tracks experiments, runs, and metrics in SQLite."""

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize experiment tracker.

        Args:
            db_path: Path to SQLite database. Defaults to ./data/experiments.db
        """
        self.db_path = db_path or "data/experiments.db"
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                run_name TEXT NOT NULL,
                model_type TEXT NOT NULL,
                model_class TEXT NOT NULL,
                hyperparameters TEXT,
                features TEXT,
                target TEXT,
                status TEXT DEFAULT 'running',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            );

            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                step INTEGER DEFAULT 0,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES runs(id)
            );

            CREATE TABLE IF NOT EXISTS artifacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                artifact_type TEXT NOT NULL,
                artifact_path TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES runs(id)
            );

            CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs(experiment_id);
            CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics(run_id);
            CREATE INDEX IF NOT EXISTS idx_artifacts_run ON artifacts(run_id);
        """
        )
        self.conn.commit()

    def create_experiment(self, name: str, description: Optional[str] = None) -> int:
        """Create a new experiment."""
        cursor = self.conn.execute(
            "INSERT OR IGNORE INTO experiments (name, description) VALUES (?, ?)",
            (name, description),
        )
        self.conn.commit()

        # Get experiment ID
        result = self.conn.execute(
            "SELECT id FROM experiments WHERE name = ?", (name,)
        ).fetchone()
        return result["id"]

    def create_run(
        self,
        experiment_id: int,
        run_name: str,
        model_type: str,
        model_class: str,
        hyperparameters: Dict[str, Any],
        features: List[str],
        target: str,
    ) -> int:
        """Create a new run within an experiment."""
        cursor = self.conn.execute(
            """
            INSERT INTO runs (
                experiment_id, run_name, model_type, model_class,
                hyperparameters, features, target
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                run_name,
                model_type,
                model_class,
                json.dumps(hyperparameters),
                json.dumps(features),
                target,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def log_metric(self, run_id: int, metric_name: str, metric_value: float, step: int = 0) -> None:
        """Log a metric for a run."""
        self.conn.execute(
            "INSERT INTO metrics (run_id, metric_name, metric_value, step) VALUES (?, ?, ?, ?)",
            (run_id, metric_name, metric_value, step),
        )
        self.conn.commit()

    def log_artifact(self, run_id: int, artifact_type: str, artifact_path: str) -> None:
        """Log an artifact for a run."""
        self.conn.execute(
            "INSERT INTO artifacts (run_id, artifact_type, artifact_path) VALUES (?, ?, ?)",
            (run_id, artifact_type, artifact_path),
        )
        self.conn.commit()

    def complete_run(self, run_id: int, status: str = "completed") -> None:
        """Mark a run as completed."""
        self.conn.execute(
            "UPDATE runs SET status = ?, completed_at = ? WHERE id = ?",
            (status, datetime.now(), run_id),
        )
        self.conn.commit()

    def get_experiment_runs(self, experiment_id: int) -> List[Dict[str, Any]]:
        """Get all runs for an experiment."""
        cursor = self.conn.execute(
            """
            SELECT r.*,
                   GROUP_CONCAT(m.metric_name || ':' || m.metric_value) as metrics
            FROM runs r
            LEFT JOIN metrics m ON r.id = m.run_id
            WHERE r.experiment_id = ?
            GROUP BY r.id
            ORDER BY r.created_at DESC
            """,
            (experiment_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_best_run(self, experiment_id: int, metric_name: str, maximize: bool = True) -> Optional[Dict[str, Any]]:
        """Get the best run for an experiment based on a metric."""
        order = "DESC" if maximize else "ASC"
        cursor = self.conn.execute(
            f"""
            SELECT r.*, m.metric_value
            FROM runs r
            JOIN metrics m ON r.id = m.run_id
            WHERE r.experiment_id = ? AND m.metric_name = ?
            ORDER BY m.metric_value {order}
            LIMIT 1
            """,
            (experiment_id, metric_name),
        )
        result = cursor.fetchone()
        return dict(result) if result else None

    def close(self) -> None:
        """Close database connection."""
        self.conn.close()
