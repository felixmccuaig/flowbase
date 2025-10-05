"""Experiment runner for training multiple models."""

import importlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    r2_score,
)

from flowbase.core.config.models import ExperimentConfig, ModelConfig
from flowbase.experiments.tracker import ExperimentTracker
from flowbase.query.base import QueryEngine
from flowbase.query.engines.duckdb_engine import DuckDBEngine
from flowbase.storage.base import StorageBackend
from flowbase.storage.local.filesystem import LocalFileSystem

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Runs experiments and trains multiple models."""

    def __init__(
        self,
        storage: Optional[StorageBackend] = None,
        query_engine: Optional[QueryEngine] = None,
        tracker: Optional[ExperimentTracker] = None,
    ):
        """
        Initialize experiment runner.

        Args:
            storage: Storage backend
            query_engine: Query engine for loading datasets
            tracker: Experiment tracker
        """
        self.storage = storage or LocalFileSystem()
        self.query_engine = query_engine or DuckDBEngine()
        self.tracker = tracker or ExperimentTracker()

    def run(self, config: ExperimentConfig) -> Dict[str, Any]:
        """
        Run a complete experiment with multiple models.

        Args:
            config: Experiment configuration

        Returns:
            Summary of experiment results
        """
        logger.info(f"Starting experiment: {config.name}")

        # Create experiment in tracker
        experiment_id = self.tracker.create_experiment(config.name, config.description)

        # Load dataset
        logger.info(f"Loading dataset: {config.dataset}")
        df = self._load_dataset(config.dataset)

        # Train each model
        results = []
        for model_config in config.models:
            logger.info(f"Training model: {model_config.name}")
            result = self._train_model(experiment_id, model_config, df, config)
            results.append(result)

        # Get best model
        best_run = self.tracker.get_best_run(
            experiment_id, metric_name=config.metrics[0], maximize=True
        )

        summary = {
            "experiment_id": experiment_id,
            "experiment_name": config.name,
            "total_runs": len(results),
            "best_run": best_run,
            "results": results,
        }

        logger.info(f"Experiment completed. Best model: {best_run['run_name'] if best_run else 'N/A'}")
        return summary

    def _load_dataset(self, dataset: str) -> pd.DataFrame:
        """Load a dataset from a table or file."""
        # Check if it's a registered table
        if hasattr(self.query_engine, "list_tables"):
            tables = self.query_engine.list_tables()
            if dataset in tables:
                return self.query_engine.execute(f"SELECT * FROM {dataset}")

        # Otherwise treat as a file path
        if Path(dataset).exists():
            self.query_engine.register_file("temp_dataset", dataset)
            return self.query_engine.execute("SELECT * FROM temp_dataset")

        raise ValueError(f"Dataset not found: {dataset}")

    def _train_model(
        self,
        experiment_id: int,
        model_config: ModelConfig,
        df: pd.DataFrame,
        experiment_config: ExperimentConfig,
    ) -> Dict[str, Any]:
        """Train a single model."""
        # Create run
        run_id = self.tracker.create_run(
            experiment_id=experiment_id,
            run_name=model_config.name,
            model_type=model_config.type,
            model_class=model_config.class_name,
            hyperparameters=model_config.hyperparameters,
            features=model_config.features,
            target=model_config.target,
        )

        try:
            # Prepare data
            X = df[model_config.features]
            y = df[model_config.target]

            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=experiment_config.test_size, random_state=experiment_config.random_state
            )

            # Load model class
            model = self._instantiate_model(model_config)

            # Train model
            logger.info(f"Training {model_config.class_name}...")
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)
            metrics = self._compute_metrics(y_test, y_pred, model, X_test, experiment_config.metrics)

            # Log metrics
            for metric_name, metric_value in metrics.items():
                self.tracker.log_metric(run_id, metric_name, metric_value)

            # Save model
            model_path = f"models/{experiment_config.name}/{model_config.name}.pkl"
            self._save_model(model, model_path)
            self.tracker.log_artifact(run_id, "model", model_path)

            # Mark run as completed
            self.tracker.complete_run(run_id, "completed")

            return {
                "run_id": run_id,
                "model_name": model_config.name,
                "status": "completed",
                "metrics": metrics,
            }

        except Exception as e:
            logger.error(f"Error training {model_config.name}: {e}")
            self.tracker.complete_run(run_id, "failed")
            return {
                "run_id": run_id,
                "model_name": model_config.name,
                "status": "failed",
                "error": str(e),
            }

    def _instantiate_model(self, config: ModelConfig) -> Any:
        """Instantiate a model from configuration."""
        # Parse module and class name
        if config.type == "sklearn":
            module_name = f"sklearn.{config.class_name.rsplit('.', 1)[0]}"
            class_name = config.class_name.rsplit(".", 1)[-1]
        elif config.type == "xgboost":
            module_name = "xgboost"
            class_name = config.class_name
        elif config.type == "lightgbm":
            module_name = "lightgbm"
            class_name = config.class_name
        else:
            raise ValueError(f"Unsupported model type: {config.type}")

        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)
        return model_class(**config.hyperparameters)

    def _compute_metrics(
        self, y_true: pd.Series, y_pred: pd.Series, model: Any, X_test: pd.DataFrame, metric_names: List[str]
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        metrics = {}

        for metric_name in metric_names:
            if metric_name == "accuracy":
                metrics["accuracy"] = accuracy_score(y_true, y_pred)
            elif metric_name == "f1":
                metrics["f1"] = f1_score(y_true, y_pred, average="weighted")
            elif metric_name == "roc_auc":
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_test)
                    metrics["roc_auc"] = roc_auc_score(y_true, y_proba, multi_class="ovr")
            elif metric_name == "mse":
                metrics["mse"] = mean_squared_error(y_true, y_pred)
            elif metric_name == "r2":
                metrics["r2"] = r2_score(y_true, y_pred)

        return metrics

    def _save_model(self, model: Any, path: str) -> None:
        """Save a trained model."""
        import pickle

        model_bytes = pickle.dumps(model)
        self.storage.write(path, model_bytes)
        logger.info(f"Model saved to: {path}")

    def close(self) -> None:
        """Clean up resources."""
        self.query_engine.close()
        self.tracker.close()
