"""Model training functionality."""

import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pickle
import json

import pandas as pd
from sklearn.model_selection import train_test_split


class ModelTrainer:
    """Handles model training from config."""

    def __init__(self, models_dir: str = "data/models"):
        """Initialize trainer.

        Args:
            models_dir: Directory to save trained models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def load_features(self, feature_path: str) -> pd.DataFrame:
        """Load features from parquet file."""
        return pd.read_parquet(feature_path)

    def prepare_data(
        self,
        df: pd.DataFrame,
        features: List[str],
        target: str,
        split_config: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare train/test splits."""
        X = df[features]
        y = df[target]

        # Handle missing values - fill numeric with median, categorical with mode
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 0)

        method = split_config.get("method", "random")

        if method == "random":
            test_size = split_config.get("test_size", 0.2)
            random_state = split_config.get("random_state", 42)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        elif method == "time":
            time_column = split_config.get("time_column")
            cutoff = split_config.get("cutoff")

            if not time_column or not cutoff:
                raise ValueError("time method requires time_column and cutoff")

            train_mask = df[time_column] < cutoff
            X_train, X_test = X[train_mask], X[~train_mask]
            y_train, y_test = y[train_mask], y[~train_mask]
        else:
            raise ValueError(f"Unknown split method: {method}")

        return X_train, X_test, y_train, y_test

    def create_model(self, model_config: Dict[str, Any]) -> Any:
        """Create model instance from config."""
        model_type = model_config.get("type", "sklearn")

        if model_type == "sklearn":
            class_path = model_config["class"]
            module_path, class_name = class_path.rsplit(".", 1)

            module = importlib.import_module(f"sklearn.{module_path}")
            model_class = getattr(module, class_name)

            hyperparams = model_config.get("hyperparameters", {})
            return model_class(**hyperparams)
        elif model_type == "xgboost":
            import xgboost as xgb
            return xgb.XGBClassifier(**model_config.get("hyperparameters", {}))
        elif model_type == "lightgbm":
            import lightgbm as lgb
            return lgb.LGBMClassifier(**model_config.get("hyperparameters", {}))
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def train(
        self,
        config: Dict[str, Any],
        feature_path: str
    ) -> Dict[str, Any]:
        """Train a model from config.

        Args:
            config: Model configuration
            feature_path: Path to feature parquet file

        Returns:
            Dict with model info and metrics
        """
        # Load features
        df = self.load_features(feature_path)

        # Prepare data
        features = config["features"]
        target = config["target"]
        split_config = config.get("split", {"method": "random", "test_size": 0.2})

        X_train, X_test, y_train, y_test = self.prepare_data(
            df, features, target, split_config
        )

        # Create and train model
        # Config might already be unwrapped by CLI
        model_config = config.get("model", config)
        model = self.create_model(model_config)

        model.fit(X_train, y_train)

        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        # Get predictions for detailed metrics
        y_pred = model.predict(X_test)

        # Calculate additional metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score

        # Determine if classification or regression
        # Check if it's a regressor first (more reliable than checking unique values)
        is_regression = (
            hasattr(model, "_estimator_type") and model._estimator_type == "regressor"
        ) or (
            "Regressor" in model.__class__.__name__ or "Ridge" in model.__class__.__name__
        )

        metrics = {
            "train_score": float(train_score),
            "test_score": float(test_score)
        }

        if is_regression:
            # Regression metrics
            mse = mean_squared_error(y_test, y_pred)
            metrics.update({
                "mse": float(mse),
                "rmse": float(mse ** 0.5),
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "r2": float(r2_score(y_test, y_pred))
            })
        else:
            # Classification metrics
            metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
            # Multi-class metrics with averaging
            avg_method = "weighted" if len(set(y_train)) > 2 else "binary"
            metrics.update({
                "precision": float(precision_score(y_test, y_pred, average=avg_method, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, average=avg_method, zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, average=avg_method, zero_division=0))
            })

        # Save model
        model_name = config["name"]
        model_path = self.models_dir / f"{model_name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Save metadata
        metadata = {
            "name": model_name,
            "model_type": model_config["type"],
            "model_class": model_config.get("class"),
            "features": features,
            "target": target,
            "hyperparameters": model_config.get("hyperparameters", {}),
            "split_config": split_config,
            "metrics": metrics,
            "train_size": len(X_train),
            "test_size": len(X_test)
        }

        metadata_path = self.models_dir / f"{model_name}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return {
            "model_path": str(model_path),
            "metadata_path": str(metadata_path),
            "metrics": metrics,
            "train_size": len(X_train),
            "test_size": len(X_test)
        }
