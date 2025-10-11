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
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, log_loss

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

        confusion_mat = None
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

            # Add log loss if model supports probability predictions
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_test)
                metrics["log_loss"] = float(log_loss(y_test, y_pred_proba))

            # Generate confusion matrix
            confusion_mat = confusion_matrix(y_test, y_pred).tolist()

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

        # Add confusion matrix for classification models
        if confusion_mat is not None:
            # Get class labels if available
            if hasattr(model, "classes_"):
                classes = model.classes_.tolist()
            else:
                classes = sorted(set(y_test.tolist()))

            metadata["confusion_matrix"] = {
                "matrix": confusion_mat,
                "labels": classes
            }

        # Add feature importance if available
        if hasattr(model, "feature_importances_"):
            # Tree-based models (Random Forest, XGBoost, LightGBM, etc.)
            importances = model.feature_importances_.tolist()
            metadata["feature_importance"] = [
                {"feature": feat, "importance": float(imp)}
                for feat, imp in zip(features, importances)
            ]
        elif hasattr(model, "coef_"):
            # Linear models (Logistic Regression, Linear Regression, etc.)
            if len(model.coef_.shape) == 1:
                # Binary classification or regression
                coefs = model.coef_.tolist()
            else:
                # Multi-class classification - use absolute mean across classes
                coefs = [abs(model.coef_[:, i]).mean() for i in range(model.coef_.shape[1])]

            metadata["feature_importance"] = [
                {"feature": feat, "importance": float(abs(coef))}
                for feat, coef in zip(features, coefs)
            ]

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

    def load_model(self, model_name: str) -> Tuple[Any, Dict[str, Any]]:
        """Load a trained model and its metadata.

        Args:
            model_name: Name of the model to load

        Returns:
            Tuple of (model, metadata)
        """
        model_path = self.models_dir / f"{model_name}.pkl"
        metadata_path = self.models_dir / f"{model_name}_metadata.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Load model
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Load metadata
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        return model, metadata

    def predict(self, model_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a prediction using a trained model.

        Args:
            model_name: Name of the model to use
            input_data: Dictionary of feature values

        Returns:
            Dictionary with prediction results
        """
        # Load model and metadata
        model, metadata = self.load_model(model_name)

        # Get expected features
        expected_features = metadata["features"]

        # Validate input
        missing_features = set(expected_features) - set(input_data.keys())
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        # Create DataFrame with correct feature order
        input_df = pd.DataFrame([input_data])[expected_features]

        # Handle missing values same way as training
        for col in input_df.columns:
            if input_df[col].dtype in ['float64', 'int64']:
                if pd.isna(input_df[col].iloc[0]):
                    input_df[col] = 0  # Default to 0 for inference
            else:
                if pd.isna(input_df[col].iloc[0]):
                    input_df[col] = 0

        # Make prediction
        prediction = model.predict(input_df)[0]

        # Get prediction probability if classifier
        result = {
            "prediction": float(prediction) if isinstance(prediction, (int, float)) else str(prediction),
            "model": model_name,
            "features_used": expected_features
        }

        # Add probabilities for classifiers
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            classes = model.classes_ if hasattr(model, "classes_") else list(range(len(probabilities)))
            result["probabilities"] = {
                str(cls): float(prob) for cls, prob in zip(classes, probabilities)
            }

        return result

    def predict_from_query(
        self,
        model_name: str,
        feature_path: str,
        where_clause: str,
        select_columns: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Make predictions by querying a feature table/parquet file.

        Args:
            model_name: Name of the model to use
            feature_path: Path to feature parquet file or table name
            where_clause: SQL WHERE clause to filter rows
            select_columns: Optional comma-separated list of identifier columns to include in output

        Returns:
            List of dictionaries with prediction results for each row
        """
        from flowbase.query.engines.duckdb_engine import DuckDBEngine

        # Load model and metadata
        model, metadata = self.load_model(model_name)
        expected_features = metadata["features"]

        # Build SQL query to fetch rows
        select_clause = ", ".join(expected_features)
        if select_columns:
            select_clause = f"{select_columns}, {select_clause}"

        sql = f"""
        SELECT {select_clause}
        FROM read_parquet('{feature_path}')
        WHERE {where_clause}
        """

        # Execute query
        engine = DuckDBEngine()
        result_df = engine.execute(sql)

        if result_df.empty:
            return []

        # Extract identifier columns if provided
        identifier_cols = []
        if select_columns:
            identifier_cols = [col.strip() for col in select_columns.split(",")]

        # Make predictions for each row
        results = []
        for _, row in result_df.iterrows():
            # Extract feature values
            feature_values = {feat: row[feat] for feat in expected_features}

            # Create DataFrame with correct feature order
            input_df = pd.DataFrame([feature_values])[expected_features]

            # Handle missing values same way as training
            for col in input_df.columns:
                if input_df[col].dtype in ['float64', 'int64']:
                    if pd.isna(input_df[col].iloc[0]):
                        input_df[col] = 0  # Default to 0 for inference
                else:
                    if pd.isna(input_df[col].iloc[0]):
                        input_df[col] = 0

            # Make prediction
            prediction = model.predict(input_df)[0]

            # Build result
            result = {
                "prediction": float(prediction) if isinstance(prediction, (int, float)) else str(prediction),
                "model": model_name
            }

            # Add identifier columns
            if identifier_cols:
                result["identifiers"] = {col: row[col] for col in identifier_cols}

            # Add probabilities for classifiers
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_df)[0]
                classes = model.classes_ if hasattr(model, "classes_") else list(range(len(probabilities)))
                result["probabilities"] = {
                    str(cls): float(prob) for cls, prob in zip(classes, probabilities)
                }

            results.append(result)

        return results
