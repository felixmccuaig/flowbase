"""Model training functionality."""

import importlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pickle
import json

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles model training from config."""

    def __init__(
        self,
        models_dir: str = "data/models",
        s3_bucket: Optional[str] = None,
        s3_prefix: Optional[str] = "",
        data_root: Optional[str] = None,
        query_engine_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize trainer.

        Args:
            models_dir: Directory to save trained models (relative or absolute)
            s3_bucket: Optional S3 bucket for syncing artifacts
            s3_prefix: Optional S3 prefix (e.g., "flowbase-greyhounds/")
            data_root: Optional root directory for Lambda /tmp support
            query_engine_config: Optional DuckDB configuration dict
        """
        # If data_root is provided and models_dir is relative, prepend data_root
        models_path = Path(models_dir)
        if data_root and not models_path.is_absolute():
            self.models_dir = Path(data_root) / models_path
        else:
            self.models_dir = models_path

        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.s3_sync = None
        self.query_engine_config = query_engine_config

        if s3_bucket:
            try:
                from flowbase.storage.s3_sync import S3Sync
                self.s3_sync = S3Sync(bucket=s3_bucket, prefix=s3_prefix)
                logger.info(f"S3 sync enabled: s3://{s3_bucket}/{s3_prefix}")
            except ImportError:
                logger.warning("boto3 not installed. S3 sync disabled.")
                self.s3_sync = None

    def load_features(self, feature_path: str) -> pd.DataFrame:
        """Load features from parquet file."""
        return pd.read_parquet(feature_path)

    def prepare_data(
        self,
        df: pd.DataFrame,
        features: List[str],
        target: str,
        split_config: Dict[str, Any],
        model_type: str = "sklearn"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare train/test splits."""
        # Drop rows where target is NaN (labels can't have NaN even for xgboost)
        df = df[df[target].notna()].copy()

        X = df[features].copy()
        y = df[target]

        # Handle missing values - xgboost handles NaN natively, others need imputation
        if model_type != "xgboost":
            for col in X.columns:
                if X[col].dtype in ['float64', 'int64']:
                    X[col] = X[col].fillna(X[col].median())
                else:
                    X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 0)
        else:
            # For XGBoost, convert object columns to categorical for enable_categorical support
            for col in X.columns:
                if X[col].dtype == 'object':
                    X[col] = X[col].astype('category')

        method = split_config.get("method", "random")

        if method == "random":
            test_size = split_config.get("test_size", 0.2)
            random_state = split_config.get("random_state", 42)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        elif method == "time":
            time_column = split_config.get("time_column")

            if not time_column:
                raise ValueError("time method requires time_column")

            # Sort by time column first (oldest to newest)
            df_sorted = df.sort_values(by=time_column)
            X_sorted = df_sorted[features]
            y_sorted = df_sorted[target]

            # Handle missing values in sorted data - xgboost handles NaN natively
            if model_type != "xgboost":
                for col in X_sorted.columns:
                    if X_sorted[col].dtype in ['float64', 'int64']:
                        X_sorted[col] = X_sorted[col].fillna(X_sorted[col].median())
                    else:
                        X_sorted[col] = X_sorted[col].fillna(X_sorted[col].mode()[0] if not X_sorted[col].mode().empty else 0)

            # Use cutoff if provided, otherwise use test_size
            cutoff = split_config.get("cutoff")
            if cutoff:
                train_mask = df_sorted[time_column] < cutoff
                X_train, X_test = X_sorted[train_mask], X_sorted[~train_mask]
                y_train, y_test = y_sorted[train_mask], y_sorted[~train_mask]
            else:
                # Split by test_size (last N% as test set)
                test_size = split_config.get("test_size", 0.2)
                split_idx = int(len(df_sorted) * (1 - test_size))
                X_train, X_test = X_sorted.iloc[:split_idx], X_sorted.iloc[split_idx:]
                y_train, y_test = y_sorted.iloc[:split_idx], y_sorted.iloc[split_idx:]
        else:
            raise ValueError(f"Unknown split method: {method}")

        return X_train, X_test, y_train, y_test

    def prepare_group_data(
        self,
        df: pd.DataFrame,
        features: List[str],
        target: str,
        split_config: Dict[str, Any],
        group_column: str,
        model_type: str = "xgboost"
    ) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[int], List[int], pd.DataFrame, pd.DataFrame
    ]:
        """Prepare group-aware train/test splits."""
        if group_column not in df.columns:
            raise ValueError(f"group_column not found in data: {group_column}")

        method = split_config.get("method", "random")
        random_state = split_config.get("random_state", 42)

        if method == "time":
            time_column = split_config.get("time_column")
            if not time_column:
                raise ValueError("time method requires time_column")

            group_times = (
                df.groupby(group_column)[time_column]
                .min()
                .sort_values()
            )
            cutoff = split_config.get("cutoff")
            if cutoff:
                train_groups = group_times[group_times < cutoff].index
                test_groups = group_times[group_times >= cutoff].index
            else:
                test_size = split_config.get("test_size", 0.2)
                split_idx = int(len(group_times) * (1 - test_size))
                train_groups = group_times.index[:split_idx]
                test_groups = group_times.index[split_idx:]
        elif method == "random":
            test_size = split_config.get("test_size", 0.2)
            group_ids = df[group_column].dropna().unique()
            rng = np.random.RandomState(random_state)
            rng.shuffle(group_ids)
            split_idx = int(len(group_ids) * (1 - test_size))
            train_groups = group_ids[:split_idx]
            test_groups = group_ids[split_idx:]
        else:
            raise ValueError(f"Unknown split method: {method}")

        train_df = df[df[group_column].isin(train_groups)].copy()
        test_df = df[df[group_column].isin(test_groups)].copy()

        # Sort to keep group ordering stable for ranker training.
        train_df = train_df.sort_values(by=[group_column])
        test_df = test_df.sort_values(by=[group_column])

        X_train = train_df[features].copy()
        y_train = train_df[target]
        X_test = test_df[features].copy()
        y_test = test_df[target]

        # Handle missing values - xgboost handles NaN natively, others need imputation
        if model_type != "xgboost":
            for col in X_train.columns:
                if X_train[col].dtype in ['float64', 'int64']:
                    X_train[col] = X_train[col].fillna(X_train[col].median())
                    X_test[col] = X_test[col].fillna(X_test[col].median())
                else:
                    train_mode = X_train[col].mode()
                    fill_value = train_mode[0] if not train_mode.empty else 0
                    X_train[col] = X_train[col].fillna(fill_value)
                    X_test[col] = X_test[col].fillna(fill_value)

        group_train = train_df.groupby(group_column).size().tolist()
        group_test = test_df.groupby(group_column).size().tolist()

        return X_train, X_test, y_train, y_test, group_train, group_test, train_df, test_df

    def create_model(self, model_config: Dict[str, Any]) -> Any:
        """Create model instance from config."""
        model_type = model_config.get("type", "sklearn")
        group_column = model_config.get("group_column")

        if model_type == "sklearn":
            class_path = model_config["class"]
            module_path, class_name = class_path.rsplit(".", 1)

            module = importlib.import_module(f"sklearn.{module_path}")
            model_class = getattr(module, class_name)

            hyperparams = model_config.get("hyperparameters", {})
            return model_class(**hyperparams)
        elif model_type == "xgboost":
            import xgboost as xgb
            # Check for explicit class specification
            model_class_name = model_config.get("class", "XGBClassifier")
            if group_column:
                return xgb.XGBRanker(**model_config.get("hyperparameters", {}))
            elif model_class_name == "XGBRegressor":
                return xgb.XGBRegressor(**model_config.get("hyperparameters", {}))
            else:
                return xgb.XGBClassifier(**model_config.get("hyperparameters", {}))
        elif model_type == "lightgbm":
            import lightgbm as lgb
            if group_column:
                return lgb.LGBMRanker(**model_config.get("hyperparameters", {}))
            return lgb.LGBMClassifier(**model_config.get("hyperparameters", {}))
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _compute_group_metrics(
        self,
        df: pd.DataFrame,
        scores: np.ndarray,
        group_column: str,
        target: str
    ) -> Dict[str, float]:
        """Compute race-level metrics using softmax within each group."""
        work = df[[group_column, target]].copy()
        work["_score"] = scores

        # Softmax per group with numerical stability
        group_max = work.groupby(group_column)["_score"].transform("max")
        exp_scores = np.exp(work["_score"] - group_max)
        group_sum = exp_scores.groupby(work[group_column]).transform("sum")
        work["_prob"] = exp_scores / group_sum

        group_wins = work.groupby(group_column)[target].sum()
        valid_groups = group_wins[group_wins == 1].index

        if len(valid_groups) == 0:
            return {
                "group_log_loss": float("nan"),
                "group_accuracy": float("nan"),
                "group_count": 0
            }

        valid_mask = work[group_column].isin(valid_groups) & (work[target] == 1)
        win_probs = work.loc[valid_mask, "_prob"]
        win_probs = np.clip(win_probs, 1e-15, 1.0)
        group_log_loss = -float(np.mean(np.log(win_probs)))

        pred_winner_idx = work.groupby(group_column)["_prob"].idxmax()
        true_winner_idx = (
            work.loc[valid_mask]
            .groupby(group_column)
            .apply(lambda group: group.index[0])
        )
        aligned_pred = pred_winner_idx.loc[valid_groups]
        aligned_true = true_winner_idx.loc[valid_groups]
        group_accuracy = float((aligned_pred == aligned_true).mean())

        return {
            "group_log_loss": group_log_loss,
            "group_accuracy": group_accuracy,
            "group_count": int(len(valid_groups))
        }

    def _apply_probability_expression(
        self,
        y_pred: np.ndarray,
        expression: str,
        target: str,
    ) -> np.ndarray:
        """Apply a DuckDB SQL expression to convert predictions to probabilities.

        Uses the same DuckDB pattern as InferenceRunner._apply_post_processing().
        """
        import duckdb

        conn = duckdb.connect(":memory:")
        pred_df = pd.DataFrame({"prediction": y_pred.astype(float)})
        conn.register("predictions", pred_df)

        sql_expr = expression.replace("{prediction}", "prediction")
        sql = f"SELECT ({sql_expr}) AS prob FROM predictions"

        result = conn.execute(sql).fetchdf()
        conn.close()

        return result["prob"].values

    def _compute_eval_metrics(
        self,
        df: pd.DataFrame,
        X_test: pd.DataFrame,
        y_pred: np.ndarray,
        test_output: pd.DataFrame,
        eval_config: Dict[str, Any],
        target: str,
        is_regression: bool,
        model: Any,
        group_column: Optional[str] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """Compute universal evaluation metrics for cross-model comparison.

        Converts predictions to probabilities and evaluates against a binary target,
        enabling apples-to-apples comparison between regression and classification models.
        """
        from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

        binary_target_name = eval_config["binary_target"]

        # Resolve binary target values
        if binary_target_name == target:
            y_binary = test_output[f"{target}_actual"].values
        elif binary_target_name in test_output.columns:
            y_binary = test_output[binary_target_name].values
        elif group_column and test_df is not None and binary_target_name in test_df.columns:
            y_binary = test_df[binary_target_name].values
        elif binary_target_name in df.columns:
            y_binary = df.loc[X_test.index, binary_target_name].values
        else:
            logger.warning(f"eval.binary_target '{binary_target_name}' not found, skipping eval metrics")
            return {}

        # Derive probabilities
        probability_expression = eval_config.get("probability_expression")
        if probability_expression:
            probabilities = self._apply_probability_expression(
                y_pred, probability_expression, target
            )
        else:
            proba_col = f"{target}_pred_proba"
            if proba_col in test_output.columns:
                probabilities = test_output[proba_col].values
            else:
                logger.warning(
                    f"No probability_expression and no {proba_col} column found, "
                    "skipping eval metrics"
                )
                return {}

        # Build working arrays, drop NaN rows
        y_binary = pd.to_numeric(y_binary, errors="coerce")
        probabilities = pd.to_numeric(probabilities, errors="coerce")
        mask = ~(np.isnan(y_binary) | np.isnan(probabilities))
        nan_count = (~mask).sum()
        if nan_count > 0:
            logger.warning(f"Dropping {nan_count} rows with NaN values for eval metrics")
        y_binary = y_binary[mask].astype(int)
        probabilities = probabilities[mask]

        if len(y_binary) == 0:
            logger.warning("No valid rows for eval metrics after NaN removal")
            return {}

        # Clip probabilities for numerical stability
        probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15)

        eval_metrics = {
            "auc_roc": float(roc_auc_score(y_binary, probabilities)),
            "log_loss": float(log_loss(y_binary, probabilities)),
            "brier_score": float(brier_score_loss(y_binary, probabilities)),
        }

        # Group-level metrics
        eval_group_col = eval_config.get("group_column")
        if eval_group_col:
            # Resolve group column values
            if eval_group_col in test_output.columns:
                group_values = test_output[eval_group_col].values
            elif group_column and test_df is not None and eval_group_col in test_df.columns:
                group_values = test_df[eval_group_col].values
            elif eval_group_col in df.columns:
                group_values = df.loc[X_test.index, eval_group_col].values
            else:
                logger.warning(f"eval.group_column '{eval_group_col}' not found, skipping group metrics")
                return eval_metrics

            # Apply same NaN mask
            group_values = group_values[mask]

            work = pd.DataFrame({
                "group": group_values,
                "prob": probabilities,
                "actual": y_binary,
            })

            # Top-1 accuracy: highest-prob runner in each group is the winner
            pred_winner_idx = work.groupby("group")["prob"].idxmax()
            top1_correct = work.loc[pred_winner_idx, "actual"].sum()
            n_groups = work["group"].nunique()
            eval_metrics["top_1_accuracy"] = float(top1_correct / n_groups) if n_groups > 0 else 0.0

            # Group log loss: -log(prob assigned to actual winner), only groups with 1 winner
            group_wins = work.groupby("group")["actual"].sum()
            valid_groups = group_wins[group_wins == 1].index
            if len(valid_groups) > 0:
                valid_winners = work[(work["group"].isin(valid_groups)) & (work["actual"] == 1)]
                winner_probs = np.clip(valid_winners["prob"].values, 1e-15, 1.0)
                eval_metrics["group_log_loss"] = -float(np.mean(np.log(winner_probs)))
                eval_metrics["group_count"] = int(len(valid_groups))
            else:
                eval_metrics["group_log_loss"] = float("nan")
                eval_metrics["group_count"] = 0

        return eval_metrics

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
        group_column = config.get("group_column")

        # Leakage guard
        leakage_columns = config.get("leakage_columns") or []
        if leakage_columns:
            leaked = set(features) & set(leakage_columns)
            if leaked:
                raise ValueError(
                    f"Features contain leakage columns: {sorted(leaked)}. "
                    "These columns must not be used as training features."
                )
            logger.info(f"Leakage columns declared (pass-through only): {leakage_columns}")
        # Support both "split" and "train_test_split" keys
        split_config = config.get("train_test_split") or config.get("split", {"method": "random", "test_size": 0.2})

        # Get model type to determine how to handle missing values
        model_config = config.get("model", config)
        model_type = model_config.get("type", "sklearn")

        test_df = None
        if group_column:
            if model_type not in ("xgboost", "lightgbm"):
                raise ValueError("group_column currently supports only xgboost/lightgbm models")

            X_train, X_test, y_train, y_test, group_train, group_test, train_df, test_df = self.prepare_group_data(
                df, features, target, split_config, group_column, model_type
            )
        else:
            X_train, X_test, y_train, y_test = self.prepare_data(
                df, features, target, split_config, model_type
            )

        # For XGBoost with categorical columns, ensure they're properly typed after split
        if model_type == "xgboost":
            for col in X_train.columns:
                if X_train[col].dtype == 'object':
                    X_train[col] = X_train[col].astype('category')
                    X_test[col] = X_test[col].astype('category')

        # Create and train model
        # Config might already be unwrapped by CLI
        model_config = config.get("model", config)
        group_objective = None
        if group_column:
            model_config = dict(model_config)
            model_config["group_column"] = group_column
            group_objective = config.get("group_objective", "rank:softmax")
            hyperparameters = dict(model_config.get("hyperparameters", {}))
            if "objective" not in hyperparameters:
                hyperparameters["objective"] = group_objective
            group_objective = hyperparameters.get("objective")
            model_config["hyperparameters"] = hyperparameters

        model = self.create_model(model_config)

        if group_column:
            model.fit(X_train, y_train, group=group_train)
        else:
            model.fit(X_train, y_train)

        # Evaluate
        if group_column:
            train_scores = model.predict(X_train)
            test_scores = model.predict(X_test)
            train_group_metrics = self._compute_group_metrics(train_df, train_scores, group_column, target)
            test_group_metrics = self._compute_group_metrics(test_df, test_scores, group_column, target)
            train_score = train_group_metrics["group_accuracy"]
            test_score = test_group_metrics["group_accuracy"]
            y_pred = test_scores
        else:
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
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
            "train_score": float(train_score) if train_score is not None else None,
            "test_score": float(test_score) if test_score is not None else None
        }

        confusion_mat = None
        if group_column:
            metrics.update({
                "group_log_loss": float(test_group_metrics["group_log_loss"]),
                "group_accuracy": float(test_group_metrics["group_accuracy"]),
                "group_log_loss_train": float(train_group_metrics["group_log_loss"]),
                "group_accuracy_train": float(train_group_metrics["group_accuracy"]),
                "group_count_train": int(train_group_metrics["group_count"]),
                "group_count_test": int(test_group_metrics["group_count"])
            })
        elif is_regression:
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

        # Save test set with predictions
        model_name = config["name"]
        test_dir = self.models_dir.parent / "test"
        test_dir.mkdir(parents=True, exist_ok=True)

        # Create test dataframe with predictions
        if group_column:
            test_output = test_df[[group_column]].join(X_test).copy()
        else:
            test_output = pd.DataFrame(X_test).copy()
        test_output[f"{target}_actual"] = y_test.values
        test_output[f"{target}_pred"] = y_pred

        # Add probabilities if available
        if group_column:
            prob_frame = pd.DataFrame({
                group_column: test_df[group_column].values,
                "_score": y_pred
            })
            group_max = prob_frame.groupby(group_column)["_score"].transform("max")
            exp_scores = np.exp(prob_frame["_score"] - group_max)
            group_sum = exp_scores.groupby(prob_frame[group_column]).transform("sum")
            test_output[f"{target}_pred_proba"] = (exp_scores / group_sum).values
        elif hasattr(model, "predict_proba"):
            if not is_regression:
                # For classification, get probability of positive class
                y_pred_proba = model.predict_proba(X_test)
                test_output[f"{target}_pred_proba"] = y_pred_proba[:, 1]

        # Append any declared leakage columns for post-hoc evaluation
        for col in leakage_columns:
            if col in df.columns:
                if group_column:
                    test_output[col] = test_df[col].values
                else:
                    test_output[col] = df.loc[X_test.index, col].values

        # Compute universal eval metrics if eval config is present
        eval_config = config.get("eval")
        if eval_config:
            # Ensure eval group_column is in test_output
            eval_group_col = eval_config.get("group_column")
            if eval_group_col and eval_group_col not in test_output.columns:
                if group_column and test_df is not None and eval_group_col in test_df.columns:
                    test_output[eval_group_col] = test_df[eval_group_col].values
                elif eval_group_col in df.columns:
                    test_output[eval_group_col] = df.loc[X_test.index, eval_group_col].values

            eval_metrics = self._compute_eval_metrics(
                df=df,
                X_test=X_test,
                y_pred=y_pred,
                test_output=test_output,
                eval_config=eval_config,
                target=target,
                is_regression=is_regression,
                model=model,
                group_column=group_column,
                test_df=test_df if group_column else None,
            )
            if eval_metrics:
                metrics["eval"] = eval_metrics
                logger.info(f"Eval metrics: {eval_metrics}")

        # Save to parquet
        test_output_path = test_dir / f"{model_name}_test.parquet"
        test_output.to_parquet(test_output_path)
        logger.info(f"Test set saved to {test_output_path}")

        # Save model
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
            "leakage_columns": leakage_columns,
            "group_column": group_column,
            "group_objective": group_objective,
            "eval_config": eval_config,
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

        # Sync to S3 if configured
        s3_url = None
        if self.s3_sync:
            logger.info("Syncing model artifacts to S3...")
            # Upload model file
            model_s3_key = f"data/models/{model_name}.pkl"
            self.s3_sync.upload_file(model_path, model_s3_key)

            # Upload metadata file
            metadata_s3_key = f"data/models/{model_name}_metadata.json"
            s3_url = self.s3_sync.upload_file(metadata_path, metadata_s3_key)

            if s3_url:
                logger.info(f"Model synced to S3: s3://{self.s3_bucket}/{self.s3_prefix}/data/models/{model_name}")

        return {
            "model_path": str(model_path),
            "metadata_path": str(metadata_path),
            "test_output_path": str(test_output_path),
            "s3_url": s3_url if s3_url else None,
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

        # If model doesn't exist locally and S3 is enabled, try to download from S3
        if not model_path.exists() and self.s3_sync:
            logger.info(f"Model not found locally, attempting to download from S3: {model_name}")
            try:
                # Download model file from S3
                model_s3_key = f"data/models/{model_name}.pkl"
                self.s3_sync.download_file(model_s3_key, str(model_path))
                logger.info(f"Downloaded model from S3: {model_name}")

                # Download metadata file from S3
                metadata_s3_key = f"data/models/{model_name}_metadata.json"
                self.s3_sync.download_file(metadata_s3_key, str(metadata_path))
                logger.info(f"Downloaded model metadata from S3: {model_name}")
            except Exception as e:
                logger.error(f"Failed to download model from S3: {e}")
                raise FileNotFoundError(f"Model not found locally or in S3: {model_path}") from e

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

        # Handle missing values same way as training - xgboost handles NaN natively
        model_type = metadata.get("model_type", "sklearn")
        if model_type != "xgboost":
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
        engine = DuckDBEngine(config=self.query_engine_config)
        result_df = engine.execute(sql)

        if result_df.empty:
            return []

        # Extract identifier columns if provided
        identifier_cols = []
        if select_columns:
            identifier_cols = [col.strip() for col in select_columns.split(",")]

        # Make predictions for each row
        results = []
        model_type = metadata.get("model_type", "sklearn")
        for _, row in result_df.iterrows():
            # Extract feature values
            feature_values = {feat: row[feat] for feat in expected_features}

            # Create DataFrame with correct feature order
            input_df = pd.DataFrame([feature_values])[expected_features]

            # Handle missing values same way as training - xgboost handles NaN natively
            if model_type != "xgboost":
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
