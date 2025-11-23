"""Baseline forecasting model for DER load and solar generation prediction.

This module provides a Gradient Boosting-based forecaster for predicting
load (kW) and solar generation (kW) 24 hours ahead.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from cloud.forecasting.config import ForecastingConfig

logger = logging.getLogger(__name__)

# Optional sklearn imports
try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    GradientBoostingRegressor = None  # type: ignore
    mean_absolute_error = None  # type: ignore
    mean_squared_error = None  # type: ignore
    r2_score = None  # type: ignore
    train_test_split = None  # type: ignore

# Optional joblib imports
try:
    import joblib

    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    joblib = None  # type: ignore

# Optional Supabase import
try:
    from supabase import Client, create_client

    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    Client = None  # type: ignore
    create_client = None  # type: ignore


@dataclass
class ModelMetrics:
    """Metrics from model evaluation.

    Attributes:
        mae: Mean Absolute Error
        rmse: Root Mean Squared Error
        r2: R-squared (coefficient of determination)
        mape: Mean Absolute Percentage Error (relative MAE)
        target_mean: Mean of target values (for context)
        mae_percent: MAE as percentage of target mean
        meets_target: Whether MAE% meets the configured target
    """

    mae: float
    rmse: float
    r2: float
    mape: float
    target_mean: float
    mae_percent: float
    meets_target: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "mae": self.mae,
            "rmse": self.rmse,
            "r2": self.r2,
            "mape": self.mape,
            "target_mean": self.target_mean,
            "mae_percent": self.mae_percent,
            "meets_target": self.meets_target,
        }


class BaselineForecaster:
    """Baseline forecasting model using Gradient Boosting.

    This class handles:
    - Loading training data from Supabase
    - Creating 24-hour ahead target variables
    - Training separate models for load and solar generation
    - Evaluating model performance against MAE target
    - Serializing trained models for deployment

    Example:
        >>> config = ForecastingConfig(
        ...     supabase_url="https://your-project.supabase.co",
        ...     supabase_key="your-key",
        ...     enabled=True
        ... )
        >>> forecaster = BaselineForecaster(config)
        >>> result = forecaster.train_all()
        >>> print(f"Load MAE: {result['load']['metrics']['mae_percent']:.2f}%")
        >>> forecaster.save_models()
    """

    def __init__(self, config: ForecastingConfig) -> None:
        """Initialize baseline forecaster.

        Args:
            config: Forecasting configuration

        Raises:
            ImportError: If required libraries are not installed
            ValueError: If configuration is invalid
        """
        self.config = config
        self._supabase_client: Optional[Client] = None
        self._load_model: Optional[GradientBoostingRegressor] = None
        self._solar_model: Optional[GradientBoostingRegressor] = None
        self._load_metrics: Optional[ModelMetrics] = None
        self._solar_metrics: Optional[ModelMetrics] = None
        self._feature_names: list[str] = []

        if not config.enabled:
            logger.info("Forecasting is disabled")
            return

        self._validate_dependencies()
        config.validate()
        self._connect()

    def _validate_dependencies(self) -> None:
        """Validate required libraries are installed."""
        missing = []
        if not SKLEARN_AVAILABLE:
            missing.append("scikit-learn")
        if not JOBLIB_AVAILABLE:
            missing.append("joblib")
        if not SUPABASE_AVAILABLE:
            missing.append("supabase")

        if missing:
            raise ImportError(
                f"Required libraries not installed: {', '.join(missing)}. "
                f"Install with: pip install {' '.join(missing)}"
            )

    def _connect(self) -> None:
        """Establish connection to Supabase."""
        try:
            self._supabase_client = create_client(
                self.config.supabase_url,
                self.config.supabase_key,
            )
            logger.info(
                "Connected to Supabase at %s for forecasting",
                self.config.supabase_url,
            )
        except Exception:
            logger.exception("Failed to connect to Supabase")
            raise

    def is_connected(self) -> bool:
        """Check if Supabase connection is established.

        Returns:
            True if connected to Supabase
        """
        if not self.config.enabled:
            return False
        return self._supabase_client is not None

    def load_training_data(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 50000,
    ) -> pd.DataFrame:
        """Load training data from Supabase.

        Args:
            start_time: Start of time range (optional)
            end_time: End of time range (optional)
            limit: Maximum number of records to load

        Returns:
            DataFrame with training data
        """
        if not self.is_connected():
            logger.warning("Not connected, cannot load data")
            return pd.DataFrame()

        logger.info("Loading training data from %s", self.config.source_table)

        try:
            # Build query (assert for type narrowing - is_connected() already checks this)
            assert self._supabase_client is not None
            query = self._supabase_client.table(self.config.source_table).select("*")

            # Add time filters if specified
            if start_time:
                query = query.gte("time", start_time.isoformat())
            if end_time:
                query = query.lte("time", end_time.isoformat())

            # Add site filter if specified
            if self.config.site_id:
                query = query.eq("site_id", self.config.site_id)

            # Order by time and limit
            query = query.order("time").limit(limit)

            # Execute query
            result = query.execute()

            if not result.data:
                logger.info("No training data found")
                return pd.DataFrame()

            df = pd.DataFrame(result.data)
            logger.info("Loaded %d training records", len(df))
            return df

        except Exception as e:
            logger.exception("Failed to load training data: %s", e)
            raise

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature matrix from training data.

        Normalizes column names (replaces dots with underscores) and
        selects only the configured feature columns.

        Args:
            df: Raw training data DataFrame

        Returns:
            DataFrame with prepared features
        """
        if df.empty:
            return pd.DataFrame()

        logger.info("Preparing features from %d records", len(df))

        df = df.copy()

        # Normalize column names (replace dots with underscores for consistency)
        df.columns = [col.replace(".", "_") for col in df.columns]

        # Get feature columns (also normalize feature names)
        feature_cols = [f.replace(".", "_") for f in self.config.get_all_features()]

        # Filter to available features
        available_features = [col for col in feature_cols if col in df.columns]
        missing_features = set(feature_cols) - set(available_features)

        if missing_features:
            logger.warning(
                "Missing %d features: %s",
                len(missing_features),
                sorted(missing_features)[:5],  # Show first 5
            )

        self._feature_names = available_features
        logger.info("Using %d features for training", len(available_features))

        return df

    def create_target_variable(
        self, df: pd.DataFrame, target_col: str
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Create 24-hour ahead target variable.

        Shifts the target column forward by the configured horizon to create
        a prediction target for 24 hours ahead.

        Args:
            df: DataFrame with prepared features
            target_col: Name of the target column (normalized)

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        if df.empty:
            return pd.DataFrame(), pd.Series(dtype=float)

        # Normalize target column name
        target_col_normalized = target_col.replace(".", "_")

        if target_col_normalized not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")

        # Calculate shift for 24-hour horizon
        # Assuming 5-minute intervals: 24 hours = 24 * 12 = 288 samples
        samples_per_hour = 12  # 5-minute intervals
        if len(df) > 1 and "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
            time_diff_minutes = (
                df["time"].iloc[1] - df["time"].iloc[0]
            ).total_seconds() / 60
            if time_diff_minutes > 0:
                samples_per_hour = round(60 / time_diff_minutes)

        horizon_samples = self.config.horizon_hours * samples_per_hour
        logger.info(
            "Creating target with %d-hour horizon (%d samples)",
            self.config.horizon_hours,
            horizon_samples,
        )

        # Create target (future value)
        df = df.copy()
        df["target"] = df[target_col_normalized].shift(-horizon_samples)

        # Drop rows where we don't have future data
        df = df.dropna(subset=["target"])

        # Get features and target
        X = df[self._feature_names].copy()
        y = df["target"].copy()

        # Drop rows with missing features
        valid_idx = X.dropna().index
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]

        logger.info(
            "Created target variable with %d valid samples (dropped %d)",
            len(X),
            len(df) - len(X),
        )

        return X, y

    def _create_model(self) -> GradientBoostingRegressor:
        """Create a new Gradient Boosting model with configured hyperparameters.

        Returns:
            Configured GradientBoostingRegressor
        """
        return GradientBoostingRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=self.config.random_state,
            verbose=0,
        )

    def _evaluate_model(
        self,
        model: GradientBoostingRegressor,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> ModelMetrics:
        """Evaluate model performance on test data.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets

        Returns:
            ModelMetrics with evaluation results
        """
        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Calculate MAPE (handling zero values)
        non_zero_mask = y_test != 0
        if non_zero_mask.any():
            mape = np.mean(np.abs((y_test[non_zero_mask] - y_pred[non_zero_mask]) / y_test[non_zero_mask])) * 100
        else:
            mape = 0.0

        # Calculate MAE as percentage of mean
        target_mean = y_test.mean()
        mae_percent = (mae / target_mean * 100) if target_mean != 0 else 0.0

        # Check if meets target
        meets_target = mae_percent <= self.config.target_mae_percent

        return ModelMetrics(
            mae=mae,
            rmse=rmse,
            r2=r2,
            mape=mape,
            target_mean=target_mean,
            mae_percent=mae_percent,
            meets_target=meets_target,
        )

    def train_load_model(
        self, df: pd.DataFrame
    ) -> dict[str, Any]:
        """Train the load forecasting model.

        Args:
            df: Prepared training data

        Returns:
            Dictionary with training results including metrics
        """
        logger.info("Training load forecasting model")

        target_col = self.config.TARGET_LOAD

        # Create target variable
        X, y = self.create_target_variable(df, target_col)

        if len(X) < 100:
            raise ValueError(
                f"Insufficient data for training: {len(X)} samples (need at least 100)"
            )

        # Train/test split (time-based, not random for time series)
        split_idx = int(len(X) * (1 - self.config.test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        logger.info(
            "Split data: %d train, %d test (%.1f%%)",
            len(X_train),
            len(X_test),
            self.config.test_size * 100,
        )

        # Create and train model
        self._load_model = self._create_model()
        self._load_model.fit(X_train, y_train)

        # Evaluate
        self._load_metrics = self._evaluate_model(self._load_model, X_test, y_test)

        logger.info(
            "Load model trained - MAE: %.4f (%.2f%%), R2: %.4f, Target met: %s",
            self._load_metrics.mae,
            self._load_metrics.mae_percent,
            self._load_metrics.r2,
            self._load_metrics.meets_target,
        )

        return {
            "target": target_col,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "metrics": self._load_metrics.to_dict(),
            "feature_importance": self._get_feature_importance(self._load_model),
        }

    def train_solar_model(
        self, df: pd.DataFrame
    ) -> dict[str, Any]:
        """Train the solar generation forecasting model.

        Args:
            df: Prepared training data

        Returns:
            Dictionary with training results including metrics
        """
        logger.info("Training solar generation forecasting model")

        target_col = self.config.TARGET_SOLAR

        # Create target variable
        X, y = self.create_target_variable(df, target_col)

        if len(X) < 100:
            raise ValueError(
                f"Insufficient data for training: {len(X)} samples (need at least 100)"
            )

        # Train/test split (time-based, not random for time series)
        split_idx = int(len(X) * (1 - self.config.test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        logger.info(
            "Split data: %d train, %d test (%.1f%%)",
            len(X_train),
            len(X_test),
            self.config.test_size * 100,
        )

        # Create and train model
        self._solar_model = self._create_model()
        self._solar_model.fit(X_train, y_train)

        # Evaluate
        self._solar_metrics = self._evaluate_model(self._solar_model, X_test, y_test)

        logger.info(
            "Solar model trained - MAE: %.4f (%.2f%%), R2: %.4f, Target met: %s",
            self._solar_metrics.mae,
            self._solar_metrics.mae_percent,
            self._solar_metrics.r2,
            self._solar_metrics.meets_target,
        )

        return {
            "target": target_col,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "metrics": self._solar_metrics.to_dict(),
            "feature_importance": self._get_feature_importance(self._solar_model),
        }

    def _get_feature_importance(
        self, model: GradientBoostingRegressor
    ) -> dict[str, float]:
        """Get feature importance from trained model.

        Args:
            model: Trained model

        Returns:
            Dictionary mapping feature names to importance scores
        """
        importance = dict(zip(self._feature_names, model.feature_importances_))
        # Sort by importance descending
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def train_all(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """Train both load and solar forecasting models.

        Args:
            start_time: Start of time range for training data
            end_time: End of time range for training data

        Returns:
            Dictionary with results for both models
        """
        if not self.config.enabled:
            return {
                "status": "disabled",
                "load": None,
                "solar": None,
            }

        logger.info("Starting model training pipeline")
        start = datetime.now(timezone.utc)

        try:
            # Load training data
            df = self.load_training_data(start_time, end_time)
            if df.empty:
                return {
                    "status": "no_data",
                    "load": None,
                    "solar": None,
                }

            # Prepare features
            df = self.prepare_features(df)

            # Train both models
            load_result = self.train_load_model(df)
            solar_result = self.train_solar_model(df)

            elapsed = (datetime.now(timezone.utc) - start).total_seconds()

            result = {
                "status": "success",
                "load": load_result,
                "solar": solar_result,
                "elapsed_seconds": elapsed,
                "features_used": len(self._feature_names),
            }

            logger.info(
                "Model training completed in %.2fs - Load MAE: %.2f%%, Solar MAE: %.2f%%",
                elapsed,
                load_result["metrics"]["mae_percent"],
                solar_result["metrics"]["mae_percent"],
            )

            return result

        except Exception as e:
            logger.exception("Model training failed")
            return {
                "status": "error",
                "error": str(e),
                "load": None,
                "solar": None,
            }

    def save_models(self) -> dict[str, str]:
        """Save trained models to disk using joblib.

        Returns:
            Dictionary mapping model names to file paths
        """
        if not JOBLIB_AVAILABLE:
            raise ImportError("joblib is required for model serialization")

        # Ensure model directory exists
        model_dir = Path(self.config.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        saved = {}

        # Save load model
        if self._load_model is not None:
            load_path = self.config.get_model_path("load")
            joblib.dump(
                {
                    "model": self._load_model,
                    "features": self._feature_names,
                    "metrics": self._load_metrics.to_dict() if self._load_metrics else None,
                    "config": {
                        "horizon_hours": self.config.horizon_hours,
                        "target": self.config.TARGET_LOAD,
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                load_path,
            )
            saved["load"] = str(load_path)
            logger.info("Saved load model to %s", load_path)

        # Save solar model
        if self._solar_model is not None:
            solar_path = self.config.get_model_path("solar")
            joblib.dump(
                {
                    "model": self._solar_model,
                    "features": self._feature_names,
                    "metrics": self._solar_metrics.to_dict() if self._solar_metrics else None,
                    "config": {
                        "horizon_hours": self.config.horizon_hours,
                        "target": self.config.TARGET_SOLAR,
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                solar_path,
            )
            saved["solar"] = str(solar_path)
            logger.info("Saved solar model to %s", solar_path)

        return saved

    def load_models(self) -> dict[str, bool]:
        """Load trained models from disk.

        Returns:
            Dictionary indicating which models were loaded successfully
        """
        if not JOBLIB_AVAILABLE:
            raise ImportError("joblib is required for model deserialization")

        loaded = {}

        # Load load model
        load_path = self.config.get_model_path("load")
        if load_path.exists():
            try:
                data = joblib.load(load_path)
                self._load_model = data["model"]
                self._feature_names = data["features"]
                if data.get("metrics"):
                    self._load_metrics = ModelMetrics(**data["metrics"])
                loaded["load"] = True
                logger.info("Loaded load model from %s", load_path)
            except Exception as e:
                logger.error("Failed to load load model: %s", e)
                loaded["load"] = False
        else:
            loaded["load"] = False

        # Load solar model
        solar_path = self.config.get_model_path("solar")
        if solar_path.exists():
            try:
                data = joblib.load(solar_path)
                self._solar_model = data["model"]
                self._feature_names = data["features"]
                if data.get("metrics"):
                    self._solar_metrics = ModelMetrics(**data["metrics"])
                loaded["solar"] = True
                logger.info("Loaded solar model from %s", solar_path)
            except Exception as e:
                logger.error("Failed to load solar model: %s", e)
                loaded["solar"] = False
        else:
            loaded["solar"] = False

        return loaded

    def predict_load(self, features: pd.DataFrame) -> np.ndarray:
        """Predict load 24 hours ahead.

        Args:
            features: DataFrame with feature values

        Returns:
            Array of predicted load values (kW)
        """
        if self._load_model is None:
            raise ValueError("Load model not trained or loaded")

        # Normalize column names
        features = features.copy()
        features.columns = [col.replace(".", "_") for col in features.columns]

        # Select required features
        X = features[self._feature_names]

        return self._load_model.predict(X)

    def predict_solar(self, features: pd.DataFrame) -> np.ndarray:
        """Predict solar generation 24 hours ahead.

        Args:
            features: DataFrame with feature values

        Returns:
            Array of predicted solar generation values (kW)
        """
        if self._solar_model is None:
            raise ValueError("Solar model not trained or loaded")

        # Normalize column names
        features = features.copy()
        features.columns = [col.replace(".", "_") for col in features.columns]

        # Select required features
        X = features[self._feature_names]

        return self._solar_model.predict(X)

    def get_metrics(self) -> dict[str, Optional[dict]]:
        """Get metrics for both trained models.

        Returns:
            Dictionary with metrics for load and solar models
        """
        return {
            "load": self._load_metrics.to_dict() if self._load_metrics else None,
            "solar": self._solar_metrics.to_dict() if self._solar_metrics else None,
        }

    def close(self) -> None:
        """Close Supabase connection."""
        self._supabase_client = None
        logger.info("Closed forecaster connections")

    def __enter__(self) -> "BaselineForecaster":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
