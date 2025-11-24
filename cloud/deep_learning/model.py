"""Deep Learning forecasting models for DER load and solar generation prediction.

This module provides LSTM and Transformer-based models for predicting
load (kW) and solar generation (kW) 24 hours ahead, with comparison
against the baseline Gradient Boosting model.
"""

import gc
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from cloud.deep_learning.config import DeepLearningConfig
from cloud.forecasting.model import ModelMetrics

logger = logging.getLogger(__name__)

# Optional TensorFlow/Keras imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks, Model
    from tensorflow.keras.optimizers import Adam

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None  # type: ignore
    keras = None  # type: ignore
    layers = None  # type: ignore
    callbacks = None  # type: ignore
    Model = None  # type: ignore
    Adam = None  # type: ignore

# Optional sklearn imports for scaling
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    StandardScaler = None  # type: ignore
    mean_absolute_error = None  # type: ignore
    mean_squared_error = None  # type: ignore
    r2_score = None  # type: ignore

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
class TrainingHistory:
    """Training history from deep learning model.

    Attributes:
        train_loss: Training loss per epoch
        val_loss: Validation loss per epoch
        best_epoch: Epoch with best validation loss
        stopped_early: Whether training was stopped early
        total_epochs: Total epochs trained
    """

    train_loss: list[float]
    val_loss: list[float]
    best_epoch: int
    stopped_early: bool
    total_epochs: int

    def to_dict(self) -> dict[str, Any]:
        """Convert history to dictionary."""
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "best_epoch": self.best_epoch,
            "stopped_early": self.stopped_early,
            "total_epochs": self.total_epochs,
        }


class DeepLearningForecaster:
    """Deep Learning forecaster using LSTM or Transformer architectures.

    This class handles:
    - Loading training data from Supabase
    - Creating sequences for time series modeling
    - Building LSTM or Transformer models
    - Training with early stopping
    - Evaluating model performance against baseline
    - Serializing trained models for deployment

    Example:
        >>> config = DeepLearningConfig(
        ...     supabase_url="https://your-project.supabase.co",
        ...     supabase_key="your-key",
        ...     enabled=True,
        ...     model_type="lstm",
        ... )
        >>> forecaster = DeepLearningForecaster(config)
        >>> result = forecaster.train_all()
        >>> print(f"Load MAE: {result['load']['metrics']['mae_percent']:.2f}%")
        >>> forecaster.save_models()
    """

    def __init__(self, config: DeepLearningConfig) -> None:
        """Initialize deep learning forecaster.

        Args:
            config: Deep learning configuration

        Raises:
            ImportError: If required libraries are not installed
            ValueError: If configuration is invalid
        """
        self.config = config
        self._supabase_client: Optional[Client] = None
        self._load_model: Optional[Model] = None
        self._solar_model: Optional[Model] = None
        self._load_metrics: Optional[ModelMetrics] = None
        self._solar_metrics: Optional[ModelMetrics] = None
        self._load_history: Optional[TrainingHistory] = None
        self._solar_history: Optional[TrainingHistory] = None
        # Per-target feature names and scalers to avoid cross-contamination
        self._feature_names_load: list[str] = []
        self._feature_names_solar: list[str] = []
        self._feature_scaler_load: Optional[StandardScaler] = None
        self._feature_scaler_solar: Optional[StandardScaler] = None
        self._target_scaler_load: Optional[StandardScaler] = None
        self._target_scaler_solar: Optional[StandardScaler] = None
        # Temporary storage for features from prepare_features
        self._available_features: list[str] = []

        if not config.enabled:
            logger.info("Deep learning forecasting is disabled")
            return

        self._validate_dependencies()
        config.validate()
        self._connect()

    def _validate_dependencies(self) -> None:
        """Validate required libraries are installed."""
        missing = []
        if not TF_AVAILABLE:
            missing.append("tensorflow")
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
                "Connected to Supabase at %s for deep learning forecasting",
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
            assert self._supabase_client is not None
            query = self._supabase_client.table(self.config.source_table).select("*")

            if start_time:
                query = query.gte("time", start_time.isoformat())
            if end_time:
                query = query.lte("time", end_time.isoformat())

            if self.config.site_id:
                query = query.eq("site_id", self.config.site_id)

            query = query.order("time").limit(limit)
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

        Normalizes column names and selects configured feature columns.

        Args:
            df: Raw training data DataFrame

        Returns:
            DataFrame with prepared features
        """
        if df.empty:
            return pd.DataFrame()

        logger.info("Preparing features from %d records", len(df))

        df = df.copy()
        df.columns = [col.replace(".", "_") for col in df.columns]

        feature_cols = [f.replace(".", "_") for f in self.config.get_all_features()]
        available_features = [col for col in feature_cols if col in df.columns]
        missing_features = set(feature_cols) - set(available_features)

        if missing_features:
            logger.warning(
                "Missing %d features: %s",
                len(missing_features),
                sorted(missing_features)[:5],
            )

        # Store available features temporarily for create_sequences to use
        self._available_features = available_features
        logger.info("Using %d features for training", len(available_features))

        return df

    def create_sequences(
        self,
        df: pd.DataFrame,
        target_col: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series modeling.

        Converts tabular data into sequences of (seq_length, n_features)
        for input to LSTM/Transformer models.

        Args:
            df: DataFrame with prepared features
            target_col: Name of the target column (normalized)

        Returns:
            Tuple of (X_sequences, y_targets) where:
                X_sequences: shape (n_samples, seq_length, n_features)
                y_targets: shape (n_samples,)
        """
        if df.empty:
            return np.array([]), np.array([])

        target_col_normalized = target_col.replace(".", "_")

        if target_col_normalized not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")

        # Calculate samples per hour for horizon
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
            "Creating sequences with length %d, horizon %d samples",
            self.config.sequence_length,
            horizon_samples,
        )

        # Determine which target we're training for (load vs solar)
        is_load_target = "load" in target_col_normalized.lower()

        # Get feature names (from prepare_features) and store per-target
        feature_names = self._available_features
        if is_load_target:
            self._feature_names_load = feature_names
        else:
            self._feature_names_solar = feature_names

        # Get feature and target data
        feature_data = df[feature_names].values
        target_data = df[target_col_normalized].values

        # Scale features if configured - use per-target scaler
        if self.config.scale_features:
            if is_load_target:
                if self._feature_scaler_load is None:
                    self._feature_scaler_load = StandardScaler()
                    feature_data = self._feature_scaler_load.fit_transform(feature_data)
                else:
                    feature_data = self._feature_scaler_load.transform(feature_data)
            else:
                if self._feature_scaler_solar is None:
                    self._feature_scaler_solar = StandardScaler()
                    feature_data = self._feature_scaler_solar.fit_transform(feature_data)
                else:
                    feature_data = self._feature_scaler_solar.transform(feature_data)

        # Create sequences
        X_list: list[np.ndarray] = []
        y_list: list[float] = []
        total_length = self.config.sequence_length + horizon_samples

        for i in range(len(feature_data) - total_length + 1):
            X_list.append(feature_data[i : i + self.config.sequence_length])
            y_list.append(target_data[i + self.config.sequence_length + horizon_samples - 1])

        X = np.array(X_list)
        y = np.array(y_list)

        # Remove NaN values
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]

        logger.info(
            "Created %d sequences with shape %s",
            len(X),
            X.shape,
        )

        return X, y

    def _build_lstm_model(self, n_features: int) -> Model:
        """Build LSTM model architecture.

        Args:
            n_features: Number of input features

        Returns:
            Compiled Keras model
        """
        inputs = keras.Input(shape=(self.config.sequence_length, n_features))
        x = inputs

        # Add LSTM layers
        for i in range(self.config.num_layers):
            return_sequences = i < self.config.num_layers - 1
            lstm_layer = layers.LSTM(
                self.config.hidden_units,
                return_sequences=return_sequences or self.config.use_attention,
                dropout=self.config.dropout,
                recurrent_dropout=self.config.dropout / 2,
            )
            if self.config.bidirectional:
                x = layers.Bidirectional(lstm_layer)(x)
            else:
                x = lstm_layer(x)

        # Add attention mechanism if configured
        if self.config.use_attention:
            # Simple attention: weighted average of LSTM outputs
            attention_weights = layers.Dense(1, activation="tanh")(x)
            attention_weights = layers.Softmax(axis=1)(attention_weights)
            x = layers.Multiply()([x, attention_weights])
            x = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(x)

        # Dense layers
        x = layers.Dense(self.config.hidden_units // 2, activation="relu")(x)
        x = layers.Dropout(self.config.dropout)(x)
        outputs = layers.Dense(1)(x)

        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=self.config.dl_learning_rate),
            loss="mse",
            metrics=["mae"],
        )

        logger.info("Built LSTM model with %d parameters", model.count_params())
        return model

    def _build_transformer_model(self, n_features: int) -> Model:
        """Build Transformer model architecture.

        Args:
            n_features: Number of input features

        Returns:
            Compiled Keras model
        """
        inputs = keras.Input(shape=(self.config.sequence_length, n_features))

        # Positional encoding
        positions = tf.range(start=0, limit=self.config.sequence_length, delta=1)
        position_embedding = layers.Embedding(
            input_dim=self.config.sequence_length,
            output_dim=n_features,
        )(positions)

        x = inputs + position_embedding

        # Transformer encoder blocks
        for _ in range(self.config.num_layers):
            # Multi-head self-attention
            attention_output = layers.MultiHeadAttention(
                num_heads=self.config.num_heads,
                key_dim=n_features // self.config.num_heads,
                dropout=self.config.dropout,
            )(x, x)
            attention_output = layers.Dropout(self.config.dropout)(attention_output)
            x = layers.LayerNormalization(epsilon=1e-6)(x + attention_output)

            # Feed-forward network
            ffn = layers.Dense(self.config.ff_dim, activation="relu")(x)
            ffn = layers.Dense(n_features)(ffn)
            ffn = layers.Dropout(self.config.dropout)(ffn)
            x = layers.LayerNormalization(epsilon=1e-6)(x + ffn)

        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)

        # Output layers
        x = layers.Dense(self.config.hidden_units, activation="relu")(x)
        x = layers.Dropout(self.config.dropout)(x)
        outputs = layers.Dense(1)(x)

        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=self.config.dl_learning_rate),
            loss="mse",
            metrics=["mae"],
        )

        logger.info("Built Transformer model with %d parameters", model.count_params())
        return model

    def _build_hybrid_model(self, n_features: int) -> Model:
        """Build hybrid LSTM + Transformer model architecture.

        Combines LSTM for sequential pattern learning with
        Transformer attention for capturing long-range dependencies.

        Args:
            n_features: Number of input features

        Returns:
            Compiled Keras model
        """
        inputs = keras.Input(shape=(self.config.sequence_length, n_features))

        # LSTM branch for sequential patterns
        lstm_out = layers.LSTM(
            self.config.hidden_units,
            return_sequences=True,
            dropout=self.config.dropout,
        )(inputs)

        # Transformer attention branch
        attention_out = layers.MultiHeadAttention(
            num_heads=self.config.num_heads,
            key_dim=n_features // self.config.num_heads,
            dropout=self.config.dropout,
        )(lstm_out, lstm_out)

        # Combine branches
        x = layers.Concatenate()([lstm_out, attention_out])
        x = layers.GlobalAveragePooling1D()(x)

        # Output layers
        x = layers.Dense(self.config.hidden_units, activation="relu")(x)
        x = layers.Dropout(self.config.dropout)(x)
        outputs = layers.Dense(1)(x)

        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=self.config.dl_learning_rate),
            loss="mse",
            metrics=["mae"],
        )

        logger.info("Built Hybrid model with %d parameters", model.count_params())
        return model

    def _build_model(self, n_features: int) -> Model:
        """Build model based on configured type.

        Args:
            n_features: Number of input features

        Returns:
            Compiled Keras model
        """
        if self.config.model_type == "lstm":
            return self._build_lstm_model(n_features)
        elif self.config.model_type == "transformer":
            return self._build_transformer_model(n_features)
        elif self.config.model_type == "hybrid":
            return self._build_hybrid_model(n_features)
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

    def _evaluate_model(
        self,
        model: Model,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> ModelMetrics:
        """Evaluate model performance on test data.

        Args:
            model: Trained Keras model
            X_test: Test feature sequences
            y_test: Test targets

        Returns:
            ModelMetrics with evaluation results
        """
        y_pred = model.predict(X_test, verbose=0).flatten()

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Calculate MAPE (handling zero values)
        non_zero_mask = y_test != 0
        if non_zero_mask.any():
            mape = np.mean(
                np.abs((y_test[non_zero_mask] - y_pred[non_zero_mask]) / y_test[non_zero_mask])
            ) * 100
        else:
            mape = 0.0

        target_mean = y_test.mean()
        mae_percent = (mae / target_mean * 100) if target_mean != 0 else 0.0
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

    def _train_model(
        self,
        df: pd.DataFrame,
        target_col: str,
        model_attr: str,
        metrics_attr: str,
        history_attr: str,
    ) -> dict[str, Any]:
        """Generic model training method.

        Args:
            df: Prepared training data
            target_col: Name of the target column
            model_attr: Name of the model attribute to set
            metrics_attr: Name of the metrics attribute to set
            history_attr: Name of the history attribute to set

        Returns:
            Dictionary with training results including metrics
        """
        logger.info("Training %s forecasting model with %s", target_col, self.config.model_type)

        # Create sequences
        X, y = self.create_sequences(df, target_col)

        if len(X) < 100:
            raise ValueError(
                f"Insufficient data for training: {len(X)} sequences (need at least 100)"
            )

        # Time-based train/test split
        split_idx = int(len(X) * (1 - self.config.test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Further split train for validation
        val_split_idx = int(len(X_train) * (1 - self.config.validation_split))
        X_train_final, X_val = X_train[:val_split_idx], X_train[val_split_idx:]
        y_train_final, y_val = y_train[:val_split_idx], y_train[val_split_idx:]

        logger.info(
            "Split data: %d train, %d val, %d test",
            len(X_train_final),
            len(X_val),
            len(X_test),
        )

        # Build model
        n_features = X_train.shape[2]
        model = self._build_model(n_features)

        # Set up callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=self.config.early_stopping_patience // 2,
            min_lr=1e-6,
            verbose=1,
        )

        # Train model
        history = model.fit(
            X_train_final,
            y_train_final,
            validation_data=(X_val, y_val),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1,
        )

        setattr(self, model_attr, model)

        # Create training history
        train_losses = history.history.get("loss", [])
        val_losses = history.history.get("val_loss", [])

        # Compute best_epoch from validation loss (or train loss if no validation)
        if val_losses:
            best_epoch_idx = int(np.argmin(val_losses))
        elif train_losses:
            best_epoch_idx = int(np.argmin(train_losses))
        else:
            best_epoch_idx = 0

        train_history = TrainingHistory(
            train_loss=train_losses,
            val_loss=val_losses,
            best_epoch=best_epoch_idx + 1,  # 1-based epoch index
            stopped_early=getattr(early_stopping, "stopped_epoch", 0) > 0,
            total_epochs=len(train_losses),
        )
        setattr(self, history_attr, train_history)

        # Evaluate
        metrics = self._evaluate_model(model, X_test, y_test)
        setattr(self, metrics_attr, metrics)

        logger.info(
            "%s model trained - MAE: %.4f (%.2f%%), R2: %.4f, Target met: %s",
            target_col,
            metrics.mae,
            metrics.mae_percent,
            metrics.r2,
            metrics.meets_target,
        )

        return {
            "target": target_col,
            "train_samples": len(X_train_final),
            "validation_samples": len(X_val),
            "test_samples": len(X_test),
            "metrics": metrics.to_dict(),
            "training_history": train_history.to_dict(),
            "model_type": self.config.model_type,
            "sequence_length": self.config.sequence_length,
            "parameters": model.count_params(),
        }

    def train_load_model(self, df: pd.DataFrame) -> dict[str, Any]:
        """Train the load forecasting model.

        Args:
            df: Prepared training data

        Returns:
            Dictionary with training results including metrics
        """
        return self._train_model(
            df,
            self.config.TARGET_LOAD,
            "_load_model",
            "_load_metrics",
            "_load_history",
        )

    def train_solar_model(self, df: pd.DataFrame) -> dict[str, Any]:
        """Train the solar generation forecasting model.

        Args:
            df: Prepared training data

        Returns:
            Dictionary with training results including metrics
        """
        return self._train_model(
            df,
            self.config.TARGET_SOLAR,
            "_solar_model",
            "_solar_metrics",
            "_solar_history",
        )

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

        logger.info("Starting deep learning model training pipeline")
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

            # Train both models (each uses its own per-target scaler)
            load_result = self.train_load_model(df)
            solar_result = self.train_solar_model(df)

            elapsed = (datetime.now(timezone.utc) - start).total_seconds()

            result = {
                "status": "success",
                "load": load_result,
                "solar": solar_result,
                "elapsed_seconds": elapsed,
                "features_used": len(self._feature_names_load),
                "model_type": self.config.model_type,
            }

            logger.info(
                "Deep learning training completed in %.2fs - Load MAE: %.2f%%, Solar MAE: %.2f%%",
                elapsed,
                load_result["metrics"]["mae_percent"],
                solar_result["metrics"]["mae_percent"],
            )

            return result

        except Exception as e:
            logger.exception("Deep learning model training failed")
            return {
                "status": "error",
                "error": str(e),
                "load": None,
                "solar": None,
            }

    def save_models(self) -> dict[str, str]:
        """Save trained models to disk.

        Saves both Keras models (.keras) and metadata with scalers (.joblib).

        Returns:
            Dictionary mapping model names to file paths
        """
        if not JOBLIB_AVAILABLE:
            raise ImportError("joblib is required for model serialization")

        model_dir = Path(self.config.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        saved = {}

        # Save load model
        if self._load_model is not None:
            # Save Keras model
            keras_path = self.config.get_keras_model_path("load")
            self._load_model.save(keras_path)
            saved["load_keras"] = str(keras_path)

            # Save metadata with scalers (per-target)
            joblib_path = self.config.get_model_path("load")
            joblib.dump(
                {
                    "features": self._feature_names_load,
                    "feature_scaler": self._feature_scaler_load,
                    "metrics": self._load_metrics.to_dict() if self._load_metrics else None,
                    "history": self._load_history.to_dict() if self._load_history else None,
                    "config": {
                        "horizon_hours": self.config.horizon_hours,
                        "target": self.config.TARGET_LOAD,
                        "model_type": self.config.model_type,
                        "sequence_length": self.config.sequence_length,
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                joblib_path,
            )
            saved["load_metadata"] = str(joblib_path)
            logger.info("Saved load model to %s", keras_path)

        # Save solar model
        if self._solar_model is not None:
            keras_path = self.config.get_keras_model_path("solar")
            self._solar_model.save(keras_path)
            saved["solar_keras"] = str(keras_path)

            # Save metadata with scalers (per-target)
            joblib_path = self.config.get_model_path("solar")
            joblib.dump(
                {
                    "features": self._feature_names_solar,
                    "feature_scaler": self._feature_scaler_solar,
                    "metrics": self._solar_metrics.to_dict() if self._solar_metrics else None,
                    "history": self._solar_history.to_dict() if self._solar_history else None,
                    "config": {
                        "horizon_hours": self.config.horizon_hours,
                        "target": self.config.TARGET_SOLAR,
                        "model_type": self.config.model_type,
                        "sequence_length": self.config.sequence_length,
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                joblib_path,
            )
            saved["solar_metadata"] = str(joblib_path)
            logger.info("Saved solar model to %s", keras_path)

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
        keras_path = self.config.get_keras_model_path("load")
        joblib_path = self.config.get_model_path("load")

        if keras_path.exists() and joblib_path.exists():
            try:
                self._load_model = keras.models.load_model(keras_path)
                data = joblib.load(joblib_path)
                self._feature_names_load = data["features"]
                self._feature_scaler_load = data.get("feature_scaler")
                if data.get("metrics"):
                    self._load_metrics = ModelMetrics(**data["metrics"])
                if data.get("history"):
                    self._load_history = TrainingHistory(**data["history"])
                loaded["load"] = True
                logger.info("Loaded load model from %s", keras_path)
            except Exception as e:
                logger.exception("Failed to load load model: %s", e)
                loaded["load"] = False
        else:
            loaded["load"] = False

        # Load solar model
        keras_path = self.config.get_keras_model_path("solar")
        joblib_path = self.config.get_model_path("solar")

        if keras_path.exists() and joblib_path.exists():
            try:
                self._solar_model = keras.models.load_model(keras_path)
                data = joblib.load(joblib_path)
                self._feature_names_solar = data["features"]
                self._feature_scaler_solar = data.get("feature_scaler")
                if data.get("metrics"):
                    self._solar_metrics = ModelMetrics(**data["metrics"])
                if data.get("history"):
                    self._solar_history = TrainingHistory(**data["history"])
                loaded["solar"] = True
                logger.info("Loaded solar model from %s", keras_path)
            except Exception as e:
                logger.exception("Failed to load solar model: %s", e)
                loaded["solar"] = False
        else:
            loaded["solar"] = False

        return loaded

    def predict_load(self, features: pd.DataFrame) -> np.ndarray:
        """Predict load 24 hours ahead.

        Args:
            features: DataFrame with feature values (must have sequence_length rows)

        Returns:
            Array of predicted load values (kW)
        """
        if self._load_model is None:
            raise ValueError("Load model not trained or loaded")

        return self._predict(
            self._load_model,
            features,
            self._feature_names_load,
            self._feature_scaler_load,
        )

    def predict_solar(self, features: pd.DataFrame) -> np.ndarray:
        """Predict solar generation 24 hours ahead.

        Args:
            features: DataFrame with feature values (must have sequence_length rows)

        Returns:
            Array of predicted solar generation values (kW)
        """
        if self._solar_model is None:
            raise ValueError("Solar model not trained or loaded")

        return self._predict(
            self._solar_model,
            features,
            self._feature_names_solar,
            self._feature_scaler_solar,
        )

    def _predict(
        self,
        model: Model,
        features: pd.DataFrame,
        feature_names: list[str],
        feature_scaler: Optional[StandardScaler],
    ) -> np.ndarray:
        """Make predictions with a trained model.

        Args:
            model: Trained Keras model
            features: DataFrame with feature values
            feature_names: List of feature column names for this target
            feature_scaler: Scaler for this target's features (or None)

        Returns:
            Array of predicted values
        """
        features = features.copy()
        features.columns = [col.replace(".", "_") for col in features.columns]

        # Get feature values
        X = features[feature_names].values

        # Scale if we have a scaler
        if feature_scaler is not None and self.config.scale_features:
            X = feature_scaler.transform(X)

        # Create sequences if needed
        if len(X) >= self.config.sequence_length:
            # Create overlapping sequences
            sequences = []
            for i in range(len(X) - self.config.sequence_length + 1):
                sequences.append(X[i : i + self.config.sequence_length])
            X_seq = np.array(sequences)
        else:
            # Pad to sequence length
            padding = np.zeros((self.config.sequence_length - len(X), X.shape[1]))
            X_padded = np.vstack([padding, X])
            X_seq = np.expand_dims(X_padded, axis=0)

        return model.predict(X_seq, verbose=0).flatten()

    def get_metrics(self) -> dict[str, Optional[dict]]:
        """Get metrics for both trained models.

        Returns:
            Dictionary with metrics for load and solar models
        """
        return {
            "load": self._load_metrics.to_dict() if self._load_metrics else None,
            "solar": self._solar_metrics.to_dict() if self._solar_metrics else None,
        }

    def get_training_history(self) -> dict[str, Optional[dict]]:
        """Get training history for both models.

        Returns:
            Dictionary with training history for load and solar models
        """
        return {
            "load": self._load_history.to_dict() if self._load_history else None,
            "solar": self._solar_history.to_dict() if self._solar_history else None,
        }

    def close(self) -> None:
        """Close Supabase connection and clear models from memory."""
        self._supabase_client = None
        # Clear Keras models to free GPU memory
        if self._load_model is not None or self._solar_model is not None:
            keras.backend.clear_session()
            gc.collect()
        self._load_model = None
        self._solar_model = None
        logger.info("Closed deep learning forecaster connections")

    def __enter__(self) -> "DeepLearningForecaster":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()


def compare_with_baseline(
    dl_metrics: ModelMetrics,
    baseline_metrics: ModelMetrics,
    target_mae_percent: float = 5.0,
) -> dict[str, Any]:
    """Compare deep learning model metrics with baseline.

    Args:
        dl_metrics: Metrics from deep learning model
        baseline_metrics: Metrics from baseline model
        target_mae_percent: Target MAE percentage threshold (default: 5.0)

    Returns:
        Dictionary with comparison results
    """
    mae_improvement = (
        (baseline_metrics.mae - dl_metrics.mae) / baseline_metrics.mae * 100
        if baseline_metrics.mae > 0
        else 0.0
    )
    rmse_improvement = (
        (baseline_metrics.rmse - dl_metrics.rmse) / baseline_metrics.rmse * 100
        if baseline_metrics.rmse > 0
        else 0.0
    )
    r2_improvement = dl_metrics.r2 - baseline_metrics.r2

    return {
        "baseline": baseline_metrics.to_dict(),
        "deep_learning": dl_metrics.to_dict(),
        "improvements": {
            "mae_percent_improvement": mae_improvement,
            "rmse_percent_improvement": rmse_improvement,
            "r2_absolute_improvement": r2_improvement,
        },
        "dl_is_better": mae_improvement > 0 and r2_improvement > 0,
        "meets_target": dl_metrics.mae_percent <= target_mae_percent,
    }
