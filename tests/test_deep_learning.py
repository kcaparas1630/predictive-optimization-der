"""Tests for the deep learning forecasting model module."""

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from cloud.deep_learning.config import DeepLearningConfig
from cloud.deep_learning.model import (
    DeepLearningForecaster,
    TrainingHistory,
    compare_with_baseline,
)
from cloud.forecasting.model import ModelMetrics


class TestDeepLearningConfig:
    """Tests for DeepLearningConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        with patch.dict(
            os.environ,
            {
                "SUPABASE_URL": "",
                "SUPABASE_KEY": "",
                "DL_MODEL_TYPE": "",
                "DL_EPOCHS": "",
                "DL_BATCH_SIZE": "",
            },
            clear=True,
        ):
            config = DeepLearningConfig()

            # Inherited defaults
            assert config.supabase_url == ""
            assert config.supabase_key == ""
            assert config.enabled is False
            assert config.model_dir == "models"
            assert config.test_size == 0.2
            assert config.horizon_hours == 24

            # Deep learning specific defaults
            assert config.model_type == "lstm"
            assert config.sequence_length == 288
            assert config.epochs == 100
            assert config.batch_size == 32
            assert config.dl_learning_rate == 0.001
            assert config.validation_split == 0.15
            assert config.early_stopping_patience == 10
            assert config.hidden_units == 64
            assert config.num_layers == 2
            assert config.dropout == 0.2
            assert config.num_heads == 4
            assert config.ff_dim == 128
            assert config.use_attention is True
            assert config.bidirectional is False
            assert config.target_mae_percent == 5.0
            assert config.scale_features is True

    def test_explicit_values_override_defaults(self):
        """Test that explicit values override defaults."""
        config = DeepLearningConfig(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
            enabled=True,
            model_type="transformer",
            sequence_length=144,
            epochs=200,
            batch_size=64,
            dl_learning_rate=0.0005,
            hidden_units=128,
            num_layers=3,
            dropout=0.3,
            num_heads=8,
        )

        assert config.supabase_url == "https://test.supabase.co"
        assert config.supabase_key == "test-key"
        assert config.enabled is True
        assert config.model_type == "transformer"
        assert config.sequence_length == 144
        assert config.epochs == 200
        assert config.batch_size == 64
        assert config.dl_learning_rate == 0.0005
        assert config.hidden_units == 128
        assert config.num_layers == 3
        assert config.dropout == 0.3
        assert config.num_heads == 8

    def test_env_var_overrides(self):
        """Test environment variable overrides."""
        env_vars = {
            "SUPABASE_URL": "https://env.supabase.co",
            "SUPABASE_KEY": "env-key",
            "DL_MODEL_TYPE": "transformer",
            "DL_SEQUENCE_LENGTH": "144",
            "DL_EPOCHS": "200",
            "DL_BATCH_SIZE": "64",
            "DL_LEARNING_RATE": "0.0005",
            "DL_HIDDEN_UNITS": "128",
            "DL_NUM_LAYERS": "3",
            "DL_DROPOUT": "0.3",
            "DL_EARLY_STOPPING_PATIENCE": "15",
            "DL_VALIDATION_SPLIT": "0.2",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = DeepLearningConfig()

            assert config.supabase_url == "https://env.supabase.co"
            assert config.supabase_key == "env-key"
            assert config.model_type == "transformer"
            assert config.sequence_length == 144
            assert config.epochs == 200
            assert config.batch_size == 64
            assert config.dl_learning_rate == 0.0005
            assert config.hidden_units == 128
            assert config.num_layers == 3
            assert config.dropout == 0.3
            assert config.early_stopping_patience == 15
            assert config.validation_split == 0.2

    def test_explicit_values_override_env_vars(self):
        """Test that explicit values take precedence over env vars."""
        env_vars = {
            "DL_EPOCHS": "200",
            "DL_BATCH_SIZE": "128",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = DeepLearningConfig(
                epochs=50,
                batch_size=16,
            )

            assert config.epochs == 50
            assert config.batch_size == 16

    def test_invalid_sequence_length_env_var(self):
        """Test that invalid DL_SEQUENCE_LENGTH raises ValueError."""
        with patch.dict(os.environ, {"DL_SEQUENCE_LENGTH": "not-a-number"}, clear=True):
            with pytest.raises(ValueError, match="DL_SEQUENCE_LENGTH must be an integer"):
                DeepLearningConfig()

    def test_invalid_epochs_env_var(self):
        """Test that invalid DL_EPOCHS raises ValueError."""
        with patch.dict(os.environ, {"DL_EPOCHS": "invalid"}, clear=True):
            with pytest.raises(ValueError, match="DL_EPOCHS must be an integer"):
                DeepLearningConfig()

    def test_invalid_learning_rate_env_var(self):
        """Test that invalid DL_LEARNING_RATE raises ValueError."""
        with patch.dict(os.environ, {"DL_LEARNING_RATE": "abc"}, clear=True):
            with pytest.raises(ValueError, match="DL_LEARNING_RATE must be a float"):
                DeepLearningConfig()

    def test_validate_invalid_model_type(self):
        """Test validation fails for invalid model_type."""
        config = DeepLearningConfig(
            supabase_url="https://test.supabase.co",
            supabase_key="key",
        )
        config.model_type = "invalid"  # type: ignore
        with pytest.raises(ValueError, match="model_type must be one of"):
            config.validate()

    def test_validate_invalid_sequence_length(self):
        """Test validation fails for non-positive sequence_length."""
        config = DeepLearningConfig(
            supabase_url="https://test.supabase.co",
            supabase_key="key",
            sequence_length=0,
        )
        with pytest.raises(ValueError, match="sequence_length must be positive"):
            config.validate()

    def test_validate_invalid_epochs(self):
        """Test validation fails for non-positive epochs."""
        config = DeepLearningConfig(
            supabase_url="https://test.supabase.co",
            supabase_key="key",
            epochs=0,
        )
        with pytest.raises(ValueError, match="epochs must be positive"):
            config.validate()

    def test_validate_invalid_batch_size(self):
        """Test validation fails for non-positive batch_size."""
        config = DeepLearningConfig(
            supabase_url="https://test.supabase.co",
            supabase_key="key",
            batch_size=0,
        )
        with pytest.raises(ValueError, match="batch_size must be positive"):
            config.validate()

    def test_validate_invalid_learning_rate(self):
        """Test validation fails for non-positive learning_rate."""
        config = DeepLearningConfig(
            supabase_url="https://test.supabase.co",
            supabase_key="key",
            dl_learning_rate=0.0,
        )
        with pytest.raises(ValueError, match="dl_learning_rate must be positive"):
            config.validate()

    def test_validate_invalid_validation_split(self):
        """Test validation fails for invalid validation_split."""
        config = DeepLearningConfig(
            supabase_url="https://test.supabase.co",
            supabase_key="key",
            validation_split=1.0,
        )
        with pytest.raises(ValueError, match="validation_split must be between"):
            config.validate()

    def test_validate_invalid_dropout(self):
        """Test validation fails for invalid dropout."""
        config = DeepLearningConfig(
            supabase_url="https://test.supabase.co",
            supabase_key="key",
            dropout=1.5,
        )
        with pytest.raises(ValueError, match="dropout must be between"):
            config.validate()

    def test_get_model_path(self):
        """Test get_model_path returns correct path with model type."""
        config = DeepLearningConfig(model_dir="my_models", model_type="lstm")
        load_path = config.get_model_path("load")
        solar_path = config.get_model_path("solar")

        assert load_path == Path("my_models/dl_lstm_load_forecaster.joblib")
        assert solar_path == Path("my_models/dl_lstm_solar_forecaster.joblib")

        # Test with transformer
        config.model_type = "transformer"
        load_path = config.get_model_path("load")
        assert load_path == Path("my_models/dl_transformer_load_forecaster.joblib")

    def test_get_keras_model_path(self):
        """Test get_keras_model_path returns correct path."""
        config = DeepLearningConfig(model_dir="my_models", model_type="lstm")
        load_path = config.get_keras_model_path("load")
        solar_path = config.get_keras_model_path("solar")

        assert load_path == Path("my_models/dl_lstm_load_forecaster.keras")
        assert solar_path == Path("my_models/dl_lstm_solar_forecaster.keras")

    def test_model_types_constant(self):
        """Test MODEL_TYPES constant is defined correctly."""
        assert DeepLearningConfig.MODEL_TYPES == ["lstm", "transformer", "hybrid"]


class TestTrainingHistory:
    """Tests for TrainingHistory dataclass."""

    def test_history_creation(self):
        """Test creating TrainingHistory."""
        history = TrainingHistory(
            train_loss=[0.5, 0.4, 0.3, 0.25],
            val_loss=[0.6, 0.45, 0.35, 0.3],
            best_epoch=3,
            stopped_early=True,
            total_epochs=4,
        )

        assert history.train_loss == [0.5, 0.4, 0.3, 0.25]
        assert history.val_loss == [0.6, 0.45, 0.35, 0.3]
        assert history.best_epoch == 3
        assert history.stopped_early is True
        assert history.total_epochs == 4

    def test_to_dict(self):
        """Test converting history to dictionary."""
        history = TrainingHistory(
            train_loss=[0.5, 0.4],
            val_loss=[0.6, 0.45],
            best_epoch=2,
            stopped_early=False,
            total_epochs=2,
        )

        result = history.to_dict()

        assert isinstance(result, dict)
        assert result["train_loss"] == [0.5, 0.4]
        assert result["val_loss"] == [0.6, 0.45]
        assert result["best_epoch"] == 2
        assert result["stopped_early"] is False
        assert result["total_epochs"] == 2


class TestDeepLearningForecaster:
    """Tests for DeepLearningForecaster."""

    @pytest.fixture
    def disabled_config(self):
        """Create a disabled configuration."""
        return DeepLearningConfig(enabled=False)

    @pytest.fixture
    def mock_config(self):
        """Create a mock enabled configuration."""
        return DeepLearningConfig(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
            enabled=True,
            model_dir=tempfile.mkdtemp(),
            test_size=0.2,
            epochs=2,  # Very small for fast tests
            batch_size=16,
            sequence_length=24,  # Small for tests
            hidden_units=8,  # Small for tests
            num_layers=1,  # Minimal
            early_stopping_patience=1,
        )

    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data with all required features."""
        np.random.seed(42)
        n_samples = 500  # Enough for sequences

        base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        times = pd.date_range(start=base_time, periods=n_samples, freq="5min")

        data = {
            "time": times,
            "site_id": ["site-1"] * n_samples,
            "device_id": ["device-1"] * n_samples,
            # Targets
            "home_load.total_load_kw": 2.0 + 0.5 * np.sin(np.linspace(0, 10*np.pi, n_samples)) + np.random.normal(0, 0.2, n_samples),
            "solar.generation_kw": np.maximum(0, 3.0 * np.sin(np.linspace(0, 10*np.pi, n_samples)) + np.random.normal(0, 0.3, n_samples)),
            # Temporal features
            "hour_of_day": [t.hour for t in times],
            "day_of_week": [t.dayofweek for t in times],
            "day_of_month": [t.day for t in times],
            "month": [t.month for t in times],
            "is_weekend": [1 if t.dayofweek >= 5 else 0 for t in times],
            "hour_sin": np.sin(2 * np.pi * np.array([t.hour for t in times]) / 24),
            "hour_cos": np.cos(2 * np.pi * np.array([t.hour for t in times]) / 24),
            "day_sin": np.sin(2 * np.pi * np.array([t.dayofweek for t in times]) / 7),
            "day_cos": np.cos(2 * np.pi * np.array([t.dayofweek for t in times]) / 7),
            # Categorical features
            "tou_peak": [1 if 16 <= t.hour < 21 else 0 for t in times],
            "tou_off_peak": [1 if t.hour < 7 or t.hour >= 22 else 0 for t in times],
            "tou_shoulder": [1 if 7 <= t.hour < 16 or t.hour == 21 else 0 for t in times],
            # Rolling features
            "solar_generation_kw_rolling_avg_7d": np.random.uniform(1, 3, n_samples),
            "solar_generation_kw_rolling_std_7d": np.random.uniform(0.5, 1.5, n_samples),
            "home_load_total_load_kw_rolling_avg_7d": np.random.uniform(1.5, 2.5, n_samples),
            "home_load_total_load_kw_rolling_std_7d": np.random.uniform(0.3, 0.8, n_samples),
            "battery_soc_percent_rolling_avg_7d": np.random.uniform(40, 60, n_samples),
            "battery_soc_percent_rolling_std_7d": np.random.uniform(5, 15, n_samples),
            "grid_price_price_per_kwh_rolling_avg_7d": np.random.uniform(0.1, 0.2, n_samples),
            "grid_price_price_per_kwh_rolling_std_7d": np.random.uniform(0.02, 0.05, n_samples),
            "system_net_grid_flow_kw_rolling_avg_7d": np.random.uniform(-1, 1, n_samples),
            "system_net_grid_flow_kw_rolling_std_7d": np.random.uniform(0.5, 1.5, n_samples),
            # Lag features
            "solar_generation_kw_lag_1": np.roll(np.random.uniform(0, 5, n_samples), 1),
            "solar_generation_kw_lag_1h": np.roll(np.random.uniform(0, 5, n_samples), 12),
            "home_load_total_load_kw_lag_1": np.roll(np.random.uniform(1, 3, n_samples), 1),
            "home_load_total_load_kw_lag_1h": np.roll(np.random.uniform(1, 3, n_samples), 12),
            "grid_price_price_per_kwh_lag_1": np.roll(np.random.uniform(0.1, 0.3, n_samples), 1),
            "grid_price_price_per_kwh_lag_1h": np.roll(np.random.uniform(0.1, 0.3, n_samples), 12),
            # Raw metric features
            "solar_irradiance_w_m2": np.random.uniform(0, 1000, n_samples),
            "solar_panel_temp_celsius": np.random.uniform(20, 60, n_samples),
            "battery_soc_percent": np.random.uniform(20, 80, n_samples),
            "grid_price_price_per_kwh": np.random.uniform(0.08, 0.25, n_samples),
        }

        return pd.DataFrame(data)

    def test_init_disabled(self, disabled_config):
        """Test initialization with disabled config."""
        forecaster = DeepLearningForecaster(disabled_config)

        assert forecaster.config.enabled is False
        assert not forecaster.is_connected()

    @patch("cloud.deep_learning.model.create_client")
    def test_init_enabled(self, mock_create_client, mock_config):
        """Test initialization with enabled config."""
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client

        forecaster = DeepLearningForecaster(mock_config)

        assert forecaster.config.enabled is True
        assert forecaster.is_connected()
        mock_create_client.assert_called_once()

    def test_missing_tensorflow_import(self, mock_config):
        """Test that missing TensorFlow raises ImportError."""
        with patch("cloud.deep_learning.model.TF_AVAILABLE", False):
            with pytest.raises(ImportError, match="tensorflow"):
                DeepLearningForecaster(mock_config)

    @patch("cloud.deep_learning.model.create_client")
    def test_prepare_features(self, mock_create_client, mock_config, sample_training_data):
        """Test feature preparation."""
        mock_create_client.return_value = MagicMock()

        forecaster = DeepLearningForecaster(mock_config)
        df = forecaster.prepare_features(sample_training_data)

        # Check column name normalization
        assert "home_load_total_load_kw" in df.columns
        assert "solar_generation_kw" in df.columns

        # Check features were identified
        assert len(forecaster._feature_names) > 0

    @patch("cloud.deep_learning.model.create_client")
    def test_prepare_features_empty_dataframe(self, mock_create_client, mock_config):
        """Test prepare_features with empty DataFrame."""
        mock_create_client.return_value = MagicMock()

        forecaster = DeepLearningForecaster(mock_config)
        result = forecaster.prepare_features(pd.DataFrame())

        assert result.empty

    @patch("cloud.deep_learning.model.create_client")
    def test_create_sequences(self, mock_create_client, mock_config, sample_training_data):
        """Test sequence creation for time series."""
        mock_create_client.return_value = MagicMock()

        forecaster = DeepLearningForecaster(mock_config)
        df = forecaster.prepare_features(sample_training_data)

        X, y = forecaster.create_sequences(df, "home_load.total_load_kw")

        # Check shapes
        assert len(X.shape) == 3  # (n_samples, seq_length, n_features)
        assert X.shape[1] == mock_config.sequence_length
        assert X.shape[2] == len(forecaster._feature_names)
        assert len(y.shape) == 1  # (n_samples,)
        assert len(X) == len(y)

    @patch("cloud.deep_learning.model.create_client")
    def test_create_sequences_empty_dataframe(self, mock_create_client, mock_config):
        """Test create_sequences with empty DataFrame."""
        mock_create_client.return_value = MagicMock()

        forecaster = DeepLearningForecaster(mock_config)
        forecaster._feature_names = ["feature1", "feature2"]

        X, y = forecaster.create_sequences(pd.DataFrame(), "target")

        assert len(X) == 0
        assert len(y) == 0

    @patch("cloud.deep_learning.model.create_client")
    def test_create_sequences_missing_target(self, mock_create_client, mock_config, sample_training_data):
        """Test create_sequences with missing target column."""
        mock_create_client.return_value = MagicMock()

        forecaster = DeepLearningForecaster(mock_config)
        df = forecaster.prepare_features(sample_training_data)

        with pytest.raises(ValueError, match="Target column"):
            forecaster.create_sequences(df, "nonexistent.column")

    @patch("cloud.deep_learning.model.create_client")
    def test_build_lstm_model(self, mock_create_client, mock_config):
        """Test LSTM model building."""
        mock_create_client.return_value = MagicMock()

        forecaster = DeepLearningForecaster(mock_config)
        model = forecaster._build_lstm_model(n_features=10)

        assert model is not None
        # Check input shape
        assert model.input_shape == (None, mock_config.sequence_length, 10)
        # Check output shape
        assert model.output_shape == (None, 1)

    @patch("cloud.deep_learning.model.create_client")
    def test_build_transformer_model(self, mock_create_client, mock_config):
        """Test Transformer model building."""
        mock_create_client.return_value = MagicMock()
        mock_config.model_type = "transformer"

        forecaster = DeepLearningForecaster(mock_config)
        model = forecaster._build_transformer_model(n_features=10)

        assert model is not None
        assert model.input_shape == (None, mock_config.sequence_length, 10)
        assert model.output_shape == (None, 1)

    @patch("cloud.deep_learning.model.create_client")
    def test_build_hybrid_model(self, mock_create_client, mock_config):
        """Test Hybrid model building."""
        mock_create_client.return_value = MagicMock()
        mock_config.model_type = "hybrid"

        forecaster = DeepLearningForecaster(mock_config)
        model = forecaster._build_hybrid_model(n_features=10)

        assert model is not None
        assert model.input_shape == (None, mock_config.sequence_length, 10)
        assert model.output_shape == (None, 1)

    @patch("cloud.deep_learning.model.create_client")
    def test_build_model_selects_correct_architecture(self, mock_create_client, mock_config):
        """Test that _build_model selects the correct architecture."""
        mock_create_client.return_value = MagicMock()

        for model_type in ["lstm", "transformer", "hybrid"]:
            mock_config.model_type = model_type
            forecaster = DeepLearningForecaster(mock_config)
            model = forecaster._build_model(n_features=10)
            assert model is not None

    @patch("cloud.deep_learning.model.create_client")
    def test_build_model_unknown_type(self, mock_create_client, mock_config):
        """Test that _build_model raises for unknown type."""
        mock_create_client.return_value = MagicMock()

        forecaster = DeepLearningForecaster(mock_config)
        forecaster.config.model_type = "unknown"  # type: ignore

        with pytest.raises(ValueError, match="Unknown model type"):
            forecaster._build_model(n_features=10)

    @patch("cloud.deep_learning.model.create_client")
    def test_evaluate_model(self, mock_create_client, mock_config):
        """Test model evaluation."""
        mock_create_client.return_value = MagicMock()

        forecaster = DeepLearningForecaster(mock_config)

        # Build a simple model for testing
        model = forecaster._build_lstm_model(n_features=10)

        # Create test data
        np.random.seed(42)
        X_test = np.random.randn(50, mock_config.sequence_length, 10)
        y_test = np.random.randn(50) + 5  # Add offset to avoid zeros

        metrics = forecaster._evaluate_model(model, X_test, y_test)

        assert isinstance(metrics, ModelMetrics)
        assert metrics.mae >= 0
        assert metrics.rmse >= 0
        assert isinstance(metrics.r2, (float, np.floating))
        assert metrics.mape >= 0
        assert metrics.target_mean > 0
        # meets_target can be bool or numpy bool
        assert metrics.meets_target in (True, False)

    @patch("cloud.deep_learning.model.create_client")
    def test_train_all_disabled(self, mock_create_client, disabled_config):
        """Test train_all with disabled config."""
        forecaster = DeepLearningForecaster(disabled_config)
        result = forecaster.train_all()

        assert result["status"] == "disabled"
        assert result["load"] is None
        assert result["solar"] is None

    @patch("cloud.deep_learning.model.create_client")
    def test_train_all_no_data(self, mock_create_client, mock_config):
        """Test train_all when no data is returned."""
        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value.data = []
        mock_create_client.return_value = mock_client

        forecaster = DeepLearningForecaster(mock_config)
        result = forecaster.train_all()

        assert result["status"] == "no_data"

    @patch("cloud.deep_learning.model.create_client")
    def test_get_metrics_no_training(self, mock_create_client, mock_config):
        """Test get_metrics when no models are trained."""
        mock_create_client.return_value = MagicMock()

        forecaster = DeepLearningForecaster(mock_config)
        metrics = forecaster.get_metrics()

        assert metrics["load"] is None
        assert metrics["solar"] is None

    @patch("cloud.deep_learning.model.create_client")
    def test_get_training_history_no_training(self, mock_create_client, mock_config):
        """Test get_training_history when no models are trained."""
        mock_create_client.return_value = MagicMock()

        forecaster = DeepLearningForecaster(mock_config)
        history = forecaster.get_training_history()

        assert history["load"] is None
        assert history["solar"] is None

    @patch("cloud.deep_learning.model.create_client")
    def test_predict_load_no_model(self, mock_create_client, mock_config):
        """Test predict_load raises error when model not trained."""
        mock_create_client.return_value = MagicMock()

        forecaster = DeepLearningForecaster(mock_config)

        with pytest.raises(ValueError, match="Load model not trained"):
            forecaster.predict_load(pd.DataFrame())

    @patch("cloud.deep_learning.model.create_client")
    def test_predict_solar_no_model(self, mock_create_client, mock_config):
        """Test predict_solar raises error when model not trained."""
        mock_create_client.return_value = MagicMock()

        forecaster = DeepLearningForecaster(mock_config)

        with pytest.raises(ValueError, match="Solar model not trained"):
            forecaster.predict_solar(pd.DataFrame())

    @patch("cloud.deep_learning.model.create_client")
    def test_context_manager(self, mock_create_client, mock_config):
        """Test context manager functionality."""
        mock_create_client.return_value = MagicMock()

        with DeepLearningForecaster(mock_config) as forecaster:
            assert forecaster.is_connected()

        # After exit, should be cleaned up
        assert not forecaster.is_connected()

    @patch("cloud.deep_learning.model.create_client")
    def test_close(self, mock_create_client, mock_config):
        """Test close method."""
        mock_create_client.return_value = MagicMock()

        forecaster = DeepLearningForecaster(mock_config)
        assert forecaster.is_connected()

        forecaster.close()
        assert not forecaster.is_connected()


class TestCompareWithBaseline:
    """Tests for compare_with_baseline function."""

    def test_compare_better_dl_model(self):
        """Test comparison when DL model is better."""
        dl_metrics = ModelMetrics(
            mae=0.3,
            rmse=0.4,
            r2=0.95,
            mape=3.0,
            target_mean=10.0,
            mae_percent=3.0,
            meets_target=True,
        )
        baseline_metrics = ModelMetrics(
            mae=0.5,
            rmse=0.6,
            r2=0.85,
            mape=5.0,
            target_mean=10.0,
            mae_percent=5.0,
            meets_target=True,
        )

        result = compare_with_baseline(dl_metrics, baseline_metrics)

        assert result["dl_is_better"] is True
        assert result["meets_target"] is True
        assert result["improvements"]["mae_percent_improvement"] > 0
        assert result["improvements"]["r2_absolute_improvement"] > 0

    def test_compare_worse_dl_model(self):
        """Test comparison when DL model is worse."""
        dl_metrics = ModelMetrics(
            mae=0.8,
            rmse=0.9,
            r2=0.75,
            mape=8.0,
            target_mean=10.0,
            mae_percent=8.0,
            meets_target=False,
        )
        baseline_metrics = ModelMetrics(
            mae=0.5,
            rmse=0.6,
            r2=0.85,
            mape=5.0,
            target_mean=10.0,
            mae_percent=5.0,
            meets_target=True,
        )

        result = compare_with_baseline(dl_metrics, baseline_metrics)

        assert result["dl_is_better"] is False
        assert result["improvements"]["mae_percent_improvement"] < 0
        assert result["improvements"]["r2_absolute_improvement"] < 0

    def test_compare_equal_models(self):
        """Test comparison when models have equal performance."""
        metrics = ModelMetrics(
            mae=0.5,
            rmse=0.6,
            r2=0.85,
            mape=5.0,
            target_mean=10.0,
            mae_percent=5.0,
            meets_target=True,
        )

        result = compare_with_baseline(metrics, metrics)

        assert result["improvements"]["mae_percent_improvement"] == 0
        assert result["improvements"]["r2_absolute_improvement"] == 0

    def test_compare_contains_all_required_keys(self):
        """Test that comparison result contains all required keys."""
        dl_metrics = ModelMetrics(
            mae=0.3,
            rmse=0.4,
            r2=0.95,
            mape=3.0,
            target_mean=10.0,
            mae_percent=3.0,
            meets_target=True,
        )
        baseline_metrics = ModelMetrics(
            mae=0.5,
            rmse=0.6,
            r2=0.85,
            mape=5.0,
            target_mean=10.0,
            mae_percent=5.0,
            meets_target=True,
        )

        result = compare_with_baseline(dl_metrics, baseline_metrics)

        assert "baseline" in result
        assert "deep_learning" in result
        assert "improvements" in result
        assert "dl_is_better" in result
        assert "meets_target" in result

        assert "mae_percent_improvement" in result["improvements"]
        assert "rmse_percent_improvement" in result["improvements"]
        assert "r2_absolute_improvement" in result["improvements"]


class TestDeepLearningTrainingRunner:
    """Tests for the CLI DeepLearningTrainingRunner."""

    @patch("cloud.deep_learning.model.create_client")
    def test_runner_initialization(self, mock_create_client):
        """Test runner initialization with default parameters."""
        mock_create_client.return_value = MagicMock()

        # Import here to avoid import errors before patching
        from run_deep_learning_training import DeepLearningTrainingRunner

        runner = DeepLearningTrainingRunner(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
        )

        assert runner.config.model_type == "lstm"
        assert runner.config.epochs == 100
        assert runner.config.batch_size == 32
        runner.close()

    @patch("cloud.deep_learning.model.create_client")
    def test_runner_custom_parameters(self, mock_create_client):
        """Test runner initialization with custom parameters."""
        mock_create_client.return_value = MagicMock()

        from run_deep_learning_training import DeepLearningTrainingRunner

        runner = DeepLearningTrainingRunner(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
            model_type="transformer",
            epochs=50,
            batch_size=64,
            hidden_units=128,
        )

        assert runner.config.model_type == "transformer"
        assert runner.config.epochs == 50
        assert runner.config.batch_size == 64
        assert runner.config.hidden_units == 128
        runner.close()

    @patch("cloud.deep_learning.model.create_client")
    def test_runner_get_status(self, mock_create_client):
        """Test runner get_status method."""
        mock_create_client.return_value = MagicMock()

        from run_deep_learning_training import DeepLearningTrainingRunner

        runner = DeepLearningTrainingRunner(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
        )

        status = runner.get_status()

        assert "connected" in status
        assert "config" in status
        assert "models" in status
        assert status["connected"] is True
        runner.close()


class TestCLIMain:
    """Tests for the CLI main function."""

    def test_cli_help(self, monkeypatch):
        """Test CLI help output."""
        monkeypatch.setattr("sys.argv", ["run_deep_learning_training.py", "--help"])

        from run_deep_learning_training import main

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 0

    def test_cli_status_requires_credentials(self, monkeypatch):
        """Test CLI status mode requires credentials."""
        monkeypatch.setattr(
            "sys.argv",
            ["run_deep_learning_training.py", "--status"],
        )
        # Clear environment variables
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.delenv("SUPABASE_KEY", raising=False)

        from run_deep_learning_training import main

        exit_code = main()
        # Should fail due to missing credentials
        assert exit_code == 1

    @patch("cloud.deep_learning.model.create_client")
    def test_cli_status_json_output(self, mock_create_client, monkeypatch, capsys):
        """Test CLI status with JSON output."""
        mock_create_client.return_value = MagicMock()

        monkeypatch.setattr(
            "sys.argv",
            [
                "run_deep_learning_training.py",
                "--status",
                "--supabase-url", "https://test.supabase.co",
                "--supabase-key", "test-key",
                "--json",
            ],
        )

        from run_deep_learning_training import main
        import json

        exit_code = main()
        captured = capsys.readouterr()

        assert exit_code == 0
        # Verify JSON output is valid
        output = json.loads(captured.out)
        assert "connected" in output
        assert "config" in output

    def test_cli_invalid_model_type(self, monkeypatch):
        """Test CLI with invalid model type."""
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_deep_learning_training.py",
                "--train",
                "--model-type", "invalid",
            ],
        )

        from run_deep_learning_training import main

        with pytest.raises(SystemExit) as exc_info:
            main()

        # argparse exits with code 2 for invalid arguments
        assert exc_info.value.code == 2

    def test_cli_mutually_exclusive_modes(self, monkeypatch):
        """Test CLI rejects multiple mode flags."""
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_deep_learning_training.py",
                "--train",
                "--evaluate",
            ],
        )

        from run_deep_learning_training import main

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 2
