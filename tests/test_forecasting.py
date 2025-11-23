"""Tests for the baseline forecasting model module."""

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from cloud.forecasting.config import ForecastingConfig
from cloud.forecasting.model import BaselineForecaster, ModelMetrics


class TestForecastingConfig:
    """Tests for ForecastingConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        with patch.dict(os.environ, {}, clear=True):
            config = ForecastingConfig()

            assert config.supabase_url == ""
            assert config.supabase_key == ""
            assert config.enabled is False
            assert config.model_dir == "models"
            assert config.test_size == 0.2
            assert config.random_state == 42
            assert config.horizon_hours == 24
            assert config.target_mae_percent == 10.0
            assert config.n_estimators == 100
            assert config.max_depth == 6
            assert config.learning_rate == 0.1

    def test_explicit_values_override_defaults(self):
        """Test that explicit values override defaults."""
        config = ForecastingConfig(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
            enabled=True,
            model_dir="custom_models",
            test_size=0.3,
            random_state=123,
            horizon_hours=12,
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
        )

        assert config.supabase_url == "https://test.supabase.co"
        assert config.supabase_key == "test-key"
        assert config.enabled is True
        assert config.model_dir == "custom_models"
        assert config.test_size == 0.3
        assert config.random_state == 123
        assert config.horizon_hours == 12
        assert config.n_estimators == 200
        assert config.max_depth == 8
        assert config.learning_rate == 0.05

    def test_env_var_overrides(self):
        """Test environment variable overrides."""
        env_vars = {
            "SUPABASE_URL": "https://env.supabase.co",
            "SUPABASE_KEY": "env-key",
            "FORECAST_SITE_ID": "test-site",
            "FORECAST_MODEL_DIR": "env_models",
            "FORECAST_TEST_SIZE": "0.25",
            "FORECAST_RANDOM_STATE": "99",
            "FORECAST_HORIZON_HOURS": "48",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = ForecastingConfig()

            assert config.supabase_url == "https://env.supabase.co"
            assert config.supabase_key == "env-key"
            assert config.site_id == "test-site"
            assert config.model_dir == "env_models"
            assert config.test_size == 0.25
            assert config.random_state == 99
            assert config.horizon_hours == 48

    def test_explicit_values_override_env_vars(self):
        """Test that explicit values take precedence over env vars."""
        env_vars = {
            "SUPABASE_URL": "https://env.supabase.co",
            "FORECAST_TEST_SIZE": "0.3",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = ForecastingConfig(
                supabase_url="https://explicit.supabase.co",
                test_size=0.15,
            )

            assert config.supabase_url == "https://explicit.supabase.co"
            assert config.test_size == 0.15

    def test_invalid_test_size_env_var(self):
        """Test that invalid FORECAST_TEST_SIZE raises ValueError."""
        with patch.dict(os.environ, {"FORECAST_TEST_SIZE": "not-a-number"}, clear=True):
            with pytest.raises(ValueError, match="FORECAST_TEST_SIZE must be a float"):
                ForecastingConfig()

    def test_invalid_random_state_env_var(self):
        """Test that invalid FORECAST_RANDOM_STATE raises ValueError."""
        with patch.dict(os.environ, {"FORECAST_RANDOM_STATE": "invalid"}, clear=True):
            with pytest.raises(
                ValueError, match="FORECAST_RANDOM_STATE must be an integer"
            ):
                ForecastingConfig()

    def test_validate_missing_url(self):
        """Test validation fails when URL is missing."""
        config = ForecastingConfig(supabase_key="key")
        with pytest.raises(ValueError, match="Supabase URL is required"):
            config.validate()

    def test_validate_missing_key(self):
        """Test validation fails when key is missing."""
        config = ForecastingConfig(supabase_url="https://test.supabase.co")
        with pytest.raises(ValueError, match="Supabase key is required"):
            config.validate()

    def test_validate_invalid_test_size(self):
        """Test validation fails for invalid test_size."""
        config = ForecastingConfig(
            supabase_url="https://test.supabase.co",
            supabase_key="key",
            test_size=0.0,
        )
        with pytest.raises(ValueError, match="test_size must be between"):
            config.validate()

    def test_validate_invalid_horizon_hours(self):
        """Test validation fails for non-positive horizon_hours."""
        config = ForecastingConfig(
            supabase_url="https://test.supabase.co",
            supabase_key="key",
            horizon_hours=0,
        )
        with pytest.raises(ValueError, match="horizon_hours must be positive"):
            config.validate()

    def test_validate_invalid_n_estimators(self):
        """Test validation fails for non-positive n_estimators."""
        config = ForecastingConfig(
            supabase_url="https://test.supabase.co",
            supabase_key="key",
            n_estimators=0,
        )
        with pytest.raises(ValueError, match="n_estimators must be positive"):
            config.validate()

    def test_get_model_path(self):
        """Test get_model_path returns correct path."""
        config = ForecastingConfig(model_dir="my_models")
        load_path = config.get_model_path("load")
        solar_path = config.get_model_path("solar")

        assert load_path == Path("my_models/baseline_load_forecaster.joblib")
        assert solar_path == Path("my_models/baseline_solar_forecaster.joblib")

    def test_get_all_features(self):
        """Test get_all_features returns all configured features."""
        config = ForecastingConfig()
        features = config.get_all_features()

        # Check it includes features from all categories
        assert "hour_of_day" in features  # Temporal
        assert "tou_peak" in features  # Categorical
        assert "solar_generation_kw_rolling_avg_7d" in features  # Rolling
        assert "solar_generation_kw_lag_1" in features  # Lag
        assert "solar.irradiance_w_m2" in features  # Metric

    def test_target_columns_defined(self):
        """Test that target columns are defined."""
        assert ForecastingConfig.TARGET_LOAD == "home_load.total_load_kw"
        assert ForecastingConfig.TARGET_SOLAR == "solar.generation_kw"


class TestModelMetrics:
    """Tests for ModelMetrics dataclass."""

    def test_metrics_creation(self):
        """Test creating ModelMetrics."""
        metrics = ModelMetrics(
            mae=0.5,
            rmse=0.7,
            r2=0.85,
            mape=5.0,
            target_mean=10.0,
            mae_percent=5.0,
            meets_target=True,
        )

        assert metrics.mae == 0.5
        assert metrics.rmse == 0.7
        assert metrics.r2 == 0.85
        assert metrics.mape == 5.0
        assert metrics.target_mean == 10.0
        assert metrics.mae_percent == 5.0
        assert metrics.meets_target is True

    def test_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = ModelMetrics(
            mae=0.5,
            rmse=0.7,
            r2=0.85,
            mape=5.0,
            target_mean=10.0,
            mae_percent=5.0,
            meets_target=True,
        )

        result = metrics.to_dict()

        assert isinstance(result, dict)
        assert result["mae"] == 0.5
        assert result["rmse"] == 0.7
        assert result["r2"] == 0.85
        assert result["mape"] == 5.0
        assert result["target_mean"] == 10.0
        assert result["mae_percent"] == 5.0
        assert result["meets_target"] is True


class TestBaselineForecaster:
    """Tests for BaselineForecaster."""

    @pytest.fixture
    def disabled_config(self):
        """Create a disabled configuration."""
        return ForecastingConfig(enabled=False)

    @pytest.fixture
    def mock_config(self):
        """Create a mock enabled configuration."""
        return ForecastingConfig(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
            enabled=True,
            model_dir=tempfile.mkdtemp(),
            test_size=0.2,
            n_estimators=10,  # Small for fast tests
            max_depth=3,
        )

    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data with all required features."""
        np.random.seed(42)
        n_samples = 1000  # Enough for train/test split

        # Create time series
        base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        times = pd.date_range(start=base_time, periods=n_samples, freq="5min")

        data = {
            "time": times,
            "site_id": ["site-1"] * n_samples,
            "device_id": ["device-1"] * n_samples,
            # Targets (with some pattern + noise)
            "home_load.total_load_kw": 2.0 + 0.5 * np.sin(np.linspace(0, 20*np.pi, n_samples)) + np.random.normal(0, 0.2, n_samples),
            "solar.generation_kw": np.maximum(0, 3.0 * np.sin(np.linspace(0, 20*np.pi, n_samples)) + np.random.normal(0, 0.3, n_samples)),
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
            "solar_generation_kw_lag_1": np.roll(np.maximum(0, 3.0 * np.sin(np.linspace(0, 20*np.pi, n_samples))), 1),
            "solar_generation_kw_lag_1h": np.roll(np.maximum(0, 3.0 * np.sin(np.linspace(0, 20*np.pi, n_samples))), 12),
            "home_load_total_load_kw_lag_1": np.roll(2.0 + 0.5 * np.sin(np.linspace(0, 20*np.pi, n_samples)), 1),
            "home_load_total_load_kw_lag_1h": np.roll(2.0 + 0.5 * np.sin(np.linspace(0, 20*np.pi, n_samples)), 12),
            "grid_price_price_per_kwh_lag_1": np.random.uniform(0.1, 0.3, n_samples),
            "grid_price_price_per_kwh_lag_1h": np.random.uniform(0.1, 0.3, n_samples),
            # Metric features
            "solar.irradiance_w_m2": np.random.uniform(0, 1000, n_samples),
            "solar.panel_temp_celsius": np.random.uniform(20, 45, n_samples),
            "battery.soc_percent": np.random.uniform(20, 80, n_samples),
            "grid_price.price_per_kwh": np.random.uniform(0.1, 0.3, n_samples),
        }

        return pd.DataFrame(data)

    def test_disabled_forecaster_not_connected(self, disabled_config):
        """Test that disabled forecaster is not connected."""
        forecaster = BaselineForecaster(disabled_config)
        assert not forecaster.is_connected()

    def test_disabled_forecaster_train_returns_disabled(self, disabled_config):
        """Test that disabled forecaster returns disabled status."""
        forecaster = BaselineForecaster(disabled_config)
        result = forecaster.train_all()

        assert result["status"] == "disabled"
        assert result["load"] is None
        assert result["solar"] is None

    @patch("cloud.forecasting.model.create_client")
    def test_forecaster_connects_when_enabled(self, mock_create_client, mock_config):
        """Test that enabled forecaster establishes connection."""
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client

        forecaster = BaselineForecaster(mock_config)

        assert forecaster.is_connected()
        mock_create_client.assert_called_once_with(
            mock_config.supabase_url,
            mock_config.supabase_key,
        )

    def test_prepare_features_empty_dataframe(self, disabled_config):
        """Test prepare_features with empty DataFrame."""
        forecaster = BaselineForecaster(disabled_config)
        result = forecaster.prepare_features(pd.DataFrame())

        assert result.empty

    def test_prepare_features_normalizes_column_names(
        self, disabled_config, sample_training_data
    ):
        """Test that prepare_features normalizes column names."""
        forecaster = BaselineForecaster(disabled_config)
        result = forecaster.prepare_features(sample_training_data)

        # Check that dots are replaced with underscores
        assert "home_load_total_load_kw" in result.columns
        assert "solar_generation_kw" in result.columns

    def test_create_target_variable_empty_dataframe(self, disabled_config):
        """Test create_target_variable with empty DataFrame."""
        forecaster = BaselineForecaster(disabled_config)
        forecaster._feature_names = ["hour_of_day"]

        X, y = forecaster.create_target_variable(pd.DataFrame(), "home_load.total_load_kw")

        assert X.empty
        assert len(y) == 0

    def test_create_target_variable_missing_target(self, disabled_config, sample_training_data):
        """Test create_target_variable raises error for missing target."""
        forecaster = BaselineForecaster(disabled_config)
        forecaster.prepare_features(sample_training_data)

        with pytest.raises(ValueError, match="Target column.*not found"):
            forecaster.create_target_variable(
                sample_training_data, "nonexistent_column"
            )

    @patch("cloud.forecasting.model.create_client")
    def test_train_load_model(
        self, mock_create_client, mock_config, sample_training_data
    ):
        """Test training the load forecasting model."""
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client

        forecaster = BaselineForecaster(mock_config)

        # Prepare features
        df = forecaster.prepare_features(sample_training_data)

        # Train model
        result = forecaster.train_load_model(df)

        assert result["target"] == mock_config.TARGET_LOAD
        assert result["train_samples"] > 0
        assert result["test_samples"] > 0
        assert "metrics" in result
        assert "mae" in result["metrics"]
        assert "rmse" in result["metrics"]
        assert "r2" in result["metrics"]
        assert "mae_percent" in result["metrics"]
        assert "meets_target" in result["metrics"]
        assert "feature_importance" in result
        assert forecaster._load_model is not None

    @patch("cloud.forecasting.model.create_client")
    def test_train_solar_model(
        self, mock_create_client, mock_config, sample_training_data
    ):
        """Test training the solar forecasting model."""
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client

        forecaster = BaselineForecaster(mock_config)

        # Prepare features
        df = forecaster.prepare_features(sample_training_data)

        # Train model
        result = forecaster.train_solar_model(df)

        assert result["target"] == mock_config.TARGET_SOLAR
        assert result["train_samples"] > 0
        assert result["test_samples"] > 0
        assert "metrics" in result
        assert "mae" in result["metrics"]
        assert forecaster._solar_model is not None

    @patch("cloud.forecasting.model.create_client")
    def test_save_and_load_models(
        self, mock_create_client, mock_config, sample_training_data
    ):
        """Test saving and loading models."""
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client

        forecaster = BaselineForecaster(mock_config)

        # Prepare features and train
        df = forecaster.prepare_features(sample_training_data)
        forecaster.train_load_model(df)
        forecaster.train_solar_model(df)

        # Save models
        saved = forecaster.save_models()

        assert "load" in saved
        assert "solar" in saved
        assert Path(saved["load"]).exists()
        assert Path(saved["solar"]).exists()

        # Create new forecaster and load models
        forecaster2 = BaselineForecaster(mock_config)
        loaded = forecaster2.load_models()

        assert loaded["load"] is True
        assert loaded["solar"] is True
        assert forecaster2._load_model is not None
        assert forecaster2._solar_model is not None

    @patch("cloud.forecasting.model.create_client")
    def test_predict_load(
        self, mock_create_client, mock_config, sample_training_data
    ):
        """Test making predictions with the load model."""
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client

        forecaster = BaselineForecaster(mock_config)

        # Prepare features and train
        df = forecaster.prepare_features(sample_training_data)
        forecaster.train_load_model(df)

        # Make predictions on sample data
        predictions = forecaster.predict_load(sample_training_data.head(10))

        assert len(predictions) == 10
        assert isinstance(predictions, np.ndarray)

    @patch("cloud.forecasting.model.create_client")
    def test_predict_solar(
        self, mock_create_client, mock_config, sample_training_data
    ):
        """Test making predictions with the solar model."""
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client

        forecaster = BaselineForecaster(mock_config)

        # Prepare features and train
        df = forecaster.prepare_features(sample_training_data)
        forecaster.train_solar_model(df)

        # Make predictions on sample data
        predictions = forecaster.predict_solar(sample_training_data.head(10))

        assert len(predictions) == 10
        assert isinstance(predictions, np.ndarray)

    @patch("cloud.forecasting.model.create_client")
    def test_predict_load_without_model_raises_error(
        self, mock_create_client, mock_config, sample_training_data
    ):
        """Test that predicting without a trained model raises error."""
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client

        forecaster = BaselineForecaster(mock_config)
        forecaster._feature_names = ["hour_of_day"]

        with pytest.raises(ValueError, match="Load model not trained"):
            forecaster.predict_load(sample_training_data.head(10))

    @patch("cloud.forecasting.model.create_client")
    def test_get_metrics(
        self, mock_create_client, mock_config, sample_training_data
    ):
        """Test getting metrics from trained models."""
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client

        forecaster = BaselineForecaster(mock_config)

        # Initially no metrics
        metrics = forecaster.get_metrics()
        assert metrics["load"] is None
        assert metrics["solar"] is None

        # Train models
        df = forecaster.prepare_features(sample_training_data)
        forecaster.train_load_model(df)
        forecaster.train_solar_model(df)

        # Now metrics should be available
        metrics = forecaster.get_metrics()
        assert metrics["load"] is not None
        assert metrics["solar"] is not None
        assert "mae" in metrics["load"]
        assert "mae" in metrics["solar"]

    def test_context_manager_closes_connection(self, disabled_config):
        """Test that context manager properly closes connection."""
        with BaselineForecaster(disabled_config) as forecaster:
            pass  # Just test that context manager works

        assert not forecaster.is_connected()

    @patch("cloud.forecasting.model.create_client")
    def test_train_insufficient_data_raises_error(
        self, mock_create_client, mock_config
    ):
        """Test that training with insufficient data raises error."""
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client

        forecaster = BaselineForecaster(mock_config)

        # Create very small dataset
        small_data = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=50, freq="5min"),
            "home_load.total_load_kw": np.random.uniform(1, 3, 50),
            "hour_of_day": list(range(50)),
        })

        df = forecaster.prepare_features(small_data)

        with pytest.raises(ValueError, match="Insufficient data"):
            forecaster.train_load_model(df)


class TestModelTrainingRunner:
    """Tests for the CLI runner."""

    @patch("run_model_training.BaselineForecaster")
    @patch("run_model_training.ForecastingConfig")
    def test_runner_initialization(self, mock_config_class, mock_forecaster_class):
        """Test runner initializes correctly."""
        from run_model_training import ModelTrainingRunner

        mock_config = MagicMock()
        mock_config_class.return_value = mock_config

        mock_forecaster = MagicMock()
        mock_forecaster_class.return_value = mock_forecaster

        ModelTrainingRunner(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
            model_dir="test_models",
        )

        mock_config_class.assert_called_once()
        mock_forecaster_class.assert_called_once()

    @patch("run_model_training.BaselineForecaster")
    @patch("run_model_training.ForecastingConfig")
    def test_runner_train(self, mock_config_class, mock_forecaster_class):
        """Test runner train method."""
        from run_model_training import ModelTrainingRunner

        mock_config = MagicMock()
        mock_config.target_mae_percent = 10.0
        mock_config.model_dir = "models"
        mock_config_class.return_value = mock_config

        mock_forecaster = MagicMock()
        mock_forecaster.train_all.return_value = {
            "status": "success",
            "load": {
                "metrics": {
                    "mae": 0.2,
                    "rmse": 0.3,
                    "r2": 0.9,
                    "mae_percent": 5.0,
                    "meets_target": True,
                }
            },
            "solar": {
                "metrics": {
                    "mae": 0.3,
                    "rmse": 0.4,
                    "r2": 0.85,
                    "mae_percent": 8.0,
                    "meets_target": True,
                }
            },
        }
        mock_forecaster.save_models.return_value = {
            "load": "models/load.joblib",
            "solar": "models/solar.joblib",
        }
        mock_forecaster_class.return_value = mock_forecaster

        runner = ModelTrainingRunner(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
        )

        result = runner.train()

        assert result["status"] == "success"
        mock_forecaster.train_all.assert_called_once()
        mock_forecaster.save_models.assert_called_once()

    @patch("run_model_training.BaselineForecaster")
    @patch("run_model_training.ForecastingConfig")
    def test_runner_get_status(self, mock_config_class, mock_forecaster_class):
        """Test runner get_status method."""
        from run_model_training import ModelTrainingRunner

        mock_config = MagicMock()
        mock_config.model_dir = "models"
        mock_config.test_size = 0.2
        mock_config.horizon_hours = 24
        mock_config.target_mae_percent = 10.0
        mock_config.n_estimators = 100
        mock_config.max_depth = 6
        mock_config.learning_rate = 0.1
        mock_config.get_model_path.side_effect = lambda x: Path(f"models/baseline_{x}_forecaster.joblib")
        mock_config_class.return_value = mock_config

        mock_forecaster = MagicMock()
        mock_forecaster.is_connected.return_value = True
        mock_forecaster_class.return_value = mock_forecaster

        runner = ModelTrainingRunner(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
        )

        status = runner.get_status()

        assert status["connected"] is True
        assert "config" in status
        assert "models" in status
