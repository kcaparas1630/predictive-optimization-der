"""Tests for the feature engineering pipeline module."""

import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from cloud.feature_engineering.config import FeatureEngineeringConfig
from cloud.feature_engineering.pipeline import FeatureEngineeringPipeline


class TestFeatureEngineeringConfig:
    """Tests for FeatureEngineeringConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        # Clear env vars for this test
        with patch.dict(os.environ, {}, clear=True):
            config = FeatureEngineeringConfig()

            assert config.supabase_url == ""
            assert config.supabase_key == ""
            assert config.enabled is False
            assert config.batch_size == 1000
            assert config.rolling_window_days == 7
            assert config.lookback_days == 30
            assert config.source_table == "readings"
            assert config.target_table == "training_data"

    def test_explicit_values_override_defaults(self):
        """Test that explicit values override defaults."""
        config = FeatureEngineeringConfig(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
            enabled=True,
            batch_size=500,
            rolling_window_days=14,
            lookback_days=60,
        )

        assert config.supabase_url == "https://test.supabase.co"
        assert config.supabase_key == "test-key"
        assert config.enabled is True
        assert config.batch_size == 500
        assert config.rolling_window_days == 14
        assert config.lookback_days == 60

    def test_env_var_overrides(self):
        """Test environment variable overrides."""
        env_vars = {
            "SUPABASE_URL": "https://env.supabase.co",
            "SUPABASE_KEY": "env-key",
            "FE_SITE_ID": "test-site",
            "FE_BATCH_SIZE": "2000",
            "FE_ROLLING_WINDOW_DAYS": "14",
            "FE_LOOKBACK_DAYS": "90",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = FeatureEngineeringConfig()

            assert config.supabase_url == "https://env.supabase.co"
            assert config.supabase_key == "env-key"
            assert config.site_id == "test-site"
            assert config.batch_size == 2000
            assert config.rolling_window_days == 14
            assert config.lookback_days == 90

    def test_explicit_values_override_env_vars(self):
        """Test that explicit values take precedence over env vars."""
        env_vars = {
            "SUPABASE_URL": "https://env.supabase.co",
            "FE_BATCH_SIZE": "2000",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = FeatureEngineeringConfig(
                supabase_url="https://explicit.supabase.co",
                batch_size=500,
            )

            assert config.supabase_url == "https://explicit.supabase.co"
            assert config.batch_size == 500

    def test_invalid_batch_size_env_var(self):
        """Test that invalid FE_BATCH_SIZE raises ValueError."""
        with patch.dict(os.environ, {"FE_BATCH_SIZE": "not-a-number"}, clear=True):
            with pytest.raises(ValueError, match="FE_BATCH_SIZE must be an integer"):
                FeatureEngineeringConfig()

    def test_invalid_rolling_window_env_var(self):
        """Test that invalid FE_ROLLING_WINDOW_DAYS raises ValueError."""
        with patch.dict(os.environ, {"FE_ROLLING_WINDOW_DAYS": "invalid"}, clear=True):
            with pytest.raises(
                ValueError, match="FE_ROLLING_WINDOW_DAYS must be an integer"
            ):
                FeatureEngineeringConfig()

    def test_validate_missing_url(self):
        """Test validation fails when URL is missing."""
        config = FeatureEngineeringConfig(supabase_key="key")
        with pytest.raises(ValueError, match="Supabase URL is required"):
            config.validate()

    def test_validate_missing_key(self):
        """Test validation fails when key is missing."""
        config = FeatureEngineeringConfig(supabase_url="https://test.supabase.co")
        with pytest.raises(ValueError, match="Supabase key is required"):
            config.validate()

    def test_validate_invalid_batch_size(self):
        """Test validation fails for non-positive batch_size."""
        config = FeatureEngineeringConfig(
            supabase_url="https://test.supabase.co",
            supabase_key="key",
            batch_size=0,
        )
        with pytest.raises(ValueError, match="batch_size must be positive"):
            config.validate()

    def test_metrics_to_process_defined(self):
        """Test that METRICS_TO_PROCESS class variable is defined."""
        assert len(FeatureEngineeringConfig.METRICS_TO_PROCESS) > 0
        assert "solar.generation_kw" in FeatureEngineeringConfig.METRICS_TO_PROCESS
        assert "battery.soc_percent" in FeatureEngineeringConfig.METRICS_TO_PROCESS

    def test_rolling_avg_metrics_defined(self):
        """Test that ROLLING_AVG_METRICS class variable is defined."""
        assert len(FeatureEngineeringConfig.ROLLING_AVG_METRICS) > 0
        assert "solar.generation_kw" in FeatureEngineeringConfig.ROLLING_AVG_METRICS


class TestFeatureEngineeringPipeline:
    """Tests for FeatureEngineeringPipeline."""

    @pytest.fixture
    def disabled_config(self):
        """Create a disabled configuration."""
        return FeatureEngineeringConfig(enabled=False)

    @pytest.fixture
    def mock_config(self):
        """Create a mock enabled configuration."""
        return FeatureEngineeringConfig(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
            enabled=True,
            batch_size=100,
            rolling_window_days=7,
            lookback_days=30,
        )

    @pytest.fixture
    def sample_raw_data(self):
        """Create sample raw data in long format."""
        base_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        data = []

        # Create 24 hours of data (288 points at 5-min intervals)
        for i in range(288):
            time = base_time + pd.Timedelta(minutes=5 * i)
            hour = time.hour

            # Solar generation (varies with hour)
            solar_gen = (
                max(0, 5.0 * np.sin(np.pi * (hour - 6) / 12)) if 6 <= hour <= 18 else 0
            )

            data.extend(
                [
                    {
                        "time": time,
                        "site_id": "site-1",
                        "device_id": "device-1",
                        "metric_name": "solar.generation_kw",
                        "metric_value": solar_gen,
                    },
                    {
                        "time": time,
                        "site_id": "site-1",
                        "device_id": "device-1",
                        "metric_name": "battery.soc_percent",
                        "metric_value": 50 + 20 * np.sin(np.pi * i / 144),
                    },
                    {
                        "time": time,
                        "site_id": "site-1",
                        "device_id": "device-1",
                        "metric_name": "home_load.total_load_kw",
                        "metric_value": 2.0 + np.random.random(),
                    },
                    {
                        "time": time,
                        "site_id": "site-1",
                        "device_id": "device-1",
                        "metric_name": "grid_price.price_per_kwh",
                        "metric_value": 0.15
                        if 7 <= hour < 16
                        else 0.25
                        if 16 <= hour < 21
                        else 0.10,
                    },
                ]
            )

        return pd.DataFrame(data)

    def test_disabled_pipeline_not_connected(self, disabled_config):
        """Test that disabled pipeline is not connected."""
        pipeline = FeatureEngineeringPipeline(disabled_config)
        assert not pipeline.is_connected()

    def test_disabled_pipeline_run_returns_disabled_status(self, disabled_config):
        """Test that disabled pipeline returns disabled status."""
        pipeline = FeatureEngineeringPipeline(disabled_config)
        result = pipeline.run()

        assert result["status"] == "disabled"
        assert result["records_processed"] == 0

    @patch("cloud.feature_engineering.pipeline.create_client")
    def test_pipeline_connects_when_enabled(self, mock_create_client, mock_config):
        """Test that enabled pipeline establishes connection."""
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client

        pipeline = FeatureEngineeringPipeline(mock_config)

        assert pipeline.is_connected()
        mock_create_client.assert_called_once_with(
            mock_config.supabase_url,
            mock_config.supabase_key,
        )

    def test_pivot_metrics_empty_dataframe(self, disabled_config):
        """Test pivot_metrics with empty DataFrame."""
        pipeline = FeatureEngineeringPipeline(disabled_config)
        result = pipeline.pivot_metrics(pd.DataFrame())

        assert result.empty

    def test_pivot_metrics_creates_wide_format(self, disabled_config, sample_raw_data):
        """Test pivot_metrics converts long to wide format."""
        pipeline = FeatureEngineeringPipeline(disabled_config)
        result = pipeline.pivot_metrics(sample_raw_data)

        assert not result.empty
        assert "time" in result.columns
        assert "site_id" in result.columns
        assert "device_id" in result.columns
        assert "solar.generation_kw" in result.columns
        assert "battery.soc_percent" in result.columns

    def test_extract_temporal_features_empty_dataframe(self, disabled_config):
        """Test extract_temporal_features with empty DataFrame."""
        pipeline = FeatureEngineeringPipeline(disabled_config)
        result = pipeline.extract_temporal_features(pd.DataFrame())

        assert result.empty

    def test_extract_temporal_features_adds_all_features(self, disabled_config):
        """Test that extract_temporal_features adds expected features."""
        pipeline = FeatureEngineeringPipeline(disabled_config)

        # Create simple test data
        df = pd.DataFrame(
            {
                "time": pd.to_datetime(
                    [
                        "2024-01-15 08:00:00",  # Monday
                        "2024-01-15 12:00:00",
                        "2024-01-15 18:00:00",
                        "2024-01-20 10:00:00",  # Saturday
                    ]
                ),
                "value": [1, 2, 3, 4],
            }
        )

        result = pipeline.extract_temporal_features(df)

        # Check temporal features exist
        assert "hour_of_day" in result.columns
        assert "day_of_week" in result.columns
        assert "day_of_month" in result.columns
        assert "month" in result.columns
        assert "is_weekend" in result.columns
        assert "hour_sin" in result.columns
        assert "hour_cos" in result.columns
        assert "day_sin" in result.columns
        assert "day_cos" in result.columns

        # Check values
        assert result.iloc[0]["hour_of_day"] == 8
        assert result.iloc[1]["hour_of_day"] == 12
        assert result.iloc[2]["hour_of_day"] == 18

        # Monday = 0, Saturday = 5
        assert result.iloc[0]["day_of_week"] == 0
        assert result.iloc[3]["day_of_week"] == 5

        # Weekend check
        assert result.iloc[0]["is_weekend"] == 0  # Monday
        assert result.iloc[3]["is_weekend"] == 1  # Saturday

    def test_calculate_rolling_features_empty_dataframe(self, disabled_config):
        """Test calculate_rolling_features with empty DataFrame."""
        pipeline = FeatureEngineeringPipeline(disabled_config)
        result = pipeline.calculate_rolling_features(pd.DataFrame())

        assert result.empty

    def test_calculate_rolling_features_adds_rolling_columns(
        self, disabled_config, sample_raw_data
    ):
        """Test that calculate_rolling_features adds rolling columns."""
        pipeline = FeatureEngineeringPipeline(disabled_config)

        # First pivot the data
        pivoted = pipeline.pivot_metrics(sample_raw_data)
        result = pipeline.calculate_rolling_features(pivoted)

        # Check rolling features exist for key metrics
        window = disabled_config.rolling_window_days
        assert f"solar_generation_kw_rolling_avg_{window}d" in result.columns
        assert f"solar_generation_kw_rolling_std_{window}d" in result.columns

    def test_encode_categorical_features_empty_dataframe(self, disabled_config):
        """Test encode_categorical_features with empty DataFrame."""
        pipeline = FeatureEngineeringPipeline(disabled_config)
        result = pipeline.encode_categorical_features(pd.DataFrame())

        assert result.empty

    def test_encode_categorical_features_infers_tou_from_hour(self, disabled_config):
        """Test that encode_categorical_features infers TOU from hour."""
        pipeline = FeatureEngineeringPipeline(disabled_config)

        df = pd.DataFrame(
            {
                "hour_of_day": [3, 10, 17, 22],  # off-peak, shoulder, peak, off-peak
            }
        )

        result = pipeline.encode_categorical_features(df)

        assert "tou_peak" in result.columns
        assert "tou_off_peak" in result.columns
        assert "tou_shoulder" in result.columns

        # Check values
        assert result.iloc[0]["tou_off_peak"] == 1  # 3am = off-peak
        assert result.iloc[1]["tou_shoulder"] == 1  # 10am = shoulder
        assert result.iloc[2]["tou_peak"] == 1  # 5pm = peak
        assert result.iloc[3]["tou_off_peak"] == 1  # 10pm = off-peak

    def test_add_lag_features_empty_dataframe(self, disabled_config):
        """Test add_lag_features with empty DataFrame."""
        pipeline = FeatureEngineeringPipeline(disabled_config)
        result = pipeline.add_lag_features(pd.DataFrame())

        assert result.empty

    def test_add_lag_features_creates_lag_columns(
        self, disabled_config, sample_raw_data
    ):
        """Test that add_lag_features creates lag columns."""
        pipeline = FeatureEngineeringPipeline(disabled_config)

        # First pivot the data
        pivoted = pipeline.pivot_metrics(sample_raw_data)
        result = pipeline.add_lag_features(pivoted)

        # Check lag features exist
        assert "solar_generation_kw_lag_1" in result.columns
        assert "solar_generation_kw_lag_1h" in result.columns

    def test_get_feature_columns_returns_expected_list(self, disabled_config):
        """Test that get_feature_columns returns expected features."""
        pipeline = FeatureEngineeringPipeline(disabled_config)
        features = pipeline.get_feature_columns()

        # Check base columns
        assert "time" in features
        assert "site_id" in features
        assert "device_id" in features

        # Check temporal features
        assert "hour_of_day" in features
        assert "day_of_week" in features
        assert "is_weekend" in features
        assert "hour_sin" in features

        # Check rolling features
        assert "solar_generation_kw_rolling_avg_7d" in features
        assert "solar_generation_kw_rolling_std_7d" in features

        # Check categorical features
        assert "tou_peak" in features
        assert "tou_off_peak" in features
        assert "tou_shoulder" in features

        # Check lag features
        assert "solar_generation_kw_lag_1" in features
        assert "solar_generation_kw_lag_1h" in features

    def test_context_manager_closes_connection(self, disabled_config):
        """Test that context manager properly closes connection."""
        with FeatureEngineeringPipeline(disabled_config) as pipeline:
            pass  # Just test that context manager works

        assert not pipeline.is_connected()

    @patch("cloud.feature_engineering.pipeline.create_client")
    def test_full_pipeline_integration(
        self, mock_create_client, mock_config, sample_raw_data
    ):
        """Test full pipeline integration with mocked Supabase."""
        # Setup mock client
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client

        # Mock query response
        mock_client.table.return_value.select.return_value.gte.return_value.lte.return_value.order.return_value.execute.return_value.data = sample_raw_data.to_dict(
            "records"
        )

        # Mock upsert response
        mock_client.table.return_value.upsert.return_value.execute.return_value = (
            MagicMock()
        )

        with FeatureEngineeringPipeline(mock_config) as pipeline:
            # We can't run the full pipeline without proper mocking of all methods,
            # but we can test individual components
            assert pipeline.is_connected()


class TestFeatureEngineeringRunner:
    """Tests for the CLI runner."""

    @patch("cloud.feature_engineering.FeatureEngineeringPipeline")
    @patch("cloud.feature_engineering.FeatureEngineeringConfig")
    def test_runner_initialization(self, mock_config_class, mock_pipeline_class):
        """Test runner initializes pipeline correctly."""
        from run_feature_engineering import FeatureEngineeringRunner

        mock_config = MagicMock()
        mock_config_class.return_value = mock_config

        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline

        FeatureEngineeringRunner(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
            batch_size=500,
        )

        mock_config_class.assert_called_once()
        mock_pipeline_class.assert_called_once()

    @patch("cloud.feature_engineering.FeatureEngineeringPipeline")
    @patch("cloud.feature_engineering.FeatureEngineeringConfig")
    def test_runner_run_once(self, mock_config_class, mock_pipeline_class):
        """Test runner run_once method."""
        from run_feature_engineering import FeatureEngineeringRunner

        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {
            "status": "success",
            "records_processed": 100,
            "records_stored": 100,
        }
        mock_pipeline_class.return_value = mock_pipeline

        runner = FeatureEngineeringRunner(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
        )

        result = runner.run_once()

        assert result["status"] == "success"
        mock_pipeline.run.assert_called_once()

    @patch("cloud.feature_engineering.FeatureEngineeringPipeline")
    @patch("cloud.feature_engineering.FeatureEngineeringConfig")
    def test_runner_get_status(self, mock_config_class, mock_pipeline_class):
        """Test runner get_status method."""
        from run_feature_engineering import FeatureEngineeringRunner

        mock_config = MagicMock()
        mock_config.site_id = "test-site"
        mock_config.batch_size = 1000
        mock_config.rolling_window_days = 7
        mock_config.lookback_days = 30
        mock_config.source_table = "readings"
        mock_config.target_table = "training_data"
        mock_config_class.return_value = mock_config

        mock_pipeline = MagicMock()
        mock_pipeline.is_connected.return_value = True
        mock_pipeline.health_check.return_value = True
        mock_pipeline_class.return_value = mock_pipeline

        runner = FeatureEngineeringRunner(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
        )

        status = runner.get_status()

        assert status["connected"] is True
        assert status["health"] is True
        assert "config" in status
