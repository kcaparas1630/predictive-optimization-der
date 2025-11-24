"""Configuration for the forecasting model pipeline.

This module provides configuration management for the baseline forecasting
model, following the same patterns as the feature engineering module.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


@dataclass
class ForecastingConfig:
    """Configuration for baseline forecasting model.

    Supports environment variable overrides:
    - SUPABASE_URL: Supabase project URL
    - SUPABASE_KEY: Supabase API key
    - FORECAST_MODEL_DIR: Directory to save trained models
    - FORECAST_TEST_SIZE: Fraction of data for testing (0.0-1.0)
    - FORECAST_RANDOM_STATE: Random seed for reproducibility
    - FORECAST_HORIZON_HOURS: Prediction horizon in hours

    Attributes:
        supabase_url: Supabase project URL
        supabase_key: Supabase API key
        enabled: Whether forecasting is enabled
        model_dir: Directory to save trained models
        test_size: Fraction of data for testing (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
        horizon_hours: Prediction horizon in hours (default: 24)
        target_mae_percent: Target MAE as percentage of mean (default: 10.0)
        source_table: Source table for training data (default: training_data)
    """

    # Supabase connection settings
    supabase_url: str = ""
    supabase_key: str = ""
    enabled: bool = False

    # Model settings
    model_dir: str = "models"
    test_size: float = 0.2
    random_state: int = 42
    horizon_hours: int = 24

    # Performance targets
    target_mae_percent: float = 10.0

    # Data settings
    source_table: str = "training_data"
    site_id: str = ""

    # Model hyperparameters for Gradient Boosting
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    min_samples_split: int = 10
    min_samples_leaf: int = 5

    # Target columns (metrics to predict)
    TARGET_LOAD: ClassVar[str] = "home_load.total_load_kw"
    TARGET_SOLAR: ClassVar[str] = "solar.generation_kw"

    # Features to use for prediction
    # Temporal features
    TEMPORAL_FEATURES: ClassVar[list[str]] = [
        "hour_of_day",
        "day_of_week",
        "day_of_month",
        "month",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
    ]

    # Categorical features (one-hot encoded)
    CATEGORICAL_FEATURES: ClassVar[list[str]] = [
        "tou_peak",
        "tou_off_peak",
        "tou_shoulder",
    ]

    # Rolling features (7-day windows)
    ROLLING_FEATURES: ClassVar[list[str]] = [
        "solar_generation_kw_rolling_avg_7d",
        "solar_generation_kw_rolling_std_7d",
        "home_load_total_load_kw_rolling_avg_7d",
        "home_load_total_load_kw_rolling_std_7d",
        "battery_soc_percent_rolling_avg_7d",
        "battery_soc_percent_rolling_std_7d",
        "grid_price_price_per_kwh_rolling_avg_7d",
        "grid_price_price_per_kwh_rolling_std_7d",
        "system_net_grid_flow_kw_rolling_avg_7d",
        "system_net_grid_flow_kw_rolling_std_7d",
    ]

    # Lag features
    LAG_FEATURES: ClassVar[list[str]] = [
        "solar_generation_kw_lag_1",
        "solar_generation_kw_lag_1h",
        "home_load_total_load_kw_lag_1",
        "home_load_total_load_kw_lag_1h",
        "grid_price_price_per_kwh_lag_1",
        "grid_price_price_per_kwh_lag_1h",
    ]

    # Additional numeric features from raw metrics
    METRIC_FEATURES: ClassVar[list[str]] = [
        "solar.irradiance_w_m2",
        "solar.panel_temp_celsius",
        "battery.soc_percent",
        "grid_price.price_per_kwh",
    ]

    def __post_init__(self) -> None:
        """Apply environment variable overrides only when values are at defaults.

        Precedence: explicit args > env vars > defaults
        """
        if self.supabase_url == "":
            self.supabase_url = os.environ.get("SUPABASE_URL", self.supabase_url)
        if self.supabase_key == "":
            self.supabase_key = os.environ.get("SUPABASE_KEY", self.supabase_key)
        if self.site_id == "":
            self.site_id = os.environ.get("FORECAST_SITE_ID", self.site_id)
        if self.model_dir == "models":
            self.model_dir = os.environ.get("FORECAST_MODEL_DIR", self.model_dir)
        if self.test_size == 0.2:
            env_test_size = os.environ.get("FORECAST_TEST_SIZE")
            if env_test_size:
                try:
                    self.test_size = float(env_test_size)
                except ValueError as e:
                    raise ValueError(
                        f"FORECAST_TEST_SIZE must be a float, got: {env_test_size!r}"
                    ) from e
        if self.random_state == 42:
            env_random_state = os.environ.get("FORECAST_RANDOM_STATE")
            if env_random_state:
                try:
                    self.random_state = int(env_random_state)
                except ValueError as e:
                    raise ValueError(
                        f"FORECAST_RANDOM_STATE must be an integer, got: {env_random_state!r}"
                    ) from e
        if self.horizon_hours == 24:
            env_horizon = os.environ.get("FORECAST_HORIZON_HOURS")
            if env_horizon:
                try:
                    self.horizon_hours = int(env_horizon)
                except ValueError as e:
                    raise ValueError(
                        f"FORECAST_HORIZON_HOURS must be an integer, got: {env_horizon!r}"
                    ) from e

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If required configuration is missing or invalid
        """
        if not self.supabase_url:
            raise ValueError("Supabase URL is required for forecasting")
        if not self.supabase_key:
            raise ValueError("Supabase key is required for forecasting")
        if not 0.0 < self.test_size < 1.0:
            raise ValueError("test_size must be between 0.0 and 1.0 (exclusive)")
        if self.horizon_hours <= 0:
            raise ValueError("horizon_hours must be positive")
        if self.target_mae_percent <= 0:
            raise ValueError("target_mae_percent must be positive")
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if self.max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

    def get_model_path(self, target: str) -> Path:
        """Get the path for a model file.

        Args:
            target: Target variable name (e.g., 'load' or 'solar')

        Returns:
            Path to the model file
        """
        return Path(self.model_dir) / f"baseline_{target}_forecaster.joblib"

    def get_all_features(self) -> list[str]:
        """Get list of all feature columns to use for training.

        Returns:
            List of feature column names
        """
        features = []
        features.extend(self.TEMPORAL_FEATURES)
        features.extend(self.CATEGORICAL_FEATURES)
        features.extend(self.ROLLING_FEATURES)
        features.extend(self.LAG_FEATURES)
        features.extend(self.METRIC_FEATURES)
        return features
