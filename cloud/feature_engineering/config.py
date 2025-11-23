"""Configuration for the feature engineering pipeline.

This module provides configuration management for the feature engineering
pipeline, following the same patterns as the existing edge_gateway.storage.cloud_sync
module with environment variable overrides.
"""

import os
from dataclasses import dataclass
from typing import ClassVar


@dataclass
class FeatureEngineeringConfig:
    """Configuration for feature engineering pipeline.

    Supports environment variable overrides:
    - SUPABASE_URL: Supabase project URL
    - SUPABASE_KEY: Supabase API key (anon or service role)
    - FE_SITE_ID: Site identifier to filter data (default: all sites)
    - FE_BATCH_SIZE: Number of records to process per batch
    - FE_ROLLING_WINDOW_DAYS: Days for rolling average calculations
    - FE_LOOKBACK_DAYS: Days of historical data to process

    Attributes:
        supabase_url: Supabase project URL
        supabase_key: Supabase API key
        enabled: Whether feature engineering is enabled
        site_id: Site identifier to filter data (empty string = all sites)
        batch_size: Number of records to process per batch (default: 1000)
        rolling_window_days: Days for rolling average calculations (default: 7)
        lookback_days: Days of historical data to process (default: 30)
        source_table: Source table name for raw data (default: readings)
        target_table: Target table name for training data (default: training_data)
    """

    # Supabase connection settings
    supabase_url: str = ""
    supabase_key: str = ""
    enabled: bool = False

    # Processing settings
    site_id: str = ""
    batch_size: int = 1000
    rolling_window_days: int = 7
    lookback_days: int = 30

    # Table names
    source_table: str = "readings"
    target_table: str = "training_data"

    # Metrics to include in feature engineering
    # These correspond to metric_name values in the readings table
    METRICS_TO_PROCESS: ClassVar[list[str]] = [
        # Solar metrics
        "solar.generation_kw",
        "solar.irradiance_w_m2",
        "solar.panel_temp_celsius",
        "solar.efficiency_percent",
        # Battery metrics
        "battery.soc_percent",
        "battery.power_kw",
        "battery.temperature_celsius",
        "battery.health_percent",
        # Home load metrics
        "home_load.total_load_kw",
        "home_load.hvac_kw",
        "home_load.appliances_kw",
        "home_load.lighting_kw",
        "home_load.ev_charging_kw",
        # Grid price metrics
        "grid_price.price_per_kwh",
        "grid_price.feed_in_tariff",
        "grid_price.demand_charge",
        "grid_price.carbon_intensity_g_kwh",
        # System metrics
        "system.net_grid_flow_kw",
    ]

    # Key metrics for rolling averages (subset of all metrics)
    ROLLING_AVG_METRICS: ClassVar[list[str]] = [
        "solar.generation_kw",
        "home_load.total_load_kw",
        "battery.soc_percent",
        "grid_price.price_per_kwh",
        "system.net_grid_flow_kw",
    ]

    # Categorical columns for one-hot encoding
    # These are stored as separate metric_names with categorical values
    CATEGORICAL_METRICS: ClassVar[list[str]] = [
        "grid_price.time_of_use_period",  # peak, off_peak, shoulder
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
            self.site_id = os.environ.get("FE_SITE_ID", self.site_id)
        if self.batch_size == 1000:
            env_batch = os.environ.get("FE_BATCH_SIZE")
            if env_batch:
                try:
                    self.batch_size = int(env_batch)
                except ValueError as e:
                    raise ValueError(
                        f"FE_BATCH_SIZE must be an integer, got: {env_batch!r}"
                    ) from e
        if self.rolling_window_days == 7:
            env_window = os.environ.get("FE_ROLLING_WINDOW_DAYS")
            if env_window:
                try:
                    self.rolling_window_days = int(env_window)
                except ValueError as e:
                    raise ValueError(
                        f"FE_ROLLING_WINDOW_DAYS must be an integer, got: {env_window!r}"
                    ) from e
        if self.lookback_days == 30:
            env_lookback = os.environ.get("FE_LOOKBACK_DAYS")
            if env_lookback:
                try:
                    self.lookback_days = int(env_lookback)
                except ValueError as e:
                    raise ValueError(
                        f"FE_LOOKBACK_DAYS must be an integer, got: {env_lookback!r}"
                    ) from e

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If required configuration is missing or invalid
        """
        if not self.supabase_url:
            raise ValueError("Supabase URL is required for feature engineering")
        if not self.supabase_key:
            raise ValueError("Supabase key is required for feature engineering")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.rolling_window_days <= 0:
            raise ValueError("rolling_window_days must be positive")
        if self.lookback_days <= 0:
            raise ValueError("lookback_days must be positive")
