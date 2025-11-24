"""Configuration for the optimization solver engine.

This module provides configuration management for the battery scheduling
optimization solver, following the same patterns as other modules.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


@dataclass
class OptimizationConfig:
    """Configuration for the optimization solver.

    Supports environment variable overrides:
    - SUPABASE_URL: Supabase project URL
    - SUPABASE_KEY: Supabase API key
    - INFLUXDB_URL: InfluxDB server URL
    - INFLUXDB_TOKEN: InfluxDB authentication token
    - INFLUXDB_ORG: InfluxDB organization name
    - INFLUXDB_BUCKET: InfluxDB bucket name
    - OPT_SITE_ID: Site identifier for optimization
    - OPT_DEVICE_ID: Device identifier for optimization
    - OPT_HORIZON_HOURS: Optimization horizon in hours (default: 24)
    - OPT_TIME_STEP_MINUTES: Time step in minutes (default: 60)
    - OPT_SOLVER: LP solver to use (default: CBC)

    Attributes:
        supabase_url: Supabase project URL
        supabase_key: Supabase API key
        influxdb_url: InfluxDB server URL
        influxdb_token: InfluxDB authentication token
        influxdb_org: InfluxDB organization name
        influxdb_bucket: InfluxDB bucket name
        enabled: Whether optimization is enabled
        site_id: Site identifier for optimization results
        device_id: Device identifier for optimization
        horizon_hours: Optimization horizon in hours (default: 24)
        time_step_minutes: Time step granularity in minutes (default: 60)
        solver: LP solver to use (CBC, GLPK, etc.)
        schedule_table: Supabase table for optimization schedules
    """

    # Database connection settings
    supabase_url: str = ""
    supabase_key: str = ""
    influxdb_url: str = "http://localhost:8086"
    influxdb_token: str = ""
    influxdb_org: str = "edge-gateway"
    influxdb_bucket: str = "der-data"
    enabled: bool = False

    # Identification settings
    site_id: str = "edge-gateway-site"
    device_id: str = "edge-gateway-001"

    # Optimization settings
    horizon_hours: int = 24
    time_step_minutes: int = 60
    solver: str = "CBC"

    # Data settings
    schedule_table: str = "optimization_schedule"

    # Default battery constraints (can be overridden at runtime)
    default_battery_capacity_kwh: float = 13.5
    default_max_charge_kw: float = 5.0
    default_max_discharge_kw: float = 5.0
    default_min_soc_percent: float = 10.0
    default_max_soc_percent: float = 90.0
    default_charge_efficiency: float = 0.95
    default_discharge_efficiency: float = 0.95

    # Grid constraints
    default_max_buy_kw: float = 10.0
    default_max_sell_kw: float = 10.0

    # Solver constants
    SOLVER_CBC: ClassVar[str] = "CBC"
    SOLVER_GLPK: ClassVar[str] = "GLPK"
    SOLVER_PULP_CBC_CMD: ClassVar[str] = "PULP_CBC_CMD"

    def __post_init__(self) -> None:
        """Apply environment variable overrides only when values are at defaults.

        Precedence: explicit args > env vars > defaults
        """
        if self.supabase_url == "":
            self.supabase_url = os.environ.get("SUPABASE_URL", self.supabase_url)
        if self.supabase_key == "":
            self.supabase_key = os.environ.get("SUPABASE_KEY", self.supabase_key)
        if self.influxdb_url == "http://localhost:8086":
            self.influxdb_url = os.environ.get("INFLUXDB_URL", self.influxdb_url)
        if self.influxdb_token == "":
            self.influxdb_token = os.environ.get("INFLUXDB_TOKEN", self.influxdb_token)
        if self.influxdb_org == "edge-gateway":
            self.influxdb_org = os.environ.get("INFLUXDB_ORG", self.influxdb_org)
        if self.influxdb_bucket == "der-data":
            self.influxdb_bucket = os.environ.get("INFLUXDB_BUCKET", self.influxdb_bucket)
        if self.site_id == "edge-gateway-site":
            self.site_id = os.environ.get("OPT_SITE_ID", self.site_id)
        if self.device_id == "edge-gateway-001":
            self.device_id = os.environ.get("OPT_DEVICE_ID", self.device_id)
        if self.horizon_hours == 24:
            env_horizon = os.environ.get("OPT_HORIZON_HOURS")
            if env_horizon:
                try:
                    self.horizon_hours = int(env_horizon)
                except ValueError as e:
                    raise ValueError(
                        f"OPT_HORIZON_HOURS must be an integer, got: {env_horizon!r}"
                    ) from e
        if self.time_step_minutes == 60:
            env_step = os.environ.get("OPT_TIME_STEP_MINUTES")
            if env_step:
                try:
                    self.time_step_minutes = int(env_step)
                except ValueError as e:
                    raise ValueError(
                        f"OPT_TIME_STEP_MINUTES must be an integer, got: {env_step!r}"
                    ) from e
        if self.solver == "CBC":
            self.solver = os.environ.get("OPT_SOLVER", self.solver)

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If required configuration is missing or invalid
        """
        if not self.supabase_url:
            raise ValueError(
                "Supabase URL is required for optimization. "
                "Set SUPABASE_URL environment variable or pass --supabase-url argument."
            )
        if not self.supabase_key:
            raise ValueError(
                "Supabase key is required for optimization. "
                "Set SUPABASE_KEY environment variable or pass --supabase-key argument."
            )
        if self.horizon_hours <= 0:
            raise ValueError("horizon_hours must be positive")
        if self.time_step_minutes <= 0:
            raise ValueError("time_step_minutes must be positive")
        if self.time_step_minutes > 60:
            raise ValueError("time_step_minutes cannot exceed 60")
        if 60 % self.time_step_minutes != 0:
            raise ValueError("time_step_minutes must divide evenly into 60")
        if self.default_battery_capacity_kwh <= 0:
            raise ValueError("default_battery_capacity_kwh must be positive")
        if self.default_max_charge_kw <= 0:
            raise ValueError("default_max_charge_kw must be positive")
        if self.default_max_discharge_kw <= 0:
            raise ValueError("default_max_discharge_kw must be positive")
        if not 0 <= self.default_min_soc_percent < self.default_max_soc_percent <= 100:
            raise ValueError(
                "SOC limits must satisfy: 0 <= min_soc < max_soc <= 100"
            )
        if not 0 < self.default_charge_efficiency <= 1:
            raise ValueError("charge_efficiency must be between 0 and 1")
        if not 0 < self.default_discharge_efficiency <= 1:
            raise ValueError("discharge_efficiency must be between 0 and 1")

    @property
    def num_time_steps(self) -> int:
        """Calculate number of time steps in the optimization horizon.

        Returns:
            Number of time steps (e.g., 24 for hourly steps over 24 hours)
        """
        return self.horizon_hours * 60 // self.time_step_minutes

    @property
    def time_step_hours(self) -> float:
        """Get time step duration in hours.

        Returns:
            Time step in hours (e.g., 1.0 for 60-minute steps)
        """
        return self.time_step_minutes / 60.0

    def get_schedule_path(self) -> Path:
        """Get the path for storing optimization schedules locally.

        Returns:
            Path to the schedule storage directory
        """
        return Path("schedules")
