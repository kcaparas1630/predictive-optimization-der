"""Configuration management for the DER data generator."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from edge_gateway.storage import InfluxDBConfig


@dataclass
class SolarConfig:
    """Solar panel configuration."""

    capacity_kw: float = 10.0
    latitude: float = 37.7749
    panel_efficiency: float = 0.20
    temp_coefficient: float = -0.004


@dataclass
class BatteryConfig:
    """Battery storage configuration."""

    capacity_kwh: float = 13.5
    max_charge_rate_kw: float = 5.0
    max_discharge_rate_kw: float = 5.0
    round_trip_efficiency: float = 0.90
    initial_soc: float = 50.0
    min_soc: float = 10.0
    max_soc: float = 90.0


@dataclass
class HomeLoadConfig:
    """Home load configuration."""

    base_load_kw: float = 0.5
    peak_load_kw: float = 8.0
    has_ev: bool = True
    ev_charging_kw: float = 7.2
    hvac_capacity_kw: float = 3.5


@dataclass
class GridPriceConfig:
    """Grid pricing configuration."""

    off_peak_price: float = 0.08
    shoulder_price: float = 0.15
    peak_price: float = 0.30
    base_feed_in_tariff: float = 0.05
    demand_charge: float = 10.0
    volatility: float = 0.15


@dataclass
class GeneratorConfig:
    """Main generator configuration."""

    device_id: str = "edge-gateway-001"
    interval_seconds: float = 300  # 5 minutes
    output_file: Optional[str] = None
    seed: Optional[int] = None

    solar: SolarConfig = field(default_factory=SolarConfig)
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    home_load: HomeLoadConfig = field(default_factory=HomeLoadConfig)
    grid_price: GridPriceConfig = field(default_factory=GridPriceConfig)
    influxdb: InfluxDBConfig = field(default_factory=InfluxDBConfig)

    @classmethod
    def from_dict(cls, data: dict) -> "GeneratorConfig":
        """Create config from dictionary."""
        try:
            return cls(
                device_id=data.get("device_id", "edge-gateway-001"),
                interval_seconds=data.get("interval_seconds", 300),
                output_file=data.get("output_file"),
                seed=data.get("seed"),
                solar=SolarConfig(**data.get("solar", {})),
                battery=BatteryConfig(**data.get("battery", {})),
                home_load=HomeLoadConfig(**data.get("home_load", {})),
                grid_price=GridPriceConfig(**data.get("grid_price", {})),
                influxdb=InfluxDBConfig(**data.get("influxdb", {})),
            )
        except TypeError as e:
            raise ValueError(f"Invalid configuration format: {e}") from e

    @classmethod
    def from_file(cls, path: Path) -> "GeneratorConfig":
        """Load config from JSON file."""
        try:
            with open(path) as f:
                data = json.load(f)
            return cls.from_dict(data)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Configuration file not found: {path}",
            ) from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in configuration file {path}: {exc}") from exc
        except (OSError, ValueError) as exc:
            # OSError: read problems; ValueError: invalid configuration structure
            raise RuntimeError(
                f"Failed to load configuration from {path}: {exc}",
            ) from exc

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)

    def to_file(self, path: Path) -> None:
        """Save config to JSON file."""
        try:
            with open(path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        except OSError as exc:
            raise RuntimeError(
                f"Failed to save configuration to {path}: {exc}",
            ) from exc


# Default configuration template
DEFAULT_CONFIG = GeneratorConfig()
