"""Configuration management for the DER data generator."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


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

    @classmethod
    def from_dict(cls, data: dict) -> "GeneratorConfig":
        """Create config from dictionary."""
        return cls(
            device_id=data.get("device_id", "edge-gateway-001"),
            interval_seconds=data.get("interval_seconds", 300),
            output_file=data.get("output_file"),
            seed=data.get("seed"),
            solar=SolarConfig(**data.get("solar", {})),
            battery=BatteryConfig(**data.get("battery", {})),
            home_load=HomeLoadConfig(**data.get("home_load", {})),
            grid_price=GridPriceConfig(**data.get("grid_price", {})),
        )

    @classmethod
    def from_file(cls, path: Path) -> "GeneratorConfig":
        """Load config from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "device_id": self.device_id,
            "interval_seconds": self.interval_seconds,
            "output_file": self.output_file,
            "seed": self.seed,
            "solar": {
                "capacity_kw": self.solar.capacity_kw,
                "latitude": self.solar.latitude,
                "panel_efficiency": self.solar.panel_efficiency,
                "temp_coefficient": self.solar.temp_coefficient,
            },
            "battery": {
                "capacity_kwh": self.battery.capacity_kwh,
                "max_charge_rate_kw": self.battery.max_charge_rate_kw,
                "max_discharge_rate_kw": self.battery.max_discharge_rate_kw,
                "round_trip_efficiency": self.battery.round_trip_efficiency,
                "initial_soc": self.battery.initial_soc,
                "min_soc": self.battery.min_soc,
                "max_soc": self.battery.max_soc,
            },
            "home_load": {
                "base_load_kw": self.home_load.base_load_kw,
                "peak_load_kw": self.home_load.peak_load_kw,
                "has_ev": self.home_load.has_ev,
                "ev_charging_kw": self.home_load.ev_charging_kw,
                "hvac_capacity_kw": self.home_load.hvac_capacity_kw,
            },
            "grid_price": {
                "off_peak_price": self.grid_price.off_peak_price,
                "shoulder_price": self.grid_price.shoulder_price,
                "peak_price": self.grid_price.peak_price,
                "base_feed_in_tariff": self.grid_price.base_feed_in_tariff,
                "demand_charge": self.grid_price.demand_charge,
                "volatility": self.grid_price.volatility,
            },
        }

    def to_file(self, path: Path) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# Default configuration template
DEFAULT_CONFIG = GeneratorConfig()
