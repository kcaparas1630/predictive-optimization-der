"""Data models for DER (Distributed Energy Resources) metrics."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import json


@dataclass
class SolarData:
    """Solar generation data."""

    generation_kw: float  # Current generation in kilowatts
    irradiance_w_m2: float  # Solar irradiance in W/m²
    panel_temp_celsius: float  # Panel temperature
    efficiency_percent: float  # Current efficiency percentage

    def to_dict(self) -> dict:
        return {
            "generation_kw": round(self.generation_kw, 3),
            "irradiance_w_m2": round(self.irradiance_w_m2, 2),
            "panel_temp_celsius": round(self.panel_temp_celsius, 1),
            "efficiency_percent": round(self.efficiency_percent, 2),
        }


@dataclass
class BatteryData:
    """Battery state data."""

    soc_percent: float  # State of Charge (0-100%)
    capacity_kwh: float  # Total capacity in kWh
    power_kw: float  # Current power flow (positive = charging, negative = discharging)
    voltage_v: float  # Battery voltage
    temperature_celsius: float  # Battery temperature
    cycles: int  # Number of charge cycles
    health_percent: float  # Battery health (0-100%)

    def to_dict(self) -> dict:
        return {
            "soc_percent": round(self.soc_percent, 2),
            "capacity_kwh": round(self.capacity_kwh, 2),
            "power_kw": round(self.power_kw, 3),
            "voltage_v": round(self.voltage_v, 1),
            "temperature_celsius": round(self.temperature_celsius, 1),
            "cycles": self.cycles,
            "health_percent": round(self.health_percent, 2),
        }


@dataclass
class HomeLoadData:
    """Home load consumption data."""

    total_load_kw: float  # Total home consumption in kW
    hvac_kw: float  # HVAC load
    appliances_kw: float  # Appliances load
    lighting_kw: float  # Lighting load
    ev_charging_kw: float  # EV charging load (if applicable)
    other_kw: float  # Other loads

    def to_dict(self) -> dict:
        return {
            "total_load_kw": round(self.total_load_kw, 3),
            "hvac_kw": round(self.hvac_kw, 3),
            "appliances_kw": round(self.appliances_kw, 3),
            "lighting_kw": round(self.lighting_kw, 3),
            "ev_charging_kw": round(self.ev_charging_kw, 3),
            "other_kw": round(self.other_kw, 3),
        }


@dataclass
class GridPriceData:
    """Grid price signal data."""

    price_per_kwh: float  # Current price in $/kWh
    feed_in_tariff: float  # Feed-in tariff for exported energy $/kWh
    demand_charge: float  # Demand charge $/kW
    time_of_use_period: str  # "peak", "off_peak", "shoulder"
    carbon_intensity_g_kwh: float  # Grid carbon intensity in gCO2/kWh

    def to_dict(self) -> dict:
        return {
            "price_per_kwh": round(self.price_per_kwh, 4),
            "feed_in_tariff": round(self.feed_in_tariff, 4),
            "demand_charge": round(self.demand_charge, 4),
            "time_of_use_period": self.time_of_use_period,
            "carbon_intensity_g_kwh": round(self.carbon_intensity_g_kwh, 2),
        }


@dataclass
class DERData:
    """Complete DER data snapshot at a point in time."""

    timestamp: datetime
    device_id: str
    solar: SolarData
    battery: BatteryData
    home_load: HomeLoadData
    grid_price: GridPriceData
    net_grid_flow_kw: float = field(init=False)  # Positive = importing, negative = exporting

    def __post_init__(self) -> None:
        """Calculate net grid flow after initialization."""
        self.net_grid_flow_kw = (
            self.home_load.total_load_kw
            - self.solar.generation_kw
            + self.battery.power_kw  # Positive power = charging (consuming)
        )

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "device_id": self.device_id,
            "solar": self.solar.to_dict(),
            "battery": self.battery.to_dict(),
            "home_load": self.home_load.to_dict(),
            "grid_price": self.grid_price.to_dict(),
            "net_grid_flow_kw": round(self.net_grid_flow_kw, 3),
        }

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "DERData":
        """Create DERData from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            device_id=data["device_id"],
            solar=SolarData(**data["solar"]),
            battery=BatteryData(**data["battery"]),
            home_load=HomeLoadData(**data["home_load"]),
            grid_price=GridPriceData(**data["grid_price"]),
        )
