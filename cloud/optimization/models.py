"""Data models for the optimization solver engine.

This module defines data structures for inputs, outputs, and constraints
used by the battery scheduling optimization solver.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Optional


@dataclass
class TariffStructure:
    """Time-of-use tariff structure for a single time period.

    Attributes:
        time_of_use_period: TOU period identifier (peak, off_peak, shoulder)
        price_per_kwh: Import price in $/kWh
        feed_in_tariff: Export price (feed-in tariff) in $/kWh
        demand_charge: Demand charge in $/kW (optional)
    """

    time_of_use_period: Literal["peak", "off_peak", "shoulder"]
    price_per_kwh: float
    feed_in_tariff: float
    demand_charge: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "time_of_use_period": self.time_of_use_period,
            "price_per_kwh": round(self.price_per_kwh, 4),
            "feed_in_tariff": round(self.feed_in_tariff, 4),
            "demand_charge": round(self.demand_charge, 4),
        }


@dataclass
class BatteryConstraints:
    """Battery system constraints for optimization.

    Attributes:
        capacity_kwh: Total battery capacity in kWh
        max_charge_kw: Maximum charging power in kW
        max_discharge_kw: Maximum discharging power in kW
        min_soc_percent: Minimum state of charge (0-100%)
        max_soc_percent: Maximum state of charge (0-100%)
        charge_efficiency: Charging efficiency (0-1)
        discharge_efficiency: Discharging efficiency (0-1)
        initial_soc_percent: Initial state of charge (0-100%)
    """

    capacity_kwh: float
    max_charge_kw: float
    max_discharge_kw: float
    min_soc_percent: float
    max_soc_percent: float
    charge_efficiency: float
    discharge_efficiency: float
    initial_soc_percent: float

    def __post_init__(self) -> None:
        """Validate battery constraints after initialization."""
        if self.capacity_kwh <= 0:
            raise ValueError("capacity_kwh must be positive")
        if self.max_charge_kw <= 0:
            raise ValueError("max_charge_kw must be positive")
        if self.max_discharge_kw <= 0:
            raise ValueError("max_discharge_kw must be positive")
        if not 0 <= self.min_soc_percent < self.max_soc_percent <= 100:
            raise ValueError(
                "SOC limits must satisfy: 0 <= min_soc < max_soc <= 100"
            )
        if not 0 < self.charge_efficiency <= 1:
            raise ValueError("charge_efficiency must be between 0 and 1")
        if not 0 < self.discharge_efficiency <= 1:
            raise ValueError("discharge_efficiency must be between 0 and 1")
        if not self.min_soc_percent <= self.initial_soc_percent <= self.max_soc_percent:
            raise ValueError(
                "initial_soc_percent must be within [min_soc_percent, max_soc_percent]"
            )

    @property
    def min_soc_kwh(self) -> float:
        """Minimum SOC in kWh."""
        return self.capacity_kwh * self.min_soc_percent / 100

    @property
    def max_soc_kwh(self) -> float:
        """Maximum SOC in kWh."""
        return self.capacity_kwh * self.max_soc_percent / 100

    @property
    def initial_soc_kwh(self) -> float:
        """Initial SOC in kWh."""
        return self.capacity_kwh * self.initial_soc_percent / 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "capacity_kwh": round(self.capacity_kwh, 2),
            "max_charge_kw": round(self.max_charge_kw, 2),
            "max_discharge_kw": round(self.max_discharge_kw, 2),
            "min_soc_percent": round(self.min_soc_percent, 2),
            "max_soc_percent": round(self.max_soc_percent, 2),
            "charge_efficiency": round(self.charge_efficiency, 3),
            "discharge_efficiency": round(self.discharge_efficiency, 3),
            "initial_soc_percent": round(self.initial_soc_percent, 2),
        }


@dataclass
class OptimizationInputs:
    """Input data for the optimization solver.

    Attributes:
        start_time: Start time of the optimization horizon
        horizon_hours: Optimization horizon in hours
        time_step_minutes: Time step granularity in minutes
        load_forecast_kw: Predicted load for each time step (kW)
        solar_forecast_kw: Predicted solar generation for each time step (kW)
        tariff_schedule: Tariff structure for each time step
        battery: Battery system constraints
        max_buy_kw: Maximum grid import power (kW)
        max_sell_kw: Maximum grid export power (kW)
    """

    start_time: datetime
    horizon_hours: int
    time_step_minutes: int
    load_forecast_kw: list[float]
    solar_forecast_kw: list[float]
    tariff_schedule: list[TariffStructure]
    battery: BatteryConstraints
    max_buy_kw: float = 10.0
    max_sell_kw: float = 10.0

    def __post_init__(self) -> None:
        """Validate inputs after initialization."""
        num_steps = self.num_time_steps
        if len(self.load_forecast_kw) != num_steps:
            raise ValueError(
                f"load_forecast_kw must have {num_steps} elements, "
                f"got {len(self.load_forecast_kw)}"
            )
        if len(self.solar_forecast_kw) != num_steps:
            raise ValueError(
                f"solar_forecast_kw must have {num_steps} elements, "
                f"got {len(self.solar_forecast_kw)}"
            )
        if len(self.tariff_schedule) != num_steps:
            raise ValueError(
                f"tariff_schedule must have {num_steps} elements, "
                f"got {len(self.tariff_schedule)}"
            )

    @property
    def num_time_steps(self) -> int:
        """Calculate number of time steps in the horizon."""
        return self.horizon_hours * 60 // self.time_step_minutes

    @property
    def time_step_hours(self) -> float:
        """Get time step duration in hours."""
        return self.time_step_minutes / 60.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "start_time": self.start_time.isoformat(),
            "horizon_hours": self.horizon_hours,
            "time_step_minutes": self.time_step_minutes,
            "load_forecast_kw": [round(x, 3) for x in self.load_forecast_kw],
            "solar_forecast_kw": [round(x, 3) for x in self.solar_forecast_kw],
            "tariff_schedule": [t.to_dict() for t in self.tariff_schedule],
            "battery": self.battery.to_dict(),
            "max_buy_kw": round(self.max_buy_kw, 2),
            "max_sell_kw": round(self.max_sell_kw, 2),
        }


@dataclass
class SchedulePoint:
    """A single point in the optimization schedule.

    Represents the optimal decisions for one time step.

    Attributes:
        timestamp: Time of this schedule point
        charge_kw: Battery charging power (kW, positive = charging)
        discharge_kw: Battery discharging power (kW, positive = discharging)
        buy_kw: Grid import power (kW)
        sell_kw: Grid export power (kW)
        soc_percent: Predicted battery state of charge after this step (%)
        load_kw: Predicted load at this time step (kW)
        solar_kw: Predicted solar generation at this time step (kW)
        price_per_kwh: Import price at this time step ($/kWh)
        feed_in_tariff: Export price at this time step ($/kWh)
        cost: Incremental cost for this time step ($)
    """

    timestamp: datetime
    charge_kw: float
    discharge_kw: float
    buy_kw: float
    sell_kw: float
    soc_percent: float
    load_kw: float
    solar_kw: float
    price_per_kwh: float
    feed_in_tariff: float
    cost: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "charge_kw": round(self.charge_kw, 3),
            "discharge_kw": round(self.discharge_kw, 3),
            "buy_kw": round(self.buy_kw, 3),
            "sell_kw": round(self.sell_kw, 3),
            "soc_percent": round(self.soc_percent, 2),
            "load_kw": round(self.load_kw, 3),
            "solar_kw": round(self.solar_kw, 3),
            "price_per_kwh": round(self.price_per_kwh, 4),
            "feed_in_tariff": round(self.feed_in_tariff, 4),
            "cost": round(self.cost, 4),
        }

    def to_influxdb_point(self, device_id: str) -> dict[str, Any]:
        """Convert to InfluxDB point format.

        Args:
            device_id: Device identifier for tagging

        Returns:
            Dictionary suitable for InfluxDB write
        """
        return {
            "measurement": "optimization_schedule",
            "time": self.timestamp.isoformat(),
            "tags": {
                "device_id": device_id,
            },
            "fields": {
                "charge_kw": round(self.charge_kw, 3),
                "discharge_kw": round(self.discharge_kw, 3),
                "buy_kw": round(self.buy_kw, 3),
                "sell_kw": round(self.sell_kw, 3),
                "soc_percent": round(self.soc_percent, 2),
                "load_kw": round(self.load_kw, 3),
                "solar_kw": round(self.solar_kw, 3),
                "price_per_kwh": round(self.price_per_kwh, 4),
                "feed_in_tariff": round(self.feed_in_tariff, 4),
                "cost": round(self.cost, 4),
            },
        }

    def to_supabase_record(self, site_id: str, device_id: str, run_id: str) -> dict[str, Any]:
        """Convert to Supabase record format.

        Args:
            site_id: Site identifier
            device_id: Device identifier
            run_id: Optimization run identifier

        Returns:
            Dictionary suitable for Supabase insert
        """
        return {
            "time": self.timestamp.isoformat(),
            "site_id": site_id,
            "device_id": device_id,
            "run_id": run_id,
            "charge_kw": round(self.charge_kw, 3),
            "discharge_kw": round(self.discharge_kw, 3),
            "buy_kw": round(self.buy_kw, 3),
            "sell_kw": round(self.sell_kw, 3),
            "soc_percent": round(self.soc_percent, 2),
            "load_kw": round(self.load_kw, 3),
            "solar_kw": round(self.solar_kw, 3),
            "price_per_kwh": round(self.price_per_kwh, 4),
            "feed_in_tariff": round(self.feed_in_tariff, 4),
            "cost": round(self.cost, 4),
        }


@dataclass
class OptimizationResult:
    """Result from the optimization solver.

    Attributes:
        status: Solver status (Optimal, Infeasible, etc.)
        run_id: Unique identifier for this optimization run
        start_time: Start time of the optimization horizon
        end_time: End time of the optimization horizon
        created_at: When the optimization was run
        total_cost: Total cost over the horizon ($)
        total_grid_import_kwh: Total energy imported from grid (kWh)
        total_grid_export_kwh: Total energy exported to grid (kWh)
        schedule: List of schedule points for each time step
        inputs: Input data used for optimization
        solver_time_seconds: Time taken by the solver
        message: Additional message from solver
    """

    status: Literal["Optimal", "Infeasible", "Unbounded", "Error", "Not Solved"]
    run_id: str
    start_time: datetime
    end_time: datetime
    created_at: datetime
    total_cost: float
    total_grid_import_kwh: float
    total_grid_export_kwh: float
    schedule: list[SchedulePoint]
    inputs: OptimizationInputs
    solver_time_seconds: float
    message: str = ""

    @property
    def is_optimal(self) -> bool:
        """Check if the solution is optimal."""
        return self.status == "Optimal"

    @property
    def num_time_steps(self) -> int:
        """Get number of time steps in the schedule."""
        return len(self.schedule)

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the optimization result.

        Returns:
            Dictionary with key metrics
        """
        return {
            "status": self.status,
            "run_id": self.run_id,
            "is_optimal": self.is_optimal,
            "total_cost": round(self.total_cost, 2),
            "total_grid_import_kwh": round(self.total_grid_import_kwh, 2),
            "total_grid_export_kwh": round(self.total_grid_export_kwh, 2),
            "horizon_hours": self.inputs.horizon_hours,
            "num_time_steps": self.num_time_steps,
            "solver_time_seconds": round(self.solver_time_seconds, 3),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "created_at": self.created_at.isoformat(),
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to full dictionary representation."""
        return {
            "status": self.status,
            "run_id": self.run_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "created_at": self.created_at.isoformat(),
            "total_cost": round(self.total_cost, 4),
            "total_grid_import_kwh": round(self.total_grid_import_kwh, 3),
            "total_grid_export_kwh": round(self.total_grid_export_kwh, 3),
            "schedule": [s.to_dict() for s in self.schedule],
            "inputs": self.inputs.to_dict(),
            "solver_time_seconds": round(self.solver_time_seconds, 3),
            "message": self.message,
        }

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Convert to JSON string.

        Args:
            indent: JSON indentation level

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OptimizationResult":
        """Create OptimizationResult from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            OptimizationResult instance
        """
        # Parse schedule points
        schedule = [
            SchedulePoint(
                timestamp=datetime.fromisoformat(s["timestamp"]),
                charge_kw=s["charge_kw"],
                discharge_kw=s["discharge_kw"],
                buy_kw=s["buy_kw"],
                sell_kw=s["sell_kw"],
                soc_percent=s["soc_percent"],
                load_kw=s["load_kw"],
                solar_kw=s["solar_kw"],
                price_per_kwh=s["price_per_kwh"],
                feed_in_tariff=s["feed_in_tariff"],
                cost=s["cost"],
            )
            for s in data["schedule"]
        ]

        # Parse inputs
        inputs_data = data["inputs"]
        tariff_schedule = [
            TariffStructure(**t) for t in inputs_data["tariff_schedule"]
        ]
        battery = BatteryConstraints(**inputs_data["battery"])
        inputs = OptimizationInputs(
            start_time=datetime.fromisoformat(inputs_data["start_time"]),
            horizon_hours=inputs_data["horizon_hours"],
            time_step_minutes=inputs_data["time_step_minutes"],
            load_forecast_kw=inputs_data["load_forecast_kw"],
            solar_forecast_kw=inputs_data["solar_forecast_kw"],
            tariff_schedule=tariff_schedule,
            battery=battery,
            max_buy_kw=inputs_data["max_buy_kw"],
            max_sell_kw=inputs_data["max_sell_kw"],
        )

        return cls(
            status=data["status"],
            run_id=data["run_id"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            total_cost=data["total_cost"],
            total_grid_import_kwh=data["total_grid_import_kwh"],
            total_grid_export_kwh=data["total_grid_export_kwh"],
            schedule=schedule,
            inputs=inputs,
            solver_time_seconds=data["solver_time_seconds"],
            message=data.get("message", ""),
        )
