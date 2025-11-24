"""Optimization solver engine for DER battery scheduling.

This module provides the core optimization logic using PuLP for computing
optimal 24-hour schedules for battery charge/discharge operations.

The optimization problem minimizes total energy cost subject to:
- Energy balance constraints (load = solar + battery_discharge + grid_import - battery_charge - grid_export)
- Battery SOC dynamics (SOC[t+1] = SOC[t] + charge*efficiency - discharge/efficiency)
- Battery capacity and power limits
- Grid import/export limits
- Non-negativity constraints
"""

import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from cloud.optimization.config import OptimizationConfig
from cloud.optimization.models import (
    BatteryConstraints,
    OptimizationInputs,
    OptimizationResult,
    SchedulePoint,
    TariffStructure,
)

logger = logging.getLogger(__name__)

# Optional PuLP import
try:
    import pulp

    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    pulp = None  # type: ignore

# Optional Supabase import
try:
    from supabase import Client, create_client

    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    Client = None  # type: ignore
    create_client = None  # type: ignore

# Optional InfluxDB import
try:
    from influxdb_client import InfluxDBClient, Point
    from influxdb_client.client.write_api import SYNCHRONOUS

    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False
    InfluxDBClient = None  # type: ignore
    Point = None  # type: ignore
    SYNCHRONOUS = None  # type: ignore


class OptimizationSolver:
    """Optimization solver for DER battery scheduling.

    This class handles:
    - Building and solving the linear programming optimization problem
    - Computing optimal charge/discharge schedules
    - Storing results to InfluxDB (local) and Supabase (cloud)

    The optimization minimizes total cost:
        min sum_t [ Buy[t] * import_price[t] - Sell[t] * export_price[t] ]

    Subject to:
        - Energy balance: Load[t] = Solar[t] + Discharge[t] + Buy[t] - Charge[t] - Sell[t]
        - SOC dynamics: SOC[t+1] = SOC[t] + Charge[t]*dt*eff_c - Discharge[t]*dt/eff_d
        - SOC limits: SOC_min <= SOC[t] <= SOC_max
        - Power limits: 0 <= Charge[t] <= P_charge_max
                        0 <= Discharge[t] <= P_discharge_max
                        0 <= Buy[t] <= P_buy_max
                        0 <= Sell[t] <= P_sell_max

    Example:
        >>> config = OptimizationConfig(
        ...     supabase_url="https://your-project.supabase.co",
        ...     supabase_key="your-key",
        ...     enabled=True
        ... )
        >>> solver = OptimizationSolver(config)
        >>> result = solver.optimize(inputs)
        >>> print(f"Total cost: ${result.total_cost:.2f}")
    """

    def __init__(self, config: OptimizationConfig) -> None:
        """Initialize optimization solver.

        Args:
            config: Optimization configuration

        Raises:
            ImportError: If required libraries are not installed
            ValueError: If configuration is invalid
        """
        self.config = config
        self._supabase_client: Optional[Client] = None
        self._influx_client: Optional[InfluxDBClient] = None

        if not config.enabled:
            logger.info("Optimization is disabled")
            return

        self._validate_dependencies()
        config.validate()
        self._connect()

    def _validate_dependencies(self) -> None:
        """Validate required libraries are installed."""
        missing = []
        if not PULP_AVAILABLE:
            missing.append("pulp")
        if not SUPABASE_AVAILABLE:
            missing.append("supabase")

        if missing:
            raise ImportError(
                f"Required libraries not installed: {', '.join(missing)}. "
                f"Install with: pip install {' '.join(missing)}"
            )

    def _connect(self) -> None:
        """Establish connections to Supabase and optionally InfluxDB."""
        # Connect to Supabase
        try:
            self._supabase_client = create_client(
                self.config.supabase_url,
                self.config.supabase_key,
            )
            logger.info(
                "Connected to Supabase at %s for optimization",
                self.config.supabase_url,
            )
        except Exception:
            logger.exception("Failed to connect to Supabase")
            raise

        # Optionally connect to InfluxDB for local storage
        if INFLUXDB_AVAILABLE and self.config.influxdb_token:
            try:
                self._influx_client = InfluxDBClient(
                    url=self.config.influxdb_url,
                    token=self.config.influxdb_token,
                    org=self.config.influxdb_org,
                )
                logger.info(
                    "Connected to InfluxDB at %s for local storage",
                    self.config.influxdb_url,
                )
            except Exception:
                logger.warning(
                    "Failed to connect to InfluxDB, local storage disabled",
                    exc_info=True,
                )

    def is_connected(self) -> bool:
        """Check if Supabase connection is established.

        Returns:
            True if connected to Supabase
        """
        if not self.config.enabled:
            return False
        return self._supabase_client is not None

    def health_check(self) -> dict[str, bool]:
        """Perform health checks on connections.

        Returns:
            Dictionary with health status for each service
        """
        status = {"supabase": False, "influxdb": False, "pulp": PULP_AVAILABLE}

        if not self.is_connected():
            return status

        # Check Supabase
        try:
            self._supabase_client.table(self.config.schedule_table).select("time").limit(1).execute()
            status["supabase"] = True
        except Exception as e:
            logger.warning("Supabase health check failed: %s", e)

        # Check InfluxDB if available
        if self._influx_client:
            try:
                health = self._influx_client.health()
                status["influxdb"] = health.status == "pass"
            except Exception as e:
                logger.warning("InfluxDB health check failed: %s", e)

        return status

    def create_default_battery_constraints(
        self,
        initial_soc_percent: float,
    ) -> BatteryConstraints:
        """Create battery constraints from config defaults.

        Args:
            initial_soc_percent: Current battery state of charge (%)

        Returns:
            BatteryConstraints with default values from config
        """
        return BatteryConstraints(
            capacity_kwh=self.config.default_battery_capacity_kwh,
            max_charge_kw=self.config.default_max_charge_kw,
            max_discharge_kw=self.config.default_max_discharge_kw,
            min_soc_percent=self.config.default_min_soc_percent,
            max_soc_percent=self.config.default_max_soc_percent,
            charge_efficiency=self.config.default_charge_efficiency,
            discharge_efficiency=self.config.default_discharge_efficiency,
            initial_soc_percent=initial_soc_percent,
        )

    def optimize(self, inputs: OptimizationInputs) -> OptimizationResult:
        """Run optimization and compute optimal schedule.

        Args:
            inputs: Optimization input data including forecasts and constraints

        Returns:
            OptimizationResult with optimal schedule or error status
        """
        run_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc)
        start_time_solver = time.time()

        logger.info(
            "Starting optimization run %s for %d time steps",
            run_id,
            inputs.num_time_steps,
        )

        try:
            # Build and solve the optimization problem
            result = self._solve_optimization(inputs, run_id, created_at)

            solver_time = time.time() - start_time_solver
            result = OptimizationResult(
                status=result["status"],
                run_id=run_id,
                start_time=inputs.start_time,
                end_time=inputs.start_time + timedelta(hours=inputs.horizon_hours),
                created_at=created_at,
                total_cost=result["total_cost"],
                total_grid_import_kwh=result["total_grid_import_kwh"],
                total_grid_export_kwh=result["total_grid_export_kwh"],
                schedule=result["schedule"],
                inputs=inputs,
                solver_time_seconds=solver_time,
                message=result.get("message", ""),
            )

            logger.info(
                "Optimization completed: status=%s, cost=%.2f, time=%.3fs",
                result.status,
                result.total_cost,
                solver_time,
            )

            return result

        except Exception as e:
            solver_time = time.time() - start_time_solver
            logger.exception("Optimization failed: %s", e)
            return OptimizationResult(
                status="Error",
                run_id=run_id,
                start_time=inputs.start_time,
                end_time=inputs.start_time + timedelta(hours=inputs.horizon_hours),
                created_at=created_at,
                total_cost=0.0,
                total_grid_import_kwh=0.0,
                total_grid_export_kwh=0.0,
                schedule=[],
                inputs=inputs,
                solver_time_seconds=solver_time,
                message=str(e),
            )

    def _solve_optimization(
        self,
        inputs: OptimizationInputs,
        run_id: str,
        created_at: datetime,
    ) -> dict[str, Any]:
        """Build and solve the LP optimization problem.

        Args:
            inputs: Optimization inputs
            run_id: Unique run identifier
            created_at: Timestamp of optimization run

        Returns:
            Dictionary with solution data
        """
        T = inputs.num_time_steps
        dt = inputs.time_step_hours
        battery = inputs.battery

        # Create the LP problem
        prob = pulp.LpProblem("DER_Battery_Optimization", pulp.LpMinimize)

        # Decision variables
        # Charge power at each time step (kW)
        charge = [
            pulp.LpVariable(f"charge_{t}", lowBound=0, upBound=battery.max_charge_kw)
            for t in range(T)
        ]
        # Discharge power at each time step (kW)
        discharge = [
            pulp.LpVariable(f"discharge_{t}", lowBound=0, upBound=battery.max_discharge_kw)
            for t in range(T)
        ]
        # Grid import power at each time step (kW)
        buy = [
            pulp.LpVariable(f"buy_{t}", lowBound=0, upBound=inputs.max_buy_kw)
            for t in range(T)
        ]
        # Grid export power at each time step (kW)
        sell = [
            pulp.LpVariable(f"sell_{t}", lowBound=0, upBound=inputs.max_sell_kw)
            for t in range(T)
        ]
        # State of charge at each time step (kWh)
        soc = [
            pulp.LpVariable(
                f"soc_{t}",
                lowBound=battery.min_soc_kwh,
                upBound=battery.max_soc_kwh,
            )
            for t in range(T + 1)  # T+1 to include final SOC
        ]

        # Objective: Minimize total cost
        # Cost = sum( Buy[t] * import_price[t] - Sell[t] * export_price[t] ) * dt
        prob += pulp.lpSum(
            [
                (buy[t] * inputs.tariff_schedule[t].price_per_kwh
                 - sell[t] * inputs.tariff_schedule[t].feed_in_tariff) * dt
                for t in range(T)
            ]
        ), "Total_Cost"

        # Constraints

        # Initial SOC constraint
        prob += soc[0] == battery.initial_soc_kwh, "Initial_SOC"

        for t in range(T):
            # Energy balance constraint:
            # Load[t] = Solar[t] + Discharge[t] + Buy[t] - Charge[t] - Sell[t]
            # Rearranged: Charge[t] + Sell[t] + Load[t] = Solar[t] + Discharge[t] + Buy[t]
            prob += (
                charge[t] + sell[t] + inputs.load_forecast_kw[t]
                == inputs.solar_forecast_kw[t] + discharge[t] + buy[t]
            ), f"Energy_Balance_{t}"

            # SOC dynamics constraint:
            # SOC[t+1] = SOC[t] + Charge[t] * dt * eff_charge - Discharge[t] * dt / eff_discharge
            prob += (
                soc[t + 1]
                == soc[t]
                + charge[t] * dt * battery.charge_efficiency
                - discharge[t] * dt / battery.discharge_efficiency
            ), f"SOC_Dynamics_{t}"

        # Solve the problem
        solver = self._get_solver()
        prob.solve(solver)

        # Extract solution
        status = pulp.LpStatus[prob.status]

        if status != "Optimal":
            return {
                "status": status,
                "total_cost": 0.0,
                "total_grid_import_kwh": 0.0,
                "total_grid_export_kwh": 0.0,
                "schedule": [],
                "message": f"Solver returned status: {status}",
            }

        # Build schedule from solution
        schedule = []
        total_import_kwh = 0.0
        total_export_kwh = 0.0
        total_cost = 0.0

        for t in range(T):
            timestamp = inputs.start_time + timedelta(minutes=t * inputs.time_step_minutes)

            charge_val = pulp.value(charge[t]) or 0.0
            discharge_val = pulp.value(discharge[t]) or 0.0
            buy_val = pulp.value(buy[t]) or 0.0
            sell_val = pulp.value(sell[t]) or 0.0
            soc_val = pulp.value(soc[t + 1]) or 0.0

            # Calculate cost for this time step
            step_cost = (
                buy_val * inputs.tariff_schedule[t].price_per_kwh
                - sell_val * inputs.tariff_schedule[t].feed_in_tariff
            ) * dt

            total_import_kwh += buy_val * dt
            total_export_kwh += sell_val * dt
            total_cost += step_cost

            schedule.append(
                SchedulePoint(
                    timestamp=timestamp,
                    charge_kw=charge_val,
                    discharge_kw=discharge_val,
                    buy_kw=buy_val,
                    sell_kw=sell_val,
                    soc_percent=soc_val / battery.capacity_kwh * 100,
                    load_kw=inputs.load_forecast_kw[t],
                    solar_kw=inputs.solar_forecast_kw[t],
                    price_per_kwh=inputs.tariff_schedule[t].price_per_kwh,
                    feed_in_tariff=inputs.tariff_schedule[t].feed_in_tariff,
                    cost=step_cost,
                )
            )

        return {
            "status": "Optimal",
            "total_cost": total_cost,
            "total_grid_import_kwh": total_import_kwh,
            "total_grid_export_kwh": total_export_kwh,
            "schedule": schedule,
            "message": "",
        }

    def _get_solver(self) -> Any:
        """Get the PuLP solver based on configuration.

        Returns:
            PuLP solver instance
        """
        solver_name = self.config.solver.upper()

        if solver_name == "CBC" or solver_name == "PULP_CBC_CMD":
            return pulp.PULP_CBC_CMD(msg=0)
        elif solver_name == "GLPK":
            return pulp.GLPK(msg=0)
        else:
            # Default to CBC
            logger.warning(
                "Unknown solver %s, defaulting to CBC",
                solver_name,
            )
            return pulp.PULP_CBC_CMD(msg=0)

    def store_result_locally(self, result: OptimizationResult) -> bool:
        """Store optimization result to local InfluxDB.

        Args:
            result: Optimization result to store

        Returns:
            True if storage was successful
        """
        if not self._influx_client:
            logger.warning("InfluxDB not connected, skipping local storage")
            return False

        if not result.schedule:
            logger.warning("No schedule to store")
            return False

        try:
            write_api = self._influx_client.write_api(write_options=SYNCHRONOUS)

            points = []
            for point in result.schedule:
                p = (
                    Point("optimization_schedule")
                    .tag("device_id", self.config.device_id)
                    .tag("run_id", result.run_id)
                    .field("charge_kw", point.charge_kw)
                    .field("discharge_kw", point.discharge_kw)
                    .field("buy_kw", point.buy_kw)
                    .field("sell_kw", point.sell_kw)
                    .field("soc_percent", point.soc_percent)
                    .field("load_kw", point.load_kw)
                    .field("solar_kw", point.solar_kw)
                    .field("price_per_kwh", point.price_per_kwh)
                    .field("feed_in_tariff", point.feed_in_tariff)
                    .field("cost", point.cost)
                    .time(point.timestamp)
                )
                points.append(p)

            write_api.write(
                bucket=self.config.influxdb_bucket,
                org=self.config.influxdb_org,
                record=points,
            )

            logger.info(
                "Stored %d schedule points to InfluxDB for run %s",
                len(points),
                result.run_id,
            )
            return True

        except Exception:
            logger.exception("Failed to store result to InfluxDB")
            return False

    def sync_result_to_cloud(self, result: OptimizationResult) -> bool:
        """Sync optimization result to Supabase.

        Args:
            result: Optimization result to sync

        Returns:
            True if sync was successful
        """
        if not self._supabase_client:
            logger.warning("Supabase not connected, skipping cloud sync")
            return False

        if not result.schedule:
            logger.warning("No schedule to sync")
            return False

        try:
            # Prepare records for Supabase
            records = [
                point.to_supabase_record(
                    self.config.site_id,
                    self.config.device_id,
                    result.run_id,
                )
                for point in result.schedule
            ]

            # Upsert to prevent duplicates (based on time, site_id, device_id, run_id)
            self._supabase_client.table(self.config.schedule_table).upsert(
                records,
                on_conflict="time,site_id,device_id,run_id",
            ).execute()

            logger.info(
                "Synced %d schedule points to Supabase for run %s",
                len(records),
                result.run_id,
            )
            return True

        except Exception:
            logger.exception("Failed to sync result to Supabase")
            return False

    def optimize_and_store(self, inputs: OptimizationInputs) -> OptimizationResult:
        """Run optimization and store results both locally and to cloud.

        This is the main entry point for running optimization with
        automatic storage.

        Args:
            inputs: Optimization input data

        Returns:
            OptimizationResult with optimal schedule
        """
        result = self.optimize(inputs)

        if result.is_optimal:
            # Store locally to InfluxDB
            local_success = self.store_result_locally(result)

            # Sync to Supabase
            cloud_success = self.sync_result_to_cloud(result)

            if local_success and cloud_success:
                logger.info("Results stored locally and synced to cloud")
            elif local_success:
                logger.warning("Results stored locally but cloud sync failed")
            elif cloud_success:
                logger.warning("Results synced to cloud but local storage failed")
            else:
                logger.warning("Both local storage and cloud sync failed")

        return result

    def close(self) -> None:
        """Close all connections."""
        if self._influx_client:
            try:
                self._influx_client.close()
            except Exception:
                logger.debug("Error closing InfluxDB client", exc_info=True)
            self._influx_client = None

        self._supabase_client = None
        logger.info("Closed optimization solver connections")

    def __enter__(self) -> "OptimizationSolver":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
