#!/usr/bin/env python3
"""CLI runner for the optimization solver engine.

This script provides a command-line interface for running the DER battery
scheduling optimization solver.

Usage:
    # Run optimization with sample/demo data
    python run_optimization.py --demo

    # Run optimization with forecasts from Supabase
    python run_optimization.py --optimize

    # Show optimization solver status
    python run_optimization.py --status

    # Run continuous optimization every hour
    python run_optimization.py --continuous --interval 3600

Environment variables:
    SUPABASE_URL: Supabase project URL
    SUPABASE_KEY: Supabase API key
    INFLUXDB_URL: InfluxDB server URL
    INFLUXDB_TOKEN: InfluxDB authentication token
    INFLUXDB_ORG: InfluxDB organization name
    INFLUXDB_BUCKET: InfluxDB bucket name
    OPT_SITE_ID: Site identifier
    OPT_DEVICE_ID: Device identifier
    OPT_HORIZON_HOURS: Optimization horizon in hours
"""

import argparse
import json
import logging
import math
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from types import FrameType
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

from cloud.optimization import (
    BatteryConstraints,
    OptimizationConfig,
    OptimizationInputs,
    OptimizationResult,
    OptimizationSolver,
    TariffStructure,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class OptimizationRunner:
    """Runner for optimization operations.

    Provides methods for running, monitoring, and managing
    the optimization solver through the CLI.
    """

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        influxdb_url: Optional[str] = None,
        influxdb_token: Optional[str] = None,
        influxdb_org: Optional[str] = None,
        influxdb_bucket: Optional[str] = None,
        site_id: Optional[str] = None,
        device_id: Optional[str] = None,
        horizon_hours: int = 24,
        time_step_minutes: int = 60,
    ) -> None:
        """Initialize the optimization runner.

        Args:
            supabase_url: Supabase project URL (or from env)
            supabase_key: Supabase API key (or from env)
            influxdb_url: InfluxDB server URL (or from env)
            influxdb_token: InfluxDB authentication token (or from env)
            influxdb_org: InfluxDB organization name (or from env)
            influxdb_bucket: InfluxDB bucket name (or from env)
            site_id: Site identifier (or from env)
            device_id: Device identifier (or from env)
            horizon_hours: Optimization horizon in hours
            time_step_minutes: Time step granularity in minutes
        """
        self.config = OptimizationConfig(
            supabase_url=supabase_url or "",
            supabase_key=supabase_key or "",
            influxdb_url=influxdb_url or "http://localhost:8086",
            influxdb_token=influxdb_token or "",
            influxdb_org=influxdb_org or "edge-gateway",
            influxdb_bucket=influxdb_bucket or "der-data",
            site_id=site_id or "edge-gateway-site",
            device_id=device_id or "edge-gateway-001",
            enabled=True,
            horizon_hours=horizon_hours,
            time_step_minutes=time_step_minutes,
        )
        self.solver: Optional[OptimizationSolver] = None
        self._shutdown_requested = False

    def _ensure_solver(self) -> None:
        """Ensure solver is initialized."""
        if self.solver is None:
            self.solver = OptimizationSolver(self.config)

    def run_demo(self) -> OptimizationResult:
        """Run optimization with demo/sample data.

        Creates synthetic forecast and tariff data to demonstrate
        the optimization solver capabilities.

        Returns:
            OptimizationResult with optimal schedule
        """
        logger.info("Running demo optimization with sample data...")

        self._ensure_solver()

        # Create demo inputs
        inputs = self._create_demo_inputs()

        # Run optimization and store results
        result = self.solver.optimize_and_store(inputs)

        # Display results
        self._display_result(result)

        return result

    def _create_demo_inputs(self) -> OptimizationInputs:
        """Create demo optimization inputs with realistic patterns.

        Returns:
            OptimizationInputs with sample data
        """
        num_steps = self.config.num_time_steps
        start_time = datetime.now(timezone.utc).replace(
            minute=0, second=0, microsecond=0
        )

        # Generate load forecast with daily pattern
        # Peak in morning (7-9) and evening (17-21)
        load_forecast = []
        for t in range(num_steps):
            hour = (start_time + timedelta(hours=t * self.config.time_step_hours)).hour
            # Base load + time-varying component
            base_load = 2.0
            if 7 <= hour < 9:
                load = base_load + 3.0  # Morning peak
            elif 17 <= hour < 21:
                load = base_load + 4.0  # Evening peak
            elif 0 <= hour < 6:
                load = base_load - 1.0  # Night minimum
            else:
                load = base_load + 1.0  # Daytime moderate
            load_forecast.append(max(0.5, load + (t % 3) * 0.2))  # Add some variation

        # Generate solar forecast with bell curve during daylight hours
        solar_forecast = []
        for t in range(num_steps):
            hour = (start_time + timedelta(hours=t * self.config.time_step_hours)).hour
            if 6 <= hour < 18:
                # Bell curve peaking at noon
                hours_from_noon = abs(hour - 12)
                solar = 8.0 * math.exp(-0.5 * (hours_from_noon / 3) ** 2)
            else:
                solar = 0.0
            solar_forecast.append(max(0.0, solar))

        # Generate TOU tariff schedule
        tariff_schedule = []
        for t in range(num_steps):
            hour = (start_time + timedelta(hours=t * self.config.time_step_hours)).hour
            if 17 <= hour < 21:
                # Peak period (evening)
                tariff = TariffStructure(
                    time_of_use_period="peak",
                    price_per_kwh=0.35,
                    feed_in_tariff=0.08,
                )
            elif 7 <= hour < 17 or 21 <= hour < 22:
                # Shoulder period
                tariff = TariffStructure(
                    time_of_use_period="shoulder",
                    price_per_kwh=0.20,
                    feed_in_tariff=0.06,
                )
            else:
                # Off-peak period
                tariff = TariffStructure(
                    time_of_use_period="off_peak",
                    price_per_kwh=0.12,
                    feed_in_tariff=0.05,
                )
            tariff_schedule.append(tariff)

        # Battery constraints
        battery = BatteryConstraints(
            capacity_kwh=self.config.default_battery_capacity_kwh,
            max_charge_kw=self.config.default_max_charge_kw,
            max_discharge_kw=self.config.default_max_discharge_kw,
            min_soc_percent=self.config.default_min_soc_percent,
            max_soc_percent=self.config.default_max_soc_percent,
            charge_efficiency=self.config.default_charge_efficiency,
            discharge_efficiency=self.config.default_discharge_efficiency,
            initial_soc_percent=50.0,  # Start at 50% SOC
        )

        return OptimizationInputs(
            start_time=start_time,
            horizon_hours=self.config.horizon_hours,
            time_step_minutes=self.config.time_step_minutes,
            load_forecast_kw=load_forecast,
            solar_forecast_kw=solar_forecast,
            tariff_schedule=tariff_schedule,
            battery=battery,
            max_buy_kw=self.config.default_max_buy_kw,
            max_sell_kw=self.config.default_max_sell_kw,
        )

    def _display_result(self, result: OptimizationResult) -> None:
        """Display optimization result summary.

        Args:
            result: Optimization result to display
        """
        logger.info("=" * 70)
        logger.info("OPTIMIZATION RESULT SUMMARY")
        logger.info("=" * 70)
        logger.info("Run ID: %s", result.run_id)
        logger.info("Status: %s", result.status)
        logger.info("Solver Time: %.3f seconds", result.solver_time_seconds)
        logger.info("")
        logger.info("Period: %s to %s",
                   result.start_time.strftime("%Y-%m-%d %H:%M"),
                   result.end_time.strftime("%Y-%m-%d %H:%M"))
        logger.info("Horizon: %d hours (%d time steps)",
                   result.inputs.horizon_hours,
                   result.num_time_steps)
        logger.info("")

        if result.is_optimal:
            logger.info("COST SUMMARY:")
            logger.info("  Total Cost: $%.2f", result.total_cost)
            logger.info("  Grid Import: %.2f kWh", result.total_grid_import_kwh)
            logger.info("  Grid Export: %.2f kWh", result.total_grid_export_kwh)
            logger.info("")
            logger.info("SCHEDULE PREVIEW (first 6 hours):")
            logger.info("-" * 70)
            logger.info(
                "%-20s %8s %8s %8s %8s %8s",
                "Time", "Charge", "Disch", "Buy", "Sell", "SOC%"
            )
            logger.info("-" * 70)

            for point in result.schedule[:6]:
                logger.info(
                    "%-20s %8.2f %8.2f %8.2f %8.2f %8.1f",
                    point.timestamp.strftime("%Y-%m-%d %H:%M"),
                    point.charge_kw,
                    point.discharge_kw,
                    point.buy_kw,
                    point.sell_kw,
                    point.soc_percent,
                )

            if len(result.schedule) > 6:
                logger.info("... (%d more time steps)", len(result.schedule) - 6)

        else:
            logger.warning("Optimization did not find optimal solution")
            if result.message:
                logger.warning("Message: %s", result.message)

        logger.info("=" * 70)

    def show_status(self) -> dict[str, Any]:
        """Show current optimization solver status.

        Returns:
            Dictionary with status information
        """
        logger.info("Checking optimization solver status...")

        self._ensure_solver()

        health = self.solver.health_check()
        status = {
            "config": {
                "site_id": self.config.site_id,
                "device_id": self.config.device_id,
                "horizon_hours": self.config.horizon_hours,
                "time_step_minutes": self.config.time_step_minutes,
                "num_time_steps": self.config.num_time_steps,
            },
            "health": health,
            "connected": self.solver.is_connected(),
        }

        logger.info("=" * 50)
        logger.info("OPTIMIZATION SOLVER STATUS")
        logger.info("=" * 50)
        logger.info("Configuration:")
        logger.info("  Site ID: %s", self.config.site_id)
        logger.info("  Device ID: %s", self.config.device_id)
        logger.info("  Horizon: %d hours", self.config.horizon_hours)
        logger.info("  Time Step: %d minutes", self.config.time_step_minutes)
        logger.info("  Number of Steps: %d", self.config.num_time_steps)
        logger.info("")
        logger.info("Health Check:")
        logger.info("  PuLP Available: %s", health.get("pulp", False))
        logger.info("  Supabase Connected: %s", health.get("supabase", False))
        logger.info("  InfluxDB Connected: %s", health.get("influxdb", False))
        logger.info("=" * 50)

        return status

    def run_continuous(self, interval_seconds: int = 3600) -> None:
        """Run optimization continuously at specified interval.

        Args:
            interval_seconds: Seconds between optimization runs
        """
        logger.info(
            "Starting continuous optimization (interval: %d seconds)...",
            interval_seconds,
        )

        self._ensure_solver()

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        run_count = 0
        while not self._shutdown_requested:
            run_count += 1
            logger.info("Starting optimization run #%d", run_count)

            try:
                inputs = self._create_demo_inputs()
                result = self.solver.optimize_and_store(inputs)

                if result.is_optimal:
                    logger.info(
                        "Run #%d complete: cost=$%.2f, import=%.2f kWh, export=%.2f kWh",
                        run_count,
                        result.total_cost,
                        result.total_grid_import_kwh,
                        result.total_grid_export_kwh,
                    )
                else:
                    logger.warning("Run #%d: optimization not optimal", run_count)

            except Exception:
                logger.exception("Error in optimization run #%d", run_count)

            # Wait for next interval
            if not self._shutdown_requested:
                logger.info("Next run in %d seconds...", interval_seconds)
                for _ in range(interval_seconds):
                    if self._shutdown_requested:
                        break
                    time.sleep(1)

        logger.info("Continuous optimization stopped after %d runs", run_count)

    def _signal_handler(self, signum: int, frame: Optional[FrameType]) -> None:
        """Handle shutdown signals.

        Args:
            signum: Signal number
            frame: Current stack frame
        """
        logger.info("Shutdown signal received, stopping...")
        self._shutdown_requested = True

    def close(self) -> None:
        """Close solver connections."""
        if self.solver:
            self.solver.close()
            self.solver = None


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="DER Battery Scheduling Optimization Solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run demo optimization with sample data
    python run_optimization.py --demo

    # Show solver status
    python run_optimization.py --status

    # Run with custom horizon
    python run_optimization.py --demo --horizon 12

    # Output result to JSON file
    python run_optimization.py --demo --output result.json

    # Run continuous optimization
    python run_optimization.py --continuous --interval 3600
        """,
    )

    # Operation modes
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--demo",
        action="store_true",
        help="Run optimization with demo/sample data",
    )
    mode_group.add_argument(
        "--status",
        action="store_true",
        help="Show optimization solver status",
    )
    mode_group.add_argument(
        "--continuous",
        action="store_true",
        help="Run continuous optimization at specified interval",
    )

    # Database connection
    parser.add_argument(
        "--supabase-url",
        type=str,
        help="Supabase project URL (or set SUPABASE_URL)",
    )
    parser.add_argument(
        "--supabase-key",
        type=str,
        help="Supabase API key (or set SUPABASE_KEY)",
    )
    parser.add_argument(
        "--influxdb-url",
        type=str,
        default="http://localhost:8086",
        help="InfluxDB server URL (default: http://localhost:8086)",
    )
    parser.add_argument(
        "--influxdb-token",
        type=str,
        help="InfluxDB authentication token (or set INFLUXDB_TOKEN)",
    )
    parser.add_argument(
        "--influxdb-org",
        type=str,
        default="edge-gateway",
        help="InfluxDB organization name (default: edge-gateway)",
    )
    parser.add_argument(
        "--influxdb-bucket",
        type=str,
        default="der-data",
        help="InfluxDB bucket name (default: der-data)",
    )

    # Identification
    parser.add_argument(
        "--site-id",
        type=str,
        default="edge-gateway-site",
        help="Site identifier (default: edge-gateway-site)",
    )
    parser.add_argument(
        "--device-id",
        type=str,
        default="edge-gateway-001",
        help="Device identifier (default: edge-gateway-001)",
    )

    # Optimization settings
    parser.add_argument(
        "--horizon",
        type=int,
        default=24,
        help="Optimization horizon in hours (default: 24)",
    )
    parser.add_argument(
        "--time-step",
        type=int,
        default=60,
        help="Time step in minutes (default: 60)",
    )

    # Continuous mode settings
    parser.add_argument(
        "--interval",
        type=int,
        default=3600,
        help="Interval between runs in seconds for continuous mode (default: 3600)",
    )

    # Output settings
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file for optimization result (JSON)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    args = parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        runner = OptimizationRunner(
            supabase_url=args.supabase_url,
            supabase_key=args.supabase_key,
            influxdb_url=args.influxdb_url,
            influxdb_token=args.influxdb_token,
            influxdb_org=args.influxdb_org,
            influxdb_bucket=args.influxdb_bucket,
            site_id=args.site_id,
            device_id=args.device_id,
            horizon_hours=args.horizon,
            time_step_minutes=args.time_step,
        )

        if args.demo:
            result = runner.run_demo()

            # Save to file if requested
            if args.output:
                with open(args.output, "w") as f:
                    f.write(result.to_json())
                logger.info("Result saved to %s", args.output)

            return 0 if result.is_optimal else 1

        elif args.status:
            runner.show_status()
            return 0

        elif args.continuous:
            runner.run_continuous(interval_seconds=args.interval)
            return 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0

    except Exception:
        logger.exception("Fatal error in optimization runner")
        return 1

    finally:
        if "runner" in locals():
            runner.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
