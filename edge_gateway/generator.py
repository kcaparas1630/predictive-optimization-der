"""
DER Data Generator - Main module for simulating realistic DER data.

Supports continuous generation and scheduled execution.
"""

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

from edge_gateway.models import DERData
from edge_gateway.simulators import (
    SolarSimulator,
    BatterySimulator,
    HomeLoadSimulator,
    GridPriceSimulator,
)

logger = logging.getLogger(__name__)


class DERDataGenerator:
    """
    Main data generator that orchestrates all simulators.

    Produces coordinated DER data snapshots with realistic
    interactions between solar, battery, load, and grid.
    """

    def __init__(
        self,
        device_id: str = "edge-gateway-001",
        # Solar config
        solar_capacity_kw: float = 10.0,
        latitude: float = 37.7749,
        # Battery config
        battery_capacity_kwh: float = 13.5,
        battery_max_power_kw: float = 5.0,
        # Home load config
        base_load_kw: float = 0.5,
        peak_load_kw: float = 8.0,
        has_ev: bool = True,
        # Grid config
        off_peak_price: float = 0.08,
        shoulder_price: float = 0.15,
        peak_price: float = 0.30,
        # General
        seed: Optional[int] = None,
    ):
        """
        Initialize the DER data generator.

        Args:
            device_id: Unique identifier for this edge gateway
            solar_capacity_kw: Installed solar panel capacity
            latitude: Location latitude for solar calculations
            battery_capacity_kwh: Battery capacity
            battery_max_power_kw: Battery max charge/discharge power
            base_load_kw: Base home load
            peak_load_kw: Peak home load
            has_ev: Whether home has an EV
            off_peak_price: Off-peak electricity price
            shoulder_price: Shoulder period price
            peak_price: Peak period price
            seed: Random seed for reproducibility
        """
        self.device_id = device_id

        # Initialize simulators
        self.solar = SolarSimulator(
            panel_capacity_kw=solar_capacity_kw,
            latitude=latitude,
            seed=seed,
        )

        self.battery = BatterySimulator(
            capacity_kwh=battery_capacity_kwh,
            max_charge_rate_kw=battery_max_power_kw,
            max_discharge_rate_kw=battery_max_power_kw,
            seed=seed + 1 if seed else None,
        )

        self.home_load = HomeLoadSimulator(
            base_load_kw=base_load_kw,
            peak_load_kw=peak_load_kw,
            has_ev=has_ev,
            seed=seed + 2 if seed else None,
        )

        self.grid_price = GridPriceSimulator(
            off_peak_price=off_peak_price,
            shoulder_price=shoulder_price,
            peak_price=peak_price,
            seed=seed + 3 if seed else None,
        )

    def generate(self, timestamp: Optional[datetime] = None) -> DERData:
        """
        Generate a complete DER data snapshot.

        Args:
            timestamp: Timestamp for the data. Defaults to current time.

        Returns:
            DERData containing all simulated values
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Generate base data
        solar_data = self.solar.generate(timestamp)
        load_data = self.home_load.generate(timestamp)
        price_data = self.grid_price.generate(timestamp)

        # Battery responds to solar, load, and price
        battery_data = self.battery.generate(
            timestamp=timestamp,
            solar_generation_kw=solar_data.generation_kw,
            home_load_kw=load_data.total_load_kw,
            grid_price=price_data.price_per_kwh,
        )

        return DERData(
            timestamp=timestamp,
            device_id=self.device_id,
            solar=solar_data,
            battery=battery_data,
            home_load=load_data,
            grid_price=price_data,
        )

    def generate_historical(
        self,
        start: datetime,
        end: datetime,
        interval_minutes: int = 5,
    ) -> list[DERData]:
        """
        Generate historical data for a time range.

        Args:
            start: Start timestamp
            end: End timestamp
            interval_minutes: Interval between data points

        Returns:
            List of DERData snapshots
        """
        data = []
        current = start
        interval = timedelta(minutes=interval_minutes)

        while current <= end:
            data.append(self.generate(current))
            current += interval

        return data


class DataGeneratorRunner:
    """
    Runner for continuous or scheduled data generation.
    """

    def __init__(
        self,
        generator: DERDataGenerator,
        output_callback: Optional[Callable[[DERData], None]] = None,
        output_file: Optional[Path] = None,
    ):
        """
        Initialize the runner.

        Args:
            generator: The DER data generator
            output_callback: Optional callback function for each data point
            output_file: Optional file path to append JSON data
        """
        self.generator = generator
        self.output_callback = output_callback
        self.output_file = output_file
        self._running = False

    def _output_data(self, data: DERData) -> None:
        """Output data to configured destinations."""
        # Log summary
        logger.info(
            f"Generated: Solar={data.solar.generation_kw:.2f}kW, "
            f"Battery={data.battery.soc_percent:.1f}%, "
            f"Load={data.home_load.total_load_kw:.2f}kW, "
            f"Grid={data.net_grid_flow_kw:.2f}kW"
        )

        # Callback
        if self.output_callback:
            self.output_callback(data)

        # File output
        if self.output_file:
            try:
                with open(self.output_file, "a") as f:
                    f.write(data.to_json(indent=None) + "\n")
            except Exception as e:
                logger.error(f"Failed to write to {self.output_file}: {e}")

    def run_once(self) -> DERData:
        """Generate and output a single data point."""
        data = self.generator.generate()
        self._output_data(data)
        return data

    def run_continuous(
        self,
        interval_seconds: float = 300,  # 5 minutes default
        duration_seconds: Optional[float] = None,
    ) -> None:
        """
        Run generator continuously.

        Args:
            interval_seconds: Seconds between data points
            duration_seconds: Optional total duration. None = run forever.
        """
        self._running = True
        start_time = time.time()

        logger.info(f"Starting continuous generation every {interval_seconds}s")

        try:
            while self._running:
                self.run_once()

                # Check duration
                if duration_seconds is not None:
                    elapsed = time.time() - start_time
                    if elapsed >= duration_seconds:
                        logger.info("Duration reached, stopping")
                        break

                # Wait for next interval
                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self._running = False

    def run_scheduled(
        self,
        interval_seconds: float = 300,
        align_to_interval: bool = True,
    ) -> None:
        """
        Run generator on a schedule aligned to clock intervals.

        For example, with interval_seconds=300 and align_to_interval=True,
        data will be generated at :00, :05, :10, etc.

        Args:
            interval_seconds: Seconds between data points
            align_to_interval: Whether to align to clock intervals
        """
        self._running = True

        logger.info(f"Starting scheduled generation every {interval_seconds}s")

        try:
            while self._running:
                if align_to_interval:
                    # Calculate time until next interval
                    now = time.time()
                    next_interval = (now // interval_seconds + 1) * interval_seconds
                    sleep_time = next_interval - now

                    if sleep_time > 0:
                        time.sleep(sleep_time)

                self.run_once()

                if not align_to_interval:
                    time.sleep(interval_seconds)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self._running = False

    def stop(self) -> None:
        """Stop the runner."""
        self._running = False
