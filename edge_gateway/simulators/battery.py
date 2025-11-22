"""Battery state of charge simulator with realistic charge/discharge patterns."""

from datetime import datetime
from typing import Optional

from edge_gateway.models import BatteryData
from .base import BaseSimulator


class BatterySimulator(BaseSimulator):
    """
    Simulates battery state with realistic behavior.

    Models:
    - State of Charge (SoC) management
    - Charge/discharge power limits
    - Temperature effects
    - Battery degradation over time
    - Round-trip efficiency losses
    """

    def __init__(
        self,
        capacity_kwh: float = 13.5,  # Tesla Powerwall-like capacity
        max_charge_rate_kw: float = 5.0,
        max_discharge_rate_kw: float = 5.0,
        round_trip_efficiency: float = 0.90,
        initial_soc: float = 50.0,
        min_soc: float = 10.0,
        max_soc: float = 90.0,
        initial_cycles: int = 100,
        seed: Optional[int] = None,
    ):
        """
        Initialize battery simulator.

        Args:
            capacity_kwh: Battery capacity in kWh
            max_charge_rate_kw: Maximum charging power in kW
            max_discharge_rate_kw: Maximum discharging power in kW
            round_trip_efficiency: Round-trip efficiency (0-1)
            initial_soc: Initial state of charge (0-100)
            min_soc: Minimum allowed SoC (0-100)
            max_soc: Maximum allowed SoC (0-100)
            initial_cycles: Initial cycle count
            seed: Random seed for reproducibility
        """
        super().__init__(seed)
        self.capacity_kwh = capacity_kwh
        self.max_charge_rate_kw = max_charge_rate_kw
        self.max_discharge_rate_kw = max_discharge_rate_kw
        self.round_trip_efficiency = round_trip_efficiency
        self.min_soc = min_soc
        self.max_soc = max_soc

        # Mutable state
        self._soc = initial_soc
        self._cycles = initial_cycles
        self._temperature = 25.0
        self._last_timestamp: Optional[datetime] = None
        self._energy_throughput = 0.0  # kWh for cycle counting

    def _calculate_health(self) -> float:
        """
        Calculate battery health based on cycle count.

        Batteries typically retain 80% capacity after 3000-5000 cycles.
        """
        # Linear degradation model (simplified)
        cycles_to_80_percent = 4000
        degradation_per_cycle = 20.0 / cycles_to_80_percent
        health = 100.0 - (self._cycles * degradation_per_cycle)
        return self._clamp(health, 50.0, 100.0)

    def _calculate_voltage(self, soc: float) -> float:
        """
        Calculate battery voltage based on SoC.

        Models a typical lithium battery voltage curve.
        """
        # Simplified voltage curve: 48V nominal system
        min_voltage = 44.0
        max_voltage = 54.4
        # Non-linear voltage curve
        voltage = min_voltage + (max_voltage - min_voltage) * (soc / 100) ** 0.8
        return self._add_noise(voltage, 0.5)

    def _update_temperature(self, power_kw: float, ambient_temp: float = 25.0) -> float:
        """
        Update battery temperature based on power flow.

        Higher power = more heat generation.
        """
        # Heat generation proportional to power squared
        heat_factor = (abs(power_kw) / self.max_charge_rate_kw) ** 2 * 10
        target_temp = ambient_temp + heat_factor

        # Temperature changes slowly
        self._temperature = self._temperature + (target_temp - self._temperature) * 0.1
        return self._add_noise(self._temperature, 1.0)

    def _get_power_limits(self, soc: float) -> tuple[float, float]:
        """
        Get current charge/discharge power limits based on SoC.

        Limits taper near full/empty to protect battery.
        """
        # Charge rate tapers above 80% SoC
        if soc > 80:
            charge_factor = 1 - ((soc - 80) / 20) * 0.7
        else:
            charge_factor = 1.0

        # Discharge rate tapers below 20% SoC
        if soc < 20:
            discharge_factor = soc / 20
        else:
            discharge_factor = 1.0

        max_charge = self.max_charge_rate_kw * charge_factor
        max_discharge = self.max_discharge_rate_kw * discharge_factor

        return max_charge, max_discharge

    def update_soc(
        self,
        power_kw: float,
        duration_hours: float,
    ) -> float:
        """
        Update SoC based on power flow over duration.

        Args:
            power_kw: Power flow (positive = charging, negative = discharging)
            duration_hours: Duration of power flow in hours

        Returns:
            Actual power applied (may be limited by constraints)
        """
        max_charge, max_discharge = self._get_power_limits(self._soc)

        # Apply power limits
        if power_kw > 0:
            # Charging - apply efficiency loss
            actual_power = min(power_kw, max_charge)
            energy_stored = actual_power * duration_hours * self.round_trip_efficiency
        else:
            # Discharging
            actual_power = max(power_kw, -max_discharge)
            energy_stored = actual_power * duration_hours

        # Update SoC
        soc_change = (energy_stored / self.capacity_kwh) * 100
        new_soc = self._soc + soc_change
        self._soc = self._clamp(new_soc, self.min_soc, self.max_soc)

        # Track energy throughput for cycle counting
        self._energy_throughput += abs(energy_stored)
        if self._energy_throughput >= self.capacity_kwh:
            self._cycles += int(self._energy_throughput / self.capacity_kwh)
            self._energy_throughput %= self.capacity_kwh

        return actual_power

    def generate(
        self,
        timestamp: datetime,
        solar_generation_kw: float = 0.0,
        home_load_kw: float = 0.0,
        grid_price: float = 0.15,
    ) -> BatteryData:
        """
        Generate battery state data with intelligent power management.

        The battery uses a simple optimization strategy:
        - Charge from excess solar
        - Discharge during high price periods when solar is low
        - Maintain SoC for grid backup

        Args:
            timestamp: The timestamp for which to generate data
            solar_generation_kw: Current solar generation
            home_load_kw: Current home load
            grid_price: Current grid price ($/kWh)

        Returns:
            BatteryData with current state
        """
        # Calculate time delta for SoC updates
        if self._last_timestamp is not None:
            duration_hours = (timestamp - self._last_timestamp).total_seconds() / 3600
        else:
            duration_hours = 1 / 12  # Assume 5-minute intervals initially

        self._last_timestamp = timestamp

        # Simple battery management strategy
        net_home = home_load_kw - solar_generation_kw

        if net_home < 0:
            # Excess solar - charge battery
            target_power = min(-net_home, self.max_charge_rate_kw)
        elif grid_price > 0.20 and self._soc > 30:
            # High price period - discharge to offset load
            target_power = -min(net_home, self.max_discharge_rate_kw)
        elif self._soc < 30:
            # Low SoC - charge from grid during low price
            if grid_price < 0.10:
                target_power = self.max_charge_rate_kw * 0.5
            else:
                target_power = 0
        else:
            # Default - small charge/discharge to maintain grid balance
            target_power = -net_home * 0.3

        # Apply power change
        actual_power = self.update_soc(target_power, duration_hours)

        # Update temperature
        temperature = self._update_temperature(actual_power)

        return BatteryData(
            soc_percent=self._soc,
            capacity_kwh=self.capacity_kwh,
            power_kw=actual_power,
            voltage_v=self._calculate_voltage(self._soc),
            temperature_celsius=temperature,
            cycles=self._cycles,
            health_percent=self._calculate_health(),
        )
