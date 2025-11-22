"""Home load simulator with realistic daily consumption patterns."""

import math
import random
from datetime import datetime
from typing import Optional

from edge_gateway.models import HomeLoadData
from .base import BaseSimulator


class HomeLoadSimulator(BaseSimulator):
    """
    Simulates home electrical load with realistic patterns.

    Models:
    - Daily load curves (morning/evening peaks)
    - HVAC patterns (temperature-dependent)
    - Appliance usage patterns
    - EV charging (if enabled)
    - Weekend vs weekday differences
    - Seasonal variations
    """

    def __init__(
        self,
        base_load_kw: float = 0.5,  # Always-on load (fridge, standby, etc.)
        peak_load_kw: float = 8.0,  # Maximum expected load
        *,
        has_ev: bool = True,
        ev_charging_kw: float = 7.2,  # Level 2 charger
        hvac_capacity_kw: float = 3.5,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize home load simulator.

        Args:
            base_load_kw: Constant base load (always on)
            peak_load_kw: Maximum expected total load
            has_ev: Whether home has an EV
            ev_charging_kw: EV charger power when active
            hvac_capacity_kw: HVAC system capacity
            seed: Random seed for reproducibility
        """
        super().__init__(seed)
        self.base_load_kw = base_load_kw
        self.peak_load_kw = peak_load_kw
        self.has_ev = has_ev
        self.ev_charging_kw = ev_charging_kw
        self.hvac_capacity_kw = hvac_capacity_kw

        # Store base seed for deterministic day-based calculations
        self._base_seed = seed if seed is not None else self._random.randint(0, 1000000)

    def _is_weekend(self, timestamp: datetime) -> bool:
        """Check if timestamp is on a weekend."""
        return timestamp.weekday() >= 5

    def _get_occupancy_factor(self, timestamp: datetime) -> float:
        """
        Get home occupancy factor (0-1) based on time.

        Higher occupancy = more load.
        """
        hour = timestamp.hour
        is_weekend = self._is_weekend(timestamp)

        if is_weekend:
            # Weekend - people home more during day
            if 8 <= hour < 23:
                return 0.8 + self._random.uniform(-0.1, 0.1)
            elif 23 <= hour or hour < 8:
                return 0.3
        else:
            # Weekday - morning and evening peaks
            if 6 <= hour < 9:  # Morning rush
                return 0.7
            elif 9 <= hour < 17:  # Work hours
                return 0.2
            elif 17 <= hour < 22:  # Evening peak
                return 0.9
            else:
                return 0.3


    def _calculate_hvac_load(self, timestamp: datetime) -> float:
        """
        Calculate HVAC load based on time and season.

        Summer = cooling, Winter = heating.
        """
        day_of_year = timestamp.timetuple().tm_yday
        hour = timestamp.hour

        # Seasonal factor: 1 in summer/winter, 0.3 in spring/fall
        # Peak cooling around day 200 (July), peak heating around day 15 (January)
        season_rad = (day_of_year - 15) * 2 * math.pi / 365
        season_factor = 0.3 + 0.7 * abs(math.cos(season_rad))

        # Time of day factor - more HVAC in afternoon (cooling) or morning (heating)
        if 100 < day_of_year < 280:  # Cooling season
            # Peak afternoon
            if 13 <= hour < 18:
                time_factor = 1.0
            elif 10 <= hour < 13 or 18 <= hour < 21:
                time_factor = 0.6
            else:
                time_factor = 0.3
        else:  # Heating season
            # Peak morning and evening
            if 6 <= hour < 9 or 17 <= hour < 22:
                time_factor = 0.8
            else:
                time_factor = 0.4

        hvac_load = self.hvac_capacity_kw * season_factor * time_factor
        return self._add_noise(hvac_load, 10.0)

    def _calculate_appliance_load(self, timestamp: datetime, occupancy: float) -> float:
        """
        Calculate appliance load based on occupancy and time.

        Includes dishwasher, washing machine, dryer, cooking, etc.
        """
        hour = timestamp.hour

        # Cooking times
        cooking_load = 0.0
        if 7 <= hour < 9:  # Breakfast
            cooking_load = self._random.uniform(0.5, 1.5)
        elif 11 <= hour < 13:  # Lunch (weekends)
            if self._is_weekend(timestamp):
                cooking_load = self._random.uniform(0.5, 1.0)
        elif 17 <= hour < 20:  # Dinner
            cooking_load = self._random.uniform(1.0, 2.5)

        # Random appliance events
        appliance_base = 0.3 * occupancy

        # Occasional high-load appliances (laundry, etc.)
        if self._random.random() < 0.05:  # 5% chance each interval
            appliance_base += self._random.uniform(1.0, 2.0)

        return self._add_noise(appliance_base + cooking_load, 15.0)

    def _calculate_lighting_load(self, timestamp: datetime, occupancy: float) -> float:
        """
        Calculate lighting load based on time of day and occupancy.
        """
        hour = timestamp.hour

        # Natural light factor (0 at night, 1 at noon)
        if 6 <= hour < 20:
            natural_light = math.sin((hour - 6) * math.pi / 14)
        else:
            natural_light = 0

        # Lighting needed = occupancy * (1 - natural_light)
        lighting_factor = occupancy * (1 - natural_light * 0.8)
        max_lighting = 0.8  # kW

        return self._add_noise(max_lighting * lighting_factor, 5.0)

    def _calculate_ev_charging(self, timestamp: datetime) -> float:
        """
        Calculate EV charging load.

        Uses deterministic daily pattern based on timestamp only,
        making it safe for out-of-order or concurrent calls.

        Args:
            timestamp: Current timestamp

        Returns:
            EV charging power in kW (0 if not charging)
        """
        if not self.has_ev:
            return 0.0

        hour = timestamp.hour

        # Deterministic daily pattern based on timestamp only
        # Use day ordinal + base seed for reproducible but varying daily patterns
        day_seed = timestamp.date().toordinal() + self._base_seed
        day_random = random.Random(day_seed)

        # 70% chance EV needs charge on any given day
        needs_charge = day_random.random() < 0.7

        if not needs_charge:
            return 0.0

        # Prefer overnight charging (cheap electricity)
        if 22 <= hour or hour < 6:
            return self.ev_charging_kw
        # Or charge when arriving home (evening)
        elif 18 <= hour < 22:
            # Use hour-specific seed for deterministic evening charging decision
            hour_random = random.Random(day_seed + hour)
            if hour_random.random() < 0.3:
                return self.ev_charging_kw

        return 0.0

    def generate(self, timestamp: datetime) -> HomeLoadData:
        """
        Generate home load data for the given timestamp.

        Args:
            timestamp: The timestamp for which to generate data

        Returns:
            HomeLoadData with realistic consumption values
        """
        occupancy = self._get_occupancy_factor(timestamp)

        hvac = self._calculate_hvac_load(timestamp)
        appliances = self._calculate_appliance_load(timestamp, occupancy)
        lighting = self._calculate_lighting_load(timestamp, occupancy)
        ev_charging = self._calculate_ev_charging(timestamp)
        other = self.base_load_kw + self._random.uniform(-0.1, 0.2)

        unclamped_total = hvac + appliances + lighting + ev_charging + other
        total = self._clamp(unclamped_total, self.base_load_kw, self.peak_load_kw)

        # Scale components proportionally if clamped
        if unclamped_total > 0 and total != unclamped_total:
            scale = total / unclamped_total
            hvac *= scale
            appliances *= scale
            lighting *= scale
            ev_charging *= scale
            other *= scale

        return HomeLoadData(
            total_load_kw=total,
            hvac_kw=max(0, hvac),
            appliances_kw=max(0, appliances),
            lighting_kw=max(0, lighting),
            ev_charging_kw=max(0, ev_charging),
            other_kw=max(0, other),
        )
