"""Grid price signal simulator with time-of-use and dynamic pricing."""

import math
from datetime import datetime
from typing import Optional

from edge_gateway.models import GridPriceData
from .base import BaseSimulator


class GridPriceSimulator(BaseSimulator):
    """
    Simulates grid electricity prices with realistic patterns.

    Models:
    - Time-of-Use (TOU) pricing tiers
    - Dynamic real-time pricing variations
    - Feed-in tariffs for exported energy
    - Demand charges
    - Carbon intensity variations
    """

    def __init__(
        self,
        # TOU prices ($/kWh)
        off_peak_price: float = 0.08,
        shoulder_price: float = 0.15,
        peak_price: float = 0.30,
        # Feed-in tariff ($/kWh)
        base_feed_in_tariff: float = 0.05,
        # Demand charge ($/kW)
        demand_charge: float = 10.0,
        # Price volatility (for real-time pricing component)
        volatility: float = 0.15,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize grid price simulator.

        Args:
            off_peak_price: Off-peak electricity price ($/kWh)
            shoulder_price: Shoulder period price ($/kWh)
            peak_price: Peak period price ($/kWh)
            base_feed_in_tariff: Base feed-in tariff for solar export ($/kWh)
            demand_charge: Demand charge per kW of peak demand ($/kW)
            volatility: Price volatility factor for dynamic pricing
            seed: Random seed for reproducibility
        """
        super().__init__(seed)
        self.off_peak_price = off_peak_price
        self.shoulder_price = shoulder_price
        self.peak_price = peak_price
        self.base_feed_in_tariff = base_feed_in_tariff
        self.demand_charge = demand_charge
        self.volatility = volatility

        # Dynamic pricing state
        self._price_offset = 0.0
        self._last_hour = -1

    def _get_tou_period(self, timestamp: datetime) -> tuple[str, float]:
        """
        Determine TOU period and base price for the given time.

        Returns:
            Tuple of (period_name, base_price)
        """
        hour = timestamp.hour
        is_weekend = timestamp.weekday() >= 5

        if is_weekend:
            # Weekends: off-peak all day except evening
            if 17 <= hour < 21:
                return "shoulder", self.shoulder_price
            else:
                return "off_peak", self.off_peak_price
        else:
            # Weekdays: standard TOU schedule
            if 0 <= hour < 7:
                return "off_peak", self.off_peak_price
            elif 7 <= hour < 14:
                return "shoulder", self.shoulder_price
            elif 14 <= hour < 20:
                return "peak", self.peak_price
            elif 20 <= hour < 22:
                return "shoulder", self.shoulder_price
            else:
                return "off_peak", self.off_peak_price

    def _update_dynamic_price(self, timestamp: datetime) -> None:
        """
        Update dynamic price component.

        Prices fluctuate based on simulated wholesale market conditions.
        """
        if timestamp.hour != self._last_hour:
            self._last_hour = timestamp.hour
            # Random walk for price offset
            change = self._random.gauss(0, self.volatility * 0.1)
            self._price_offset += change
            # Mean reversion
            self._price_offset *= 0.95
            self._price_offset = self._clamp(self._price_offset, -0.10, 0.10)

    def _calculate_carbon_intensity(self, timestamp: datetime) -> float:
        """
        Calculate grid carbon intensity based on time.

        Carbon intensity varies with grid mix:
        - Lower during day (more solar/wind)
        - Higher during evening peak (gas peakers)
        - Varies by season
        """
        hour = timestamp.hour
        day_of_year = timestamp.timetuple().tm_yday

        # Base carbon intensity (gCO2/kWh)
        base_intensity = 400

        # Time of day factor
        # Lower during midday (solar), higher in evening (gas peakers)
        if 10 <= hour < 16:
            time_factor = 0.7  # Solar reduces intensity
        elif 17 <= hour < 21:
            time_factor = 1.3  # Peak demand, more gas
        else:
            time_factor = 1.0

        # Seasonal factor - summer has more solar
        season_factor = 1.0 - 0.2 * math.cos((day_of_year - 172) * 2 * math.pi / 365)

        intensity = base_intensity * time_factor * season_factor
        return self._add_noise(intensity, 5.0)

    def _calculate_feed_in_tariff(self, timestamp: datetime, grid_price: float) -> float:
        """
        Calculate feed-in tariff for exported energy.

        Some regions have time-varying feed-in tariffs.
        """
        hour = timestamp.hour

        # Higher FIT during peak demand periods
        if 14 <= hour < 20:
            fit_multiplier = 1.5
        elif 10 <= hour < 14:
            fit_multiplier = 1.0  # Midday - lots of solar, lower value
        else:
            fit_multiplier = 0.8

        # FIT is typically a fraction of retail price
        fit = min(self.base_feed_in_tariff * fit_multiplier, grid_price * 0.5)
        return self._add_noise(fit, 3.0)

    def generate(self, timestamp: datetime) -> GridPriceData:
        """
        Generate grid price data for the given timestamp.

        Args:
            timestamp: The timestamp for which to generate data

        Returns:
            GridPriceData with current prices
        """
        self._update_dynamic_price(timestamp)

        tou_period, base_price = self._get_tou_period(timestamp)

        # Apply dynamic pricing component
        dynamic_price = base_price * (1 + self._price_offset)
        dynamic_price = self._clamp(dynamic_price, 0.01, 1.0)

        # Add small random variation
        final_price = self._add_noise(dynamic_price, 2.0)

        feed_in = self._calculate_feed_in_tariff(timestamp, final_price)
        carbon = self._calculate_carbon_intensity(timestamp)

        return GridPriceData(
            price_per_kwh=final_price,
            feed_in_tariff=feed_in,
            demand_charge=self.demand_charge,
            time_of_use_period=tou_period,
            carbon_intensity_g_kwh=carbon,
        )
