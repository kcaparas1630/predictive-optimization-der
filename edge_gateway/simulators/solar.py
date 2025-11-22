"""Solar generation simulator with realistic time-of-day and weather patterns."""

import math
from datetime import datetime
from typing import Optional

from edge_gateway.models import SolarData
from .base import BaseSimulator


class SolarSimulator(BaseSimulator):
    """
    Simulates solar panel generation with realistic patterns.

    Models:
    - Daily solar arc (sunrise to sunset)
    - Seasonal variations
    - Cloud cover effects
    - Panel temperature effects on efficiency
    - Random weather variations
    """

    def __init__(
        self,
        panel_capacity_kw: float = 10.0,
        latitude: float = 37.7749,  # San Francisco default
        panel_efficiency: float = 0.20,  # 20% base efficiency
        temp_coefficient: float = -0.004,  # -0.4% per degree C above 25
        seed: Optional[int] = None,
    ):
        """
        Initialize solar simulator.

        Args:
            panel_capacity_kw: Installed solar panel capacity in kW
            latitude: Location latitude (affects day length and sun angle)
            panel_efficiency: Base panel efficiency (0-1)
            temp_coefficient: Temperature coefficient (efficiency change per degree C)
            seed: Random seed for reproducibility
        """
        super().__init__(seed)
        self.panel_capacity_kw = panel_capacity_kw
        self.latitude = latitude
        self.panel_efficiency = panel_efficiency
        self.temp_coefficient = temp_coefficient

        # Weather state (persists across calls for more realistic patterns)
        self._cloud_cover = 0.0  # 0-1
        self._weather_change_hour = -1

    def _get_day_of_year(self, timestamp: datetime) -> int:
        """Get day of year (1-365)."""
        return timestamp.timetuple().tm_yday

    def _calculate_day_length(self, day_of_year: int) -> tuple[float, float]:
        """
        Calculate sunrise and sunset hours for the given day.

        Returns:
            Tuple of (sunrise_hour, sunset_hour) in decimal hours
        """
        # Simplified solar calculation
        lat_rad = math.radians(self.latitude)

        # Solar declination angle
        declination = 23.45 * math.sin(math.radians((360 / 365) * (day_of_year - 81)))
        decl_rad = math.radians(declination)

        # Hour angle at sunrise/sunset
        cos_hour_angle = -math.tan(lat_rad) * math.tan(decl_rad)
        cos_hour_angle = self._clamp(cos_hour_angle, -1, 1)
        hour_angle = math.degrees(math.acos(cos_hour_angle))

        # Day length in hours
        day_length = 2 * hour_angle / 15

        # Solar noon is approximately 12:00
        sunrise = 12 - day_length / 2
        sunset = 12 + day_length / 2

        return sunrise, sunset

    def _calculate_sun_position(self, timestamp: datetime) -> float:
        """
        Calculate sun elevation factor (0-1) for the given time.

        Returns:
            Sun elevation factor where 1 is solar noon peak
        """
        day_of_year = self._get_day_of_year(timestamp)
        sunrise, sunset = self._calculate_day_length(day_of_year)

        hour = timestamp.hour + timestamp.minute / 60

        if hour < sunrise or hour > sunset:
            return 0.0

        # Normalize to 0-1 range with sine curve for sun arc
        day_progress = (hour - sunrise) / (sunset - sunrise)
        sun_factor = math.sin(day_progress * math.pi)

        # Account for seasonal intensity variation
        # Summer has higher peak irradiance
        season_factor = 0.8 + 0.2 * math.cos(math.radians((day_of_year - 172) * 360 / 365))

        return sun_factor * season_factor

    def _update_weather(self, timestamp: datetime) -> None:
        """Update weather conditions periodically."""
        if timestamp.hour != self._weather_change_hour:
            self._weather_change_hour = timestamp.hour
            # Weather changes slowly - adjust by small random amount
            change = self._random.gauss(0, 0.1)
            self._cloud_cover = self._clamp(self._cloud_cover + change, 0, 0.9)

            # Occasional sudden weather changes
            if self._random.random() < 0.05:  # 5% chance
                self._cloud_cover = self._random.uniform(0, 0.8)

    def _calculate_irradiance(self, sun_factor: float) -> float:
        """
        Calculate solar irradiance in W/m².

        Args:
            sun_factor: Sun position factor (0-1)

        Returns:
            Irradiance in W/m²
        """
        # Maximum clear sky irradiance approximately 1000 W/m²
        max_irradiance = 1000

        # Apply cloud cover reduction
        cloud_factor = 1 - (self._cloud_cover * 0.8)  # Max 80% reduction from clouds

        irradiance = max_irradiance * sun_factor * cloud_factor
        return self._add_noise(irradiance, 3.0) if irradiance > 0 else 0

    def _calculate_panel_temperature(self, irradiance: float, ambient_temp: float = 25.0) -> float:
        """
        Estimate panel temperature based on irradiance.

        Panels typically run 20-30C above ambient in full sun.
        """
        temp_rise = (irradiance / 1000) * 25  # Up to 25C rise at full irradiance
        panel_temp = ambient_temp + temp_rise
        return self._add_noise(panel_temp, 2.0)

    def _calculate_efficiency(self, panel_temp: float) -> float:
        """
        Calculate actual efficiency accounting for temperature.

        Efficiency decreases as temperature increases above 25C.
        """
        reference_temp = 25.0
        temp_delta = panel_temp - reference_temp
        efficiency_multiplier = 1 + (self.temp_coefficient * temp_delta)
        actual_efficiency = self.panel_efficiency * efficiency_multiplier
        return self._clamp(actual_efficiency, 0.05, 0.25)

    def generate(self, timestamp: datetime) -> SolarData:
        """
        Generate solar generation data for the given timestamp.

        Args:
            timestamp: The timestamp for which to generate data

        Returns:
            SolarData with realistic generation values
        """
        self._update_weather(timestamp)

        sun_factor = self._calculate_sun_position(timestamp)
        irradiance = self._calculate_irradiance(sun_factor)
        panel_temp = self._calculate_panel_temperature(irradiance)
        efficiency = self._calculate_efficiency(panel_temp)

        # Calculate actual generation
        # Power = Irradiance * Area * Efficiency
        # Simplified: capacity represents area * efficiency at STC
        if irradiance > 0:
            generation = self.panel_capacity_kw * (irradiance / 1000) * (efficiency / self.panel_efficiency)
            generation = self._add_noise(generation, 2.0)
            generation = self._clamp(generation, 0, self.panel_capacity_kw)
        else:
            generation = 0

        return SolarData(
            generation_kw=generation,
            irradiance_w_m2=irradiance,
            panel_temp_celsius=panel_temp,
            efficiency_percent=efficiency * 100,
        )
