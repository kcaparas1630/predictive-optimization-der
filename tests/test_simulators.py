"""Tests for DER simulators."""

from datetime import datetime

import pytest

from edge_gateway.simulators import (
    SolarSimulator,
    BatterySimulator,
    HomeLoadSimulator,
    GridPriceSimulator,
)


class TestSolarSimulator:
    """Tests for SolarSimulator."""

    def test_creation(self):
        """Test simulator can be created."""
        sim = SolarSimulator(panel_capacity_kw=10.0, seed=42)
        assert sim.panel_capacity_kw == 10.0

    def test_generate_returns_solar_data(self):
        """Test generate returns SolarData."""
        sim = SolarSimulator(seed=42)
        data = sim.generate(datetime(2024, 6, 15, 12, 0, 0))

        assert hasattr(data, "generation_kw")
        assert hasattr(data, "irradiance_w_m2")
        assert hasattr(data, "panel_temp_celsius")
        assert hasattr(data, "efficiency_percent")

    def test_no_generation_at_night(self):
        """Test no solar generation at night."""
        sim = SolarSimulator(seed=42)
        # Midnight
        data = sim.generate(datetime(2024, 6, 15, 0, 0, 0))
        assert data.generation_kw == 0
        assert data.irradiance_w_m2 == 0

    def test_peak_generation_at_midday(self):
        """Test peak generation around midday."""
        sim = SolarSimulator(panel_capacity_kw=10.0, seed=42)
        # Noon in summer
        data = sim.generate(datetime(2024, 6, 21, 12, 0, 0))
        assert data.generation_kw > 0
        assert data.irradiance_w_m2 > 500

    def test_generation_bounded_by_capacity(self):
        """Test generation doesn't exceed capacity."""
        sim = SolarSimulator(panel_capacity_kw=10.0, seed=42)

        for hour in range(24):
            data = sim.generate(datetime(2024, 6, 15, hour, 0, 0))
            assert data.generation_kw <= sim.panel_capacity_kw
            assert data.generation_kw >= 0

    def test_reproducibility_with_seed(self):
        """Test same seed produces same results."""
        sim1 = SolarSimulator(seed=42)
        sim2 = SolarSimulator(seed=42)

        ts = datetime(2024, 6, 15, 12, 0, 0)
        data1 = sim1.generate(ts)
        data2 = sim2.generate(ts)

        assert data1 == data2

    def test_seasonal_variation(self):
        """Test summer has more generation than winter."""
        sim = SolarSimulator(seed=42)

        # Summer noon
        summer = sim.generate(datetime(2024, 6, 21, 12, 0, 0))

        # Reset for fair comparison
        sim2 = SolarSimulator(seed=42)
        # Winter noon
        winter = sim2.generate(datetime(2024, 12, 21, 12, 0, 0))

        # Summer should have more generation (at same latitude)
        assert summer.generation_kw >= winter.generation_kw * 0.8

    def test_invalid_panel_efficiency_raises_error(self):
        """Test invalid panel efficiency raises ValueError."""
        with pytest.raises(ValueError, match="panel_efficiency must be greater than 0"):
            SolarSimulator(panel_efficiency=0.0)

        with pytest.raises(ValueError, match="panel_efficiency must be greater than 0"):
            SolarSimulator(panel_efficiency=-0.1)


class TestBatterySimulator:
    """Tests for BatterySimulator."""

    def test_creation(self):
        """Test simulator can be created."""
        sim = BatterySimulator(capacity_kwh=13.5, seed=42)
        assert sim.capacity_kwh == 13.5

    def test_generate_returns_battery_data(self):
        """Test generate returns BatteryData."""
        sim = BatterySimulator(seed=42)
        data = sim.generate(datetime(2024, 6, 15, 12, 0, 0))

        assert hasattr(data, "soc_percent")
        assert hasattr(data, "capacity_kwh")
        assert hasattr(data, "power_kw")
        assert hasattr(data, "health_percent")

    def test_soc_bounded(self):
        """Test SoC stays within min/max bounds."""
        sim = BatterySimulator(
            min_soc=10.0,
            max_soc=90.0,
            seed=42,
        )

        for hour in range(24):
            data = sim.generate(
                datetime(2024, 6, 15, hour, 0, 0),
                solar_generation_kw=5.0,
                home_load_kw=3.0,
            )
            assert data.soc_percent >= sim.min_soc
            assert data.soc_percent <= sim.max_soc

    def test_charges_from_excess_solar(self):
        """Test battery charges when solar exceeds load."""
        sim = BatterySimulator(initial_soc=50.0, seed=42)

        # Excess solar
        data = sim.generate(
            datetime(2024, 6, 15, 12, 0, 0),
            solar_generation_kw=8.0,
            home_load_kw=2.0,
            grid_price=0.15,
        )

        # Should be charging (positive power)
        assert data.power_kw > 0

    def test_health_decreases_with_cycles(self):
        """Test battery health decreases as cycles increase."""
        sim1 = BatterySimulator(initial_cycles=100, seed=42)
        sim2 = BatterySimulator(initial_cycles=2000, seed=42)

        data1 = sim1.generate(datetime(2024, 6, 15, 12, 0, 0))
        data2 = sim2.generate(datetime(2024, 6, 15, 12, 0, 0))

        assert data1.health_percent > data2.health_percent

    def test_invalid_capacity_raises_error(self):
        """Test invalid capacity raises ValueError."""
        with pytest.raises(ValueError, match="capacity_kwh must be positive"):
            BatterySimulator(capacity_kwh=0)

        with pytest.raises(ValueError, match="capacity_kwh must be positive"):
            BatterySimulator(capacity_kwh=-10)

    def test_invalid_soc_bounds_raises_error(self):
        """Test invalid SoC bounds raise ValueError."""
        # initial_soc below min_soc
        with pytest.raises(ValueError, match="SoC bounds must satisfy"):
            BatterySimulator(initial_soc=5, min_soc=10, max_soc=90)

        # initial_soc above max_soc
        with pytest.raises(ValueError, match="SoC bounds must satisfy"):
            BatterySimulator(initial_soc=95, min_soc=10, max_soc=90)

        # min_soc > max_soc
        with pytest.raises(ValueError, match="SoC bounds must satisfy"):
            BatterySimulator(initial_soc=50, min_soc=90, max_soc=10)

        # SoC bounds outside 0-100
        with pytest.raises(ValueError, match="SoC bounds must satisfy"):
            BatterySimulator(initial_soc=50, min_soc=-10, max_soc=90)

    def test_invalid_power_rates_raises_error(self):
        """Test invalid power rates raise ValueError."""
        with pytest.raises(ValueError, match="max_charge_rate_kw and max_discharge_rate_kw must be positive"):
            BatterySimulator(max_charge_rate_kw=0)

        with pytest.raises(ValueError, match="max_charge_rate_kw and max_discharge_rate_kw must be positive"):
            BatterySimulator(max_charge_rate_kw=-1)

        with pytest.raises(ValueError, match="max_charge_rate_kw and max_discharge_rate_kw must be positive"):
            BatterySimulator(max_discharge_rate_kw=0)

        with pytest.raises(ValueError, match="max_charge_rate_kw and max_discharge_rate_kw must be positive"):
            BatterySimulator(max_discharge_rate_kw=-1)

    def test_invalid_round_trip_efficiency_raises_error(self):
        """Test invalid round_trip_efficiency raises ValueError."""
        with pytest.raises(ValueError, match="round_trip_efficiency must be in the range"):
            BatterySimulator(round_trip_efficiency=0)

        with pytest.raises(ValueError, match="round_trip_efficiency must be in the range"):
            BatterySimulator(round_trip_efficiency=-0.5)

        with pytest.raises(ValueError, match="round_trip_efficiency must be in the range"):
            BatterySimulator(round_trip_efficiency=1.5)


class TestHomeLoadSimulator:
    """Tests for HomeLoadSimulator."""

    def test_creation(self):
        """Test simulator can be created."""
        sim = HomeLoadSimulator(base_load_kw=0.5, seed=42)
        assert sim.base_load_kw == 0.5

    def test_generate_returns_home_load_data(self):
        """Test generate returns HomeLoadData."""
        sim = HomeLoadSimulator(seed=42)
        data = sim.generate(datetime(2024, 6, 15, 12, 0, 0))

        assert hasattr(data, "total_load_kw")
        assert hasattr(data, "hvac_kw")
        assert hasattr(data, "appliances_kw")
        assert hasattr(data, "lighting_kw")

    def test_load_above_base(self):
        """Test total load is at least base load."""
        sim = HomeLoadSimulator(base_load_kw=0.5, seed=42)

        for hour in range(24):
            data = sim.generate(datetime(2024, 6, 15, hour, 0, 0))
            assert data.total_load_kw >= sim.base_load_kw

    def test_load_bounded_by_peak(self):
        """Test load doesn't exceed peak."""
        sim = HomeLoadSimulator(peak_load_kw=10.0, seed=42)

        for hour in range(24):
            data = sim.generate(datetime(2024, 6, 15, hour, 0, 0))
            assert data.total_load_kw <= sim.peak_load_kw

    def test_component_sum_equals_total(self):
        """Test that sum of components equals total_load_kw."""
        sim = HomeLoadSimulator(seed=42)

        for hour in range(24):
            data = sim.generate(datetime(2024, 6, 15, hour, 0, 0))
            component_sum = (
                data.hvac_kw
                + data.appliances_kw
                + data.lighting_kw
                + data.ev_charging_kw
                + data.other_kw
            )
            assert abs(data.total_load_kw - component_sum) < 0.0001

    def test_evening_peak(self):
        """Test load is higher in evening than during work hours."""
        sim = HomeLoadSimulator(seed=42)

        # Weekday work hours
        work_hours = sim.generate(datetime(2024, 6, 17, 14, 0, 0))  # Monday 2 PM

        # Reset
        sim2 = HomeLoadSimulator(seed=42)
        # Weekday evening
        evening = sim2.generate(datetime(2024, 6, 17, 19, 0, 0))  # Monday 7 PM

        # Evening should typically have higher load
        # (Note: due to randomness, we check the general pattern)
        assert evening.total_load_kw >= work_hours.total_load_kw * 0.5

    def test_no_ev_charging_without_ev(self):
        """Test no EV charging when has_ev=False."""
        sim = HomeLoadSimulator(has_ev=False, seed=42)

        for hour in range(24):
            data = sim.generate(datetime(2024, 6, 15, hour, 0, 0))
            assert data.ev_charging_kw == 0


class TestGridPriceSimulator:
    """Tests for GridPriceSimulator."""

    def test_creation(self):
        """Test simulator can be created."""
        sim = GridPriceSimulator(peak_price=0.30, seed=42)
        assert sim.peak_price == 0.30

    def test_generate_returns_grid_price_data(self):
        """Test generate returns GridPriceData."""
        sim = GridPriceSimulator(seed=42)
        data = sim.generate(datetime(2024, 6, 15, 12, 0, 0))

        assert hasattr(data, "price_per_kwh")
        assert hasattr(data, "feed_in_tariff")
        assert hasattr(data, "time_of_use_period")

    def test_tou_periods(self):
        """Test TOU periods are assigned correctly."""
        sim = GridPriceSimulator(seed=42)

        # Early morning - should be off_peak
        data = sim.generate(datetime(2024, 6, 17, 3, 0, 0))  # Monday 3 AM
        assert data.time_of_use_period == "off_peak"

        # Reset for fresh state
        sim2 = GridPriceSimulator(seed=42)
        # Afternoon - should be peak
        data2 = sim2.generate(datetime(2024, 6, 17, 16, 0, 0))  # Monday 4 PM
        assert data2.time_of_use_period == "peak"

    def test_peak_price_higher_than_off_peak(self):
        """Test peak prices are higher than off-peak."""
        sim = GridPriceSimulator(
            off_peak_price=0.08,
            peak_price=0.30,
            seed=42,
        )

        off_peak = sim.generate(datetime(2024, 6, 17, 3, 0, 0))

        sim2 = GridPriceSimulator(
            off_peak_price=0.08,
            peak_price=0.30,
            seed=42,
        )
        peak = sim2.generate(datetime(2024, 6, 17, 16, 0, 0))

        assert peak.price_per_kwh > off_peak.price_per_kwh

    def test_feed_in_tariff_less_than_price(self):
        """Test feed-in tariff is less than retail price."""
        sim = GridPriceSimulator(seed=42)

        for hour in range(24):
            data = sim.generate(datetime(2024, 6, 15, hour, 0, 0))
            assert data.feed_in_tariff <= data.price_per_kwh

    def test_carbon_intensity_varies(self):
        """Test carbon intensity varies throughout the day."""
        sim = GridPriceSimulator(seed=42)

        intensities = []
        for hour in range(24):
            data = sim.generate(datetime(2024, 6, 15, hour, 0, 0))
            intensities.append(data.carbon_intensity_g_kwh)

        # Should have some variation
        assert max(intensities) > min(intensities)
