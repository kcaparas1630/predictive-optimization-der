"""Tests for DER data models."""

import json
from datetime import datetime, timezone


from edge_gateway.models import (
    DERData,
    SolarData,
    BatteryData,
    HomeLoadData,
    GridPriceData,
)


class TestSolarData:
    """Tests for SolarData model."""

    def test_creation(self):
        """Test SolarData can be created with valid values."""
        data = SolarData(
            generation_kw=5.5,
            irradiance_w_m2=800.0,
            panel_temp_celsius=35.0,
            efficiency_percent=18.5,
        )
        assert data.generation_kw == 5.5
        assert data.irradiance_w_m2 == 800.0
        assert data.panel_temp_celsius == 35.0
        assert data.efficiency_percent == 18.5

    def test_to_dict(self):
        """Test SolarData serializes correctly."""
        data = SolarData(
            generation_kw=5.555,
            irradiance_w_m2=800.123,
            panel_temp_celsius=35.05,
            efficiency_percent=18.567,
        )
        d = data.to_dict()
        assert d["generation_kw"] == 5.555
        assert d["irradiance_w_m2"] == 800.12
        assert d["panel_temp_celsius"] == 35.0
        assert d["efficiency_percent"] == 18.57


class TestBatteryData:
    """Tests for BatteryData model."""

    def test_creation(self):
        """Test BatteryData can be created with valid values."""
        data = BatteryData(
            soc_percent=75.0,
            capacity_kwh=13.5,
            power_kw=2.5,
            voltage_v=52.0,
            temperature_celsius=28.0,
            cycles=150,
            health_percent=98.5,
        )
        assert data.soc_percent == 75.0
        assert data.capacity_kwh == 13.5
        assert data.power_kw == 2.5
        assert data.cycles == 150

    def test_to_dict(self):
        """Test BatteryData serializes correctly."""
        data = BatteryData(
            soc_percent=75.123,
            capacity_kwh=13.5,
            power_kw=2.567,
            voltage_v=52.12,
            temperature_celsius=28.05,
            cycles=150,
            health_percent=98.567,
        )
        d = data.to_dict()
        assert d["soc_percent"] == 75.12
        assert d["capacity_kwh"] == 13.5
        assert d["power_kw"] == 2.567
        assert d["voltage_v"] == 52.1
        assert d["temperature_celsius"] == 28.1
        assert d["cycles"] == 150
        assert d["health_percent"] == 98.57


class TestHomeLoadData:
    """Tests for HomeLoadData model."""

    def test_creation(self):
        """Test HomeLoadData can be created with valid values."""
        data = HomeLoadData(
            total_load_kw=4.5,
            hvac_kw=2.0,
            appliances_kw=1.0,
            lighting_kw=0.5,
            ev_charging_kw=0.0,
            other_kw=1.0,
        )
        assert data.total_load_kw == 4.5
        assert data.hvac_kw == 2.0

    def test_to_dict(self):
        """Test HomeLoadData serializes correctly."""
        data = HomeLoadData(
            total_load_kw=4.567,
            hvac_kw=2.0,
            appliances_kw=1.0,
            lighting_kw=0.5,
            ev_charging_kw=0.0,
            other_kw=1.067,
        )
        d = data.to_dict()
        assert d["total_load_kw"] == 4.567
        assert d["hvac_kw"] == 2.0
        assert d["appliances_kw"] == 1.0
        assert d["lighting_kw"] == 0.5
        assert d["ev_charging_kw"] == 0.0
        assert d["other_kw"] == 1.067


class TestGridPriceData:
    """Tests for GridPriceData model."""

    def test_creation(self):
        """Test GridPriceData can be created with valid values."""
        data = GridPriceData(
            price_per_kwh=0.25,
            feed_in_tariff=0.08,
            demand_charge=10.0,
            time_of_use_period="peak",
            carbon_intensity_g_kwh=450.0,
        )
        assert data.price_per_kwh == 0.25
        assert data.time_of_use_period == "peak"

    def test_to_dict(self):
        """Test GridPriceData serializes correctly."""
        data = GridPriceData(
            price_per_kwh=0.2567,
            feed_in_tariff=0.0812,
            demand_charge=10.05,
            time_of_use_period="shoulder",
            carbon_intensity_g_kwh=450.123,
        )
        d = data.to_dict()
        assert d["price_per_kwh"] == 0.2567
        assert d["feed_in_tariff"] == 0.0812
        assert d["demand_charge"] == 10.05
        assert d["time_of_use_period"] == "shoulder"
        assert d["carbon_intensity_g_kwh"] == 450.12


class TestDERData:
    """Tests for DERData composite model."""

    def create_sample_der_data(self):
        """Create sample DER data for testing."""
        return DERData(
            timestamp=datetime(2024, 6, 15, 14, 30, 0, tzinfo=timezone.utc),
            device_id="test-gateway-001",
            solar=SolarData(
                generation_kw=7.5,
                irradiance_w_m2=900.0,
                panel_temp_celsius=40.0,
                efficiency_percent=17.5,
            ),
            battery=BatteryData(
                soc_percent=65.0,
                capacity_kwh=13.5,
                power_kw=2.0,  # Charging
                voltage_v=51.5,
                temperature_celsius=30.0,
                cycles=200,
                health_percent=97.0,
            ),
            home_load=HomeLoadData(
                total_load_kw=3.5,
                hvac_kw=1.5,
                appliances_kw=1.0,
                lighting_kw=0.3,
                ev_charging_kw=0.0,
                other_kw=0.7,
            ),
            grid_price=GridPriceData(
                price_per_kwh=0.28,
                feed_in_tariff=0.08,
                demand_charge=10.0,
                time_of_use_period="peak",
                carbon_intensity_g_kwh=380.0,
            ),
        )

    def test_creation(self):
        """Test DERData can be created with valid values."""
        data = self.create_sample_der_data()
        assert data.device_id == "test-gateway-001"
        assert data.solar.generation_kw == 7.5
        assert data.battery.soc_percent == 65.0

    def test_net_grid_flow_calculation(self):
        """Test net grid flow is calculated correctly."""
        data = self.create_sample_der_data()
        # net = load - solar + battery_power
        # net = 3.5 - 7.5 + 2.0 = -2.0 (exporting to grid)
        assert data.net_grid_flow_kw == -2.0

    def test_to_dict(self):
        """Test DERData serializes correctly."""
        data = self.create_sample_der_data()
        d = data.to_dict()

        assert d["device_id"] == "test-gateway-001"
        assert d["timestamp"] == "2024-06-15T14:30:00+00:00"
        assert "solar" in d
        assert "battery" in d
        assert "home_load" in d
        assert "grid_price" in d
        assert d["net_grid_flow_kw"] == -2.0

    def test_to_json(self):
        """Test DERData converts to valid JSON."""
        data = self.create_sample_der_data()
        json_str = data.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["device_id"] == "test-gateway-001"

    def test_from_dict(self):
        """Test DERData can be recreated from dictionary."""
        original = self.create_sample_der_data()
        d = original.to_dict()
        restored = DERData.from_dict(d)

        assert restored.device_id == original.device_id
        assert restored.timestamp == original.timestamp
        assert restored.timestamp.tzinfo is not None
        assert restored.net_grid_flow_kw == original.net_grid_flow_kw
        assert restored.solar.generation_kw == original.solar.generation_kw
        assert restored.battery.soc_percent == original.battery.soc_percent
