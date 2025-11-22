"""Tests for the DER data generator."""

from datetime import datetime, timedelta, timezone

import pytest

from edge_gateway.generator import DERDataGenerator, DataGeneratorRunner
from edge_gateway.models import DERData


class TestDERDataGenerator:
    """Tests for DERDataGenerator."""

    def test_creation(self):
        """Test generator can be created with defaults."""
        gen = DERDataGenerator()
        assert gen.device_id == "edge-gateway-001"

    def test_creation_with_custom_config(self):
        """Test generator can be created with custom config."""
        gen = DERDataGenerator(
            device_id="custom-device",
            solar_capacity_kw=15.0,
            battery_capacity_kwh=20.0,
            seed=42,
        )
        assert gen.device_id == "custom-device"

    def test_generate_returns_der_data(self):
        """Test generate returns DERData."""
        gen = DERDataGenerator(seed=42)
        data = gen.generate(datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc))

        assert isinstance(data, DERData)
        assert data.device_id == "edge-gateway-001"
        assert hasattr(data, "solar")
        assert hasattr(data, "battery")
        assert hasattr(data, "home_load")
        assert hasattr(data, "grid_price")

    def test_generate_uses_current_time_by_default(self):
        """Test generate uses current time when no timestamp provided."""
        gen = DERDataGenerator(seed=42)
        before = datetime.now(timezone.utc)
        data = gen.generate()
        after = datetime.now(timezone.utc)

        assert before <= data.timestamp <= after

    def test_generate_historical(self):
        """Test generating historical data."""
        gen = DERDataGenerator(seed=42)

        start = datetime(2024, 6, 15, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 6, 15, 1, 0, 0, tzinfo=timezone.utc)

        data = gen.generate_historical(start, end, interval_minutes=5)

        # Should have 13 data points (0, 5, 10, ..., 60 minutes)
        assert len(data) == 13

        # All should be DERData
        for d in data:
            assert isinstance(d, DERData)

        # Timestamps should be in order
        for i in range(1, len(data)):
            assert data[i].timestamp > data[i - 1].timestamp

    def test_generate_historical_longer_period(self):
        """Test generating a full day of historical data."""
        gen = DERDataGenerator(seed=42)

        start = datetime(2024, 6, 15, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 6, 15, 23, 55, 0, tzinfo=timezone.utc)

        data = gen.generate_historical(start, end, interval_minutes=5)

        # Should have 288 data points (24 hours * 12 per hour)
        assert len(data) == 288

    def test_reproducibility_with_seed(self):
        """Test same seed produces same results."""
        gen1 = DERDataGenerator(seed=42)
        gen2 = DERDataGenerator(seed=42)

        ts = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        data1 = gen1.generate(ts)
        data2 = gen2.generate(ts)

        assert data1.solar.generation_kw == data2.solar.generation_kw
        assert data1.home_load.total_load_kw == data2.home_load.total_load_kw

    def test_net_grid_flow_calculation(self):
        """Test net grid flow is calculated correctly."""
        gen = DERDataGenerator(seed=42)
        data = gen.generate(datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc))

        expected_net = (
            data.home_load.total_load_kw
            - data.solar.generation_kw
            + data.battery.power_kw
        )

        assert abs(data.net_grid_flow_kw - expected_net) < 0.001

    def test_data_serialization(self):
        """Test generated data can be serialized to JSON."""
        gen = DERDataGenerator(seed=42)
        data = gen.generate(datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc))

        json_str = data.to_json()
        assert isinstance(json_str, str)
        assert "solar" in json_str
        assert "battery" in json_str


class TestDataGeneratorRunner:
    """Tests for DataGeneratorRunner."""

    def test_creation(self):
        """Test runner can be created."""
        gen = DERDataGenerator(seed=42)
        runner = DataGeneratorRunner(generator=gen)
        assert runner.generator is gen

    def test_run_once(self):
        """Test run_once generates a single data point."""
        gen = DERDataGenerator(seed=42)
        captured_data = []

        def capture(data) -> None:
            captured_data.append(data)

        runner = DataGeneratorRunner(generator=gen, output_callback=capture)
        result = runner.run_once()

        assert isinstance(result, DERData)
        assert len(captured_data) == 1

    def test_run_once_with_callback(self):
        """Test callback is invoked."""
        gen = DERDataGenerator(seed=42)
        captured = []

        runner = DataGeneratorRunner(
            generator=gen,
            output_callback=lambda d: captured.append(d),
        )
        runner.run_once()

        assert len(captured) == 1
        assert isinstance(captured[0], DERData)

    def test_stop(self):
        """Test runner can be stopped."""
        gen = DERDataGenerator(seed=42)
        runner = DataGeneratorRunner(generator=gen)

        runner._running = True
        runner.stop()

        assert runner._running is False
