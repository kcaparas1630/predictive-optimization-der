"""Tests for configuration management."""

import json
import tempfile
from pathlib import Path

import pytest

from edge_gateway.config import (
    GeneratorConfig,
    SolarConfig,
    BatteryConfig,
    HomeLoadConfig,
    GridPriceConfig,
    DEFAULT_CONFIG,
)


class TestSolarConfig:
    """Tests for SolarConfig."""

    def test_defaults(self):
        """Test default values."""
        config = SolarConfig()
        assert config.capacity_kw == 10.0
        assert config.latitude == 37.7749


class TestBatteryConfig:
    """Tests for BatteryConfig."""

    def test_defaults(self):
        """Test default values."""
        config = BatteryConfig()
        assert config.capacity_kwh == 13.5
        assert config.round_trip_efficiency == 0.90


class TestHomeLoadConfig:
    """Tests for HomeLoadConfig."""

    def test_defaults(self):
        """Test default values."""
        config = HomeLoadConfig()
        assert config.base_load_kw == 0.5
        assert config.has_ev is True


class TestGridPriceConfig:
    """Tests for GridPriceConfig."""

    def test_defaults(self):
        """Test default values."""
        config = GridPriceConfig()
        assert config.off_peak_price == 0.08
        assert config.peak_price == 0.30


class TestGeneratorConfig:
    """Tests for GeneratorConfig."""

    def test_defaults(self):
        """Test default configuration values."""
        config = GeneratorConfig()
        assert config.device_id == "edge-gateway-001"
        assert config.interval_seconds == 300
        assert config.seed is None

    def test_nested_configs(self):
        """Test nested config objects are created."""
        config = GeneratorConfig()
        assert isinstance(config.solar, SolarConfig)
        assert isinstance(config.battery, BatteryConfig)
        assert isinstance(config.home_load, HomeLoadConfig)
        assert isinstance(config.grid_price, GridPriceConfig)

    def test_to_dict(self):
        """Test config serializes to dictionary."""
        config = GeneratorConfig(device_id="test-device")
        d = config.to_dict()

        assert d["device_id"] == "test-device"
        assert "solar" in d
        assert "battery" in d
        assert "home_load" in d
        assert "grid_price" in d

    def test_from_dict(self):
        """Test config can be created from dictionary."""
        data = {
            "device_id": "custom-device",
            "interval_seconds": 600,
            "solar": {"capacity_kw": 15.0},
            "battery": {"capacity_kwh": 20.0},
        }

        config = GeneratorConfig.from_dict(data)

        assert config.device_id == "custom-device"
        assert config.interval_seconds == 600
        assert config.solar.capacity_kw == 15.0
        assert config.battery.capacity_kwh == 20.0

    def test_from_dict_with_partial_data(self):
        """Test from_dict uses defaults for missing values."""
        data = {"device_id": "minimal-device"}

        config = GeneratorConfig.from_dict(data)

        assert config.device_id == "minimal-device"
        # Defaults should be used
        assert config.solar.capacity_kw == 10.0
        assert config.battery.capacity_kwh == 13.5

    def test_to_file_and_from_file(self):
        """Test config can be saved and loaded from file."""
        config = GeneratorConfig(
            device_id="file-test-device",
            seed=42,
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            temp_path = Path(f.name)

        try:
            config.to_file(temp_path)

            # File should exist and be valid JSON
            assert temp_path.exists()
            with open(temp_path) as f:
                data = json.load(f)
            assert data["device_id"] == "file-test-device"

            # Should be able to load it back
            loaded = GeneratorConfig.from_file(temp_path)
            assert loaded.device_id == "file-test-device"
            assert loaded.seed == 42

        finally:
            temp_path.unlink()

    def test_default_config_exists(self):
        """Test DEFAULT_CONFIG is available."""
        assert DEFAULT_CONFIG is not None
        assert isinstance(DEFAULT_CONFIG, GeneratorConfig)
