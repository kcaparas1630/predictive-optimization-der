"""Tests for configuration management."""

import json
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

    def test_to_file_and_from_file(self, tmp_path):
        """Test config can be saved and loaded from file."""
        config = GeneratorConfig(
            device_id="file-test-device",
            seed=42,
        )

        config_path = tmp_path / "config.json"
        config.to_file(config_path)

        # File should exist and be valid JSON
        assert config_path.exists()
        with open(config_path) as f:
            data = json.load(f)
        assert data["device_id"] == "file-test-device"

        # Should be able to load it back
        loaded = GeneratorConfig.from_file(config_path)
        assert loaded.device_id == "file-test-device"
        assert loaded.seed == 42

    def test_default_config_exists(self):
        """Test DEFAULT_CONFIG is available."""
        assert DEFAULT_CONFIG is not None
        assert isinstance(DEFAULT_CONFIG, GeneratorConfig)

    def test_from_file_missing_file_raises_error(self):
        """Test from_file raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="does_not_exist.json"):
            GeneratorConfig.from_file(Path("does_not_exist.json"))

    def test_to_file_unwritable_path_raises_error(self, tmp_path):
        """Test to_file raises RuntimeError for unwritable path."""
        config = GeneratorConfig()
        # Try to write to a directory path (not a file)
        with pytest.raises(RuntimeError, match="Failed to save configuration"):
            config.to_file(tmp_path)

    def test_from_file_invalid_json_raises_error(self, tmp_path):
        """Test from_file raises ValueError for invalid JSON."""
        invalid_json = tmp_path / "invalid.json"
        invalid_json.write_text("{ invalid json content }")

        with pytest.raises(ValueError, match="Invalid JSON in configuration file"):
            GeneratorConfig.from_file(invalid_json)

    def test_from_file_invalid_config_structure_raises_error(self, tmp_path):
        """Test from_file raises RuntimeError for invalid config structure."""
        invalid_config = tmp_path / "bad_config.json"
        # Config with unexpected keys in solar section
        invalid_config.write_text('{"solar": {"unknown_key": 123}}')

        with pytest.raises(RuntimeError, match="Failed to load configuration"):
            GeneratorConfig.from_file(invalid_config)
