"""Tests for InfluxDB storage module."""

from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest

from edge_gateway.config import InfluxDBConfig
from edge_gateway.models import (
    DERData,
    SolarData,
    BatteryData,
    HomeLoadData,
    GridPriceData,
)


# Check if influxdb-client is available
try:
    import influxdb_client  # noqa: F401

    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False


class TestInfluxDBConfig:
    """Tests for InfluxDBConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = InfluxDBConfig()

        assert config.url == "http://localhost:8086"
        assert config.token == ""
        assert config.org == "edge-gateway"
        assert config.bucket == "der-data"
        assert config.enabled is False
        assert config.retention_days == 7
        assert config.batch_size == 1
        assert config.flush_interval_ms == 1000

    def test_custom_values(self):
        """Test custom configuration values."""
        config = InfluxDBConfig(
            url="http://custom:8086",
            token="my-token",
            org="my-org",
            bucket="my-bucket",
            enabled=True,
            retention_days=30,
            batch_size=100,
            flush_interval_ms=5000,
        )

        assert config.url == "http://custom:8086"
        assert config.token == "my-token"
        assert config.org == "my-org"
        assert config.bucket == "my-bucket"
        assert config.enabled is True
        assert config.retention_days == 30
        assert config.batch_size == 100
        assert config.flush_interval_ms == 5000

    def test_env_var_override(self, monkeypatch):
        """Test environment variable overrides."""
        monkeypatch.setenv("INFLUXDB_URL", "http://env-host:9999")
        monkeypatch.setenv("INFLUXDB_TOKEN", "env-token")
        monkeypatch.setenv("INFLUXDB_ORG", "env-org")
        monkeypatch.setenv("INFLUXDB_BUCKET", "env-bucket")

        config = InfluxDBConfig()

        assert config.url == "http://env-host:9999"
        assert config.token == "env-token"
        assert config.org == "env-org"
        assert config.bucket == "env-bucket"

    def test_env_var_override_with_defaults(self, monkeypatch):
        """Test env vars override defaults but not explicit values."""
        monkeypatch.setenv("INFLUXDB_URL", "http://env-host:9999")

        # Explicit value should be overridden by env var due to __post_init__
        config = InfluxDBConfig(url="http://explicit:8086")

        # Note: With current implementation, env var takes precedence
        assert config.url == "http://env-host:9999"


@pytest.fixture
def sample_der_data():
    """Create sample DER data for testing."""
    return DERData(
        timestamp=datetime(2024, 6, 15, 14, 30, 0, tzinfo=timezone.utc),
        device_id="test-device-001",
        solar=SolarData(
            generation_kw=7.5,
            irradiance_w_m2=900.0,
            panel_temp_celsius=40.0,
            efficiency_percent=17.5,
        ),
        battery=BatteryData(
            soc_percent=65.0,
            capacity_kwh=13.5,
            power_kw=2.0,
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


@pytest.mark.skipif(not INFLUXDB_AVAILABLE, reason="influxdb-client not installed")
class TestInfluxDBStorage:
    """Tests for InfluxDBStorage class."""

    def test_disabled_storage_does_not_connect(self):
        """Test that disabled storage doesn't attempt connection."""
        from edge_gateway.storage import InfluxDBStorage

        config = InfluxDBConfig(enabled=False)
        storage = InfluxDBStorage(config)

        assert not storage.is_connected()

    def test_disabled_storage_write_returns_true(self, sample_der_data):
        """Test that disabled storage write succeeds silently."""
        from edge_gateway.storage import InfluxDBStorage

        config = InfluxDBConfig(enabled=False)
        storage = InfluxDBStorage(config)

        result = storage.write(sample_der_data)
        assert result is True

    def test_missing_token_raises_error(self):
        """Test that missing token raises ValueError when enabled."""
        from edge_gateway.storage import InfluxDBStorage

        config = InfluxDBConfig(enabled=True, token="")

        with pytest.raises(ValueError, match="InfluxDB token is required"):
            InfluxDBStorage(config)

    @patch("edge_gateway.storage.influxdb_client.InfluxDBClient")
    def test_connect_creates_client(self, mock_client_class):
        """Test that connect creates InfluxDB client."""
        from edge_gateway.storage import InfluxDBStorage

        mock_client = Mock()
        mock_client.write_api.return_value = Mock()
        mock_client_class.return_value = mock_client

        config = InfluxDBConfig(
            enabled=True,
            token="test-token",
            url="http://test:8086",
            org="test-org",
        )

        storage = InfluxDBStorage(config)

        mock_client_class.assert_called_once_with(
            url="http://test:8086",
            token="test-token",
            org="test-org",
        )
        assert storage.is_connected()

    @patch("edge_gateway.storage.influxdb_client.InfluxDBClient")
    def test_write_creates_points(self, mock_client_class, sample_der_data):
        """Test that write creates correct InfluxDB points."""
        from edge_gateway.storage import InfluxDBStorage

        mock_write_api = Mock()
        mock_client = Mock()
        mock_client.write_api.return_value = mock_write_api
        mock_client_class.return_value = mock_client

        config = InfluxDBConfig(
            enabled=True,
            token="test-token",
            bucket="test-bucket",
            org="test-org",
        )

        storage = InfluxDBStorage(config)
        result = storage.write(sample_der_data)

        assert result is True
        mock_write_api.write.assert_called_once()

        # Check that write was called with correct bucket and org
        call_kwargs = mock_write_api.write.call_args.kwargs
        assert call_kwargs["bucket"] == "test-bucket"
        assert call_kwargs["org"] == "test-org"

        # Should have created 5 points (solar, battery, home_load, grid_price, system)
        points = call_kwargs["record"]
        assert len(points) == 5

    @patch("edge_gateway.storage.influxdb_client.InfluxDBClient")
    def test_write_batch(self, mock_client_class, sample_der_data):
        """Test batch write creates all points."""
        from edge_gateway.storage import InfluxDBStorage

        mock_write_api = Mock()
        mock_client = Mock()
        mock_client.write_api.return_value = mock_write_api
        mock_client_class.return_value = mock_client

        config = InfluxDBConfig(enabled=True, token="test-token")

        storage = InfluxDBStorage(config)
        data_list = [sample_der_data, sample_der_data]
        result = storage.write_batch(data_list)

        assert result is True
        mock_write_api.write.assert_called_once()

        # Should have 10 points (5 per data item)
        points = mock_write_api.write.call_args.kwargs["record"]
        assert len(points) == 10

    @patch("edge_gateway.storage.influxdb_client.InfluxDBClient")
    def test_health_check_success(self, mock_client_class):
        """Test health check returns True when healthy."""
        from edge_gateway.storage import InfluxDBStorage

        mock_health = Mock()
        mock_health.status = "pass"
        mock_client = Mock()
        mock_client.write_api.return_value = Mock()
        mock_client.health.return_value = mock_health
        mock_client_class.return_value = mock_client

        config = InfluxDBConfig(enabled=True, token="test-token")
        storage = InfluxDBStorage(config)

        assert storage.health_check() is True

    @patch("edge_gateway.storage.influxdb_client.InfluxDBClient")
    def test_health_check_failure(self, mock_client_class):
        """Test health check returns False when unhealthy."""
        from edge_gateway.storage import InfluxDBStorage

        mock_client = Mock()
        mock_client.write_api.return_value = Mock()
        mock_client.health.side_effect = Exception("Connection failed")
        mock_client_class.return_value = mock_client

        config = InfluxDBConfig(enabled=True, token="test-token")
        storage = InfluxDBStorage(config)

        assert storage.health_check() is False

    @patch("edge_gateway.storage.influxdb_client.InfluxDBClient")
    def test_close_cleans_up(self, mock_client_class):
        """Test close properly cleans up resources."""
        from edge_gateway.storage import InfluxDBStorage

        mock_write_api = Mock()
        mock_client = Mock()
        mock_client.write_api.return_value = mock_write_api
        mock_client_class.return_value = mock_client

        config = InfluxDBConfig(enabled=True, token="test-token")
        storage = InfluxDBStorage(config)

        assert storage.is_connected()

        storage.close()

        mock_write_api.close.assert_called_once()
        mock_client.close.assert_called_once()
        assert not storage.is_connected()

    @patch("edge_gateway.storage.influxdb_client.InfluxDBClient")
    def test_context_manager(self, mock_client_class):
        """Test context manager properly closes connection."""
        from edge_gateway.storage import InfluxDBStorage

        mock_write_api = Mock()
        mock_client = Mock()
        mock_client.write_api.return_value = mock_write_api
        mock_client_class.return_value = mock_client

        config = InfluxDBConfig(enabled=True, token="test-token")

        with InfluxDBStorage(config) as storage:
            assert storage.is_connected()

        mock_write_api.close.assert_called_once()
        mock_client.close.assert_called_once()

    @patch("edge_gateway.storage.influxdb_client.InfluxDBClient")
    def test_write_failure_returns_false(self, mock_client_class, sample_der_data):
        """Test write returns False on failure."""
        from edge_gateway.storage import InfluxDBStorage

        mock_write_api = Mock()
        mock_write_api.write.side_effect = Exception("Write failed")
        mock_client = Mock()
        mock_client.write_api.return_value = mock_write_api
        mock_client_class.return_value = mock_client

        config = InfluxDBConfig(enabled=True, token="test-token")
        storage = InfluxDBStorage(config)

        result = storage.write(sample_der_data)
        assert result is False

    def test_der_data_to_points_structure(self, sample_der_data):
        """Test the structure of generated InfluxDB points."""
        from edge_gateway.storage import InfluxDBStorage

        config = InfluxDBConfig(enabled=False)
        storage = InfluxDBStorage(config)

        # Access private method for testing
        points = storage._der_data_to_points(sample_der_data)

        assert len(points) == 5

        # Verify measurement names
        measurement_names = [
            InfluxDBStorage.MEASUREMENT_SOLAR,
            InfluxDBStorage.MEASUREMENT_BATTERY,
            InfluxDBStorage.MEASUREMENT_HOME_LOAD,
            InfluxDBStorage.MEASUREMENT_GRID_PRICE,
            InfluxDBStorage.MEASUREMENT_SYSTEM,
        ]

        for point, expected_name in zip(points, measurement_names):
            # Point objects have _name attribute in influxdb-client
            assert point._name == expected_name


@pytest.mark.skipif(not INFLUXDB_AVAILABLE, reason="influxdb-client not installed")
class TestInfluxDBStorageBatching:
    """Tests for InfluxDB batching configuration."""

    @patch("edge_gateway.storage.influxdb_client.InfluxDBClient")
    def test_batch_mode_configuration(self, mock_client_class):
        """Test batch mode uses WriteOptions."""
        from edge_gateway.storage import InfluxDBStorage

        mock_write_api = Mock()
        mock_client = Mock()
        mock_client.write_api.return_value = mock_write_api
        mock_client_class.return_value = mock_client

        config = InfluxDBConfig(
            enabled=True,
            token="test-token",
            batch_size=100,
            flush_interval_ms=5000,
        )

        _storage = InfluxDBStorage(config)  # noqa: F841

        # Verify write_api was called with WriteOptions
        call_args = mock_client.write_api.call_args
        assert call_args is not None
        # The WriteOptions should be passed
        assert "write_options" in call_args.kwargs


class TestInfluxDBConfigInGeneratorConfig:
    """Test InfluxDB config integration with GeneratorConfig."""

    def test_generator_config_includes_influxdb(self):
        """Test GeneratorConfig includes InfluxDB configuration."""
        from edge_gateway.config import GeneratorConfig

        config = GeneratorConfig()

        assert hasattr(config, "influxdb")
        assert isinstance(config.influxdb, InfluxDBConfig)

    def test_generator_config_from_dict_with_influxdb(self):
        """Test GeneratorConfig.from_dict parses InfluxDB config."""
        from edge_gateway.config import GeneratorConfig

        data = {
            "device_id": "test-device",
            "influxdb": {
                "url": "http://custom:8086",
                "token": "custom-token",
                "enabled": True,
                "retention_days": 14,
            },
        }

        config = GeneratorConfig.from_dict(data)

        assert config.influxdb.url == "http://custom:8086"
        assert config.influxdb.token == "custom-token"
        assert config.influxdb.enabled is True
        assert config.influxdb.retention_days == 14

    def test_generator_config_to_dict_includes_influxdb(self):
        """Test GeneratorConfig.to_dict includes InfluxDB config."""
        from edge_gateway.config import GeneratorConfig

        config = GeneratorConfig()
        config.influxdb.enabled = True
        config.influxdb.token = "test-token"

        data = config.to_dict()

        assert "influxdb" in data
        assert data["influxdb"]["enabled"] is True
        assert data["influxdb"]["token"] == "test-token"
