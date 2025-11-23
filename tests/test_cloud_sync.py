"""Tests for cloud sync module."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from edge_gateway.storage.cloud_sync import (
    CloudSync,
    CloudSyncConfig,
    SyncState,
)


# Check if required libraries are available
try:
    import influxdb_client  # noqa: F401

    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False

try:
    import supabase  # noqa: F401

    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False


class TestCloudSyncConfig:
    """Tests for CloudSyncConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CloudSyncConfig()

        assert config.supabase_url == ""
        assert config.supabase_key == ""
        assert config.enabled is False
        assert config.batch_size == 100
        assert config.sync_interval_seconds == 60
        assert config.state_file == ".sync_state.json"
        assert config.site_id == "edge-gateway-site"
        assert config.max_retries == 3
        assert config.retry_delay_seconds == 5

    def test_custom_values(self):
        """Test custom configuration values."""
        config = CloudSyncConfig(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
            enabled=True,
            batch_size=50,
            sync_interval_seconds=120,
            state_file="/custom/state.json",
            site_id="custom-site",
            max_retries=5,
            retry_delay_seconds=10,
        )

        assert config.supabase_url == "https://test.supabase.co"
        assert config.supabase_key == "test-key"
        assert config.enabled is True
        assert config.batch_size == 50
        assert config.sync_interval_seconds == 120
        assert config.state_file == "/custom/state.json"
        assert config.site_id == "custom-site"
        assert config.max_retries == 5
        assert config.retry_delay_seconds == 10

    def test_env_var_override(self, monkeypatch):
        """Test environment variable overrides."""
        monkeypatch.setenv("SUPABASE_URL", "https://env.supabase.co")
        monkeypatch.setenv("SUPABASE_KEY", "env-key")
        monkeypatch.setenv("SYNC_BATCH_SIZE", "200")
        monkeypatch.setenv("SYNC_INTERVAL_SECONDS", "30")
        monkeypatch.setenv("SYNC_STATE_FILE", "/env/state.json")
        monkeypatch.setenv("SYNC_SITE_ID", "env-site")

        config = CloudSyncConfig()

        assert config.supabase_url == "https://env.supabase.co"
        assert config.supabase_key == "env-key"
        assert config.batch_size == 200
        assert config.sync_interval_seconds == 30
        assert config.state_file == "/env/state.json"
        assert config.site_id == "env-site"


class TestSyncState:
    """Tests for SyncState."""

    def test_initial_state(self):
        """Test initial state values."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            state_file = f.name

        try:
            state = SyncState(state_file)

            assert state.last_sync_time is None
            assert state.total_synced == 0
        finally:
            Path(state_file).unlink(missing_ok=True)

    def test_update_success(self):
        """Test updating state after successful sync."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            state_file = f.name

        try:
            state = SyncState(state_file)
            sync_time = datetime(2024, 6, 15, 14, 30, 0, tzinfo=timezone.utc)

            state.update_success(sync_time, 100)

            assert state.last_sync_time == sync_time
            assert state.total_synced == 100

            # Update again
            sync_time2 = datetime(2024, 6, 15, 15, 0, 0, tzinfo=timezone.utc)
            state.update_success(sync_time2, 50)

            assert state.last_sync_time == sync_time2
            assert state.total_synced == 150
        finally:
            Path(state_file).unlink(missing_ok=True)

    def test_update_error(self):
        """Test updating state after failed sync."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            state_file = f.name

        try:
            state = SyncState(state_file)

            state.update_error("Connection timeout")

            status = state.get_status()
            assert status["last_error"] == "Connection timeout"
            assert status["last_error_time"] is not None
        finally:
            Path(state_file).unlink(missing_ok=True)

    def test_persistence(self):
        """Test state persistence across instances."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            state_file = f.name

        try:
            # Create state and update it
            state1 = SyncState(state_file)
            sync_time = datetime(2024, 6, 15, 14, 30, 0, tzinfo=timezone.utc)
            state1.update_success(sync_time, 100)

            # Create new instance from same file
            state2 = SyncState(state_file)

            assert state2.last_sync_time == sync_time
            assert state2.total_synced == 100
        finally:
            Path(state_file).unlink(missing_ok=True)

    def test_corrupted_state_file(self):
        """Test handling of corrupted state file."""
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            f.write("not valid json")
            state_file = f.name

        try:
            # Should not raise, but start fresh
            state = SyncState(state_file)

            assert state.last_sync_time is None
            assert state.total_synced == 0
        finally:
            Path(state_file).unlink(missing_ok=True)

    def test_get_status(self):
        """Test get_status returns complete information."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            state_file = f.name

        try:
            state = SyncState(state_file)
            sync_time = datetime(2024, 6, 15, 14, 30, 0, tzinfo=timezone.utc)
            state.update_success(sync_time, 100)

            status = state.get_status()

            assert "last_sync_time" in status
            assert "last_sync_count" in status
            assert "total_synced" in status
            assert "last_error" in status
            assert "last_error_time" in status

            assert status["last_sync_count"] == 100
            assert status["total_synced"] == 100
            assert status["last_error"] is None
        finally:
            Path(state_file).unlink(missing_ok=True)


@pytest.mark.skipif(
    not (INFLUXDB_AVAILABLE and SUPABASE_AVAILABLE),
    reason="influxdb-client and supabase not installed",
)
class TestCloudSync:
    """Tests for CloudSync class."""

    def test_disabled_sync_does_not_connect(self):
        """Test that disabled sync doesn't attempt connection."""
        config = CloudSyncConfig(enabled=False)

        sync = CloudSync(
            config,
            influxdb_url="http://localhost:8086",
            influxdb_token="test-token",
            influxdb_org="test-org",
            influxdb_bucket="test-bucket",
        )

        assert not sync.is_connected()

    def test_disabled_sync_returns_zeros(self):
        """Test that disabled sync returns (0, 0)."""
        config = CloudSyncConfig(enabled=False)

        sync = CloudSync(
            config,
            influxdb_url="http://localhost:8086",
            influxdb_token="test-token",
            influxdb_org="test-org",
            influxdb_bucket="test-bucket",
        )

        successful, failed = sync.sync()
        assert successful == 0
        assert failed == 0

    def test_missing_supabase_url_raises_error(self):
        """Test that missing Supabase URL raises ValueError."""
        config = CloudSyncConfig(
            enabled=True,
            supabase_url="",
            supabase_key="test-key",
        )

        with pytest.raises(ValueError, match="Supabase URL is required"):
            CloudSync(
                config,
                influxdb_url="http://localhost:8086",
                influxdb_token="test-token",
                influxdb_org="test-org",
                influxdb_bucket="test-bucket",
            )

    def test_missing_supabase_key_raises_error(self):
        """Test that missing Supabase key raises ValueError."""
        config = CloudSyncConfig(
            enabled=True,
            supabase_url="https://test.supabase.co",
            supabase_key="",
        )

        with pytest.raises(ValueError, match="Supabase key is required"):
            CloudSync(
                config,
                influxdb_url="http://localhost:8086",
                influxdb_token="test-token",
                influxdb_org="test-org",
                influxdb_bucket="test-bucket",
            )

    def test_missing_influxdb_token_raises_error(self):
        """Test that missing InfluxDB token raises ValueError."""
        config = CloudSyncConfig(
            enabled=True,
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
        )

        with pytest.raises(ValueError, match="InfluxDB token is required"):
            CloudSync(
                config,
                influxdb_url="http://localhost:8086",
                influxdb_token="",
                influxdb_org="test-org",
                influxdb_bucket="test-bucket",
            )

    @patch("edge_gateway.storage.cloud_sync.create_client")
    @patch("edge_gateway.storage.cloud_sync.InfluxDBClient")
    def test_connect_creates_clients(
        self, mock_influx_class, mock_supabase_create
    ):
        """Test that connect creates both clients."""
        mock_influx = Mock()
        mock_influx_class.return_value = mock_influx

        mock_supabase = Mock()
        mock_supabase_create.return_value = mock_supabase

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            state_file = f.name

        try:
            config = CloudSyncConfig(
                enabled=True,
                supabase_url="https://test.supabase.co",
                supabase_key="test-key",
                state_file=state_file,
            )

            sync = CloudSync(
                config,
                influxdb_url="http://localhost:8086",
                influxdb_token="test-token",
                influxdb_org="test-org",
                influxdb_bucket="test-bucket",
            )

            mock_influx_class.assert_called_once_with(
                url="http://localhost:8086",
                token="test-token",
                org="test-org",
            )
            mock_supabase_create.assert_called_once_with(
                "https://test.supabase.co",
                "test-key",
            )
            assert sync.is_connected()

            sync.close()
        finally:
            Path(state_file).unlink(missing_ok=True)

    @patch("edge_gateway.storage.cloud_sync.create_client")
    @patch("edge_gateway.storage.cloud_sync.InfluxDBClient")
    def test_health_check(self, mock_influx_class, mock_supabase_create):
        """Test health check returns correct status."""
        mock_influx = Mock()
        mock_influx.health.return_value = Mock(status="pass")
        mock_influx_class.return_value = mock_influx

        mock_table = Mock()
        mock_table.select.return_value.limit.return_value.execute.return_value = None
        mock_supabase = Mock()
        mock_supabase.table.return_value = mock_table
        mock_supabase_create.return_value = mock_supabase

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            state_file = f.name

        try:
            config = CloudSyncConfig(
                enabled=True,
                supabase_url="https://test.supabase.co",
                supabase_key="test-key",
                state_file=state_file,
            )

            sync = CloudSync(
                config,
                influxdb_url="http://localhost:8086",
                influxdb_token="test-token",
                influxdb_org="test-org",
                influxdb_bucket="test-bucket",
            )

            health = sync.health_check()

            assert health["influxdb"] is True
            assert health["supabase"] is True

            sync.close()
        finally:
            Path(state_file).unlink(missing_ok=True)

    @patch("edge_gateway.storage.cloud_sync.create_client")
    @patch("edge_gateway.storage.cloud_sync.InfluxDBClient")
    def test_transform_record(self, mock_influx_class, mock_supabase_create):
        """Test record transformation to Supabase format."""
        mock_influx = Mock()
        mock_influx_class.return_value = mock_influx

        mock_supabase = Mock()
        mock_supabase_create.return_value = mock_supabase

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            state_file = f.name

        try:
            config = CloudSyncConfig(
                enabled=True,
                supabase_url="https://test.supabase.co",
                supabase_key="test-key",
                state_file=state_file,
                site_id="test-site",
            )

            sync = CloudSync(
                config,
                influxdb_url="http://localhost:8086",
                influxdb_token="test-token",
                influxdb_org="test-org",
                influxdb_bucket="test-bucket",
            )

            # Create mock InfluxDB record
            mock_record = Mock()
            mock_record.get_time.return_value = datetime(
                2024, 6, 15, 14, 30, 0, tzinfo=timezone.utc
            )
            mock_record.values = {"device_id": "test-device"}
            mock_record.get_field.return_value = "generation_kw"
            mock_record.get_value.return_value = 7.5

            result = sync._transform_record("solar", mock_record)

            assert result is not None
            assert result["site_id"] == "test-site"
            assert result["device_id"] == "test-device"
            assert result["metric_name"] == "solar.generation_kw"
            assert result["metric_value"] == 7.5
            assert "time" in result

            sync.close()
        finally:
            Path(state_file).unlink(missing_ok=True)

    @patch("edge_gateway.storage.cloud_sync.create_client")
    @patch("edge_gateway.storage.cloud_sync.InfluxDBClient")
    def test_build_flux_query(self, mock_influx_class, mock_supabase_create):
        """Test Flux query building."""
        mock_influx = Mock()
        mock_influx_class.return_value = mock_influx

        mock_supabase = Mock()
        mock_supabase_create.return_value = mock_supabase

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            state_file = f.name

        try:
            config = CloudSyncConfig(
                enabled=True,
                supabase_url="https://test.supabase.co",
                supabase_key="test-key",
                state_file=state_file,
            )

            sync = CloudSync(
                config,
                influxdb_url="http://localhost:8086",
                influxdb_token="test-token",
                influxdb_org="test-org",
                influxdb_bucket="test-bucket",
            )

            # Test query without start time
            query = sync._build_flux_query("solar", None, 100)
            assert "test-bucket" in query
            assert 'r._measurement == "solar"' in query
            assert "generation_kw" in query
            assert "limit(n: 100)" in query

            # Test query with start time
            start = datetime(2024, 6, 15, 14, 30, 0, tzinfo=timezone.utc)
            query_with_start = sync._build_flux_query("solar", start, 50)
            assert "2024-06-15" in query_with_start
            assert "limit(n: 50)" in query_with_start

            sync.close()
        finally:
            Path(state_file).unlink(missing_ok=True)

    @patch("edge_gateway.storage.cloud_sync.create_client")
    @patch("edge_gateway.storage.cloud_sync.InfluxDBClient")
    def test_push_to_supabase_with_retries(
        self, mock_influx_class, mock_supabase_create
    ):
        """Test Supabase push with retry logic."""
        mock_influx = Mock()
        mock_influx_class.return_value = mock_influx

        # Mock Supabase to fail twice then succeed
        mock_upsert = Mock()
        mock_upsert.execute.side_effect = [
            Exception("Network error"),
            Exception("Network error"),
            Mock(),  # Success on third try
        ]
        mock_table = Mock()
        mock_table.upsert.return_value = mock_upsert
        mock_supabase = Mock()
        mock_supabase.table.return_value = mock_table
        mock_supabase_create.return_value = mock_supabase

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            state_file = f.name

        try:
            config = CloudSyncConfig(
                enabled=True,
                supabase_url="https://test.supabase.co",
                supabase_key="test-key",
                state_file=state_file,
                max_retries=3,
                retry_delay_seconds=0,  # No delay for tests
            )

            sync = CloudSync(
                config,
                influxdb_url="http://localhost:8086",
                influxdb_token="test-token",
                influxdb_org="test-org",
                influxdb_bucket="test-bucket",
            )

            records = [
                {
                    "time": "2024-06-15T14:30:00+00:00",
                    "site_id": "test-site",
                    "device_id": "test-device",
                    "metric_name": "solar.generation_kw",
                    "metric_value": 7.5,
                }
            ]

            successful, failed = sync._push_to_supabase(records)

            # Should succeed after retries
            assert successful == 1
            assert failed == 0
            assert mock_upsert.execute.call_count == 3

            sync.close()
        finally:
            Path(state_file).unlink(missing_ok=True)

    @patch("edge_gateway.storage.cloud_sync.create_client")
    @patch("edge_gateway.storage.cloud_sync.InfluxDBClient")
    def test_push_to_supabase_all_retries_fail(
        self, mock_influx_class, mock_supabase_create
    ):
        """Test Supabase push when all retries fail."""
        mock_influx = Mock()
        mock_influx_class.return_value = mock_influx

        # Mock Supabase to always fail
        mock_upsert = Mock()
        mock_upsert.execute.side_effect = Exception("Network error")
        mock_table = Mock()
        mock_table.upsert.return_value = mock_upsert
        mock_supabase = Mock()
        mock_supabase.table.return_value = mock_table
        mock_supabase_create.return_value = mock_supabase

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            state_file = f.name

        try:
            config = CloudSyncConfig(
                enabled=True,
                supabase_url="https://test.supabase.co",
                supabase_key="test-key",
                state_file=state_file,
                max_retries=3,
                retry_delay_seconds=0,
            )

            sync = CloudSync(
                config,
                influxdb_url="http://localhost:8086",
                influxdb_token="test-token",
                influxdb_org="test-org",
                influxdb_bucket="test-bucket",
            )

            records = [
                {
                    "time": "2024-06-15T14:30:00+00:00",
                    "site_id": "test-site",
                    "device_id": "test-device",
                    "metric_name": "solar.generation_kw",
                    "metric_value": 7.5,
                }
            ]

            successful, failed = sync._push_to_supabase(records)

            # Should fail after all retries
            assert successful == 0
            assert failed == 1
            assert mock_upsert.execute.call_count == 3

            sync.close()
        finally:
            Path(state_file).unlink(missing_ok=True)

    @patch("edge_gateway.storage.cloud_sync.create_client")
    @patch("edge_gateway.storage.cloud_sync.InfluxDBClient")
    def test_context_manager(self, mock_influx_class, mock_supabase_create):
        """Test context manager properly closes connections."""
        mock_influx = Mock()
        mock_influx_class.return_value = mock_influx

        mock_supabase = Mock()
        mock_supabase_create.return_value = mock_supabase

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            state_file = f.name

        try:
            config = CloudSyncConfig(
                enabled=True,
                supabase_url="https://test.supabase.co",
                supabase_key="test-key",
                state_file=state_file,
            )

            with CloudSync(
                config,
                influxdb_url="http://localhost:8086",
                influxdb_token="test-token",
                influxdb_org="test-org",
                influxdb_bucket="test-bucket",
            ) as sync:
                assert sync.is_connected()

            mock_influx.close.assert_called_once()
        finally:
            Path(state_file).unlink(missing_ok=True)

    @patch("edge_gateway.storage.cloud_sync.create_client")
    @patch("edge_gateway.storage.cloud_sync.InfluxDBClient")
    def test_get_sync_status(self, mock_influx_class, mock_supabase_create):
        """Test get_sync_status returns complete information."""
        mock_influx = Mock()
        mock_influx.health.return_value = Mock(status="pass")
        mock_influx_class.return_value = mock_influx

        mock_table = Mock()
        mock_table.select.return_value.limit.return_value.execute.return_value = None
        mock_supabase = Mock()
        mock_supabase.table.return_value = mock_table
        mock_supabase_create.return_value = mock_supabase

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            state_file = f.name

        try:
            config = CloudSyncConfig(
                enabled=True,
                supabase_url="https://test.supabase.co",
                supabase_key="test-key",
                state_file=state_file,
            )

            sync = CloudSync(
                config,
                influxdb_url="http://localhost:8086",
                influxdb_token="test-token",
                influxdb_org="test-org",
                influxdb_bucket="test-bucket",
            )

            status = sync.get_sync_status()

            assert "enabled" in status
            assert "connected" in status
            assert "health" in status
            assert "sync_state" in status

            assert status["enabled"] is True
            assert status["connected"] is True

            sync.close()
        finally:
            Path(state_file).unlink(missing_ok=True)


class TestCloudSyncConfigInGeneratorConfig:
    """Test CloudSyncConfig integration with GeneratorConfig."""

    def test_generator_config_includes_cloud_sync(self):
        """Test GeneratorConfig includes CloudSync configuration."""
        from edge_gateway.config import GeneratorConfig

        config = GeneratorConfig()

        assert hasattr(config, "cloud_sync")
        assert isinstance(config.cloud_sync, CloudSyncConfig)

    def test_generator_config_from_dict_with_cloud_sync(self):
        """Test GeneratorConfig.from_dict parses CloudSync config."""
        from edge_gateway.config import GeneratorConfig

        data = {
            "device_id": "test-device",
            "cloud_sync": {
                "supabase_url": "https://test.supabase.co",
                "supabase_key": "test-key",
                "enabled": True,
                "batch_size": 50,
            },
        }

        config = GeneratorConfig.from_dict(data)

        assert config.cloud_sync.supabase_url == "https://test.supabase.co"
        assert config.cloud_sync.supabase_key == "test-key"
        assert config.cloud_sync.enabled is True
        assert config.cloud_sync.batch_size == 50

    def test_generator_config_to_dict_includes_cloud_sync(self):
        """Test GeneratorConfig.to_dict includes CloudSync config."""
        from edge_gateway.config import GeneratorConfig

        config = GeneratorConfig()
        config.cloud_sync.enabled = True
        config.cloud_sync.supabase_url = "https://test.supabase.co"

        data = config.to_dict()

        assert "cloud_sync" in data
        assert data["cloud_sync"]["enabled"] is True
        assert data["cloud_sync"]["supabase_url"] == "https://test.supabase.co"
