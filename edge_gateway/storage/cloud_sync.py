"""Cloud sync module for syncing local InfluxDB data to Supabase.

This module provides functionality to:
1. Query data from local InfluxDB
2. Transform data for Supabase's readings table schema
3. Sync data to Supabase with duplicate prevention
4. Track sync state for data integrity
5. Log all sync operations (success and failure)
"""

import json
import logging
import os
import time as time_module
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Optional imports for InfluxDB and Supabase
try:
    from influxdb_client import InfluxDBClient

    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False
    InfluxDBClient = None  # type: ignore

try:
    from supabase import Client, create_client

    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    Client = None  # type: ignore
    create_client = None  # type: ignore


@dataclass
class CloudSyncConfig:
    """Configuration for cloud sync operations.

    Supports environment variable overrides:
    - SUPABASE_URL: Supabase project URL
    - SUPABASE_KEY: Supabase API key (anon or service role)
    - SYNC_BATCH_SIZE: Number of records per batch
    - SYNC_INTERVAL_SECONDS: Seconds between sync operations
    - SYNC_STATE_FILE: Path to sync state file

    Attributes:
        supabase_url: Supabase project URL
        supabase_key: Supabase API key
        enabled: Whether cloud sync is enabled
        batch_size: Number of records to sync per batch (default: 100)
        sync_interval_seconds: Seconds between sync cycles (default: 60)
        state_file: Path to the sync state file for tracking progress
        site_id: Site identifier for the readings table
        max_retries: Maximum retry attempts for failed operations (default: 3)
        retry_delay_seconds: Delay between retries (default: 5)
    """

    supabase_url: str = ""
    supabase_key: str = ""
    enabled: bool = False
    batch_size: int = 100
    sync_interval_seconds: int = 60
    state_file: str = ".sync_state.json"
    site_id: str = "edge-gateway-site"
    max_retries: int = 3
    retry_delay_seconds: int = 5

    def __post_init__(self) -> None:
        """Apply environment variable overrides only when values are at defaults.

        Precedence: explicit args > env vars > defaults
        """
        if self.supabase_url == "":
            self.supabase_url = os.environ.get("SUPABASE_URL", self.supabase_url)
        if self.supabase_key == "":
            self.supabase_key = os.environ.get("SUPABASE_KEY", self.supabase_key)
        if self.batch_size == 100:
            env_batch = os.environ.get("SYNC_BATCH_SIZE")
            if env_batch:
                self.batch_size = int(env_batch)
        if self.sync_interval_seconds == 60:
            env_interval = os.environ.get("SYNC_INTERVAL_SECONDS")
            if env_interval:
                self.sync_interval_seconds = int(env_interval)
        if self.state_file == ".sync_state.json":
            self.state_file = os.environ.get("SYNC_STATE_FILE", self.state_file)
        if self.site_id == "edge-gateway-site":
            self.site_id = os.environ.get("SYNC_SITE_ID", self.site_id)


class SyncState:
    """Manages sync state for tracking last synced timestamp.

    Stores the last successfully synced timestamp to prevent
    duplicate data and enable resume after failures.
    """

    def __init__(self, state_file: str) -> None:
        """Initialize sync state manager.

        Args:
            state_file: Path to the JSON file storing sync state
        """
        self.state_file = Path(state_file)
        self._state: dict[str, Any] = {
            "last_sync_time": None,
            "last_sync_count": 0,
            "total_synced": 0,
            "last_error": None,
            "last_error_time": None,
        }
        self._load_state()

    def _load_state(self) -> None:
        """Load state from file if it exists."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    loaded = json.load(f)
                    self._state.update(loaded)
                logger.info("Loaded sync state from %s", self.state_file)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load sync state, starting fresh: %s", e)

    def _save_state(self) -> None:
        """Save current state to file."""
        try:
            with open(self.state_file, "w") as f:
                json.dump(self._state, f, indent=2, default=str)
            logger.debug("Saved sync state to %s", self.state_file)
        except OSError as e:
            logger.error("Failed to save sync state: %s", e)

    @property
    def last_sync_time(self) -> Optional[datetime]:
        """Get the last successful sync timestamp."""
        time_str = self._state.get("last_sync_time")
        if time_str:
            return datetime.fromisoformat(time_str)
        return None

    @property
    def total_synced(self) -> int:
        """Get total number of records synced."""
        return self._state.get("total_synced", 0)

    def update_success(self, sync_time: datetime, record_count: int) -> None:
        """Update state after successful sync.

        Args:
            sync_time: Timestamp of the last synced record
            record_count: Number of records synced in this batch
        """
        self._state["last_sync_time"] = sync_time.isoformat()
        self._state["last_sync_count"] = record_count
        self._state["total_synced"] = self._state.get("total_synced", 0) + record_count
        self._state["last_error"] = None
        self._state["last_error_time"] = None
        self._save_state()
        logger.info(
            "Sync state updated: synced %d records, total: %d",
            record_count,
            self._state["total_synced"],
        )

    def update_error(self, error: str) -> None:
        """Update state after failed sync.

        Args:
            error: Error message from the failed sync
        """
        self._state["last_error"] = error
        self._state["last_error_time"] = datetime.now(timezone.utc).isoformat()
        self._save_state()
        logger.error("Sync state updated with error: %s", error)

    def get_status(self) -> dict[str, Any]:
        """Get current sync status.

        Returns:
            Dictionary containing sync state information
        """
        return {
            "last_sync_time": self._state.get("last_sync_time"),
            "last_sync_count": self._state.get("last_sync_count", 0),
            "total_synced": self._state.get("total_synced", 0),
            "last_error": self._state.get("last_error"),
            "last_error_time": self._state.get("last_error_time"),
        }


class CloudSync:
    """Cloud sync client for syncing InfluxDB data to Supabase.

    This class handles:
    - Querying unsynced data from InfluxDB
    - Transforming DER metrics to Supabase readings format
    - Batch uploading to Supabase with duplicate prevention
    - Tracking sync progress for reliability

    Example:
        >>> config = CloudSyncConfig(
        ...     supabase_url="https://your-project.supabase.co",
        ...     supabase_key="your-key",
        ...     enabled=True
        ... )
        >>> sync = CloudSync(config, influxdb_config)
        >>> sync.sync()
    """

    # Measurements to sync from InfluxDB
    MEASUREMENTS = ["solar", "battery", "home_load", "grid_price", "system"]

    # Field mappings from InfluxDB to Supabase metric names
    FIELD_MAPPINGS = {
        "solar": [
            "generation_kw",
            "irradiance_w_m2",
            "panel_temp_celsius",
            "efficiency_percent",
        ],
        "battery": [
            "soc_percent",
            "capacity_kwh",
            "power_kw",
            "voltage_v",
            "temperature_celsius",
            "cycles",
            "health_percent",
        ],
        "home_load": [
            "total_load_kw",
            "hvac_kw",
            "appliances_kw",
            "lighting_kw",
            "ev_charging_kw",
            "other_kw",
        ],
        "grid_price": [
            "price_per_kwh",
            "feed_in_tariff",
            "demand_charge",
            "carbon_intensity_g_kwh",
        ],
        "system": [
            "net_grid_flow_kw",
            "solar_generation_kw",
            "battery_soc_percent",
            "total_load_kw",
            "price_per_kwh",
        ],
    }

    def __init__(
        self,
        config: CloudSyncConfig,
        influxdb_url: str,
        influxdb_token: str,
        influxdb_org: str,
        influxdb_bucket: str,
    ) -> None:
        """Initialize cloud sync client.

        Args:
            config: Cloud sync configuration
            influxdb_url: InfluxDB server URL
            influxdb_token: InfluxDB authentication token
            influxdb_org: InfluxDB organization name
            influxdb_bucket: InfluxDB bucket name

        Raises:
            ImportError: If required libraries are not installed
            ValueError: If configuration is invalid
        """
        self.config = config
        self._influxdb_url = influxdb_url
        self._influxdb_token = influxdb_token
        self._influxdb_org = influxdb_org
        self._influxdb_bucket = influxdb_bucket

        self._influx_client: Optional[InfluxDBClient] = None
        self._supabase_client: Optional[Client] = None
        self._sync_state: Optional[SyncState] = None

        if not config.enabled:
            logger.info("Cloud sync is disabled")
            return

        self._validate_dependencies()
        self._validate_config()
        self._connect()
        self._sync_state = SyncState(config.state_file)

    def _validate_dependencies(self) -> None:
        """Validate required libraries are installed."""
        if not INFLUXDB_AVAILABLE:
            raise ImportError(
                "influxdb-client is not installed. "
                "Install it with: pip install influxdb-client"
            )
        if not SUPABASE_AVAILABLE:
            raise ImportError(
                "supabase is not installed. " "Install it with: pip install supabase"
            )

    def _validate_config(self) -> None:
        """Validate configuration values."""
        if not self.config.supabase_url:
            raise ValueError("Supabase URL is required when cloud sync is enabled")
        if not self.config.supabase_key:
            raise ValueError("Supabase key is required when cloud sync is enabled")
        if not self._influxdb_token:
            raise ValueError("InfluxDB token is required for cloud sync")

    def _connect(self) -> None:
        """Establish connections to InfluxDB and Supabase."""
        # Connect to InfluxDB
        try:
            self._influx_client = InfluxDBClient(
                url=self._influxdb_url,
                token=self._influxdb_token,
                org=self._influxdb_org,
            )
            logger.info(
                "Connected to InfluxDB at %s for cloud sync",
                self._influxdb_url,
            )
        except Exception as e:
            logger.exception("Failed to connect to InfluxDB: %s", e)
            raise

        # Connect to Supabase
        try:
            self._supabase_client = create_client(
                self.config.supabase_url,
                self.config.supabase_key,
            )
            logger.info(
                "Connected to Supabase at %s",
                self.config.supabase_url,
            )
        except Exception as e:
            logger.exception("Failed to connect to Supabase: %s", e)
            raise

    def is_connected(self) -> bool:
        """Check if both connections are established.

        Returns:
            True if connected to both InfluxDB and Supabase
        """
        if not self.config.enabled:
            return False
        return self._influx_client is not None and self._supabase_client is not None

    def health_check(self) -> dict[str, bool]:
        """Perform health checks on both connections.

        Returns:
            Dictionary with health status for each service
        """
        status = {"influxdb": False, "supabase": False}

        if not self.is_connected():
            return status

        # Check InfluxDB
        try:
            health = self._influx_client.health()
            status["influxdb"] = health.status == "pass"
        except Exception as e:
            logger.warning("InfluxDB health check failed: %s", e)

        # Check Supabase
        try:
            # Simple query to verify connection
            self._supabase_client.table("readings").select("time").limit(1).execute()
            status["supabase"] = True
        except Exception as e:
            logger.warning("Supabase health check failed: %s", e)

        return status

    def query_unsynced_data(
        self,
        start_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Query data from InfluxDB that hasn't been synced yet.

        Args:
            start_time: Start time for query (uses last sync time if None)
            limit: Maximum number of records to return (uses batch_size if None)

        Returns:
            List of records ready for Supabase insertion
        """
        if not self.is_connected():
            logger.warning("Not connected, cannot query data")
            return []

        # Determine start time
        if start_time is None and self._sync_state:
            start_time = self._sync_state.last_sync_time

        # Build Flux query
        query_api = self._influx_client.query_api()
        records = []

        for measurement in self.MEASUREMENTS:
            flux_query = self._build_flux_query(
                measurement,
                start_time,
                limit or self.config.batch_size,
            )

            try:
                tables = query_api.query(flux_query, org=self._influxdb_org)

                for table in tables:
                    for record in table.records:
                        # Transform to Supabase format
                        supabase_record = self._transform_record(
                            measurement,
                            record,
                        )
                        if supabase_record:
                            records.append(supabase_record)

            except Exception as e:
                logger.error(
                    "Failed to query %s from InfluxDB: %s",
                    measurement,
                    e,
                )

        logger.info("Queried %d records from InfluxDB", len(records))
        return records

    def _build_flux_query(
        self,
        measurement: str,
        start_time: Optional[datetime],
        limit: int,
    ) -> str:
        """Build Flux query for a measurement.

        Args:
            measurement: The measurement to query
            start_time: Start time filter
            limit: Maximum records per measurement

        Returns:
            Flux query string
        """
        # Start time handling
        if start_time:
            start_filter = f'start: {start_time.isoformat()}Z'
        else:
            # Default to last 24 hours if no sync state
            start_filter = "start: -24h"

        fields = self.FIELD_MAPPINGS.get(measurement, [])
        field_filter = " or ".join([f'r._field == "{f}"' for f in fields])

        query = f'''
from(bucket: "{self._influxdb_bucket}")
  |> range({start_filter})
  |> filter(fn: (r) => r._measurement == "{measurement}")
  |> filter(fn: (r) => {field_filter})
  |> sort(columns: ["_time"])
  |> limit(n: {limit})
'''
        return query

    def _transform_record(
        self,
        measurement: str,
        record: Any,
    ) -> Optional[dict[str, Any]]:
        """Transform InfluxDB record to Supabase format.

        The Supabase readings table uses:
        - time: TIMESTAMPTZ
        - site_id: TEXT
        - device_id: TEXT
        - metric_name: TEXT (measurement.field)
        - metric_value: DOUBLE PRECISION

        Args:
            measurement: The measurement name
            record: InfluxDB record object

        Returns:
            Dictionary ready for Supabase insertion, or None if invalid
        """
        try:
            timestamp = record.get_time()
            if timestamp is None:
                return None

            # Format timestamp for Supabase
            if hasattr(timestamp, "isoformat"):
                time_str = timestamp.isoformat()
            else:
                time_str = str(timestamp)

            # Get device_id from tags
            device_id = record.values.get("device_id", "unknown")

            # Build metric name from measurement and field
            field_name = record.get_field()
            metric_name = f"{measurement}.{field_name}"

            # Get the value
            value = record.get_value()
            if value is None:
                return None

            return {
                "time": time_str,
                "site_id": self.config.site_id,
                "device_id": device_id,
                "metric_name": metric_name,
                "metric_value": float(value),
            }
        except Exception as e:
            logger.warning("Failed to transform record: %s", e)
            return None

    def sync(self) -> tuple[int, int]:
        """Perform a sync operation.

        Queries unsynced data from InfluxDB and pushes to Supabase.

        Returns:
            Tuple of (successful_count, failed_count)
        """
        if not self.config.enabled:
            logger.debug("Cloud sync is disabled, skipping")
            return (0, 0)

        if not self.is_connected():
            logger.error("Not connected, cannot sync")
            if self._sync_state:
                self._sync_state.update_error("Not connected to data sources")
            return (0, 0)

        logger.info("Starting cloud sync operation")

        # Query unsynced data
        records = self.query_unsynced_data()
        if not records:
            logger.info("No new data to sync")
            return (0, 0)

        # Sync to Supabase
        successful, failed = self._push_to_supabase(records)

        # Update sync state with the latest timestamp
        if successful > 0:
            # Find the latest timestamp from synced records
            latest_time = max(
                datetime.fromisoformat(r["time"].replace("Z", "+00:00"))
                for r in records[:successful]
            )
            self._sync_state.update_success(latest_time, successful)

        if failed > 0:
            self._sync_state.update_error(f"Failed to sync {failed} records")

        logger.info(
            "Cloud sync completed: %d successful, %d failed",
            successful,
            failed,
        )
        return (successful, failed)

    def _push_to_supabase(
        self,
        records: list[dict[str, Any]],
    ) -> tuple[int, int]:
        """Push records to Supabase with duplicate prevention.

        Uses upsert to prevent duplicate records based on
        (time, site_id, device_id, metric_name) combination.

        Args:
            records: List of records to push

        Returns:
            Tuple of (successful_count, failed_count)
        """
        successful = 0
        failed = 0

        # Process in batches
        batch_size = self.config.batch_size
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]

            for retry in range(self.config.max_retries):
                try:
                    # Use upsert for duplicate prevention
                    # The readings table should have a unique constraint on
                    # (time, site_id, device_id, metric_name)
                    (
                        self._supabase_client.table("readings")
                        .upsert(batch, on_conflict="time,site_id,device_id,metric_name")
                        .execute()
                    )

                    successful += len(batch)
                    logger.debug(
                        "Successfully pushed batch of %d records to Supabase",
                        len(batch),
                    )
                    break  # Success, exit retry loop

                except Exception as e:
                    logger.warning(
                        "Batch push failed (attempt %d/%d): %s",
                        retry + 1,
                        self.config.max_retries,
                        e,
                    )
                    if retry == self.config.max_retries - 1:
                        failed += len(batch)
                        logger.error(
                            "Failed to push batch of %d records after %d retries",
                            len(batch),
                            self.config.max_retries,
                        )
                    else:
                        time_module.sleep(self.config.retry_delay_seconds)

        return (successful, failed)

    def get_sync_status(self) -> dict[str, Any]:
        """Get current sync status.

        Returns:
            Dictionary containing sync state and health information
        """
        status = {
            "enabled": self.config.enabled,
            "connected": self.is_connected(),
            "health": self.health_check() if self.is_connected() else {},
        }

        if self._sync_state:
            status["sync_state"] = self._sync_state.get_status()

        return status

    def close(self) -> None:
        """Close all connections."""
        if self._influx_client:
            try:
                self._influx_client.close()
            except Exception as e:
                logger.debug("Error closing InfluxDB client: %s", e)
            self._influx_client = None

        self._supabase_client = None
        logger.info("Closed cloud sync connections")

    def __enter__(self) -> "CloudSync":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
