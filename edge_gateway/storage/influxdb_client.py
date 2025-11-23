"""InfluxDB client for local time series data storage.

This module provides a client for writing DER data to a local InfluxDB instance,
designed for edge gateway deployments where data needs to be stored locally
before potential sync to cloud services.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

from edge_gateway.models import DERData

logger = logging.getLogger(__name__)

# InfluxDB client imports - these are optional dependencies
try:
    from influxdb_client import InfluxDBClient, Point
    from influxdb_client.client.write_api import SYNCHRONOUS

    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False
    InfluxDBClient = None  # type: ignore
    Point = None  # type: ignore
    SYNCHRONOUS = None  # type: ignore


@dataclass
class InfluxDBConfig:
    """Configuration for InfluxDB connection.

    Supports environment variable overrides:
    - INFLUXDB_URL: Server URL
    - INFLUXDB_TOKEN: Authentication token
    - INFLUXDB_ORG: Organization name
    - INFLUXDB_BUCKET: Bucket name

    Attributes:
        url: InfluxDB server URL (e.g., "http://localhost:8086")
        token: Authentication token for InfluxDB
        org: Organization name in InfluxDB
        bucket: Bucket name for storing data
        enabled: Whether InfluxDB storage is enabled
        retention_days: Data retention period in days (default: 7)
        batch_size: Number of points to batch before writing (default: 1)
        flush_interval_ms: Milliseconds between batch flushes (default: 1000)
    """

    url: str = "http://localhost:8086"
    token: str = ""
    org: str = "edge-gateway"
    bucket: str = "der-data"
    enabled: bool = False
    retention_days: int = 7
    batch_size: int = 1
    flush_interval_ms: int = 1000

    def __post_init__(self) -> None:
        """Apply environment variable overrides only when values are at defaults.

        Precedence: explicit args > env vars > defaults
        """
        if self.url == "http://localhost:8086":
            self.url = os.environ.get("INFLUXDB_URL", self.url)
        if self.token == "":
            self.token = os.environ.get("INFLUXDB_TOKEN", self.token)
        if self.org == "edge-gateway":
            self.org = os.environ.get("INFLUXDB_ORG", self.org)
        if self.bucket == "der-data":
            self.bucket = os.environ.get("INFLUXDB_BUCKET", self.bucket)


class InfluxDBStorage:
    """InfluxDB storage client for DER data.

    This class handles writing DER data to a local InfluxDB instance.
    It converts DERData objects into InfluxDB points with appropriate
    measurements, tags, and fields.

    Example:
        >>> config = InfluxDBConfig(
        ...     url="http://localhost:8086",
        ...     token="my-token",
        ...     org="my-org",
        ...     bucket="der-data",
        ...     enabled=True
        ... )
        >>> storage = InfluxDBStorage(config)
        >>> storage.write(der_data)
    """

    # Measurement names for different data types
    MEASUREMENT_SOLAR = "solar"
    MEASUREMENT_BATTERY = "battery"
    MEASUREMENT_HOME_LOAD = "home_load"
    MEASUREMENT_GRID_PRICE = "grid_price"
    MEASUREMENT_SYSTEM = "system"

    def __init__(self, config: InfluxDBConfig) -> None:
        """Initialize InfluxDB storage.

        Args:
            config: InfluxDB configuration

        Raises:
            ImportError: If influxdb-client is not installed
            ValueError: If configuration is invalid
        """
        self.config = config
        self._client: Optional[InfluxDBClient] = None
        self._write_api = None

        if not config.enabled:
            logger.info("InfluxDB storage is disabled")
            return

        if not INFLUXDB_AVAILABLE:
            raise ImportError(
                "influxdb-client is not installed. "
                "Install it with: pip install influxdb-client"
            )

        if not config.token:
            raise ValueError("InfluxDB token is required when storage is enabled")

        self._connect()

    def _connect(self) -> None:
        """Establish connection to InfluxDB."""
        if not self.config.enabled:
            return

        try:
            self._client = InfluxDBClient(
                url=self.config.url,
                token=self.config.token,
                org=self.config.org,
            )

            # Use synchronous writes for simplicity and reliability
            # For high-throughput scenarios, consider batching
            if self.config.batch_size > 1:
                from influxdb_client.client.write_api import WriteOptions

                self._write_api = self._client.write_api(
                    write_options=WriteOptions(
                        batch_size=self.config.batch_size,
                        flush_interval=self.config.flush_interval_ms,
                    )
                )
            else:
                self._write_api = self._client.write_api(write_options=SYNCHRONOUS)

            logger.info(
                "Connected to InfluxDB at %s (org=%s, bucket=%s)",
                self.config.url,
                self.config.org,
                self.config.bucket,
            )
        except Exception as e:
            logger.exception("Failed to connect to InfluxDB: %s", e)
            raise

    def is_connected(self) -> bool:
        """Check if connected to InfluxDB.

        Returns:
            True if connected and ready to write
        """
        if not self.config.enabled:
            return False
        return self._client is not None and self._write_api is not None

    def health_check(self) -> bool:
        """Perform a health check on the InfluxDB connection.

        Returns:
            True if InfluxDB is healthy and accessible
        """
        if not self.is_connected():
            return False

        try:
            health = self._client.health()
            return health.status == "pass"
        except Exception as e:
            logger.warning("InfluxDB health check failed: %s", e)
            return False

    def write(self, data: DERData) -> bool:
        """Write DER data to InfluxDB.

        This method writes the DER data as multiple measurements:
        - solar: Solar generation metrics
        - battery: Battery state metrics
        - home_load: Home consumption metrics
        - grid_price: Grid pricing metrics
        - system: System-level metrics (net grid flow)

        Args:
            data: DER data to write

        Returns:
            True if write was successful, False otherwise
        """
        if not self.config.enabled:
            return True  # Silently succeed when disabled

        if not self.is_connected():
            logger.warning("Not connected to InfluxDB, skipping write")
            return False

        try:
            points = self._der_data_to_points(data)
            self._write_api.write(
                bucket=self.config.bucket,
                org=self.config.org,
                record=points,
            )
            logger.debug(
                "Wrote %d points to InfluxDB for device %s",
                len(points),
                data.device_id,
            )
            return True
        except Exception as e:
            logger.exception("Failed to write to InfluxDB: %s", e)
            return False

    def write_batch(self, data_list: list[DERData]) -> bool:
        """Write multiple DER data points to InfluxDB.

        Args:
            data_list: List of DER data to write

        Returns:
            True if all writes were successful, False otherwise
        """
        if not self.config.enabled:
            return True

        if not self.is_connected():
            logger.warning("Not connected to InfluxDB, skipping batch write")
            return False

        try:
            all_points = []
            for data in data_list:
                all_points.extend(self._der_data_to_points(data))

            self._write_api.write(
                bucket=self.config.bucket,
                org=self.config.org,
                record=all_points,
            )
            logger.debug(
                "Wrote %d points to InfluxDB (%d data records)",
                len(all_points),
                len(data_list),
            )
            return True
        except Exception as e:
            logger.exception("Failed to batch write to InfluxDB: %s", e)
            return False

    def _der_data_to_points(self, data: DERData) -> list:
        """Convert DERData to InfluxDB points.

        Args:
            data: DER data to convert

        Returns:
            List of InfluxDB Point objects
        """
        timestamp = data.timestamp
        device_id = data.device_id

        points = []

        # Solar measurement
        solar_point = (
            Point(self.MEASUREMENT_SOLAR)
            .tag("device_id", device_id)
            .field("generation_kw", data.solar.generation_kw)
            .field("irradiance_w_m2", data.solar.irradiance_w_m2)
            .field("panel_temp_celsius", data.solar.panel_temp_celsius)
            .field("efficiency_percent", data.solar.efficiency_percent)
            .time(timestamp)
        )
        points.append(solar_point)

        # Battery measurement
        battery_point = (
            Point(self.MEASUREMENT_BATTERY)
            .tag("device_id", device_id)
            .field("soc_percent", data.battery.soc_percent)
            .field("capacity_kwh", data.battery.capacity_kwh)
            .field("power_kw", data.battery.power_kw)
            .field("voltage_v", data.battery.voltage_v)
            .field("temperature_celsius", data.battery.temperature_celsius)
            .field("cycles", data.battery.cycles)
            .field("health_percent", data.battery.health_percent)
            .time(timestamp)
        )
        points.append(battery_point)

        # Home load measurement
        home_load_point = (
            Point(self.MEASUREMENT_HOME_LOAD)
            .tag("device_id", device_id)
            .field("total_load_kw", data.home_load.total_load_kw)
            .field("hvac_kw", data.home_load.hvac_kw)
            .field("appliances_kw", data.home_load.appliances_kw)
            .field("lighting_kw", data.home_load.lighting_kw)
            .field("ev_charging_kw", data.home_load.ev_charging_kw)
            .field("other_kw", data.home_load.other_kw)
            .time(timestamp)
        )
        points.append(home_load_point)

        # Grid price measurement
        grid_price_point = (
            Point(self.MEASUREMENT_GRID_PRICE)
            .tag("device_id", device_id)
            .tag("time_of_use_period", data.grid_price.time_of_use_period)
            .field("price_per_kwh", data.grid_price.price_per_kwh)
            .field("feed_in_tariff", data.grid_price.feed_in_tariff)
            .field("demand_charge", data.grid_price.demand_charge)
            .field("carbon_intensity_g_kwh", data.grid_price.carbon_intensity_g_kwh)
            .time(timestamp)
        )
        points.append(grid_price_point)

        # System-level measurement (aggregated metrics)
        system_point = (
            Point(self.MEASUREMENT_SYSTEM)
            .tag("device_id", device_id)
            .field("net_grid_flow_kw", data.net_grid_flow_kw)
            .field("solar_generation_kw", data.solar.generation_kw)
            .field("battery_soc_percent", data.battery.soc_percent)
            .field("total_load_kw", data.home_load.total_load_kw)
            .field("price_per_kwh", data.grid_price.price_per_kwh)
            .time(timestamp)
        )
        points.append(system_point)

        return points

    def setup_retention_policy(self) -> bool:
        """Set up retention policy for the bucket.

        Creates or updates the bucket with the configured retention period.

        Returns:
            True if successful, False otherwise
        """
        if not self.config.enabled or not self.is_connected():
            return False

        try:
            buckets_api = self._client.buckets_api()
            org_api = self._client.organizations_api()

            # Find the organization
            orgs = org_api.find_organizations(org=self.config.org)
            if not orgs:
                logger.error("Organization '%s' not found", self.config.org)
                return False
            org_id = orgs[0].id

            # Check if bucket exists
            existing_bucket = buckets_api.find_bucket_by_name(self.config.bucket)

            retention_seconds = self.config.retention_days * 24 * 60 * 60

            if existing_bucket:
                # Update existing bucket's retention
                existing_bucket.retention_rules = [
                    {"type": "expire", "everySeconds": retention_seconds}
                ]
                buckets_api.update_bucket(bucket=existing_bucket)
                logger.info(
                    "Updated bucket '%s' retention to %d days",
                    self.config.bucket,
                    self.config.retention_days,
                )
            else:
                # Create new bucket with retention policy
                from influxdb_client import BucketRetentionRules

                retention_rules = BucketRetentionRules(
                    type="expire", every_seconds=retention_seconds
                )
                buckets_api.create_bucket(
                    bucket_name=self.config.bucket,
                    org_id=org_id,
                    retention_rules=retention_rules,
                )
                logger.info(
                    "Created bucket '%s' with %d days retention",
                    self.config.bucket,
                    self.config.retention_days,
                )

            return True
        except Exception as e:
            logger.exception("Failed to setup retention policy: %s", e)
            return False

    def close(self) -> None:
        """Close the InfluxDB connection."""
        if self._write_api:
            try:
                self._write_api.close()
            except Exception as e:
              logger.debug("Error closing InfluxDB write API: %s", e)
            self._write_api = None

        if self._client:
            try:
                self._client.close()
            except Exception as e:
                logger.debug("Error closing InfluxDB client: %s", e)
            self._client = None

        logger.info("Closed InfluxDB connection")

    def __enter__(self) -> "InfluxDBStorage":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
