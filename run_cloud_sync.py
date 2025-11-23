#!/usr/bin/env python3
"""Cloud sync runner for Edge Gateway.

This script runs a continuous process that syncs data from the local
InfluxDB database to the Supabase cloud database.

Usage:
    python run_cloud_sync.py [options]

Options:
    --once              Run sync once and exit
    --continuous        Run continuous sync (default)
    --interval SECONDS  Sync interval in seconds (default: 60)
    --config FILE       Path to configuration file (JSON)
    --quiet             Reduce logging verbosity
    --debug             Enable debug logging
    --status            Show current sync status and exit

Environment variables:
    SUPABASE_URL        Supabase project URL
    SUPABASE_KEY        Supabase API key
    INFLUXDB_URL        InfluxDB server URL
    INFLUXDB_TOKEN      InfluxDB authentication token
    INFLUXDB_ORG        InfluxDB organization
    INFLUXDB_BUCKET     InfluxDB bucket name
    SYNC_BATCH_SIZE     Records per sync batch
    SYNC_SITE_ID        Site identifier for readings
"""

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
# Must happen before importing modules that use env vars
load_dotenv()

from edge_gateway.storage.cloud_sync import CloudSync, CloudSyncConfig  # noqa: E402

logger = logging.getLogger(__name__)


class CloudSyncRunner:
    """Runner for continuous cloud sync operations."""

    def __init__(
        self,
        config: CloudSyncConfig,
        influxdb_url: str,
        influxdb_token: str,
        influxdb_org: str,
        influxdb_bucket: str,
    ) -> None:
        """Initialize the sync runner.

        Args:
            config: Cloud sync configuration
            influxdb_url: InfluxDB server URL
            influxdb_token: InfluxDB authentication token
            influxdb_org: InfluxDB organization
            influxdb_bucket: InfluxDB bucket name
        """
        self.config = config
        self._influxdb_url = influxdb_url
        self._influxdb_token = influxdb_token
        self._influxdb_org = influxdb_org
        self._influxdb_bucket = influxdb_bucket
        self._running = False
        self._sync: CloudSync | None = None

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum: int, frame) -> None:
        """Handle shutdown signals."""
        logger.info("Received signal %d, initiating shutdown...", signum)
        self._running = False

    def run_once(self) -> tuple[int, int]:
        """Run a single sync operation.

        Returns:
            Tuple of (successful_count, failed_count)
        """
        logger.info("Running single sync operation")

        with CloudSync(
            self.config,
            self._influxdb_url,
            self._influxdb_token,
            self._influxdb_org,
            self._influxdb_bucket,
        ) as sync:
            return sync.sync()

    def run_continuous(self) -> None:
        """Run continuous sync operations."""
        logger.info(
            "Starting continuous cloud sync (interval: %ds)",
            self.config.sync_interval_seconds,
        )

        self._setup_signal_handlers()
        self._running = True

        self._sync = CloudSync(
            self.config,
            self._influxdb_url,
            self._influxdb_token,
            self._influxdb_org,
            self._influxdb_bucket,
        )

        total_successful = 0
        total_failed = 0
        cycle_count = 0

        try:
            while self._running:
                cycle_count += 1
                logger.info("Starting sync cycle %d", cycle_count)

                try:
                    successful, failed = self._sync.sync()
                    total_successful += successful
                    total_failed += failed

                    logger.info(
                        "Cycle %d complete: %d synced, %d failed (total: %d/%d)",
                        cycle_count,
                        successful,
                        failed,
                        total_successful,
                        total_failed,
                    )

                except Exception as e:
                    logger.exception("Sync cycle %d failed: %s", cycle_count, e)

                # Wait for next interval
                if self._running:
                    logger.debug(
                        "Waiting %d seconds until next sync",
                        self.config.sync_interval_seconds,
                    )
                    # Use interruptible sleep
                    for _ in range(self.config.sync_interval_seconds):
                        if not self._running:
                            break
                        time.sleep(1)

        finally:
            if self._sync:
                self._sync.close()

            logger.info(
                "Cloud sync stopped. Total synced: %d, Total failed: %d",
                total_successful,
                total_failed,
            )

    def get_status(self) -> dict:
        """Get current sync status.

        Returns:
            Dictionary with sync status information
        """
        with CloudSync(
            self.config,
            self._influxdb_url,
            self._influxdb_token,
            self._influxdb_org,
            self._influxdb_bucket,
        ) as sync:
            return sync.get_sync_status()


def setup_logging(quiet: bool = False, debug: bool = False) -> None:
    """Configure logging for the application.

    Args:
        quiet: If True, only show warnings and errors
        debug: If True, show debug messages
    """
    if debug:
        level = logging.DEBUG
    elif quiet:
        level = logging.WARNING
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_config_from_file(path: Path) -> dict:
    """Load configuration from JSON file.

    Args:
        path: Path to JSON configuration file

    Returns:
        Dictionary with configuration values
    """
    import json

    with open(path) as f:
        return json.load(f)


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    parser = argparse.ArgumentParser(
        description="Cloud sync runner for Edge Gateway",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Execution mode
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--once",
        action="store_true",
        help="Run sync once and exit",
    )
    mode_group.add_argument(
        "--continuous",
        action="store_true",
        default=True,
        help="Run continuous sync (default)",
    )
    mode_group.add_argument(
        "--status",
        action="store_true",
        help="Show current sync status and exit",
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to JSON configuration file",
    )
    parser.add_argument(
        "--interval",
        type=int,
        help="Sync interval in seconds (default: 60)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Records per sync batch (default: 100)",
    )
    parser.add_argument(
        "--site-id",
        type=str,
        help="Site identifier for readings",
    )

    # Supabase configuration
    parser.add_argument(
        "--supabase-url",
        type=str,
        help="Supabase project URL",
    )
    parser.add_argument(
        "--supabase-key",
        type=str,
        help="Supabase API key",
    )

    # InfluxDB configuration
    parser.add_argument(
        "--influxdb-url",
        type=str,
        default="http://localhost:8086",
        help="InfluxDB server URL (default: http://localhost:8086)",
    )
    parser.add_argument(
        "--influxdb-token",
        type=str,
        help="InfluxDB authentication token",
    )
    parser.add_argument(
        "--influxdb-org",
        type=str,
        default="edge-gateway",
        help="InfluxDB organization (default: edge-gateway)",
    )
    parser.add_argument(
        "--influxdb-bucket",
        type=str,
        default="der-data",
        help="InfluxDB bucket (default: der-data)",
    )

    # Logging
    logging_group = parser.add_mutually_exclusive_group()
    logging_group.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging verbosity",
    )
    logging_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(quiet=args.quiet, debug=args.debug)

    # Load configuration from file if provided
    file_config = {}
    if args.config:
        try:
            file_config = load_config_from_file(args.config)
            logger.info("Loaded configuration from %s", args.config)
        except Exception as e:
            logger.error("Failed to load config file: %s", e)
            return 1

    # Build CloudSyncConfig with CLI overrides
    cloud_sync_config = file_config.get("cloud_sync", {})

    sync_config = CloudSyncConfig(
        supabase_url=args.supabase_url or cloud_sync_config.get("supabase_url", ""),
        supabase_key=args.supabase_key or cloud_sync_config.get("supabase_key", ""),
        enabled=True,  # Always enabled when running this script
        batch_size=args.batch_size or cloud_sync_config.get("batch_size", 100),
        sync_interval_seconds=args.interval
        or cloud_sync_config.get("sync_interval_seconds", 60),
        site_id=args.site_id or cloud_sync_config.get("site_id", "edge-gateway-site"),
    )

    # Get InfluxDB configuration
    influxdb_config = file_config.get("influxdb", {})
    influxdb_url = args.influxdb_url or influxdb_config.get(
        "url", "http://localhost:8086"
    )
    influxdb_token = args.influxdb_token or influxdb_config.get("token", "")
    influxdb_org = args.influxdb_org or influxdb_config.get("org", "edge-gateway")
    influxdb_bucket = args.influxdb_bucket or influxdb_config.get("bucket", "der-data")

    # Use environment variables as fallback
    import os

    influxdb_url = influxdb_url or os.environ.get(
        "INFLUXDB_URL", "http://localhost:8086"
    )
    influxdb_token = influxdb_token or os.environ.get("INFLUXDB_TOKEN", "")
    influxdb_org = influxdb_org or os.environ.get("INFLUXDB_ORG", "edge-gateway")
    influxdb_bucket = influxdb_bucket or os.environ.get("INFLUXDB_BUCKET", "der-data")

    # Validate required configuration
    if not sync_config.supabase_url or not sync_config.supabase_key:
        logger.error(
            "Supabase URL and key are required. "
            "Set SUPABASE_URL and SUPABASE_KEY environment variables "
            "or use --supabase-url and --supabase-key arguments."
        )
        return 1

    if not influxdb_token:
        logger.error(
            "InfluxDB token is required. "
            "Set INFLUXDB_TOKEN environment variable "
            "or use --influxdb-token argument."
        )
        return 1

    # Create runner
    try:
        runner = CloudSyncRunner(
            config=sync_config,
            influxdb_url=influxdb_url,
            influxdb_token=influxdb_token,
            influxdb_org=influxdb_org,
            influxdb_bucket=influxdb_bucket,
        )
    except Exception as e:
        logger.error("Failed to initialize sync runner: %s", e)
        return 1

    # Execute based on mode
    try:
        if args.status:
            status = runner.get_status()
            print("\nCloud Sync Status:")
            print("-" * 40)
            print(f"  Enabled: {status.get('enabled', False)}")
            print(f"  Connected: {status.get('connected', False)}")

            health = status.get("health", {})
            print(f"  InfluxDB Health: {'OK' if health.get('influxdb') else 'FAIL'}")
            print(f"  Supabase Health: {'OK' if health.get('supabase') else 'FAIL'}")

            sync_state = status.get("sync_state", {})
            if sync_state:
                print("\nSync State:")
                print(f"  Last Sync Time: {sync_state.get('last_sync_time', 'Never')}")
                print(f"  Last Sync Count: {sync_state.get('last_sync_count', 0)}")
                print(f"  Total Synced: {sync_state.get('total_synced', 0)}")
                if sync_state.get("last_error"):
                    print(f"  Last Error: {sync_state.get('last_error')}")
                    print(f"  Error Time: {sync_state.get('last_error_time')}")

            return 0

        elif args.once:
            successful, failed = runner.run_once()
            print(f"\nSync complete: {successful} successful, {failed} failed")
            return 0 if failed == 0 else 1

        else:
            # Continuous mode
            runner.run_continuous()
            return 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
