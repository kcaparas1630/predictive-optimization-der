#!/usr/bin/env python3
"""CLI runner for the AI Feature Engineering Pipeline.

This script runs the feature engineering pipeline to:
1. Query raw DER data from Supabase
2. Engineer features (temporal, rolling, categorical, lag)
3. Store results in the training_data table

Usage:
    # Run once with default settings
    python run_feature_engineering.py --once

    # Run with custom lookback period
    python run_feature_engineering.py --once --lookback-days 60

    # Run in continuous mode (scheduled runs)
    python run_feature_engineering.py --continuous --interval 3600

    # Show status and exit
    python run_feature_engineering.py --status

Environment variables:
    SUPABASE_URL: Supabase project URL
    SUPABASE_KEY: Supabase API key
    FE_SITE_ID: Site identifier to filter data
    FE_BATCH_SIZE: Records per batch (default: 1000)
    FE_ROLLING_WINDOW_DAYS: Rolling window days (default: 7)
    FE_LOOKBACK_DAYS: Historical data lookback (default: 30)
"""

import argparse
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class FeatureEngineeringRunner:
    """Runner class for the feature engineering pipeline.

    Handles execution modes (once, continuous), signal handling,
    and provides status reporting.
    """

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        site_id: str = "",
        batch_size: int = 1000,
        rolling_window_days: int = 7,
        lookback_days: int = 30,
        interval_seconds: int = 3600,
    ) -> None:
        """Initialize the feature engineering runner.

        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase API key
            site_id: Site identifier to filter data
            batch_size: Records per batch
            rolling_window_days: Days for rolling averages
            lookback_days: Days of historical data to process
            interval_seconds: Seconds between continuous runs
        """
        self._running = False
        self._interval_seconds = interval_seconds
        self._cycles_completed = 0
        self._total_records_stored = 0

        # Import here to avoid circular imports and handle missing dependencies
        try:
            from cloud.feature_engineering import (
                FeatureEngineeringConfig,
                FeatureEngineeringPipeline,
            )
        except ImportError as e:
            logger.error("Failed to import feature engineering module: %s", e)
            raise

        # Create configuration
        self._config = FeatureEngineeringConfig(
            supabase_url=supabase_url or "",
            supabase_key=supabase_key or "",
            enabled=True,
            site_id=site_id,
            batch_size=batch_size,
            rolling_window_days=rolling_window_days,
            lookback_days=lookback_days,
        )

        # Create pipeline
        self._pipeline = FeatureEngineeringPipeline(self._config)

    def run_once(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> dict:
        """Run the feature engineering pipeline once.

        Args:
            start_time: Start of time range
            end_time: End of time range

        Returns:
            Result dictionary from pipeline execution
        """
        logger.info("Running feature engineering pipeline (single run)")

        result = self._pipeline.run(start_time, end_time)

        if result["status"] == "success":
            logger.info(
                "Pipeline completed: %d records processed, %d stored",
                result.get("records_processed", 0),
                result.get("records_stored", 0),
            )
        elif result["status"] == "no_data":
            logger.info("No data found in specified time range")
        else:
            logger.error("Pipeline failed: %s", result.get("error", "Unknown error"))

        return result

    def run_continuous(self) -> None:
        """Run the feature engineering pipeline continuously.

        Runs the pipeline at regular intervals until stopped
        via signal (SIGINT/SIGTERM).
        """
        self._running = True
        self._setup_signal_handlers()

        logger.info(
            "Starting continuous feature engineering (interval: %ds)",
            self._interval_seconds,
        )

        while self._running:
            cycle_start = datetime.now(timezone.utc)

            try:
                result = self._pipeline.run()

                if result["status"] == "success":
                    self._cycles_completed += 1
                    self._total_records_stored += result.get("records_stored", 0)
                    logger.info(
                        "Cycle %d completed: %d records stored (total: %d)",
                        self._cycles_completed,
                        result.get("records_stored", 0),
                        self._total_records_stored,
                    )
                elif result["status"] == "no_data":
                    logger.debug("No new data to process")
                else:
                    logger.warning(
                        "Cycle failed: %s", result.get("error", "Unknown error")
                    )

            except Exception:
                logger.exception("Error during feature engineering cycle")

            # Wait for next cycle
            if self._running:
                elapsed = (datetime.now(timezone.utc) - cycle_start).total_seconds()
                sleep_time = max(0, self._interval_seconds - elapsed)
                if sleep_time > 0:
                    logger.debug("Sleeping for %.1f seconds", sleep_time)
                    time.sleep(sleep_time)

        logger.info(
            "Feature engineering stopped. Total cycles: %d, records stored: %d",
            self._cycles_completed,
            self._total_records_stored,
        )

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            logger.info("Received signal %d, stopping...", signum)
            self._running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def stop(self) -> None:
        """Stop the continuous runner."""
        self._running = False

    def get_status(self) -> dict:
        """Get current status of the runner.

        Returns:
            Status dictionary
        """
        health = (
            self._pipeline.health_check() if self._pipeline.is_connected() else False
        )

        return {
            "connected": self._pipeline.is_connected(),
            "health": health,
            "cycles_completed": self._cycles_completed,
            "total_records_stored": self._total_records_stored,
            "config": {
                "site_id": self._config.site_id or "(all sites)",
                "batch_size": self._config.batch_size,
                "rolling_window_days": self._config.rolling_window_days,
                "lookback_days": self._config.lookback_days,
                "source_table": self._config.source_table,
                "target_table": self._config.target_table,
            },
        }

    def close(self) -> None:
        """Close the pipeline connection."""
        if self._pipeline:
            self._pipeline.close()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AI Feature Engineering Pipeline for DER data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run once with default settings
    python run_feature_engineering.py --once

    # Run with custom lookback
    python run_feature_engineering.py --once --lookback-days 60

    # Run continuously every hour
    python run_feature_engineering.py --continuous --interval 3600

    # Show connection status
    python run_feature_engineering.py --status
        """,
    )

    # Execution mode
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--once",
        action="store_true",
        help="Run feature engineering once and exit",
    )
    mode_group.add_argument(
        "--continuous",
        action="store_true",
        help="Run feature engineering continuously",
    )
    mode_group.add_argument(
        "--status",
        action="store_true",
        help="Show connection status and exit",
    )

    # Supabase connection
    parser.add_argument(
        "--supabase-url",
        type=str,
        help="Supabase project URL (or set SUPABASE_URL env var)",
    )
    parser.add_argument(
        "--supabase-key",
        type=str,
        help="Supabase API key (or set SUPABASE_KEY env var)",
    )

    # Processing options
    parser.add_argument(
        "--site-id",
        type=str,
        default="",
        help="Site ID to filter data (default: all sites)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Records per batch (default: 1000)",
    )
    parser.add_argument(
        "--rolling-window-days",
        type=int,
        default=7,
        help="Days for rolling averages (default: 7)",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=30,
        help="Days of historical data to process (default: 30)",
    )

    # Continuous mode options
    parser.add_argument(
        "--interval",
        type=int,
        default=3600,
        help="Seconds between continuous runs (default: 3600)",
    )

    # Time range options (for --once mode)
    parser.add_argument(
        "--start-time",
        type=str,
        help="Start time in ISO format (default: lookback-days ago)",
    )
    parser.add_argument(
        "--end-time",
        type=str,
        help="End time in ISO format (default: now)",
    )

    # Logging
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    args = parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("cloud.feature_engineering").setLevel(logging.DEBUG)

    try:
        runner = FeatureEngineeringRunner(
            supabase_url=args.supabase_url,
            supabase_key=args.supabase_key,
            site_id=args.site_id,
            batch_size=args.batch_size,
            rolling_window_days=args.rolling_window_days,
            lookback_days=args.lookback_days,
            interval_seconds=args.interval,
        )
    except Exception as e:
        logger.error("Failed to initialize feature engineering runner: %s", e)
        return 1

    try:
        if args.status:
            # Show status and exit
            status = runner.get_status()
            print("\nFeature Engineering Pipeline Status")
            print("=" * 40)
            print(f"Connected: {status['connected']}")
            print(f"Health: {'OK' if status['health'] else 'FAILED'}")
            print("\nConfiguration:")
            for key, value in status["config"].items():
                print(f"  {key}: {value}")
            return 0 if status["connected"] else 1

        elif args.once:
            # Parse time range if provided
            start_time = None
            end_time = None

            if args.start_time:
                start_time = datetime.fromisoformat(args.start_time)
                if start_time.tzinfo is None:
                    start_time = start_time.replace(tzinfo=timezone.utc)

            if args.end_time:
                end_time = datetime.fromisoformat(args.end_time)
                if end_time.tzinfo is None:
                    end_time = end_time.replace(tzinfo=timezone.utc)

            # Run once
            result = runner.run_once(start_time, end_time)
            return 0 if result["status"] in ("success", "no_data") else 1

        elif args.continuous:
            # Run continuously
            runner.run_continuous()
            return 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0
    except Exception as e:
        logger.exception("Feature engineering failed: %s", e)
        return 1
    finally:
        runner.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
