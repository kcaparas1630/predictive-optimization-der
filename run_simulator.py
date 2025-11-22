#!/usr/bin/env python3
"""
DER Data Simulator - Command Line Interface

Simulates realistic hourly DER data including:
- Solar generation
- Battery State of Charge
- Home Load
- Grid Price signals

Usage:
    # Run once
    python run_simulator.py --once

    # Run continuously every 5 minutes
    python run_simulator.py --continuous --interval 300

    # Run on schedule (aligned to clock)
    python run_simulator.py --scheduled --interval 300

    # Generate historical data
    python run_simulator.py --historical --start 2024-01-01 --end 2024-01-02

    # Use custom config file
    python run_simulator.py --config config.json --continuous
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

from edge_gateway.config import GeneratorConfig, DEFAULT_CONFIG
from edge_gateway.generator import DERDataGenerator, DataGeneratorRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def create_generator_from_config(config: GeneratorConfig) -> DERDataGenerator:
    """Create a DERDataGenerator from configuration."""
    return DERDataGenerator(
        device_id=config.device_id,
        solar_capacity_kw=config.solar.capacity_kw,
        latitude=config.solar.latitude,
        battery_capacity_kwh=config.battery.capacity_kwh,
        battery_max_power_kw=config.battery.max_charge_rate_kw,
        base_load_kw=config.home_load.base_load_kw,
        peak_load_kw=config.home_load.peak_load_kw,
        has_ev=config.home_load.has_ev,
        off_peak_price=config.grid_price.off_peak_price,
        shoulder_price=config.grid_price.shoulder_price,
        peak_price=config.grid_price.peak_price,
        seed=config.seed,
    )


def print_data(data) -> None:
    """Print data as formatted JSON to stdout."""
    print(data.to_json())


def run_once(config: GeneratorConfig) -> None:
    """Generate a single data point."""
    generator = create_generator_from_config(config)
    output_file = Path(config.output_file) if config.output_file else None

    runner = DataGeneratorRunner(
        generator=generator,
        output_callback=print_data,
        output_file=output_file,
    )
    runner.run_once()


def run_continuous(config: GeneratorConfig, duration: int | None = None) -> None:
    """Run generator continuously."""
    generator = create_generator_from_config(config)
    output_file = Path(config.output_file) if config.output_file else None

    runner = DataGeneratorRunner(
        generator=generator,
        output_callback=print_data,
        output_file=output_file,
    )
    runner.run_continuous(
        interval_seconds=config.interval_seconds,
        duration_seconds=duration,
    )


def run_scheduled(config: GeneratorConfig) -> None:
    """Run generator on schedule."""
    generator = create_generator_from_config(config)
    output_file = Path(config.output_file) if config.output_file else None

    runner = DataGeneratorRunner(
        generator=generator,
        output_callback=print_data,
        output_file=output_file,
    )
    runner.run_scheduled(
        interval_seconds=config.interval_seconds,
        align_to_interval=True,
    )


def generate_historical(
    config: GeneratorConfig,
    start: datetime,
    end: datetime,
    interval_minutes: int = 5,
) -> None:
    """Generate historical data for a time range."""
    generator = create_generator_from_config(config)
    data = generator.generate_historical(start, end, interval_minutes)

    logger.info(f"Generated {len(data)} data points from {start} to {end}")

    # Output to file or stdout
    if config.output_file:
        output_path = Path(config.output_file)
        with open(output_path, "w") as f:
            for d in data:
                f.write(d.to_json(indent=None) + "\n")
        logger.info(f"Data written to {output_path}")
    else:
        for d in data:
            print(d.to_json())


def generate_sample_config(output_path: Path) -> None:
    """Generate a sample configuration file."""
    DEFAULT_CONFIG.to_file(output_path)
    logger.info(f"Sample config written to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="DER Data Simulator - Generate realistic distributed energy resource data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--once",
        action="store_true",
        help="Generate a single data point and exit",
    )
    mode_group.add_argument(
        "--continuous",
        action="store_true",
        help="Run continuously at specified interval",
    )
    mode_group.add_argument(
        "--scheduled",
        action="store_true",
        help="Run on schedule aligned to clock intervals",
    )
    mode_group.add_argument(
        "--historical",
        action="store_true",
        help="Generate historical data for a time range",
    )
    mode_group.add_argument(
        "--generate-config",
        action="store_true",
        help="Generate a sample configuration file",
    )

    # Configuration options
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration JSON file",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=300,
        help="Interval in seconds between data points (default: 300)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        help="Duration in seconds for continuous mode (default: run forever)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path (JSON lines format)",
    )
    parser.add_argument(
        "--device-id",
        type=str,
        default="edge-gateway-001",
        help="Device ID for the edge gateway",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )

    # Historical mode options
    parser.add_argument(
        "--start",
        type=str,
        help="Start date for historical data (YYYY-MM-DD or YYYY-MM-DD HH:MM)",
    )
    parser.add_argument(
        "--end",
        type=str,
        help="End date for historical data (YYYY-MM-DD or YYYY-MM-DD HH:MM)",
    )

    # Verbosity
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress log output",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Configure logging level
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    elif args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load or create configuration
    if args.config and args.config.exists():
        config = GeneratorConfig.from_file(args.config)
        logger.info(f"Loaded config from {args.config}")
    else:
        config = GeneratorConfig()

    # Apply command line overrides
    if args.interval:
        config.interval_seconds = args.interval
    if args.output:
        config.output_file = str(args.output)
    if args.device_id:
        config.device_id = args.device_id
    if args.seed is not None:
        config.seed = args.seed

    # Execute selected mode
    if args.generate_config:
        output_path = args.config or Path("config.json")
        generate_sample_config(output_path)

    elif args.once:
        run_once(config)

    elif args.continuous:
        run_continuous(config, args.duration)

    elif args.scheduled:
        run_scheduled(config)

    elif args.historical:
        if not args.start or not args.end:
            parser.error("--historical requires --start and --end dates")

        # Parse dates
        try:
            start = datetime.fromisoformat(args.start)
        except ValueError:
            start = datetime.strptime(args.start, "%Y-%m-%d")

        try:
            end = datetime.fromisoformat(args.end)
        except ValueError:
            end = datetime.strptime(args.end, "%Y-%m-%d") + timedelta(days=1)

        generate_historical(config, start, end, int(args.interval / 60))


if __name__ == "__main__":
    main()
