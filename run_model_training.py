#!/usr/bin/env python3
"""CLI runner for baseline forecasting model training.

This script provides a command-line interface for training and evaluating
the baseline forecasting models for load and solar generation prediction.

Usage:
    # Train models with default settings
    python run_model_training.py --train

    # Train with custom model directory
    python run_model_training.py --train --model-dir ./trained_models

    # Evaluate existing models
    python run_model_training.py --evaluate

    # Show model status
    python run_model_training.py --status

Environment variables:
    SUPABASE_URL: Supabase project URL
    SUPABASE_KEY: Supabase API key
    FORECAST_MODEL_DIR: Directory to save trained models
    FORECAST_TEST_SIZE: Fraction of data for testing (0.0-1.0)
"""

import argparse
import json
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

from cloud.forecasting import BaselineForecaster, ForecastingConfig  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ModelTrainingRunner:
    """Runner for model training operations.

    Provides methods for training, evaluating, and managing
    forecasting models through the CLI.
    """

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        model_dir: str = "models",
        test_size: float = 0.2,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
    ) -> None:
        """Initialize the model training runner.

        Args:
            supabase_url: Supabase project URL (or from env)
            supabase_key: Supabase API key (or from env)
            model_dir: Directory to save trained models
            test_size: Fraction of data for testing
            n_estimators: Number of trees in the ensemble
            max_depth: Maximum depth of each tree
            learning_rate: Learning rate for gradient boosting
        """
        self.config = ForecastingConfig(
            supabase_url=supabase_url or "",
            supabase_key=supabase_key or "",
            enabled=True,
            model_dir=model_dir,
            test_size=test_size,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
        )
        self.forecaster = BaselineForecaster(self.config)
        self._shutdown_requested = False

    def train(self) -> dict[str, Any]:
        """Train both load and solar forecasting models.

        Returns:
            Dictionary with training results
        """
        logger.info("Starting model training...")

        result = self.forecaster.train_all()

        if result["status"] == "success":
            # Save trained models
            saved = self.forecaster.save_models()
            result["saved_models"] = saved

            # Log summary
            load_metrics = result["load"]["metrics"]
            solar_metrics = result["solar"]["metrics"]

            logger.info("=" * 60)
            logger.info("MODEL TRAINING SUMMARY")
            logger.info("=" * 60)
            logger.info("Load Forecasting Model:")
            logger.info("  - MAE: %.4f kW (%.2f%%)", load_metrics["mae"], load_metrics["mae_percent"])
            logger.info("  - RMSE: %.4f kW", load_metrics["rmse"])
            logger.info("  - R2: %.4f", load_metrics["r2"])
            logger.info("  - Target (MAE <= %.1f%%): %s",
                       self.config.target_mae_percent,
                       "PASS" if load_metrics["meets_target"] else "FAIL")
            logger.info("")
            logger.info("Solar Forecasting Model:")
            logger.info("  - MAE: %.4f kW (%.2f%%)", solar_metrics["mae"], solar_metrics["mae_percent"])
            logger.info("  - RMSE: %.4f kW", solar_metrics["rmse"])
            logger.info("  - R2: %.4f", solar_metrics["r2"])
            logger.info("  - Target (MAE <= %.1f%%): %s",
                       self.config.target_mae_percent,
                       "PASS" if solar_metrics["meets_target"] else "FAIL")
            logger.info("")
            logger.info("Models saved to: %s", self.config.model_dir)
            logger.info("=" * 60)

        return result

    def evaluate(self) -> dict[str, Any]:
        """Load and evaluate existing models.

        Returns:
            Dictionary with model metrics
        """
        logger.info("Loading existing models...")

        loaded = self.forecaster.load_models()

        if not any(loaded.values()):
            logger.warning("No trained models found in %s", self.config.model_dir)
            return {
                "status": "no_models",
                "models_loaded": loaded,
            }

        metrics = self.forecaster.get_metrics()

        logger.info("=" * 60)
        logger.info("MODEL EVALUATION")
        logger.info("=" * 60)

        if loaded["load"] and metrics["load"]:
            m = metrics["load"]
            logger.info("Load Forecasting Model:")
            logger.info("  - MAE: %.4f kW (%.2f%%)", m["mae"], m["mae_percent"])
            logger.info("  - RMSE: %.4f kW", m["rmse"])
            logger.info("  - R2: %.4f", m["r2"])
            logger.info("  - Target met: %s", m["meets_target"])
        else:
            logger.info("Load model: Not available")

        logger.info("")

        if loaded["solar"] and metrics["solar"]:
            m = metrics["solar"]
            logger.info("Solar Forecasting Model:")
            logger.info("  - MAE: %.4f kW (%.2f%%)", m["mae"], m["mae_percent"])
            logger.info("  - RMSE: %.4f kW", m["rmse"])
            logger.info("  - R2: %.4f", m["r2"])
            logger.info("  - Target met: %s", m["meets_target"])
        else:
            logger.info("Solar model: Not available")

        logger.info("=" * 60)

        return {
            "status": "success",
            "models_loaded": loaded,
            "metrics": metrics,
        }

    def get_status(self) -> dict[str, Any]:
        """Get current status of models and configuration.

        Returns:
            Dictionary with status information
        """
        # Check for existing models
        load_path = self.config.get_model_path("load")
        solar_path = self.config.get_model_path("solar")

        status = {
            "connected": self.forecaster.is_connected(),
            "config": {
                "model_dir": self.config.model_dir,
                "test_size": self.config.test_size,
                "horizon_hours": self.config.horizon_hours,
                "target_mae_percent": self.config.target_mae_percent,
                "n_estimators": self.config.n_estimators,
                "max_depth": self.config.max_depth,
                "learning_rate": self.config.learning_rate,
            },
            "models": {
                "load": {
                    "path": str(load_path),
                    "exists": load_path.exists(),
                },
                "solar": {
                    "path": str(solar_path),
                    "exists": solar_path.exists(),
                },
            },
        }

        return status

    def close(self) -> None:
        """Close connections and clean up resources."""
        self.forecaster.close()


def setup_signal_handlers(runner: ModelTrainingRunner) -> None:
    """Setup signal handlers for graceful shutdown."""

    def handle_signal(signum, frame):
        logger.info("Received shutdown signal")
        runner._shutdown_requested = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)


def main() -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate baseline forecasting models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train models
    python run_model_training.py --train

    # Train with custom settings
    python run_model_training.py --train --model-dir ./my_models --n-estimators 200

    # Evaluate existing models
    python run_model_training.py --evaluate

    # Check status
    python run_model_training.py --status
        """,
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--train",
        action="store_true",
        help="Train new models",
    )
    mode_group.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate existing models",
    )
    mode_group.add_argument(
        "--status",
        action="store_true",
        help="Show model and configuration status",
    )

    # Connection settings
    parser.add_argument(
        "--supabase-url",
        help="Supabase project URL (or set SUPABASE_URL env var)",
    )
    parser.add_argument(
        "--supabase-key",
        help="Supabase API key (or set SUPABASE_KEY env var)",
    )

    # Model settings
    parser.add_argument(
        "--model-dir",
        default="models",
        help="Directory to save/load models (default: models)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for testing (default: 0.2)",
    )

    # Hyperparameters
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees in the ensemble (default: 100)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum depth of each tree (default: 6)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate for gradient boosting (default: 0.1)",
    )

    # Output settings
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Create runner
        runner = ModelTrainingRunner(
            supabase_url=args.supabase_url,
            supabase_key=args.supabase_key,
            model_dir=args.model_dir,
            test_size=args.test_size,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
        )

        # Setup signal handlers
        setup_signal_handlers(runner)

        # Execute requested operation
        if args.train:
            result = runner.train()
        elif args.evaluate:
            result = runner.evaluate()
        elif args.status:
            result = runner.get_status()
        else:
            parser.print_help()
            return 1

        # Output results
        if args.json:
            # Convert any non-serializable values
            def serialize(obj):
                if isinstance(obj, Path):
                    return str(obj)
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return obj

            print(json.dumps(result, indent=2, default=serialize))

        # Return appropriate exit code
        if result.get("status") == "success":
            return 0
        elif result.get("status") in ("disabled", "no_data", "no_models"):
            return 0  # Not an error, just nothing to do
        else:
            return 1

    except ValueError as e:
        logger.error("Configuration error: %s", e)
        return 1
    except ImportError as e:
        logger.error("Missing dependency: %s", e)
        return 1
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        return 1
    finally:
        if "runner" in locals():
            runner.close()


if __name__ == "__main__":
    sys.exit(main())
