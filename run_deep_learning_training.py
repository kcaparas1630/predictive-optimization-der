#!/usr/bin/env python3
"""CLI runner for deep learning forecasting model training.

This script provides a command-line interface for training and evaluating
deep learning models (LSTM, Transformer, Hybrid) for load and solar generation
prediction, with comparison against the baseline Gradient Boosting model.

Usage:
    # Train LSTM model with default settings
    python run_deep_learning_training.py --train

    # Train Transformer model
    python run_deep_learning_training.py --train --model-type transformer

    # Train with custom hyperparameters
    python run_deep_learning_training.py --train --epochs 200 --batch-size 64

    # Compare with baseline model
    python run_deep_learning_training.py --compare

    # Evaluate existing deep learning models
    python run_deep_learning_training.py --evaluate

    # Show model status
    python run_deep_learning_training.py --status

Environment variables:
    SUPABASE_URL: Supabase project URL
    SUPABASE_KEY: Supabase API key
    DL_MODEL_TYPE: Model type (lstm, transformer, hybrid)
    DL_EPOCHS: Number of training epochs
    DL_BATCH_SIZE: Training batch size
    DL_LEARNING_RATE: Learning rate
"""

import argparse
import json
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path
from types import FrameType
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

from cloud.deep_learning import DeepLearningConfig, DeepLearningForecaster  # noqa: E402
from cloud.deep_learning.model import compare_with_baseline  # noqa: E402
from cloud.forecasting import BaselineForecaster, ForecastingConfig  # noqa: E402
from cloud.forecasting.model import ModelMetrics  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class DeepLearningTrainingRunner:
    """Runner for deep learning model training operations.

    Provides methods for training, evaluating, and comparing
    deep learning forecasting models through the CLI.
    """

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        model_dir: str = "models",
        test_size: float = 0.2,
        model_type: str = "lstm",
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        hidden_units: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        sequence_length: int = 288,
        early_stopping_patience: int = 10,
    ) -> None:
        """Initialize the deep learning training runner.

        Args:
            supabase_url: Supabase project URL (or from env)
            supabase_key: Supabase API key (or from env)
            model_dir: Directory to save trained models
            test_size: Fraction of data for testing
            model_type: Model architecture (lstm, transformer, hybrid)
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
            hidden_units: Number of hidden units
            num_layers: Number of layers
            dropout: Dropout rate
            sequence_length: Input sequence length
            early_stopping_patience: Epochs for early stopping
        """
        self.config = DeepLearningConfig(
            supabase_url=supabase_url or "",
            supabase_key=supabase_key or "",
            enabled=True,
            model_dir=model_dir,
            test_size=test_size,
            model_type=model_type,  # type: ignore
            epochs=epochs,
            batch_size=batch_size,
            dl_learning_rate=learning_rate,
            hidden_units=hidden_units,
            num_layers=num_layers,
            dropout=dropout,
            sequence_length=sequence_length,
            early_stopping_patience=early_stopping_patience,
        )
        self.forecaster = DeepLearningForecaster(self.config)
        self._shutdown_requested = False

    def train(self) -> dict[str, Any]:
        """Train both load and solar forecasting models.

        Returns:
            Dictionary with training results
        """
        logger.info("Starting deep learning model training...")
        logger.info("Model type: %s", self.config.model_type)
        logger.info("Sequence length: %d", self.config.sequence_length)
        logger.info("Epochs: %d", self.config.epochs)
        logger.info("Batch size: %d", self.config.batch_size)

        result = self.forecaster.train_all()

        if result["status"] == "success":
            # Save trained models
            saved = self.forecaster.save_models()
            result["saved_models"] = saved

            # Log summary
            load_metrics = result["load"]["metrics"]
            solar_metrics = result["solar"]["metrics"]

            logger.info("=" * 60)
            logger.info("DEEP LEARNING MODEL TRAINING SUMMARY")
            logger.info("=" * 60)
            logger.info("Model Type: %s", self.config.model_type.upper())
            logger.info("Sequence Length: %d samples", self.config.sequence_length)
            logger.info("")
            logger.info("Load Forecasting Model:")
            logger.info("  - MAE: %.4f kW (%.2f%%)", load_metrics["mae"], load_metrics["mae_percent"])
            logger.info("  - RMSE: %.4f kW", load_metrics["rmse"])
            logger.info("  - R2: %.4f", load_metrics["r2"])
            logger.info("  - Target (MAE <= %.1f%%): %s",
                       self.config.target_mae_percent,
                       "PASS" if load_metrics["meets_target"] else "FAIL")
            logger.info("  - Epochs trained: %d", result["load"]["training_history"]["total_epochs"])
            logger.info("  - Parameters: %d", result["load"]["parameters"])
            logger.info("")
            logger.info("Solar Forecasting Model:")
            logger.info("  - MAE: %.4f kW (%.2f%%)", solar_metrics["mae"], solar_metrics["mae_percent"])
            logger.info("  - RMSE: %.4f kW", solar_metrics["rmse"])
            logger.info("  - R2: %.4f", solar_metrics["r2"])
            logger.info("  - Target (MAE <= %.1f%%): %s",
                       self.config.target_mae_percent,
                       "PASS" if solar_metrics["meets_target"] else "FAIL")
            logger.info("  - Epochs trained: %d", result["solar"]["training_history"]["total_epochs"])
            logger.info("  - Parameters: %d", result["solar"]["parameters"])
            logger.info("")
            logger.info("Models saved to: %s", self.config.model_dir)
            logger.info("=" * 60)

        return result

    def evaluate(self) -> dict[str, Any]:
        """Load and evaluate existing deep learning models.

        Returns:
            Dictionary with model metrics
        """
        logger.info("Loading existing deep learning models...")

        loaded = self.forecaster.load_models()

        if not any(loaded.values()):
            logger.warning("No trained deep learning models found in %s", self.config.model_dir)
            return {
                "status": "no_models",
                "models_loaded": loaded,
            }

        metrics = self.forecaster.get_metrics()
        history = self.forecaster.get_training_history()

        logger.info("=" * 60)
        logger.info("DEEP LEARNING MODEL EVALUATION")
        logger.info("=" * 60)

        if loaded["load"] and metrics["load"]:
            m = metrics["load"]
            logger.info("Load Forecasting Model:")
            logger.info("  - MAE: %.4f kW (%.2f%%)", m["mae"], m["mae_percent"])
            logger.info("  - RMSE: %.4f kW", m["rmse"])
            logger.info("  - R2: %.4f", m["r2"])
            logger.info("  - Target met (<=5%%): %s", m["meets_target"])
        else:
            logger.info("Load model: Not available")

        logger.info("")

        if loaded["solar"] and metrics["solar"]:
            m = metrics["solar"]
            logger.info("Solar Forecasting Model:")
            logger.info("  - MAE: %.4f kW (%.2f%%)", m["mae"], m["mae_percent"])
            logger.info("  - RMSE: %.4f kW", m["rmse"])
            logger.info("  - R2: %.4f", m["r2"])
            logger.info("  - Target met (<=5%%): %s", m["meets_target"])
        else:
            logger.info("Solar model: Not available")

        logger.info("=" * 60)

        return {
            "status": "success",
            "models_loaded": loaded,
            "metrics": metrics,
            "training_history": history,
        }

    def compare_with_baseline(self) -> dict[str, Any]:
        """Compare deep learning models with baseline Gradient Boosting.

        Returns:
            Dictionary with comparison results
        """
        logger.info("Comparing deep learning models with baseline...")

        # Load deep learning models
        dl_loaded = self.forecaster.load_models()
        dl_metrics = self.forecaster.get_metrics()

        if not any(dl_loaded.values()):
            logger.warning("No deep learning models found")
            return {
                "status": "no_dl_models",
                "comparison": None,
            }

        # Load baseline models
        baseline_config = ForecastingConfig(
            supabase_url=self.config.supabase_url,
            supabase_key=self.config.supabase_key,
            enabled=True,
            model_dir=self.config.model_dir,
        )
        baseline_forecaster = BaselineForecaster(baseline_config)
        baseline_loaded = baseline_forecaster.load_models()
        baseline_metrics = baseline_forecaster.get_metrics()

        if not any(baseline_loaded.values()):
            logger.warning("No baseline models found")
            return {
                "status": "no_baseline_models",
                "comparison": None,
            }

        comparison = {}

        # Compare load models
        if dl_loaded["load"] and baseline_loaded["load"] and dl_metrics["load"] and baseline_metrics["load"]:
            dl_load = ModelMetrics(**dl_metrics["load"])
            baseline_load = ModelMetrics(**baseline_metrics["load"])
            comparison["load"] = compare_with_baseline(dl_load, baseline_load)

        # Compare solar models
        if dl_loaded["solar"] and baseline_loaded["solar"] and dl_metrics["solar"] and baseline_metrics["solar"]:
            dl_solar = ModelMetrics(**dl_metrics["solar"])
            baseline_solar = ModelMetrics(**baseline_metrics["solar"])
            comparison["solar"] = compare_with_baseline(dl_solar, baseline_solar)

        # Log comparison
        logger.info("=" * 60)
        logger.info("MODEL COMPARISON: Deep Learning vs Baseline")
        logger.info("=" * 60)

        if "load" in comparison:
            c = comparison["load"]
            logger.info("LOAD FORECASTING:")
            logger.info("  Baseline (Gradient Boosting):")
            logger.info("    - MAE: %.4f kW (%.2f%%)", c["baseline"]["mae"], c["baseline"]["mae_percent"])
            logger.info("    - R2: %.4f", c["baseline"]["r2"])
            logger.info("  Deep Learning (%s):", self.config.model_type.upper())
            logger.info("    - MAE: %.4f kW (%.2f%%)", c["deep_learning"]["mae"], c["deep_learning"]["mae_percent"])
            logger.info("    - R2: %.4f", c["deep_learning"]["r2"])
            logger.info("  Improvement:")
            logger.info("    - MAE: %.2f%%", c["improvements"]["mae_percent_improvement"])
            logger.info("    - R2: %.4f", c["improvements"]["r2_absolute_improvement"])
            logger.info("    - DL is better: %s", c["dl_is_better"])
            logger.info("    - Meets 5%% target: %s", c["meets_5_percent_target"])
            logger.info("")

        if "solar" in comparison:
            c = comparison["solar"]
            logger.info("SOLAR FORECASTING:")
            logger.info("  Baseline (Gradient Boosting):")
            logger.info("    - MAE: %.4f kW (%.2f%%)", c["baseline"]["mae"], c["baseline"]["mae_percent"])
            logger.info("    - R2: %.4f", c["baseline"]["r2"])
            logger.info("  Deep Learning (%s):", self.config.model_type.upper())
            logger.info("    - MAE: %.4f kW (%.2f%%)", c["deep_learning"]["mae"], c["deep_learning"]["mae_percent"])
            logger.info("    - R2: %.4f", c["deep_learning"]["r2"])
            logger.info("  Improvement:")
            logger.info("    - MAE: %.2f%%", c["improvements"]["mae_percent_improvement"])
            logger.info("    - R2: %.4f", c["improvements"]["r2_absolute_improvement"])
            logger.info("    - DL is better: %s", c["dl_is_better"])
            logger.info("    - Meets 5%% target: %s", c["meets_5_percent_target"])

        logger.info("=" * 60)

        baseline_forecaster.close()

        return {
            "status": "success",
            "comparison": comparison,
        }

    def get_status(self) -> dict[str, Any]:
        """Get current status of deep learning models and configuration.

        Returns:
            Dictionary with status information
        """
        # Check for existing models
        load_keras_path = self.config.get_keras_model_path("load")
        solar_keras_path = self.config.get_keras_model_path("solar")
        load_meta_path = self.config.get_model_path("load")
        solar_meta_path = self.config.get_model_path("solar")

        status = {
            "status": "success",
            "connected": self.forecaster.is_connected(),
            "config": {
                "model_dir": self.config.model_dir,
                "model_type": self.config.model_type,
                "sequence_length": self.config.sequence_length,
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.dl_learning_rate,
                "hidden_units": self.config.hidden_units,
                "num_layers": self.config.num_layers,
                "dropout": self.config.dropout,
                "test_size": self.config.test_size,
                "target_mae_percent": self.config.target_mae_percent,
            },
            "models": {
                "load": {
                    "keras_path": str(load_keras_path),
                    "metadata_path": str(load_meta_path),
                    "exists": load_keras_path.exists() and load_meta_path.exists(),
                },
                "solar": {
                    "keras_path": str(solar_keras_path),
                    "metadata_path": str(solar_meta_path),
                    "exists": solar_keras_path.exists() and solar_meta_path.exists(),
                },
            },
        }

        return status

    def close(self) -> None:
        """Close connections and clean up resources."""
        self.forecaster.close()


def setup_signal_handlers(runner: DeepLearningTrainingRunner) -> None:
    """Setup signal handlers for graceful shutdown."""

    def handle_signal(_signum: int, _frame: Optional[FrameType]) -> None:
        logger.info("Received shutdown signal; requesting shutdown")
        runner._shutdown_requested = True
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)


def main() -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate deep learning forecasting models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train LSTM model
    python run_deep_learning_training.py --train

    # Train Transformer model
    python run_deep_learning_training.py --train --model-type transformer

    # Train with custom settings
    python run_deep_learning_training.py --train --epochs 200 --batch-size 64

    # Compare with baseline
    python run_deep_learning_training.py --compare

    # Evaluate existing models
    python run_deep_learning_training.py --evaluate

    # Check status
    python run_deep_learning_training.py --status
        """,
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--train",
        action="store_true",
        help="Train new deep learning models",
    )
    mode_group.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate existing deep learning models",
    )
    mode_group.add_argument(
        "--compare",
        action="store_true",
        help="Compare deep learning models with baseline",
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
        "--model-type",
        choices=["lstm", "transformer", "hybrid"],
        default="lstm",
        help="Model architecture (default: lstm)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for testing (default: 0.2)",
    )

    # Deep learning hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (default: 32)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--hidden-units",
        type=int,
        default=64,
        help="Number of hidden units (default: 64)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of LSTM/Transformer layers (default: 2)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate (default: 0.2)",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=288,
        help="Input sequence length in samples (default: 288 = 24 hours at 5-min intervals)",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Epochs for early stopping (default: 10)",
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
        runner = DeepLearningTrainingRunner(
            supabase_url=args.supabase_url,
            supabase_key=args.supabase_key,
            model_dir=args.model_dir,
            test_size=args.test_size,
            model_type=args.model_type,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            hidden_units=args.hidden_units,
            num_layers=args.num_layers,
            dropout=args.dropout,
            sequence_length=args.sequence_length,
            early_stopping_patience=args.early_stopping_patience,
        )

        # Setup signal handlers
        setup_signal_handlers(runner)

        # Execute requested operation
        if args.train:
            result = runner.train()
        elif args.evaluate:
            result = runner.evaluate()
        elif args.compare:
            result = runner.compare_with_baseline()
        elif args.status:
            result = runner.get_status()
        else:
            parser.print_help()
            return 1

        # Output results
        if args.json:
            def serialize(obj: Any) -> Any:
                if isinstance(obj, Path):
                    return str(obj)
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return obj

            print(json.dumps(result, indent=2, default=serialize))

        # Return appropriate exit code
        if result.get("status") == "success":
            return 0
        elif result.get("status") in ("disabled", "no_data", "no_models", "no_dl_models", "no_baseline_models"):
            return 0  # Not an error, just nothing to do
        else:
            return 1

    except KeyboardInterrupt:
        logger.info("Interrupted by user; shutting down.")
        return 130
    except ValueError as e:
        logger.error("Configuration error: %s", e)
        return 1
    except ImportError as e:
        logger.error("Missing dependency: %s", e)
        return 1
    except Exception:
        logger.exception("Unexpected error")
        return 1
    finally:
        if "runner" in locals():
            runner.close()


if __name__ == "__main__":
    sys.exit(main())
