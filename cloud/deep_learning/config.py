"""Configuration for deep learning forecasting models.

This module provides configuration management for LSTM and Transformer-based
forecasting models, extending the baseline forecasting configuration.
"""

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, ClassVar, Literal

from cloud.forecasting.config import ForecastingConfig


def _get_dataclass_defaults(cls: type) -> dict[str, Any]:
    """Get default values for all fields in a dataclass hierarchy."""
    defaults: dict[str, Any] = {}
    for field in fields(cls):
        if field.default is not field.default_factory:
            if field.default is not dataclass:
                defaults[field.name] = field.default
        elif field.default_factory is not dataclass:
            defaults[field.name] = field.default_factory()
    return defaults


@dataclass
class DeepLearningConfig(ForecastingConfig):
    """Configuration for deep learning forecasting models.

    Extends ForecastingConfig with deep learning-specific parameters.

    Environment variable overrides:
    - DL_MODEL_TYPE: Model architecture (lstm, transformer, hybrid)
    - DL_SEQUENCE_LENGTH: Input sequence length in samples
    - DL_EPOCHS: Number of training epochs
    - DL_BATCH_SIZE: Training batch size
    - DL_LEARNING_RATE: Learning rate for optimizer
    - DL_HIDDEN_UNITS: Hidden layer size
    - DL_NUM_LAYERS: Number of layers
    - DL_DROPOUT: Dropout rate
    - DL_EARLY_STOPPING_PATIENCE: Early stopping patience
    - DL_VALIDATION_SPLIT: Validation data fraction

    Attributes:
        model_type: Type of model architecture (lstm, transformer, hybrid)
        sequence_length: Number of time steps for input sequences
        epochs: Number of training epochs
        batch_size: Training batch size
        dl_learning_rate: Learning rate for deep learning optimizer
        hidden_units: Number of hidden units in layers
        num_layers: Number of LSTM/Transformer layers
        dropout: Dropout rate for regularization
        validation_split: Fraction of training data for validation
        early_stopping_patience: Epochs to wait before early stopping
        use_attention: Whether to use attention mechanism in LSTM
        num_heads: Number of attention heads for Transformer
        ff_dim: Feed-forward dimension for Transformer
    """

    # Model architecture
    model_type: Literal["lstm", "transformer", "hybrid"] = "lstm"
    sequence_length: int = 288  # 24 hours at 5-minute intervals

    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    dl_learning_rate: float = 0.001
    validation_split: float = 0.15
    early_stopping_patience: int = 10

    # Architecture parameters
    hidden_units: int = 64
    num_layers: int = 2
    dropout: float = 0.2

    # Transformer-specific parameters
    num_heads: int = 4
    ff_dim: int = 128

    # LSTM-specific parameters
    use_attention: bool = True
    bidirectional: bool = False

    # Target MAE as percentage of target mean (model meets target if MAE% <= this value)
    target_mae_percent: float = 5.0

    # Feature scaling
    scale_features: bool = True

    # Model types available
    MODEL_TYPES: ClassVar[list[str]] = ["lstm", "transformer", "hybrid"]

    # Class-level cache for dataclass defaults
    _defaults: ClassVar[dict[str, Any] | None] = None

    def _get_defaults(self) -> dict[str, Any]:
        """Get cached dataclass defaults."""
        if DeepLearningConfig._defaults is None:
            DeepLearningConfig._defaults = _get_dataclass_defaults(DeepLearningConfig)
        return DeepLearningConfig._defaults

    def _apply_env_override(
        self,
        field_name: str,
        env_name: str,
        type_converter: type[int] | type[float],
    ) -> None:
        """Apply environment variable override if field is still at default.

        Args:
            field_name: Name of the field to override
            env_name: Environment variable name
            type_converter: Type to convert the value to (int or float)
        """
        defaults = self._get_defaults()
        if field_name not in defaults:
            return
        if getattr(self, field_name) != defaults[field_name]:
            return
        raw = os.environ.get(env_name)
        if not raw:
            return
        try:
            setattr(self, field_name, type_converter(raw))
        except ValueError as e:
            type_name = "an integer" if type_converter is int else "a float"
            raise ValueError(f"{env_name} must be {type_name}, got: {raw!r}") from e

    def __post_init__(self) -> None:
        """Apply environment variable overrides."""
        # First apply parent class overrides
        super().__post_init__()

        # Apply deep learning specific env var overrides
        defaults = self._get_defaults()

        # Handle model_type specially due to validation requirement
        if self.model_type == defaults.get("model_type"):
            env_model_type = os.environ.get("DL_MODEL_TYPE")
            if env_model_type and env_model_type in self.MODEL_TYPES:
                self.model_type = env_model_type  # type: ignore

        # Apply numeric overrides using helper
        self._apply_env_override("sequence_length", "DL_SEQUENCE_LENGTH", int)
        self._apply_env_override("epochs", "DL_EPOCHS", int)
        self._apply_env_override("batch_size", "DL_BATCH_SIZE", int)
        self._apply_env_override("dl_learning_rate", "DL_LEARNING_RATE", float)
        self._apply_env_override("hidden_units", "DL_HIDDEN_UNITS", int)
        self._apply_env_override("num_layers", "DL_NUM_LAYERS", int)
        self._apply_env_override("dropout", "DL_DROPOUT", float)
        self._apply_env_override("early_stopping_patience", "DL_EARLY_STOPPING_PATIENCE", int)
        self._apply_env_override("validation_split", "DL_VALIDATION_SPLIT", float)

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If required configuration is missing or invalid
        """
        # Validate parent configuration
        super().validate()

        # Validate deep learning specific configuration
        if self.model_type not in self.MODEL_TYPES:
            raise ValueError(
                f"model_type must be one of {self.MODEL_TYPES}, got: {self.model_type}"
            )

        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")

        if self.epochs <= 0:
            raise ValueError("epochs must be positive")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.dl_learning_rate <= 0:
            raise ValueError("dl_learning_rate must be positive")

        if not 0.0 < self.validation_split < 1.0:
            raise ValueError(
                "validation_split must be between 0.0 and 1.0 (exclusive)"
            )

        if self.hidden_units <= 0:
            raise ValueError("hidden_units must be positive")

        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be between 0.0 and 1.0")

        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive")

        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")

        if self.ff_dim <= 0:
            raise ValueError("ff_dim must be positive")

    def get_model_path(self, target: str) -> Path:
        """Get the path for a deep learning model file.

        Args:
            target: Target variable name (e.g., 'load' or 'solar')

        Returns:
            Path to the model file
        """
        return Path(self.model_dir) / f"dl_{self.model_type}_{target}_forecaster.joblib"

    def get_keras_model_path(self, target: str) -> Path:
        """Get the path for a Keras model file.

        Args:
            target: Target variable name (e.g., 'load' or 'solar')

        Returns:
            Path to the Keras model file
        """
        return Path(self.model_dir) / f"dl_{self.model_type}_{target}_forecaster.keras"
