"""Configuration for deep learning forecasting models.

This module provides configuration management for LSTM and Transformer-based
forecasting models, extending the baseline forecasting configuration.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

from cloud.forecasting.config import ForecastingConfig


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

    # Target MAE improvement over baseline (5% as per requirements)
    target_mae_percent: float = 5.0

    # Feature scaling
    scale_features: bool = True

    # Model types available
    MODEL_TYPES: ClassVar[list[str]] = ["lstm", "transformer", "hybrid"]

    def __post_init__(self) -> None:
        """Apply environment variable overrides."""
        # First apply parent class overrides
        super().__post_init__()

        # Apply deep learning specific env var overrides
        if self.model_type == "lstm":
            env_model_type = os.environ.get("DL_MODEL_TYPE")
            if env_model_type and env_model_type in self.MODEL_TYPES:
                self.model_type = env_model_type  # type: ignore

        if self.sequence_length == 288:
            env_seq_len = os.environ.get("DL_SEQUENCE_LENGTH")
            if env_seq_len:
                try:
                    self.sequence_length = int(env_seq_len)
                except ValueError as e:
                    raise ValueError(
                        f"DL_SEQUENCE_LENGTH must be an integer, got: {env_seq_len!r}"
                    ) from e

        if self.epochs == 100:
            env_epochs = os.environ.get("DL_EPOCHS")
            if env_epochs:
                try:
                    self.epochs = int(env_epochs)
                except ValueError as e:
                    raise ValueError(
                        f"DL_EPOCHS must be an integer, got: {env_epochs!r}"
                    ) from e

        if self.batch_size == 32:
            env_batch_size = os.environ.get("DL_BATCH_SIZE")
            if env_batch_size:
                try:
                    self.batch_size = int(env_batch_size)
                except ValueError as e:
                    raise ValueError(
                        f"DL_BATCH_SIZE must be an integer, got: {env_batch_size!r}"
                    ) from e

        if self.dl_learning_rate == 0.001:
            env_lr = os.environ.get("DL_LEARNING_RATE")
            if env_lr:
                try:
                    self.dl_learning_rate = float(env_lr)
                except ValueError as e:
                    raise ValueError(
                        f"DL_LEARNING_RATE must be a float, got: {env_lr!r}"
                    ) from e

        if self.hidden_units == 64:
            env_hidden = os.environ.get("DL_HIDDEN_UNITS")
            if env_hidden:
                try:
                    self.hidden_units = int(env_hidden)
                except ValueError as e:
                    raise ValueError(
                        f"DL_HIDDEN_UNITS must be an integer, got: {env_hidden!r}"
                    ) from e

        if self.num_layers == 2:
            env_layers = os.environ.get("DL_NUM_LAYERS")
            if env_layers:
                try:
                    self.num_layers = int(env_layers)
                except ValueError as e:
                    raise ValueError(
                        f"DL_NUM_LAYERS must be an integer, got: {env_layers!r}"
                    ) from e

        if self.dropout == 0.2:
            env_dropout = os.environ.get("DL_DROPOUT")
            if env_dropout:
                try:
                    self.dropout = float(env_dropout)
                except ValueError as e:
                    raise ValueError(
                        f"DL_DROPOUT must be a float, got: {env_dropout!r}"
                    ) from e

        if self.early_stopping_patience == 10:
            env_patience = os.environ.get("DL_EARLY_STOPPING_PATIENCE")
            if env_patience:
                try:
                    self.early_stopping_patience = int(env_patience)
                except ValueError as e:
                    raise ValueError(
                        f"DL_EARLY_STOPPING_PATIENCE must be an integer, got: {env_patience!r}"
                    ) from e

        if self.validation_split == 0.15:
            env_val_split = os.environ.get("DL_VALIDATION_SPLIT")
            if env_val_split:
                try:
                    self.validation_split = float(env_val_split)
                except ValueError as e:
                    raise ValueError(
                        f"DL_VALIDATION_SPLIT must be a float, got: {env_val_split!r}"
                    ) from e

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
