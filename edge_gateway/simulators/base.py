"""Base simulator class with common functionality."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any
import random


class BaseSimulator(ABC):
    """Abstract base class for all simulators."""

    def __init__(self, seed: int | None = None):
        """
        Initialize the simulator.

        Args:
            seed: Random seed for reproducibility. If None, results will vary.
        """
        self._random = random.Random(seed)

    def _add_noise(self, value: float, noise_percent: float = 5.0) -> float:
        """
        Add random noise to a value.

        Args:
            value: Base value
            noise_percent: Maximum noise as percentage of value

        Returns:
            Value with noise applied
        """
        noise_factor = self._random.uniform(-noise_percent / 100, noise_percent / 100)
        return value * (1 + noise_factor)

    def _clamp(self, value: float, min_val: float, max_val: float) -> float:
        """Clamp value between min and max."""
        return max(min_val, min(max_val, value))

    @abstractmethod
    def generate(self, timestamp: datetime) -> Any:
        """
        Generate simulated data for the given timestamp.

        Args:
            timestamp: The timestamp for which to generate data

        Returns:
            Simulated data appropriate for this simulator
        """
        pass
