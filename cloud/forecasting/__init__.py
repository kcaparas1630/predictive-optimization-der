"""Forecasting module for DER load and solar generation prediction.

This module provides baseline forecasting models for predicting:
- Home load (kW) 24 hours ahead
- Solar generation (kW) 24 hours ahead

Example:
    >>> from cloud.forecasting import ForecastingConfig, BaselineForecaster
    >>> config = ForecastingConfig()
    >>> forecaster = BaselineForecaster(config)
    >>> forecaster.train(training_data)
    >>> predictions = forecaster.predict(features)
"""

from cloud.forecasting.config import ForecastingConfig
from cloud.forecasting.model import BaselineForecaster

__all__ = ["ForecastingConfig", "BaselineForecaster"]
