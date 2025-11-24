"""Deep Learning forecasting models for DER prediction.

This module provides LSTM and Transformer-based models for predicting
load (kW) and solar generation (kW) 24 hours ahead, aiming to improve
upon the baseline Gradient Boosting model.
"""

from cloud.deep_learning.config import DeepLearningConfig
from cloud.deep_learning.model import DeepLearningForecaster

__all__ = ["DeepLearningConfig", "DeepLearningForecaster"]
