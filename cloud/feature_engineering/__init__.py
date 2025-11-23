"""Feature engineering module for AI/ML training data preparation.

This module provides functionality to:
1. Query raw DER data from Supabase
2. Engineer temporal features (hour_of_day, day_of_week, etc.)
3. Calculate rolling window features (7-day moving averages)
4. Apply categorical encoding (one-hot for weather/time-of-use categories)
5. Store engineered features in a training-ready table

Example:
    >>> from cloud.feature_engineering import FeatureEngineeringPipeline, FeatureEngineeringConfig
    >>> config = FeatureEngineeringConfig(enabled=True)
    >>> pipeline = FeatureEngineeringPipeline(config)
    >>> result = pipeline.run()
    >>> print(f"Processed {result['records_processed']} records")
"""

from cloud.feature_engineering.config import FeatureEngineeringConfig
from cloud.feature_engineering.pipeline import FeatureEngineeringPipeline

__all__ = ["FeatureEngineeringConfig", "FeatureEngineeringPipeline"]
