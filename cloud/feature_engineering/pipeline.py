"""Feature engineering pipeline for AI/ML training data preparation.

This module extracts raw DER data from Supabase, engineers features,
and stores the results in a training-ready table.
"""

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, ClassVar, Optional

import pandas as pd

from cloud.feature_engineering.config import FeatureEngineeringConfig

logger = logging.getLogger(__name__)

# Optional Supabase import
try:
    from supabase import Client, create_client

    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    Client = None  # type: ignore
    create_client = None  # type: ignore


class FeatureEngineeringPipeline:
    """Pipeline for engineering features from raw DER data.

    This class handles:
    - Querying raw data from Supabase readings table
    - Extracting temporal features (hour_of_day, day_of_week, etc.)
    - Calculating rolling window features (7-day moving averages)
    - Applying one-hot encoding for categorical features
    - Storing engineered features to training_data table

    Example:
        >>> config = FeatureEngineeringConfig(
        ...     supabase_url="https://your-project.supabase.co",
        ...     supabase_key="your-key",
        ...     enabled=True
        ... )
        >>> pipeline = FeatureEngineeringPipeline(config)
        >>> result = pipeline.run()
        >>> print(f"Processed {result['records_processed']} records")
    """

    # Key metrics for lag features
    LAG_METRICS: ClassVar[list[str]] = [
        "solar.generation_kw",
        "home_load.total_load_kw",
        "grid_price.price_per_kwh",
    ]

    def __init__(self, config: FeatureEngineeringConfig) -> None:
        """Initialize feature engineering pipeline.

        Args:
            config: Feature engineering configuration

        Raises:
            ImportError: If required libraries are not installed
            ValueError: If configuration is invalid
        """
        self.config = config
        self._supabase_client: Optional[Client] = None

        if not config.enabled:
            logger.info("Feature engineering pipeline is disabled")
            return

        self._validate_dependencies()
        config.validate()
        self._connect()

    def _validate_dependencies(self) -> None:
        """Validate required libraries are installed."""
        if not SUPABASE_AVAILABLE:
            raise ImportError(
                "supabase is not installed; install with `pip install supabase`"
            )

    def _connect(self) -> None:
        """Establish connection to Supabase."""
        try:
            self._supabase_client = create_client(
                self.config.supabase_url,
                self.config.supabase_key,
            )
            logger.info(
                "Connected to Supabase at %s for feature engineering",
                self.config.supabase_url,
            )
        except Exception:
            logger.exception("Failed to connect to Supabase")
            raise

    def is_connected(self) -> bool:
        """Check if Supabase connection is established.

        Returns:
            True if connected to Supabase
        """
        if not self.config.enabled:
            return False
        return self._supabase_client is not None

    def health_check(self) -> bool:
        """Perform health check on Supabase connection.

        Returns:
            True if Supabase is accessible
        """
        if not self.is_connected():
            return False

        try:
            self._supabase_client.table(self.config.source_table).select("time").limit(
                1
            ).execute()
            return True
        except Exception as e:
            logger.warning("Supabase health check failed: %s", e)
            return False

    def query_raw_data(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Query raw data from Supabase readings table.

        Args:
            start_time: Start of time range (default: lookback_days ago)
            end_time: End of time range (default: now)

        Returns:
            DataFrame with raw readings data
        """
        if not self.is_connected():
            logger.warning("Not connected, cannot query data")
            return pd.DataFrame()

        # Set default time range
        if end_time is None:
            end_time = datetime.now(timezone.utc)
        if start_time is None:
            start_time = end_time - timedelta(days=self.config.lookback_days)

        logger.info(
            "Querying raw data from %s to %s",
            start_time.isoformat(),
            end_time.isoformat(),
        )

        try:
            # Paginate through results (Supabase/PostgREST has 1000-row default limit)
            all_data = []
            page_size = 1000
            offset = 0

            while True:
                # Build query
                query = self._supabase_client.table(self.config.source_table).select(
                    "*"
                )

                # Add time range filter
                query = query.gte("time", start_time.isoformat())
                query = query.lte("time", end_time.isoformat())

                # Add site filter if specified
                if self.config.site_id:
                    query = query.eq("site_id", self.config.site_id)

                # Order by time for proper processing
                query = query.order("time")

                # Paginate
                query = query.range(offset, offset + page_size - 1)

                # Execute query
                result = query.execute()

                if not result.data:
                    break

                all_data.extend(result.data)
                logger.debug(
                    "Fetched %d records (total: %d)", len(result.data), len(all_data)
                )

                # If we got less than page_size, we're done
                if len(result.data) < page_size:
                    break

                offset += page_size

            if not all_data:
                logger.info("No data found in specified time range")
                return pd.DataFrame()

            df = pd.DataFrame(all_data)
            logger.info("Queried %d raw records", len(df))
            return df

        except Exception as e:
            logger.exception("Failed to query raw data: %s", e)
            raise

    def pivot_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pivot the readings table from long format to wide format.

        The readings table stores data in long format:
            time | site_id | device_id | metric_name | metric_value

        This pivots to wide format for feature engineering:
            time | site_id | device_id | solar.generation_kw | battery.soc_percent | ...

        Args:
            df: DataFrame with raw readings in long format

        Returns:
            DataFrame with metrics as columns
        """
        if df.empty:
            return pd.DataFrame()

        logger.info("Pivoting %d records to wide format", len(df))

        # Ensure time is datetime
        df["time"] = pd.to_datetime(df["time"])

        # Pivot to wide format
        # Group by time, site_id, device_id and spread metric_name into columns
        # Check for duplicates that will be dropped
        dup_check = df.groupby(["time", "site_id", "device_id", "metric_name"]).size()
        if (dup_check > 1).any():
            num_dups = (dup_check > 1).sum()
            logger.warning(
                "Found %d duplicate metric readings (same time/site/device/metric); "
                "using first value",
                num_dups,
            )

        pivot_df = df.pivot_table(
            index=["time", "site_id", "device_id"],
            columns="metric_name",
            values="metric_value",
            aggfunc="first",  # Take first value if duplicates
        ).reset_index()

        # Flatten column names
        pivot_df.columns.name = None

        logger.info(
            "Pivoted to %d rows with %d columns",
            len(pivot_df),
            len(pivot_df.columns),
        )
        return pivot_df

    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features from timestamp.

        Adds the following features:
        - hour_of_day: Hour (0-23)
        - day_of_week: Day of week (0=Monday, 6=Sunday)
        - day_of_month: Day of month (1-31)
        - month: Month (1-12)
        - is_weekend: Boolean (Saturday or Sunday)
        - hour_sin: Sine of hour (cyclical encoding)
        - hour_cos: Cosine of hour (cyclical encoding)
        - day_sin: Sine of day of week (cyclical encoding)
        - day_cos: Cosine of day of week (cyclical encoding)

        Args:
            df: DataFrame with 'time' column

        Returns:
            DataFrame with added temporal features
        """
        if df.empty or "time" not in df.columns:
            return df

        logger.info("Extracting temporal features")

        import numpy as np

        df = df.copy()

        # Ensure time is datetime
        df["time"] = pd.to_datetime(df["time"])

        # Basic temporal features
        df["hour_of_day"] = df["time"].dt.hour
        df["day_of_week"] = df["time"].dt.dayofweek
        df["day_of_month"] = df["time"].dt.day
        df["month"] = df["time"].dt.month
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        # Cyclical encoding for hour (24-hour cycle)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)

        # Cyclical encoding for day of week (7-day cycle)
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        logger.info("Added 9 temporal features")
        return df

    def calculate_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling window features (moving averages).

        For each metric in ROLLING_AVG_METRICS, calculates:
        - {metric}_rolling_avg_{window}d: Rolling mean over window days
        - {metric}_rolling_std_{window}d: Rolling std over window days

        Args:
            df: DataFrame with metric columns

        Returns:
            DataFrame with added rolling features
        """
        if df.empty:
            return df

        logger.info(
            "Calculating %d-day rolling features", self.config.rolling_window_days
        )

        df = df.copy()

        # Sort by time for rolling calculations
        df = df.sort_values("time")

        # Calculate rolling features for each key metric
        window_days = self.config.rolling_window_days

        # Calculate window size based on actual data interval
        expected_interval_minutes = 5
        actual_interval_minutes = expected_interval_minutes

        if len(df) > 1 and "time" in df.columns:
            actual_interval_minutes = (
                df["time"].iloc[1] - df["time"].iloc[0]
            ).total_seconds() / 60

            if abs(actual_interval_minutes - expected_interval_minutes) > 0.1:
                logger.warning(
                    "Data interval (%.1fmin) differs from expected (%dmin). "
                    "Adjusting rolling window to use actual interval.",
                    actual_interval_minutes,
                    expected_interval_minutes,
                )

        # Calculate samples per day based on actual interval
        if actual_interval_minutes > 0:
            samples_per_day = round((24 * 60) / actual_interval_minutes)
        else:
            samples_per_day = 288  # fallback to 5-min intervals

        window_size = max(1, samples_per_day * window_days)
        logger.debug(
            "Rolling window: %d samples (%d days at %.1f min intervals)",
            window_size,
            window_days,
            actual_interval_minutes,
        )

        features_added = 0
        for metric in self.config.ROLLING_AVG_METRICS:
            # Clean metric name for column (replace . with _)
            col_name = metric.replace(".", "_")

            if metric not in df.columns:
                logger.debug("Metric %s not found in data, skipping", metric)
                continue

            # Rolling mean (use min_periods=window_size to avoid data leakage)
            avg_col = f"{col_name}_rolling_avg_{window_days}d"
            df[avg_col] = df[metric].rolling(
                window=window_size, min_periods=window_size
            ).mean()
            features_added += 1

            # Rolling standard deviation
            std_col = f"{col_name}_rolling_std_{window_days}d"
            df[std_col] = df[metric].rolling(
                window=window_size, min_periods=window_size
            ).std()
            features_added += 1

        logger.info("Added %d rolling features", features_added)
        return df

    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply one-hot encoding for categorical features.

        Encodes time_of_use_period into:
        - tou_peak: 1 if peak period, else 0
        - tou_off_peak: 1 if off-peak period, else 0
        - tou_shoulder: 1 if shoulder period, else 0

        Args:
            df: DataFrame with categorical columns

        Returns:
            DataFrame with one-hot encoded features
        """
        if df.empty:
            return df

        logger.info("Encoding categorical features")

        df = df.copy()
        features_added = 0

        # Time of use period encoding
        # The time_of_use_period is stored as a tag in grid_price measurement
        # We need to infer it from the hour if not available
        if "grid_price.time_of_use_period" in df.columns:
            # One-hot encode existing column
            tou_col = "grid_price.time_of_use_period"
            df["tou_peak"] = (df[tou_col] == "peak").astype(int)
            df["tou_off_peak"] = (df[tou_col] == "off_peak").astype(int)
            df["tou_shoulder"] = (df[tou_col] == "shoulder").astype(int)
            features_added += 3
        elif "hour_of_day" in df.columns:
            # Infer time-of-use from hour using configurable schedule
            hour = df["hour_of_day"]
            peak_start, peak_end = self.config.tou_peak_hours
            df["tou_peak"] = ((hour >= peak_start) & (hour < peak_end)).astype(int)

            # Build shoulder mask from configured ranges
            shoulder_mask = False
            for start, end in self.config.tou_shoulder_hours:
                shoulder_mask = shoulder_mask | ((hour >= start) & (hour < end))
            df["tou_shoulder"] = shoulder_mask.astype(int)

            # Off-peak is everything else
            df["tou_off_peak"] = (
                (df["tou_peak"] == 0) & (df["tou_shoulder"] == 0)
            ).astype(int)
            features_added += 3

        logger.info("Added %d categorical features", features_added)
        return df

    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lag features for key metrics.

        Adds previous time step values for key prediction targets:
        - {metric}_lag_1: Value from 1 time step ago
        - {metric}_lag_1h: Value from 1 hour ago (dynamic based on sampling rate)

        Args:
            df: DataFrame with metric columns

        Returns:
            DataFrame with lag features
        """
        if df.empty:
            return df

        logger.info("Adding lag features")

        df = df.copy()
        df = df.sort_values("time")

        # Calculate samples per hour from actual data interval
        samples_per_hour = 12  # default for 5-min intervals
        if len(df) > 1 and "time" in df.columns:
            time_diff_minutes = (
                df["time"].iloc[1] - df["time"].iloc[0]
            ).total_seconds() / 60
            if time_diff_minutes > 0:
                samples_per_hour = round(60 / time_diff_minutes)

        features_added = 0
        for metric in self.LAG_METRICS:
            col_name = metric.replace(".", "_")

            if metric not in df.columns:
                continue

            # Lag 1 (previous time step)
            df[f"{col_name}_lag_1"] = df[metric].shift(1)
            features_added += 1

            # Lag for 1 hour ago (dynamic based on sampling rate)
            df[f"{col_name}_lag_1h"] = df[metric].shift(samples_per_hour)
            features_added += 1

        logger.info("Added %d lag features", features_added)
        return df

    def store_training_data(self, df: pd.DataFrame) -> int:
        """Store engineered features to training_data table.

        Args:
            df: DataFrame with engineered features

        Returns:
            Number of records stored
        """
        if df.empty:
            logger.info("No data to store")
            return 0

        if not self.is_connected():
            logger.warning("Not connected, cannot store data")
            return 0

        logger.info("Storing %d records to %s", len(df), self.config.target_table)

        # Convert DataFrame to list of dicts for Supabase
        # Handle NaN values and datetime serialization
        df = df.copy()

        # Filter to only expected columns (data may have extra columns from pivot)
        expected_columns = self.get_feature_columns()
        available_columns = [col for col in expected_columns if col in df.columns]
        extra_columns = [col for col in df.columns if col not in expected_columns]
        if extra_columns:
            logger.debug(
                "Dropping %d extra columns not in schema: %s",
                len(extra_columns),
                extra_columns[:5],  # Log first 5
            )
        df = df[available_columns]

        # Convert time to ISO string
        if "time" in df.columns:
            df["time"] = df["time"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")

        # Convert to records and replace NaN/None values
        # (df.where doesn't properly convert NaN to None for JSON)
        records = df.to_dict(orient="records")

        # Clean NaN values from records (JSON doesn't support NaN)
        for record in records:
            for key, value in record.items():
                if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                    record[key] = None

        # Store in batches
        stored = 0
        batch_size = self.config.batch_size

        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]

            try:
                self._supabase_client.table(self.config.target_table).upsert(
                    batch,
                    on_conflict="time,site_id,device_id",
                ).execute()
                stored += len(batch)
                logger.debug("Stored batch of %d records", len(batch))
            except Exception as e:
                logger.exception("Failed to store batch: %s", e)
                raise

        logger.info("Successfully stored %d records", stored)
        return stored

    def run(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """Run the complete feature engineering pipeline.

        Steps:
        1. Query raw data from Supabase
        2. Pivot to wide format
        3. Extract temporal features
        4. Calculate rolling features
        5. Encode categorical features
        6. Add lag features
        7. Store to training_data table

        Args:
            start_time: Start of time range (default: lookback_days ago)
            end_time: End of time range (default: now)

        Returns:
            Dictionary with pipeline execution results
        """
        if not self.config.enabled:
            logger.info("Feature engineering pipeline is disabled")
            return {
                "status": "disabled",
                "records_processed": 0,
                "records_stored": 0,
            }

        logger.info("Starting feature engineering pipeline")
        start = datetime.now(timezone.utc)

        try:
            # Step 1: Query raw data
            raw_df = self.query_raw_data(start_time, end_time)
            if raw_df.empty:
                return {
                    "status": "no_data",
                    "records_processed": 0,
                    "records_stored": 0,
                }

            # Step 2: Pivot to wide format
            df = self.pivot_metrics(raw_df)
            if df.empty:
                return {
                    "status": "pivot_failed",
                    "records_processed": 0,
                    "records_stored": 0,
                }

            # Step 3: Extract temporal features
            df = self.extract_temporal_features(df)

            # Step 4: Calculate rolling features
            df = self.calculate_rolling_features(df)

            # Drop rows with incomplete rolling features (NaN from min_periods)
            rolling_cols = [col for col in df.columns if "rolling" in col]
            if rolling_cols:
                rows_before = len(df)
                df = df.dropna(subset=rolling_cols)
                rows_dropped = rows_before - len(df)
                if rows_dropped > 0:
                    logger.info(
                        "Dropped %d rows with incomplete rolling features", rows_dropped
                    )

            # Step 5: Encode categorical features
            df = self.encode_categorical_features(df)

            # Step 6: Add lag features
            df = self.add_lag_features(df)

            # Step 7: Store to training_data table
            records_stored = self.store_training_data(df)

            elapsed = (datetime.now(timezone.utc) - start).total_seconds()

            result = {
                "status": "success",
                "records_processed": len(df),
                "records_stored": records_stored,
                "features_count": len(df.columns),
                "elapsed_seconds": elapsed,
            }

            logger.info(
                "Feature engineering completed: %d records, %d features, %.2fs",
                records_stored,
                len(df.columns),
                elapsed,
            )

            return result

        except Exception as e:
            logger.exception("Feature engineering pipeline failed")
            return {
                "status": "error",
                "error": str(e),
                "records_processed": 0,
                "records_stored": 0,
            }

    def get_feature_columns(self) -> list[str]:
        """Get list of all feature column names that will be generated.

        Returns:
            List of feature column names
        """
        features = []

        # Base columns
        features.extend(["time", "site_id", "device_id"])

        # Original metrics (after pivot)
        features.extend(self.config.METRICS_TO_PROCESS)

        # Temporal features
        features.extend(
            [
                "hour_of_day",
                "day_of_week",
                "day_of_month",
                "month",
                "is_weekend",
                "hour_sin",
                "hour_cos",
                "day_sin",
                "day_cos",
            ]
        )

        # Rolling features
        window = self.config.rolling_window_days
        for metric in self.config.ROLLING_AVG_METRICS:
            col_name = metric.replace(".", "_")
            features.append(f"{col_name}_rolling_avg_{window}d")
            features.append(f"{col_name}_rolling_std_{window}d")

        # Categorical features
        features.extend(["tou_peak", "tou_off_peak", "tou_shoulder"])

        # Lag features
        for metric in self.LAG_METRICS:
            col_name = metric.replace(".", "_")
            features.append(f"{col_name}_lag_1")
            features.append(f"{col_name}_lag_1h")

        return features

    def close(self) -> None:
        """Close Supabase connection."""
        self._supabase_client = None
        logger.info("Closed feature engineering pipeline connections")

    def __enter__(self) -> "FeatureEngineeringPipeline":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
