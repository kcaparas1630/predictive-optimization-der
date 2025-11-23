"""Database setup script for the training_data table.

Run these SQL commands in the Supabase SQL Editor (https://supabase.com/dashboard):

================================================================================
STEP 1: Create the training_data Table
================================================================================

This table stores engineered features for ML training. It uses a wide format
where each row represents a single time point with all features as columns.

CREATE TABLE public.training_data (
    -- Primary identifiers
    time TIMESTAMPTZ NOT NULL,
    site_id TEXT NOT NULL,
    device_id TEXT NOT NULL,

    -- Original metrics (from readings table pivot)
    "solar.generation_kw" DOUBLE PRECISION,
    "solar.irradiance_w_m2" DOUBLE PRECISION,
    "solar.panel_temp_celsius" DOUBLE PRECISION,
    "solar.efficiency_percent" DOUBLE PRECISION,
    "battery.soc_percent" DOUBLE PRECISION,
    "battery.power_kw" DOUBLE PRECISION,
    "battery.temperature_celsius" DOUBLE PRECISION,
    "battery.health_percent" DOUBLE PRECISION,
    "home_load.total_load_kw" DOUBLE PRECISION,
    "home_load.hvac_kw" DOUBLE PRECISION,
    "home_load.appliances_kw" DOUBLE PRECISION,
    "home_load.lighting_kw" DOUBLE PRECISION,
    "home_load.ev_charging_kw" DOUBLE PRECISION,
    "grid_price.price_per_kwh" DOUBLE PRECISION,
    "grid_price.feed_in_tariff" DOUBLE PRECISION,
    "grid_price.demand_charge" DOUBLE PRECISION,
    "grid_price.carbon_intensity_g_kwh" DOUBLE PRECISION,
    "system.net_grid_flow_kw" DOUBLE PRECISION,

    -- Temporal features
    hour_of_day INTEGER,
    day_of_week INTEGER,
    day_of_month INTEGER,
    month INTEGER,
    is_weekend INTEGER,
    hour_sin DOUBLE PRECISION,
    hour_cos DOUBLE PRECISION,
    day_sin DOUBLE PRECISION,
    day_cos DOUBLE PRECISION,

    -- Rolling window features (7-day)
    solar_generation_kw_rolling_avg_7d DOUBLE PRECISION,
    solar_generation_kw_rolling_std_7d DOUBLE PRECISION,
    home_load_total_load_kw_rolling_avg_7d DOUBLE PRECISION,
    home_load_total_load_kw_rolling_std_7d DOUBLE PRECISION,
    battery_soc_percent_rolling_avg_7d DOUBLE PRECISION,
    battery_soc_percent_rolling_std_7d DOUBLE PRECISION,
    grid_price_price_per_kwh_rolling_avg_7d DOUBLE PRECISION,
    grid_price_price_per_kwh_rolling_std_7d DOUBLE PRECISION,
    system_net_grid_flow_kw_rolling_avg_7d DOUBLE PRECISION,
    system_net_grid_flow_kw_rolling_std_7d DOUBLE PRECISION,

    -- Time-of-use categorical features (one-hot encoded)
    tou_peak INTEGER,
    tou_off_peak INTEGER,
    tou_shoulder INTEGER,

    -- Lag features
    solar_generation_kw_lag_1 DOUBLE PRECISION,
    solar_generation_kw_lag_12 DOUBLE PRECISION,
    home_load_total_load_kw_lag_1 DOUBLE PRECISION,
    home_load_total_load_kw_lag_12 DOUBLE PRECISION,
    grid_price_price_per_kwh_lag_1 DOUBLE PRECISION,
    grid_price_price_per_kwh_lag_12 DOUBLE PRECISION,

    -- Composite primary key
    PRIMARY KEY (time, site_id, device_id)
);

-- Create indexes for common query patterns
CREATE INDEX idx_training_data_time ON public.training_data (time);
CREATE INDEX idx_training_data_site_time ON public.training_data (site_id, time);

================================================================================
STEP 2: Enable Row Level Security (Optional)
================================================================================

ALTER TABLE public.training_data ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Allow all access" ON public.training_data FOR ALL USING (true);

================================================================================
STEP 3: Create a view for easy feature selection
================================================================================

This view provides a cleaned version of the training data with standardized
column names (replacing dots with underscores for ML framework compatibility).

CREATE OR REPLACE VIEW public.training_features AS
SELECT
    time,
    site_id,
    device_id,
    -- Original metrics (renamed for ML compatibility)
    "solar.generation_kw" AS solar_generation_kw,
    "solar.irradiance_w_m2" AS solar_irradiance_w_m2,
    "solar.panel_temp_celsius" AS solar_panel_temp_celsius,
    "solar.efficiency_percent" AS solar_efficiency_percent,
    "battery.soc_percent" AS battery_soc_percent,
    "battery.power_kw" AS battery_power_kw,
    "battery.temperature_celsius" AS battery_temperature_celsius,
    "battery.health_percent" AS battery_health_percent,
    "home_load.total_load_kw" AS home_load_total_load_kw,
    "home_load.hvac_kw" AS home_load_hvac_kw,
    "home_load.appliances_kw" AS home_load_appliances_kw,
    "home_load.lighting_kw" AS home_load_lighting_kw,
    "home_load.ev_charging_kw" AS home_load_ev_charging_kw,
    "grid_price.price_per_kwh" AS grid_price_price_per_kwh,
    "grid_price.feed_in_tariff" AS grid_price_feed_in_tariff,
    "grid_price.demand_charge" AS grid_price_demand_charge,
    "grid_price.carbon_intensity_g_kwh" AS grid_price_carbon_intensity_g_kwh,
    "system.net_grid_flow_kw" AS system_net_grid_flow_kw,
    -- Temporal features
    hour_of_day,
    day_of_week,
    day_of_month,
    month,
    is_weekend,
    hour_sin,
    hour_cos,
    day_sin,
    day_cos,
    -- Rolling features
    solar_generation_kw_rolling_avg_7d,
    solar_generation_kw_rolling_std_7d,
    home_load_total_load_kw_rolling_avg_7d,
    home_load_total_load_kw_rolling_std_7d,
    battery_soc_percent_rolling_avg_7d,
    battery_soc_percent_rolling_std_7d,
    grid_price_price_per_kwh_rolling_avg_7d,
    grid_price_price_per_kwh_rolling_std_7d,
    system_net_grid_flow_kw_rolling_avg_7d,
    system_net_grid_flow_kw_rolling_std_7d,
    -- Categorical features
    tou_peak,
    tou_off_peak,
    tou_shoulder,
    -- Lag features
    solar_generation_kw_lag_1,
    solar_generation_kw_lag_12,
    home_load_total_load_kw_lag_1,
    home_load_total_load_kw_lag_12,
    grid_price_price_per_kwh_lag_1,
    grid_price_price_per_kwh_lag_12
FROM public.training_data;

"""

import os
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

# Import Supabase client
try:
    from supabase import create_client, Client
    from postgrest.exceptions import APIError as PostgrestAPIError

    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    PostgrestAPIError = Exception  # type: ignore
    print("Warning: supabase not installed. Run: pip install supabase")


# PostgreSQL error code for "relation does not exist"
POSTGRES_UNDEFINED_TABLE = "42P01"


def get_client() -> "Client":
    """Get Supabase client from environment variables."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_KEY must be set in environment variables."
        )

    return create_client(url, key)


def test_connection() -> bool:
    """Test the Supabase connection.

    Returns:
        True if connection successful and table exists, False if table missing.

    Raises:
        ValueError: If SUPABASE_URL or SUPABASE_KEY not configured.
        PostgrestAPIError: For unexpected API errors (auth, network, etc.).
    """
    client = get_client()  # Let ValueError propagate for misconfiguration

    try:
        result = client.table("training_data").select("*").limit(1).execute()
        print("Supabase connection successful!")
        print(f"  training_data table accessible: {result.data is not None}")
        if result.data:
            print(f"  Sample data exists: {len(result.data)} record(s)")
        return True
    except PostgrestAPIError as e:
        # Check for "relation does not exist" using structured error code
        if getattr(e, "code", None) == POSTGRES_UNDEFINED_TABLE:
            print("Supabase connection successful!")
            print(
                "  training_data table not yet created - "
                "run SQL commands in Supabase SQL Editor"
            )
            return False
        # Re-raise unexpected API errors (auth issues, network errors, etc.)
        raise


def insert_test_record() -> bool:
    """Insert a test record to verify write access."""
    try:
        client = get_client()

        test_data = {
            "time": datetime.now(timezone.utc).isoformat(),
            "site_id": "test-site",
            "device_id": "test-device",
            "solar.generation_kw": 5.5,
            "hour_of_day": 12,
            "day_of_week": 2,
            "is_weekend": 0,
            "tou_peak": 0,
            "tou_off_peak": 0,
            "tou_shoulder": 1,
        }

        result = client.table("training_data").insert(test_data).execute()
        print("Test record inserted successfully!")
        print(f"  Data: {result.data}")
        return True
    except Exception as e:
        print(f"Insert error: {e}")
        return False


def get_table_stats() -> dict:
    """Get statistics about the training_data table.

    Returns:
        Dictionary with table statistics.

    Raises:
        ValueError: If SUPABASE_URL or SUPABASE_KEY not configured.
        PostgrestAPIError: For API errors (table missing, auth issues, etc.).
    """
    client = get_client()  # Let ValueError propagate for misconfiguration

    # Count total records
    result = client.table("training_data").select("*", count="exact").execute()

    # Get time range
    earliest = (
        client.table("training_data")
        .select("time")
        .order("time", desc=False)
        .limit(1)
        .execute()
    )
    latest = (
        client.table("training_data")
        .select("time")
        .order("time", desc=True)
        .limit(1)
        .execute()
    )

    return {
        "total_records": result.count if result.count else 0,
        "earliest_time": earliest.data[0]["time"] if earliest.data else None,
        "latest_time": latest.data[0]["time"] if latest.data else None,
    }


if __name__ == "__main__":
    print("Testing training_data table setup...\n")

    if not SUPABASE_AVAILABLE:
        print("Please install supabase: pip install supabase")
        exit(1)

    print("1. Testing connection...")
    if test_connection():
        print("\n2. Getting table statistics...")
        stats = get_table_stats()
        if stats:
            print(f"   Total records: {stats.get('total_records', 0)}")
            print(
                f"   Time range: {stats.get('earliest_time')} to {stats.get('latest_time')}"
            )
        else:
            print("   No statistics available (table may be empty)")
    else:
        print("\nPlease create the training_data table using the SQL commands above.")
