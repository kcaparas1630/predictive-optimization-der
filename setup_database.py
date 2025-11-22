"""
Database setup script using Native PostgreSQL Partitioning.

Run these SQL commands in the Supabase SQL Editor (https://supabase.com/dashboard):

================================================================================
STEP 1: Create the Parent Table (Partitioned by Time)
================================================================================

CREATE TABLE public.readings (
    time TIMESTAMPTZ NOT NULL,
    site_id TEXT NOT NULL,
    device_id TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL
) PARTITION BY RANGE (time);

CREATE INDEX ON public.readings (time);

================================================================================
STEP 2: Create the Partition Management Function
================================================================================

CREATE OR REPLACE FUNCTION create_new_partition(p_date TIMESTAMPTZ)
RETURNS TEXT AS $$
DECLARE
    partition_name TEXT;
    start_time TEXT;
    end_time TEXT;
BEGIN
    start_time := date_trunc('month', p_date)::TEXT;
    end_time := (date_trunc('month', p_date) + INTERVAL '1 month')::TEXT;
    partition_name := 'readings_' || to_char(p_date, 'YYYY_MM');

    IF NOT EXISTS (
        SELECT 1 FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE c.relname = partition_name AND n.nspname = 'public'
    ) THEN
        EXECUTE 'CREATE TABLE public.' || quote_ident(partition_name) ||
                ' PARTITION OF public.readings FOR VALUES FROM (''' || start_time || ''') TO (''' || end_time || ''');';
        EXECUTE 'CREATE INDEX ON public.' || quote_ident(partition_name) || ' (time);';
        EXECUTE 'CREATE INDEX ON public.' || quote_ident(partition_name) || ' (site_id, time);';
        RETURN 'Partition ' || partition_name || ' created successfully.';
    ELSE
        RETURN 'Partition ' || partition_name || ' already exists.';
    END IF;
END;
$$ LANGUAGE plpgsql;

================================================================================
STEP 3: Create Initial Partition and Test Insert
================================================================================

SELECT create_new_partition(NOW());

INSERT INTO public.readings (time, site_id, device_id, metric_name, metric_value)
VALUES (NOW(), 'Site_001', 'Battery', 'SoC', 95.5);

================================================================================
STEP 4: Enable Row Level Security (Optional)
================================================================================

ALTER TABLE public.readings ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Allow all access" ON public.readings FOR ALL USING (true);

"""

import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def test_connection():
    """Test the Supabase connection."""
    try:
        result = supabase.table("readings").select("*").limit(1).execute()
        print("✓ Supabase connection successful!")
        print(f"  Readings table accessible: {result.data is not None}")
        if result.data:
            print(f"  Sample data: {result.data}")
        return True
    except Exception as e:
        if "relation" in str(e) and "does not exist" in str(e):
            print("✓ Supabase connection successful!")
            print("  ✗ Readings table not yet created - run SQL commands in Supabase SQL Editor")
        else:
            print(f"✗ Connection error: {e}")
        return False


def insert_test_reading():
    """Insert a test reading to verify write access."""
    from datetime import datetime, timezone

    test_data = {
        "time": datetime.now(timezone.utc).isoformat(),
        "site_id": "Site_001",
        "device_id": "Battery",
        "metric_name": "SoC",
        "metric_value": 95.5
    }

    try:
        result = supabase.table("readings").insert(test_data).execute()
        print("✓ Test reading inserted successfully!")
        print(f"  Data: {result.data}")
        return True
    except Exception as e:
        print(f"✗ Insert error: {e}")
        return False


if __name__ == "__main__":
    print("Testing Supabase connection...\n")
    test_connection()