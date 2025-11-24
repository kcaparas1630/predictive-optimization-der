# predictive-optimization-der

An app or dashboard that uses local weather data, individual consumption patterns, and dynamic grid pricing signals to offer hyper-localized optimization forecasts, helping asset owners or aggregators make more informed decisions about when to buy, sell, or store energy.

## Edge Gateway Simulator

The Edge Gateway Simulator generates realistic DER (Distributed Energy Resources) data for testing and development purposes.

### Features

- **Solar Generation**: Realistic patterns based on time of day, season, and weather variations
- **Battery State of Charge**: Intelligent charge/discharge management with efficiency and degradation modeling
- **Home Load**: Daily consumption patterns including HVAC, appliances, lighting, and EV charging
- **Grid Price Signals**: Time-of-Use (TOU) pricing with dynamic real-time pricing variations

### Installation

```bash
# Install with pip (development mode)
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

### Quick Start

```bash
# Generate a single data point
python run_simulator.py --once

# Run continuously every 5 minutes (300 seconds)
python run_simulator.py --continuous --interval 300

# Run on schedule aligned to clock intervals
python run_simulator.py --scheduled --interval 300

# Generate historical data for a date range
python run_simulator.py --historical --start 2024-06-15 --end 2024-06-16 --output data.jsonl

# Generate a sample configuration file
python run_simulator.py --generate-config --config config.json

# Enable InfluxDB storage (requires running InfluxDB instance)
python run_simulator.py --continuous --enable-influxdb --influxdb-token edge-gateway-dev-token

# Or use environment variables for InfluxDB configuration
export INFLUXDB_URL=http://localhost:8086
export INFLUXDB_TOKEN=mytoken
export INFLUXDB_ORG=edge-gateway
export INFLUXDB_BUCKET=der-data
python run_simulator.py --continuous --enable-influxdb
```

### InfluxDB Storage

The simulator supports writing data to a local InfluxDB instance for time series storage.

**Requirements:**
- InfluxDB 2.x running locally (default: `http://localhost:8086`)
- An authentication token with write permissions
- Install the optional dependency: `pip install influxdb-client`

**CLI Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--enable-influxdb` | Enable InfluxDB storage | disabled |
| `--influxdb-url` | InfluxDB server URL | `http://localhost:8086` |
| `--influxdb-token` | Authentication token | (required) |
| `--influxdb-org` | Organization name | `edge-gateway` |
| `--influxdb-bucket` | Bucket name | `der-data` |
| `--influxdb-retention-days` | Data retention period | 7 |

**Environment Variables:**
- `INFLUXDB_URL` - Server URL
- `INFLUXDB_TOKEN` - Authentication token
- `INFLUXDB_ORG` - Organization name
- `INFLUXDB_BUCKET` - Bucket name

**Docker Setup:**
```bash
docker-compose up -d influxdb
```

### Configuration

Create a `config.json` file to customize the simulator. All fields are optional and will use sensible defaults if omitted:

```json
{
  "device_id": "edge-gateway-001",
  "interval_seconds": 300,
  "output_file": null,
  "seed": null,
  "solar": {
    "capacity_kw": 10.0,
    "latitude": 37.7749,
    "panel_efficiency": 0.20,
    "temp_coefficient": -0.004
  },
  "battery": {
    "capacity_kwh": 13.5,
    "max_charge_rate_kw": 5.0,
    "max_discharge_rate_kw": 5.0,
    "round_trip_efficiency": 0.90,
    "initial_soc": 50.0,
    "min_soc": 10.0,
    "max_soc": 90.0
  },
  "home_load": {
    "base_load_kw": 0.5,
    "peak_load_kw": 8.0,
    "has_ev": true,
    "ev_charging_kw": 7.2,
    "hvac_capacity_kw": 3.5
  },
  "grid_price": {
    "off_peak_price": 0.08,
    "shoulder_price": 0.15,
    "peak_price": 0.30,
    "base_feed_in_tariff": 0.05,
    "demand_charge": 10.0,
    "volatility": 0.15
  }
}
```

### Output Format

Data is output as JSON with the following structure:

```json
{
  "timestamp": "2024-06-15T14:30:00",
  "device_id": "edge-gateway-001",
  "solar": {
    "generation_kw": 7.5,
    "irradiance_w_m2": 900.0,
    "panel_temp_celsius": 40.0,
    "efficiency_percent": 17.5
  },
  "battery": {
    "soc_percent": 65.0,
    "capacity_kwh": 13.5,
    "power_kw": 2.0,
    "voltage_v": 51.5,
    "temperature_celsius": 30.0,
    "cycles": 200,
    "health_percent": 97.0
  },
  "home_load": {
    "total_load_kw": 3.5,
    "hvac_kw": 1.5,
    "appliances_kw": 1.0,
    "lighting_kw": 0.3,
    "ev_charging_kw": 0.0,
    "other_kw": 0.7
  },
  "grid_price": {
    "price_per_kwh": 0.28,
    "feed_in_tariff": 0.08,
    "demand_charge": 10.0,
    "time_of_use_period": "peak",
    "carbon_intensity_g_kwh": 380.0
  },
  "net_grid_flow_kw": -2.0
}
```

### Running Tests

```bash
pytest tests/ -v
```

## Cloud Sync

Syncs data from local InfluxDB to Supabase cloud database.

### Commands

```bash
# Run sync once
python run_cloud_sync.py --once

# Run continuous sync (default interval: 60s)
python run_cloud_sync.py --continuous

# Run with custom interval
python run_cloud_sync.py --continuous --interval 120

# Show sync status
python run_cloud_sync.py --status

# With InfluxDB token
python run_cloud_sync.py --continuous --influxdb-token edge-gateway-dev-token
```

### Environment Variables

- `SUPABASE_URL` - Supabase project URL
- `SUPABASE_KEY` - Supabase API key
- `INFLUXDB_URL` - InfluxDB server URL
- `INFLUXDB_TOKEN` - InfluxDB authentication token
- `INFLUXDB_ORG` - InfluxDB organization
- `INFLUXDB_BUCKET` - InfluxDB bucket name
- `SYNC_BATCH_SIZE` - Records per sync batch
- `SYNC_INTERVAL_SECONDS` - Sync interval in seconds

## Feature Engineering

Processes raw DER data into ML-ready training features.

### Commands

```bash
# Run once with default settings
python run_feature_engineering.py --once

# Run with custom lookback period
python run_feature_engineering.py --once --lookback-days 60

# Run in continuous mode (default interval: 1 hour)
python run_feature_engineering.py --continuous

# Run with custom interval
python run_feature_engineering.py --continuous --interval 3600

# Show status
python run_feature_engineering.py --status
```

### Environment Variables

- `SUPABASE_URL` - Supabase project URL
- `SUPABASE_KEY` - Supabase API key
- `FE_SITE_ID` - Site identifier to filter data
- `FE_BATCH_SIZE` - Records per batch (default: 1000)
- `FE_ROLLING_WINDOW_DAYS` - Rolling window days (default: 7)
- `FE_LOOKBACK_DAYS` - Historical data lookback (default: 30)

## Model Training

Trains baseline forecasting models for load and solar prediction.

### Commands

```bash
# Train models with default settings
python run_model_training.py --train

# Train with custom model directory
python run_model_training.py --train --model-dir ./trained_models

# Evaluate existing models
python run_model_training.py --evaluate

# Show model status
python run_model_training.py --status
```

### Environment Variables

- `SUPABASE_URL` - Supabase project URL
- `SUPABASE_KEY` - Supabase API key
- `FORECAST_MODEL_DIR` - Directory to save trained models
- `FORECAST_TEST_SIZE` - Fraction of data for testing (0.0-1.0)

## Full Pipeline Example

Run the complete pipeline with InfluxDB token `edge-gateway-dev-token`:

```bash
# Terminal 1: Start InfluxDB
docker-compose up -d influxdb

# Terminal 2: Run simulator (generates data every 10 seconds)
python run_simulator.py --continuous --interval 10 --enable-influxdb --influxdb-token edge-gateway-dev-token

# Terminal 3: Run cloud sync (syncs to Supabase every 60 seconds)
python run_cloud_sync.py --continuous --interval 60

# Terminal 4: Run feature engineering (processes data every hour)
python run_feature_engineering.py --continuous --interval 3600

# Or run feature engineering once after accumulating data
python run_feature_engineering.py --once --lookback-days 7

# Train models after feature engineering completes
python run_model_training.py --train
```

### Project Structure

```text
edge_gateway/
  __init__.py
  config.py          # Configuration management
  generator.py       # Main data generator and runner
  models/
    __init__.py
    der_data.py      # Data models for DER metrics
  simulators/
    __init__.py
    base.py          # Base simulator class
    solar.py         # Solar generation simulator
    battery.py       # Battery state simulator
    home_load.py     # Home load simulator
    grid_price.py    # Grid price simulator
cloud/
  feature_engineering/  # Feature engineering pipeline
  forecasting/          # ML forecasting models
tests/
  __init__.py
  test_config.py
  test_generator.py
  test_models.py
  test_simulators.py
run_simulator.py           # Simulator CLI
run_cloud_sync.py          # Cloud sync CLI
run_feature_engineering.py # Feature engineering CLI
run_model_training.py      # Model training CLI
pyproject.toml             # Project configuration
```
