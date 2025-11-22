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
```

### Configuration

Create a `config.json` file to customize the simulator:

```json
{
  "device_id": "edge-gateway-001",
  "interval_seconds": 300,
  "seed": 42,
  "solar": {
    "capacity_kw": 10.0,
    "latitude": 37.7749
  },
  "battery": {
    "capacity_kwh": 13.5,
    "max_charge_rate_kw": 5.0
  },
  "home_load": {
    "base_load_kw": 0.5,
    "peak_load_kw": 8.0,
    "has_ev": true
  },
  "grid_price": {
    "off_peak_price": 0.08,
    "shoulder_price": 0.15,
    "peak_price": 0.30
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
tests/
  test_config.py
  test_generator.py
  test_models.py
  test_simulators.py
run_simulator.py     # CLI entry point
```
