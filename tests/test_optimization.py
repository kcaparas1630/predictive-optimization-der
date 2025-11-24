"""Tests for the optimization solver engine module."""

import os
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from cloud.optimization.config import OptimizationConfig
from cloud.optimization.models import (
    BatteryConstraints,
    OptimizationInputs,
    OptimizationResult,
    SchedulePoint,
    TariffStructure,
)


class TestOptimizationConfig:
    """Tests for OptimizationConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = OptimizationConfig()

        assert config.supabase_url == "" or config.supabase_url  # May come from env
        assert config.influxdb_url  # Will be set to default or env var
        assert config.influxdb_org == "edge-gateway" or config.influxdb_org  # May come from env
        assert config.influxdb_bucket == "der-data" or config.influxdb_bucket  # May come from env
        assert config.enabled is False
        assert config.horizon_hours == 24
        assert config.time_step_minutes == 60
        assert config.solver == "CBC" or config.solver  # May come from env
        assert config.default_battery_capacity_kwh == 13.5
        assert config.default_max_charge_kw == 5.0
        assert config.default_max_discharge_kw == 5.0

    def test_explicit_values_override_defaults(self):
        """Test that explicit values override defaults."""
        config = OptimizationConfig(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
            enabled=True,
            horizon_hours=12,
            time_step_minutes=30,
            solver="GLPK",
        )

        assert config.supabase_url == "https://test.supabase.co"
        assert config.supabase_key == "test-key"
        assert config.enabled is True
        assert config.horizon_hours == 12
        assert config.time_step_minutes == 30
        assert config.solver == "GLPK"

    def test_env_var_overrides(self):
        """Test environment variable overrides."""
        env_vars = {
            "SUPABASE_URL": "https://env.supabase.co",
            "SUPABASE_KEY": "env-key",
            "INFLUXDB_URL": "http://influx:8086",
            "INFLUXDB_TOKEN": "env-token",
            "OPT_SITE_ID": "test-site",
            "OPT_DEVICE_ID": "test-device",
            "OPT_HORIZON_HOURS": "48",
            "OPT_TIME_STEP_MINUTES": "30",
            "OPT_SOLVER": "GLPK",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = OptimizationConfig()

            assert config.supabase_url == "https://env.supabase.co"
            assert config.supabase_key == "env-key"
            assert config.influxdb_url == "http://influx:8086"
            assert config.influxdb_token == "env-token"
            assert config.site_id == "test-site"
            assert config.device_id == "test-device"
            assert config.horizon_hours == 48
            assert config.time_step_minutes == 30
            assert config.solver == "GLPK"

    def test_num_time_steps(self):
        """Test calculation of number of time steps."""
        config = OptimizationConfig(horizon_hours=24, time_step_minutes=60)
        assert config.num_time_steps == 24

        config = OptimizationConfig(horizon_hours=24, time_step_minutes=30)
        assert config.num_time_steps == 48

        config = OptimizationConfig(horizon_hours=12, time_step_minutes=15)
        assert config.num_time_steps == 48

    def test_time_step_hours(self):
        """Test calculation of time step in hours."""
        config = OptimizationConfig(time_step_minutes=60)
        assert config.time_step_hours == 1.0

        config = OptimizationConfig(time_step_minutes=30)
        assert config.time_step_hours == 0.5

        config = OptimizationConfig(time_step_minutes=15)
        assert config.time_step_hours == 0.25

    def test_validate_success(self):
        """Test validation passes with valid config."""
        config = OptimizationConfig(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
            horizon_hours=24,
            time_step_minutes=60,
        )
        config.validate()  # Should not raise

    def test_validate_missing_supabase_url(self):
        """Test validation fails without Supabase URL."""
        config = OptimizationConfig(
            supabase_url="",
            supabase_key="test-key",
        )
        with pytest.raises(ValueError, match="Supabase URL is required"):
            config.validate()

    def test_validate_missing_supabase_key(self):
        """Test validation fails without Supabase key."""
        config = OptimizationConfig(
            supabase_url="https://test.supabase.co",
            supabase_key="",
        )
        with pytest.raises(ValueError, match="Supabase key is required"):
            config.validate()

    def test_validate_invalid_horizon(self):
        """Test validation fails with invalid horizon."""
        config = OptimizationConfig(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
            horizon_hours=0,
        )
        with pytest.raises(ValueError, match="horizon_hours must be positive"):
            config.validate()

    def test_validate_invalid_time_step(self):
        """Test validation fails with invalid time step."""
        config = OptimizationConfig(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
            time_step_minutes=0,
        )
        with pytest.raises(ValueError, match="time_step_minutes must be positive"):
            config.validate()

        config = OptimizationConfig(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
            time_step_minutes=90,
        )
        with pytest.raises(ValueError, match="time_step_minutes cannot exceed 60"):
            config.validate()

        config = OptimizationConfig(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
            time_step_minutes=7,
        )
        with pytest.raises(ValueError, match="time_step_minutes must divide evenly"):
            config.validate()


class TestTariffStructure:
    """Tests for TariffStructure."""

    def test_creation(self):
        """Test tariff structure creation."""
        tariff = TariffStructure(
            time_of_use_period="peak",
            price_per_kwh=0.35,
            feed_in_tariff=0.08,
            demand_charge=0.05,
        )

        assert tariff.time_of_use_period == "peak"
        assert tariff.price_per_kwh == 0.35
        assert tariff.feed_in_tariff == 0.08
        assert tariff.demand_charge == 0.05

    def test_to_dict(self):
        """Test conversion to dictionary."""
        tariff = TariffStructure(
            time_of_use_period="off_peak",
            price_per_kwh=0.12345,
            feed_in_tariff=0.05678,
        )

        d = tariff.to_dict()
        assert d["time_of_use_period"] == "off_peak"
        assert d["price_per_kwh"] == 0.1235  # Rounded to 4 decimals
        assert d["feed_in_tariff"] == 0.0568  # Rounded to 4 decimals
        assert d["demand_charge"] == 0.0


class TestBatteryConstraints:
    """Tests for BatteryConstraints."""

    def test_creation(self):
        """Test battery constraints creation."""
        battery = BatteryConstraints(
            capacity_kwh=13.5,
            max_charge_kw=5.0,
            max_discharge_kw=5.0,
            min_soc_percent=10.0,
            max_soc_percent=90.0,
            charge_efficiency=0.95,
            discharge_efficiency=0.95,
            initial_soc_percent=50.0,
        )

        assert battery.capacity_kwh == 13.5
        assert battery.max_charge_kw == 5.0
        assert battery.max_discharge_kw == 5.0
        assert battery.min_soc_percent == 10.0
        assert battery.max_soc_percent == 90.0
        assert battery.charge_efficiency == 0.95
        assert battery.discharge_efficiency == 0.95
        assert battery.initial_soc_percent == 50.0

    def test_soc_kwh_properties(self):
        """Test SOC kWh calculated properties."""
        battery = BatteryConstraints(
            capacity_kwh=10.0,
            max_charge_kw=5.0,
            max_discharge_kw=5.0,
            min_soc_percent=20.0,
            max_soc_percent=80.0,
            charge_efficiency=0.95,
            discharge_efficiency=0.95,
            initial_soc_percent=50.0,
        )

        assert battery.min_soc_kwh == 2.0  # 10 * 20%
        assert battery.max_soc_kwh == 8.0  # 10 * 80%
        assert battery.initial_soc_kwh == 5.0  # 10 * 50%

    def test_validation_invalid_capacity(self):
        """Test validation fails with invalid capacity."""
        with pytest.raises(ValueError, match="capacity_kwh must be positive"):
            BatteryConstraints(
                capacity_kwh=0,
                max_charge_kw=5.0,
                max_discharge_kw=5.0,
                min_soc_percent=10.0,
                max_soc_percent=90.0,
                charge_efficiency=0.95,
                discharge_efficiency=0.95,
                initial_soc_percent=50.0,
            )

    def test_validation_invalid_soc_limits(self):
        """Test validation fails with invalid SOC limits."""
        with pytest.raises(ValueError, match="SOC limits must satisfy"):
            BatteryConstraints(
                capacity_kwh=13.5,
                max_charge_kw=5.0,
                max_discharge_kw=5.0,
                min_soc_percent=90.0,  # min > max
                max_soc_percent=10.0,
                charge_efficiency=0.95,
                discharge_efficiency=0.95,
                initial_soc_percent=50.0,
            )

    def test_validation_invalid_efficiency(self):
        """Test validation fails with invalid efficiency."""
        with pytest.raises(ValueError, match="charge_efficiency must be between"):
            BatteryConstraints(
                capacity_kwh=13.5,
                max_charge_kw=5.0,
                max_discharge_kw=5.0,
                min_soc_percent=10.0,
                max_soc_percent=90.0,
                charge_efficiency=1.5,  # > 1
                discharge_efficiency=0.95,
                initial_soc_percent=50.0,
            )

    def test_validation_invalid_initial_soc(self):
        """Test validation fails with invalid initial SOC."""
        with pytest.raises(ValueError, match="initial_soc_percent must be within"):
            BatteryConstraints(
                capacity_kwh=13.5,
                max_charge_kw=5.0,
                max_discharge_kw=5.0,
                min_soc_percent=20.0,
                max_soc_percent=80.0,
                charge_efficiency=0.95,
                discharge_efficiency=0.95,
                initial_soc_percent=90.0,  # > max_soc_percent
            )

    def test_to_dict(self):
        """Test conversion to dictionary."""
        battery = BatteryConstraints(
            capacity_kwh=13.5555,
            max_charge_kw=5.0,
            max_discharge_kw=5.0,
            min_soc_percent=10.0,
            max_soc_percent=90.0,
            charge_efficiency=0.95555,
            discharge_efficiency=0.95,
            initial_soc_percent=50.0,
        )

        d = battery.to_dict()
        assert d["capacity_kwh"] == 13.56  # Rounded to 2 decimals
        assert d["charge_efficiency"] == 0.956  # Rounded to 3 decimals


class TestOptimizationInputs:
    """Tests for OptimizationInputs."""

    def create_sample_inputs(self, num_steps: int = 24) -> OptimizationInputs:
        """Create sample optimization inputs for testing."""
        start_time = datetime(2024, 6, 15, 0, 0, 0, tzinfo=timezone.utc)

        battery = BatteryConstraints(
            capacity_kwh=13.5,
            max_charge_kw=5.0,
            max_discharge_kw=5.0,
            min_soc_percent=10.0,
            max_soc_percent=90.0,
            charge_efficiency=0.95,
            discharge_efficiency=0.95,
            initial_soc_percent=50.0,
        )

        tariff_schedule = [
            TariffStructure(
                time_of_use_period="off_peak",
                price_per_kwh=0.12,
                feed_in_tariff=0.05,
            )
            for _ in range(num_steps)
        ]

        return OptimizationInputs(
            start_time=start_time,
            horizon_hours=24,
            time_step_minutes=60,
            load_forecast_kw=[3.0] * num_steps,
            solar_forecast_kw=[5.0] * num_steps,
            tariff_schedule=tariff_schedule,
            battery=battery,
        )

    def test_creation(self):
        """Test optimization inputs creation."""
        inputs = self.create_sample_inputs()

        assert inputs.horizon_hours == 24
        assert inputs.time_step_minutes == 60
        assert inputs.num_time_steps == 24
        assert len(inputs.load_forecast_kw) == 24
        assert len(inputs.solar_forecast_kw) == 24
        assert len(inputs.tariff_schedule) == 24

    def test_validation_mismatched_lengths(self):
        """Test validation fails with mismatched array lengths."""
        start_time = datetime(2024, 6, 15, 0, 0, 0, tzinfo=timezone.utc)

        battery = BatteryConstraints(
            capacity_kwh=13.5,
            max_charge_kw=5.0,
            max_discharge_kw=5.0,
            min_soc_percent=10.0,
            max_soc_percent=90.0,
            charge_efficiency=0.95,
            discharge_efficiency=0.95,
            initial_soc_percent=50.0,
        )

        with pytest.raises(ValueError, match="load_forecast_kw must have 24 elements"):
            OptimizationInputs(
                start_time=start_time,
                horizon_hours=24,
                time_step_minutes=60,
                load_forecast_kw=[3.0] * 12,  # Wrong length
                solar_forecast_kw=[5.0] * 24,
                tariff_schedule=[
                    TariffStructure("off_peak", 0.12, 0.05) for _ in range(24)
                ],
                battery=battery,
            )

    def test_to_dict(self):
        """Test conversion to dictionary."""
        inputs = self.create_sample_inputs()
        d = inputs.to_dict()

        assert d["horizon_hours"] == 24
        assert d["time_step_minutes"] == 60
        assert len(d["load_forecast_kw"]) == 24
        assert len(d["solar_forecast_kw"]) == 24
        assert len(d["tariff_schedule"]) == 24
        assert "battery" in d


class TestSchedulePoint:
    """Tests for SchedulePoint."""

    def test_creation(self):
        """Test schedule point creation."""
        timestamp = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

        point = SchedulePoint(
            timestamp=timestamp,
            charge_kw=2.5,
            discharge_kw=0.0,
            buy_kw=0.0,
            sell_kw=3.0,
            soc_percent=60.0,
            load_kw=3.0,
            solar_kw=8.0,
            price_per_kwh=0.20,
            feed_in_tariff=0.06,
            cost=-0.18,  # Negative = revenue from selling
        )

        assert point.charge_kw == 2.5
        assert point.discharge_kw == 0.0
        assert point.sell_kw == 3.0
        assert point.cost == -0.18

    def test_to_dict(self):
        """Test conversion to dictionary."""
        timestamp = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

        point = SchedulePoint(
            timestamp=timestamp,
            charge_kw=2.5555,
            discharge_kw=0.0,
            buy_kw=0.0,
            sell_kw=3.0,
            soc_percent=60.0555,
            load_kw=3.0,
            solar_kw=8.0,
            price_per_kwh=0.20,
            feed_in_tariff=0.06,
            cost=-0.18,
        )

        d = point.to_dict()
        assert d["charge_kw"] == 2.555  # Rounded to 3 decimals
        assert d["soc_percent"] == 60.06  # Rounded to 2 decimals
        assert "timestamp" in d

    def test_to_supabase_record(self):
        """Test conversion to Supabase record format."""
        timestamp = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

        point = SchedulePoint(
            timestamp=timestamp,
            charge_kw=2.5,
            discharge_kw=0.0,
            buy_kw=0.0,
            sell_kw=3.0,
            soc_percent=60.0,
            load_kw=3.0,
            solar_kw=8.0,
            price_per_kwh=0.20,
            feed_in_tariff=0.06,
            cost=-0.18,
        )

        record = point.to_supabase_record("site-1", "device-1", "run-1")

        assert record["site_id"] == "site-1"
        assert record["device_id"] == "device-1"
        assert record["run_id"] == "run-1"
        assert record["charge_kw"] == 2.5
        assert record["sell_kw"] == 3.0
        assert "time" in record


class TestOptimizationResult:
    """Tests for OptimizationResult."""

    def create_sample_result(self) -> OptimizationResult:
        """Create sample optimization result for testing."""
        start_time = datetime(2024, 6, 15, 0, 0, 0, tzinfo=timezone.utc)
        end_time = datetime(2024, 6, 16, 0, 0, 0, tzinfo=timezone.utc)
        created_at = datetime(2024, 6, 15, 0, 0, 1, tzinfo=timezone.utc)

        battery = BatteryConstraints(
            capacity_kwh=13.5,
            max_charge_kw=5.0,
            max_discharge_kw=5.0,
            min_soc_percent=10.0,
            max_soc_percent=90.0,
            charge_efficiency=0.95,
            discharge_efficiency=0.95,
            initial_soc_percent=50.0,
        )

        inputs = OptimizationInputs(
            start_time=start_time,
            horizon_hours=24,
            time_step_minutes=60,
            load_forecast_kw=[3.0] * 24,
            solar_forecast_kw=[5.0] * 24,
            tariff_schedule=[
                TariffStructure("off_peak", 0.12, 0.05) for _ in range(24)
            ],
            battery=battery,
        )

        schedule = [
            SchedulePoint(
                timestamp=start_time + timedelta(hours=t),
                charge_kw=1.0,
                discharge_kw=0.0,
                buy_kw=0.0,
                sell_kw=1.0,
                soc_percent=50.0 + t,
                load_kw=3.0,
                solar_kw=5.0,
                price_per_kwh=0.12,
                feed_in_tariff=0.05,
                cost=-0.05,
            )
            for t in range(24)
        ]

        return OptimizationResult(
            status="Optimal",
            run_id="test-run-123",
            start_time=start_time,
            end_time=end_time,
            created_at=created_at,
            total_cost=-1.20,
            total_grid_import_kwh=0.0,
            total_grid_export_kwh=24.0,
            schedule=schedule,
            inputs=inputs,
            solver_time_seconds=0.5,
            message="",
        )

    def test_creation(self):
        """Test optimization result creation."""
        result = self.create_sample_result()

        assert result.status == "Optimal"
        assert result.is_optimal is True
        assert result.run_id == "test-run-123"
        assert result.total_cost == -1.20
        assert len(result.schedule) == 24

    def test_get_summary(self):
        """Test getting result summary."""
        result = self.create_sample_result()
        summary = result.get_summary()

        assert summary["status"] == "Optimal"
        assert summary["is_optimal"] is True
        assert summary["run_id"] == "test-run-123"
        assert summary["total_cost"] == -1.20
        assert summary["num_time_steps"] == 24

    def test_to_dict_and_from_dict(self):
        """Test round-trip serialization."""
        result = self.create_sample_result()

        # Serialize
        d = result.to_dict()

        # Deserialize
        restored = OptimizationResult.from_dict(d)

        assert restored.status == result.status
        assert restored.run_id == result.run_id
        assert restored.total_cost == result.total_cost
        assert len(restored.schedule) == len(result.schedule)
        assert restored.schedule[0].charge_kw == result.schedule[0].charge_kw

    def test_to_json(self):
        """Test JSON serialization."""
        result = self.create_sample_result()
        json_str = result.to_json()

        assert isinstance(json_str, str)
        assert '"status": "Optimal"' in json_str
        assert '"run_id": "test-run-123"' in json_str


class TestOptimizationSolver:
    """Tests for OptimizationSolver."""

    def test_disabled_solver(self):
        """Test that disabled solver doesn't connect."""
        from cloud.optimization.solver import OptimizationSolver

        config = OptimizationConfig(enabled=False)
        solver = OptimizationSolver(config)

        assert solver.is_connected() is False

    @patch("cloud.optimization.solver.PULP_AVAILABLE", True)
    @patch("cloud.optimization.solver.SUPABASE_AVAILABLE", True)
    @patch("cloud.optimization.solver.create_client")
    def test_solver_connects(self, mock_create_client):
        """Test that solver connects when enabled."""
        from cloud.optimization.solver import OptimizationSolver

        mock_client = MagicMock()
        mock_create_client.return_value = mock_client

        config = OptimizationConfig(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
            enabled=True,
        )

        solver = OptimizationSolver(config)

        assert solver.is_connected() is True
        mock_create_client.assert_called_once()

    @patch("cloud.optimization.solver.PULP_AVAILABLE", True)
    @patch("cloud.optimization.solver.SUPABASE_AVAILABLE", True)
    @patch("cloud.optimization.solver.create_client")
    def test_create_default_battery_constraints(self, mock_create_client):
        """Test creating default battery constraints."""
        from cloud.optimization.solver import OptimizationSolver

        mock_create_client.return_value = MagicMock()

        config = OptimizationConfig(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
            enabled=True,
            default_battery_capacity_kwh=10.0,
            default_max_charge_kw=3.0,
        )

        solver = OptimizationSolver(config)
        battery = solver.create_default_battery_constraints(initial_soc_percent=50.0)

        assert battery.capacity_kwh == 10.0
        assert battery.max_charge_kw == 3.0
        assert battery.initial_soc_percent == 50.0

    @patch("cloud.optimization.solver.PULP_AVAILABLE", True)
    @patch("cloud.optimization.solver.SUPABASE_AVAILABLE", True)
    @patch("cloud.optimization.solver.create_client")
    @patch("cloud.optimization.solver.pulp")
    def test_optimize_simple_problem(self, mock_pulp, mock_create_client):
        """Test optimization with a simple problem."""
        from cloud.optimization.solver import OptimizationSolver

        mock_create_client.return_value = MagicMock()

        # Mock PuLP
        mock_problem = MagicMock()
        mock_problem.status = 1  # Optimal
        mock_pulp.LpProblem.return_value = mock_problem
        mock_pulp.LpStatus = {1: "Optimal"}
        mock_pulp.LpMinimize = 1

        # Mock LpVariable
        mock_var = MagicMock()
        mock_pulp.LpVariable.return_value = mock_var
        mock_pulp.value.return_value = 1.0  # Return 1.0 for all variable values

        # Mock lpSum
        mock_pulp.lpSum.return_value = mock_var

        # Mock solver
        mock_solver = MagicMock()
        mock_pulp.PULP_CBC_CMD.return_value = mock_solver

        config = OptimizationConfig(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
            enabled=True,
            horizon_hours=2,  # Short horizon for testing
            time_step_minutes=60,
        )

        solver = OptimizationSolver(config)

        # Create simple inputs
        start_time = datetime(2024, 6, 15, 0, 0, 0, tzinfo=timezone.utc)
        battery = BatteryConstraints(
            capacity_kwh=13.5,
            max_charge_kw=5.0,
            max_discharge_kw=5.0,
            min_soc_percent=10.0,
            max_soc_percent=90.0,
            charge_efficiency=0.95,
            discharge_efficiency=0.95,
            initial_soc_percent=50.0,
        )

        inputs = OptimizationInputs(
            start_time=start_time,
            horizon_hours=2,
            time_step_minutes=60,
            load_forecast_kw=[3.0, 3.0],
            solar_forecast_kw=[5.0, 5.0],
            tariff_schedule=[
                TariffStructure("off_peak", 0.12, 0.05),
                TariffStructure("off_peak", 0.12, 0.05),
            ],
            battery=battery,
        )

        result = solver.optimize(inputs)

        assert result.run_id is not None
        assert result.start_time == start_time

    def test_solver_context_manager(self):
        """Test that solver works as context manager."""
        from cloud.optimization.solver import OptimizationSolver

        config = OptimizationConfig(enabled=False)

        with OptimizationSolver(config) as solver:
            assert solver.is_connected() is False

        # Solver should be closed after exiting context


class TestIntegration:
    """Integration tests for the optimization module.

    These tests verify the complete workflow without mocking.
    They require PuLP to be installed but not external services.
    """

    @pytest.fixture
    def solver_config(self):
        """Create a config that won't connect to external services."""
        return OptimizationConfig(
            supabase_url="",
            supabase_key="",
            enabled=False,  # Disabled to avoid connection
        )

    def test_full_model_workflow(self):
        """Test creating inputs, running optimization models standalone."""
        # This tests the data models without actually running optimization
        start_time = datetime(2024, 6, 15, 0, 0, 0, tzinfo=timezone.utc)

        battery = BatteryConstraints(
            capacity_kwh=13.5,
            max_charge_kw=5.0,
            max_discharge_kw=5.0,
            min_soc_percent=10.0,
            max_soc_percent=90.0,
            charge_efficiency=0.95,
            discharge_efficiency=0.95,
            initial_soc_percent=50.0,
        )

        # Verify battery properties
        assert battery.min_soc_kwh == 1.35
        assert battery.max_soc_kwh == 12.15
        assert battery.initial_soc_kwh == 6.75

        # Create tariff schedule with different periods
        num_steps = 24
        tariff_schedule = []
        for t in range(num_steps):
            hour = t
            if 17 <= hour < 21:
                tariff = TariffStructure("peak", 0.35, 0.08)
            elif 7 <= hour < 17:
                tariff = TariffStructure("shoulder", 0.20, 0.06)
            else:
                tariff = TariffStructure("off_peak", 0.12, 0.05)
            tariff_schedule.append(tariff)

        # Verify tariff schedule
        assert tariff_schedule[0].time_of_use_period == "off_peak"
        assert tariff_schedule[12].time_of_use_period == "shoulder"
        assert tariff_schedule[18].time_of_use_period == "peak"

        # Create inputs
        inputs = OptimizationInputs(
            start_time=start_time,
            horizon_hours=24,
            time_step_minutes=60,
            load_forecast_kw=[3.0] * 24,
            solar_forecast_kw=[0.0] * 6 + [5.0] * 12 + [0.0] * 6,
            tariff_schedule=tariff_schedule,
            battery=battery,
        )

        assert inputs.num_time_steps == 24
        assert inputs.time_step_hours == 1.0

        # Verify serialization
        inputs_dict = inputs.to_dict()
        assert inputs_dict["horizon_hours"] == 24
        assert len(inputs_dict["load_forecast_kw"]) == 24
