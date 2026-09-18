from __future__ import annotations

import copy

import pytest

from app.final_validator import FinalValidationError, validate_final_response
from app.optimizer import optimize_energy


def make_scenario(
    *,
    demand=10.0,
    solar=0.0,
    tariff=5.0,
    capacity=20.0,
    initial=10.0,
    minimum=0.0,
    max_charge=5.0,
    max_discharge=5.0,
):
    def value(v, h):
        if callable(v):
            return float(v(h))
        if isinstance(v, (list, tuple)):
            return float(v[h])
        return float(v)

    return {
        "scenario_id": "TEST",
        "operator_notes": ["synthetic test note"],
        "hours": [
            {
                "hour": h,
                "demand_kwh": value(demand, h),
                "solar_kwh": value(solar, h),
                "tariff_bdt_per_kwh": value(tariff, h),
            }
            for h in range(24)
        ],
        "battery": {
            "capacity_kwh": float(capacity),
            "initial_energy_kwh": float(initial),
            "minimum_energy_kwh": float(minimum),
            "max_charge_kwh_per_hour": float(max_charge),
            "max_discharge_kwh_per_hour": float(max_discharge),
        },
    }


def no_op():
    return {
        "note_index": 0,
        "applies": False,
        "directive_type": "no_op",
        "structured_adjustment": None,
        "explanation": "test no-op",
    }


def directive(kind, adjustment, index=0):
    return {
        "note_index": index,
        "applies": True,
        "directive_type": kind,
        "structured_adjustment": adjustment,
        "explanation": "test directive",
    }


def b_signed(row):
    if row["battery_action"] == "charge":
        return row["battery_kwh"]
    if row["battery_action"] == "discharge":
        return -row["battery_kwh"]
    return 0.0


def assert_valid(scenario, out, directives):
    validate_final_response(scenario, out, directives=directives)
    assert len(out["hourly_plan"]) == 24
    assert out["hourly_plan"][-1]["battery_energy_after_kwh"] == pytest.approx(
        scenario["battery"]["initial_energy_kwh"], abs=1e-6
    )


def test_baseline_case():
    scenario = make_scenario()
    directives = [no_op()]
    out = optimize_energy(scenario, directives)
    assert_valid(scenario, out, directives)
    assert out["total_cost_bdt"] == pytest.approx(24 * 10 * 5, abs=1e-6)


def test_solar_reduction():
    scenario = make_scenario(solar=lambda h: 8 if h in (10, 11) else 0)
    directives = [directive("solar_reduction", {"hours": [10, 11], "factor": 0.25})]
    out = optimize_energy(scenario, directives)
    assert_valid(scenario, out, directives)
    for h in (10, 11):
        assert out["hourly_plan"][h]["solar_used_kwh"] <= 2.0 + 1e-6


def test_minimum_reserve():
    tariff = [5.0] * 24
    tariff[0] = 1.0
    tariff[18] = tariff[19] = 20.0
    scenario = make_scenario(tariff=tariff, capacity=20, initial=10, minimum=2)
    directives = [
        directive(
            "minimum_battery_reserve",
            {"hours": [18, 19], "minimum_energy_kwh": 8.0},
        )
    ]
    out = optimize_energy(scenario, directives)
    assert_valid(scenario, out, directives)
    assert out["hourly_plan"][18]["battery_energy_after_kwh"] >= 8.0 - 1e-6
    assert out["hourly_plan"][19]["battery_energy_after_kwh"] >= 8.0 - 1e-6


def test_no_charge_window():
    tariff = [5.0] * 24
    tariff[2] = tariff[3] = 1.0
    tariff[18] = 20.0
    scenario = make_scenario(tariff=tariff)
    directives = [directive("no_charge_window", {"hours": [2, 3]})]
    out = optimize_energy(scenario, directives)
    assert_valid(scenario, out, directives)
    assert all(b_signed(out["hourly_plan"][h]) <= 1e-6 for h in (2, 3))


def test_no_discharge_window():
    tariff = [5.0] * 24
    tariff[18] = tariff[19] = 20.0
    scenario = make_scenario(tariff=tariff)
    directives = [directive("no_discharge_window", {"hours": [18, 19]})]
    out = optimize_energy(scenario, directives)
    assert_valid(scenario, out, directives)
    assert all(b_signed(out["hourly_plan"][h]) >= -1e-6 for h in (18, 19))


def test_grid_cap():
    tariff = [5.0] * 24
    tariff[18] = 20.0
    scenario = make_scenario(tariff=tariff, initial=10, minimum=0, max_discharge=5)
    directives = [directive("max_grid_window", {"hours": [18], "max_grid_kwh": 6.0})]
    out = optimize_energy(scenario, directives)
    assert_valid(scenario, out, directives)
    assert out["hourly_plan"][18]["grid_kwh"] <= 6.0 + 1e-6
    assert b_signed(out["hourly_plan"][18]) <= -4.0 + 1e-6


def test_multiple_directives():
    scenario = make_scenario(
        solar=lambda h: 8 if h in (10, 11) else 0,
        tariff=lambda h: 20 if h == 18 else 5,
    )
    directives = [
        directive("solar_reduction", {"hours": [10, 11], "factor": 0.5}, 0),
        directive("no_charge_window", {"hours": [14, 15]}, 1),
        {
            "note_index": 2,
            "applies": False,
            "directive_type": "no_op",
            "structured_adjustment": None,
            "explanation": "irrelevant",
        },
    ]
    out = optimize_energy(scenario, directives)
    assert_valid(scenario, out, directives)
    assert out["hourly_plan"][10]["solar_used_kwh"] <= 4 + 1e-6
    assert b_signed(out["hourly_plan"][14]) <= 1e-6
    assert b_signed(out["hourly_plan"][15]) <= 1e-6


def test_reserve_plus_grid_cap():
    tariff = [5.0] * 24
    tariff[0] = 1.0
    tariff[18] = 20.0
    scenario = make_scenario(tariff=tariff, initial=10, capacity=20, max_charge=5, max_discharge=5)
    directives = [
        directive(
            "minimum_battery_reserve",
            {"hours": [18], "minimum_energy_kwh": 8.0},
            0,
        ),
        directive("max_grid_window", {"hours": [18], "max_grid_kwh": 6.0}, 1),
    ]
    out = optimize_energy(scenario, directives)
    assert_valid(scenario, out, directives)
    assert out["hourly_plan"][18]["grid_kwh"] <= 6.0 + 1e-6
    assert out["hourly_plan"][18]["battery_energy_after_kwh"] >= 8.0 - 1e-6


def test_separate_charge_and_discharge_windows():
    tariff = [5.0] * 24
    tariff[1] = 1.0
    tariff[18] = 20.0
    scenario = make_scenario(tariff=tariff)
    directives = [
        directive("no_charge_window", {"hours": [11, 12]}, 0),
        directive("no_discharge_window", {"hours": [17, 18]}, 1),
    ]
    out = optimize_energy(scenario, directives)
    assert_valid(scenario, out, directives)
    assert all(b_signed(out["hourly_plan"][h]) <= 1e-6 for h in (11, 12))
    assert all(b_signed(out["hourly_plan"][h]) >= -1e-6 for h in (17, 18))


def test_very_cheap_early_hours_and_expensive_evening():
    tariff = [5.0] * 24
    tariff[0] = 1.0
    tariff[18] = 20.0
    scenario = make_scenario(tariff=tariff, capacity=15, initial=10, minimum=0, max_charge=5, max_discharge=5)
    out = optimize_energy(scenario, [no_op()])
    assert_valid(scenario, out, [no_op()])
    assert b_signed(out["hourly_plan"][0]) > 0
    assert b_signed(out["hourly_plan"][18]) < 0


def test_zero_solar():
    scenario = make_scenario(solar=0.0)
    out = optimize_energy(scenario, [no_op()])
    assert_valid(scenario, out, [no_op()])
    assert all(row["solar_used_kwh"] == 0 for row in out["hourly_plan"])


def test_high_solar():
    scenario = make_scenario(
        demand=5.0,
        solar=lambda h: 20.0 if h in (11, 12) else 0.0,
        tariff=10.0,
        capacity=20.0,
        initial=5.0,
        minimum=0.0,
        max_charge=5.0,
        max_discharge=5.0,
    )
    out = optimize_energy(scenario, [no_op()])
    assert_valid(scenario, out, [no_op()])
    assert out["hourly_plan"][11]["grid_kwh"] == pytest.approx(0.0, abs=1e-6)
    assert out["hourly_plan"][12]["grid_kwh"] == pytest.approx(0.0, abs=1e-6)
    assert all(row["solar_used_kwh"] <= scenario["hours"][row["hour"]]["solar_kwh"] + 1e-6 for row in out["hourly_plan"])


def test_battery_initially_at_minimum():
    tariff = [5.0] * 24
    tariff[0] = 1.0
    tariff[18] = 20.0
    scenario = make_scenario(tariff=tariff, initial=2, minimum=2, capacity=12)
    out = optimize_energy(scenario, [no_op()])
    assert_valid(scenario, out, [no_op()])
    assert min(row["battery_energy_after_kwh"] for row in out["hourly_plan"]) >= 2 - 1e-6


def test_battery_initially_full():
    tariff = [5.0] * 24
    tariff[0] = 20.0
    tariff[23] = 1.0
    scenario = make_scenario(tariff=tariff, initial=20, minimum=0, capacity=20)
    out = optimize_energy(scenario, [no_op()])
    assert_valid(scenario, out, [no_op()])
    assert max(row["battery_energy_after_kwh"] for row in out["hourly_plan"]) <= 20 + 1e-6


def test_end_of_day_neutrality():
    tariff = [1.0] + [10.0] * 23
    scenario = make_scenario(tariff=tariff, initial=7.5, capacity=20, minimum=2.5)
    out = optimize_energy(scenario, [no_op()])
    assert_valid(scenario, out, [no_op()])
    assert out["hourly_plan"][23]["battery_energy_after_kwh"] == pytest.approx(7.5, abs=1e-6)


def test_floating_point_edge_values():
    tariff = [0.31 + 0.001 * h for h in range(24)]
    scenario = make_scenario(
        demand=0.1,
        solar=lambda h: 0.03 if 8 <= h <= 15 else 0.0,
        tariff=tariff,
        capacity=0.3,
        initial=0.15,
        minimum=0.05,
        max_charge=0.07,
        max_discharge=0.06,
    )
    directives = [directive("max_grid_window", {"hours": [12], "max_grid_kwh": 0.2})]
    out = optimize_energy(scenario, directives)
    assert_valid(scenario, out, directives)
    assert out["hourly_plan"][23]["battery_energy_after_kwh"] == pytest.approx(0.15, abs=1e-6)


def test_final_validator_rejects_energy_balance_corruption():
    scenario = make_scenario()
    directives = [no_op()]
    out = optimize_energy(scenario, directives)
    broken = copy.deepcopy(out)
    broken["hourly_plan"][5]["grid_kwh"] += 0.1
    broken["total_grid_kwh"] += 0.1
    broken["total_cost_bdt"] += 0.5
    with pytest.raises(FinalValidationError, match="energy balance"):
        validate_final_response(scenario, broken, directives=directives)


def test_final_validator_rejects_wrong_aggregate():
    scenario = make_scenario()
    directives = [no_op()]
    out = optimize_energy(scenario, directives)
    broken = copy.deepcopy(out)
    broken["total_cost_bdt"] += 0.02
    with pytest.raises(FinalValidationError, match="total_cost_bdt mismatch"):
        validate_final_response(scenario, broken, directives=directives)
