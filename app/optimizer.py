"""Mathematical optimizer for the BUP CSE FEST 2026 GridWise preliminary.

Canonical LP variables for each hour h:
    b[h] > 0  -> charge
    b[h] < 0  -> discharge
    b[h] == 0 -> idle

    E[h] = E[h-1] + b[h]
    grid[h] + solar_used[h] = demand[h] + b[h]

The objective is exactly the total grid-electricity cost.  No battery efficiency,
degradation, export, demand shifting, or other unstated feature is modeled.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linprog

from .final_validator import FinalValidationError, build_replay_directives, validate_final_response


HOURS = 24
CLEAN_EPSILON = 1e-9
OUTPUT_DECIMALS = 8


class OptimizationError(RuntimeError):
    """Controlled internal error for infeasible/failed optimization."""


class InvalidOptimizationInput(ValueError):
    """Raised when optimizer input is structurally or numerically invalid."""


def _finite(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise InvalidOptimizationInput(f"{label} must be a finite number")
    return float(value)


def _ordered_hours(scenario: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = scenario.get("hours")
    if not isinstance(raw, list) or len(raw) != HOURS:
        raise InvalidOptimizationInput("scenario.hours must contain exactly 24 entries")

    by_hour: dict[int, Mapping[str, Any]] = {}
    for entry in raw:
        if not isinstance(entry, Mapping):
            raise InvalidOptimizationInput("every scenario hour must be an object")
        hour = entry.get("hour")
        if not isinstance(hour, int) or isinstance(hour, bool) or not 0 <= hour < HOURS:
            raise InvalidOptimizationInput("scenario hour values must be integers 0..23")
        if hour in by_hour:
            raise InvalidOptimizationInput(f"duplicate scenario hour {hour}")
        by_hour[hour] = entry

    if set(by_hour) != set(range(HOURS)):
        raise InvalidOptimizationInput("scenario hours must cover 0..23 exactly")
    return [by_hour[h] for h in range(HOURS)]


def _clean(value: float) -> float:
    """Remove harmless solver noise and emit stable precision."""
    if abs(value) <= CLEAN_EPSILON:
        return 0.0
    rounded = round(float(value), OUTPUT_DECIMALS)
    if abs(rounded) <= CLEAN_EPSILON:
        return 0.0
    return rounded


def _validate_base_scenario(scenario: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    hours = _ordered_hours(scenario)
    battery = scenario.get("battery")
    if not isinstance(battery, Mapping):
        raise InvalidOptimizationInput("scenario.battery must be an object")

    capacity = _finite(battery.get("capacity_kwh"), "battery.capacity_kwh")
    initial = _finite(battery.get("initial_energy_kwh"), "battery.initial_energy_kwh")
    minimum = _finite(battery.get("minimum_energy_kwh"), "battery.minimum_energy_kwh")
    max_charge = _finite(battery.get("max_charge_kwh_per_hour"), "battery.max_charge_kwh_per_hour")
    max_discharge = _finite(
        battery.get("max_discharge_kwh_per_hour"),
        "battery.max_discharge_kwh_per_hour",
    )

    if capacity < 0 or minimum < 0 or max_charge < 0 or max_discharge < 0:
        raise InvalidOptimizationInput("battery capacity, minimum, and rate limits must be non-negative")
    if minimum > capacity:
        raise InvalidOptimizationInput("battery.minimum_energy_kwh cannot exceed capacity_kwh")
    if initial < minimum or initial > capacity:
        raise InvalidOptimizationInput("battery.initial_energy_kwh must be within [minimum, capacity]")

    for h, entry in enumerate(hours):
        demand = _finite(entry.get("demand_kwh"), f"hours[{h}].demand_kwh")
        solar = _finite(entry.get("solar_kwh"), f"hours[{h}].solar_kwh")
        _finite(entry.get("tariff_bdt_per_kwh"), f"hours[{h}].tariff_bdt_per_kwh")
        if demand < 0 or solar < 0:
            raise InvalidOptimizationInput(f"hours[{h}] demand_kwh and solar_kwh must be non-negative")

    return hours, battery


def optimize_energy(
    scenario: Mapping[str, Any],
    directive_interpretation: Sequence[Mapping[str, Any]],
    *,
    validate: bool = True,
) -> dict[str, Any]:
    """Solve the 24-hour GridWise LP and return a complete response payload.

    ``directive_interpretation`` is expected to have already passed the LLM
    guardrail layer.  This function still refuses malformed/unsupported directive
    data rather than silently changing the mathematical model.

    If ``validate`` is true (default), the result is independently replayed by
    :func:`app.final_validator.validate_final_response` before being returned.
    """

    hours, battery = _validate_base_scenario(scenario)

    try:
        replay = build_replay_directives(scenario, list(directive_interpretation))
    except (TypeError, ValueError) as exc:
        raise InvalidOptimizationInput(f"invalid directive interpretation: {exc}") from exc

    capacity = float(battery["capacity_kwh"])
    initial = float(battery["initial_energy_kwh"])
    max_charge = float(battery["max_charge_kwh_per_hour"])
    max_discharge = float(battery["max_discharge_kwh_per_hour"])

    demand = np.array([float(entry["demand_kwh"]) for entry in hours], dtype=float)
    tariff = np.array([float(entry["tariff_bdt_per_kwh"]) for entry in hours], dtype=float)

    # Variable layout: b[0:24], grid[24:48], solar[48:72], E[72:96]
    b0 = 0
    g0 = HOURS
    s0 = 2 * HOURS
    e0 = 3 * HOURS
    nvars = 4 * HOURS

    objective = np.zeros(nvars, dtype=float)
    objective[g0 : g0 + HOURS] = tariff

    bounds: list[tuple[float | None, float | None]] = []

    # Signed battery-flow bounds, adjusted by no-charge/no-discharge directives.
    for h in range(HOURS):
        lower = -max_discharge
        upper = max_charge
        if h in replay.no_charge:
            upper = min(upper, 0.0)
        if h in replay.no_discharge:
            lower = max(lower, 0.0)
        bounds.append((lower, upper))

    # Grid bounds, adjusted by max-grid directives.
    for h in range(HOURS):
        bounds.append((0.0, replay.max_grid[h]))

    # Solar-used bounds after solar-reduction directives.
    for h in range(HOURS):
        bounds.append((0.0, replay.effective_solar[h]))

    # Battery energy-after bounds, including hour-specific reserve directives.
    for h in range(HOURS):
        bounds.append((replay.minimum_energy[h], capacity))

    a_eq: list[np.ndarray] = []
    b_eq: list[float] = []

    # Energy balance: grid[h] + solar_used[h] - b[h] = demand[h].
    for h in range(HOURS):
        row = np.zeros(nvars, dtype=float)
        row[b0 + h] = -1.0
        row[g0 + h] = 1.0
        row[s0 + h] = 1.0
        a_eq.append(row)
        b_eq.append(demand[h])

    # Battery state transition.
    # Hour 0: E[0] - b[0] = initial_energy.
    row = np.zeros(nvars, dtype=float)
    row[e0] = 1.0
    row[b0] = -1.0
    a_eq.append(row)
    b_eq.append(initial)

    # Hours 1..23: E[h] - E[h-1] - b[h] = 0.
    for h in range(1, HOURS):
        row = np.zeros(nvars, dtype=float)
        row[e0 + h] = 1.0
        row[e0 + h - 1] = -1.0
        row[b0 + h] = -1.0
        a_eq.append(row)
        b_eq.append(0.0)

    # Mandatory end-of-day battery neutrality: E[23] = initial_energy.
    row = np.zeros(nvars, dtype=float)
    row[e0 + HOURS - 1] = 1.0
    a_eq.append(row)
    b_eq.append(initial)

    result = linprog(
        c=objective,
        A_eq=np.vstack(a_eq),
        b_eq=np.asarray(b_eq, dtype=float),
        bounds=bounds,
        method="highs",
        options={"presolve": True},
    )

    if not result.success or result.x is None:
        status = getattr(result, "status", "unknown")
        message = getattr(result, "message", "solver failed")
        raise OptimizationError(f"HiGHS optimization failed (status={status}): {message}")

    raw_b = result.x[b0 : b0 + HOURS]
    raw_solar = result.x[s0 : s0 + HOURS]

    # Clean only tiny numerical noise.  Then reconstruct state and grid from the
    # canonical equations so the returned JSON remains internally self-consistent.
    signed_battery = np.array([_clean(v) for v in raw_b], dtype=float)
    solar_used = np.array([_clean(v) for v in raw_solar], dtype=float)

    energy_after = np.zeros(HOURS, dtype=float)
    running_energy = initial
    for h in range(HOURS):
        running_energy = _clean(running_energy + signed_battery[h])
        energy_after[h] = running_energy

    grid = np.zeros(HOURS, dtype=float)
    for h in range(HOURS):
        grid[h] = _clean(demand[h] + signed_battery[h] - solar_used[h])
        if grid[h] < 0 and abs(grid[h]) <= CLEAN_EPSILON:
            grid[h] = 0.0

    hourly_plan: list[dict[str, Any]] = []
    for h in range(HOURS):
        b = signed_battery[h]
        if b > CLEAN_EPSILON:
            action = "charge"
            magnitude = _clean(b)
        elif b < -CLEAN_EPSILON:
            action = "discharge"
            magnitude = _clean(-b)
        else:
            action = "idle"
            magnitude = 0.0

        hourly_plan.append(
            {
                "hour": h,
                "grid_kwh": _clean(grid[h]),
                "solar_used_kwh": _clean(solar_used[h]),
                "battery_action": action,
                "battery_kwh": magnitude,
                "battery_energy_after_kwh": _clean(energy_after[h]),
            }
        )

    # Aggregates are deliberately recalculated from the final returned plan,
    # never copied from result.fun.
    final_grid = [float(row["grid_kwh"]) for row in hourly_plan]
    total_grid = _clean(sum(final_grid))
    total_cost = _clean(sum(final_grid[h] * tariff[h] for h in range(HOURS)))
    peak_grid = _clean(max(final_grid))

    response: dict[str, Any] = {
        "scenario_id": scenario.get("scenario_id"),
        "directive_interpretation": copy.deepcopy(list(directive_interpretation)),
        "hourly_plan": hourly_plan,
        "total_grid_kwh": total_grid,
        "total_cost_bdt": total_cost,
        "peak_grid_kwh": peak_grid,
        "plan_summary": "24-hour least-cost grid schedule satisfying battery, solar, and validated operator directives.",
    }

    if validate:
        try:
            validate_final_response(
                scenario,
                response,
                directives=directive_interpretation,
            )
        except FinalValidationError as exc:
            # Keep this as a controlled internal optimizer failure so an API layer
            # can map it to HTTP 500 without exposing a raw traceback.
            raise OptimizationError(str(exc)) from exc

    return response
