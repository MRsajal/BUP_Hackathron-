"""Independent replay validator for GridWise optimization responses.

This module intentionally does not depend on the LP implementation.  It replays the
returned 24-hour plan from the initial battery energy and checks every canonical
energy, battery, directive, and aggregate rule from the problem statement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


DEFAULT_TOLERANCE = 1e-6
SUPPORTED_DIRECTIVES = {
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
}


class FinalValidationError(RuntimeError):
    """Controlled internal error raised when a produced plan fails replay validation."""

    def __init__(self, errors: Sequence[str]):
        self.errors = list(errors)
        preview = "; ".join(self.errors[:8])
        if len(self.errors) > 8:
            preview += f"; ... ({len(self.errors) - 8} more)"
        super().__init__(f"Final plan validation failed: {preview}")


@dataclass(frozen=True)
class ReplayDirectives:
    effective_solar: tuple[float, ...]
    minimum_energy: tuple[float, ...]
    no_charge: frozenset[int]
    no_discharge: frozenset[int]
    max_grid: tuple[float | None, ...]


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _as_float(value: Any, label: str) -> float:
    if not _is_finite_number(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _extract_hours(scenario: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    hours = scenario.get("hours")
    if not isinstance(hours, list) or len(hours) != 24:
        raise ValueError("scenario.hours must contain exactly 24 entries")

    by_hour: dict[int, Mapping[str, Any]] = {}
    for entry in hours:
        if not isinstance(entry, Mapping):
            raise ValueError("every scenario hour entry must be an object")
        hour = entry.get("hour")
        if not isinstance(hour, int) or isinstance(hour, bool) or not 0 <= hour <= 23:
            raise ValueError("scenario hour values must be unique integers 0..23")
        if hour in by_hour:
            raise ValueError(f"duplicate scenario hour {hour}")
        by_hour[hour] = entry

    if set(by_hour) != set(range(24)):
        raise ValueError("scenario hours must cover 0..23 exactly")
    return [by_hour[h] for h in range(24)]


def _directive_hours(adj: Mapping[str, Any], label: str) -> list[int]:
    hours = adj.get("hours")
    if not isinstance(hours, list) or not hours:
        raise ValueError(f"{label}.hours must be a non-empty list")
    if any(not isinstance(h, int) or isinstance(h, bool) or not 0 <= h <= 23 for h in hours):
        raise ValueError(f"{label}.hours must contain integers 0..23")
    if hours != sorted(set(hours)):
        raise ValueError(f"{label}.hours must be unique and ascending")
    return hours


def build_replay_directives(
    scenario: Mapping[str, Any],
    directives: Sequence[Mapping[str, Any]],
) -> ReplayDirectives:
    """Convert validated directive interpretations into replay constraints.

    Overlapping directives are combined conservatively and deterministically:
    reserve requirements use the highest minimum, grid caps use the smallest cap,
    and overlapping solar reductions use the smallest remaining-solar factor.
    This makes every overlapping hard restriction hold simultaneously.
    """

    hours = _extract_hours(scenario)
    battery = scenario.get("battery")
    if not isinstance(battery, Mapping):
        raise ValueError("scenario.battery must be an object")

    capacity = _as_float(battery.get("capacity_kwh"), "battery.capacity_kwh")
    base_min = _as_float(battery.get("minimum_energy_kwh"), "battery.minimum_energy_kwh")
    if capacity < 0 or base_min < 0 or base_min > capacity:
        raise ValueError("invalid battery capacity/minimum energy")

    original_solar = []
    for h, entry in enumerate(hours):
        solar = _as_float(entry.get("solar_kwh"), f"hours[{h}].solar_kwh")
        if solar < 0:
            raise ValueError(f"hours[{h}].solar_kwh must be non-negative")
        original_solar.append(solar)

    solar_factor = [1.0] * 24
    minimum_energy = [base_min] * 24
    no_charge: set[int] = set()
    no_discharge: set[int] = set()
    max_grid: list[float | None] = [None] * 24

    for i, directive in enumerate(directives):
        if not isinstance(directive, Mapping):
            raise ValueError(f"directive[{i}] must be an object")
        directive_type = directive.get("directive_type")
        if directive_type not in SUPPORTED_DIRECTIVES:
            raise ValueError(f"directive[{i}] has unsupported directive_type {directive_type!r}")

        applies = directive.get("applies")
        adj = directive.get("structured_adjustment")

        if directive_type == "no_op":
            if applies is not False or adj is not None:
                raise ValueError(f"directive[{i}] invalid no_op semantics")
            continue

        if applies is not True or not isinstance(adj, Mapping):
            raise ValueError(f"directive[{i}] must have applies=true and an adjustment object")

        listed_hours = _directive_hours(adj, f"directive[{i}]")

        if directive_type == "solar_reduction":
            factor = _as_float(adj.get("factor"), f"directive[{i}].factor")
            if not 0 <= factor <= 1:
                raise ValueError(f"directive[{i}].factor must be in [0, 1]")
            for h in listed_hours:
                solar_factor[h] = min(solar_factor[h], factor)

        elif directive_type == "minimum_battery_reserve":
            reserve = _as_float(
                adj.get("minimum_energy_kwh"),
                f"directive[{i}].minimum_energy_kwh",
            )
            if reserve < 0 or reserve > capacity:
                raise ValueError(f"directive[{i}] reserve must be between 0 and capacity")
            for h in listed_hours:
                minimum_energy[h] = max(minimum_energy[h], reserve)

        elif directive_type == "no_charge_window":
            no_charge.update(listed_hours)

        elif directive_type == "no_discharge_window":
            no_discharge.update(listed_hours)

        elif directive_type == "max_grid_window":
            cap = _as_float(adj.get("max_grid_kwh"), f"directive[{i}].max_grid_kwh")
            if cap < 0:
                raise ValueError(f"directive[{i}].max_grid_kwh must be non-negative")
            for h in listed_hours:
                max_grid[h] = cap if max_grid[h] is None else min(max_grid[h], cap)

    effective_solar = tuple(original_solar[h] * solar_factor[h] for h in range(24))
    return ReplayDirectives(
        effective_solar=effective_solar,
        minimum_energy=tuple(minimum_energy),
        no_charge=frozenset(no_charge),
        no_discharge=frozenset(no_discharge),
        max_grid=tuple(max_grid),
    )


def validate_final_response(
    scenario: Mapping[str, Any],
    response: Mapping[str, Any],
    *,
    directives: Sequence[Mapping[str, Any]] | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> None:
    """Replay and validate a final GridWise response.

    Raises:
        FinalValidationError: if any returned-plan or aggregate rule is violated.

    The function returns ``None`` on success so callers can use it as a hard gate
    before returning HTTP 200.
    """

    if tolerance <= 0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be a positive finite number")

    errors: list[str] = []

    try:
        scenario_hours = _extract_hours(scenario)
        battery = scenario.get("battery")
        if not isinstance(battery, Mapping):
            raise ValueError("scenario.battery must be an object")
        capacity = _as_float(battery.get("capacity_kwh"), "battery.capacity_kwh")
        initial_energy = _as_float(battery.get("initial_energy_kwh"), "battery.initial_energy_kwh")
        base_minimum = _as_float(battery.get("minimum_energy_kwh"), "battery.minimum_energy_kwh")
        max_charge = _as_float(battery.get("max_charge_kwh_per_hour"), "battery.max_charge_kwh_per_hour")
        max_discharge = _as_float(
            battery.get("max_discharge_kwh_per_hour"),
            "battery.max_discharge_kwh_per_hour",
        )

        replay = build_replay_directives(
            scenario,
            list(directives) if directives is not None else list(response.get("directive_interpretation", [])),
        )
    except (TypeError, ValueError) as exc:
        raise FinalValidationError([f"cannot replay scenario/directives: {exc}"]) from exc

    if capacity < -tolerance or base_minimum < -tolerance or base_minimum > capacity + tolerance:
        errors.append("invalid base battery bounds")
    if initial_energy < base_minimum - tolerance or initial_energy > capacity + tolerance:
        errors.append("initial battery energy is outside base battery bounds")
    if max_charge < -tolerance or max_discharge < -tolerance:
        errors.append("battery rate limits must be non-negative")

    if "scenario_id" in response and response.get("scenario_id") != scenario.get("scenario_id"):
        errors.append("response scenario_id does not match request scenario_id")

    plan = response.get("hourly_plan")
    if not isinstance(plan, list) or len(plan) != 24:
        errors.append("hourly_plan must contain exactly 24 entries")
        raise FinalValidationError(errors)

    plan_by_hour: dict[int, Mapping[str, Any]] = {}
    for idx, entry in enumerate(plan):
        if not isinstance(entry, Mapping):
            errors.append(f"hourly_plan[{idx}] must be an object")
            continue
        hour = entry.get("hour")
        if not isinstance(hour, int) or isinstance(hour, bool) or not 0 <= hour <= 23:
            errors.append(f"hourly_plan[{idx}].hour must be an integer 0..23")
            continue
        if hour in plan_by_hour:
            errors.append(f"duplicate hourly_plan hour {hour}")
            continue
        plan_by_hour[hour] = entry

    if set(plan_by_hour) != set(range(24)):
        errors.append("hourly_plan hours must cover 0..23 exactly")
    if errors:
        raise FinalValidationError(errors)

    previous_energy = initial_energy
    grid_values: list[float] = []
    tariffs: list[float] = []

    for h in range(24):
        entry = plan_by_hour[h]
        source = scenario_hours[h]

        try:
            demand = _as_float(source.get("demand_kwh"), f"scenario hour {h} demand_kwh")
            tariff = _as_float(source.get("tariff_bdt_per_kwh"), f"scenario hour {h} tariff")
            grid = _as_float(entry.get("grid_kwh"), f"hour {h} grid_kwh")
            solar_used = _as_float(entry.get("solar_used_kwh"), f"hour {h} solar_used_kwh")
            battery_kwh = _as_float(entry.get("battery_kwh"), f"hour {h} battery_kwh")
            energy_after = _as_float(
                entry.get("battery_energy_after_kwh"),
                f"hour {h} battery_energy_after_kwh",
            )
        except ValueError as exc:
            errors.append(str(exc))
            continue

        action = entry.get("battery_action")
        if action not in {"charge", "discharge", "idle"}:
            errors.append(f"hour {h}: invalid battery_action {action!r}")
            signed_battery = 0.0
        elif action == "charge":
            if battery_kwh <= tolerance:
                errors.append(f"hour {h}: charge action must have positive battery_kwh")
            signed_battery = battery_kwh
        elif action == "discharge":
            if battery_kwh <= tolerance:
                errors.append(f"hour {h}: discharge action must have positive battery_kwh")
            signed_battery = -battery_kwh
        else:
            if abs(battery_kwh) > tolerance:
                errors.append(f"hour {h}: idle action must have battery_kwh=0")
            signed_battery = 0.0

        # Required returned numeric values are non-negative except the signed flow,
        # which is represented by action + non-negative magnitude.
        if grid < -tolerance:
            errors.append(f"hour {h}: grid_kwh is negative")
        if solar_used < -tolerance:
            errors.append(f"hour {h}: solar_used_kwh is negative")
        if battery_kwh < -tolerance:
            errors.append(f"hour {h}: battery_kwh magnitude is negative")
        if energy_after < -tolerance:
            errors.append(f"hour {h}: battery_energy_after_kwh is negative")

        if solar_used > replay.effective_solar[h] + tolerance:
            errors.append(
                f"hour {h}: solar_used_kwh {solar_used} exceeds effective solar {replay.effective_solar[h]}"
            )

        if signed_battery > max_charge + tolerance:
            errors.append(f"hour {h}: battery charge rate exceeds max_charge_kwh_per_hour")
        if -signed_battery > max_discharge + tolerance:
            errors.append(f"hour {h}: battery discharge rate exceeds max_discharge_kwh_per_hour")

        expected_energy = previous_energy + signed_battery
        if abs(energy_after - expected_energy) > tolerance:
            errors.append(
                f"hour {h}: battery state transition mismatch; expected {expected_energy}, got {energy_after}"
            )

        if energy_after < base_minimum - tolerance:
            errors.append(f"hour {h}: battery energy below base minimum")
        if energy_after < replay.minimum_energy[h] - tolerance:
            errors.append(
                f"hour {h}: battery energy below active reserve {replay.minimum_energy[h]}"
            )
        if energy_after > capacity + tolerance:
            errors.append(f"hour {h}: battery energy exceeds capacity")

        if h in replay.no_charge and signed_battery > tolerance:
            errors.append(f"hour {h}: no_charge_window violated")
        if h in replay.no_discharge and signed_battery < -tolerance:
            errors.append(f"hour {h}: no_discharge_window violated")

        cap = replay.max_grid[h]
        if cap is not None and grid > cap + tolerance:
            errors.append(f"hour {h}: max_grid_window violated ({grid} > {cap})")

        # signed_battery > 0 is charge, signed_battery < 0 is discharge.
        # Canonical balance: grid + solar = demand + signed_battery.
        balance_residual = grid + solar_used - demand - signed_battery
        if abs(balance_residual) > tolerance:
            errors.append(
                f"hour {h}: energy balance residual {balance_residual} exceeds tolerance"
            )

        previous_energy = energy_after
        grid_values.append(max(0.0, grid) if abs(grid) <= tolerance else grid)
        tariffs.append(tariff)

    if len(grid_values) == 24:
        if abs(previous_energy - initial_energy) > tolerance:
            errors.append(
                f"end-of-day battery neutrality violated: final {previous_energy}, initial {initial_energy}"
            )

        recalculated_total_grid = sum(grid_values)
        recalculated_total_cost = sum(grid_values[h] * tariffs[h] for h in range(24))
        recalculated_peak_grid = max(grid_values)

        for field, expected in (
            ("total_grid_kwh", recalculated_total_grid),
            ("total_cost_bdt", recalculated_total_cost),
            ("peak_grid_kwh", recalculated_peak_grid),
        ):
            value = response.get(field)
            if not _is_finite_number(value):
                errors.append(f"{field} must be a finite number")
            elif abs(float(value) - expected) > tolerance:
                errors.append(f"{field} mismatch: reported {value}, recalculated {expected}")

    if errors:
        raise FinalValidationError(errors)
