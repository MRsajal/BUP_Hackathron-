"""
guardrails.py
-------------
Deterministic post-LLM validation of directive interpretations.

The LLM is untrusted. Even when it looks right, hidden test cases may
produce edge outputs (duplicate hours, factor > 1, reserve > capacity,
wrong applies semantics, etc.). This module enforces every rule the
problem statement defines, and normalizes safe issues (sorting/deduping
hours) rather than crashing.

Public function:
    validate_directives(directives, request) -> list[DirectiveInterpretation]
"""

from typing import Any, List

from app.directives import (
    DirectiveInterpretation,
    DIRECTIVE_TYPES,
    ADJUSTMENT_MODELS,
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _get_attr(obj: Any, name: str):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _normalize_hours(hours: Any) -> list[int]:
    """
    Force hours into unique sorted integers 0..23.
    Rejects (raises) if any hour is out of range or not an integer.
    """
    if not isinstance(hours, list) or len(hours) == 0:
        raise ValueError(f"hours must be a non-empty list, got {hours!r}")

    clean: set[int] = set()
    for h in hours:
        # Coerce numeric strings like "13" -> 13, and floats like 13.0 -> 13.
        if isinstance(h, bool):  # bool is a subclass of int; reject explicitly
            raise ValueError(f"hours contains a boolean: {h!r}")
        try:
            hi = int(h)
        except (TypeError, ValueError):
            raise ValueError(f"hour value not an integer: {h!r}")
        if hi != float(h):  # 13.5 would fail this
            raise ValueError(f"hour value not a whole hour: {h!r}")
        if hi < 0 or hi > 23:
            raise ValueError(f"hour out of range 0..23: {hi}")
        clean.add(hi)

    return sorted(clean)


def _force_no_op(note_index: int, reason: str) -> DirectiveInterpretation:
    """
    Build a safe no_op entry. Used only when we DECIDE the note is irrelevant.
    NOTE: guardrails should NOT silently convert a malformed 'applies=true'
    directive into no_op — that would hide real bugs. Use this only for
    entries the LLM itself marked no_op but got a shape detail wrong (e.g.
    included a non-null structured_adjustment).
    """
    return DirectiveInterpretation(
        note_index=note_index,
        applies=False,
        directive_type="no_op",
        structured_adjustment=None,
        explanation=reason,
    )


# ---------------------------------------------------------------------------
# Per-entry validation.
#
# For each entry we enforce:
#   - directive_type is one of the six allowed values
#   - no_op iff applies=False, and structured_adjustment is None
#   - every other directive: applies=True, and structured_adjustment matches
#     the exact shape required by directives.ADJUSTMENT_MODELS
#   - hours arrays: unique sorted integers 0..23
#   - factor in [0, 1]
#   - reserve >= 0, and <= battery capacity
#   - max_grid_kwh >= 0
# ---------------------------------------------------------------------------
def _validate_one(entry: DirectiveInterpretation, battery_capacity_kwh: float) -> DirectiveInterpretation:
    dt = entry.directive_type
    if dt not in DIRECTIVE_TYPES:
        raise ValueError(f"note_index {entry.note_index}: unknown directive_type {dt!r}")

    # --- no_op branch ---
    if dt == "no_op":
        if entry.applies is not False:
            raise ValueError(f"note_index {entry.note_index}: no_op must have applies=false")
        if entry.structured_adjustment is not None:
            # LLM said no_op but stuck data in the adjustment — normalize to null.
            entry = entry.model_copy(update={"structured_adjustment": None})
        return entry

    # --- All other directives ---
    if entry.applies is not True:
        raise ValueError(
            f"note_index {entry.note_index}: non-no_op directive {dt!r} must have applies=true"
        )

    adj = entry.structured_adjustment
    if not isinstance(adj, dict):
        raise ValueError(
            f"note_index {entry.note_index}: structured_adjustment must be an object for {dt!r}"
        )

    # Normalize hours first (all directives except no_op carry hours).
    hours = _normalize_hours(adj.get("hours"))

    if dt == "solar_reduction":
        factor = adj.get("factor")
        if factor is None or not isinstance(factor, (int, float)):
            raise ValueError(f"note_index {entry.note_index}: solar_reduction.factor missing/invalid")
        if factor < 0.0 or factor > 1.0:
            raise ValueError(f"note_index {entry.note_index}: factor out of [0,1]: {factor}")
        new_adj = {"hours": hours, "factor": float(factor)}

    elif dt == "minimum_battery_reserve":
        reserve = adj.get("minimum_energy_kwh")
        if reserve is None or not isinstance(reserve, (int, float)):
            raise ValueError(f"note_index {entry.note_index}: minimum_energy_kwh missing/invalid")
        if reserve < 0:
            raise ValueError(f"note_index {entry.note_index}: reserve is negative: {reserve}")
        if battery_capacity_kwh is not None and reserve > battery_capacity_kwh + 1e-6:
            raise ValueError(
                f"note_index {entry.note_index}: reserve {reserve} exceeds battery capacity {battery_capacity_kwh}"
            )
        new_adj = {"hours": hours, "minimum_energy_kwh": float(reserve)}

    elif dt == "no_charge_window":
        new_adj = {"hours": hours}

    elif dt == "no_discharge_window":
        new_adj = {"hours": hours}

    elif dt == "max_grid_window":
        cap = adj.get("max_grid_kwh")
        if cap is None or not isinstance(cap, (int, float)):
            raise ValueError(f"note_index {entry.note_index}: max_grid_kwh missing/invalid")
        if cap < 0:
            raise ValueError(f"note_index {entry.note_index}: max_grid_kwh is negative: {cap}")
        new_adj = {"hours": hours, "max_grid_kwh": float(cap)}

    else:
        # Defensive; should be unreachable because of DIRECTIVE_TYPES check above.
        raise ValueError(f"note_index {entry.note_index}: unhandled directive_type {dt!r}")

    # Sanity: pass through the type-specific Pydantic model to be sure.
    ADJUSTMENT_MODELS[dt](**new_adj)

    return entry.model_copy(update={"structured_adjustment": new_adj})


# ---------------------------------------------------------------------------
# PUBLIC API — called by Dev 1's service.py after interpret_notes().
# ---------------------------------------------------------------------------
def validate_directives(
    directives: List[DirectiveInterpretation],
    request: Any,
) -> List[DirectiveInterpretation]:
    """
    Validate and normalize a list of DirectiveInterpretation entries.

    - Enforces: exactly one entry per note, note_index values 0..N-1 in order.
    - Normalizes hours arrays (dedup + sort).
    - Rejects invalid factor/reserve/cap values by raising ValueError.
    - Never silently rewrites an applies=true directive into no_op.
    """
    notes = _get_attr(request, "operator_notes") or []
    n = len(notes)

    if len(directives) != n:
        raise ValueError(
            f"Expected {n} directive_interpretation entries, got {len(directives)}"
        )

    # Enforce order and 0..N-1 note_index values.
    seen: set[int] = set()
    for i, d in enumerate(directives):
        if d.note_index != i:
            raise ValueError(
                f"Entry at position {i} has note_index={d.note_index}; must be {i}"
            )
        if d.note_index in seen:
            raise ValueError(f"Duplicate note_index {d.note_index}")
        seen.add(d.note_index)

    battery = _get_attr(request, "battery")
    capacity = _get_attr(battery, "capacity_kwh")

    validated: List[DirectiveInterpretation] = []
    for entry in directives:
        validated.append(_validate_one(entry, capacity))

    return validated