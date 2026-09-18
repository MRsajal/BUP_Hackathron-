"""
directives.py
-------------
Defines the six supported directive types and the Pydantic models
that represent a validated LLM interpretation of an operator note.

This file is the single source of truth for directive shapes.
Both llm_interpreter.py and guardrails.py import from here.
"""

from typing import Optional, List, Literal, Union
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# 1. The six directive type names (exact strings the judge expects)
# ---------------------------------------------------------------------------
DIRECTIVE_TYPES = [
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
]


# ---------------------------------------------------------------------------
# 2. Structured adjustment shapes (one per directive type)
#
#    Each directive that "applies" must carry a structured_adjustment object
#    whose fields match EXACTLY what the problem statement specifies.
#    no_op is the only directive where structured_adjustment is null.
# ---------------------------------------------------------------------------

class SolarReductionAdjustment(BaseModel):
    """Reduce usable solar to `factor` fraction during listed hours."""
    hours: List[int] = Field(..., description="Unique sorted integers 0..23")
    factor: float = Field(..., ge=0.0, le=1.0, description="Fraction remaining, e.g. 0.2 = 20% remains")


class MinimumBatteryReserveAdjustment(BaseModel):
    """Keep battery energy at or above minimum_energy_kwh during listed hours."""
    hours: List[int] = Field(..., description="Unique sorted integers 0..23")
    minimum_energy_kwh: float = Field(..., ge=0.0)


class NoChargeWindowAdjustment(BaseModel):
    """Battery charging is disallowed during listed hours."""
    hours: List[int] = Field(..., description="Unique sorted integers 0..23")


class NoDischargeWindowAdjustment(BaseModel):
    """Battery discharging is disallowed during listed hours."""
    hours: List[int] = Field(..., description="Unique sorted integers 0..23")


class MaxGridWindowAdjustment(BaseModel):
    """Grid import must not exceed max_grid_kwh during listed hours."""
    hours: List[int] = Field(..., description="Unique sorted integers 0..23")
    max_grid_kwh: float = Field(..., ge=0.0)


# ---------------------------------------------------------------------------
# 3. The response entry the judge scores against.
#
#    - note_index: which operator_note this entry corresponds to (0-based)
#    - applies: true for every real directive; false ONLY for no_op
#    - directive_type: one of the six exact strings above
#    - structured_adjustment: shape must match directive_type, or null for no_op
#    - explanation: short human-readable text (not byte-matched by the judge)
# ---------------------------------------------------------------------------
class DirectiveInterpretation(BaseModel):
    note_index: int = Field(..., ge=0)
    applies: bool
    directive_type: str
    structured_adjustment: Optional[dict] = None
    explanation: str = ""

    @field_validator("directive_type")
    @classmethod
    def check_type(cls, v: str) -> str:
        if v not in DIRECTIVE_TYPES:
            raise ValueError(f"directive_type must be one of {DIRECTIVE_TYPES}, got {v!r}")
        return v


# ---------------------------------------------------------------------------
# 4. Convenience: map directive_type -> adjustment model class.
#    guardrails.py uses this to pick the right validator per directive.
# ---------------------------------------------------------------------------
ADJUSTMENT_MODELS = {
    "solar_reduction": SolarReductionAdjustment,
    "minimum_battery_reserve": MinimumBatteryReserveAdjustment,
    "no_charge_window": NoChargeWindowAdjustment,
    "no_discharge_window": NoDischargeWindowAdjustment,
    "max_grid_window": MaxGridWindowAdjustment,
    # no_op: no adjustment model — must be null
}