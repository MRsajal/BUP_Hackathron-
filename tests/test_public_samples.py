from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.final_validator import validate_final_response
from app.optimizer import optimize_energy


SAMPLE_FILE = Path(__file__).with_name("BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json")
OFFICIAL_TOLERANCE = 0.01


def _cases():
    data = json.loads(SAMPLE_FILE.read_text(encoding="utf-8"))
    return data["cases"]


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["input"]["scenario_id"])
def test_public_sample_optimal_cost_and_validity(case):
    scenario = case["input"]
    expected = case["expected_output"]
    directives = expected["directive_interpretation"]

    result = optimize_energy(scenario, directives)

    # Independent replay covers directive compliance, all canonical constraints,
    # end-of-day neutrality, and the three reported aggregate values.
    validate_final_response(scenario, result, directives=directives)

    # Equivalent optimal schedules are explicitly allowed; do not compare the
    # 24 hourly actions or peak grid byte-for-byte with the reference schedule.
    assert result["total_cost_bdt"] == pytest.approx(
        expected["total_cost_bdt"], abs=OFFICIAL_TOLERANCE
    )
