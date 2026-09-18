"""
tests/test_public_samples.py
----------------------------
Runs the 10 public sample cases from public_samples.json through the
full LLM interpreter + guardrails pipeline and reports how many the
interpretation part gets right.

This uses REAL LLM calls. Run it sparingly.

Usage:
    python -m tests.test_public_samples
"""

import json
import asyncio
import time
from pathlib import Path

from app.llm_interpreter import interpret_notes
from app.guardrails import validate_directives


# ---------------------------------------------------------------------------
# Load the samples
# ---------------------------------------------------------------------------
SAMPLES_PATH = Path(__file__).resolve().parent.parent / "public_samples.json"


def _load_samples():
    with SAMPLES_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # The public samples file may be a list of cases OR wrap them under a key.
    if isinstance(data, list):
        return data
    for key in ("cases", "samples", "public_cases"):
        if key in data:
            return data[key]
    # Fallback: treat any top-level list-valued field as cases.
    for v in data.values():
        if isinstance(v, list):
            return v
    raise RuntimeError("Could not locate a list of cases in public_samples.json")


# ---------------------------------------------------------------------------
# Compare our interpretation against expected (for one entry)
# ---------------------------------------------------------------------------
def _compare_entry(ours, expected):
    """
    Returns (passed: bool, reason: str).
    - directive_type must match exactly
    - applies must match exactly
    - if applies=True, hours must be equal set, and numeric values close.
    - explanation text is not compared.
    """
    if ours.directive_type != expected.get("directive_type"):
        return False, f"directive_type mismatch: ours={ours.directive_type} expected={expected.get('directive_type')}"
    if ours.applies != expected.get("applies"):
        return False, f"applies mismatch: ours={ours.applies} expected={expected.get('applies')}"

    if not ours.applies:
        return True, "no_op OK"

    ours_adj = ours.structured_adjustment or {}
    exp_adj = expected.get("structured_adjustment") or {}

    ours_hours = sorted(ours_adj.get("hours", []))
    exp_hours = sorted(exp_adj.get("hours", []))
    if ours_hours != exp_hours:
        return False, f"hours mismatch: ours={ours_hours} expected={exp_hours}"

    # Compare numeric values if present
    for numeric_field in ("factor", "minimum_energy_kwh", "max_grid_kwh"):
        if numeric_field in exp_adj:
            ours_v = ours_adj.get(numeric_field)
            exp_v = exp_adj.get(numeric_field)
            if ours_v is None or abs(float(ours_v) - float(exp_v)) > 0.01:
                return False, f"{numeric_field} mismatch: ours={ours_v} expected={exp_v}"

    return True, "OK"


# ---------------------------------------------------------------------------
# Discover the "expected" answer for each note in a sample
#
# Each sample file may store expected outputs under one of several keys
# ("expected_directive_interpretation", "expected_response",
# "directive_interpretation", etc.). We try a few common shapes.
# ---------------------------------------------------------------------------
def _extract_expected(sample):
    # Try a bunch of common shapes.
    for key in (
        "expected_directive_interpretation",
        "directive_interpretation",
        "expected",
        "expected_output",
        "reference",
        "answer",
    ):
        v = sample.get(key)
        if isinstance(v, list):
            return v
        if isinstance(v, dict) and "directive_interpretation" in v:
            return v["directive_interpretation"]

    # Also check under "output" / "expected_response"
    for key in ("output", "expected_response"):
        v = sample.get(key)
        if isinstance(v, dict) and "directive_interpretation" in v:
            return v["directive_interpretation"]

    return None


# ---------------------------------------------------------------------------
# Run one sample
# ---------------------------------------------------------------------------
async def _run_one(sample, idx):
    scenario_id = sample.get("id") or sample.get("scenario_id") or f"SAMPLE-{idx+1:02d}"

    # Cases in this file wrap the request under "input".
    input_block = sample.get("input") or sample.get("request") or sample

    request = {
        "operator_notes": input_block.get("operator_notes"),
        "battery": input_block.get("battery"),
    }

    if not request["operator_notes"]:
        return scenario_id, False, "no operator_notes found in sample"
    if not request["battery"]:
        return scenario_id, False, "no battery info found in sample"

    try:
        raw = await interpret_notes(request)
        validated = validate_directives(raw, request)
    except Exception as e:
        return scenario_id, False, f"interpreter/guardrail crashed: {e}"

    expected = _extract_expected(sample)
    if expected is None:
        return scenario_id, None, "no expected answer in sample (informational only)"

    entry_results = []
    for i, (ours, exp) in enumerate(zip(validated, expected)):
        ok, reason = _compare_entry(ours, exp)
        entry_results.append((i, ok, reason))

    all_ok = all(r[1] for r in entry_results)
    if all_ok:
        return scenario_id, True, "all entries match"
    else:
        failures = "; ".join(f"note {r[0]}: {r[2]}" for r in entry_results if not r[1])
        return scenario_id, False, failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    samples = _load_samples()
    print(f"Loaded {len(samples)} public samples from {SAMPLES_PATH.name}\n")

    passed = 0
    failed = 0
    unknown = 0
    start = time.time()

    for idx, sample in enumerate(samples):
        scenario_id, result, reason = await _run_one(sample, idx)
        if result is True:
            print(f"[PASS] {scenario_id}: {reason}")
            passed += 1
        elif result is False:
            print(f"[FAIL] {scenario_id}: {reason}")
            failed += 1
        else:
            print(f"[??  ] {scenario_id}: {reason}")
            unknown += 1

    elapsed = time.time() - start
    print(f"\nResult: {passed} passed, {failed} failed, {unknown} unknown in {elapsed:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())