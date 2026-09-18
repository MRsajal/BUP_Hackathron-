# Developer 3 - Optimization & Final Validation Report

## Mathematical formulation

For each hour h = 0..23, the LP uses four variables:

- `b[h]`: signed battery flow; positive = charge, negative = discharge.
- `grid[h]`: non-negative grid import.
- `solar_used[h]`: used solar after operator-directed reductions.
- `E[h]`: battery energy after hour h.

Primary objective:

`min sum(grid[h] * tariff[h])`

Equalities:

- `grid[h] + solar_used[h] - b[h] = demand[h]`
- `E[0] - b[0] = initial_energy`
- `E[h] - E[h-1] - b[h] = 0` for h = 1..23
- `E[23] = initial_energy`

Bounds enforce charge/discharge rates, non-negative grid, effective solar, base/hour-specific battery reserve, battery capacity, no-charge/no-discharge windows, and max-grid windows.

No battery efficiency, degradation, export, export income, demand shifting, or charging penalty was added.

## Solver

- `scipy.optimize.linprog`
- HiGHS backend (`method="highs"`)

## Files

- `app/optimizer.py`
- `app/final_validator.py`
- `tests/test_optimizer.py`
- `tests/test_public_samples.py`
- `tests/conftest.py`
- `tests/BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json` (organizer public sample pack copied for reproducible tests)
- `sample_costs.txt`

## Validation design

The final validator is independent of the LP solver and replays the returned plan hour by hour from the initial battery energy. It verifies:

- exactly 24 unique hours 0..23
- finite/non-negative returned energy quantities
- effective-solar upper bounds
- action/magnitude consistency
- charge/discharge rate limits
- state transitions
- base and directive reserve bounds
- capacity
- no-charge/no-discharge windows
- max-grid windows
- hourly energy balance
- end-of-day battery neutrality
- total grid, total cost, and peak-grid aggregates

Internal replay tolerance: `1e-6`, stricter than the official `0.01` tolerance.

## Tests executed

- `pytest -q`: 28 passed
- `python -m py_compile`: passed
- 200 randomized feasible stress scenarios: passed optimization + replay validation

Covered requested cases: baseline, solar reduction, minimum reserve, no charge, no discharge, grid cap, multiple directives, reserve + grid cap, separate charge/discharge windows, cheap early vs expensive evening, zero solar, high solar, battery initially at minimum, battery initially full, end-of-day neutrality, and floating-point edge values. Additional negative tests confirm replay rejection for energy-balance corruption and wrong aggregates.

## Public sample costs

All 10 public sample costs exactly match the organizer reference costs:

| Case | Obtained cost (BDT) | Reference cost (BDT) |
|---|---:|---:|
| SAMPLE-01 | 38,365.00 | 38,365.00 |
| SAMPLE-02 | 42,885.00 | 42,885.00 |
| SAMPLE-03 | 35,480.00 | 35,480.00 |
| SAMPLE-04 | 40,495.00 | 40,495.00 |
| SAMPLE-05 | 33,950.00 | 33,950.00 |
| SAMPLE-06 | 34,090.00 | 34,090.00 |
| SAMPLE-07 | 38,550.00 | 38,550.00 |
| SAMPLE-08 | 37,665.00 | 37,665.00 |
| SAMPLE-09 | 34,873.00 | 34,873.00 |
| SAMPLE-10 | 41,620.00 | 41,620.00 |

`total_grid_kwh` also matches all 10 reference outputs. Peak grid differs on SAMPLE-01 and SAMPLE-09 because the cost LP has multiple equally optimal schedules; peak grid is not part of the canonical objective. The returned peaks are still valid and are independently recalculated from the returned plan.

## Numerical / feasibility risks

1. HiGHS can return tiny values around zero. Values below `1e-9` are clamped to zero and output is kept to 8 decimal places.
2. The validator uses `1e-6`; this is deliberately stricter than the judge's documented `0.01` tolerance but still comfortable for HiGHS on this small LP.
3. If validated directives make a scenario infeasible, `OptimizationError` is raised. A caller/API should translate it into a controlled internal error rather than return HTTP 200.
4. Equivalent cost-optimal schedules can differ in hourly battery movement and peak grid. Tests therefore compare public optimal cost, not exact hourly-plan equality.
5. Overlapping reserve directives use the highest reserve; overlapping grid caps use the smallest cap; overlapping solar reductions use the smallest remaining-solar factor so all simultaneous restrictions are respected.
