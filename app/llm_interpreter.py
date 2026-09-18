"""
llm_interpreter.py
------------------
Uses a language-capable generative model (Groq / OpenAI-compatible API)
to convert operator_notes into structured directive interpretations.

The LLM is treated as UNTRUSTED. Its output is parsed, then handed to
guardrails.py for deterministic validation before it reaches the optimizer.

Public function:
    interpret_notes(request) -> list[DirectiveInterpretation]
"""

import os
import json
import logging
from typing import Any, List

from dotenv import load_dotenv
from openai import OpenAI

from app.directives import DirectiveInterpretation, DIRECTIVE_TYPES

load_dotenv()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM client — created once per process, not per request.
# base_url lets us point the OpenAI SDK at Groq (or any OpenAI-compatible API).
# ---------------------------------------------------------------------------
_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Lazy-initialize the LLM client from environment variables."""
    global _client
    if _client is None:
        api_key = os.getenv("LLM_API_KEY")
        base_url = os.getenv("LLM_BASE_URL")  # e.g. https://api.groq.com/openai/v1
        if not api_key:
            raise RuntimeError("LLM_API_KEY is not set")
        _client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    return _client


def get_model_name() -> str:
    return os.getenv("LLM_MODEL", "openai/gpt-oss-120b")


# ---------------------------------------------------------------------------
# THE SYSTEM PROMPT
#
# This is the single most important string in the codebase. Every rule here
# maps to something the judge checks. Do NOT loosen anything without reason.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an operator-note interpreter for a smart-campus energy optimization system.
Your ONLY job is to convert 1-3 natural-language operator notes into a structured JSON array.
You do NOT plan energy, schedule batteries, or optimize cost. You only interpret language.

You will receive:
- The operator notes (with 0-based indexes).
- The battery capacity in kWh (needed for percentage-of-capacity reserves).

Return a JSON object of exactly this shape:
{
  "interpretations": [
    {
      "note_index": <int>,
      "applies": <true|false>,
      "directive_type": "<one of the six types>",
      "structured_adjustment": <object or null>,
      "explanation": "<short reason>"
    },
    ...
  ]
}

Rules — every one matters:

1. Return EXACTLY ONE entry per operator note, in note_index order 0, 1, 2, ...
2. The six allowed directive_type values are EXACTLY:
   solar_reduction, minimum_battery_reserve, no_charge_window,
   no_discharge_window, max_grid_window, no_op
3. If a note is irrelevant to today's 24-hour energy schedule (cafeteria menus,
   weather chit-chat, personnel gossip, unrelated announcements), set:
     applies = false
     directive_type = "no_op"
     structured_adjustment = null
4. For every other directive, applies = true and structured_adjustment MUST match:
     solar_reduction:          {"hours": [...], "factor": <0..1>}
     minimum_battery_reserve:  {"hours": [...], "minimum_energy_kwh": <number>}
     no_charge_window:         {"hours": [...]}
     no_discharge_window:      {"hours": [...]}
     max_grid_window:          {"hours": [...], "max_grid_kwh": <number>}

TIME WINDOW RULES (critical):
- Windows are START-INCLUSIVE and END-EXCLUSIVE.
- "1 PM to 3 PM"        -> [13, 14]
- "2 AM until 5 AM"     -> [2, 3, 4]
- "from 18:00 to 21:00" -> [18, 19, 20]
- "between 10 AM and noon" -> [10, 11]
- hours must be UNIQUE integers 0..23, sorted ASCENDING.

PERCENTAGE / FACTOR RULES (critical):
- "factor" for solar_reduction means the FRACTION THAT REMAINS.
- "80% reduction" -> factor = 0.2  (only 20% remains)
- "solar drops to 20%" -> factor = 0.2
- "roughly one-fifth of normal" -> factor = 0.2
- "reduced by half" -> factor = 0.5
- Factor is between 0 and 1.

RESERVE PERCENTAGES:
- "keep at least 50% of battery capacity" with capacity 200 kWh -> minimum_energy_kwh = 100
- Multiply the percentage by the battery capacity provided in the user message.

DO NOT invent demand, tariff, solar values, battery limits, or unsupported directive types.
DO NOT include any commentary outside the JSON object.
Output MUST be valid JSON. Nothing else.
"""


# ---------------------------------------------------------------------------
# Build the user message that carries the notes + battery capacity.
# ---------------------------------------------------------------------------
def _build_user_message(operator_notes: list[str], battery_capacity_kwh: float) -> str:
    notes_block = "\n".join(f"  {i}: {note}" for i, note in enumerate(operator_notes))
    return (
        f"Battery capacity (for percentage-of-capacity reserves): {battery_capacity_kwh} kWh\n\n"
        f"Operator notes (index: text):\n{notes_block}\n\n"
        f"Return the JSON object as instructed. No prose."
    )


# ---------------------------------------------------------------------------
# Parse the LLM's raw text response into a list[DirectiveInterpretation].
# We DO NOT trust the LLM. Anything malformed here raises so the guardrail
# layer or caller can decide what to do (retry once, then fail loudly).
# ---------------------------------------------------------------------------
def _parse_llm_response(raw_text: str, expected_note_count: int) -> List[DirectiveInterpretation]:
    # Some models wrap JSON in ```json ... ```; strip that if present.
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        # Remove leading ``` or ```json line
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
        # Remove trailing ```
        if cleaned.endswith("```"):
            cleaned = cleaned[: -3].strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON: {e}\n---\n{raw_text[:500]}")

    if "interpretations" not in data:
        raise ValueError(f"LLM JSON missing 'interpretations' key. Got keys: {list(data.keys())}")

    items = data["interpretations"]
    if not isinstance(items, list):
        raise ValueError("'interpretations' is not a list")

    if len(items) != expected_note_count:
        raise ValueError(
            f"Expected {expected_note_count} interpretation entries, got {len(items)}"
        )

    # Parse each item through the Pydantic model to catch type/enum errors early.
    parsed: List[DirectiveInterpretation] = []
    for i, item in enumerate(items):
        try:
            parsed.append(DirectiveInterpretation(**item))
        except Exception as e:
            raise ValueError(f"Interpretation entry {i} failed to parse: {e}\n{item}")

    return parsed


# ---------------------------------------------------------------------------
# Single LLM call. Uses JSON response format when the provider supports it.
# ---------------------------------------------------------------------------
def _call_llm(operator_notes: list[str], battery_capacity_kwh: float) -> str:
    client = get_client()
    model = get_model_name()

    response = client.chat.completions.create(
        model=model,
        temperature=0,  # deterministic
        response_format={"type": "json_object"},  # force JSON output
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_message(operator_notes, battery_capacity_kwh)},
        ],
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# PUBLIC API — this is what Dev 1's service.py will import and call.
#
# `request` is a Pydantic OptimizeRequest (Dev 1 owns that model), but we only
# rely on two attributes: operator_notes and battery.capacity_kwh. We access
# them defensively so it also works if a plain dict is passed during tests.
# ---------------------------------------------------------------------------
async def interpret_notes(request: Any) -> List[DirectiveInterpretation]:
    """
    Turn `request.operator_notes` into a list of validated DirectiveInterpretation
    objects. Attempts one repair retry if the first response is malformed.

    Raises RuntimeError on repeated failure — caller should return HTTP 500.
    """
    # Support both Pydantic model and dict inputs.
    notes = _get_attr(request, "operator_notes")
    battery = _get_attr(request, "battery")
    capacity = _get_attr(battery, "capacity_kwh")

    if not notes or not isinstance(notes, list):
        raise ValueError("request.operator_notes missing or not a list")

    # Attempt 1
    try:
        raw = _call_llm(notes, capacity)
        return _parse_llm_response(raw, expected_note_count=len(notes))
    except Exception as first_err:
        logger.warning("LLM interpretation attempt 1 failed: %s", first_err)

    # Attempt 2 — one repair retry with an explicit reminder appended.
    try:
        raw = _call_llm(
            notes + ["[system: previous response was invalid JSON. Return ONLY valid JSON matching the schema.]"],
            capacity,
        )
        # We don't want the reminder-note included in output count, so subtract it back:
        parsed = _parse_llm_response(raw, expected_note_count=len(notes) + 1)
        # Trim the reminder entry if the model included it
        return parsed[: len(notes)]
    except Exception as second_err:
        logger.error("LLM interpretation attempt 2 failed: %s", second_err)
        raise RuntimeError(f"LLM interpretation failed after 2 attempts: {second_err}")


def _get_attr(obj: Any, name: str):
    """Read `name` from either a Pydantic model / object or a dict."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)