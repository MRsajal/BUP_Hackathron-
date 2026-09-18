"""
tests/test_llm_interpreter.py
-----------------------------
Unit tests for the LLM interpreter and guardrails.

These tests MOCK the LLM call so they run instantly, don't need internet,
and don't consume Groq quota. Production code still calls the real LLM.
"""

import json
import asyncio
import pytest
from unittest.mock import patch, MagicMock

from app.llm_interpreter import interpret_notes, _parse_llm_response
from app.guardrails import validate_directives
from app.directives import DirectiveInterpretation


# ---------------------------------------------------------------------------
# Helper — build a fake OpenAI response object shaped like the real thing.
# ---------------------------------------------------------------------------
def _fake_response(json_dict: dict) -> MagicMock:
    m = MagicMock()
    m.choices = [MagicMock()]
    m.choices[0].message.content = json.dumps(json_dict)
    return m


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)


# ---------------------------------------------------------------------------
# _parse_llm_response — the low-level JSON parser
# ---------------------------------------------------------------------------
class TestParseLLMResponse:
    def test_clean_json(self):
        raw = json.dumps({
            "interpretations": [
                {"note_index": 0, "applies": False, "directive_type": "no_op",
                 "structured_adjustment": None, "explanation": "n/a"}
            ]
        })
        result = _parse_llm_response(raw, expected_note_count=1)
        assert len(result) == 1
        assert result[0].directive_type == "no_op"

    def test_json_wrapped_in_markdown_fences(self):
        raw = "```json\n" + json.dumps({
            "interpretations": [
                {"note_index": 0, "applies": False, "directive_type": "no_op",
                 "structured_adjustment": None, "explanation": ""}
            ]
        }) + "\n```"
        result = _parse_llm_response(raw, expected_note_count=1)
        assert result[0].directive_type == "no_op"

    def test_malformed_json_raises(self):
        with pytest.raises(ValueError, match="valid JSON"):
            _parse_llm_response("not json at all", expected_note_count=1)

    def test_missing_interpretations_key_raises(self):
        raw = json.dumps({"wrong_key": []})
        with pytest.raises(ValueError, match="interpretations"):
            _parse_llm_response(raw, expected_note_count=1)

    def test_wrong_entry_count_raises(self):
        raw = json.dumps({
            "interpretations": [
                {"note_index": 0, "applies": False, "directive_type": "no_op",
                 "structured_adjustment": None, "explanation": ""}
            ]
        })
        with pytest.raises(ValueError, match="Expected 3"):
            _parse_llm_response(raw, expected_note_count=3)


# ---------------------------------------------------------------------------
# interpret_notes — full mocked LLM flow
# ---------------------------------------------------------------------------
class TestInterpretNotes:
    def _mock_client(self, response_dict):
        client_mock = MagicMock()
        client_mock.chat.completions.create.return_value = _fake_response(response_dict)
        return client_mock

    def test_three_notes_end_to_end(self):
        request = {
            "operator_notes": ["a", "b", "c"],
            "battery": {"capacity_kwh": 500},
        }
        mocked = {
            "interpretations": [
                {"note_index": 0, "applies": True, "directive_type": "solar_reduction",
                 "structured_adjustment": {"hours": [13, 14], "factor": 0.2},
                 "explanation": "solar"},
                {"note_index": 1, "applies": True, "directive_type": "no_charge_window",
                 "structured_adjustment": {"hours": [14, 15]},
                 "explanation": "no charge"},
                {"note_index": 2, "applies": False, "directive_type": "no_op",
                 "structured_adjustment": None,
                 "explanation": "irrelevant"},
            ]
        }
        with patch("app.llm_interpreter.get_client", return_value=self._mock_client(mocked)):
            result = _run(interpret_notes(request))
        assert len(result) == 3
        assert result[0].directive_type == "solar_reduction"
        assert result[2].applies is False

    def test_retry_on_first_failure(self):
        """First call returns garbage, second returns valid JSON."""
        request = {"operator_notes": ["a"], "battery": {"capacity_kwh": 500}}

        good = {
            "interpretations": [
                {"note_index": 0, "applies": False, "directive_type": "no_op",
                 "structured_adjustment": None, "explanation": "ok"}
            ]
        }
        # Retry sends notes + [reminder], so expected_note_count = 2 on second call.
        good_retry = {
            "interpretations": [
                {"note_index": 0, "applies": False, "directive_type": "no_op",
                 "structured_adjustment": None, "explanation": "ok"},
                {"note_index": 1, "applies": False, "directive_type": "no_op",
                 "structured_adjustment": None, "explanation": "reminder"},
            ]
        }

        bad_response = MagicMock()
        bad_response.choices = [MagicMock()]
        bad_response.choices[0].message.content = "not json"

        client_mock = MagicMock()
        client_mock.chat.completions.create.side_effect = [
            bad_response,
            _fake_response(good_retry),
        ]

        with patch("app.llm_interpreter.get_client", return_value=client_mock):
            result = _run(interpret_notes(request))
        assert len(result) == 1  # reminder trimmed
        assert result[0].directive_type == "no_op"


# ---------------------------------------------------------------------------
# validate_directives — guardrail behaviour
# ---------------------------------------------------------------------------
def _mk(note_index, dt, adj, applies=True, expl=""):
    return DirectiveInterpretation(
        note_index=note_index,
        applies=applies,
        directive_type=dt,
        structured_adjustment=adj,
        explanation=expl,
    )


class TestGuardrails:
    def test_valid_solar_reduction_passes(self):
        req = {"operator_notes": ["x"], "battery": {"capacity_kwh": 500}}
        d = [_mk(0, "solar_reduction", {"hours": [13, 14], "factor": 0.2})]
        result = validate_directives(d, req)
        assert result[0].structured_adjustment == {"hours": [13, 14], "factor": 0.2}

    def test_hours_get_sorted_and_deduped(self):
        req = {"operator_notes": ["x"], "battery": {"capacity_kwh": 500}}
        d = [_mk(0, "no_charge_window", {"hours": [15, 14, 14, 13]})]
        result = validate_directives(d, req)
        assert result[0].structured_adjustment["hours"] == [13, 14, 15]

    def test_factor_out_of_range_raises(self):
        req = {"operator_notes": ["x"], "battery": {"capacity_kwh": 500}}
        d = [_mk(0, "solar_reduction", {"hours": [13], "factor": 1.5})]
        with pytest.raises(ValueError, match="factor out of"):
            validate_directives(d, req)

    def test_reserve_exceeds_capacity_raises(self):
        req = {"operator_notes": ["x"], "battery": {"capacity_kwh": 100}}
        d = [_mk(0, "minimum_battery_reserve",
                 {"hours": [18], "minimum_energy_kwh": 200})]
        with pytest.raises(ValueError, match="exceeds battery capacity"):
            validate_directives(d, req)

    def test_negative_grid_cap_raises(self):
        req = {"operator_notes": ["x"], "battery": {"capacity_kwh": 500}}
        d = [_mk(0, "max_grid_window",
                 {"hours": [18], "max_grid_kwh": -10})]
        with pytest.raises(ValueError, match="negative"):
            validate_directives(d, req)

    def test_hour_out_of_range_raises(self):
        req = {"operator_notes": ["x"], "battery": {"capacity_kwh": 500}}
        d = [_mk(0, "no_charge_window", {"hours": [24]})]
        with pytest.raises(ValueError, match="out of range"):
            validate_directives(d, req)

    def test_no_op_with_extra_adjustment_gets_nulled(self):
        req = {"operator_notes": ["x"], "battery": {"capacity_kwh": 500}}
        d = [_mk(0, "no_op", {"hours": [13]}, applies=False)]
        result = validate_directives(d, req)
        assert result[0].structured_adjustment is None

    def test_wrong_applies_semantics_raises(self):
        req = {"operator_notes": ["x"], "battery": {"capacity_kwh": 500}}
        # non-no_op with applies=false should fail
        d = [_mk(0, "solar_reduction", {"hours": [13], "factor": 0.2}, applies=False)]
        with pytest.raises(ValueError, match="applies=true"):
            validate_directives(d, req)

    def test_wrong_entry_count_raises(self):
        req = {"operator_notes": ["a", "b"], "battery": {"capacity_kwh": 500}}
        d = [_mk(0, "no_op", None, applies=False)]  # only 1 entry for 2 notes
        with pytest.raises(ValueError, match="Expected 2"):
            validate_directives(d, req)

    def test_out_of_order_note_index_raises(self):
        req = {"operator_notes": ["a", "b"], "battery": {"capacity_kwh": 500}}
        d = [
            _mk(1, "no_op", None, applies=False),
            _mk(0, "no_op", None, applies=False),
        ]
        with pytest.raises(ValueError, match="note_index"):
            validate_directives(d, req)