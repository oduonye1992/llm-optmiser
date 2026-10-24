import importlib.util
import io
import sys
import urllib.error
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_optimiser.bernard_runner import BernardProgram, hard_rule_errors, post_json
from llm_optimiser.cases import ReviewCase
from llm_optimiser.config import load_settings


def fake_settings():
    return replace(load_settings(), typesafe_key="fake", openai_key="fake")


def test_two_stage_runner_uses_both_prompts_and_current_fields():
    settings = fake_settings()
    calls = []

    def post(url, key, payload):
        calls.append((url, payload))
        if "questions" in payload:
            return {"answers": {"action": {"choice": "speak"}}}
        return {
            "choices": [{"message": {"content": "Ava, what is your deal breaker?"}}]
        }

    program = BernardProgram(
        settings,
        settings.jev_prompt.read_text(),
        settings.luna_prompt.read_text(),
        post,
    )
    context = "Dater names: Ava, Leo\n\nFull labeled conversation so far:\nAva: Hi."
    result = program(
        name1="Ava",
        name2="Leo",
        user_context="Ava likes chess.",
        call_context=context,
        chattiness=4,
    )
    assert (result.action, result.line) == ("speak", "Ava, what is your deal breaker?")
    assert len(calls) == 2
    jev = calls[0][1]
    luna = calls[1][1]
    assert set(jev) == {"model", "state", "questions"}
    assert set(luna) == {
        "model",
        "reasoning_effort",
        "max_completion_tokens",
        "messages",
    }
    assert luna["max_completion_tokens"] == settings.luna_max_completion_tokens
    assert '(("action"' not in jev["state"]
    assert "Pick only one action: `speak` or `silent`" in jev["state"]
    assert '{"action":"silent"}' not in jev["state"]
    assert luna["messages"][1]["content"].startswith(context)


def test_trial_can_raise_luna_token_cap():
    settings = replace(fake_settings(), luna_max_completion_tokens=512)

    def post(url, key, payload):
        if "questions" in payload:
            return {"answers": {"action": {"choice": "speak"}}}
        assert payload["max_completion_tokens"] == 512
        return {"choices": [{"message": {"content": "What happened next?"}}]}

    program = BernardProgram(
        settings,
        settings.jev_prompt.read_text(),
        settings.luna_prompt.read_text(),
        post,
    )
    assert (
        program(
            name1="Ava",
            name2="Leo",
            user_context="",
            call_context="Ava: Hi.",
            chattiness=4,
        ).line
        == "What happened next?"
    )


def test_provider_400_shows_safe_fields(monkeypatch):
    import llm_optimiser.bernard_runner as runner

    body = b'{"error":{"type":"invalid_request_error",'
    body += b'"code":"bad_request","param":"max_completion_tokens",'
    body += b'"message":"private call text"}}'

    def fail(_request, timeout):
        raise urllib.error.HTTPError(
            "https://example.test", 400, "Bad Request", {}, io.BytesIO(body)
        )

    monkeypatch.setattr(runner.urllib.request, "urlopen", fail)
    with pytest.raises(
        ValueError, match="HTTP 400.*bad_request.*max_completion_tokens"
    ) as error:
        post_json("https://example.test", "fake", {"model": "fake"})
    assert "private call text" not in str(error.value)


def test_empty_luna_line_is_marked_as_a_bad_draft():
    settings = fake_settings()

    def post(url, key, payload):
        if "questions" in payload:
            return {"answers": {"action": {"choice": "speak"}}}
        return {"choices": [{"message": {"content": ""}}]}

    program = BernardProgram(
        settings,
        settings.jev_prompt.read_text(),
        settings.luna_prompt.read_text(),
        post,
    )
    with pytest.raises(Exception, match="empty line"):
        program(
            name1="Ava",
            name2="Leo",
            user_context="",
            call_context="Dater names: Ava, Leo\nAva: Hi.",
            chattiness=4,
        )
    assert program.errors.types == ["luna:empty_line"]


def test_request_shape_matches_local_bernard(monkeypatch):
    agent_root = Path(__file__).resolve().parents[2] / "livekit-agent" / "src"
    if not (agent_root / "decision_client.py").exists():
        pytest.skip("Standalone clone has no local Bernard checkout")
    monkeypatch.setitem(
        sys.modules,
        "braintrust_telemetry",
        SimpleNamespace(log_decision_trace=lambda *args, **kwargs: None),
    )

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    original = load("source_decision_client", agent_root / "decision_client.py")
    prompts = load("source_bernard_prompts", agent_root / "bernard_prompts.py")
    original_calls = []

    def original_post(url, key, payload):
        original_calls.append(payload)
        if "questions" in payload:
            return {"answers": {"action": {"choice": "speak"}}}
        return {
            "choices": [{"message": {"content": "Ava, what is your deal breaker?"}}]
        }

    original._post_json = original_post
    context = "Dater names: Ava, Leo\n\nFull labeled conversation so far:\nAva: Hi."
    client = original.JevLunaDecisionClient(
        typesafe_api_key="fake",
        openai_api_key="fake",
        openai_model="gpt-6-luna",
        reasoning_effort="medium",
        spoken_line_system_prompt=prompts.format_assistant_instructions(
            name1="Ava", name2="Leo", user_context="Ava likes chess."
        ),
    )
    state = (
        "Bernard policy:\n"
        + prompts.format_decision_system_prompt(name1="Ava", name2="Leo", chattiness=4)
        + "\n\nCall context:\n"
        + context
    )
    assert client._choose_action(state) == "speak"
    assert client._write_line(context) == "Ava, what is your deal breaker?"
    new_calls = []

    def new_post(url, key, payload):
        new_calls.append(payload)
        return original_post(url, key, payload)

    settings = fake_settings()
    program = BernardProgram(
        settings,
        settings.jev_prompt.read_text(),
        settings.luna_prompt.read_text(),
        new_post,
    )
    program(
        name1="Ava",
        name2="Leo",
        user_context="Ava likes chess.",
        call_context=context,
        chattiness=4,
    )
    assert new_calls[0]["model"] == original_calls[0]["model"]
    assert new_calls[0]["questions"] == original_calls[0]["questions"]
    assert new_calls[0]["state"].endswith("\n\nCall context:\n" + context)
    assert new_calls[1]["model"] == original_calls[1]["model"]
    assert new_calls[1]["reasoning_effort"] == original_calls[1]["reasoning_effort"]
    assert (
        new_calls[1]["max_completion_tokens"]
        == original_calls[1]["max_completion_tokens"]
    )
    assert new_calls[1]["messages"][1] == original_calls[1]["messages"][1]


def test_hard_rules_stop_long_and_internal_line():
    case = ReviewCase(
        "fake", "room", "Ava", "Leo", "", "", 4, "speak", "", 1, 1, "", ""
    )
    assert hard_rule_errors(case, "speak", "Bernard, " + "word " * 20)


def test_hard_rules_stop_event_and_made_up_name():
    case = ReviewCase(
        "fake",
        "room",
        "Ava",
        "Leo",
        "Event: Sunset Night. Ava likes cooking.",
        "Full labeled conversation so far:\nAva: What do you cook?",
        4,
        "speak",
        "",
        1,
        1,
        "",
        "",
    )
    errors = hard_rule_errors(case, "speak", "Milo, are you going to Sunset Night?")
    assert errors == [
        "Line names the event before a dater does",
        "Line uses a name that was not supplied",
    ]


@pytest.mark.parametrize(
    "line",
    [
        "Mate, do you need a minute?",
        "Wait, lighter how?",
        "Dude, tell us what happened.",
    ],
)
def test_hard_rules_allow_plain_openers_after_reviewed_question(line):
    case = ReviewCase(
        "fake",
        "room",
        "Ava",
        "Leo",
        "",
        "Full labeled conversation so far:\nLeo: What's up?",
        4,
        "speak",
        "",
        1,
        0,
        "A quick cut-in fits here.",
        "Keep it short.",
    )
    assert hard_rule_errors(case, "speak", line) == []
