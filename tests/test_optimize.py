from dataclasses import replace
from pathlib import Path

import dspy
import pytest
from dspy.teleprompt.gepa import InstructionProposer

from llm_optimiser.bernard_runner import BernardProgram, case_inputs
from llm_optimiser.cases import read_snapshot
from llm_optimiser.config import load_settings
from llm_optimiser.judge import check_judge
from llm_optimiser.optimize import (
    SafeInstructionProposer,
    _check_prompt,
    _private_text,
    compare_programs,
    gate_reviews,
    run_gepa,
    split_cases,
)


def fake_cases():
    root = Path(__file__).resolve().parents[1]
    return read_snapshot(root / "examples/fake_reviews.jsonl")


def test_prompt_jobs_stay_separate():
    seed = load_settings()
    _check_prompt("choose", seed.jev_prompt.read_text(), set(), set())
    _check_prompt("write", seed.luna_prompt.read_text(), set(), set())
    with pytest.raises(ValueError, match="Jev prompt"):
        _check_prompt(
            "choose", 'Output JSON with {"action":"speak","line":"Hi"}', set(), set()
        )
    with pytest.raises(ValueError, match="Luna prompt"):
        _check_prompt(
            "write", "First decide whether to speak, then stay silent.", set(), set()
        )
    _check_prompt("choose", "Pick speak for a fresh personal question.", set(), set())
    _check_prompt("write", "Jev chose speak. Ask one short question.", set(), set())


def test_prompt_cannot_copy_private_review_text():
    case = replace(
        fake_cases()[0],
        name1="Zorinda Bayfell",
        name2="Tameron",
        user_context="Nickname: Moonfinch",
        call_context="A duck painted the moon blue at noon yesterday.",
    )
    seed = {"choose": "Pick speak or silent.", "write": "Write a short line."}
    names, phrases = _private_text([case], seed)
    for body in (
        "Ask Zorinda about the date.",
        "Ask Moonfinch about the date.",
        "A duck painted the moon blue at noon yesterday.",
    ):
        with pytest.raises(ValueError, match="private review text"):
            _check_prompt("write", body, names, phrases)
    _check_prompt("write", "Ask about a choice the daters just made.", names, phrases)


def test_gepa_checks_a_draft_before_it_can_run(monkeypatch):
    case = replace(fake_cases()[0], name1="Zorinda Bayfell")
    seed = {"choose": "Pick speak or silent.", "write": "Write a short line."}
    proposer = SafeInstructionProposer([case], seed)

    def bad_draft(self, candidate, reflective_dataset, components_to_update):
        return {"write": "Ask Zorinda a personal question."}

    monkeypatch.setattr(InstructionProposer, "__call__", bad_draft)
    with pytest.raises(ValueError, match="private review text"):
        proposer(seed, {"write": []}, ["write"])


def test_room_split_keeps_calls_apart_and_balanced():
    cases = fake_cases()
    gate_reviews(cases, load_settings())
    train, val, test = split_cases(cases)
    rooms = [{case.room_id for case in part} for part in (train, val, test)]
    assert not (rooms[0] & rooms[1] or rooms[0] & rooms[2] or rooms[1] & rooms[2])
    assert {case.source_id for case in train + val + test} == {
        case.source_id for case in cases
    }
    assert {case.wanted_action for case in val} == {"speak", "silent"}
    assert {case.luna_score for case in val if case.original_action == "speak"} == {
        0,
        1,
    }


def test_judge_must_match_both_human_classes():
    _, val, _ = split_cases(fake_cases())

    def good_judge(case, line):
        return float(case.luna_score), "fake judge"

    assert check_judge(val, good_judge, 0.8)["agreement"] == 1

    with pytest.raises(ValueError, match="agreement"):
        check_judge(val, lambda case, line: (1.0, "always pass"), 0.8)


def test_baseline_and_candidate_use_same_heldout_cases():
    _, _, test = split_cases(fake_cases())
    settings = replace(load_settings(), typesafe_key="fake", openai_key="fake")

    def post(url, key, payload):
        if "questions" in payload:
            return {"answers": {"action": {"choice": "speak"}}}
        return {"choices": [{"message": {"content": "Ava, what about date two?"}}]}

    program = BernardProgram(
        settings,
        settings.jev_prompt.read_text(),
        settings.luna_prompt.read_text(),
        post,
    )
    result = compare_programs(
        test, program, program, lambda case, line: (0.8, "fake judge")
    )
    assert not result["improved"]
    assert [row["source_id"] for row in result["baseline"]] == [
        row["source_id"] for row in result["candidate"]
    ]


def test_same_prompt_cannot_win_from_random_reply_changes():
    case = replace(fake_cases()[0], original_action="speak", jev_score=1, luna_score=1)
    settings = replace(load_settings(), typesafe_key="fake", openai_key="fake")
    lines = iter(["Bad line.", "Good line."])

    def post(url, key, payload):
        if "questions" in payload:
            return {"answers": {"action": {"choice": "speak"}}}
        return {"choices": [{"message": {"content": next(lines)}}]}

    program = BernardProgram(
        settings,
        settings.jev_prompt.read_text(),
        settings.luna_prompt.read_text(),
        post,
    )
    result = compare_programs(
        [case],
        program,
        program,
        lambda case, line: (float(line.startswith("Good")), "fake"),
    )
    assert result["candidate_mean"] > result["baseline_mean"]
    assert result["prompt_changed"] is False
    assert result["improved"] is False


def test_same_line_gets_one_judge_score_in_comparison():
    case = fake_cases()[0]
    settings = replace(load_settings(), typesafe_key="fake", openai_key="fake")

    def post(url, key, payload):
        if "questions" in payload:
            return {"answers": {"action": {"choice": "speak"}}}
        return {"choices": [{"message": {"content": "What happened next?"}}]}

    program = BernardProgram(
        settings,
        settings.jev_prompt.read_text(),
        settings.luna_prompt.read_text(),
        post,
    )
    judge_calls = []

    def judge(case, line):
        judge_calls.append(line)
        return (0.2 * len(judge_calls), "fake")

    result = compare_programs([case], program, program, judge)
    assert result["baseline_mean"] == result["candidate_mean"]
    assert len(judge_calls) == 1


def test_no_reviews_stop_before_provider_call():
    with pytest.raises(ValueError, match="20 complete"):
        gate_reviews([], load_settings())


def test_gepa_can_change_both_provider_prompts():
    settings = replace(load_settings(), typesafe_key="fake", openai_key="fake")

    def post(url, key, payload):
        if "questions" in payload:
            action = "speak" if "OPTIMIZED" in payload["state"] else "silent"
            return {"answers": {"action": {"choice": action}}}
        line = "good" if "OPTIMIZED" in payload["messages"][0]["content"] else "plain"
        return {"choices": [{"message": {"content": line}}]}

    program = BernardProgram(
        settings,
        settings.jev_prompt.read_text(),
        settings.luna_prompt.read_text(),
        post,
    )
    sample = dspy.Example(
        name1="Ava",
        name2="Leo",
        user_context="",
        call_context="Dater names: Ava, Leo\n\nAva: Hi.",
        chattiness=4,
    ).with_inputs("name1", "name2", "user_context", "call_context", "chattiness")

    def metric(example, pred, trace=None, pred_name=None, pred_trace=None):
        score = 0 if pred.action == "silent" else 1 if pred.line == "good" else 0.5
        return dspy.Prediction(score=score, feedback="Improve both instructions")

    def proposer(candidate, reflective_dataset, components_to_update):
        return {name: body + "\nOPTIMIZED" for name, body in candidate.items()}

    gepa = dspy.GEPA(
        metric=metric,
        instruction_proposer=proposer,
        reflection_lm=dspy.LM("openai/gpt-6-astra"),
        max_metric_calls=20,
        num_threads=1,
    )
    result = gepa.compile(program, trainset=[sample, sample], valset=[sample, sample])
    assert "OPTIMIZED" in result.choose.signature.instructions
    assert "OPTIMIZED" in result.write.signature.instructions
    assert result(**sample.inputs()).line == "good"


def test_gepa_keeps_a_failed_draft_but_checks_the_chosen_prompt(monkeypatch, tmp_path):
    settings = replace(load_settings(), typesafe_key="fake", openai_key="fake")
    calls = []

    def post(url, key, payload):
        calls.append(payload)
        if "questions" in payload:
            return {"answers": {"action": {"choice": "speak"}}}
        return {"choices": [{"message": {"content": "What happened next?"}}]}

    program = BernardProgram(
        settings,
        settings.jev_prompt.read_text(),
        settings.luna_prompt.read_text(),
        post,
    )

    class FakeGEPA:
        def __init__(self, **kwargs):
            pass

        def compile(self, program, trainset, valset):
            program.errors.types.append("luna:empty_line")
            return program

    monkeypatch.setattr(dspy, "GEPA", FakeGEPA)
    chosen = run_gepa(
        settings,
        fake_cases()[:1],
        fake_cases()[1:2],
        lambda case, line: (1.0, "good"),
        program,
        tmp_path,
    )
    assert chosen is program
    assert len(calls) == 4


def test_gepa_rejects_a_bad_selected_prompt(monkeypatch, tmp_path):
    settings = replace(load_settings(), typesafe_key="fake", openai_key="fake")

    def post(url, key, payload):
        raise AssertionError("A bad selected prompt must stop before replay")

    program = BernardProgram(
        settings,
        settings.jev_prompt.read_text(),
        settings.luna_prompt.read_text(),
        post,
    )

    class FakeGEPA:
        def __init__(self, **kwargs):
            pass

        def compile(self, program, trainset, valset):
            return BernardProgram(
                settings,
                program.choose.signature.instructions,
                "First decide whether to speak. Stay silent if nothing fits.",
                post,
            )

    monkeypatch.setattr(dspy, "GEPA", FakeGEPA)
    with pytest.raises(ValueError, match="Luna prompt asks to choose silence"):
        run_gepa(
            settings,
            fake_cases()[:1],
            fake_cases()[1:2],
            lambda case, line: (1.0, "good"),
            program,
            tmp_path,
        )


def test_gepa_stops_when_chosen_prompt_returns_no_line(monkeypatch, tmp_path):
    settings = replace(load_settings(), typesafe_key="fake", openai_key="fake")

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

    class FakeGEPA:
        def __init__(self, **kwargs):
            pass

        def compile(self, program, trainset, valset):
            return program

    monkeypatch.setattr(dspy, "GEPA", FakeGEPA)
    with pytest.raises(ValueError, match="Selected GEPA prompt returned an empty line"):
        run_gepa(
            settings,
            fake_cases()[:1],
            fake_cases()[1:2],
            lambda case, line: (1.0, "good"),
            program,
            tmp_path,
        )


def test_gepa_reuses_judge_score_for_feedback(monkeypatch, tmp_path):
    settings = replace(load_settings(), typesafe_key="fake", openai_key="fake")

    def post(url, key, payload):
        if "questions" in payload:
            return {"answers": {"action": {"choice": "speak"}}}
        return {"choices": [{"message": {"content": "What happened next?"}}]}

    program = BernardProgram(
        settings,
        settings.jev_prompt.read_text(),
        settings.luna_prompt.read_text(),
        post,
    )
    judge_calls = []

    def judge(case, line):
        judge_calls.append(line)
        return (0.2 * len(judge_calls), "fake")

    class FakeGEPA:
        def __init__(self, metric, **kwargs):
            self.metric = metric

        def compile(self, program, trainset, valset):
            case = trainset[0].review_case
            output = program(**case_inputs(case))
            first = self.metric(trainset[0], output)
            second = self.metric(trainset[0], output, pred_name="write")
            assert first.score == second.score
            return program

    monkeypatch.setattr(dspy, "GEPA", FakeGEPA)
    run_gepa(
        settings,
        fake_cases()[:1],
        fake_cases()[1:2],
        judge,
        program,
        tmp_path,
    )
    assert len(judge_calls) == 1
