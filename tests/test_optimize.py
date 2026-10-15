from pathlib import Path

import dspy
import pytest

from llm_optimiser.bernard_runner import BernardProgram
from llm_optimiser.cases import read_snapshot
from llm_optimiser.config import load_settings
from llm_optimiser.judge import check_judge
from llm_optimiser.optimize import (
    compare_programs,
    gate_reviews,
    split_cases,
)


def fake_cases():
    root = Path(__file__).resolve().parents[1]
    return read_snapshot(root / "examples/fake_reviews.jsonl")


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
    settings = load_settings()

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


def test_no_reviews_stop_before_provider_call():
    with pytest.raises(ValueError, match="20 complete"):
        gate_reviews([], load_settings())


def test_gepa_can_change_both_provider_prompts():
    settings = load_settings()

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
