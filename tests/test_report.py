from dataclasses import replace

import pytest

from llm_optimiser.bernard_runner import BernardProgram
from llm_optimiser.cases import read_snapshot
from llm_optimiser.config import load_settings
from llm_optimiser.report import publish_experiments


def test_two_experiments_use_the_same_heldout_ids(monkeypatch, tmp_path):
    import llm_optimiser.report as module

    settings = replace(
        load_settings(), braintrust_key="fake", typesafe_key="fake", openai_key="fake"
    )
    cases = read_snapshot(settings.root / "examples/fake_reviews.jsonl")[:3]

    def post(url, key, payload):
        if "questions" in payload:
            return {"answers": {"action": {"choice": "silent"}}}
        raise AssertionError("A silent choice must not call Luna")

    program = BernardProgram(
        settings,
        settings.jev_prompt.read_text(),
        settings.luna_prompt.read_text(),
        post,
    )
    experiments = []

    class FakeExperiment:
        def __init__(self, name):
            self.id = name
            self.rows = []

        def log(self, **row):
            self.rows.append(row)

        def flush(self):
            pass

    def init(**kwargs):
        experiment = FakeExperiment(kwargs["experiment"])
        experiments.append(experiment)
        return experiment

    monkeypatch.setattr(module.braintrust, "init", init)
    monkeypatch.setattr(
        module,
        "run_safety_cases",
        lambda candidate, path: {"checked": 12, "failed": 0},
    )
    run_dir = tmp_path / "fake-run"
    result = publish_experiments(
        settings,
        cases,
        program,
        program,
        lambda case, line: (0.8, "fake judge"),
        run_dir,
        "fake-version",
        1.0,
    )
    assert len(experiments) == 2
    first_ids = [row["dataset_record_id"] for row in experiments[0].rows]
    second_ids = [row["dataset_record_id"] for row in experiments[1].rows]
    assert first_ids == second_ids == [case.source_id for case in cases]
    assert result["report"] == str(run_dir / "report.json")


def test_provider_error_stops_comparison_before_experiment(monkeypatch, tmp_path):
    import llm_optimiser.report as module

    settings = replace(
        load_settings(), braintrust_key="fake", typesafe_key="fake", openai_key="fake"
    )
    case = read_snapshot(settings.root / "examples/fake_reviews.jsonl")[0]

    def broken_post(url, key, payload):
        raise RuntimeError("fake provider failure")

    program = BernardProgram(
        settings,
        settings.jev_prompt.read_text(),
        settings.luna_prompt.read_text(),
        broken_post,
    )
    calls = []
    monkeypatch.setattr(
        module.braintrust, "init", lambda **kwargs: calls.append(kwargs)
    )
    with pytest.raises(Exception, match="fake provider failure"):
        publish_experiments(
            settings,
            [case],
            program,
            program,
            lambda case, line: (0.8, "fake judge"),
            tmp_path / "bad-run",
            "fake-version",
            1.0,
        )
    assert not calls
