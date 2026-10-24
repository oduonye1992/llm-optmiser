import json
import sys
from dataclasses import replace

from llm_optimiser import cli
from llm_optimiser.cases import read_snapshot
from llm_optimiser.config import load_settings
from llm_optimiser.optimize import prompt_hash


def test_unchanged_prompt_skips_judge_and_heldout_calls(monkeypatch, tmp_path, capsys):
    settings = replace(load_settings(), root=tmp_path)
    cases = read_snapshot(
        settings.jev_prompt.parents[1] / "examples/fake_reviews.jsonl"
    )
    test = cases[:2]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    jev = settings.jev_prompt.read_text()
    luna = settings.luna_prompt.read_text()
    (run_dir / "candidate_jev.txt").write_text(jev.rstrip())
    (run_dir / "candidate_luna.txt").write_text(luna.rstrip())
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": "same",
                "test_ids": [case.source_id for case in test],
                "seed_jev_hash": prompt_hash(jev),
                "seed_luna_hash": prompt_hash(luna),
                "candidate_hashes": {
                    "jev": prompt_hash(jev.rstrip()),
                    "luna": prompt_hash(luna.rstrip()),
                },
                "models": {
                    "jev": settings.jev_model,
                    "luna": settings.luna_model,
                    "luna_reasoning_effort": settings.luna_reasoning_effort,
                    "luna_max_completion_tokens": settings.luna_max_completion_tokens,
                    "judge": settings.judge_model,
                    "reflection": settings.reflection_model,
                },
            }
        )
    )

    monkeypatch.setattr(cli, "load_settings", lambda _path: settings)
    monkeypatch.setattr(
        cli, "load_ready_cases", lambda _settings, _snapshot: (cases, ([], [], test))
    )
    monkeypatch.setattr(cli, "snapshot_hash", lambda _path: "same")
    monkeypatch.setattr(cli, "require_private_data_approval", lambda: None)
    monkeypatch.setattr(
        cli, "check_judge", lambda *_args: (_ for _ in ()).throw(AssertionError())
    )
    monkeypatch.setattr(
        cli,
        "publish_experiments",
        lambda *_args: (_ for _ in ()).throw(AssertionError()),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["llm-optmiser", "evaluate", "--run", str(run_dir), "--snapshot", "fake"],
    )
    cli.main()
    result = json.loads(capsys.readouterr().out)
    assert result["prompt_changed"] is False
    assert result["experiments"] == {}
