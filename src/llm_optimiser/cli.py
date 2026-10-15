"""Run the review, judge, search, and comparison steps."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from llm_optimiser.bernard_runner import BernardProgram
from llm_optimiser.braintrust_sync import sync
from llm_optimiser.cases import read_snapshot
from llm_optimiser.config import Settings, load_settings
from llm_optimiser.judge import LineJudge, check_judge
from llm_optimiser.optimize import (
    gate_reviews,
    prompt_hash,
    run_gepa,
    save_candidate,
    save_report,
    split_cases,
)
from llm_optimiser.report import publish_experiments


def require_private_data_approval() -> None:
    if os.getenv("PRIVATE_DATA_APPROVED", "").lower() != "true":
        raise ValueError(
            "Real call text is blocked. Close the private-data gate, then set "
            "PRIVATE_DATA_APPROVED=true in .env."
        )


def require_provider_keys(settings: Settings, *, braintrust: bool = False) -> None:
    missing = [
        name
        for name, value in (
            ("TYPESAFE_API_KEY", settings.typesafe_key),
            ("OPENAI_API_KEY", settings.openai_key),
            ("BRAINTRUST_API_KEY", settings.braintrust_key if braintrust else "ready"),
        )
        if not value
    ]
    if missing:
        raise ValueError("Set these keys in .env: " + ", ".join(missing))


def snapshot_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_ready_cases(settings: Settings, snapshot: Path):
    if not snapshot.exists():
        raise ValueError(
            "No reviewed snapshot. Review moments in Braintrust, then run sync."
        )
    cases = read_snapshot(snapshot)
    gate_reviews(cases, settings)
    return cases, split_cases(cases)


def make_program(settings: Settings, jev: Path, luna: Path) -> BernardProgram:
    return BernardProgram(settings, jev.read_text(), luna.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("sync", "check-judge", "optimize", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--snapshot", type=Path)
        if name == "evaluate":
            command.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    try:
        settings = load_settings(args.config)
        snapshot = args.snapshot or settings.root / "data/reviewed.jsonl"
        if args.command == "sync":
            require_private_data_approval()
            result = sync(settings, snapshot)
        else:
            cases, (train, val, test) = load_ready_cases(settings, snapshot)
            require_private_data_approval()
            judge = LineJudge(settings)
            if args.command == "check-judge":
                result = check_judge(val, judge, max(0.8, settings.min_judge_agreement))
            else:
                require_provider_keys(settings, braintrust=args.command == "evaluate")
                check = check_judge(val, judge, max(0.8, settings.min_judge_agreement))
                if args.command == "optimize":
                    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
                    run_dir = settings.root / "runs" / stamp
                    baseline = make_program(
                        settings, settings.jev_prompt, settings.luna_prompt
                    )
                    candidate = run_gepa(settings, train, val, judge, baseline, run_dir)
                    hashes = save_candidate(candidate, run_dir)
                    sidecar = snapshot.with_suffix(snapshot.suffix + ".meta.json")
                    sync_metadata = (
                        json.loads(sidecar.read_text()) if sidecar.exists() else {}
                    )
                    manifest = {
                        "dataset_version": snapshot_hash(snapshot),
                        "braintrust_version": sync_metadata.get("braintrust_version"),
                        "braintrust_dataset_id": sync_metadata.get("dataset_id"),
                        "snapshot": str(snapshot.resolve()),
                        "source_count": len(cases),
                        "train_ids": [case.source_id for case in train],
                        "validation_ids": [case.source_id for case in val],
                        "test_ids": [case.source_id for case in test],
                        "judge_agreement": check["agreement"],
                        "seed_jev_hash": prompt_hash(settings.jev_prompt.read_text()),
                        "seed_luna_hash": prompt_hash(settings.luna_prompt.read_text()),
                        "candidate_hashes": hashes,
                        "models": {
                            "jev": settings.jev_model,
                            "luna": settings.luna_model,
                            "judge": settings.judge_model,
                            "reflection": settings.reflection_model,
                        },
                    }
                    save_report(run_dir / "manifest.json", manifest)
                    result = {"run": str(run_dir), "candidate_hashes": hashes}
                else:
                    run_dir = args.run.resolve()
                    manifest = json.loads((run_dir / "manifest.json").read_text())
                    if snapshot_hash(snapshot) != manifest["dataset_version"]:
                        raise ValueError(
                            "Snapshot changed since this run. Use its saved snapshot."
                        )
                    if [case.source_id for case in test] != manifest["test_ids"]:
                        raise ValueError("Held-out test IDs changed since optimization")
                    current_models = {
                        "jev": settings.jev_model,
                        "luna": settings.luna_model,
                        "judge": settings.judge_model,
                        "reflection": settings.reflection_model,
                    }
                    if current_models != manifest["models"]:
                        raise ValueError("Model settings changed since optimization")
                    if (
                        prompt_hash(settings.jev_prompt.read_text())
                        != manifest["seed_jev_hash"]
                        or prompt_hash(settings.luna_prompt.read_text())
                        != manifest["seed_luna_hash"]
                    ):
                        raise ValueError("Seed prompts changed since optimization")
                    baseline = make_program(
                        settings, settings.jev_prompt, settings.luna_prompt
                    )
                    candidate = make_program(
                        settings,
                        run_dir / "candidate_jev.txt",
                        run_dir / "candidate_luna.txt",
                    )
                    candidate_hashes = {
                        "jev": prompt_hash((run_dir / "candidate_jev.txt").read_text()),
                        "luna": prompt_hash(
                            (run_dir / "candidate_luna.txt").read_text()
                        ),
                    }
                    if candidate_hashes != manifest["candidate_hashes"]:
                        raise ValueError("Candidate prompts changed since optimization")
                    result = publish_experiments(
                        settings,
                        test,
                        baseline,
                        candidate,
                        judge,
                        run_dir,
                        manifest["dataset_version"],
                        check["agreement"],
                    )
        print(json.dumps(result, indent=2))
    except (ValueError, OSError) as error:
        parser.exit(1, f"llm-optmiser: {error}\n")


if __name__ == "__main__":
    main()
