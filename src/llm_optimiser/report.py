"""Save strict Braintrust experiments and a safe local summary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import braintrust

from llm_optimiser.bernard_runner import BernardProgram, case_inputs, hard_rule_errors
from llm_optimiser.cases import ReviewCase
from llm_optimiser.config import Settings
from llm_optimiser.optimize import compare_programs, prompt_hash, save_report


def run_safety_cases(program: BernardProgram, path: Path) -> dict[str, Any]:
    cases = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    failures: list[str] = []
    for item in cases:
        data = item["input"]
        context = (
            f"Dater names: {data['name1']}, {data['name2']}\n\n"
            "Call time is unknown because the server clock has not arrived."
        )
        if data.get("user_context"):
            context += f"\n\nEvent and participant context:\n{data['user_context']}"
        conversation = "\n".join(
            f"{turn['name']}: {turn['text']}" for turn in data["conversation"]
        )
        context += f"\n\nFull labeled conversation so far:\n{conversation}"
        result = program(
            name1=data["name1"],
            name2=data["name2"],
            user_context=data.get("user_context", ""),
            call_context=context,
            chattiness=data["chattiness"],
        )
        if result.action != item["expected"]["action"]:
            failures.append(item["id"] + ": wrong action")
        simple_case = ReviewCase(
            source_id=item["id"],
            room_id="fake",
            name1=data["name1"],
            name2=data["name2"],
            user_context=data.get("user_context", ""),
            call_context=context,
            chattiness=data["chattiness"],
            original_action=item["expected"]["action"],
            original_line="",
            jev_score=1,
            luna_score=None,
            jev_reason="synthetic rule",
            luna_reason="",
        )
        failures.extend(
            item["id"] + ": " + error
            for error in hard_rule_errors(simple_case, result.action, result.line)
        )
        event_name = item.get("metadata", {}).get("event_name", "")
        if event_name and not item.get("metadata", {}).get("event_questioned"):
            if event_name.casefold() in result.line.casefold():
                failures.append(item["id"] + ": event name leaked")
    if failures:
        raise ValueError("Safety cases failed: " + "; ".join(failures))
    return {"checked": len(cases), "failed": 0}


def publish_experiments(
    settings: Settings,
    test_cases: list[ReviewCase],
    baseline: BernardProgram,
    candidate: BernardProgram,
    judge,
    run_dir: Path,
    dataset_version: str,
    judge_agreement: float,
) -> dict[str, Any]:
    if not settings.braintrust_key:
        raise ValueError("BRAINTRUST_API_KEY is missing")
    if not test_cases:
        raise ValueError("Held-out test set is empty")
    comparison = compare_programs(test_cases, baseline, candidate, judge)
    safety = run_safety_cases(
        candidate, settings.root / "examples/bernard_safety_cases.jsonl"
    )
    metadata = {
        "dataset_version": dataset_version,
        "jev_model": settings.jev_model,
        "luna_model": settings.luna_model,
        "luna_reasoning_effort": settings.luna_reasoning_effort,
        "luna_max_completion_tokens": settings.luna_max_completion_tokens,
        "judge_model": settings.judge_model,
        "judge_agreement": judge_agreement,
        "baseline_jev_hash": prompt_hash(baseline.choose.signature.instructions),
        "baseline_luna_hash": prompt_hash(baseline.write.signature.instructions),
        "candidate_jev_hash": prompt_hash(candidate.choose.signature.instructions),
        "candidate_luna_hash": prompt_hash(candidate.write.signature.instructions),
        "safety_count": safety["checked"],
    }
    ids = {}
    for label in ("baseline", "candidate"):
        experiment = braintrust.init(
            project=settings.project,
            experiment=f"Bernard {run_dir.name} {label}",
            api_key=settings.braintrust_key,
            metadata={**metadata, "prompt_pair": label},
        )
        for case, row in zip(test_cases, comparison[label], strict=True):
            if row["source_id"] != case.source_id or row["score"] is None:
                raise ValueError("Experiment row has a missing or mixed score")
            experiment.log(
                id=f"{run_dir.name}-{label}-{case.source_id}",
                dataset_record_id=case.source_id,
                input=case_inputs(case),
                output={"action": row["action"], "line": row["line"]},
                expected={"action": case.wanted_action},
                scores={"Bernard Quality": row["score"]},
                metadata={
                    "feedback": row["feedback"],
                    "hard_errors": row["hard_errors"],
                    "room_id": case.room_id,
                },
            )
        experiment.flush()
        ids[label] = experiment.id
    report = {**metadata, **comparison, "experiments": ids, "safety": safety}
    save_report(run_dir / "report.json", report)
    return {
        "baseline_mean": comparison["baseline_mean"],
        "candidate_mean": comparison["candidate_mean"],
        "improved": comparison["improved"],
        "experiments": ids,
        "report": str(run_dir / "report.json"),
    }
