"""Split reviews, check gates, and search for better Jev and Luna prompts."""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import dspy
from dspy.teleprompt.gepa import InstructionProposer

from llm_optimiser.bernard_runner import (
    BernardProgram,
    case_inputs,
    hard_rule_errors,
)
from llm_optimiser.cases import ReviewCase
from llm_optimiser.config import Settings
from llm_optimiser.judge import Judge, cache_judge

PROPOSAL_RULES = """Improve the current instruction using the examples and feedback.
Write rules that can work on new calls.
Do not copy names, quotes, call facts, or review text.
Jev only picks speak or silent. Jev does not write words to say or return JSON.
Luna runs only after Jev picked speak. Luna always writes one spoken line.
Do not ask Luna to decide if it should speak, stay silent, or return blank text.
Keep the two jobs separate. Return only the new instruction."""

JEV_MIXED_JOB = re.compile(
    r"\b(?:output|return)\s+json\b|[\"']line[\"']|"
    r"\b(?:write|make|generate|produce)\b[^.!?\n]{0,40}\bline\b|"
    r"\bspoken\s+line\b|^\s*line\s*:",
    re.IGNORECASE | re.MULTILINE,
)
LUNA_SILENCE = re.compile(
    r"\b(?:choose|pick|return|output|stay|remain)\s+(?:to\s+)?silent\b|"
    r"\b(?:decide|choose)\s+(?:if|whether|when)\s+to\s+speak\b|"
    r"\b(?:do\s+not|don't)\s+speak\b|\bdo\s+not\s+interject\b",
    re.IGNORECASE,
)


def _words(value: str) -> list[str]:
    return re.findall(r"[\w]+(?:['’][\w]+)?", value.casefold())


def _private_text(
    cases: list[ReviewCase], seed: dict[str, str]
) -> tuple[set[str], set[tuple[str, ...]]]:
    seed_words = _words("\n".join(seed.values()))
    seed_names = set(seed_words)
    seed_phrases = set(zip(*(seed_words[i:] for i in range(6)), strict=False))
    names: set[str] = set()
    phrases: set[tuple[str, ...]] = set()
    for case in cases:
        for name in (case.name1, case.name2):
            names.update(word for word in _words(name) if word not in seed_names)
        for value in (
            case.user_context,
            case.call_context,
            case.original_line,
            case.jev_reason,
            case.luna_reason,
        ):
            words = _words(value)
            phrases.update(
                phrase
                for phrase in zip(*(words[i:] for i in range(6)), strict=False)
                if phrase not in seed_phrases
            )
            for match in re.finditer(r"\bnicknames?\s*[:=]\s*([^\n,;]+)", value, re.I):
                names.update(
                    word for word in _words(match.group(1)) if word not in seed_names
                )
    return names, phrases


def _check_prompt(
    name: str,
    body: str,
    names: set[str],
    phrases: set[tuple[str, ...]],
) -> None:
    if not body.strip():
        raise ValueError(f"{name} prompt is empty")
    if name == "choose" and JEV_MIXED_JOB.search(body):
        raise ValueError("Jev prompt asks for a spoken line or JSON")
    if name == "write" and LUNA_SILENCE.search(body):
        raise ValueError("Luna prompt asks to choose silence")
    words = _words(body)
    if names.intersection(words) or any(
        phrase in phrases
        for phrase in zip(*(words[i:] for i in range(6)), strict=False)
    ):
        raise ValueError("Prompt copies private review text")


class SafeInstructionProposer(InstructionProposer):
    def __init__(self, cases: list[ReviewCase], seed: dict[str, str]):
        super().__init__(base_instructions=PROPOSAL_RULES)
        self.names, self.phrases = _private_text(cases, seed)

    def __call__(self, candidate, reflective_dataset, components_to_update):
        proposed = super().__call__(candidate, reflective_dataset, components_to_update)
        for name, body in proposed.items():
            _check_prompt(name, body, self.names, self.phrases)
        return proposed


def gate_reviews(cases: list[ReviewCase], settings: Settings) -> None:
    if len(cases) < max(20, settings.min_reviews):
        raise ValueError(
            f"Need at least {max(20, settings.min_reviews)} complete reviews"
        )
    rooms = {case.room_id for case in cases}
    if len(rooms) < max(6, settings.min_rooms):
        raise ValueError(f"Need at least {max(6, settings.min_rooms)} reviewed rooms")
    actions = Counter(case.wanted_action for case in cases)
    lines = Counter(
        case.luna_score for case in cases if case.original_action == "speak"
    )
    if actions["speak"] < 6 or actions["silent"] < 6:
        raise ValueError("Need 6 wanted speak and 6 wanted silent reviews")
    if lines[0] < 2 or lines[1] < 2:
        raise ValueError("Need 2 Luna Pass and 2 Luna Fail reviews")
    if len({case.source_id for case in cases}) != len(cases):
        raise ValueError("Reviews have duplicate source IDs")


def split_cases(
    cases: list[ReviewCase],
) -> tuple[list[ReviewCase], list[ReviewCase], list[ReviewCase]]:
    groups: dict[str, list[ReviewCase]] = {}
    for case in cases:
        groups.setdefault(case.room_id, []).append(case)
    rooms = sorted(groups)
    if len(rooms) < 6:
        raise ValueError("A split needs at least 6 rooms")
    test_n = max(2, round(len(rooms) * 0.2))
    val_n = max(2, round(len(rooms) * 0.2))
    for seed in range(1000):
        order = rooms.copy()
        random.Random(seed).shuffle(order)
        train_rooms = set(order[: -val_n - test_n])
        val_rooms = set(order[-val_n - test_n : -test_n])
        test_rooms = set(order[-test_n:])
        splits = tuple(
            [case for case in cases if case.room_id in room_set]
            for room_set in (train_rooms, val_rooms, test_rooms)
        )
        train, val, test = splits
        if (
            len(train_rooms) >= 2
            and all(
                {c.wanted_action for c in part} == {"speak", "silent"}
                for part in (val, test)
            )
            and {c.luna_score for c in val if c.original_action == "speak"} >= {0, 1}
        ):
            return train, val, test
    raise ValueError("No room split has both actions and Luna review classes")


def score_case(
    case: ReviewCase, output: Any, judge: Judge
) -> tuple[float, str, list[str]]:
    action = output.action
    line = output.line
    errors = hard_rule_errors(case, action, line)
    if errors:
        return 0.0, "; ".join(errors), errors
    if action != case.wanted_action:
        return 0.0, f"Wanted {case.wanted_action}; got {action}. {case.jev_reason}", []
    if action == "silent":
        return 1.0, "Jev stayed silent at the right time", []
    line_quality, reason = judge(case, line)
    return 0.5 + 0.5 * line_quality, reason, []


def examples(cases: list[ReviewCase]) -> list[dspy.Example]:
    return [
        dspy.Example(**case_inputs(case), review_case=case).with_inputs(
            *case_inputs(case)
        )
        for case in cases
    ]


def run_gepa(
    settings: Settings,
    train: list[ReviewCase],
    val: list[ReviewCase],
    judge: Judge,
    program: BernardProgram,
    run_dir: Path,
) -> BernardProgram:
    run_dir.mkdir(parents=True, exist_ok=True)
    program.errors.types.clear()
    metric_errors: list[str] = []
    score_once = cache_judge(judge)
    seed = {
        "choose": program.choose.signature.instructions,
        "write": program.write.signature.instructions,
    }
    proposer = SafeInstructionProposer(train + val, seed)
    for name, body in seed.items():
        _check_prompt(name, body, proposer.names, proposer.phrases)

    def metric(example, pred, trace=None, pred_name=None, pred_trace=None):
        del trace, pred_name, pred_trace
        try:
            score, reason, _ = score_case(example.review_case, pred, score_once)
        except Exception as error:
            metric_errors.append(type(error).__name__)
            raise
        return dspy.Prediction(score=score, feedback=reason)

    reflection = dspy.LM(settings.reflection_model, cache=False)
    gepa = dspy.GEPA(
        metric=metric,
        reflection_lm=reflection,
        instruction_proposer=proposer,
        max_metric_calls=settings.max_metric_calls,
        num_threads=1,
        log_dir=str(run_dir / "gepa"),
    )
    # GEPA can print prompt text. Real call text must stay in ignored files.
    logger = logging.getLogger("dspy.teleprompt.gepa.gepa")
    old_level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        candidate = gepa.compile(
            program, trainset=examples(train), valset=examples(val)
        )
        for name, body in (
            ("choose", candidate.choose.signature.instructions),
            ("write", candidate.write.signature.instructions),
        ):
            _check_prompt(name, body, proposer.names, proposer.phrases)
        provider_errors = [
            error for error in program.errors.types if error != "luna:empty_line"
        ]
        if provider_errors or metric_errors:
            raise ValueError(
                "GEPA had provider or judge errors: "
                + ", ".join(sorted(set(provider_errors + metric_errors)))
            )
        for case in train + val:
            prior_errors = len(candidate.errors.types)
            try:
                candidate(**case_inputs(case))
            except Exception as error:
                if "luna:empty_line" in candidate.errors.types[prior_errors:]:
                    raise ValueError(
                        "Selected GEPA prompt returned an empty line on replay"
                    ) from error
                raise
        return candidate
    finally:
        logger.setLevel(old_level)


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def save_candidate(program: BernardProgram, run_dir: Path) -> dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    prompts = {
        "jev": program.choose.signature.instructions,
        "luna": program.write.signature.instructions,
    }
    for name, body in prompts.items():
        path = run_dir / f"candidate_{name}.txt"
        path.write_text(body)
        path.chmod(0o600)
    return {name: prompt_hash(body) for name, body in prompts.items()}


def compare_programs(
    cases: list[ReviewCase],
    baseline: BernardProgram,
    candidate: BernardProgram,
    judge: Judge,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    score_once = cache_judge(judge)
    for label, program in (("baseline", baseline), ("candidate", candidate)):
        rows = []
        for case in cases:
            output = program(**case_inputs(case))
            score, feedback, hard_errors = score_case(case, output, score_once)
            rows.append(
                {
                    "source_id": case.source_id,
                    "action": output.action,
                    "line": output.line,
                    "score": score,
                    "feedback": feedback,
                    "hard_errors": hard_errors,
                }
            )
        results[label] = rows
    if [r["source_id"] for r in results["baseline"]] != [
        r["source_id"] for r in results["candidate"]
    ]:
        raise ValueError("Baseline and candidate did not use the same test cases")
    baseline_mean = sum(row["score"] for row in results["baseline"]) / len(cases)
    candidate_mean = sum(row["score"] for row in results["candidate"]) / len(cases)
    prompt_changed = any(
        original.signature.instructions.strip()
        != updated.signature.instructions.strip()
        for original, updated in (
            (baseline.choose, candidate.choose),
            (baseline.write, candidate.write),
        )
    )
    return {
        **results,
        "baseline_mean": baseline_mean,
        "candidate_mean": candidate_mean,
        "prompt_changed": prompt_changed,
        "improved": prompt_changed
        and candidate_mean > baseline_mean
        and not any(row["hard_errors"] for row in results["candidate"]),
    }


def save_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    path.chmod(0o600)
