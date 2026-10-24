"""Check a line with a model that has been tested against human reviews."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from llm_optimiser.bernard_runner import LUNA_URL, PostJson, post_json
from llm_optimiser.cases import ReviewCase
from llm_optimiser.config import Settings

JUDGE_RULES = """You score one short line from a cheeky dating-call host.
Use the call context and the human's written style reason as guidance.
Score 1 for a fitting, kind, witty and personal line. Score 0 for a poor line.
Use a number between 0 and 1. A line must not press a declined topic,
shame a dater, invent facts, reveal private facts for shock, or name the event
unless a dater asked. Favor a short, calm, dry question over advice or praise.
Return JSON only: {"score": 0.0, "reason": "short reason"}.
"""


class LineJudge:
    def __init__(self, settings: Settings, post: PostJson = post_json):
        self.settings = settings
        self.post = post

    def __call__(self, case: ReviewCase, line: str) -> tuple[float, str]:
        model = self.settings.judge_model.removeprefix("openai/")
        payload = {
            "model": model,
            "max_completion_tokens": 512,
            "messages": [
                {"role": "system", "content": JUDGE_RULES},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "call_context": case.call_context,
                            "line": line,
                            "style_reason": case.luna_reason,
                        }
                    ),
                },
            ],
            "response_format": {"type": "json_object"},
        }
        response = self.post(LUNA_URL, self.settings.openai_key, payload)
        try:
            content = response["choices"][0]["message"]["content"]
            result = json.loads(content)
            score = result["score"]
            reason = result["reason"]
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ValueError("Judge returned an invalid score") from error
        if type(score) not in (int, float) or not 0 <= score <= 1:
            raise ValueError("Judge score must be between 0 and 1")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Judge returned no reason")
        return float(score), reason.strip()


Judge = Callable[[ReviewCase, str], tuple[float, str]]


def cache_judge(judge: Judge) -> Judge:
    """Keep one score for each case and line during this run."""
    scores: dict[tuple[str, str], tuple[float, str]] = {}

    def score_once(case: ReviewCase, line: str) -> tuple[float, str]:
        key = (case.source_id, line)
        if key not in scores:
            scores[key] = judge(case, line)
        return scores[key]

    return score_once


def check_judge(cases: list[ReviewCase], judge: Judge, floor: float) -> dict[str, Any]:
    spoken = [case for case in cases if case.original_action == "speak"]
    labels = {case.luna_score for case in spoken}
    if labels != {0, 1}:
        raise ValueError("Judge check needs both Luna Pass and Fail reviews")
    right = 0
    for case in spoken:
        score, _ = judge(case, case.original_line)
        right += int((score >= 0.5) == bool(case.luna_score))
    agreement = right / len(spoken)
    if agreement < floor:
        raise ValueError(f"Judge agreement {agreement:.0%} is below {floor:.0%}")
    return {"checked": len(spoken), "agreement": agreement}
