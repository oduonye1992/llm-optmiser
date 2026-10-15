"""Run Jev and Luna with Bernard's current request fields."""

from __future__ import annotations

import json
import re
import urllib.request
from collections.abc import Callable
from typing import Any

import dspy
from dspy.lm15 import Message, Response, Usage

from llm_optimiser.cases import GUIDANCE, LUNA_SUFFIX, ReviewCase
from llm_optimiser.config import Settings

JEV_URL = "https://api.typesafe.ai/v1/systemone"
LUNA_URL = "https://api.openai.com/v1/chat/completions"
JEV_QUESTION = {
    "type": "choice",
    "instructions": (
        "Use Bernard's policy in the state. Decide whether he should "
        "speak now. Follow the chatty level in the policy. A higher "
        "level means more fitting chances should lead to speech. A "
        "playful personal question can fit while the daters talk well. "
        "Do not wait for silence or press a declined topic. If the "
        "last line asks the other dater a question, choose silent "
        "until that dater answers, at any chatty level."
    ),
    "criteria": {
        "silent": (
            "The last line asks the other dater a question, no fresh question "
            "fits, or one would press a declined topic."
        ),
        "speak": (
            "A timely, playful personal question or sly comment fits the chatty level."
        ),
    },
}


def post_json(url: str, key: str, payload: dict[str, Any]) -> dict[str, Any]:
    if not key:
        raise ValueError("A provider key is missing")
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        result = json.loads(response.read())
    if not isinstance(result, dict):
        raise ValueError("Provider returned a non-object response")
    return result


PostJson = Callable[[str, str, dict[str, Any]], dict[str, Any]]


class RunErrors:
    def __init__(self):
        self.types: list[str] = []

    def __deepcopy__(self, memo):
        return self


class ChooseAction(dspy.Signature):
    """Choose if Bernard speaks now."""

    name1: str = dspy.InputField()
    name2: str = dspy.InputField()
    chattiness: int = dspy.InputField()
    call_context: str = dspy.InputField()
    action: str = dspy.OutputField()


class WriteLine(dspy.Signature):
    """Write one spoken line for Bernard."""

    name1: str = dspy.InputField()
    name2: str = dspy.InputField()
    user_context: str = dspy.InputField()
    call_context: str = dspy.InputField()
    line: str = dspy.OutputField()


def _fill(template: str, values: dict[str, Any]) -> str:
    for key, value in values.items():
        template = template.replace("{" + key + "}", str(value))
    return template.replace("{{", "{").replace("}}", "}")


class BernardAdapter(dspy.Adapter):
    def __init__(self, stage: str):
        super().__init__()
        self.stage = stage

    def format(self, signature, demos, inputs):
        if self.stage == "jev":
            level = max(1, min(5, int(inputs["chattiness"])))
            system = _fill(signature.instructions, inputs)
            system += f"\n\nChatty level: {level}/5. {GUIDANCE[level]}\n"
            user = inputs["call_context"]
        else:
            values = dict(inputs)
            values["user_context"] = (
                inputs["user_context"].strip() or "(no extra context yet)"
            )
            system = _fill(signature.instructions, values) + "\n"
            user = inputs["call_context"] + LUNA_SUFFIX
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def parse(self, signature, completion):
        value = completion.strip()
        if self.stage == "jev" and value not in ("speak", "silent"):
            raise ValueError("Jev returned an unsupported action")
        if not value:
            raise ValueError("Luna returned an empty line")
        return {"action" if self.stage == "jev" else "line": value}


class BernardEngine:
    def __init__(
        self,
        stage: str,
        settings: Settings,
        post: PostJson = post_json,
        errors: RunErrors | None = None,
    ):
        self.stage = stage
        self.settings = settings
        self.post = post
        self.errors = errors or RunErrors()

    def complete(self, request):
        try:
            return self._complete(request)
        except Exception as error:
            self.errors.types.append(f"{self.stage}:{type(error).__name__}")
            raise

    def _complete(self, request):
        system = request.system
        user = request.messages[-1].text
        if self.stage == "jev":
            payload = {
                "model": self.settings.jev_model,
                "state": f"Bernard policy:\n{system}\n\nCall context:\n{user}",
                "questions": {"action": JEV_QUESTION},
            }
            reply = self.post(JEV_URL, self.settings.typesafe_key, payload)
            try:
                value = reply["answers"]["action"]["choice"]
            except (KeyError, TypeError) as error:
                raise ValueError("Jev response lacks an action choice") from error
        else:
            payload = {
                "model": self.settings.luna_model,
                "reasoning_effort": self.settings.luna_reasoning_effort,
                "max_completion_tokens": 256,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
            reply = self.post(LUNA_URL, self.settings.openai_key, payload)
            try:
                value = reply["choices"][0]["message"]["content"]
            except (IndexError, KeyError, TypeError) as error:
                raise ValueError("Luna response lacks a line") from error
        if not isinstance(value, str):
            raise ValueError("Provider returned non-text output")
        if self.stage == "jev" and value not in ("speak", "silent"):
            raise ValueError("Jev returned an unsupported action")
        if self.stage == "luna" and not value.strip():
            raise ValueError("Luna returned an empty line")
        return Response(
            id=None,
            model=request.model,
            message=Message.assistant(value),
            finish_reason="stop",
            usage=Usage(),
        )


class BernardProgram(dspy.Module):
    def __init__(
        self,
        settings: Settings,
        jev_prompt: str,
        luna_prompt: str,
        post: PostJson = post_json,
    ):
        super().__init__()
        self.errors = RunErrors()
        self.choose = dspy.Predict(ChooseAction.with_instructions(jev_prompt))
        self.write = dspy.Predict(WriteLine.with_instructions(luna_prompt))
        self.jev_lm = dspy.LM(
            "bernard/jev",
            engine=BernardEngine("jev", settings, post, self.errors),
            cache=False,
        )
        self.luna_lm = dspy.LM(
            "bernard/luna",
            engine=BernardEngine("luna", settings, post, self.errors),
            cache=False,
        )

    def forward(self, name1, name2, user_context, call_context, chattiness):
        with dspy.context(lm=self.jev_lm, adapter=BernardAdapter("jev")):
            action = self.choose(
                name1=name1,
                name2=name2,
                chattiness=chattiness,
                call_context=call_context,
            ).action
        if action == "silent":
            return dspy.Prediction(action="silent", line="")
        with dspy.context(lm=self.luna_lm, adapter=BernardAdapter("luna")):
            line = self.write(
                name1=name1,
                name2=name2,
                user_context=user_context,
                call_context=call_context,
            ).line
        return dspy.Prediction(action="speak", line=line)


def case_inputs(case: ReviewCase) -> dict[str, Any]:
    return {
        "name1": case.name1,
        "name2": case.name2,
        "user_context": case.user_context,
        "call_context": case.call_context,
        "chattiness": case.chattiness,
    }


def hard_rule_errors(case: ReviewCase, action: str, line: str) -> list[str]:
    if action not in ("silent", "speak"):
        return ["Unknown action"]
    if action == "silent":
        return ["Silent action has text"] if line else []
    errors = []
    if len(line.split()) > 18:
        errors.append("Line is over 18 words")
    if re.search(
        r"\bBernard\b|\bTR_[A-Za-z0-9]+\b|\b[0-9a-f]{8}-[0-9a-f-]{27,}\b", line, re.I
    ):
        errors.append("Line has an internal name or ID")
    if not line.strip():
        errors.append("Spoken line is empty")
    transcript = case.call_context.split("Full labeled conversation so far:\n", 1)[-1]
    last_line = transcript.strip().splitlines()[-1] if transcript.strip() else ""
    if last_line.rstrip().endswith("?"):
        errors.append("The other dater has not answered a question")
    event = re.search(r"(?:^|\n)Event(?: name)?:\s*([^\n.]+)", case.user_context, re.I)
    if event and event[1].casefold() in line.casefold():
        if event[1].casefold() not in transcript.casefold():
            errors.append("Line names the event before a dater does")
    addressed = re.match(r"([A-Z][a-z]+),", line.strip())
    if addressed:
        first_names = {case.name1.split()[0], case.name2.split()[0]}
        extra_names = set(
            re.findall(r"nickname:\s*([A-Z][a-z]+)", case.user_context, re.I)
        )
        if addressed[1] not in first_names | extra_names:
            errors.append("Line uses a name that was not supplied")
    return errors
