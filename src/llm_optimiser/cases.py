"""Read one human review for one Bernard decision."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

STATE_SPLIT = "\n\nCall context:\n"
LUNA_SUFFIX = (
    "\n\nJev chose speak. Write one short line now. "
    "Return only the spoken line. Do not explain your reasoning."
)
GUIDANCE = {
    1: "Speak only when a question has a clear, strong fit for the call.",
    2: "Speak for clear chances to ask a fresh question. Let small ones pass.",
    3: "Ask fitting questions while leaving room for the daters to lead.",
    4: (
        "Often take a good chance to ask a bold, personal question, "
        "even when the chat is going well."
    ),
    5: (
        "Be eager to ask a fresh, fitting question. Speak for most good chances, "
        "even when the daters are talking easily."
    ),
}


@dataclass(frozen=True)
class ReviewCase:
    source_id: str
    room_id: str
    name1: str
    name2: str
    user_context: str
    call_context: str
    chattiness: int
    original_action: str
    original_line: str
    jev_score: int
    luna_score: int | None
    jev_reason: str
    luna_reason: str

    @property
    def wanted_action(self) -> str:
        if self.jev_score == 1:
            return self.original_action
        return "silent" if self.original_action == "speak" else "speak"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _score(value: Any, required: bool) -> int | None:
    if value is None and not required:
        return None
    if type(value) not in (int, float) or value not in (0, 1):
        raise ValueError("Human review score must be Pass=1 or Fail=0")
    return int(value)


def _reasons(comments: Any) -> dict[str, str]:
    if not isinstance(comments, list):
        return {}
    found: dict[str, str] = {}
    for comment in comments:
        text = comment.get("comment", "") if isinstance(comment, dict) else ""
        for label, reason in re.findall(r"^(Jev|Luna):\s*(.+)$", text, re.MULTILINE):
            if label in found and found[label] != reason.strip():
                raise ValueError("Conflicting review reasons on one row")
            found[label] = reason.strip()
    return found


def case_from_log(row: dict[str, Any]) -> ReviewCase | None:
    metadata = row.get("metadata") or {}
    if not isinstance(metadata, dict) or metadata.get("kind") == "full_call_transcript":
        return None
    trace_input, output = row.get("input") or {}, row.get("output") or {}
    if not isinstance(trace_input, dict) or not isinstance(output, dict):
        return None
    jev_request = trace_input.get("jev_request") or {}
    state = jev_request.get("state")
    action = output.get("action")
    if not isinstance(state, str) or action not in ("speak", "silent"):
        return None
    scores = row.get("scores") or {}
    reasons = _reasons(row.get("comments"))
    for name in ("Jev Choice", "Luna Line"):
        if name in scores:
            _score(scores[name], True)
    if "Jev Choice" not in scores or not reasons.get("Jev"):
        return None
    jev_score = _score(scores["Jev Choice"], True)
    if action == "speak" and ("Luna Line" not in scores or not reasons.get("Luna")):
        return None
    luna_score = _score(scores.get("Luna Line"), action == "speak")
    if not state.startswith("Bernard policy:\n") or STATE_SPLIT not in state:
        raise ValueError("Reviewed decision has an unknown state shape")
    policy, context = state.split(STATE_SPLIT, 1)
    names = re.match(r"Dater names: ([^\n,]+), ([^\n]+)", context)
    level = re.search(r"\nChatty level: ([1-5])/5\.", policy)
    if not names or not level:
        raise ValueError("Reviewed decision lacks names or chatty level")
    user_context = ""
    marker = "Event and participant context:\n"
    if marker in context:
        rest = context.split(marker, 1)[1]
        transcript_marker = "\n\nFull labeled conversation so far:"
        if transcript_marker not in rest:
            raise ValueError("Reviewed decision lacks a transcript boundary")
        user_context = rest.split(transcript_marker, 1)[0]
    line = output.get("line", "") if action == "speak" else ""
    if action == "speak":
        luna_messages = (trace_input.get("luna_request") or {}).get("messages") or []
        if (
            not isinstance(line, str)
            or not line.strip()
            or len(luna_messages) != 2
            or luna_messages[1].get("content") != context + LUNA_SUFFIX
        ):
            raise ValueError("Reviewed Luna call does not match the Jev context")
    room_id = metadata.get("room_id")
    source_id = row.get("id")
    if not isinstance(room_id, str) or not room_id or not isinstance(source_id, str):
        raise ValueError("Reviewed decision lacks a room or row ID")
    return ReviewCase(
        source_id=source_id,
        room_id=room_id,
        name1=names[1].strip(),
        name2=names[2].strip(),
        user_context=user_context,
        call_context=context,
        chattiness=int(level[1]),
        original_action=action,
        original_line=line.strip(),
        jev_score=jev_score,
        luna_score=luna_score,
        jev_reason=reasons["Jev"],
        luna_reason=reasons.get("Luna", ""),
    )


def read_snapshot(path: Path) -> list[ReviewCase]:
    cases = [
        ReviewCase(**json.loads(line)) for line in path.read_text().splitlines() if line
    ]
    ids = [case.source_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Snapshot has duplicate source IDs")
    return cases


def write_snapshot(path: Path, cases: list[ReviewCase]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(
        json.dumps(case.to_dict(), sort_keys=True) + "\n"
        for case in sorted(cases, key=lambda c: c.source_id)
    )
    path.write_text(body)
    path.chmod(0o600)
    return hashlib.sha256(body.encode()).hexdigest()
