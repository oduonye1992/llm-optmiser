import json
from pathlib import Path

import pytest

from llm_optimiser.cases import case_from_log, read_snapshot, write_snapshot


def log_row(action="speak", *, complete=True):
    context = (
        "Dater names: Ava, Leo\n\nCall time is unknown.\n\n"
        "Event and participant context:\nAva likes chess.\n\n"
        "Full labeled conversation so far:\nAva: I like chess."
    )
    row = {
        "id": "source-1",
        "metadata": {"room_id": "room-1"},
        "input": {
            "jev_request": {
                "state": (
                    "Bernard policy:\nUse names.\nChatty level: 4/5. "
                    "Often ask.\n\nCall context:\n"
                )
                + context
            },
        },
        "output": {"action": action, "line": "Ava, who wins at chess?"},
        "scores": {"Jev Choice": 1, "Luna Line": 0},
        "comments": [{"comment": "Jev: Good time to ask.\nLuna: Too plain."}],
    }
    if action == "speak":
        row["input"]["luna_request"] = {
            "messages": [
                {"role": "system", "content": "prompt"},
                {
                    "role": "user",
                    "content": context
                    + "\n\nJev chose speak. Write one short line now. "
                    "Return only the spoken line. Do not explain your reasoning.",
                },
            ]
        }
    if not complete:
        row["comments"] = [{"comment": "Jev: Good time to ask."}]
    return row


def test_review_requires_both_reasons_for_spoken_line():
    assert case_from_log(log_row(complete=False)) is None
    case = case_from_log(log_row())
    assert case is not None
    assert case.wanted_action == "speak"
    assert case.luna_score == 0
    assert case.user_context == "Ava likes chess."


def test_silent_row_only_needs_jev_review():
    row = log_row("silent", complete=False)
    row["scores"].pop("Luna Line")
    assert case_from_log(row).luna_score is None


def test_reject_transcript_invalid_score_and_conflict():
    row = log_row()
    row["metadata"]["kind"] = "full_call_transcript"
    assert case_from_log(row) is None
    row = log_row()
    row["scores"]["Jev Choice"] = 0.6
    with pytest.raises(ValueError, match="Pass=1"):
        case_from_log(row)
    row = log_row(complete=False)
    row["scores"]["Luna Line"] = 0.4
    with pytest.raises(ValueError, match="Pass=1"):
        case_from_log(row)
    row = log_row()
    row["comments"].append({"comment": "Jev: Wrong time."})
    with pytest.raises(ValueError, match="Conflicting"):
        case_from_log(row)


def test_snapshot_is_stable_and_private(tmp_path: Path):
    case = case_from_log(log_row())
    path = tmp_path / "data" / "reviews.jsonl"
    first = write_snapshot(path, [case])
    second = write_snapshot(path, [case])
    assert first == second
    assert read_snapshot(path) == [case]
    assert path.stat().st_mode & 0o077 == 0
    path.write_text(path.read_text() + json.dumps(case.to_dict()) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_snapshot(path)
