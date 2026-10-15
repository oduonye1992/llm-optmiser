"""Copy complete human reviews into a private Braintrust dataset."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import braintrust

from llm_optimiser.cases import ReviewCase, case_from_log, write_snapshot
from llm_optimiser.config import Settings

API = "https://api.braintrust.dev/v1"


def api_json(key: str, method: str, path: str, body: dict | None = None) -> dict:
    if not key:
        raise ValueError("BRAINTRUST_API_KEY is missing")
    request = urllib.request.Request(
        API + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.loads(response.read())
    if not isinstance(result, dict):
        raise ValueError("Braintrust returned a non-object response")
    return result


ApiJson = Callable[[str, str, str, dict | None], dict]


def project_id(settings: Settings, api: ApiJson = api_json) -> str:
    query = urllib.parse.urlencode({"project_name": settings.project, "limit": 100})
    projects = api(settings.braintrust_key, "GET", f"/project?{query}", None).get(
        "objects", []
    )
    matches = [p for p in projects if p.get("name") == settings.project]
    if len(matches) != 1:
        raise ValueError("Braintrust project was not found by name")
    return matches[0]["id"]


def fetch_logs(settings: Settings, api: ApiJson = api_json) -> Iterator[dict[str, Any]]:
    project_identifier = project_id(settings, api)
    cursor: str | None = None
    seen: set[str] = set()
    cursors: set[str] = set()
    while True:
        page = api(
            settings.braintrust_key,
            "POST",
            f"/project_logs/{project_identifier}/fetch",
            {"limit": 100, **({"cursor": cursor} if cursor else {})},
        )
        events = page.get("events")
        if not isinstance(events, list):
            raise ValueError("Braintrust fetch response lacks events")
        for row in events:
            if not isinstance(row, dict):
                raise ValueError("Braintrust returned a malformed log row")
            row_identifier = row.get("id")
            if row_identifier and row_identifier not in seen:
                seen.add(row_identifier)
                yield row
        cursor = page.get("cursor")
        if not cursor:
            break
        if cursor in cursors:
            raise ValueError("Braintrust repeated a page cursor")
        cursors.add(cursor)


def select_reviews(
    rows: Iterator[dict[str, Any]],
) -> tuple[list[ReviewCase], dict[str, int]]:
    cases: list[ReviewCase] = []
    counts = {"seen": 0, "reviewed": 0, "skipped": 0}
    ids: set[str] = set()
    for row in rows:
        counts["seen"] += 1
        case = case_from_log(row)
        if case is None:
            counts["skipped"] += 1
            continue
        if case.source_id in ids:
            raise ValueError("Duplicate reviewed source ID")
        ids.add(case.source_id)
        cases.append(case)
        counts["reviewed"] += 1
    return cases, counts


def sync(settings: Settings, snapshot: Path) -> dict[str, Any]:
    cases, counts = select_reviews(fetch_logs(settings))
    if not cases:
        return {**counts, "dataset_version": None, "snapshot": None}
    dataset = braintrust.init_dataset(
        project=settings.project,
        name=settings.dataset,
        api_key=settings.braintrust_key,
    )
    for case in cases:
        dataset.insert(
            id=case.source_id,
            input={
                "name1": case.name1,
                "name2": case.name2,
                "user_context": case.user_context,
                "call_context": case.call_context,
                "chattiness": case.chattiness,
            },
            expected={
                "wanted_action": case.wanted_action,
                "original_action": case.original_action,
                "original_line": case.original_line,
                "jev_score": case.jev_score,
                "luna_score": case.luna_score,
                "jev_reason": case.jev_reason,
                "luna_reason": case.luna_reason,
            },
            metadata={"source_id": case.source_id, "room_id": case.room_id},
        )
    dataset.flush()
    version = write_snapshot(snapshot, cases)
    braintrust_version = dataset.version
    metadata = {
        "snapshot_sha256": version,
        "braintrust_version": braintrust_version,
        "dataset_id": dataset.id,
    }
    sidecar = snapshot.with_suffix(snapshot.suffix + ".meta.json")
    sidecar.write_text(json.dumps(metadata, indent=2))
    sidecar.chmod(0o600)
    return {
        **counts,
        "snapshot_sha256": version,
        "braintrust_version": braintrust_version,
        "snapshot": str(snapshot),
    }
