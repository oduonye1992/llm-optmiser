from dataclasses import replace

from test_cases import log_row

from llm_optimiser.braintrust_sync import fetch_logs, select_reviews
from llm_optimiser.config import load_settings


def test_fetch_pages_drops_old_row_versions():
    settings = replace(load_settings(), braintrust_key="fake")
    calls = []

    def api(key, method, path, body):
        calls.append((method, path, body))
        if path.startswith("/project?"):
            return {"objects": [{"name": settings.project, "id": "project-1"}]}
        if body.get("cursor") == "next":
            return {"events": [log_row()], "cursor": None}
        return {"events": [log_row()], "cursor": "next"}

    rows = list(fetch_logs(settings, api))
    cases, counts = select_reviews(iter(rows))
    assert len(cases) == 1
    assert counts == {"seen": 1, "reviewed": 1, "skipped": 0}
    assert len(calls) == 3
    assert calls[1][1] == calls[2][1] == "/project_logs/project-1/fetch"
