from dataclasses import replace

from test_cases import log_row

from llm_optimiser.braintrust_sync import fetch_logs, select_reviews, sync
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


def test_sync_keeps_source_id_on_repeat(monkeypatch, tmp_path):
    import llm_optimiser.braintrust_sync as module

    settings = replace(load_settings(), braintrust_key="fake")
    writes = []

    class FakeDataset:
        id = "fake-dataset"
        version = "77"

        def insert(self, **kwargs):
            writes.append(kwargs)

        def flush(self):
            pass

    monkeypatch.setattr(module, "fetch_logs", lambda settings: iter([log_row()]))
    monkeypatch.setattr(
        module.braintrust, "init_dataset", lambda **kwargs: FakeDataset()
    )
    path = tmp_path / "data" / "reviews.jsonl"
    first = sync(settings, path)
    second = sync(settings, path)
    assert first["snapshot_sha256"] == second["snapshot_sha256"]
    assert [item["id"] for item in writes] == ["source-1", "source-1"]
    assert first["braintrust_version"] == "77"
    assert path.with_suffix(".jsonl.meta.json").exists()
