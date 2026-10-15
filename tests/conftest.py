import pytest


@pytest.fixture(autouse=True)
def use_fake_keys(monkeypatch):
    for name in ("BRAINTRUST_API_KEY", "TYPESAFE_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(name, "fake-test-key")
