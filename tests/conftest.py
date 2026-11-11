import pytest

from local_llm import catalog


@pytest.fixture(autouse=True)
def macos_catalog(monkeypatch):
    """The gateway catalog as macOS serves it, on every machine: most tests describe that full
    list. On Linux the MLX-only models drop out; test_gateway checks that view explicitly."""
    monkeypatch.setattr(catalog, "PLATFORM", "darwin")
