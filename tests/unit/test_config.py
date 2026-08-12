"""Settings behaviour for the shared spine.

`_env_file=None` keeps these tests deterministic: a developer's real .env
must never change whether they pass.
"""

import pytest
from pydantic import ValidationError

from mesh.models.config import Settings


def test_route_confidence_threshold_is_read_from_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("MESH_ROUTE_CONFIDENCE_THRESHOLD", "0.75")

    settings = Settings(_env_file=None)

    assert settings.route_confidence_threshold == 0.75


def test_chroma_location_defaults_to_the_compose_service(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("CHROMA_HOST", raising=False)
    monkeypatch.delenv("CHROMA_PORT", raising=False)

    settings = Settings(_env_file=None)

    assert (settings.chroma_host, settings.chroma_port) == ("localhost", 8001)


def test_missing_api_key_is_rejected_at_construction(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
