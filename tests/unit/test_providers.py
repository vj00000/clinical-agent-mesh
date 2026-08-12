"""Provider wiring.

These construct clients only — no network calls, so they run in CI without a key.
"""

from mesh.models.config import Settings
from mesh.models.providers import build_chat_model, build_embeddings


def _settings(monkeypatch) -> Settings:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("MESH_CHAT_MODEL", "gpt-5-nano")
    monkeypatch.setenv("MESH_EMBED_MODEL", "text-embedding-3-small")
    return Settings(_env_file=None)


def test_chat_model_uses_the_configured_model_name(monkeypatch):
    model = build_chat_model(_settings(monkeypatch))

    assert model.model_name == "gpt-5-nano"


def test_chat_model_retries_transient_failures(monkeypatch):
    model = build_chat_model(_settings(monkeypatch))

    assert model.max_retries == 3


def test_embeddings_use_the_configured_model_name(monkeypatch):
    embeddings = build_embeddings(_settings(monkeypatch))

    assert embeddings.model == "text-embedding-3-small"
