"""Keys never exist as plaintext beyond .env parsing."""

import json

import pytest

from rag_engine.api.auth import key_digest
from rag_engine.config import get_settings

KEYS = {
    "sk-acme-admin": {"workspace": "acme", "role": "admin"},
    "sk-acme-reader": {"workspace": "acme", "role": "reader"},
}


@pytest.fixture()
def _keys_env(monkeypatch):
    monkeypatch.setenv("RAG_API_KEYS", json.dumps(KEYS))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_stored_keys_are_digests(_keys_env):
    keys = get_settings().api_keys
    assert set(keys.keys()) == {key_digest(k) for k in KEYS}
    assert "sk-acme-admin" not in str(keys)
    assert "sk-acme-reader" not in str(keys)
    # Entries survive hashing intact.
    assert keys[key_digest("sk-acme-admin")].workspace == "acme"


def test_settings_repr_carries_no_keys(_keys_env):
    r = repr(get_settings())
    assert "api_keys" not in r
    assert "sk-acme-admin" not in r and "sk-acme-reader" not in r


def test_blank_keys_are_dropped(monkeypatch):
    monkeypatch.setenv(
        "RAG_API_KEYS",
        json.dumps({"sk-real": {"workspace": "acme"}, "": {"workspace": "ghost"}}),
    )
    get_settings.cache_clear()
    keys = get_settings().api_keys
    assert set(keys.keys()) == {key_digest("sk-real")}
    get_settings.cache_clear()


def test_malformed_entry_rejected(monkeypatch):
    monkeypatch.setenv(
        "RAG_API_KEYS",
        json.dumps({"sk-bad": {"workspace": "acme", "role": "boss"}}),
    )
    get_settings.cache_clear()
    # Validation is lazy — it happens when the map is materialized (which the
    # app forces at startup via app.py's lifespan), so the assertion touches it.
    with pytest.raises(Exception, match=r"(?i)role|input"):
        api_keys = get_settings().api_keys
        assert isinstance(api_keys, dict)  # parsed inside pytest.raises only
    get_settings.cache_clear()


def test_digest_is_not_a_replayable_credential(_keys_env):
    """Knowing a stored digest must not authenticate as the original key."""
    from fastapi import HTTPException

    from rag_engine.api.auth import authenticate

    for k in KEYS:
        with pytest.raises(HTTPException):
            authenticate(key_digest(k))
