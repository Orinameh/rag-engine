"""Tests for the simple API-key auth + workspace scoping."""

import json

import pytest
from fastapi import HTTPException

from rag_engine.api.auth import (
    authenticate,
    require_admin,
    require_caller,
    scoped_metadata_filter,
)
from rag_engine.config import get_settings

KEYS = {
    "sk-acme-admin": {"workspace": "acme", "role": "admin"},
    "sk-acme-reader": {"workspace": "acme", "role": "reader"},
    "sk-globex-reader": {"workspace": "globex", "role": "reader"},
}


@pytest.fixture()
def _keys_env(monkeypatch):
    monkeypatch.setenv("RAG_API_KEYS", json.dumps(KEYS))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ---------- key resolution ----------


@pytest.mark.usefixtures("_keys_env")
def test_authenticate_resolves_workspace_and_role():
    admin = authenticate("sk-acme-admin")
    assert admin.workspace == "acme" and admin.role == "admin"
    reader = authenticate("sk-globex-reader")
    assert reader.workspace == "globex" and reader.role == "reader"


@pytest.mark.usefixtures("_keys_env")
def test_unknown_key_is_401():
    with pytest.raises(HTTPException) as err:
        authenticate("sk-nope")
    assert err.value.status_code == 401


def test_no_keys_configured_is_401():
    # Default settings: api_keys == {} → lockout, not a free-for-all.
    get_settings.cache_clear()
    with pytest.raises(HTTPException) as err:
        authenticate("anything")
    assert err.value.status_code == 401


# ---------- transport headers ----------


@pytest.mark.anyio
@pytest.mark.usefixtures("_keys_env")
async def test_bearer_header_accepted():
    caller = await require_caller(authorization="Bearer sk-acme-admin", x_api_key=None)
    assert caller.workspace == "acme"


@pytest.mark.anyio
@pytest.mark.usefixtures("_keys_env")
async def test_x_api_key_header_accepted():
    caller = await require_caller(authorization=None, x_api_key="sk-acme-reader")
    assert caller.role == "reader"


@pytest.mark.anyio
async def test_missing_credentials_is_401():
    with pytest.raises(HTTPException) as err:
        await require_caller(authorization=None, x_api_key=None)
    assert err.value.status_code == 401


@pytest.mark.anyio
@pytest.mark.usefixtures("_keys_env")
async def test_wrong_scheme_is_401():
    with pytest.raises(HTTPException) as err:
        await require_caller(authorization="Basic sk-acme-admin", x_api_key=None)
    assert err.value.status_code == 401


# ---------- roles ----------


@pytest.mark.anyio
@pytest.mark.usefixtures("_keys_env")
async def test_admin_passes_require_admin():
    caller = await require_admin(
        await require_caller(authorization="Bearer sk-acme-admin", x_api_key=None)
    )
    assert caller.role == "admin"


@pytest.mark.anyio
@pytest.mark.usefixtures("_keys_env")
async def test_reader_is_403_on_ingest():
    caller = await require_caller(authorization=None, x_api_key="sk-acme-reader")
    with pytest.raises(HTTPException) as err:
        await require_admin(caller)
    assert err.value.status_code == 403


# ---------- workspace scoping ----------


@pytest.mark.usefixtures("_keys_env")
def test_scoped_filter_forces_caller_workspace():
    out = scoped_metadata_filter({"authority": "official"}, "acme")
    assert out == {"authority": "official", "workspace": "acme"}


@pytest.mark.usefixtures("_keys_env")
def test_client_cannot_widen_into_another_workspace():
    # The single most important property: whatever the client sends for the
    # workspace key, the server's value wins.
    out = scoped_metadata_filter({"workspace": "globex"}, "acme")
    assert out["workspace"] == "acme"


@pytest.mark.usefixtures("_keys_env")
def test_scoped_filter_on_none():
    assert scoped_metadata_filter(None, "acme") == {"workspace": "acme"}
