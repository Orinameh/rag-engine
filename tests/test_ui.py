"""The /ui test page: served exactly when the docs conveniences are on, and
self-contained by construction."""

from pathlib import Path

from rag_engine.api.app import app
from rag_engine.config import get_settings

UI_HTML = Path(__file__).parents[1] / "src" / "rag_engine" / "api" / "static" / "ui.html"


def test_ui_served_only_with_docs_enabled():
    """/ui rides the same DOCS_ENABLED toggle as /docs — one switch for all
    unauthenticated dev conveniences, nothing extra to reason about."""
    served = any(getattr(r, "path", None) == "/ui" for r in app.routes)
    assert served is get_settings().docs_enabled


def test_ui_is_self_contained():
    html = UI_HTML.read_text()
    # No build step, no CDN: a single file must actually be single.
    assert "<script src" not in html
    assert "<link " not in html
    assert "@import" not in html


def test_ui_never_uses_innerhtml():
    # Document content is untrusted data; every render path must go through
    # textContent/createElement so corpus text can't execute as markup.
    # Match call sites (".innerHTML"), not prose — the header comment mentions
    # the property by name.
    html = UI_HTML.read_text()
    assert ".innerHTML" not in html
    assert ".insertAdjacentHTML" not in html
