"""Tests for markdown parsing (header-based section split)."""

from rag_engine.ingestion.parsers import _parse_markdown

DEEP_DOC = """# Root Title

Root intro text.

## Deployment

Rollback uses the scripts/rollback.sh helper.

#### Deep detail

The nested deep detail sentence lives here.

##### Deeper still

And this is a fifth-level section all by itself.
"""


def test_all_heading_levels_split():
    blocks = _parse_markdown(_write(DEEP_DOC))
    headings = [
        (b["metadata"].get("h1"), b["metadata"].get("h2"),
         b["metadata"].get("h4") or b["metadata"].get("h5"))
        for b in blocks
    ]
    # Section with an h4-level heading must exist and inherit h1/h2 context.
    assert any(h1 == "Root Title" and h2 == "Deployment" and h4 == "Deep detail"
               for h1, h2, h4 in headings)


def test_deep_section_keeps_heading_text_in_content():
    blocks = _parse_markdown(_write(DEEP_DOC))
    deep = next(b for b in blocks if b["metadata"].get("h5") == "Deeper still")
    # strip_headers=False: the heading words stay inside chunk content, so
    # BM25/embeddings can match queries naming the section title itself.
    assert "Deeper still" in deep["text"]


def test_unheaded_document_is_single_block():
    blocks = _parse_markdown(_write("just some plain text, one section\nsecond line\n"))
    assert len(blocks) == 1
    assert "second line" in blocks[0]["text"]


def _write(text: str) -> str:
    import tempfile

    fd, name = tempfile.mkstemp(suffix=".md")
    with open(fd, "w") as f:
        f.write(text)
    return name
