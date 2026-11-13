"""Magic-byte sniffing at upload: the client-supplied extension is untrusted,
so the first bytes must agree with it before the file reaches the parser."""

import pytest

from rag_engine.api.routes.ingest import _content_matches_ext


@pytest.mark.parametrize(
    ("ext", "head", "ok"),
    [
        # Real headers.
        (".pdf", b"%PDF-1.7 ...", True),
        (".docx", b"PK\x03\x04\x14\x00\x00\x00...", True),
        # Mislabeled payloads — the actual attack.
        (".pdf", b"PK\x03\x04(zip)", False),  # a zip renamed .pdf
        (".pdf", b"\x7fELF", False),  # an ELF binary renamed .pdf
        (".docx", b"%PDF-1.7", False),  # a pdf renamed .docx
        (".docx", b"#!/bin/sh\nrm -rf /\n", False),
        # Text formats: no fixed magic, but NUL bytes mean binary.
        (".md", b"# Acme Runbook\n\n85 \xc2\xb0 C...\n", True),
        (".html", b"<!DOCTYPE html>\n<html>...", True),
        (".md", b"MZ\x90\x00\x03\x00\x00\x00...", False),  # PE binary as .md
        (".md", b"\x00\x01\x02", False),
        # Pathological shortness: a head shorter than the magic fails closed.
        (".pdf", b"%PD", False),
        (".docx", b"PK", False),
        # Empty upload: passes for text (parser yields "empty"), fails for
        # formats where emptiness is meaningless.
        (".md", b"", True),
        (".html", b"", True),
        (".pdf", b"", False),
        (".docx", b"", False),
    ],
)
def test_content_matches_ext(ext: str, head: bytes, ok: bool):
    assert _content_matches_ext(ext, head) is ok
