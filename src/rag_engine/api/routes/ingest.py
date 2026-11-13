import hashlib
import uuid
from pathlib import Path

from celery.result import AsyncResult
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from rag_engine.api.auth import AdminDep
from rag_engine.api.ownership import record_owner
from rag_engine.config import get_settings
from rag_engine.worker import ingest_document

router = APIRouter(prefix="/ingest", tags=["ingest"])

UPLOAD_DIR = Path("/tmp/rag_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_CHUNK = 1 << 20  # 1 MB read chunks
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".html", ".htm", ".md"}

# Magic bytes for formats that have them. The extension comes from the
# client-supplied filename, so it is untrusted — these checks make a file
# named "report.pdf" that is really a ZIP or ELF fail at the API (415)
# instead of costing the worker a parse attempt.
_MAGIC_HEADERS = {
    ".pdf": b"%PDF",
    ".docx": b"PK\x03\x04",  # ZIP local file header (OOXML container)
}


def _content_matches_ext(ext: str, head: bytes) -> bool:
    """True when the first bytes agree with the claimed extension.

    Formats without a fixed magic (.html/.htm/.md) are validated negatively:
    binary payloads masquerading as text almost always contain NUL bytes in
    their head. An empty head (0-byte upload) passes for text formats — the
    parser turns it into a clean "empty" result — but fails for .pdf/.docx,
    which cannot be meaningfully empty.
    """
    magic = _MAGIC_HEADERS.get(ext)
    if magic is not None:
        # head may be shorter than the magic for a pathologically tiny file;
        # startswith then fails, which is the safe answer.
        return head.startswith(magic)
    return b"\x00" not in head


class IngestResponse(BaseModel):
    task_id: str
    status_url: str


@router.post("", response_model=IngestResponse, status_code=202)
async def ingest(
    caller: AdminDep,
    file: UploadFile = File(...),  # noqa: B008 — FastAPI idiom for multipart parameters
    title: str = Form(...),
    authority: str = Form("unknown"),
) -> IngestResponse:
    s = get_settings()
    workspace = caller.workspace

    # Only the suffix is used from the client-supplied name; the stored filename
    # is derived from the content hash, so path traversal is impossible and
    # identical content in one workspace always maps to the same source_uri
    # (idempotent re-ingest). The workspace prefix keeps tenants' file trees
    # separate, so two workspaces uploading identical content both get indexed.
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type {ext!r}. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    digest = hashlib.sha256()
    size = 0
    tmp = UPLOAD_DIR / workspace / f".tmp-{uuid.uuid4()}{ext}"
    try:
        tmp.parent.mkdir(parents=True, exist_ok=True)
        first = True
        with tmp.open("wb") as out:
            while chunk := await file.read(UPLOAD_CHUNK):
                size += len(chunk)
                if size > s.upload_max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {s.upload_max_bytes // (1024 * 1024)} MB upload limit",
                    )
                if first:
                    # Content sniff on the very first chunk, before anything is
                    # kept: the extension was client-supplied, bytes are the
                    # truth. tmp is unlinked by the handler below.
                    if not _content_matches_ext(ext, chunk):
                        raise HTTPException(
                            status_code=415,
                            detail=f"File content does not look like {ext!r} "
                            "(magic-byte mismatch)",
                        )
                    first = False
                digest.update(chunk)
                out.write(chunk)
        dest = UPLOAD_DIR / workspace / f"{digest.hexdigest()}{ext}"
        tmp.rename(dest)
    except HTTPException:
        tmp.unlink(missing_ok=True)
        raise
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    # .apply_async() is the explicit Celery API; .delay() is just a shorthand
    # that type checkers often can't resolve due to incomplete Celery stubs.
    task: AsyncResult = ingest_document.apply_async(
        args=(str(dest), title, authority, workspace)
    )
    # Tenancy stamping: tasks polls must belong to the dispatching workspace
    # (api/ownership.py). Best-effort: a Redis hiccup yields an un-pollable
    # task rather than a pollable-by-anyone task (fail-closed).
    await record_owner(task.id, workspace)
    return IngestResponse(task_id=task.id, status_url=f"/tasks/{task.id}")
