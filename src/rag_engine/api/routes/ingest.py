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
        with tmp.open("wb") as out:
            while chunk := await file.read(UPLOAD_CHUNK):
                size += len(chunk)
                if size > s.upload_max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {s.upload_max_bytes // (1024 * 1024)} MB upload limit",
                    )
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
