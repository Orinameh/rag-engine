"""Structured (JSON) logging for application loggers.

Formats every rag_engine.* log record — and uvicorn's own handlers, which the
configure function also reaches — as one JSON line:
``{"ts": ..., "level": ..., "logger": ..., "msg": ..., "exc": ...}`` so
container log shippers (Loki, Beats, CloudWatch) parse without regex.
"""

import json
import logging
from datetime import UTC, datetime


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, UTC)
            .isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_structured_logging(level: int = logging.INFO) -> None:
    """Swap the root handler formatting to JSON (idempotent, up to one line)."""
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler()
        root.addHandler(handler)
    for handler in root.handlers:
        handler.setFormatter(JsonFormatter())
    root.setLevel(level)
    # Uvicorn installs its own handlers AFTER app import; keep them honest.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        for handler in lg.handlers:
            handler.setFormatter(JsonFormatter())
