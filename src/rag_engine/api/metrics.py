"""API metrics: in-process counters + latency summaries, Prometheus format.

Implemented without a prometheus-client dependency — the exposition is plain
text. Counters live in this process's memory (per replica; aggregating
across replicas is the networking backend's job, standard for counter
rescrape semantics). Memory is bounded: dynamic path segments collapse to
templates (/tasks/<uuid> → /tasks/{id}) and latency samples are capped.
"""

import re
import time
from collections import defaultdict

_uuid_re = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
)

_requests: dict[tuple[str, str, str], int] = defaultdict(int)
_latencies: dict[tuple[str, str], list[float]] = defaultdict(list)
_MAX_SAMPLES = 5000  # per route, bounded memory
_started_at = time.monotonic()


def path_template(path: str) -> str:
    """Collapses dynamic segments so counter cardinality stays bounded."""
    return _uuid_re.sub("{id}", path)


def record(method: str, path: str, status: int, seconds: float) -> None:
    rank = (method, path_template(path))
    _requests[(method, path_template(path), str(status))] += 1
    bucket = _latencies[rank]
    if len(bucket) < _MAX_SAMPLES:
        bucket.append(max(0.0, seconds))


def build_exposition(queue_depth: int | None = None) -> str:
    """Prometheus text exposition (format 0.0.4)."""
    lines: list[str] = [
        "# HELP rag_engine_requests_total HTTP requests served",
        "# TYPE rag_engine_requests_total counter",
    ]
    for (method, path, status), count in sorted(_requests.items()):
        labels = f'method="{method}",path="{path}",status="{status}"'
        lines.append(f"rag_engine_requests_total{{{labels}}} {count}")

    lines.append(
        "# HELP rag_engine_request_latency_seconds Per-route latency (last N samples)"
    )
    lines.append("# TYPE rag_engine_request_latency_seconds summary")
    for rank, samples in sorted(_latencies.items()):
        if not samples:
            continue
        method, path = rank
        labels = f'method="{method}",path="{path}"'
        lines.append(
            f'rag_engine_request_latency_seconds_sum{{{labels}}} {sum(samples):.4f}'
        )
        lines.append(
            f'rag_engine_request_latency_seconds_count{{{labels}}} {len(samples)}'
        )
        ordered = sorted(samples)
        for q in (0.5, 0.9, 0.99):
            idx = min(round(q * (len(ordered) - 1)), len(ordered) - 1)
            lines.append(
                f'rag_engine_request_latency_seconds_quantile{{{labels},'
                f'quantile="{q}"}} {ordered[idx]:.4f}'
            )

    lines.append("# HELP rag_engine_celery_queue_depth Ingest tasks waiting to run")
    lines.append("# TYPE rag_engine_celery_queue_depth gauge")
    lines.append(f"rag_engine_celery_queue_depth {queue_depth or 0}")

    lines.append("# HELP rag_engine_uptime_seconds Process uptime")
    lines.append("# TYPE rag_engine_uptime_seconds gauge")
    lines.append(f"rag_engine_uptime_seconds {time.monotonic() - _started_at:.1f}")
    return "\n".join(lines) + "\n"
