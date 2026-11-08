"""Metrics exposition builder (pure, no I/O)."""

from rag_engine.api import metrics as m


def test_records_group_by_route_template():
    m.record("GET", "/tasks/4b1f1e2e-aaaa-4bbb-8ccc-dddd0000eeee", 200, 0.01)
    m.record("GET", "/tasks/4b1f1e2e-aaaa-4bbb-8ccc-dddd0000eeee", 200, 0.03)
    m.record("GET", "/health", 200, 0.002)
    out = m.build_exposition(queue_depth=2)
    # UUIDs collapse into one route template; counter aggregates both.
    assert 'path="/tasks/{id}",status="200"}' in out
    assert 'method="GET",path="/health",status="200"' in out
    # gauge + uptime present
    assert "rag_engine_celery_queue_depth 2" in out
    assert "rag_engine_uptime_seconds " in out
    # latency summary includes count for the recorded route
    assert "rag_engine_request_latency_seconds_count{" in out
    # quantiles labeled
    assert 'quantile="0.9"' in out


def test_exposition_never_prints_a_bearer_token():
    m.record("POST", "/query", 200, 0.5)
    out = m.build_exposition(0)
    assert "sk-" not in out
