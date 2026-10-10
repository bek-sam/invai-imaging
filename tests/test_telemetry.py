"""T-32-5: imaging continues the caller's trace, exports only allowlisted attributes, and is
untouched when tracing is off."""

from fastapi import FastAPI
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind

from app.guard import GuardMiddleware
from app.telemetry import SPAN_ATTRIBUTE_ALLOWLIST, allowlist_exporter, setup_tracing

TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
PARENT_ID = "00f067aa0ba902b7"


def traced_app():
    app = FastAPI()
    app.add_middleware(GuardMiddleware)

    @app.post("/compose")
    def compose(key: str = ""):
        raise ValueError(f"cannot read {key}")

    @app.get("/health")
    def health():
        return {"ok": True}

    memory = InMemorySpanExporter()
    provider = setup_tracing(app, processor=SimpleSpanProcessor(allowlist_exporter(memory)))
    return app, memory, provider


def test_off_without_endpoint(monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    app = FastAPI()
    assert setup_tracing(app) is None
    assert not getattr(app, "_is_instrumented_by_opentelemetry", False)


def test_continues_incoming_traceparent_with_allowlisted_attributes(monkeypatch):
    from app.config import settings

    secret = settings.shared_secrets()[0]
    app, memory, provider = traced_app()
    client = TestClient(app, raise_server_exceptions=False)
    res = client.post(
        "/compose?key=personalization/MARIA-GONZALEZ.png&email=maria@example.com",
        headers={"traceparent": f"00-{TRACE_ID}-{PARENT_ID}-01", "x-imaging-secret": secret},
    )
    assert res.status_code == 500
    client.get("/health")
    spans = memory.get_finished_spans()
    server = [s for s in spans if s.kind == SpanKind.SERVER]
    assert len(server) == 1  # /health is not traced
    span = server[0]
    assert format(span.context.trace_id, "032x") == TRACE_ID
    assert format(span.parent.span_id, "016x") == PARENT_ID
    assert span.attributes.get("http.route") == "/compose"
    for s in spans:
        for key in list(s.attributes) + [k for e in s.events for k in (e.attributes or {})]:
            assert key in SPAN_ATTRIBUTE_ALLOWLIST, key
        assert not s.status.description
    dump = repr([(s.attributes, [e.attributes for e in s.events]) for s in spans])
    assert "MARIA" not in dump and "example.com" not in dump
    provider.shutdown()
