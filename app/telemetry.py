"""OpenTelemetry tracing for imaging (T-32-5, B-18).

Off unless OTEL_EXPORTER_OTLP_ENDPOINT is set: then no OpenTelemetry module is imported, no
exporter exists and the app is not instrumented. On, FastAPI requests get a SERVER span that
continues the caller's W3C `traceparent` (the backend's imaging client sends it), exported over
OTLP/HTTP to `<endpoint>/v1/traces` as OTEL_SERVICE_NAME (default `invai-imaging`).

Only the attribute keys in SPAN_ATTRIBUTE_ALLOWLIST leave the process: method, route template,
status, path-free sizes. No URL query strings, headers, storage keys or bodies (buyer text can sit
in a personalization key), and exception events keep only their type.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import FastAPI
    from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor, TracerProvider

SPAN_ATTRIBUTE_ALLOWLIST = frozenset(
    {
        # old and stable HTTP semantic conventions (the instrumentation emits either)
        "http.method",
        "http.request.method",
        "http.route",
        "http.status_code",
        "http.response.status_code",
        "http.scheme",
        "url.scheme",
        "server.port",
        "net.host.port",
        "http.request.body.size",
        "http.response.body.size",
        "http.request_content_length",
        "http.response_content_length",
        "error.type",
        "exception.type",
    }
)

DEFAULT_SERVICE_NAME = "invai-imaging"


def tracing_endpoint() -> str | None:
    value = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    return value if value.startswith(("http://", "https://")) else None


def _allowed(attrs: Any) -> dict:
    return {k: v for k, v in (attrs or {}).items() if k in SPAN_ATTRIBUTE_ALLOWLIST}


def allowlist_span(span: ReadableSpan) -> ReadableSpan:
    """A copy of the span with only allowlisted attributes and no status description."""
    from opentelemetry.sdk.trace import Event, ReadableSpan
    from opentelemetry.trace import Link, Status

    return ReadableSpan(
        name=span.name,
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes=_allowed(span.attributes),
        events=tuple(Event(e.name, _allowed(e.attributes), e.timestamp) for e in span.events),
        links=tuple(Link(link.context, _allowed(link.attributes)) for link in span.links),
        kind=span.kind,
        status=Status(span.status.status_code),
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


def allowlist_exporter(inner: Any) -> Any:
    """Wraps a SpanExporter so it only ever sees allowlisted spans."""
    from opentelemetry.sdk.trace.export import SpanExporter

    class AllowlistExporter(SpanExporter):
        def export(self, spans):
            return inner.export([allowlist_span(s) for s in spans])

        def shutdown(self):
            return inner.shutdown()

        def force_flush(self, timeout_millis: int = 30000):
            return inner.force_flush(timeout_millis)

    return AllowlistExporter()


def setup_tracing(app: FastAPI, processor: SpanProcessor | None = None) -> TracerProvider | None:
    """Instruments `app` when tracing is on (or a test passes a processor); else does nothing."""
    endpoint = tracing_endpoint()
    if processor is None and endpoint is None:
        return None

    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    if processor is None:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        exporter = OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces")
        processor = BatchSpanProcessor(allowlist_exporter(exporter))
    name = os.environ.get("OTEL_SERVICE_NAME", "").strip() or DEFAULT_SERVICE_NAME
    provider = TracerProvider(resource=Resource.create({"service.name": name}))
    provider.add_span_processor(processor)
    FastAPIInstrumentor.instrument_app(
        app,
        tracer_provider=provider,
        excluded_urls="/health",
        exclude_spans=["receive", "send"],
    )
    return provider
