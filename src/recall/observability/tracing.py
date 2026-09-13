"""OpenTelemetry tracing bootstrap and span helpers for Recall."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.trace import Tracer

_initialized = False


def init_tracing(
    service_name: str = "recall-kit",
    enabled: bool = True,
    otlp_endpoint: str | None = None,
    extra_exporter: SpanExporter | None = None,
) -> None:
    """Initializes the global tracer provider once per process."""
    global _initialized
    if _initialized or not enabled:
        return

    # CI sets OTEL_SDK_DISABLED=true globally; explicit init must re-enable the SDK.
    os.environ["OTEL_SDK_DISABLED"] = "false"

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    if otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        endpoint = otlp_endpoint.rstrip("/")
        if not endpoint.endswith("/v1/traces"):
            endpoint = f"{endpoint}/v1/traces"
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))

    if extra_exporter is not None:
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        provider.add_span_processor(SimpleSpanProcessor(extra_exporter))

    trace.set_tracer_provider(provider)
    _initialized = True


def reset_tracing() -> None:
    """Resets tracing state for tests."""
    global _initialized
    _initialized = False
    try:
        from opentelemetry import trace as ot_trace

        once = getattr(ot_trace, "_TRACER_PROVIDER_SET_ONCE", None)
        if once is not None and hasattr(once, "_done"):
            once._done = False  # type: ignore[attr-defined]
        provider = trace.get_tracer_provider()
        if isinstance(provider, TracerProvider):
            provider.shutdown()
    except Exception:
        pass


def get_tracer(name: str) -> Tracer:
    return trace.get_tracer(name)


@contextmanager
def trace_span(
    tracer: Tracer,
    name: str,
    attributes: dict[str, Any] | None = None,
) -> Iterator[Any]:
    """Creates a span and attaches optional attributes."""
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                if value is not None:
                    span.set_attribute(key, value)
        yield span
