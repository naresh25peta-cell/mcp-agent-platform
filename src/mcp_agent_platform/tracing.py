"""
tracing.py — Phase 5 observability: OpenTelemetry spans for a run.

What is traced (names follow the OpenTelemetry GenAI semantic conventions
where one exists, e.g. ``gen_ai.operation.name`` / ``gen_ai.tool.name``):

* ``ask``                 — one span per question, parent of everything below
* ``guardrail.<stage>``   — each of the four guardrail checks, with
                            ``guardrail.layer`` / ``guardrail.allowed``
* ``execute_tool <name>`` — each MCP tool call the supervisor makes

Privacy: spans record **lengths and names only** — never the question text,
the answer, tool arguments or tool results — so traces are safe to export.

Zero cost when unused: this module only depends on ``opentelemetry-api``
at runtime. Without a configured SDK provider every span is a no-op.
Call :func:`setup_tracing` (or set ``MCP_TRACE=console``) to actually see
spans; pass an exporter to send them somewhere else (OTLP, Langfuse's OTel
endpoint, ...).

Not covered yet: the LLM calls themselves (token counts, latency per model
request). That needs a LangChain callback handler — see the README roadmap.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

from opentelemetry import trace

TRACER_NAME = "mcp_agent_platform"


def get_tracer() -> trace.Tracer:
    return trace.get_tracer(TRACER_NAME)


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[trace.Span]:
    """Start a span as the current span. A no-op if no SDK is configured."""
    with get_tracer().start_as_current_span(name, attributes=attributes) as s:
        yield s


def setup_tracing(exporter: Any | None = None):
    """Install a global TracerProvider (needs the ``opentelemetry-sdk`` package).

    ``exporter`` is any OpenTelemetry ``SpanExporter``. Omitted, spans are
    printed to stdout when ``MCP_TRACE=console`` and nothing is installed
    otherwise. Returns the provider (or ``None`` when tracing stays off).
    """
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )

    if exporter is None:
        if os.environ.get("MCP_TRACE", "").lower() != "console":
            return None
        exporter = ConsoleSpanExporter()
    provider = TracerProvider(resource=Resource.create({"service.name": TRACER_NAME}))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return provider
