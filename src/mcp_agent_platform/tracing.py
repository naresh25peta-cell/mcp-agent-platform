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


# ---------------------------------------------------------------------------
# LLM-call tracing (LangChain callback)
# ---------------------------------------------------------------------------


def _make_callback_class():
    from langchain_core.callbacks import BaseCallbackHandler

    class LLMTraceCallback(BaseCallbackHandler):
        """One ``chat <model>`` span per LLM request, recording the model
        name, token usage and finish reason (GenAI conventions:
        ``gen_ai.request.model``, ``gen_ai.usage.input_tokens``,
        ``gen_ai.usage.output_tokens``).

        Privacy: prompts and completions are never recorded.
        """

        def __init__(self) -> None:
            self._spans: dict[Any, Any] = {}

        def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
            invocation = kwargs.get("invocation_params") or {}
            model = (
                invocation.get("model")
                or invocation.get("model_name")
                or (serialized or {}).get("name")
                or "unknown"
            )
            s = get_tracer().start_span(
                f"chat {model}",
                attributes={
                    "gen_ai.operation.name": "chat",
                    "gen_ai.request.model": str(model),
                    "gen_ai.request.message_count": sum(len(m) for m in messages),
                },
            )
            self._spans[run_id] = s

        def on_llm_end(self, response, *, run_id, **kwargs):
            s = self._spans.pop(run_id, None)
            if s is None:
                return
            try:
                msg = response.generations[0][0].message
                usage = getattr(msg, "usage_metadata", None) or {}
                if "input_tokens" in usage:
                    s.set_attribute("gen_ai.usage.input_tokens", int(usage["input_tokens"]))
                if "output_tokens" in usage:
                    s.set_attribute("gen_ai.usage.output_tokens", int(usage["output_tokens"]))
                finish = (getattr(msg, "response_metadata", None) or {}).get("finish_reason")
                if finish:
                    s.set_attribute("gen_ai.response.finish_reasons", [str(finish)])
            except (IndexError, AttributeError):
                pass
            s.end()

        def on_llm_error(self, error, *, run_id, **kwargs):
            s = self._spans.pop(run_id, None)
            if s is not None:
                s.set_attribute("error.type", type(error).__name__)
                s.end()

    return LLMTraceCallback


def llm_trace_callbacks() -> list:
    """Callbacks to pass as ``config={"callbacks": llm_trace_callbacks()}``."""
    return [_make_callback_class()()]
