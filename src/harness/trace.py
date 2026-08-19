"""Structured, privacy-conscious trace recording for agent executions."""

import hashlib
import logging
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from harness.context import ExecutionContext

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_HANDLERS: dict[Path, logging.Handler] = {}
_HANDLERS_LOCK = threading.Lock()
_FALLBACK_LOGGER = logging.getLogger(__name__)
_SENSITIVE_VALUE_PATTERN = re.compile(
    r"(?i)(authorization|api[_ -]?key|auth[_ -]?secret|password|token)"
    r"\s*[:=]\s*([^\s,;]+)"
)
_MAX_ERROR_MESSAGE_LENGTH = 500


class TraceEvent(BaseModel):
    """One JSON-serializable execution event."""

    model_config = ConfigDict(frozen=True)

    event_id: str
    request_id: str
    user_id: str
    thread_id: str
    agent_name: str
    endpoint: str
    event_type: str
    status: str
    timestamp: datetime
    duration_ms: float | None = None
    result_type: str | None = None
    error_type: str | None = None
    error_message: str | None = None

    @classmethod
    def create(
        cls,
        *,
        context: "ExecutionContext",
        event_type: str,
        status: str,
        duration_ms: float | None = None,
        result_type: str | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> "TraceEvent":
        """Build an event using the context's stable request identifiers."""

        return cls(
            event_id=str(uuid4()),
            request_id=context.request_id,
            user_id=context.user_id,
            thread_id=context.thread_id,
            agent_name=context.agent_name,
            endpoint=context.endpoint,
            event_type=event_type,
            status=status,
            timestamp=datetime.now(UTC),
            duration_ms=duration_ms,
            result_type=result_type,
            error_type=error_type,
            error_message=error_message,
        )


def sanitize_error_message(error: BaseException) -> str:
    """Keep useful error context without copying secrets or large payloads."""

    message = str(error).replace("\r", " ").replace("\n", " ")
    message = _SENSITIVE_VALUE_PATTERN.sub(r"\1=[REDACTED]", message)
    message = message[:_MAX_ERROR_MESSAGE_LENGTH]
    if len(str(error)) > _MAX_ERROR_MESSAGE_LENGTH:
        message += "..."
    return f"{type(error).__name__}: {message}"[:_MAX_ERROR_MESSAGE_LENGTH]


def _resolve_trace_path(path: str | Path) -> Path:
    trace_path = Path(path)
    if not trace_path.is_absolute():
        trace_path = _PROJECT_ROOT / trace_path
    return trace_path.resolve()


def _handler_for(path: Path) -> logging.Handler:
    """Create one UTF-8 handler per path, even across recorder instances."""

    with _HANDLERS_LOCK:
        if path in _HANDLERS:
            return _HANDLERS[path]

        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        _HANDLERS[path] = handler
        return handler


class TraceRecorder:
    """Write one-line JSON events without affecting the agent run."""

    def __init__(
        self,
        path: str | Path = "data/traces/agent_trace.jsonl",
        *,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.path = _resolve_trace_path(path)
        self._logger: logging.Logger | None = None

        if not enabled:
            return

        try:
            logger_name = (
                "harness.trace."
                + hashlib.sha1(str(self.path).encode("utf-8")).hexdigest()
            )
            trace_logger = logging.getLogger(logger_name)
            trace_logger.setLevel(logging.INFO)
            trace_logger.propagate = False
            handler = _handler_for(self.path)
            if handler not in trace_logger.handlers:
                trace_logger.addHandler(handler)
            self._logger = trace_logger
        except Exception as error:  # pragma: no cover - platform/filesystem dependent
            _FALLBACK_LOGGER.warning(
                "Unable to initialize agent trace recorder: %s",
                type(error).__name__,
            )

    def record(self, event: TraceEvent) -> None:
        """Persist an event, degrading to a normal log on trace failure."""

        if not self.enabled or self._logger is None:
            return

        try:
            self._logger.info(event.model_dump_json(exclude_none=False))
        except Exception as error:  # pragma: no cover - platform/filesystem dependent
            _FALLBACK_LOGGER.warning(
                "Unable to write agent trace event: %s",
                type(error).__name__,
            )

    def record_event(
        self,
        *,
        context: "ExecutionContext",
        event_type: str,
        status: str,
        duration_ms: float | None = None,
        result_type: str | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> TraceEvent:
        """Create and persist an event, returning it for tests/callers."""

        event = TraceEvent.create(
            context=context,
            event_type=event_type,
            status=status,
            duration_ms=duration_ms,
            result_type=result_type,
            error_type=error_type,
            error_message=error_message,
        )
        self.record(event)
        return event
