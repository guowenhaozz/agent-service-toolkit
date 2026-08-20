"""Generic lifecycle wrapper around any LangGraph-compatible agent."""

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from core import settings
from harness.context import ExecutionContext
from harness.trace import TraceRecorder, sanitize_error_message
from tool_gateway.runtime import execution_context_scope

logger = logging.getLogger(__name__)


class AgentHarness:
    """Record execution lifecycle while preserving the wrapped agent contract."""

    def __init__(self, recorder: TraceRecorder | None = None) -> None:
        self.recorder = recorder or TraceRecorder(
            settings.TRACE_LOG_PATH,
            enabled=settings.TRACE_ENABLED,
        )

    @staticmethod
    def create_context(
        *,
        user_id: str,
        thread_id: str,
        agent_name: str,
        endpoint: str,
    ) -> ExecutionContext:
        """Create a context from IDs already resolved by the service layer."""

        return ExecutionContext.create(
            user_id=user_id,
            thread_id=thread_id,
            agent_name=agent_name,
            endpoint=endpoint,
        )

    def _record(
        self,
        *,
        context: ExecutionContext,
        event_type: str,
        status: str,
        started: float,
        result_type: str | None = None,
        error: BaseException | None = None,
    ) -> None:
        duration_ms = None if event_type == "RUN_STARTED" else max(
            0.0,
            (time.perf_counter() - started) * 1000,
        )
        try:
            self.recorder.record_event(
                context=context,
                event_type=event_type,
                status=status,
                duration_ms=duration_ms,
                result_type=result_type,
                error_type=type(error).__name__ if error else None,
                error_message=sanitize_error_message(error) if error else None,
            )
        except Exception as trace_error:  # pragma: no cover - defensive fallback
            logger.warning(
                "Unable to record agent lifecycle event: %s",
                type(trace_error).__name__,
            )

    async def invoke(
        self,
        agent: Any,
        context: ExecutionContext,
        **kwargs: Any,
    ) -> Any:
        """Invoke an agent and re-raise every original exception."""

        with execution_context_scope(context):
            return await self._invoke_with_context(agent, context, **kwargs)

    async def _invoke_with_context(
        self,
        agent: Any,
        context: ExecutionContext,
        **kwargs: Any,
    ) -> Any:
        """Run the invoke lifecycle while the ContextVars are bound."""

        started = time.perf_counter()
        self._record(
            context=context,
            event_type="RUN_STARTED",
            status="started",
            started=started,
        )

        try:
            result = await agent.ainvoke(**kwargs)
        except asyncio.CancelledError as error:
            self._record(
                context=context,
                event_type="RUN_CANCELLED",
                status="cancelled",
                started=started,
                error=error,
            )
            raise
        except Exception as error:
            self._record(
                context=context,
                event_type="RUN_FAILED",
                status="failed",
                started=started,
                error=error,
            )
            raise
        else:
            self._record(
                context=context,
                event_type="RUN_COMPLETED",
                status="completed",
                started=started,
                result_type=type(result).__name__,
            )
            return result

    async def stream(
        self,
        agent: Any,
        context: ExecutionContext,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Forward every stream chunk unchanged and record terminal state."""

        with execution_context_scope(context):
            async for chunk in self._stream_with_context(agent, context, **kwargs):
                yield chunk

    async def _stream_with_context(
        self,
        agent: Any,
        context: ExecutionContext,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Run the stream lifecycle while the ContextVars are bound."""

        started = time.perf_counter()
        self._record(
            context=context,
            event_type="RUN_STARTED",
            status="started",
            started=started,
        )

        try:
            async for chunk in agent.astream(**kwargs):
                yield chunk
        except asyncio.CancelledError as error:
            self._record(
                context=context,
                event_type="RUN_CANCELLED",
                status="cancelled",
                started=started,
                error=error,
            )
            raise
        except Exception as error:
            self._record(
                context=context,
                event_type="RUN_FAILED",
                status="failed",
                started=started,
                error=error,
            )
            raise
        else:
            self._record(
                context=context,
                event_type="RUN_COMPLETED",
                status="completed",
                started=started,
                result_type="stream",
            )
