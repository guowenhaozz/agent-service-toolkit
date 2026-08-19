"""Policy, timeout, retry, idempotency, and trace wrapper for BaseTool."""

import asyncio
import hashlib
import inspect
import json
import logging
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextvars import copy_context
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, ToolException
from langchain_core.tools.base import _format_output
from langgraph.errors import GraphBubbleUp
from pydantic import ConfigDict, Field, ValidationError

from core import settings
from harness.trace import TraceRecorder, sanitize_error_message
from tool_gateway.policy import (
    TOOL_POLICIES,
    ToolOperationType,
    ToolPolicy,
)
from tool_gateway.runtime import (
    get_current_execution_context,
    get_current_tool_runtime,
)

logger = logging.getLogger(__name__)


class ToolGatewayError(ToolException):
    """Base exception for gateway-level policy or execution failures."""


class ToolPolicyError(ToolGatewayError):
    """Raised when a tool is not registered or the agent is not allowed."""


class ToolOutcomeUnknownError(ToolGatewayError):
    """A WRITE operation timed out before its final outcome was observable."""


class GatewayTool(BaseTool):
    """A BaseTool-compatible facade that delegates through ToolGateway."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    wrapped_tool: BaseTool = Field(exclude=True)
    policy: ToolPolicy = Field(exclude=True)
    gateway: Any = Field(exclude=True)

    def __init__(
        self,
        *,
        wrapped_tool: BaseTool,
        policy: ToolPolicy,
        gateway: "ToolGateway",
    ) -> None:
        values = {
            "name": wrapped_tool.name,
            "description": wrapped_tool.description,
            "args_schema": wrapped_tool.args_schema,
            "return_direct": wrapped_tool.return_direct,
            "verbose": wrapped_tool.verbose,
            "callbacks": wrapped_tool.callbacks,
            "tags": wrapped_tool.tags,
            "metadata": wrapped_tool.metadata,
            "handle_tool_error": wrapped_tool.handle_tool_error,
            "handle_validation_error": wrapped_tool.handle_validation_error,
            "response_format": wrapped_tool.response_format,
            "extras": getattr(wrapped_tool, "extras", None),
            "wrapped_tool": wrapped_tool,
            "policy": policy,
            "gateway": gateway,
        }
        super().__init__(**values)

    async def ainvoke(
        self,
        input: str | dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        """Validate input and execute while preserving ToolNode output types."""

        tool_input, tool_call_id = _split_tool_input(input, kwargs)
        try:
            args, parsed_kwargs = self._to_args_and_kwargs(tool_input, tool_call_id)
        except (TypeError, ValueError, ValidationError) as error:
            self.gateway.record_rejected(
                self.policy,
                error=error,
                error_message="工具参数校验失败。",
            )
            raise

        result = await self.gateway.execute_async(
            self,
            args=args,
            kwargs=parsed_kwargs,
            config=config,
        )
        return _format_gateway_output(self, result, tool_call_id)

    def invoke(
        self,
        input: str | dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        """Synchronous counterpart used by sync ToolNode callers."""

        tool_input, tool_call_id = _split_tool_input(input, kwargs)
        try:
            args, parsed_kwargs = self._to_args_and_kwargs(tool_input, tool_call_id)
        except (TypeError, ValueError, ValidationError) as error:
            self.gateway.record_rejected(
                self.policy,
                error=error,
                error_message="工具参数校验失败。",
            )
            raise

        result = self.gateway.execute_sync(
            self,
            args=args,
            kwargs=parsed_kwargs,
            config=config,
        )
        return _format_gateway_output(self, result, tool_call_id)

    async def _arun(
        self,
        *args: Any,
        config: RunnableConfig | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Support direct ``arun`` callers after BaseTool validation."""

        return await self.gateway.execute_async(
            self,
            args=args,
            kwargs=kwargs,
            config=config,
            run_manager=run_manager,
        )

    def _run(
        self,
        *args: Any,
        config: RunnableConfig | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Support direct ``run`` callers after BaseTool validation."""

        return self.gateway.execute_sync(
            self,
            args=args,
            kwargs=kwargs,
            config=config,
            run_manager=run_manager,
        )


class ToolGateway:
    """Wrap registered tools with policy controls and tool-level traces."""

    _RETRYABLE_EXCEPTION_TYPES = (TimeoutError, ConnectionError)

    def __init__(
        self,
        *,
        policies: dict[str, ToolPolicy] | None = None,
        recorder: TraceRecorder | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.policies = dict(policies or TOOL_POLICIES)
        self.recorder = recorder or TraceRecorder(
            settings.TRACE_LOG_PATH,
            enabled=settings.TRACE_ENABLED,
        )
        self.enabled = settings.TOOL_GATEWAY_ENABLED if enabled is None else enabled

    def wrap(self, tool: BaseTool) -> GatewayTool:
        """Wrap one tool; missing policy is a startup/configuration error."""

        policy = self.policies.get(tool.name)
        if policy is None:
            raise ToolPolicyError(f"Tool policy is not registered: {tool.name}")
        if policy.tool_name != tool.name:
            raise ToolPolicyError(f"Tool policy name mismatch: {tool.name}")
        return GatewayTool(wrapped_tool=tool, policy=policy, gateway=self)

    def wrap_many(self, tools: list[BaseTool]) -> list[GatewayTool]:
        """Wrap all tools explicitly registered for an Agent."""

        return [self.wrap(tool) for tool in tools]

    def _context(self):
        context = get_current_execution_context()
        if self.enabled and context is None:
            raise ToolPolicyError("Tool Gateway requires an active ExecutionContext")
        return context

    def _check_agent_allowed(self, policy: ToolPolicy, context: Any) -> None:
        if not self.enabled:
            return
        if context is None or context.agent_name not in policy.allowed_agents:
            agent_name = context.agent_name if context is not None else "<none>"
            error = ToolPolicyError(
                f"Agent {agent_name} is not allowed to use {policy.tool_name}"
            )
            if context is not None:
                self._record(
                    context=context,
                    policy=policy,
                    event_type="TOOL_REJECTED",
                    status="rejected",
                    attempt=0,
                    max_attempts=policy.max_attempts,
                    error=error,
                    error_message="当前 Agent 不在工具白名单中。",
                    duration_ms=0.0,
                )
            raise error

    def _idempotency_hash(
        self,
        policy: ToolPolicy,
        kwargs: dict[str, Any],
    ) -> str | None:
        if policy.operation_type != ToolOperationType.WRITE:
            return None

        values = {
            field: _normalize_for_hash(kwargs.get(field))
            for field in policy.idempotency_fields
        }
        payload = json.dumps(
            {"tool_name": policy.tool_name, "fields": values},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def record_rejected(
        self,
        policy: ToolPolicy,
        *,
        error: BaseException,
        error_message: str | None = None,
    ) -> None:
        context = get_current_execution_context()
        if context is None:
            return
        self._record(
            context=context,
            policy=policy,
            event_type="TOOL_REJECTED",
            status="rejected",
            attempt=0,
            max_attempts=policy.max_attempts,
            error=error,
            error_message=error_message,
            duration_ms=0.0,
        )

    async def execute_async(
        self,
        tool: GatewayTool,
        *,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        config: RunnableConfig | None,
        run_manager: Any = None,
    ) -> Any:
        context = self._context()
        self._check_agent_allowed(tool.policy, context)
        idempotency_hash = self._idempotency_hash(tool.policy, kwargs)
        runtime = get_current_tool_runtime()

        if (
            self.enabled
            and tool.policy.operation_type == ToolOperationType.WRITE
            and (context is None or runtime is None)
        ):
            raise ToolPolicyError("WRITE tools require request-scoped runtime state")

        if idempotency_hash and runtime is not None:
            cache_key = (tool.name, idempotency_hash)
            if cache_key in runtime.idempotency_cache:
                result = runtime.idempotency_cache[cache_key]
                if context is not None:
                    self._record(
                        context=context,
                        policy=tool.policy,
                        event_type="TOOL_DEDUPLICATED",
                        status="deduplicated",
                        attempt=1,
                        max_attempts=1,
                        result=result,
                        idempotency_key_hash=idempotency_hash,
                        duration_ms=0.0,
                    )
                return result

        max_attempts = tool.policy.max_attempts if self.enabled else 1
        for attempt in range(1, max_attempts + 1):
            started = time.perf_counter()
            if context is not None:
                self._record(
                    context=context,
                    policy=tool.policy,
                    event_type="TOOL_STARTED",
                    status="started",
                    attempt=attempt,
                    max_attempts=max_attempts,
                    idempotency_key_hash=idempotency_hash,
                )
            try:
                if self.enabled:
                    async with asyncio.timeout(tool.policy.timeout_seconds):
                        result = await self._call_wrapped_async(
                            tool.wrapped_tool,
                            args=args,
                            kwargs=kwargs,
                            config=config,
                            run_manager=run_manager,
                        )
                else:
                    result = await self._call_wrapped_async(
                        tool.wrapped_tool,
                        args=args,
                        kwargs=kwargs,
                        config=config,
                        run_manager=run_manager,
                    )
            except GraphBubbleUp as error:
                if context is not None:
                    self._record(
                        context=context,
                        policy=tool.policy,
                        event_type="TOOL_INTERRUPTED",
                        status="interrupted",
                        attempt=attempt,
                        max_attempts=max_attempts,
                        idempotency_key_hash=idempotency_hash,
                        error=error,
                        duration_ms=_elapsed_ms(started),
                    )
                raise
            except asyncio.CancelledError:
                raise
            except Exception as error:
                if self.enabled and self._should_retry(tool.policy, error, attempt):
                    if context is not None:
                        self._record(
                            context=context,
                            policy=tool.policy,
                            event_type="TOOL_RETRY",
                            status="retrying",
                            attempt=attempt,
                            max_attempts=max_attempts,
                            idempotency_key_hash=idempotency_hash,
                            error=error,
                            duration_ms=_elapsed_ms(started),
                        )
                    await asyncio.sleep(settings.TOOL_RETRY_BACKOFF_SECONDS)
                    continue

                unknown_outcome = (
                    tool.policy.operation_type == ToolOperationType.WRITE
                    and isinstance(error, TimeoutError)
                )
                if context is not None:
                    self._record(
                        context=context,
                        policy=tool.policy,
                        event_type="TOOL_FAILED",
                        status="failed",
                        attempt=attempt,
                        max_attempts=max_attempts,
                        idempotency_key_hash=idempotency_hash,
                        error=error,
                        result_type="UNKNOWN_OUTCOME" if unknown_outcome else None,
                        duration_ms=_elapsed_ms(started),
                    )
                if unknown_outcome:
                    raise ToolOutcomeUnknownError(
                        f"{tool.name} timed out; outcome is unknown. "
                        "Query the business state before retrying."
                    ) from error
                raise
            else:
                if context is not None:
                    self._record(
                        context=context,
                        policy=tool.policy,
                        event_type="TOOL_COMPLETED",
                        status="completed",
                        attempt=attempt,
                        max_attempts=max_attempts,
                        idempotency_key_hash=idempotency_hash,
                        result=result,
                        duration_ms=_elapsed_ms(started),
                    )
                if idempotency_hash and runtime is not None:
                    runtime.idempotency_cache[(tool.name, idempotency_hash)] = result
                return result

        raise AssertionError("Tool Gateway retry loop exited unexpectedly")

    def execute_sync(
        self,
        tool: GatewayTool,
        *,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        config: RunnableConfig | None,
        run_manager: Any = None,
    ) -> Any:
        context = self._context()
        self._check_agent_allowed(tool.policy, context)
        idempotency_hash = self._idempotency_hash(tool.policy, kwargs)
        runtime = get_current_tool_runtime()

        if (
            self.enabled
            and tool.policy.operation_type == ToolOperationType.WRITE
            and (context is None or runtime is None)
        ):
            raise ToolPolicyError("WRITE tools require request-scoped runtime state")

        if idempotency_hash and runtime is not None:
            cache_key = (tool.name, idempotency_hash)
            if cache_key in runtime.idempotency_cache:
                result = runtime.idempotency_cache[cache_key]
                if context is not None:
                    self._record(
                        context=context,
                        policy=tool.policy,
                        event_type="TOOL_DEDUPLICATED",
                        status="deduplicated",
                        attempt=1,
                        max_attempts=1,
                        result=result,
                        idempotency_key_hash=idempotency_hash,
                        duration_ms=0.0,
                    )
                return result

        max_attempts = tool.policy.max_attempts if self.enabled else 1
        for attempt in range(1, max_attempts + 1):
            started = time.perf_counter()
            if context is not None:
                self._record(
                    context=context,
                    policy=tool.policy,
                    event_type="TOOL_STARTED",
                    status="started",
                    attempt=attempt,
                    max_attempts=max_attempts,
                    idempotency_key_hash=idempotency_hash,
                )
            try:
                result = self._call_wrapped_sync_with_timeout(
                    tool,
                    args=args,
                    kwargs=kwargs,
                    config=config,
                    run_manager=run_manager,
                    timeout_seconds=tool.policy.timeout_seconds
                    if self.enabled
                    else None,
                )
            except GraphBubbleUp as error:
                if context is not None:
                    self._record(
                        context=context,
                        policy=tool.policy,
                        event_type="TOOL_INTERRUPTED",
                        status="interrupted",
                        attempt=attempt,
                        max_attempts=max_attempts,
                        idempotency_key_hash=idempotency_hash,
                        error=error,
                        duration_ms=_elapsed_ms(started),
                    )
                raise
            except Exception as error:
                if self.enabled and self._should_retry(tool.policy, error, attempt):
                    if context is not None:
                        self._record(
                            context=context,
                            policy=tool.policy,
                            event_type="TOOL_RETRY",
                            status="retrying",
                            attempt=attempt,
                            max_attempts=max_attempts,
                            idempotency_key_hash=idempotency_hash,
                            error=error,
                            duration_ms=_elapsed_ms(started),
                        )
                    time.sleep(settings.TOOL_RETRY_BACKOFF_SECONDS)
                    continue

                unknown_outcome = (
                    tool.policy.operation_type == ToolOperationType.WRITE
                    and isinstance(error, TimeoutError)
                )
                if context is not None:
                    self._record(
                        context=context,
                        policy=tool.policy,
                        event_type="TOOL_FAILED",
                        status="failed",
                        attempt=attempt,
                        max_attempts=max_attempts,
                        idempotency_key_hash=idempotency_hash,
                        error=error,
                        result_type="UNKNOWN_OUTCOME" if unknown_outcome else None,
                        duration_ms=_elapsed_ms(started),
                    )
                if unknown_outcome:
                    raise ToolOutcomeUnknownError(
                        f"{tool.name} timed out; outcome is unknown. "
                        "Query the business state before retrying."
                    ) from error
                raise
            else:
                if context is not None:
                    self._record(
                        context=context,
                        policy=tool.policy,
                        event_type="TOOL_COMPLETED",
                        status="completed",
                        attempt=attempt,
                        max_attempts=max_attempts,
                        idempotency_key_hash=idempotency_hash,
                        result=result,
                        duration_ms=_elapsed_ms(started),
                    )
                if idempotency_hash and runtime is not None:
                    runtime.idempotency_cache[(tool.name, idempotency_hash)] = result
                return result

        raise AssertionError("Tool Gateway retry loop exited unexpectedly")

    async def _call_wrapped_async(
        self,
        tool: BaseTool,
        *,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        config: RunnableConfig | None,
        run_manager: Any,
    ) -> Any:
        call_kwargs = dict(kwargs)
        parameters = inspect.signature(tool._arun).parameters
        if "config" in parameters:
            call_kwargs["config"] = config
        if "run_manager" in parameters:
            call_kwargs["run_manager"] = run_manager
        result = tool._arun(*args, **call_kwargs)
        return await result if inspect.isawaitable(result) else result

    def _call_wrapped_sync_with_timeout(
        self,
        gateway_tool: GatewayTool,
        *,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        config: RunnableConfig | None,
        run_manager: Any,
        timeout_seconds: float | None,
    ) -> Any:
        def call() -> Any:
            call_kwargs = dict(kwargs)
            parameters = inspect.signature(gateway_tool.wrapped_tool._run).parameters
            if "config" in parameters:
                call_kwargs["config"] = config
            if "run_manager" in parameters:
                call_kwargs["run_manager"] = run_manager
            return gateway_tool.wrapped_tool._run(*args, **call_kwargs)

        if timeout_seconds is None:
            return call()

        executor = ThreadPoolExecutor(max_workers=1)
        context = copy_context()
        future = executor.submit(context.run, call)
        try:
            return future.result(timeout=timeout_seconds)
        except FutureTimeoutError as error:
            future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise TimeoutError("synchronous tool execution timed out") from error
        finally:
            if not future.done():
                executor.shutdown(wait=False, cancel_futures=True)
            else:
                executor.shutdown(wait=True)

    def _should_retry(
        self,
        policy: ToolPolicy,
        error: BaseException,
        attempt: int,
    ) -> bool:
        if (
            not policy.retryable
            or policy.operation_type != ToolOperationType.READ
        ):
            return False
        if attempt >= policy.max_attempts:
            return False
        if isinstance(error, self._RETRYABLE_EXCEPTION_TYPES):
            return True
        return isinstance(error, sqlite3.OperationalError) and "locked" in str(error).lower()

    def _record(
        self,
        *,
        context: Any,
        policy: ToolPolicy,
        event_type: str,
        status: str,
        attempt: int,
        max_attempts: int,
        idempotency_key_hash: str | None = None,
        result: Any = None,
        result_type: str | None = None,
        error: BaseException | None = None,
        error_message: str | None = None,
        duration_ms: float | None = None,
    ) -> None:
        try:
            self.recorder.record_event(
                context=context,
                event_type=event_type,
                status=status,
                duration_ms=duration_ms,
                result_type=result_type or (type(result).__name__ if result is not None else None),
                error_type=type(error).__name__ if error else None,
                error_message=error_message or _safe_error_message(error),
                tool_name=policy.tool_name,
                operation_type=policy.operation_type.value,
                attempt=attempt,
                max_attempts=max_attempts,
                idempotency_key_hash=idempotency_key_hash,
            )
        except Exception as trace_error:  # pragma: no cover - defensive fallback
            logger.warning(
                "Unable to record tool lifecycle event: %s",
                type(trace_error).__name__,
            )


def _split_tool_input(
    input: str | dict[str, Any],
    kwargs: dict[str, Any],
) -> tuple[str | dict[str, Any], str | None]:
    if isinstance(input, dict) and input.get("type") == "tool_call":
        return dict(input.get("args", {})), input.get("id")
    return input, kwargs.get("tool_call_id")


def _format_gateway_output(
    tool: GatewayTool,
    result: Any,
    tool_call_id: str | None,
) -> Any:
    """Mirror BaseTool output formatting for ToolNode and direct callers."""

    artifact = None
    content = result
    if tool.response_format == "content_and_artifact":
        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError(
                "Since response_format='content_and_artifact' a two-tuple "
                f"of the message content and raw tool output is expected. "
                f"Instead, generated response is of type: {type(result)}."
            )
        content, artifact = result

    return _format_output(
        content,
        artifact,
        tool_call_id,
        tool.name,
        "success",
    )


def _normalize_for_hash(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return {str(key): _normalize_for_hash(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_hash(item) for item in value]
    return value


def _elapsed_ms(started: float) -> float:
    return max(0.0, (time.perf_counter() - started) * 1000)


def _safe_error_message(error: BaseException | None) -> str | None:
    if error is None:
        return None
    if isinstance(error, ValidationError):
        return "工具参数校验失败。"
    return sanitize_error_message(error)
