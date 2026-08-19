"""Unit tests for the policy-driven Tool Gateway."""

import asyncio
import json
import sqlite3
from typing import Annotated

import pytest
from langchain_core.tools import InjectedToolCallId, StructuredTool
from langgraph.errors import GraphInterrupt
from langgraph.types import Command
from pydantic import BaseModel

from core import settings
from harness.context import ExecutionContext
from harness.trace import TraceRecorder
from tool_gateway import (
    TOOL_POLICIES,
    ToolGateway,
    ToolGatewayError,
    ToolOperationType,
    ToolOutcomeUnknownError,
    ToolPolicy,
    ToolPolicyError,
)
from tool_gateway.runtime import (
    execution_context_scope,
    get_current_execution_context,
)


class ValueInput(BaseModel):
    value: str


class ToolCallInput(BaseModel):
    value: str
    tool_call_id: Annotated[str, InjectedToolCallId]


def make_context(
    *,
    agent_name: str = "device-assistant",
    request_id: str | None = None,
) -> ExecutionContext:
    context = ExecutionContext.create(
        user_id="user-1",
        thread_id="thread-1",
        agent_name=agent_name,
        endpoint="/test",
    )
    if request_id is not None:
        return context.model_copy(update={"request_id": request_id})
    return context


def make_policy(
    name: str,
    operation_type: ToolOperationType = ToolOperationType.READ,
    *,
    timeout_seconds: float = 1.0,
    max_attempts: int = 1,
    retryable: bool = False,
    idempotency_fields: tuple[str, ...] = (),
) -> ToolPolicy:
    return ToolPolicy(
        tool_name=name,
        operation_type=operation_type,
        allowed_agents={"device-assistant"},
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
        retryable=retryable,
        idempotency_fields=idempotency_fields,
    )


def make_async_tool(name: str, coroutine, args_schema=ValueInput) -> StructuredTool:
    return StructuredTool.from_function(
        coroutine=coroutine,
        name=name,
        description=f"Test tool {name}",
        args_schema=args_schema,
    )


def make_sync_tool(name: str, function, args_schema=ValueInput) -> StructuredTool:
    return StructuredTool.from_function(
        func=function,
        name=name,
        description=f"Test tool {name}",
        args_schema=args_schema,
    )


def read_events(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.asyncio
async def test_registered_tool_preserves_identity_schema_and_result(tmp_path):
    calls = []

    async def read(value: str) -> str:
        calls.append(value)
        return f"ok:{value}"

    source = make_async_tool("FakeRead", read)
    gateway = ToolGateway(
        policies={"FakeRead": make_policy("FakeRead")},
        recorder=TraceRecorder(tmp_path / "trace.jsonl"),
    )
    wrapped = gateway.wrap(source)

    assert wrapped.name == source.name
    assert wrapped.description == source.description
    assert wrapped.args_schema is source.args_schema
    with execution_context_scope(make_context()):
        result = await wrapped.ainvoke({"value": "one"})

    assert result == "ok:one"
    assert calls == ["one"]


def test_missing_policy_is_denied_at_wrap_time():
    async def read(value: str) -> str:
        return value

    source = make_async_tool("UnregisteredTool", read)
    gateway = ToolGateway(policies={})

    with pytest.raises(ToolPolicyError, match="not registered"):
        gateway.wrap(source)


@pytest.mark.asyncio
async def test_enabled_gateway_requires_active_execution_context():
    async def read(value: str) -> str:
        return value

    gateway = ToolGateway(
        policies={"ContextRequired": make_policy("ContextRequired")},
    )
    wrapped = gateway.wrap(make_async_tool("ContextRequired", read))

    with pytest.raises(ToolPolicyError, match="active ExecutionContext"):
        await wrapped.ainvoke({"value": "blocked"})


@pytest.mark.asyncio
async def test_disabled_gateway_preserves_execution_without_context(tmp_path):
    calls = 0

    async def read(value: str) -> str:
        nonlocal calls
        calls += 1
        return f"ok:{value}"

    path = tmp_path / "trace-disabled.jsonl"
    gateway = ToolGateway(
        policies={"DisabledRead": make_policy("DisabledRead")},
        recorder=TraceRecorder(path),
        enabled=False,
    )
    wrapped = gateway.wrap(make_async_tool("DisabledRead", read))

    assert await wrapped.ainvoke({"value": "plain"}) == "ok:plain"
    assert calls == 1
    if path.exists():
        assert path.read_text(encoding="utf-8") == ""


@pytest.mark.asyncio
async def test_agent_outside_allowlist_is_rejected_without_call(tmp_path):
    calls = 0

    async def read(value: str) -> str:
        nonlocal calls
        calls += 1
        return value

    path = tmp_path / "trace.jsonl"
    gateway = ToolGateway(
        policies={"FakeRead": make_policy("FakeRead")},
        recorder=TraceRecorder(path),
    )
    wrapped = gateway.wrap(make_async_tool("FakeRead", read))

    with execution_context_scope(make_context(agent_name="other-agent")):
        with pytest.raises(ToolGatewayError, match="not allowed"):
            await wrapped.ainvoke({"value": "blocked"})

    assert calls == 0
    assert [event["event_type"] for event in read_events(path)] == [
        "TOOL_REJECTED"
    ]


@pytest.mark.asyncio
async def test_invalid_parameters_are_rejected_before_underlying_call(tmp_path):
    calls = 0

    async def read(value: str) -> str:
        nonlocal calls
        calls += 1
        return value

    path = tmp_path / "trace.jsonl"
    gateway = ToolGateway(
        policies={"FakeRead": make_policy("FakeRead")},
        recorder=TraceRecorder(path),
    )
    wrapped = gateway.wrap(make_async_tool("FakeRead", read))

    with execution_context_scope(make_context()):
        with pytest.raises(Exception):
            await wrapped.ainvoke({})

    assert calls == 0
    events = read_events(path)
    assert events[0]["event_type"] == "TOOL_REJECTED"
    assert "value" not in path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_read_retry_only_retries_transient_error(monkeypatch, tmp_path):
    attempts = 0

    async def read(value: str) -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("temporary connection problem")
        return value

    monkeypatch.setattr(settings, "TOOL_RETRY_BACKOFF_SECONDS", 0.0)
    path = tmp_path / "trace.jsonl"
    policy = make_policy("RetryRead", max_attempts=2, retryable=True)
    gateway = ToolGateway(
        policies={"RetryRead": policy},
        recorder=TraceRecorder(path),
    )
    wrapped = gateway.wrap(make_async_tool("RetryRead", read))

    with execution_context_scope(make_context()):
        assert await wrapped.ainvoke({"value": "ok"}) == "ok"

    assert attempts == 2
    assert [event["event_type"] for event in read_events(path)] == [
        "TOOL_STARTED",
        "TOOL_RETRY",
        "TOOL_STARTED",
        "TOOL_COMPLETED",
    ]


@pytest.mark.asyncio
async def test_read_retries_sqlite_locked_error(monkeypatch, tmp_path):
    attempts = 0

    async def read(value: str) -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise sqlite3.OperationalError("database is locked")
        return value

    monkeypatch.setattr(settings, "TOOL_RETRY_BACKOFF_SECONDS", 0.0)
    gateway = ToolGateway(
        policies={"LockedRead": make_policy("LockedRead", max_attempts=2, retryable=True)},
        recorder=TraceRecorder(tmp_path / "trace.jsonl"),
    )
    wrapped = gateway.wrap(make_async_tool("LockedRead", read))

    with execution_context_scope(make_context()):
        assert await wrapped.ainvoke({"value": "ok"}) == "ok"

    assert attempts == 2


@pytest.mark.asyncio
async def test_pure_tool_never_retries_even_if_misconfigured(tmp_path):
    attempts = 0

    async def pure(value: str) -> str:
        nonlocal attempts
        attempts += 1
        raise TimeoutError("pure operation failed")

    policy = make_policy(
        "PureNoRetry",
        ToolOperationType.PURE,
        max_attempts=2,
        retryable=True,
    )
    gateway = ToolGateway(
        policies={"PureNoRetry": policy},
        recorder=TraceRecorder(tmp_path / "trace.jsonl"),
    )
    wrapped = gateway.wrap(make_async_tool("PureNoRetry", pure))

    with execution_context_scope(make_context()):
        with pytest.raises(TimeoutError):
            await wrapped.ainvoke({"value": "pure"})

    assert attempts == 1


@pytest.mark.asyncio
async def test_read_non_transient_error_is_not_retried(tmp_path):
    attempts = 0

    async def read(value: str) -> str:
        nonlocal attempts
        attempts += 1
        raise ValueError("business validation failed")

    path = tmp_path / "trace.jsonl"
    policy = make_policy("NoRetryRead", max_attempts=2, retryable=True)
    gateway = ToolGateway(
        policies={"NoRetryRead": policy},
        recorder=TraceRecorder(path),
    )
    wrapped = gateway.wrap(make_async_tool("NoRetryRead", read))

    with execution_context_scope(make_context()):
        with pytest.raises(ValueError):
            await wrapped.ainvoke({"value": "bad"})

    assert attempts == 1
    assert [event["event_type"] for event in read_events(path)] == [
        "TOOL_STARTED",
        "TOOL_FAILED",
    ]


@pytest.mark.asyncio
async def test_write_timeout_is_unknown_and_never_retried(tmp_path):
    calls = 0

    async def write(value: str) -> str:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.05)
        return value

    path = tmp_path / "trace.jsonl"
    policy = make_policy(
        "SlowWrite",
        ToolOperationType.WRITE,
        timeout_seconds=0.005,
        idempotency_fields=("value",),
    )
    gateway = ToolGateway(
        policies={"SlowWrite": policy},
        recorder=TraceRecorder(path),
    )
    wrapped = gateway.wrap(make_async_tool("SlowWrite", write))

    with execution_context_scope(make_context()):
        with pytest.raises(ToolOutcomeUnknownError):
            await wrapped.ainvoke({"value": "write"})

    assert calls == 1
    events = read_events(path)
    assert [event["event_type"] for event in events] == [
        "TOOL_STARTED",
        "TOOL_FAILED",
    ]
    assert events[-1]["result_type"] == "UNKNOWN_OUTCOME"
    assert events[-1]["attempt"] == 1
    assert events[-1]["max_attempts"] == 1


@pytest.mark.asyncio
async def test_write_business_error_is_not_retried(tmp_path):
    attempts = 0

    async def write(value: str) -> str:
        nonlocal attempts
        attempts += 1
        raise ValueError("business rule rejected")

    policy = make_policy(
        "BusinessWrite",
        ToolOperationType.WRITE,
        idempotency_fields=("value",),
    )
    gateway = ToolGateway(
        policies={"BusinessWrite": policy},
        recorder=TraceRecorder(tmp_path / "trace.jsonl"),
    )
    wrapped = gateway.wrap(make_async_tool("BusinessWrite", write))

    with execution_context_scope(make_context()):
        with pytest.raises(ValueError):
            await wrapped.ainvoke({"value": "write"})

    assert attempts == 1


@pytest.mark.asyncio
async def test_write_is_deduplicated_within_one_request_only(tmp_path):
    calls = 0

    async def write(value: str) -> str:
        nonlocal calls
        calls += 1
        return f"saved:{value}:{calls}"

    path = tmp_path / "trace.jsonl"
    policy = make_policy(
        "IdempotentWrite",
        ToolOperationType.WRITE,
        idempotency_fields=("value",),
    )
    gateway = ToolGateway(
        policies={"IdempotentWrite": policy},
        recorder=TraceRecorder(path),
    )
    wrapped = gateway.wrap(make_async_tool("IdempotentWrite", write))

    with execution_context_scope(make_context()):
        first = await wrapped.ainvoke({"value": "same"})
        second = await wrapped.ainvoke({"value": "same"})

    with execution_context_scope(make_context()):
        third = await wrapped.ainvoke({"value": "same"})

    assert first == second
    assert third != first
    assert calls == 2
    assert [event["event_type"] for event in read_events(path)] == [
        "TOOL_STARTED",
        "TOOL_COMPLETED",
        "TOOL_DEDUPLICATED",
        "TOOL_STARTED",
        "TOOL_COMPLETED",
    ]


@pytest.mark.asyncio
async def test_graph_interrupt_is_re_raised_without_retry(tmp_path):
    attempts = 0

    async def interrupting(value: str) -> str:
        nonlocal attempts
        attempts += 1
        raise GraphInterrupt("approval required")

    path = tmp_path / "trace.jsonl"
    policy = make_policy("InterruptingTool", max_attempts=2, retryable=True)
    gateway = ToolGateway(
        policies={"InterruptingTool": policy},
        recorder=TraceRecorder(path),
    )
    wrapped = gateway.wrap(make_async_tool("InterruptingTool", interrupting))

    with execution_context_scope(make_context()):
        with pytest.raises(GraphInterrupt):
            await wrapped.ainvoke({"value": "approval"})

    assert attempts == 1
    assert [event["event_type"] for event in read_events(path)] == [
        "TOOL_STARTED",
        "TOOL_INTERRUPTED",
    ]


@pytest.mark.asyncio
async def test_command_return_value_and_tool_call_id_are_preserved():
    async def command_tool(value: str) -> Command:
        return Command(update={"value": value})

    async def id_tool(value: str, tool_call_id: str) -> str:
        return tool_call_id

    gateway = ToolGateway(
        policies={
            "CommandTool": make_policy("CommandTool"),
            "IdTool": make_policy("IdTool"),
        },
        enabled=False,
    )
    command_wrapper = gateway.wrap(make_async_tool("CommandTool", command_tool))
    id_wrapper = gateway.wrap(
        make_async_tool("IdTool", id_tool, args_schema=ToolCallInput)
    )

    command = await command_wrapper.ainvoke({"value": "kept"})
    tool_call_id = await id_wrapper.ainvoke(
        {
            "type": "tool_call",
            "name": "IdTool",
            "id": "call-123",
            "args": {"value": "kept"},
        }
    )

    assert isinstance(command, Command)
    assert command.update == {"value": "kept"}
    assert tool_call_id == "call-123"


@pytest.mark.asyncio
async def test_parallel_contexts_do_not_share_request_ids(tmp_path):
    async def read(value: str) -> str:
        await asyncio.sleep(0.01)
        context = get_current_execution_context()
        assert context is not None
        return context.request_id

    gateway = ToolGateway(
        policies={"ContextRead": make_policy("ContextRead")},
        recorder=TraceRecorder(tmp_path / "trace.jsonl"),
    )
    wrapped = gateway.wrap(make_async_tool("ContextRead", read))
    first_context = make_context()
    second_context = make_context()

    async def run(context: ExecutionContext) -> str:
        with execution_context_scope(context):
            return await wrapped.ainvoke({"value": "x"})

    first, second = await asyncio.gather(run(first_context), run(second_context))

    assert first == first_context.request_id
    assert second == second_context.request_id
    assert first != second


def test_sync_tool_receives_context_through_timeout_thread(tmp_path):
    def read(value: str) -> str:
        context = get_current_execution_context()
        assert context is not None
        return f"{context.request_id}:{value}"

    gateway = ToolGateway(
        policies={"SyncRead": make_policy("SyncRead")},
        recorder=TraceRecorder(tmp_path / "trace.jsonl"),
    )
    wrapped = gateway.wrap(make_sync_tool("SyncRead", read))
    context = make_context()

    with execution_context_scope(context):
        assert wrapped.invoke({"value": "sync"}) == f"{context.request_id}:sync"


@pytest.mark.asyncio
async def test_trace_contains_type_only_not_raw_result(tmp_path):
    async def read(value: str) -> str:
        return "secret-result-value"

    path = tmp_path / "trace.jsonl"
    gateway = ToolGateway(
        policies={"SafeTrace": make_policy("SafeTrace")},
        recorder=TraceRecorder(path),
    )
    wrapped = gateway.wrap(make_async_tool("SafeTrace", read))

    with execution_context_scope(make_context()):
        assert await wrapped.ainvoke({"value": "secret-input-value"}) == (
            "secret-result-value"
        )

    trace = path.read_text(encoding="utf-8")
    assert "secret-input-value" not in trace
    assert "secret-result-value" not in trace
    assert '"result_type":"str"' in trace


def test_default_policy_registry_matches_current_device_tools():
    assert set(TOOL_POLICIES) == {
        "QueryDevice",
        "QueryDeviceAlarms",
        "AssessAlarmRisk",
        "SearchMaintenanceKnowledge",
        "CreateWorkOrder",
        "QueryWorkOrder",
        "QueryWorkOrderByAlarm",
        "StartWorkOrder",
        "CompleteWorkOrder",
    }


def test_write_policy_requires_safe_configuration():
    with pytest.raises(ValueError, match="max_attempts=1"):
        make_policy(
            "UnsafeWrite",
            ToolOperationType.WRITE,
            max_attempts=2,
            idempotency_fields=("value",),
        )

    with pytest.raises(ValueError, match="idempotency_fields"):
        make_policy("UnkeyedWrite", ToolOperationType.WRITE)
