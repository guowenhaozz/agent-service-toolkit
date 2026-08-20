"""Final acceptance tests for request-level Gateway idempotency."""

import asyncio
import json

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from harness.context import ExecutionContext
from harness.trace import TraceRecorder
from tool_gateway import ToolGateway, ToolOperationType, ToolOutcomeUnknownError, ToolPolicy
from tool_gateway.runtime import execution_context_scope


class ValueInput(BaseModel):
    value: str


class PayloadInput(BaseModel):
    payload: dict[str, int]


def make_context(request_id: str) -> ExecutionContext:
    context = ExecutionContext.create(
        user_id="phase2-user",
        thread_id="phase2-idempotency",
        agent_name="device-assistant",
        endpoint="/phase2-final",
    )
    return context.model_copy(update={"request_id": request_id})


def make_write_policy(
    name: str,
    *,
    timeout_seconds: float = 1.0,
    idempotency_fields: tuple[str, ...] = ("value",),
) -> ToolPolicy:
    return ToolPolicy(
        tool_name=name,
        operation_type=ToolOperationType.WRITE,
        allowed_agents={"device-assistant"},
        timeout_seconds=timeout_seconds,
        max_attempts=1,
        retryable=False,
        idempotency_fields=idempotency_fields,
    )


def make_read_policy(name: str) -> ToolPolicy:
    return ToolPolicy(
        tool_name=name,
        operation_type=ToolOperationType.READ,
        allowed_agents={"device-assistant"},
        timeout_seconds=1.0,
        max_attempts=1,
        retryable=False,
    )


def read_events(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.asyncio
async def test_same_request_same_write_is_deduplicated_and_audited(tmp_path):
    """A successful WRITE result is reused only within one active request."""

    call_count = 0

    async def write(value: str) -> str:
        nonlocal call_count
        call_count += 1
        return f"saved:{value}:{call_count}"

    trace_path = tmp_path / "same-request.jsonl"
    source = StructuredTool.from_function(
        coroutine=write,
        name="FakeWriteTool",
        description="Final acceptance idempotent write tool",
        args_schema=ValueInput,
    )
    gateway = ToolGateway(
        policies={"FakeWriteTool": make_write_policy("FakeWriteTool")},
        recorder=TraceRecorder(trace_path),
    )
    wrapped = gateway.wrap(source)

    with execution_context_scope(make_context("request-same")):
        first = await wrapped.ainvoke({"value": "sensitive-work-order-key"})
        second = await wrapped.ainvoke({"value": "sensitive-work-order-key"})

    events = read_events(trace_path)
    assert first == second == "saved:sensitive-work-order-key:1"
    assert call_count == 1
    assert [event["event_type"] for event in events] == [
        "TOOL_STARTED",
        "TOOL_COMPLETED",
        "TOOL_DEDUPLICATED",
    ]
    assert events[-1]["idempotency_key_hash"] == events[-2]["idempotency_key_hash"]
    assert events[-1]["idempotency_key_hash"] is not None
    assert "sensitive-work-order-key" not in trace_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_different_write_arguments_or_request_ids_execute_again(tmp_path):
    """The in-memory key is scoped to semantic WRITE input and one request."""

    call_count = 0

    async def write(value: str) -> str:
        nonlocal call_count
        call_count += 1
        return f"saved:{value}:{call_count}"

    source = StructuredTool.from_function(
        coroutine=write,
        name="FakeScopedWrite",
        description="Final acceptance scoped write tool",
        args_schema=ValueInput,
    )
    gateway = ToolGateway(
        policies={"FakeScopedWrite": make_write_policy("FakeScopedWrite")},
        recorder=TraceRecorder(tmp_path / "scoped.jsonl"),
    )
    wrapped = gateway.wrap(source)

    with execution_context_scope(make_context("request-one")):
        first = await wrapped.ainvoke({"value": "alarm-one"})
        changed_args = await wrapped.ainvoke({"value": "alarm-two"})

    with execution_context_scope(make_context("request-two")):
        different_request = await wrapped.ainvoke({"value": "alarm-one"})

    assert [first, changed_args, different_request] == [
        "saved:alarm-one:1",
        "saved:alarm-two:2",
        "saved:alarm-one:3",
    ]
    assert call_count == 3


@pytest.mark.asyncio
async def test_semantically_identical_mapping_order_uses_one_idempotency_key(tmp_path):
    """Dictionary insertion order must not change a WRITE idempotency key."""

    call_count = 0

    async def write(payload: dict[str, int]) -> str:
        nonlocal call_count
        call_count += 1
        return f"saved:{call_count}:{payload['a']}:{payload['b']}"

    trace_path = tmp_path / "mapping-order.jsonl"
    source = StructuredTool.from_function(
        coroutine=write,
        name="FakeMappingWrite",
        description="Final acceptance mapping write tool",
        args_schema=PayloadInput,
    )
    gateway = ToolGateway(
        policies={
            "FakeMappingWrite": make_write_policy(
                "FakeMappingWrite",
                idempotency_fields=("payload",),
            )
        },
        recorder=TraceRecorder(trace_path),
    )
    wrapped = gateway.wrap(source)

    with execution_context_scope(make_context("request-mapping")):
        first = await wrapped.ainvoke({"payload": {"b": 2, "a": 1}})
        second = await wrapped.ainvoke({"payload": {"a": 1, "b": 2}})

    events = read_events(trace_path)
    assert first == second == "saved:1:1:2"
    assert call_count == 1
    assert events[-1]["event_type"] == "TOOL_DEDUPLICATED"
    assert events[-1]["idempotency_key_hash"] == events[-2]["idempotency_key_hash"]


@pytest.mark.asyncio
async def test_failed_and_timeout_writes_are_not_cached(tmp_path):
    """Only a successful completed WRITE may populate the request cache."""

    failed_calls = 0

    async def fails_once(value: str) -> str:
        nonlocal failed_calls
        failed_calls += 1
        if failed_calls == 1:
            raise RuntimeError("first write failed")
        return "saved-after-failure"

    failed_trace = tmp_path / "failed-not-cached.jsonl"
    failed_source = StructuredTool.from_function(
        coroutine=fails_once,
        name="FakeFailThenWrite",
        description="Final acceptance fail then write tool",
        args_schema=ValueInput,
    )
    failed_gateway = ToolGateway(
        policies={"FakeFailThenWrite": make_write_policy("FakeFailThenWrite")},
        recorder=TraceRecorder(failed_trace),
    )
    failed_wrapped = failed_gateway.wrap(failed_source)

    with execution_context_scope(make_context("request-failure")):
        with pytest.raises(RuntimeError, match="first write failed"):
            await failed_wrapped.ainvoke({"value": "same"})
        assert await failed_wrapped.ainvoke({"value": "same"}) == "saved-after-failure"

    assert failed_calls == 2
    assert [event["event_type"] for event in read_events(failed_trace)] == [
        "TOOL_STARTED",
        "TOOL_FAILED",
        "TOOL_STARTED",
        "TOOL_COMPLETED",
    ]

    timeout_calls = 0

    async def times_out_once(value: str) -> str:
        nonlocal timeout_calls
        timeout_calls += 1
        if timeout_calls == 1:
            await asyncio.sleep(0.05)
        return "saved-after-timeout"

    timeout_trace = tmp_path / "timeout-not-cached.jsonl"
    timeout_source = StructuredTool.from_function(
        coroutine=times_out_once,
        name="FakeTimeoutThenWrite",
        description="Final acceptance timeout then write tool",
        args_schema=ValueInput,
    )
    timeout_gateway = ToolGateway(
        policies={
            "FakeTimeoutThenWrite": make_write_policy(
                "FakeTimeoutThenWrite",
                timeout_seconds=0.005,
            )
        },
        recorder=TraceRecorder(timeout_trace),
    )
    timeout_wrapped = timeout_gateway.wrap(timeout_source)

    with execution_context_scope(make_context("request-timeout")):
        with pytest.raises(ToolOutcomeUnknownError):
            await timeout_wrapped.ainvoke({"value": "same"})
        assert await timeout_wrapped.ainvoke({"value": "same"}) == "saved-after-timeout"

    assert timeout_calls == 2
    assert [event["event_type"] for event in read_events(timeout_trace)] == [
        "TOOL_STARTED",
        "TOOL_FAILED",
        "TOOL_STARTED",
        "TOOL_COMPLETED",
    ]


@pytest.mark.asyncio
async def test_read_tools_are_never_deduplicated_by_write_idempotency(tmp_path):
    """READ operations execute twice even when their arguments are identical."""

    call_count = 0

    async def read(value: str) -> str:
        nonlocal call_count
        call_count += 1
        return f"read:{call_count}:{value}"

    trace_path = tmp_path / "read-not-deduplicated.jsonl"
    source = StructuredTool.from_function(
        coroutine=read,
        name="FakeRepeatRead",
        description="Final acceptance repeat read tool",
        args_schema=ValueInput,
    )
    gateway = ToolGateway(
        policies={"FakeRepeatRead": make_read_policy("FakeRepeatRead")},
        recorder=TraceRecorder(trace_path),
    )
    wrapped = gateway.wrap(source)

    with execution_context_scope(make_context("request-read")):
        first = await wrapped.ainvoke({"value": "same"})
        second = await wrapped.ainvoke({"value": "same"})

    assert [first, second] == ["read:1:same", "read:2:same"]
    assert call_count == 2
    assert not any(
        event["event_type"] == "TOOL_DEDUPLICATED"
        for event in read_events(trace_path)
    )
