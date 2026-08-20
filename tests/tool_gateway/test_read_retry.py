"""Final acceptance tests for READ retry behavior."""

import json

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from core import settings
from harness.context import ExecutionContext
from harness.trace import TraceRecorder
from tool_gateway import ToolGateway, ToolOperationType, ToolPolicy
from tool_gateway.runtime import execution_context_scope


class ReadInput(BaseModel):
    value: str


def make_context() -> ExecutionContext:
    return ExecutionContext.create(
        user_id="phase2-user",
        thread_id="phase2-read-retry",
        agent_name="device-assistant",
        endpoint="/phase2-final",
    )


def make_read_policy(name: str) -> ToolPolicy:
    return ToolPolicy(
        tool_name=name,
        operation_type=ToolOperationType.READ,
        allowed_agents={"device-assistant"},
        timeout_seconds=1.0,
        max_attempts=2,
        retryable=True,
    )


def make_async_tool(name: str, coroutine) -> StructuredTool:
    return StructuredTool.from_function(
        coroutine=coroutine,
        name=name,
        description=f"Phase 2 final acceptance tool: {name}",
        args_schema=ReadInput,
    )


def read_events(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.asyncio
async def test_flaky_read_retries_once_then_completes(monkeypatch, tmp_path):
    """A retryable READ failure runs the real function exactly twice."""

    call_count = 0

    async def flaky_read(value: str) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectionError("temporary transport failure")
        return f"read:{value}"

    monkeypatch.setattr(settings, "TOOL_RETRY_BACKOFF_SECONDS", 0.0)
    trace_path = tmp_path / "read-retry.jsonl"
    gateway = ToolGateway(
        policies={"FakeFlakyReadTool": make_read_policy("FakeFlakyReadTool")},
        recorder=TraceRecorder(trace_path),
    )
    wrapped = gateway.wrap(make_async_tool("FakeFlakyReadTool", flaky_read))

    with execution_context_scope(make_context()):
        result = await wrapped.ainvoke({"value": "device-state"})

    events = read_events(trace_path)
    assert result == "read:device-state"
    assert call_count == 2
    assert [event["event_type"] for event in events] == [
        "TOOL_STARTED",
        "TOOL_RETRY",
        "TOOL_STARTED",
        "TOOL_COMPLETED",
    ]
    assert events[0]["attempt"] == 1
    assert events[1]["attempt"] == 1
    assert events[2]["attempt"] == 2
    assert events[3]["attempt"] == 2
    assert events[3]["result_type"] == "str"


@pytest.mark.asyncio
async def test_read_validation_and_business_errors_do_not_retry(tmp_path):
    """Invalid input and explicit business errors cannot trigger READ retries."""

    validation_calls = 0

    async def validation_target(value: str) -> str:
        nonlocal validation_calls
        validation_calls += 1
        return value

    validation_trace = tmp_path / "validation.jsonl"
    validation_gateway = ToolGateway(
        policies={"FakeValidatedRead": make_read_policy("FakeValidatedRead")},
        recorder=TraceRecorder(validation_trace),
    )
    validation_tool = validation_gateway.wrap(
        make_async_tool("FakeValidatedRead", validation_target)
    )

    with execution_context_scope(make_context()):
        with pytest.raises(Exception):
            await validation_tool.ainvoke({})

    assert validation_calls == 0
    assert [event["event_type"] for event in read_events(validation_trace)] == [
        "TOOL_REJECTED"
    ]

    business_calls = 0

    async def business_target(value: str) -> str:
        nonlocal business_calls
        business_calls += 1
        raise ValueError("business rule rejected")

    business_trace = tmp_path / "business-error.jsonl"
    business_gateway = ToolGateway(
        policies={"FakeBusinessRead": make_read_policy("FakeBusinessRead")},
        recorder=TraceRecorder(business_trace),
    )
    business_tool = business_gateway.wrap(
        make_async_tool("FakeBusinessRead", business_target)
    )

    with execution_context_scope(make_context()):
        with pytest.raises(ValueError, match="business rule rejected"):
            await business_tool.ainvoke({"value": "invalid-business-state"})

    business_events = read_events(business_trace)
    assert business_calls == 1
    assert [event["event_type"] for event in business_events] == [
        "TOOL_STARTED",
        "TOOL_FAILED",
    ]
    assert business_events[-1]["attempt"] == 1
