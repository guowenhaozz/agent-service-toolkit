"""Final acceptance tests for WRITE retry safety."""

import json

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from harness.context import ExecutionContext
from harness.trace import TraceRecorder
from tool_gateway import ToolGateway, ToolOperationType, ToolPolicy
from tool_gateway.runtime import execution_context_scope


class WriteInput(BaseModel):
    value: str


def make_context() -> ExecutionContext:
    return ExecutionContext.create(
        user_id="phase2-user",
        thread_id="phase2-write-policy",
        agent_name="device-assistant",
        endpoint="/phase2-final",
    )


def make_write_policy(name: str) -> ToolPolicy:
    return ToolPolicy(
        tool_name=name,
        operation_type=ToolOperationType.WRITE,
        allowed_agents={"device-assistant"},
        timeout_seconds=1.0,
        max_attempts=1,
        retryable=False,
        idempotency_fields=("value",),
    )


def read_events(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.asyncio
async def test_failing_write_is_called_once_and_original_error_propagates(tmp_path):
    """WRITE errors must never be retried by the Gateway."""

    call_count = 0

    async def failing_write(value: str) -> str:
        nonlocal call_count
        call_count += 1
        raise RuntimeError("simulated write failure")

    trace_path = tmp_path / "write-failure.jsonl"
    policy = make_write_policy("FakeFailingWriteTool")
    source = StructuredTool.from_function(
        coroutine=failing_write,
        name="FakeFailingWriteTool",
        description="Final acceptance failing write tool",
        args_schema=WriteInput,
    )
    gateway = ToolGateway(
        policies={"FakeFailingWriteTool": policy},
        recorder=TraceRecorder(trace_path),
    )
    wrapped = gateway.wrap(source)

    with execution_context_scope(make_context()):
        with pytest.raises(RuntimeError, match="simulated write failure"):
            await wrapped.ainvoke({"value": "save"})

    events = read_events(trace_path)
    assert call_count == 1
    assert [event["event_type"] for event in events] == [
        "TOOL_STARTED",
        "TOOL_FAILED",
    ]
    assert [event["attempt"] for event in events] == [1, 1]
    assert all(event["max_attempts"] == 1 for event in events)
    assert not any(event["event_type"] == "TOOL_RETRY" for event in events)
    assert events[-1]["error_type"] == "RuntimeError"
