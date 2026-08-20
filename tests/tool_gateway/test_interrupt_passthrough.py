"""Final acceptance test for real LangGraph HITL interrupt/resume behavior."""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
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
        thread_id="phase2-hitl-thread",
        agent_name="device-assistant",
        endpoint="/phase2-final",
    )


def read_events(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.asyncio
async def test_graph_interrupt_is_not_retried_or_cached_and_resume_writes_once(tmp_path):
    """Use LangGraph's real Checkpointer and Command(resume=...) semantics."""

    write_count = 0

    async def approval_gated_write(value: str) -> str:
        nonlocal write_count
        approval = interrupt({"kind": "write_approval", "value": value})
        write_count += 1
        return f"written:{value}:{approval}"

    trace_path = tmp_path / "hitl.jsonl"
    policy = ToolPolicy(
        tool_name="FakeInterruptWrite",
        operation_type=ToolOperationType.WRITE,
        allowed_agents={"device-assistant"},
        timeout_seconds=1.0,
        max_attempts=1,
        retryable=False,
        idempotency_fields=("value",),
    )
    source = StructuredTool.from_function(
        coroutine=approval_gated_write,
        name="FakeInterruptWrite",
        description="Final acceptance HITL write tool",
        args_schema=WriteInput,
    )
    gateway = ToolGateway(
        policies={"FakeInterruptWrite": policy},
        recorder=TraceRecorder(trace_path),
    )
    wrapped = gateway.wrap(source)

    async def issue_write(state: MessagesState) -> MessagesState:
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "FakeInterruptWrite",
                            "args": {"value": "ALM-001"},
                            "id": "phase2-hitl-call",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        }

    async def finish(state: MessagesState) -> MessagesState:
        return {"messages": [AIMessage(content="write completed")]}

    builder = StateGraph(MessagesState)
    builder.add_node("issue_write", issue_write)
    builder.add_node("tools", ToolNode([wrapped]))
    builder.add_node("finish", finish)
    builder.set_entry_point("issue_write")
    builder.add_edge("issue_write", "tools")
    builder.add_edge("tools", "finish")
    builder.add_edge("finish", END)
    graph = builder.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "phase2-hitl-thread"}}

    context = make_context()
    with execution_context_scope(context):
        interrupted = await graph.ainvoke(
            {"messages": [HumanMessage(content="create the approved work order")]},
            config=config,
        )

        assert "__interrupt__" in interrupted
        assert write_count == 0

        completed = await graph.ainvoke(Command(resume="approved"), config=config)

    events = read_events(trace_path)
    tool_messages = [
        message for message in completed["messages"] if isinstance(message, ToolMessage)
    ]
    assert write_count == 1
    assert completed["messages"][-1].content == "write completed"
    assert len(tool_messages) == 1
    assert tool_messages[0].content == "written:ALM-001:approved"
    assert [event["event_type"] for event in events] == [
        "TOOL_STARTED",
        "TOOL_INTERRUPTED",
        "TOOL_STARTED",
        "TOOL_COMPLETED",
    ]
    assert [event["attempt"] for event in events] == [1, 1, 1, 1]
    assert not any(event["event_type"] == "TOOL_RETRY" for event in events)
    assert not any(event["event_type"] == "TOOL_DEDUPLICATED" for event in events)
    assert all(event["request_id"] == context.request_id for event in events)
