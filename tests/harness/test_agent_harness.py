"""Unit tests for the generic Agent Harness lifecycle."""

import asyncio
import json

import pytest

from harness import AgentHarness, ExecutionContext, TraceRecorder


def make_context() -> ExecutionContext:
    return ExecutionContext.create(
        user_id="user-1",
        thread_id="thread-1",
        agent_name="fake-agent",
        endpoint="/invoke",
    )


def read_events(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class SuccessfulAgent:
    async def ainvoke(self, **kwargs):
        self.kwargs = kwargs
        return {"answer": "model output"}


class FailingAgent:
    async def ainvoke(self, **kwargs):
        raise RuntimeError("AUTH_SECRET=top-secret")


class StreamAgent:
    def __init__(self, chunks):
        self.chunks = chunks

    async def astream(self, **kwargs):
        for chunk in self.chunks:
            yield chunk


class FailingStreamAgent:
    async def astream(self, **kwargs):
        yield "first"
        raise ValueError("stream failed")


class CancelledStreamAgent:
    async def astream(self, **kwargs):
        raise asyncio.CancelledError
        yield  # pragma: no cover


def test_execution_context_generates_unique_utc_request_ids():
    first = make_context()
    second = make_context()

    assert first.request_id != second.request_id
    assert first.started_at.tzinfo is not None
    assert first.started_at.utcoffset().total_seconds() == 0
    with pytest.raises(Exception):
        first.request_id = "changed"


@pytest.mark.asyncio
async def test_invoke_records_start_and_completion_and_preserves_result(tmp_path):
    path = tmp_path / "trace.jsonl"
    harness = AgentHarness(TraceRecorder(path))
    agent = SuccessfulAgent()
    result = await harness.invoke(agent, context=make_context(), input={"message": "hello"})

    assert result == {"answer": "model output"}
    assert agent.kwargs == {"input": {"message": "hello"}}
    events = read_events(path)
    assert [event["event_type"] for event in events] == ["RUN_STARTED", "RUN_COMPLETED"]
    assert len({event["request_id"] for event in events}) == 1
    assert events[1]["duration_ms"] >= 0
    assert events[1]["result_type"] == "dict"


@pytest.mark.asyncio
async def test_invoke_records_failure_and_reraises_original_exception(tmp_path):
    path = tmp_path / "trace.jsonl"
    harness = AgentHarness(TraceRecorder(path))

    with pytest.raises(RuntimeError, match="AUTH_SECRET=top-secret") as caught:
        await harness.invoke(FailingAgent(), context=make_context())

    assert isinstance(caught.value, RuntimeError)
    events = read_events(path)
    assert [event["event_type"] for event in events] == ["RUN_STARTED", "RUN_FAILED"]
    assert events[1]["error_type"] == "RuntimeError"
    assert "top-secret" not in path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_stream_forwards_chunks_and_records_completion(tmp_path):
    path = tmp_path / "trace.jsonl"
    harness = AgentHarness(TraceRecorder(path))

    chunks = ["one", {"two": 2}, "three"]
    received = [
        chunk
        async for chunk in harness.stream(StreamAgent(chunks), context=make_context())
    ]

    assert received == chunks
    events = read_events(path)
    assert [event["event_type"] for event in events] == ["RUN_STARTED", "RUN_COMPLETED"]
    assert events[1]["result_type"] == "stream"


@pytest.mark.asyncio
async def test_stream_failure_records_failed_and_reraises(tmp_path):
    path = tmp_path / "trace.jsonl"
    harness = AgentHarness(TraceRecorder(path))

    with pytest.raises(ValueError, match="stream failed"):
        _ = [
            chunk
            async for chunk in harness.stream(
                FailingStreamAgent(),
                context=make_context(),
            )
        ]

    assert [event["event_type"] for event in read_events(path)] == [
        "RUN_STARTED",
        "RUN_FAILED",
    ]


@pytest.mark.asyncio
async def test_stream_cancellation_records_cancelled_and_reraises(tmp_path):
    path = tmp_path / "trace.jsonl"
    harness = AgentHarness(TraceRecorder(path))

    with pytest.raises(asyncio.CancelledError):
        _ = [
            chunk
            async for chunk in harness.stream(
                CancelledStreamAgent(),
                context=make_context(),
            )
        ]

    assert [event["event_type"] for event in read_events(path)] == [
        "RUN_STARTED",
        "RUN_CANCELLED",
    ]


@pytest.mark.asyncio
async def test_trace_disabled_does_not_change_agent_result_or_create_file(tmp_path):
    path = tmp_path / "disabled.jsonl"
    harness = AgentHarness(TraceRecorder(path, enabled=False))

    assert await harness.invoke(SuccessfulAgent(), context=make_context()) == {
        "answer": "model output"
    }
    assert not path.exists()


@pytest.mark.asyncio
async def test_trace_failure_does_not_change_agent_result():
    class BrokenRecorder:
        def record_event(self, **kwargs):
            raise OSError("trace destination unavailable")

    harness = AgentHarness(BrokenRecorder())

    assert await harness.invoke(SuccessfulAgent(), context=make_context()) == {
        "answer": "model output"
    }
