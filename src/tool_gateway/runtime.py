"""Context-local state shared by one Agent Harness execution."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any

from harness.context import ExecutionContext


@dataclass
class ToolRuntimeState:
    """Request-scoped state; never shared between concurrent requests."""

    context: ExecutionContext
    idempotency_cache: dict[tuple[str, str], Any] = field(default_factory=dict)


current_execution_context: ContextVar[ExecutionContext | None] = ContextVar(
    "current_execution_context",
    default=None,
)
current_tool_runtime: ContextVar[ToolRuntimeState | None] = ContextVar(
    "current_tool_runtime",
    default=None,
)


@dataclass(frozen=True)
class ExecutionContextTokens:
    """Tokens needed to restore both request-local ContextVars."""

    context_token: Token[ExecutionContext | None]
    runtime_token: Token[ToolRuntimeState | None]


def bind_execution_context(context: ExecutionContext) -> ExecutionContextTokens:
    """Bind a context and create a fresh request-level runtime cache."""

    return ExecutionContextTokens(
        context_token=current_execution_context.set(context),
        runtime_token=current_tool_runtime.set(ToolRuntimeState(context=context)),
    )


def reset_execution_context(tokens: ExecutionContextTokens) -> None:
    """Restore the ContextVar values that existed before a run."""

    current_tool_runtime.reset(tokens.runtime_token)
    current_execution_context.reset(tokens.context_token)


@contextmanager
def execution_context_scope(context: ExecutionContext) -> Iterator[None]:
    """Keep one execution context active for a sync or async call scope."""

    tokens = bind_execution_context(context)
    try:
        yield
    finally:
        reset_execution_context(tokens)


def get_current_execution_context() -> ExecutionContext | None:
    """Return the context for the current asyncio task, if one is bound."""

    return current_execution_context.get()


def get_current_tool_runtime() -> ToolRuntimeState | None:
    """Return request-local gateway state for the current asyncio task."""

    return current_tool_runtime.get()
