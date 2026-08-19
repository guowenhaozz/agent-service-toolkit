"""Immutable execution context shared by one harness run."""

from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ExecutionContext(BaseModel):
    """Trusted request metadata used by the harness and trace events."""

    model_config = ConfigDict(frozen=True)

    request_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    thread_id: str = Field(min_length=1)
    agent_name: str = Field(min_length=1)
    endpoint: str = Field(min_length=1)
    started_at: datetime

    @field_validator("started_at")
    @classmethod
    def normalize_started_at(cls, value: datetime) -> datetime:
        """Require an aware timestamp and normalize it to UTC."""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("started_at must include timezone information")
        return value.astimezone(UTC)

    @classmethod
    def create(
        cls,
        *,
        user_id: str,
        thread_id: str,
        agent_name: str,
        endpoint: str,
    ) -> "ExecutionContext":
        """Create a context after the service has resolved trusted IDs."""

        return cls(
            request_id=str(uuid4()),
            user_id=user_id,
            thread_id=thread_id,
            agent_name=agent_name,
            endpoint=endpoint,
            started_at=datetime.now(UTC),
        )
