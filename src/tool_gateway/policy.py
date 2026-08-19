"""Tool operation policies and the default device-agent policy registry."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core import settings


class ToolOperationType(StrEnum):
    """The side-effect class used to select gateway safeguards."""

    READ = "READ"
    WRITE = "WRITE"
    PURE = "PURE"


class ToolPolicy(BaseModel):
    """Execution policy for one registered tool."""

    model_config = ConfigDict(frozen=True)

    tool_name: str = Field(min_length=1)
    operation_type: ToolOperationType
    allowed_agents: set[str] = Field(min_length=1)
    timeout_seconds: float = Field(gt=0)
    max_attempts: int = Field(ge=1)
    retryable: bool = False
    idempotency_fields: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_operation_rules(self) -> "ToolPolicy":
        if self.operation_type == ToolOperationType.WRITE:
            if self.max_attempts != 1:
                raise ValueError("WRITE tools must have max_attempts=1")
            if self.retryable:
                raise ValueError("WRITE tools cannot be retryable")
            if not self.idempotency_fields:
                raise ValueError("WRITE tools must declare idempotency_fields")
        elif self.idempotency_fields:
            raise ValueError("Only WRITE tools may declare idempotency_fields")

        return self


_DEVICE_AGENT = frozenset({"device-assistant"})


def _policy(
    tool_name: str,
    operation_type: ToolOperationType,
    *,
    retryable: bool = False,
    max_attempts: int = 1,
    idempotency_fields: tuple[str, ...] = (),
) -> ToolPolicy:
    return ToolPolicy(
        tool_name=tool_name,
        operation_type=operation_type,
        allowed_agents=set(_DEVICE_AGENT),
        timeout_seconds=settings.TOOL_DEFAULT_TIMEOUT_SECONDS,
        max_attempts=max_attempts,
        retryable=retryable,
        idempotency_fields=idempotency_fields,
    )


# This registry intentionally lists the tools that exist in the current
# device-assistant implementation. Future agents can register additional
# policies without changing the gateway's default-deny behavior.
TOOL_POLICIES: dict[str, ToolPolicy] = {
    "QueryDevice": _policy(
        "QueryDevice",
        ToolOperationType.READ,
        retryable=True,
        max_attempts=settings.TOOL_READ_MAX_ATTEMPTS,
    ),
    "QueryDeviceAlarms": _policy(
        "QueryDeviceAlarms",
        ToolOperationType.READ,
        retryable=True,
        max_attempts=settings.TOOL_READ_MAX_ATTEMPTS,
    ),
    "SearchMaintenanceKnowledge": _policy(
        "SearchMaintenanceKnowledge",
        ToolOperationType.READ,
        retryable=True,
        max_attempts=settings.TOOL_READ_MAX_ATTEMPTS,
    ),
    "QueryWorkOrder": _policy(
        "QueryWorkOrder",
        ToolOperationType.READ,
        retryable=True,
        max_attempts=settings.TOOL_READ_MAX_ATTEMPTS,
    ),
    "QueryWorkOrderByAlarm": _policy(
        "QueryWorkOrderByAlarm",
        ToolOperationType.READ,
        retryable=True,
        max_attempts=settings.TOOL_READ_MAX_ATTEMPTS,
    ),
    "AssessAlarmRisk": _policy("AssessAlarmRisk", ToolOperationType.PURE),
    "CreateWorkOrder": _policy(
        "CreateWorkOrder",
        ToolOperationType.WRITE,
        idempotency_fields=("alarm_id",),
    ),
    "StartWorkOrder": _policy(
        "StartWorkOrder",
        ToolOperationType.WRITE,
        idempotency_fields=("work_order_id",),
    ),
    "CompleteWorkOrder": _policy(
        "CompleteWorkOrder",
        ToolOperationType.WRITE,
        idempotency_fields=("work_order_id",),
    ),
}


def register_tool_policy(policy: ToolPolicy) -> None:
    """Register a policy explicitly for a future tool or agent."""

    TOOL_POLICIES[policy.tool_name] = policy


def get_tool_policy(tool_name: str) -> ToolPolicy | None:
    """Return the policy for a tool, or ``None`` when it is unregistered."""

    return TOOL_POLICIES.get(tool_name)
