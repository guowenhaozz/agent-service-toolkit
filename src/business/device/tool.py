import json

from langchain_core.tools import BaseTool, tool
from langgraph.types import interrupt

from business.device.service import (
    complete_work_order,
    create_work_order,
    get_device_active_alarms,
    get_device_info,
    get_work_order_by_alarm,
    get_work_order_info,
    prepare_work_order,
    start_work_order_processing,
)


def query_device_func(device_id: str) -> str:
    """Query device information by device ID.

    Use this tool when the user asks about a device's name,
    type, running status, location, or last maintenance date.

    Args:
        device_id: Device ID, for example DVC-001.
    """

    result = get_device_info(device_id)
    return json.dumps(result, ensure_ascii=False)


query_device: BaseTool = tool(query_device_func)
query_device.name = "QueryDevice"


def query_device_alarms_func(device_id: str) -> str:
    """Query active alarms for a device.

    Use this tool when the user asks why a device is abnormal,
    offline, under maintenance, or asks about current alarms.

    Args:
        device_id: Device ID, for example DVC-002.
    """

    result = get_device_active_alarms(device_id)
    return json.dumps(result, ensure_ascii=False)


query_device_alarms: BaseTool = tool(query_device_alarms_func)
query_device_alarms.name = "QueryDeviceAlarms"

def assess_alarm_risk_func(
    alarm_id: str,
    alarm_level: str,
    alarm_status: str,
    alarm_code: str,
) -> str:
    """Assess the operational risk of an alarm with deterministic rules.

    Use this tool after querying active alarms during device diagnosis.
    Do not infer risk level yourself.

    Args:
        alarm_id: Alarm ID, for example ALM-001.
        alarm_level: Alarm severity, for example LOW, MEDIUM, HIGH, or CRITICAL.
        alarm_status: Alarm status, for example OPEN or ACKNOWLEDGED.
        alarm_code: Alarm code, for example DEPTH_TIMEOUT.
    """

    level = alarm_level.strip().upper()
    status = alarm_status.strip().upper()
    active_statuses = {"OPEN", "ACKNOWLEDGED"}
    is_active = status in active_statuses

    if level not in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}:
        return json.dumps(
            {
                "success": False,
                "code": "INVALID_ALARM_LEVEL",
                "message": f"不支持的告警级别：{alarm_level}",
            },
            ensure_ascii=False,
        )

    requires_approval = is_active and level in {"HIGH", "CRITICAL"}

    if not is_active:
        reason = "告警当前不是活跃状态，无需触发工单审批流程。"
    elif level in {"HIGH", "CRITICAL"}:
        reason = "活跃高等级告警，属于高风险事件；创建工单前必须人工审批。"
    else:
        reason = "低、中等级活跃告警，当前仅提供排查建议和持续观察，不自动创建工单。"

    return json.dumps(
        {
            "success": True,
            "alarm_id": alarm_id,
            "alarm_code": alarm_code,
            "alarm_level": level,
            "alarm_status": status,
            "risk_level": level,
            "is_active": is_active,
            "requires_approval": requires_approval,
            "should_create_work_order": requires_approval,
            "reason": reason,
        },
        ensure_ascii=False,
    )


assess_alarm_risk: BaseTool = tool(assess_alarm_risk_func)
assess_alarm_risk.name = "AssessAlarmRisk"


def create_work_order_func(device_id: str, alarm_id: str) -> str:
    """Create a maintenance work order for an active device alarm.

    Use this tool only when the user explicitly asks to create
    a maintenance work order. User confirmation is required
    before the work order is saved.

    Args:
        device_id: Device ID, for example DVC-002.
        alarm_id: Alarm ID, for example ALM-001.
    """

    draft_result = prepare_work_order(device_id, alarm_id)

    if not draft_result["success"]:
        return json.dumps(draft_result, ensure_ascii=False)

    draft = draft_result["draft"]

    approval = interrupt(
        "即将创建维修工单：\n"
        f"- 工单编号：{draft['id']}\n"
        f"- 设备编号：{draft['device_id']}\n"
        f"- 告警编号：{draft['alarm_id']}\n"
        f"- 故障描述：{draft['description']}\n"
        f"- 优先级：{draft['priority']}\n"
        "请输入“确认创建”或“取消”。"
    )

    approved_values = {
        "确认创建",
        "确认",
        "批准",
        "approve",
        "yes",
    }

    cancelled_values = {
        "取消",
        "取消创建",
        "cancel",
        "no",
    }

    decision = str(approval).strip().lower()

    if decision in approved_values:
        result = create_work_order(draft)
        return json.dumps(result, ensure_ascii=False)

    if decision in cancelled_values:
        return json.dumps(
            {
                "success": False,
                "code": "USER_CANCELLED",
                "message": "用户取消创建维修工单。",
            },
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "success": False,
            "code": "INVALID_APPROVAL",
            "message": "无法识别审批指令，工单未创建。请重新发起并回复“确认创建”或“取消”。",
        },
        ensure_ascii=False,
    )


create_work_order_tool: BaseTool = tool(create_work_order_func)
create_work_order_tool.name = "CreateWorkOrder"


def query_work_order_func(work_order_id: str) -> str:
    """Query maintenance work order information.

    Use this tool when the user asks about a work order's
    device, alarm, priority, status, or creation time.

    Args:
        work_order_id: Work order ID, for example WO-ALM-001.
    """

    result = get_work_order_info(work_order_id)
    return json.dumps(result, ensure_ascii=False)


query_work_order: BaseTool = tool(query_work_order_func)
query_work_order.name = "QueryWorkOrder"


def query_work_order_by_alarm_func(alarm_id: str) -> str:
    """Query the maintenance work order associated with an alarm.

    Use this tool during high-risk alarm handling before creating a work
    order. It determines whether the alarm already has a work order.

    Args:
        alarm_id: Alarm ID, for example ALM-001.
    """

    result = get_work_order_by_alarm(alarm_id)
    return json.dumps(result, ensure_ascii=False)


query_work_order_by_alarm: BaseTool = tool(query_work_order_by_alarm_func)
query_work_order_by_alarm.name = "QueryWorkOrderByAlarm"


def start_work_order_func(
    work_order_id: str,
    operator: str,
) -> str:
    """Start processing a pending maintenance work order.

    Use this tool only when the user explicitly requests that
    a maintenance worker starts processing a work order.

    Args:
        work_order_id: Work order ID, for example WO-ALM-001.
        operator: Name or ID of the maintenance worker.
    """

    result = start_work_order_processing(
        work_order_id=work_order_id,
        operator=operator,
    )

    return json.dumps(result, ensure_ascii=False)


start_work_order: BaseTool = tool(start_work_order_func)
start_work_order.name = "StartWorkOrder"


def complete_work_order_func(
    work_order_id: str,
    operator: str,
    resolution: str,
) -> str:
    """Complete a processing maintenance work order.

    Use this tool only when a named maintenance worker explicitly reports
    that the repair is finished and provides a maintenance result.

    Args:
        work_order_id: Work order ID, for example WO-ALM-001.
        operator: Name or ID of the maintenance worker.
        resolution: Actual repair result and verification outcome.
    """

    result = complete_work_order(
        work_order_id=work_order_id,
        operator=operator,
        resolution=resolution,
    )
    return json.dumps(result, ensure_ascii=False)


complete_work_order_tool: BaseTool = tool(complete_work_order_func)
complete_work_order_tool.name = "CompleteWorkOrder"
