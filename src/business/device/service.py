import re
from datetime import datetime

from business.device.repository import (
    complete_work_order_and_close_alarm,
    find_active_alarms_by_device_id,
    find_alarm_by_id,
    find_device_by_id,
    find_work_order_by_alarm_id,
    find_work_order_by_id,
    insert_work_order,
    mark_work_order_processing,
)

STATUS_TEXT = {
    "RUNNING": "运行中",
    "MAINTENANCE": "维护中",
    "OFFLINE": "离线",
}


def get_device_info(device_id: str) -> dict:
    """校验设备编号并查询设备信息。"""

    if not device_id or not device_id.strip():
        return {
            "success": False,
            "message": "设备编号不能为空。",
        }

    normalized_id = device_id.strip().upper()

    if not re.fullmatch(r"DVC-\d{3}", normalized_id):
        return {
            "success": False,
            "message": "设备编号格式错误，正确格式例如：DVC-001。",
        }

    device = find_device_by_id(normalized_id)

    if device is None:
        return {
            "success": False,
            "message": f"没有找到设备：{normalized_id}。",
        }

    device["status_text"] = STATUS_TEXT.get(device["status"], "未知状态")

    return {
        "success": True,
        "device": device,
    }


ALARM_LEVEL_TEXT = {
    "HIGH": "高",
    "MEDIUM": "中",
    "LOW": "低",
}


def get_device_active_alarms(device_id: str) -> dict:
    """查询设备当前未关闭的告警。"""

    device_result = get_device_info(device_id)

    if not device_result["success"]:
        return device_result

    normalized_id = device_result["device"]["id"]
    alarms = find_active_alarms_by_device_id(normalized_id)

    for alarm in alarms:
        alarm["alarm_level_text"] = ALARM_LEVEL_TEXT.get(
            alarm["alarm_level"],
            "未知",
        )

    return {
        "success": True,
        "device_id": normalized_id,
        "alarm_count": len(alarms),
        "alarms": alarms,
    }


PRIORITY_MAPPING = {
    "HIGH": "P1",
    "MEDIUM": "P2",
    "LOW": "P3",
}


def prepare_work_order(device_id: str, alarm_id: str) -> dict:
    """校验设备和告警，生成工单草案。"""

    device_result = get_device_info(device_id)

    if not device_result["success"]:
        return device_result

    normalized_device_id = device_result["device"]["id"]
    normalized_alarm_id = alarm_id.strip().upper()

    alarm = find_alarm_by_id(normalized_alarm_id)

    if alarm is None:
        return {
            "success": False,
            "message": f"没有找到告警：{normalized_alarm_id}。",
        }

    if alarm["device_id"] != normalized_device_id:
        return {
            "success": False,
            "message": (
                f"告警{normalized_alarm_id}不属于设备"
                f"{normalized_device_id}。"
            ),
        }

    if alarm["status"] == "CLOSED":
        return {
            "success": False,
            "message": "该告警已经关闭，不能创建维修工单。",
        }

    existing_order = find_work_order_by_alarm_id(normalized_alarm_id)

    if existing_order:
        return {
            "success": False,
            "code": "WORK_ORDER_ALREADY_EXISTS",
            "message": "该告警已经创建过维修工单。",
            "work_order": existing_order,
        }

    draft = {
        "id": f"WO-{normalized_alarm_id}",
        "device_id": normalized_device_id,
        "alarm_id": normalized_alarm_id,
        "description": alarm["message"],
        "priority": PRIORITY_MAPPING.get(
            alarm["alarm_level"],
            "P3",
        ),
        "status": "PENDING",
    }

    return {
        "success": True,
        "draft": draft,
    }


def create_work_order(draft: dict) -> dict:
    """人工确认后正式创建维修工单。"""

    work_order = {
        **draft,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    saved_order = insert_work_order(work_order)

    return {
        "success": True,
        "message": "维修工单创建成功。",
        "work_order": saved_order,
    }


WORK_ORDER_STATUS_TEXT = {
    "PENDING": "待处理",
    "PROCESSING": "处理中",
    "COMPLETED": "已完成",
    "CANCELLED": "已取消",
}

WORK_ORDER_PRIORITY_TEXT = {
    "P1": "高",
    "P2": "中",
    "P3": "低",
}


def get_work_order_info(work_order_id: str) -> dict:
    """查询维修工单详情。"""

    if not work_order_id or not work_order_id.strip():
        return {
            "success": False,
            "message": "工单编号不能为空。",
        }

    normalized_id = work_order_id.strip().upper()

    work_order = find_work_order_by_id(normalized_id)

    if work_order is None:
        return {
            "success": False,
            "message": f"没有找到维修工单：{normalized_id}。",
        }

    work_order["status_text"] = WORK_ORDER_STATUS_TEXT.get(
        work_order["status"],
        "未知状态",
    )

    work_order["priority_text"] = WORK_ORDER_PRIORITY_TEXT.get(
        work_order["priority"],
        "未知优先级",
    )

    return {
        "success": True,
        "work_order": work_order,
    }


def get_work_order_by_alarm(alarm_id: str) -> dict:
    """按告警编号查询关联的维修工单。"""

    if not alarm_id or not alarm_id.strip():
        return {
            "success": False,
            "message": "告警编号不能为空。",
        }

    normalized_alarm_id = alarm_id.strip().upper()

    if not re.fullmatch(r"ALM-\d{3}", normalized_alarm_id):
        return {
            "success": False,
            "message": "告警编号格式错误，正确格式例如：ALM-001。",
        }

    work_order = find_work_order_by_alarm_id(normalized_alarm_id)

    if work_order is None:
        return {
            "success": True,
            "found": False,
            "alarm_id": normalized_alarm_id,
            "message": "该告警当前没有关联的维修工单。",
        }

    work_order["status_text"] = WORK_ORDER_STATUS_TEXT.get(
        work_order["status"],
        "未知状态",
    )
    work_order["priority_text"] = WORK_ORDER_PRIORITY_TEXT.get(
        work_order["priority"],
        "未知优先级",
    )

    return {
        "success": True,
        "found": True,
        "alarm_id": normalized_alarm_id,
        "work_order": work_order,
    }


ALLOWED_STATUS_TRANSITIONS = {
    "PENDING": {"PROCESSING", "CANCELLED"},
    "PROCESSING": {"COMPLETED", "CANCELLED"},
    "COMPLETED": set(),
    "CANCELLED": set(),
}


def start_work_order_processing(
    work_order_id: str,
    operator: str,
) -> dict:
    """维修人员开始处理工单。"""

    if not operator or not operator.strip():
        return {
            "success": False,
            "message": "维修人员姓名不能为空。",
        }

    order_result = get_work_order_info(work_order_id)

    if not order_result["success"]:
        return order_result

    work_order = order_result["work_order"]
    current_status = work_order["status"]

    if current_status == "PROCESSING":
        return {
            "success": False,
            "message": "该工单已经处于处理中状态。",
            "work_order": work_order,
        }

    if "PROCESSING" not in ALLOWED_STATUS_TRANSITIONS.get(
        current_status,
        set(),
    ):
        return {
            "success": False,
            "message": (
                f"工单不能从{current_status}"
                "变更为PROCESSING。"
            ),
        }

    operation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    updated = mark_work_order_processing(
        work_order_id=work_order["id"],
        operator=operator.strip(),
        operation_time=operation_time,
    )

    if not updated:
        return {
            "success": False,
            "message": "工单状态已经发生变化，请重新查询后再操作。",
        }

    latest_order = get_work_order_info(work_order["id"])

    return {
        "success": True,
        "message": "维修工单已经开始处理。",
        "work_order": latest_order["work_order"],
    }


def complete_work_order(
    work_order_id: str,
    operator: str,
    resolution: str,
) -> dict:
    """维修人员提交结果并完成工单。"""

    if not operator or not operator.strip():
        return {
            "success": False,
            "message": "维修人员姓名不能为空。",
        }

    if not resolution or not resolution.strip():
        return {
            "success": False,
            "message": "维修结果不能为空。",
        }

    order_result = get_work_order_info(work_order_id)

    if not order_result["success"]:
        return order_result

    work_order = order_result["work_order"]
    current_status = work_order["status"]

    if current_status == "COMPLETED":
        return {
            "success": True,
            "code": "ALREADY_COMPLETED",
            "message": "该工单已经完成，无需重复操作。",
            "work_order": work_order,
        }

    if current_status != "PROCESSING":
        return {
            "success": False,
            "message": (
                f"工单当前状态为{current_status}，"
                "只有处理中（PROCESSING）工单可以完成。"
            ),
            "work_order": work_order,
        }

    operation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    completion_result = complete_work_order_and_close_alarm(
        work_order_id=work_order["id"],
        operator=operator.strip(),
        resolution=resolution.strip(),
        operation_time=operation_time,
    )

    if completion_result is None:
        return {
            "success": False,
            "message": "工单状态已经发生变化，请重新查询后再操作。",
        }

    latest_order = get_work_order_info(work_order["id"])

    return {
        "success": True,
        "message": "工单已完成，关联告警已关闭。",
        "work_order": latest_order["work_order"],
        **completion_result,
    }
