import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DB_PATH = PROJECT_ROOT / "data" / "business.db"


def initialize_device_table() -> None:
    """创建设备表并写入演示数据。"""

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS device (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                device_type TEXT NOT NULL,
                status TEXT NOT NULL,
                location TEXT NOT NULL,
                last_maintenance TEXT
            )
            """
        )

        connection.executemany(
            """
            INSERT OR IGNORE INTO device (
                id,
                name,
                device_type,
                status,
                location,
                last_maintenance
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "DVC-001",
                    "激光疏蕾作业平台",
                    "农业机器人",
                    "RUNNING",
                    "猕猴桃试验园A区",
                    "2026-07-20",
                ),
                (
                    "DVC-002",
                    "RGB-D视觉检测终端",
                    "视觉设备",
                    "MAINTENANCE",
                    "设备实验室",
                    "2026-08-10",
                ),
                (
                    "DVC-003",
                    "双轴振镜控制器",
                    "执行设备",
                    "OFFLINE",
                    "猕猴桃试验园B区",
                    "2026-06-15",
                ),
            ],
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS device_alarm (
                id TEXT PRIMARY KEY,
                device_id TEXT NOT NULL,
                alarm_code TEXT NOT NULL,
                alarm_level TEXT NOT NULL,
                message TEXT NOT NULL,
                status TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                FOREIGN KEY (device_id) REFERENCES device(id)
            )
            """
        )

        connection.executemany(
            """
            INSERT OR IGNORE INTO device_alarm (
                id,
                device_id,
                alarm_code,
                alarm_level,
                message,
                status,
                occurred_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "ALM-001",
                    "DVC-002",
                    "DEPTH_TIMEOUT",
                    "HIGH",
                    "深度相机连续读取超时",
                    "OPEN",
                    "2026-08-17 09:20:00",
                ),
                (
                    "ALM-002",
                    "DVC-002",
                    "TEMPERATURE_HIGH",
                    "MEDIUM",
                    "设备内部温度超过预警阈值",
                    "ACKNOWLEDGED",
                    "2026-08-17 09:35:00",
                ),
                (
                    "ALM-003",
                    "DVC-003",
                    "GALVO_CONNECTION_LOST",
                    "HIGH",
                    "振镜控制器通信连接中断",
                    "OPEN",
                    "2026-08-17 10:10:00",
                ),
            ],
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS work_order (
                id TEXT PRIMARY KEY,
                device_id TEXT NOT NULL,
                alarm_id TEXT NOT NULL UNIQUE,
                description TEXT NOT NULL,
                priority TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (device_id) REFERENCES device(id),
                FOREIGN KEY (alarm_id) REFERENCES device_alarm(id)
            )
            """
        )

        work_order_columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(work_order)"
            ).fetchall()
        }

        if "assignee" not in work_order_columns:
            connection.execute(
                "ALTER TABLE work_order ADD COLUMN assignee TEXT"
            )

        if "updated_at" not in work_order_columns:
            connection.execute(
                "ALTER TABLE work_order ADD COLUMN updated_at TEXT"
            )

        if "completed_at" not in work_order_columns:
            connection.execute(
                "ALTER TABLE work_order ADD COLUMN completed_at TEXT"
            )

        if "resolution" not in work_order_columns:
            connection.execute(
                "ALTER TABLE work_order ADD COLUMN resolution TEXT"
            )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS work_order_operation_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                work_order_id TEXT NOT NULL,
                operator TEXT NOT NULL,
                action TEXT NOT NULL,
                from_status TEXT,
                to_status TEXT,
                operation_time TEXT NOT NULL,
                remark TEXT,
                FOREIGN KEY (work_order_id) REFERENCES work_order(id)
            )
            """
        )


def find_device_by_id(device_id: str) -> dict | None:
    """根据设备编号查询设备。"""

    initialize_device_table()

    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row

        row = connection.execute(
            """
            SELECT
                id,
                name,
                device_type,
                status,
                location,
                last_maintenance
            FROM device
            WHERE id = ?
            """,
            (device_id,),
        ).fetchone()

    return dict(row) if row else None


def find_active_alarms_by_device_id(device_id: str) -> list[dict]:
    """查询设备尚未关闭的告警。"""

    initialize_device_table()

    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row

        rows = connection.execute(
            """
            SELECT
                id,
                device_id,
                alarm_code,
                alarm_level,
                message,
                status,
                occurred_at
            FROM device_alarm
            WHERE device_id = ?
              AND status IN ('OPEN', 'ACKNOWLEDGED')
            ORDER BY occurred_at DESC
            """,
            (device_id,),
        ).fetchall()

    return [dict(row) for row in rows]


def find_alarm_by_id(alarm_id: str) -> dict | None:
    """根据告警编号查询告警。"""

    initialize_device_table()

    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row

        row = connection.execute(
            """
            SELECT
                id,
                device_id,
                alarm_code,
                alarm_level,
                message,
                status,
                occurred_at
            FROM device_alarm
            WHERE id = ?
            """,
            (alarm_id,),
        ).fetchone()

    return dict(row) if row else None


def find_work_order_by_alarm_id(alarm_id: str) -> dict | None:
    """查询告警是否已经创建工单。"""

    initialize_device_table()

    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row

        row = connection.execute(
            """
            SELECT
                id,
                device_id,
                alarm_id,
                description,
                priority,
                status,
                created_at
            FROM work_order
            WHERE alarm_id = ?
            """,
            (alarm_id,),
        ).fetchone()

    return dict(row) if row else None


def insert_work_order(work_order: dict) -> dict:
    """保存维修工单。"""

    initialize_device_table()

    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            INSERT INTO work_order (
                id,
                device_id,
                alarm_id,
                description,
                priority,
                status,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                work_order["id"],
                work_order["device_id"],
                work_order["alarm_id"],
                work_order["description"],
                work_order["priority"],
                work_order["status"],
                work_order["created_at"],
            ),
        )
        connection.execute(
            """
            INSERT INTO work_order_operation_log (
                work_order_id,
                operator,
                action,
                from_status,
                to_status,
                operation_time,
                remark
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                work_order["id"],
                "SYSTEM",
                "CREATE_WORK_ORDER",
                None,
                work_order["status"],
                work_order["created_at"],
                f"Agent 根据告警 {work_order['alarm_id']} 创建维修工单",
            ),
        )

    return work_order


def find_work_order_by_id(work_order_id: str) -> dict | None:
    """查询工单及其关联的设备、告警信息。"""

    initialize_device_table()

    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row

        row = connection.execute(
            """
            SELECT
                wo.id,
                wo.device_id,
                d.name AS device_name,
                wo.alarm_id,
                da.alarm_code,
                da.message AS alarm_message,
                da.alarm_level,
                wo.description,
                wo.priority,
                wo.status,
                wo.created_at,
                wo.assignee,
                wo.updated_at,
                wo.completed_at,
                wo.resolution
            FROM work_order wo
            INNER JOIN device d
                ON wo.device_id = d.id
            INNER JOIN device_alarm da
                ON wo.alarm_id = da.id
            WHERE wo.id = ?
            """,
            (work_order_id,),
        ).fetchone()

    return dict(row) if row else None


def mark_work_order_processing(
    work_order_id: str,
    operator: str,
    operation_time: str,
) -> bool:
    """将待处理工单更新为处理中，并写入操作日志。"""

    initialize_device_table()

    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.execute(
            """
            UPDATE work_order
            SET
                status = 'PROCESSING',
                assignee = ?,
                updated_at = ?
            WHERE id = ?
              AND status = 'PENDING'
            """,
            (
                operator,
                operation_time,
                work_order_id,
            ),
        )

        if cursor.rowcount != 1:
            return False

        connection.execute(
            """
            INSERT INTO work_order_operation_log (
                work_order_id,
                operator,
                action,
                from_status,
                to_status,
                operation_time,
                remark
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                work_order_id,
                operator,
                "START_PROCESSING",
                "PENDING",
                "PROCESSING",
                operation_time,
                "维修人员开始处理工单",
            ),
        )
def complete_work_order_and_close_alarm(
    work_order_id: str,
    operator: str,
    resolution: str,
    operation_time: str,
) -> dict | None:
    """完成工单、关闭关联告警，并刷新设备状态。"""

    initialize_device_table()

    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row

        work_order = connection.execute(
            """
            SELECT id, device_id, alarm_id, status
            FROM work_order
            WHERE id = ?
            """,
            (work_order_id,),
        ).fetchone()

        if work_order is None or work_order["status"] != "PROCESSING":
            return None

        cursor = connection.execute(
            """
            UPDATE work_order
            SET
                status = 'COMPLETED',
                updated_at = ?,
                completed_at = ?,
                resolution = ?
            WHERE id = ?
              AND status = 'PROCESSING'
            """,
            (
                operation_time,
                operation_time,
                resolution,
                work_order_id,
            ),
        )

        if cursor.rowcount != 1:
            return None

        connection.execute(
            """
            UPDATE device_alarm
            SET status = 'CLOSED'
            WHERE id = ?
              AND status IN ('OPEN', 'ACKNOWLEDGED')
            """,
            (work_order["alarm_id"],),
        )

        remaining_alarm_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM device_alarm
            WHERE device_id = ?
              AND status IN ('OPEN', 'ACKNOWLEDGED')
            """,
            (work_order["device_id"],),
        ).fetchone()[0]

        device = connection.execute(
            """
            SELECT status
            FROM device
            WHERE id = ?
            """,
            (work_order["device_id"],),
        ).fetchone()

        device_status = device["status"]

        if device_status == "MAINTENANCE":
            device_status = (
                "RUNNING"
                if remaining_alarm_count == 0
                else "MAINTENANCE"
            )
            connection.execute(
                """
                UPDATE device
                SET status = ?
                WHERE id = ?
                """,
                (device_status, work_order["device_id"]),
            )

        connection.execute(
            """
            INSERT INTO work_order_operation_log (
                work_order_id,
                operator,
                action,
                from_status,
                to_status,
                operation_time,
                remark
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                work_order_id,
                operator,
                "COMPLETE_WORK_ORDER",
                "PROCESSING",
                "COMPLETED",
                operation_time,
                resolution,
            ),
        )

    return {
        "device_id": work_order["device_id"],
        "alarm_id": work_order["alarm_id"],
        "alarm_status": "CLOSED",
        "remaining_active_alarm_count": remaining_alarm_count,
        "device_status": device_status,
    }


    return True
