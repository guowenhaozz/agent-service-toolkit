"""Database-backed work-order idempotency acceptance tests."""

import json
import sqlite3

import pytest

from business.device import repository as device_repository
from business.device import tool as device_tool


@pytest.mark.asyncio
async def test_real_create_work_order_is_business_idempotent_in_temporary_sqlite(
    monkeypatch,
    tmp_path,
):
    """The durable alarm_id UNIQUE key prevents duplicate work orders.

    This test redirects the repository before any business function is invoked,
    so it never reads or writes the project's real data/business.db.
    """

    real_database_path = device_repository.DB_PATH
    temporary_db = tmp_path / "business-idempotency.db"
    monkeypatch.setattr(device_repository, "DB_PATH", temporary_db)
    monkeypatch.setattr(device_tool, "interrupt", lambda _: "approve")

    first_raw = await device_tool.create_work_order_tool.ainvoke(
        {"device_id": "DVC-002", "alarm_id": "ALM-001"}
    )
    second_raw = await device_tool.create_work_order_tool.ainvoke(
        {"device_id": "DVC-002", "alarm_id": "ALM-001"}
    )

    first = json.loads(first_raw)
    second = json.loads(second_raw)

    with sqlite3.connect(temporary_db) as connection:
        work_order_count = connection.execute(
            "SELECT COUNT(*) FROM work_order WHERE alarm_id = ?",
            ("ALM-001",),
        ).fetchone()[0]
        operation_log_count = connection.execute(
            "SELECT COUNT(*) FROM work_order_operation_log WHERE work_order_id = ?",
            ("WO-ALM-001",),
        ).fetchone()[0]
        work_order_schema = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'work_order'"
        ).fetchone()[0]

    assert temporary_db.exists()
    assert temporary_db != real_database_path
    assert temporary_db.parent == tmp_path
    assert isinstance(first_raw, str)
    assert isinstance(second_raw, str)
    assert first["success"] is True
    assert first["work_order"]["id"] == "WO-ALM-001"
    assert second["success"] is False
    assert second["code"] == "WORK_ORDER_ALREADY_EXISTS"
    assert second["work_order"]["id"] == first["work_order"]["id"]
    assert work_order_count == 1
    assert operation_log_count == 1
    assert "alarm_id TEXT NOT NULL UNIQUE" in work_order_schema
