"""
SQLite-backed order CRUD
- table_id   : 1-10 for dine-in, NULL for takeaway
- overlap    : ±45 min guard in assign_table()
- booking_time handled as datetime everywhere (ISO string only inside DB)
"""
from pathlib import Path
import sqlite3
import uuid
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from react_agent.utils.react_constants import DEFAULT_TZ
import pendulum
import os
DB_PATH =f"{os.getcwd()}\\react_agent\\store\\orders_db.db"
# DB_PATH.parent.mkdir(parents=True, exist_ok=True)   # ensure folder exists

def to_aware_dt(dt) -> pendulum.DateTime:
    """Convert str or datetime to Pendulum aware datetime in DEFAULT_TZ"""
    if isinstance(dt, str):
        dt = pendulum.parse(dt)
    elif isinstance(dt, datetime):
        dt = pendulum.instance(dt)  # convert standard datetime to pendulum
    # normalize to DEFAULT_TZ
    if dt.tzinfo is None:
        dt = DEFAULT_TZ.convert(dt)
    else:
        dt = dt.in_timezone(DEFAULT_TZ)
    return dt


# ------------------------------------------------------------------
# DB connection helper
# ------------------------------------------------------------------
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ------------------------------------------------------------------
# Table creation
# ------------------------------------------------------------------
def create_table():
    print("create_table run")
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id             TEXT PRIMARY KEY,
                table_id             INTEGER CHECK(table_id BETWEEN 1 AND 10),
                guest_name           TEXT NOT NULL,
                guest_phone_number   TEXT NOT NULL,
                total_cost           REAL NOT NULL,
                is_pre_paid          INTEGER NOT NULL CHECK(is_pre_paid IN (0,1)),
                dishes               TEXT NOT NULL,
                notes                TEXT,
                booking_time         TEXT NOT NULL
            )
        """)
        conn.commit()


# ------------------------------------------------------------------
# JSON helper (copes with any stray datetime inside dishes)
# ------------------------------------------------------------------
def _json_dumps(obj) -> str:
    return json.dumps(obj, default=str, ensure_ascii=False)


# ------------------------------------------------------------------
# Create order
# ------------------------------------------------------------------
def create_order(
    guest_name: str,
    guest_phone_number: str,
    total_cost: float,
    is_takeaway: bool,
    dishes: list,
    booking_time: datetime,
    notes: Optional[str] = None
) -> tuple[str, Optional[int]]:
    print("create_order run")
    if not isinstance(booking_time, datetime):
        raise TypeError("booking_time must be a datetime object")

    table_id = None if is_takeaway else assign_table(booking_time)

    order_id = str(uuid.uuid4())[:8]
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO orders (order_id, table_id, guest_name, guest_phone_number,
                                total_cost, is_pre_paid, dishes, notes, booking_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            order_id, table_id, guest_name, guest_phone_number,
            total_cost, 0, _json_dumps(dishes), notes,
            booking_time.isoformat()
        ))
        conn.commit()
    return order_id, table_id


# ------------------------------------------------------------------
# Read order
# ------------------------------------------------------------------
def get_order(order_id: str) -> Optional[Dict]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,)).fetchone()
    if not row:
        return None

    order = dict(row)
    order["dishes"] = json.loads(order["dishes"])
    order["is_pre_paid"] = bool(order["is_pre_paid"])
    order["booking_time"] = datetime.fromisoformat(order["booking_time"])
    return order


# ------------------------------------------------------------------
# Update order
# ------------------------------------------------------------------
def update_order(
    order_id: str,
    *,
    table_id: Optional[int] = None,
    guest_name: Optional[str] = None,
    guest_phone_number: Optional[str] = None,
    total_cost: Optional[float] = None,
    is_pre_paid: Optional[bool] = None,
    dishes: Optional[List[Dict]] = None,
    booking_time: Optional[datetime] = None,
    notes: Optional[str] = None
) -> bool:
    fields, values = [], []

    for col, val in (
        ("table_id", table_id),
        ("guest_name", guest_name),
        ("guest_phone_number", guest_phone_number),
        ("total_cost", total_cost),
        ("is_pre_paid", int(is_pre_paid) if is_pre_paid is not None else None),
        ("dishes", _json_dumps(dishes) if dishes is not None else None),
        ("booking_time", booking_time.isoformat() if booking_time is not None else None),
        ("notes", notes),
    ):
        if val is not None:
            fields.append(f"{col}=?")
            values.append(val)

    if not fields:
        return False

    values.append(order_id)
    sql = f"UPDATE orders SET {', '.join(fields)} WHERE order_id=?"
    with get_connection() as conn:
        cur = conn.execute(sql, values)
        conn.commit()
        return cur.rowcount > 0


# ------------------------------------------------------------------
# Delete order
# ------------------------------------------------------------------
def delete_order(order_id: str) -> bool:
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM orders WHERE order_id=?", (order_id,))
        conn.commit()
        return cur.rowcount > 0


# ------------------------------------------------------------------
# Table assignment  (±45 min overlap guard)
# ------------------------------------------------------------------
def assign_table(booking_time) -> int:
    """
    Return the first free table (1-10) whose orders do NOT overlap
    booking_time ± 45 min. All datetimes are treated in DEFAULT_TZ.
    """
    booking_time = to_aware_dt(booking_time)

    start = booking_time - timedelta(minutes=45)
    end = booking_time + timedelta(minutes=45)

    occupied = set()
    with get_connection() as conn:
        cur = conn.execute(
            "SELECT table_id, booking_time FROM orders "
            "WHERE table_id IS NOT NULL "
            "AND is_pre_paid IN (0,1)"
        )
        for row in cur:
            existing = to_aware_dt(row["booking_time"])
            if start <= existing <= end:
                occupied.add(row["table_id"])

    for t in range(1, 11):
        if t not in occupied:
            return t

    raise ValueError("No table free in the 90-minute window around requested time.")

# ------------------------------------------------------------------
# Quick demo
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Quick demo  (all times in Asia/Ho_Chi_Minh)
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Quick demo  (all times via to_aware_dt -> Asia/Ho_Chi_Minh)
# ------------------------------------------------------------------
if __name__ == "__main__":
    create_table()

    # 1. 18:30 local
    o1, t1 = create_order(
        guest_name="Early bird 18:30",
        guest_phone_number="0900000001",
        total_cost=100_000,
        is_takeaway=False,
        dishes=[{"id": 1, "name": "Cơm cháy", "quantity": 1}],
        booking_time=to_aware_dt("2025-09-14T18:30:00")
    )
    print("18:30 order gets", get_order(o1)["table_id"])   # → 1

    # 2. 18:00 local (30 min earlier → overlap)
    o2, t2 = create_order(
        guest_name="Overlap 18:00",
        guest_phone_number="0900000002",
        total_cost=100_000,
        is_takeaway=False,
        dishes=[{"id": 1, "name": "Cơm cháy", "quantity": 1}],
        booking_time=to_aware_dt("2025-09-14T18:00:00")
    )
    print("18:00 order gets", get_order(o2)["table_id"])   # → 2

    # 3. 17:30 local (60 min earlier → free)
    o3, t3 = create_order(
        guest_name="Safe 17:30",
        guest_phone_number="0900000003",
        total_cost=100_000,
        is_takeaway=False,
        dishes=[{"id": 1, "name": "Cơm cháy", "quantity": 1}],
        booking_time=to_aware_dt("2025-09-14T17:30:00")
    )
    print("17:30 order gets", get_order(o3)["table_id"])   # → 1