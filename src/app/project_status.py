from __future__ import annotations

import argparse

from src.database.db import connect, init_db


TABLES = [
    "students",
    "face_embeddings",
    "attendance_sessions",
    "attendance_records",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Show project database status.")
    parser.add_argument("--db", default="data/attendance/attendance.sqlite3")
    args = parser.parse_args()

    init_db(args.db)
    conn = connect(args.db)
    try:
        print(f"Database: {args.db}")
        for table in TABLES:
            count = conn.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()["count"]
            print(f"{table}: {count}")

        latest = conn.execute(
            """
            SELECT r.timestamp, s.roll_no, s.name, a.name AS session_name
            FROM attendance_records r
            JOIN students s ON s.student_id = r.student_id
            JOIN attendance_sessions a ON a.session_id = r.session_id
            ORDER BY r.timestamp DESC
            LIMIT 5
            """
        ).fetchall()
    finally:
        conn.close()

    if latest:
        print()
        print("Latest attendance:")
        for row in latest:
            print(f"{row['timestamp']} | {row['session_name']} | {row['roll_no']} | {row['name']}")


if __name__ == "__main__":
    main()
