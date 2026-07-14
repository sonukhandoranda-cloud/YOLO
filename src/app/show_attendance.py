from __future__ import annotations

import argparse

from src.database.db import connect, effective_attendance_status, init_db


def main() -> None:
    parser = argparse.ArgumentParser(description="Show saved attendance records.")
    parser.add_argument("--db", default="data/attendance/attendance.sqlite3")
    args = parser.parse_args()

    init_db(args.db)
    conn = connect(args.db)
    try:
        rows = conn.execute(
            """
            SELECT
                r.record_id,
                r.timestamp,
                s.roll_no,
                s.name,
                s.branch,
                s.batch,
                a.name AS session_name,
                r.face_score,
                r.presence_score,
                r.presence_label,
                r.decision_status
            FROM attendance_records r
            JOIN students s ON s.student_id = r.student_id
            JOIN attendance_sessions a ON a.session_id = r.session_id
            ORDER BY r.timestamp DESC
            """
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        print("No attendance records yet.")
        return

    for row in rows:
        status = effective_attendance_status(
            row["decision_status"],
            row["presence_label"],
            float(row["presence_score"]),
        )
        print(
            f"{row['timestamp']} | {row['session_name']} | "
            f"{row['roll_no']} | {row['name']} | {row['branch']} | {row['batch']} | "
            f"face={row['face_score']:.2f} | footwear={row['presence_label']} | "
            f"presence={row['presence_score']:.2f} | "
            f"{status}"
        )


if __name__ == "__main__":
    main()
