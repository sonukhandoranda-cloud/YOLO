from __future__ import annotations

import argparse

from src.database.db import init_db


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize the attendance SQLite database.")
    parser.add_argument("--db", default="data/attendance/attendance.sqlite3")
    args = parser.parse_args()
    init_db(args.db)
    print(f"Database ready: {args.db}")


if __name__ == "__main__":
    main()
