from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np


SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS students (
    student_id INTEGER PRIMARY KEY AUTOINCREMENT,
    roll_no TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    branch TEXT NOT NULL,
    batch TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS face_embeddings (
    embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL,
    vector_json TEXT NOT NULL,
    image_path TEXT,
    quality_score REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (student_id) REFERENCES students(student_id)
);

CREATE TABLE IF NOT EXISTS attendance_sessions (
    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    session_date TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(name, session_date)
);

CREATE TABLE IF NOT EXISTS attendance_records (
    record_id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL,
    session_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    face_score REAL NOT NULL,
    presence_score REAL NOT NULL,
    decision_status TEXT NOT NULL,
    FOREIGN KEY (student_id) REFERENCES students(student_id),
    FOREIGN KEY (session_id) REFERENCES attendance_sessions(session_id),
    UNIQUE(student_id, session_id)
);
"""


@dataclass(frozen=True)
class StudentMatch:
    student_id: int
    roll_no: str
    name: str
    branch: str
    batch: str
    score: float
    second_score: float | None = None
    margin: float | None = None


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | Path) -> None:
    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def upsert_student(
    conn: sqlite3.Connection,
    *,
    roll_no: str,
    name: str,
    branch: str,
    batch: str,
) -> int:
    now = datetime.now().isoformat(timespec="seconds")
    conn.execute(
        """
        INSERT INTO students (roll_no, name, branch, batch, created_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(roll_no) DO UPDATE SET
            name = excluded.name,
            branch = excluded.branch,
            batch = excluded.batch,
            status = 'active'
        """,
        (roll_no, name, branch, batch, now),
    )
    row = conn.execute("SELECT student_id FROM students WHERE roll_no = ?", (roll_no,)).fetchone()
    return int(row["student_id"])


def add_face_embedding(
    conn: sqlite3.Connection,
    *,
    student_id: int,
    vector: np.ndarray,
    image_path: str | None,
    quality_score: float,
) -> None:
    conn.execute(
        """
        INSERT INTO face_embeddings
            (student_id, vector_json, image_path, quality_score, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            student_id,
            json.dumps(vector.astype(float).tolist()),
            image_path,
            float(quality_score),
            datetime.now().isoformat(timespec="seconds"),
        ),
    )


def iter_embeddings(conn: sqlite3.Connection) -> Iterable[tuple[sqlite3.Row, np.ndarray]]:
    rows = conn.execute(
        """
        SELECT s.student_id, s.roll_no, s.name, s.branch, s.batch, e.vector_json
        FROM face_embeddings e
        JOIN students s ON s.student_id = e.student_id
        WHERE s.status = 'active'
        """
    ).fetchall()
    for row in rows:
        yield row, np.asarray(json.loads(row["vector_json"]), dtype=np.float32)


def get_or_create_session(conn: sqlite3.Connection, name: str) -> int:
    session_date = datetime.now().date().isoformat()
    now = datetime.now().isoformat(timespec="seconds")
    conn.execute(
        """
        INSERT OR IGNORE INTO attendance_sessions (name, session_date, created_at)
        VALUES (?, ?, ?)
        """,
        (name, session_date, now),
    )
    row = conn.execute(
        "SELECT session_id FROM attendance_sessions WHERE name = ? AND session_date = ?",
        (name, session_date),
    ).fetchone()
    return int(row["session_id"])


def recently_marked(
    conn: sqlite3.Connection,
    *,
    student_id: int,
    cooldown_minutes: int,
) -> bool:
    since = datetime.now() - timedelta(minutes=cooldown_minutes)
    row = conn.execute(
        """
        SELECT 1
        FROM attendance_records
        WHERE student_id = ? AND timestamp >= ?
        LIMIT 1
        """,
        (student_id, since.isoformat(timespec="seconds")),
    ).fetchone()
    return row is not None


def mark_attendance(
    conn: sqlite3.Connection,
    *,
    student_id: int,
    session_id: int,
    face_score: float,
    presence_score: float,
) -> bool:
    try:
        conn.execute(
            """
            INSERT INTO attendance_records
                (student_id, session_id, timestamp, face_score, presence_score, decision_status)
            VALUES (?, ?, ?, ?, ?, 'present')
            """,
            (
                student_id,
                session_id,
                datetime.now().isoformat(timespec="seconds"),
                float(face_score),
                float(presence_score),
            ),
        )
        return True
    except sqlite3.IntegrityError:
        return False
