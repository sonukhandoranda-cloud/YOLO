from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_face_folder import IMAGE_EXTENSIONS, read_image_exif_safe
from src.database.db import (
    connect,
    get_or_create_session,
    init_db,
    mark_attendance,
    recently_marked,
)
from src.face.engine import create_face_embedder
from src.face.index import FaceIndex


def iter_images(folder: Path):
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def count_attendance(conn, student_id: int, session_id: int) -> int:
    row = conn.execute(
        """
        SELECT COUNT(*) AS count
        FROM attendance_records
        WHERE student_id = ? AND session_id = ?
        """,
        (student_id, session_id),
    ).fetchone()
    return int(row["count"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate duplicate attendance prevention using a folder of private face test images."
    )
    parser.add_argument("--folder", required=True)
    parser.add_argument("--expected-roll", required=True)
    parser.add_argument("--db", default="data/attendance/attendance.sqlite3")
    parser.add_argument("--session", default="face-demo")
    parser.add_argument("--threshold", type=float, default=0.41)
    parser.add_argument("--min-margin", type=float, default=0.04)
    parser.add_argument("--cooldown-minutes", type=int, default=30)
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    init_db(args.db)
    embedder = create_face_embedder()
    conn = connect(args.db)
    try:
        face_index = FaceIndex.load(conn)
        expected_student = next(
            (student for student in face_index.students if str(student["roll_no"]) == args.expected_roll),
            None,
        )
        if expected_student is None:
            raise RuntimeError(f"Expected roll number is not enrolled: {args.expected_roll}")

        session_id = get_or_create_session(conn, args.session)
        conn.commit()
        student_id = int(expected_student["student_id"])
        before_count = count_attendance(conn, student_id, session_id)

        recognized = 0
        already_marked = 0
        duplicate_blocked = 0
        inserted = 0
        rejected = 0
        no_face = 0

        for image_path in iter_images(folder):
            image = read_image_exif_safe(image_path)
            if image is None:
                rejected += 1
                continue

            face = embedder.detect_largest(image)
            if face is None:
                no_face += 1
                continue

            match = face_index.find_best(face.embedding, args.threshold, args.min_margin)
            if match is None or match.roll_no != args.expected_roll:
                rejected += 1
                continue

            recognized += 1
            if recently_marked(conn, student_id=match.student_id, cooldown_minutes=args.cooldown_minutes):
                already_marked += 1
                continue

            if mark_attendance(
                conn,
                student_id=match.student_id,
                session_id=session_id,
                face_score=match.score,
                presence_score=0.0,
            ):
                inserted += 1
            else:
                duplicate_blocked += 1
            conn.commit()

        after_count = count_attendance(conn, student_id, session_id)
    finally:
        conn.close()

    print("Duplicate attendance validation")
    print(f"Folder: {folder}")
    print("Privacy: images were read only; no images or embeddings were inserted into the database.")
    print(f"Session: {args.session}")
    print(f"Expected roll: {args.expected_roll}")
    print(f"Attendance count before: {before_count}")
    print(f"Attendance count after: {after_count}")
    print(f"Recognized images: {recognized}")
    print(f"Cooldown already-marked decisions: {already_marked}")
    print(f"Session duplicate blocks: {duplicate_blocked}")
    print(f"New inserts: {inserted}")
    print(f"No face: {no_face}")
    print(f"Rejected/mismatched/unreadable: {rejected}")
    if after_count == before_count and inserted == 0:
        print("Verdict: PASS - no duplicate attendance entries were created.")
    else:
        print("Verdict: CHECK - attendance count changed.")


if __name__ == "__main__":
    main()
