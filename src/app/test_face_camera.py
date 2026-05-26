from __future__ import annotations

import argparse

import cv2

from src.database.db import (
    connect,
    get_or_create_session,
    init_db,
    mark_attendance,
    recently_marked,
)
from src.face.engine import create_face_embedder
from src.face.index import FaceIndex
from src.fusion.decision import IdentityVoteWindow


def main() -> None:
    parser = argparse.ArgumentParser(description="Test face detection and recognition with one PC camera.")
    parser.add_argument("--db", default="data/attendance/attendance.sqlite3")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.41)
    parser.add_argument("--min-margin", type=float, default=0.04)
    parser.add_argument("--session", default="face-demo")
    parser.add_argument("--cooldown-minutes", type=int, default=30)
    parser.add_argument("--vote-window", type=int, default=8)
    parser.add_argument("--vote-required", type=int, default=5)
    args = parser.parse_args()

    init_db(args.db)
    embedder = create_face_embedder()
    votes = IdentityVoteWindow(args.vote_window, args.vote_required)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")

    print("Face attendance test running. Press ESC to quit.")
    conn = connect(args.db)
    last_status = "Waiting for a stable face match"
    try:
        session_id = get_or_create_session(conn, args.session)
        conn.commit()
        face_index = FaceIndex.load(conn)
        if not face_index.students:
            print("No enrolled students found. Enroll a student first.")
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            face = embedder.detect_largest(frame)
            match = face_index.find_best(face.embedding, args.threshold, args.min_margin) if face else None
            stable = votes.add(match.student_id if match else None)

            if face:
                x, y, w, h = face.bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), 2)

            if stable and match and stable.student_id == match.student_id:
                if recently_marked(
                    conn,
                    student_id=match.student_id,
                    cooldown_minutes=args.cooldown_minutes,
                ):
                    last_status = f"ALREADY MARKED: {match.roll_no} {match.name}"
                else:
                    inserted = mark_attendance(
                        conn,
                        student_id=match.student_id,
                        session_id=session_id,
                        face_score=match.score,
                        presence_score=0.0,
                    )
                    conn.commit()
                    if inserted:
                        last_status = f"ATTENDANCE MARKED: {match.roll_no} {match.name}"
                        print(last_status)
                    else:
                        last_status = f"ALREADY MARKED: {match.roll_no} {match.name}"

            if match:
                text = f"{match.roll_no} {match.name} {match.score:.2f}"
                color = (0, 220, 0)
            elif face:
                text = "Face detected, no enrolled match"
                color = (0, 180, 255)
            else:
                text = "No face detected"
                color = (0, 0, 255)

            cv2.putText(frame, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, last_status, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow("Face Recognition Test", frame)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break
    finally:
        conn.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
