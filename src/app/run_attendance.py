from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.database.db import connect, get_or_create_session, init_db, mark_attendance, recently_marked
from src.face.engine import create_face_embedder
from src.face.index import FaceIndex
from src.footwear.detector import FootwearDetector, draw_detections
from src.fusion.decision import IdentityVoteWindow


def main() -> None:
    parser = argparse.ArgumentParser(description="Run two-camera attendance prototype.")
    parser.add_argument("--db", default="data/attendance/attendance.sqlite3")
    parser.add_argument("--face-camera", type=int, default=0)
    parser.add_argument("--footwear-camera", type=int, default=1)
    parser.add_argument("--shoe-model", default="models/footwear/shoe_detect_best.pt")
    parser.add_argument("--face-threshold", type=float, default=0.41)
    parser.add_argument("--face-min-margin", type=float, default=0.04)
    parser.add_argument("--presence-threshold", type=float, default=0.50)
    parser.add_argument("--session", default="default")
    parser.add_argument("--cooldown-minutes", type=int, default=30)
    parser.add_argument("--vote-window", type=int, default=8)
    parser.add_argument("--vote-required", type=int, default=5)
    args = parser.parse_args()

    init_db(args.db)
    face_engine = create_face_embedder()
    footwear_detector = FootwearDetector(args.shoe_model, args.presence_threshold)
    votes = IdentityVoteWindow(args.vote_window, args.vote_required)

    face_cap = cv2.VideoCapture(args.face_camera)
    shoe_cap = cv2.VideoCapture(args.footwear_camera)
    if not face_cap.isOpened():
        raise RuntimeError(f"Could not open face camera {args.face_camera}")
    if not shoe_cap.isOpened():
        raise RuntimeError(f"Could not open footwear camera {args.footwear_camera}")

    print("Running attendance. Press ESC to quit.")
    conn = connect(args.db)
    try:
        session_id = get_or_create_session(conn, args.session)
        conn.commit()
        face_index = FaceIndex.load(conn)
        if not face_index.students:
            print("No enrolled students found. Enroll a student first.")
        while True:
            face_ok, face_frame = face_cap.read()
            shoe_ok, shoe_frame = shoe_cap.read()
            if not face_ok or not shoe_ok:
                print("Camera frame unavailable.")
                break

            face = face_engine.detect_largest(face_frame)
            match = face_index.find_best(face.embedding, args.face_threshold, args.face_min_margin) if face else None
            stable = votes.add(match.student_id if match else None)

            detections = footwear_detector.detect(shoe_frame)
            presence_score = max((d.confidence for d in detections), default=0.0)
            presence_ok = presence_score >= args.presence_threshold

            if stable and match and stable.student_id == match.student_id and presence_ok:
                if not recently_marked(conn, student_id=match.student_id, cooldown_minutes=args.cooldown_minutes):
                    inserted = mark_attendance(
                        conn,
                        student_id=match.student_id,
                        session_id=session_id,
                        face_score=match.score,
                        presence_score=presence_score,
                    )
                    conn.commit()
                    if inserted:
                        print(f"Marked present: {match.roll_no} {match.name} face={match.score:.2f} presence={presence_score:.2f}")

            if face:
                x, y, w, h = face.bbox
                cv2.rectangle(face_frame, (x, y), (x + w, y + h), (0, 220, 0), 2)
            if match:
                cv2.putText(face_frame, f"{match.roll_no} {match.name} {match.score:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
            else:
                cv2.putText(face_frame, "No confident face match", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            shoe_view = draw_detections(shoe_frame, detections)
            cv2.imshow("Face Recognition", face_frame)
            cv2.imshow("Footwear Presence", shoe_view)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break
    finally:
        conn.close()

    face_cap.release()
    shoe_cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
