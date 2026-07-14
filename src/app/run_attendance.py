from __future__ import annotations

import argparse
import time

import cv2

from src.cameras.threaded import LatestFrameCamera
from src.database.db import connect, get_or_create_session, init_db, mark_attendance, recently_marked
from src.face.engine import create_face_embedder
from src.face.index import FaceIndex
from src.footwear.async_detector import AsyncFootwearDetector
from src.footwear.detector import FootwearDetector, draw_detections
from src.fusion.decision import IdentityVoteWindow
from src.fusion.footwear_policy import FootwearDecisionWindow, FootwearPolicy


def parse_labels(value: str) -> set[str]:
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run two-camera attendance prototype.")
    parser.add_argument("--db", default="data/attendance/attendance.sqlite3")
    parser.add_argument("--face-camera", type=int, default=0)
    parser.add_argument("--footwear-camera", type=int, default=1)
    parser.add_argument("--shoe-model", default="models/footwear/footwear_policy_yolo11n.pt")
    parser.add_argument("--fallback-shoe-model", default="models/footwear/shoe_detect_best.pt")
    parser.add_argument("--face-threshold", type=float, default=0.41)
    parser.add_argument("--face-min-margin", type=float, default=0.04)
    parser.add_argument("--presence-threshold", type=float, default=0.20)
    parser.add_argument("--allowed-footwear", default="shoe,footwear")
    parser.add_argument("--flagged-footwear", default="sandal")
    parser.add_argument("--session", default="default")
    parser.add_argument("--cooldown-minutes", type=int, default=30)
    parser.add_argument("--vote-window", type=int, default=8)
    parser.add_argument("--vote-required", type=int, default=5)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-footwear-age-ms", type=float, default=1000.0)
    parser.add_argument("--footwear-vote-window", type=int, default=6)
    parser.add_argument("--footwear-vote-required", type=int, default=2)
    parser.add_argument("--footwear-hold-ms", type=float, default=4000.0)
    parser.add_argument("--show-latency", action="store_true", help="Overlay capture/inference latency on both windows.")
    args = parser.parse_args()

    init_db(args.db)
    face_engine = create_face_embedder()
    footwear_detector = FootwearDetector(
        args.shoe_model,
        args.presence_threshold,
        fallback_model_path=args.fallback_shoe_model,
    )
    async_footwear = AsyncFootwearDetector(footwear_detector).start()
    footwear_policy = FootwearPolicy(
        allowed_labels=parse_labels(args.allowed_footwear),
        flagged_labels=parse_labels(args.flagged_footwear),
    )
    footwear_votes = FootwearDecisionWindow(
        window_size=args.footwear_vote_window,
        required_votes=args.footwear_vote_required,
        hold_ms=args.footwear_hold_ms,
    )
    votes = IdentityVoteWindow(args.vote_window, args.vote_required)

    face_cap = LatestFrameCamera(args.face_camera, width=args.width, height=args.height, fps=args.fps).start()
    shoe_cap = LatestFrameCamera(args.footwear_camera, width=args.width, height=args.height, fps=args.fps).start()

    print("Running attendance. Press ESC to quit.")
    print(f"Face camera={args.face_camera}, footwear camera={args.footwear_camera}, size={args.width}x{args.height}@{args.fps}")
    conn = connect(args.db)
    try:
        session_id = get_or_create_session(conn, args.session)
        conn.commit()
        face_index = FaceIndex.load(conn)
        if not face_index.students:
            print("No enrolled students found. Enroll a student first.")
        last_footwear_sequence = -1
        last_absent_update = 0.0
        footwear_decision = footwear_policy.decide([])
        while True:
            face_sample = face_cap.read_latest()
            shoe_sample = shoe_cap.read_latest()
            if face_sample is None or shoe_sample is None:
                time.sleep(0.01)
                continue

            face_frame = face_sample.frame
            shoe_frame = shoe_sample.frame
            async_footwear.submit(shoe_frame, shoe_sample.timestamp, shoe_sample.sequence)

            face_start = time.perf_counter()
            face = face_engine.detect_largest(face_frame)
            face_ms = (time.perf_counter() - face_start) * 1000
            match = face_index.find_best(face.embedding, args.face_threshold, args.face_min_margin) if face else None
            stable = votes.add(match.student_id if match else None)

            footwear_result = async_footwear.latest()
            footwear_is_fresh = bool(footwear_result and footwear_result.age_ms <= args.max_footwear_age_ms)
            detections = footwear_result.detections if footwear_result and footwear_is_fresh else []
            now = time.perf_counter()
            if footwear_result and footwear_is_fresh and footwear_result.sequence != last_footwear_sequence:
                current_footwear_decision = footwear_policy.decide(detections)
                footwear_decision = footwear_votes.add(current_footwear_decision)
                last_footwear_sequence = footwear_result.sequence
                last_absent_update = now
            elif not footwear_is_fresh and now - last_absent_update >= 0.25:
                footwear_decision = footwear_votes.add(footwear_policy.decide([]))
                last_absent_update = now

            if stable and match and stable.student_id == match.student_id and footwear_decision.can_mark_attendance:
                if not recently_marked(
                    conn,
                    student_id=match.student_id,
                    cooldown_minutes=args.cooldown_minutes,
                    session_id=session_id,
                ):
                    inserted = mark_attendance(
                        conn,
                        student_id=match.student_id,
                        session_id=session_id,
                        face_score=match.score,
                        presence_score=footwear_decision.confidence,
                        presence_label=footwear_decision.label,
                        decision_status=footwear_decision.status,
                    )
                    conn.commit()
                    if inserted:
                        print(
                            f"Marked {footwear_decision.status}: {match.roll_no} {match.name} "
                            f"face={match.score:.2f} footwear={footwear_decision.label} "
                            f"presence={footwear_decision.confidence:.2f}"
                        )

            if face:
                x, y, w, h = face.bbox
                cv2.rectangle(face_frame, (x, y), (x + w, y + h), (0, 220, 0), 2)
            if match:
                cv2.putText(face_frame, f"{match.roll_no} {match.name} {match.score:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
            else:
                cv2.putText(face_frame, "No confident face match", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if args.show_latency:
                face_age_ms = (time.perf_counter() - face_sample.timestamp) * 1000
                cv2.putText(face_frame, f"face {face_ms:.0f}ms | age {face_age_ms:.0f}ms", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            shoe_view = draw_detections(shoe_frame, detections)
            cv2.putText(
                shoe_view,
                footwear_decision.message,
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 220, 0) if footwear_decision.can_mark_attendance else (0, 0, 255),
                2,
            )
            if args.show_latency and footwear_result:
                cv2.putText(
                    shoe_view,
                    f"shoe {footwear_result.inference_ms:.0f}ms | age {footwear_result.age_ms:.0f}ms",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0) if footwear_is_fresh else (0, 0, 255),
                    2,
                )
            cv2.imshow("Face Recognition", face_frame)
            cv2.imshow("Footwear Presence", shoe_view)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break
    finally:
        conn.close()
        async_footwear.stop()
        face_cap.stop()
        shoe_cap.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
