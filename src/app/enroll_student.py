from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.database.db import add_face_embedding, connect, init_db, upsert_student
from src.face.engine import create_face_embedder
from src.face.index import FaceIndex


def read_image_or_camera(image_path: str | None, camera_index: int) -> tuple[object, str | None]:
    if image_path:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return image, image_path

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")
    print("Press SPACE to capture enrollment image, or ESC to cancel.")
    captured = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imshow("Enroll Student", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == 32:
            captured = frame.copy()
            break
    cap.release()
    cv2.destroyAllWindows()
    if captured is None:
        raise RuntimeError("Enrollment cancelled or camera frame unavailable.")

    output_dir = Path("data/raw/enrollment")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "latest_capture.jpg"
    cv2.imwrite(str(output_path), captured)
    return captured, str(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Enroll a student face into the attendance database.")
    parser.add_argument("--roll-no", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--image", help="Path to enrollment image. If omitted, webcam capture is used.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--db", default="data/attendance/attendance.sqlite3")
    parser.add_argument("--duplicate-threshold", type=float, default=0.41)
    parser.add_argument("--duplicate-min-margin", type=float, default=0.04)
    parser.add_argument("--force", action="store_true", help="Allow saving even if this face already matches an enrolled person.")
    args = parser.parse_args()

    init_db(args.db)
    image, stored_image_path = read_image_or_camera(args.image, args.camera)
    embedder = create_face_embedder()
    face = embedder.detect_largest(image)
    if face is None:
        raise RuntimeError("No usable face detected. Try a clearer front-facing image.")

    conn = connect(args.db)
    try:
        face_index = FaceIndex.load(conn)
        existing_match = face_index.find_best(face.embedding, args.duplicate_threshold, args.duplicate_min_margin)
        if existing_match is not None and not args.force:
            print(
                "Already enrolled: "
                f"{existing_match.roll_no} - {existing_match.name} "
                f"(match score {existing_match.score:.2f}). "
                "Enrollment was not saved."
            )
            return

        student_id = upsert_student(
            conn,
            roll_no=args.roll_no,
            name=args.name,
            branch=args.branch,
            batch=args.batch,
        )
        add_face_embedding(
            conn,
            student_id=student_id,
            vector=face.embedding,
            image_path=stored_image_path,
            quality_score=face.quality_score,
        )
        conn.commit()
    finally:
        conn.close()

    print(f"Enrolled {args.roll_no} - {args.name} with quality {face.quality_score:.2f}")


if __name__ == "__main__":
    main()
