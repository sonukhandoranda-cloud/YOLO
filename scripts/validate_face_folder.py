from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.database.db import connect, init_db
from src.face.engine import create_face_embedder
from src.face.index import FaceIndex


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(folder: Path):
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def read_image_exif_safe(path: Path) -> np.ndarray | None:
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            rgb = np.asarray(image)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return cv2.imread(str(path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate private face test photos against already-enrolled database identities."
    )
    parser.add_argument("--folder", required=True, help="Folder containing private test photos.")
    parser.add_argument("--expected-roll", required=True, help="Expected roll number for these test photos.")
    parser.add_argument("--db", default="data/attendance/attendance.sqlite3")
    parser.add_argument("--threshold", type=float, default=0.41)
    parser.add_argument("--min-margin", type=float, default=0.04)
    parser.add_argument("--output", default="data/processed/private_face_validation.csv")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    init_db(args.db)
    embedder = create_face_embedder()
    conn = connect(args.db)
    try:
        face_index = FaceIndex.load(conn)
    finally:
        conn.close()

    if not face_index.students:
        raise RuntimeError("No enrolled students found in database. Enroll a face first.")

    rows: list[dict[str, object]] = []
    total = 0
    correct = 0
    wrong = 0
    missed = 0
    no_face = 0
    unreadable = 0

    for image_path in iter_images(folder):
        total += 1
        image = read_image_exif_safe(image_path)
        if image is None:
            unreadable += 1
            rows.append(
                {
                    "file": image_path.name,
                    "status": "unreadable",
                    "expected_roll": args.expected_roll,
                    "matched_roll": "",
                    "matched_name": "",
                    "score": "",
                    "margin": "",
                }
            )
            continue

        face = embedder.detect_largest(image)
        if face is None:
            no_face += 1
            rows.append(
                {
                    "file": image_path.name,
                    "status": "no_face_detected",
                    "expected_roll": args.expected_roll,
                    "matched_roll": "",
                    "matched_name": "",
                    "score": "",
                    "margin": "",
                }
            )
            continue

        match = face_index.find_best(face.embedding, args.threshold, args.min_margin)
        if match is None:
            missed += 1
            status = "missed"
            matched_roll = ""
            matched_name = ""
            score = ""
            margin = ""
        elif match.roll_no == args.expected_roll:
            correct += 1
            status = "correct"
            matched_roll = match.roll_no
            matched_name = match.name
            score = f"{match.score:.4f}"
            margin = "" if match.margin is None else f"{match.margin:.4f}"
        else:
            wrong += 1
            status = "wrong_identity"
            matched_roll = match.roll_no
            matched_name = match.name
            score = f"{match.score:.4f}"
            margin = "" if match.margin is None else f"{match.margin:.4f}"

        rows.append(
            {
                "file": image_path.name,
                "status": status,
                "expected_roll": args.expected_roll,
                "matched_roll": matched_roll,
                "matched_name": matched_name,
                "score": score,
                "margin": margin,
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "file",
                "status",
                "expected_roll",
                "matched_roll",
                "matched_name",
                "score",
                "margin",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    attempted = total - unreadable - no_face
    accuracy = correct / max(attempted, 1)
    print("Private face folder validation")
    print(f"Folder: {folder}")
    print("Privacy: images were read for testing only; no images or embeddings were inserted into the database.")
    print(f"Total files: {total}")
    print(f"Attempted matches: {attempted}")
    print(f"Correct: {correct}")
    print(f"Missed: {missed}")
    print(f"Wrong identity: {wrong}")
    print(f"No face detected: {no_face}")
    print(f"Unreadable: {unreadable}")
    print(f"Accuracy on attempted matches: {accuracy:.2%}")
    print(f"CSV: {output_path}")


if __name__ == "__main__":
    main()
