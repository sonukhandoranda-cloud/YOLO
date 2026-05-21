from __future__ import annotations

import argparse
import csv
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.database.db import add_face_embedding, connect, init_db, upsert_student
from src.face.engine import create_face_embedder
from src.face.index import FaceIndex


@dataclass(frozen=True)
class IdentityPair:
    name: str
    enroll_image: np.ndarray
    test_image: np.ndarray


def to_bgr(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        if float(np.max(image)) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def upscale_for_detector(image_bgr: np.ndarray, min_height: int = 360) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    if height >= min_height:
        return image_bgr
    scale = min_height / max(height, 1)
    return cv2.resize(image_bgr, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_CUBIC)


def load_lfw_pairs(
    *,
    data_home: Path,
    enrolled_count: int,
    unknown_count: int,
) -> tuple[list[IdentityPair], list[IdentityPair]]:
    from sklearn.datasets import fetch_lfw_people

    dataset = fetch_lfw_people(
        data_home=str(data_home),
        min_faces_per_person=2,
        color=True,
        resize=1.0,
    )

    grouped: dict[int, list[int]] = {}
    for index, target in enumerate(dataset.target):
        grouped.setdefault(int(target), []).append(index)

    pairs: list[IdentityPair] = []
    for target, indexes in grouped.items():
        if len(indexes) < 2:
            continue
        name = str(dataset.target_names[target]).replace(" ", "_")
        enroll = upscale_for_detector(to_bgr(dataset.images[indexes[0]]))
        test = upscale_for_detector(to_bgr(dataset.images[indexes[1]]))
        pairs.append(IdentityPair(name=name, enroll_image=enroll, test_image=test))

    pairs = sorted(pairs, key=lambda pair: pair.name)
    enrolled = pairs[:enrolled_count]
    unknown = pairs[enrolled_count : enrolled_count + unknown_count]
    return enrolled, unknown


def save_pair_images(pairs: list[IdentityPair], output_dir: Path, limit: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, pair in enumerate(pairs[:limit], start=1):
        identity_dir = output_dir / f"{index:04d}_{pair.name}"
        identity_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(identity_dir / "enroll.jpg"), pair.enroll_image)
        cv2.imwrite(str(identity_dir / "test.jpg"), pair.test_image)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate face recognition on a public LFW benchmark subset.")
    parser.add_argument("--enrolled", type=int, default=1500)
    parser.add_argument("--unknown", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.42)
    parser.add_argument("--min-margin", type=float, default=0.03)
    parser.add_argument("--data-home", default="data/raw/public_face_benchmark")
    parser.add_argument("--export-images", default="data/processed/public_face_pairs")
    parser.add_argument("--export-limit", type=int, default=1500)
    parser.add_argument("--csv", default="data/processed/public_face_benchmark_results.csv")
    parser.add_argument("--report", default="docs/PUBLIC_FACE_BENCHMARK_REPORT.md")
    args = parser.parse_args()

    data_home = Path(args.data_home)
    csv_path = Path(args.csv)
    report_path = Path(args.report)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    print("Downloading/loading public LFW face benchmark...")
    enrolled_pairs, unknown_pairs = load_lfw_pairs(
        data_home=data_home,
        enrolled_count=args.enrolled,
        unknown_count=args.unknown,
    )
    print(f"Available enrolled identities: {len(enrolled_pairs)}")
    print(f"Available unknown identities: {len(unknown_pairs)}")

    if len(enrolled_pairs) < args.enrolled:
        print(f"Warning: requested {args.enrolled} enrolled identities but dataset provided {len(enrolled_pairs)}.")

    print("Saving benchmark image pairs for inspection...")
    save_pair_images(enrolled_pairs, Path(args.export_images), args.export_limit)

    embedder = create_face_embedder()
    enrollment_failures: list[str] = []
    genuine_rows: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "public_face_benchmark.sqlite3"
        init_db(db_path)
        conn = connect(db_path)
        try:
            print("Enrolling public benchmark identities...")
            for index, pair in enumerate(enrolled_pairs, start=1):
                face = embedder.detect_largest(pair.enroll_image)
                if face is None:
                    enrollment_failures.append(pair.name)
                    continue
                student_id = upsert_student(
                    conn,
                    roll_no=f"LFW{index:04d}",
                    name=pair.name,
                    branch="PUBLIC_BENCHMARK",
                    batch="LFW",
                )
                add_face_embedding(
                    conn,
                    student_id=student_id,
                    vector=face.embedding,
                    image_path=None,
                    quality_score=face.quality_score,
                )
            conn.commit()

            face_index = FaceIndex.load(conn)
        finally:
            conn.close()

        print("Testing genuine matches...")
        genuine_correct = 0
        genuine_missed = 0
        genuine_wrong = 0
        query_failures = 0
        match_times_ms: list[float] = []

        for index, pair in enumerate(enrolled_pairs, start=1):
            if pair.name in enrollment_failures:
                genuine_rows.append(
                    {
                        "type": "genuine",
                        "expected": pair.name,
                        "matched": "",
                        "score": "",
                        "margin": "",
                        "status": "enrollment_failed",
                    }
                )
                continue

            face = embedder.detect_largest(pair.test_image)
            if face is None:
                query_failures += 1
                genuine_rows.append(
                    {
                        "type": "genuine",
                        "expected": pair.name,
                        "matched": "",
                        "score": "",
                        "margin": "",
                        "status": "query_face_not_detected",
                    }
                )
                continue

            start = time.perf_counter()
            match = face_index.find_best(face.embedding, args.threshold, args.min_margin)
            match_times_ms.append((time.perf_counter() - start) * 1000.0)
            if match is None:
                genuine_missed += 1
                status = "missed"
                matched = ""
                score = ""
                margin = ""
            elif match.name == pair.name:
                genuine_correct += 1
                status = "correct"
                matched = match.name
                score = f"{match.score:.4f}"
                margin = "" if match.margin is None else f"{match.margin:.4f}"
            else:
                genuine_wrong += 1
                status = "wrong_identity"
                matched = match.name
                score = f"{match.score:.4f}"
                margin = "" if match.margin is None else f"{match.margin:.4f}"

            genuine_rows.append(
                {
                    "type": "genuine",
                    "expected": pair.name,
                    "matched": matched,
                    "score": score,
                    "margin": margin,
                    "status": status,
                }
            )

        print("Testing unknown-person rejection...")
        unknown_false_accepts = 0
        unknown_rejected = 0
        unknown_detection_failures = 0
        for pair in unknown_pairs:
            face = embedder.detect_largest(pair.test_image)
            if face is None:
                unknown_detection_failures += 1
                continue
            start = time.perf_counter()
            match = face_index.find_best(face.embedding, args.threshold, args.min_margin)
            match_times_ms.append((time.perf_counter() - start) * 1000.0)
            if match is None:
                unknown_rejected += 1
                status = "rejected"
                matched = ""
                score = ""
                margin = ""
            else:
                unknown_false_accepts += 1
                status = "false_accept"
                matched = match.name
                score = f"{match.score:.4f}"
                margin = "" if match.margin is None else f"{match.margin:.4f}"
            genuine_rows.append(
                {
                    "type": "unknown",
                    "expected": pair.name,
                    "matched": matched,
                    "score": score,
                    "margin": margin,
                    "status": status,
                }
            )

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["type", "expected", "matched", "score", "margin", "status"])
        writer.writeheader()
        writer.writerows(genuine_rows)

    enrolled_ok = len(enrolled_pairs) - len(enrollment_failures)
    genuine_attempted = enrolled_ok - query_failures
    genuine_accuracy = genuine_correct / max(genuine_attempted, 1)
    false_accept_rate = unknown_false_accepts / max((unknown_rejected + unknown_false_accepts), 1)
    avg_match_ms = float(np.mean(match_times_ms)) if match_times_ms else 0.0
    p95_match_ms = float(np.percentile(match_times_ms, 95)) if match_times_ms else 0.0

    report = f"""# Public Face Benchmark Report

## Dataset

Public LFW benchmark loaded through scikit-learn. Each identity is treated as a simulated student.

This is not a real college-student dataset. It is used only to test the current face-recognition pipeline before real enrollment data exists.

## Setup

```text
Requested enrolled identities: {args.enrolled}
Loaded enrolled identities: {len(enrolled_pairs)}
Unknown identities reserved: {len(unknown_pairs)}
Threshold: {args.threshold}
Minimum top-2 margin: {args.min_margin}
```

## Result

```text
Enrollment success: {enrolled_ok}/{len(enrolled_pairs)}
Enrollment failures: {len(enrollment_failures)}
Genuine query face-detection failures: {query_failures}
Genuine attempted matches: {genuine_attempted}
Genuine correct: {genuine_correct}
Genuine missed: {genuine_missed}
Genuine wrong identity: {genuine_wrong}
Measured genuine accuracy: {genuine_accuracy:.2%}

Unknown tested: {unknown_rejected + unknown_false_accepts}
Unknown rejected: {unknown_rejected}
Unknown false accepts: {unknown_false_accepts}
Unknown detection failures: {unknown_detection_failures}
Measured false accept rate: {false_accept_rate:.2%}

Average in-memory match time: {avg_match_ms:.4f} ms
P95 in-memory match time: {p95_match_ms:.4f} ms
```

## Output Files

```text
{csv_path}
{args.export_images}
```

## Interpretation

This benchmark tests the current face-recognition pipeline with public face images. It is useful, but it is not the final accuracy claim for the college deployment.

Final accuracy must still be measured using real enrollment-style images and real camera-condition images.
"""
    report_path.write_text(report, encoding="utf-8")

    print()
    print("Public face benchmark result")
    print(f"Enrollment success: {enrolled_ok}/{len(enrolled_pairs)}")
    print(f"Genuine attempted matches: {genuine_attempted}")
    print(f"Genuine correct: {genuine_correct}")
    print(f"Genuine missed: {genuine_missed}")
    print(f"Genuine wrong identity: {genuine_wrong}")
    print(f"Measured genuine accuracy: {genuine_accuracy:.2%}")
    print(f"Unknown tested: {unknown_rejected + unknown_false_accepts}")
    print(f"Unknown false accepts: {unknown_false_accepts}")
    print(f"Measured false accept rate: {false_accept_rate:.2%}")
    print(f"Average in-memory match time: {avg_match_ms:.4f} ms")
    print(f"P95 in-memory match time: {p95_match_ms:.4f} ms")
    print(f"CSV: {csv_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
