from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.database.db import add_face_embedding, connect, init_db, upsert_student
from src.face.index import FaceIndex


def unit_vector(rng: np.random.Generator, size: int) -> np.ndarray:
    vector = rng.normal(size=size).astype(np.float32)
    return vector / max(float(np.linalg.norm(vector)), 1e-6)


def noisy_copy(rng: np.random.Generator, vector: np.ndarray, noise: float) -> np.ndarray:
    result = vector + rng.normal(scale=noise, size=vector.shape).astype(np.float32)
    return result / max(float(np.linalg.norm(result)), 1e-6)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def build_database(
    *,
    db_path: Path,
    students: int,
    embedding_size: int,
    rng: np.random.Generator,
) -> list[tuple[str, np.ndarray]]:
    vectors: list[tuple[str, np.ndarray]] = []
    init_db(db_path)
    conn = connect(db_path)
    try:
        for index in range(students):
            roll_no = f"SIM{index + 1:04d}"
            vector = unit_vector(rng, embedding_size)
            student_id = upsert_student(
                conn,
                roll_no=roll_no,
                name=f"Simulated Student {index + 1}",
                branch="SIM",
                batch="2026",
            )
            add_face_embedding(
                conn,
                student_id=student_id,
                vector=vector,
                image_path=None,
                quality_score=1.0,
            )
            vectors.append((roll_no, vector))
        conn.commit()
    finally:
        conn.close()
    return vectors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate 1500+ student matching readiness.")
    parser.add_argument("--students", type=int, default=1500)
    parser.add_argument("--embedding-size", type=int, default=128)
    parser.add_argument("--genuine-queries", type=int, default=300)
    parser.add_argument("--impostor-queries", type=int, default=300)
    parser.add_argument("--ambiguous-pairs", type=int, default=30)
    parser.add_argument("--noise", type=float, default=0.03)
    parser.add_argument("--threshold", type=float, default=0.72)
    parser.add_argument("--min-margin", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--db", help="Optional database path. Uses a temp database if omitted.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if args.db:
        db_path = Path(args.db)
    else:
        temp_dir = tempfile.TemporaryDirectory()
        db_path = Path(temp_dir.name) / "scale_readiness.sqlite3"

    try:
        print(f"Building simulated database with {args.students} students...")
        build_start = time.perf_counter()
        vectors = build_database(
            db_path=db_path,
            students=args.students,
            embedding_size=args.embedding_size,
            rng=rng,
        )
        build_seconds = time.perf_counter() - build_start

        conn = connect(db_path)
        try:
            index_start = time.perf_counter()
            face_index = FaceIndex.load(conn)
            index_seconds = time.perf_counter() - index_start
        finally:
            conn.close()

        genuine_correct = 0
        genuine_missed = 0
        genuine_times: list[float] = []
        for _ in range(args.genuine_queries):
            expected_roll, vector = vectors[int(rng.integers(0, len(vectors)))]
            query = noisy_copy(rng, vector, args.noise)
            start = time.perf_counter()
            match = face_index.find_best(query, args.threshold, args.min_margin)
            genuine_times.append((time.perf_counter() - start) * 1000.0)
            if match is None:
                genuine_missed += 1
            elif match.roll_no == expected_roll:
                genuine_correct += 1

        impostor_false_accepts = 0
        impostor_times: list[float] = []
        for _ in range(args.impostor_queries):
            query = unit_vector(rng, args.embedding_size)
            start = time.perf_counter()
            match = face_index.find_best(query, args.threshold, args.min_margin)
            impostor_times.append((time.perf_counter() - start) * 1000.0)
            if match is not None:
                impostor_false_accepts += 1

        ambiguous_rejected = 0
        ambiguous_total = min(args.ambiguous_pairs, len(vectors) - 1)
        for index in range(ambiguous_total):
            _, base = vectors[index]
            # Insert a near-duplicate vector into the in-memory index to simulate a lookalike/confusing identity.
            twin = noisy_copy(rng, base, noise=0.005)
            original_matrix = face_index.matrix
            original_students = face_index.students
            face_index.matrix = np.vstack([face_index.matrix, twin])
            face_index.students = [
                *face_index.students,
                {
                    "student_id": 10_000 + index,
                    "roll_no": f"TWIN{index + 1:04d}",
                    "name": "Ambiguous Twin",
                    "branch": "SIM",
                    "batch": "2026",
                },
            ]
            query = noisy_copy(rng, base, noise=0.004)
            match = face_index.find_best(query, args.threshold, args.min_margin)
            if match is None:
                ambiguous_rejected += 1
            face_index.matrix = original_matrix
            face_index.students = original_students

        all_times = genuine_times + impostor_times
        genuine_accuracy = genuine_correct / max(args.genuine_queries, 1)
        false_accept_rate = impostor_false_accepts / max(args.impostor_queries, 1)
        ambiguous_reject_rate = ambiguous_rejected / max(ambiguous_total, 1)

        print()
        print("Scale readiness result")
        print(f"Students enrolled: {args.students}")
        print(f"Database build time: {build_seconds:.2f}s")
        print(f"Face index load time: {index_seconds:.3f}s")
        print(f"Genuine queries: {args.genuine_queries}")
        print(f"Genuine correct: {genuine_correct}")
        print(f"Genuine missed: {genuine_missed}")
        print(f"Synthetic genuine accuracy: {genuine_accuracy:.2%}")
        print(f"Impostor queries: {args.impostor_queries}")
        print(f"Impostor false accepts: {impostor_false_accepts}")
        print(f"Synthetic false accept rate: {false_accept_rate:.2%}")
        print(f"Ambiguous pairs tested: {ambiguous_total}")
        print(f"Ambiguous rejected: {ambiguous_rejected}")
        print(f"Ambiguous reject rate: {ambiguous_reject_rate:.2%}")
        print(f"Average match time: {statistics.mean(all_times):.4f} ms")
        print(f"P95 match time: {percentile(all_times, 95):.4f} ms")
        print(f"Max match time: {max(all_times):.4f} ms")
        print()
        if (
            genuine_accuracy >= 0.98
            and false_accept_rate <= 0.01
            and ambiguous_reject_rate >= 0.90
            and percentile(all_times, 95) <= 10.0
        ):
            print("Verdict: PASS for 1500+ scale-readiness simulation.")
        else:
            print("Verdict: NEEDS TUNING for 1500+ scale-readiness simulation.")
        print("Note: this tests scale and matching logic, not real-world camera accuracy.")
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
