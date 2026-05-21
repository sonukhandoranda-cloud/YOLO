from __future__ import annotations

import argparse
import statistics
import sys
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


def add_noise(rng: np.random.Generator, vector: np.ndarray, noise: float) -> np.ndarray:
    noisy = vector + rng.normal(scale=noise, size=vector.shape).astype(np.float32)
    return noisy / max(float(np.linalg.norm(noisy)), 1e-6)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate 1500 enrolled students and benchmark face matching.")
    parser.add_argument("--db", default="data/attendance/simulation.sqlite3")
    parser.add_argument("--students", type=int, default=1500)
    parser.add_argument("--embedding-size", type=int, default=128)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--noise", type=float, default=0.03)
    parser.add_argument("--threshold", type=float, default=0.72)
    parser.add_argument("--min-margin", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    init_db(args.db)

    print(f"Creating {args.students} simulated students in {args.db}...")
    vectors: list[tuple[str, np.ndarray]] = []
    start = time.perf_counter()
    conn = connect(args.db)
    try:
        for index in range(args.students):
            roll_no = f"SIM{index + 1:04d}"
            vector = unit_vector(rng, args.embedding_size)
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
    write_seconds = time.perf_counter() - start

    print(f"Inserted records in {write_seconds:.2f}s")
    print(f"Running {args.queries} recognition queries...")

    timings_ms: list[float] = []
    correct = 0
    misses = 0
    conn = connect(args.db)
    try:
        face_index = FaceIndex.load(conn)
        for _ in range(args.queries):
            expected_roll, vector = vectors[int(rng.integers(0, len(vectors)))]
            query = add_noise(rng, vector, args.noise)
            q_start = time.perf_counter()
            match = face_index.find_best(query, args.threshold, args.min_margin)
            timings_ms.append((time.perf_counter() - q_start) * 1000.0)
            if match is None:
                misses += 1
            elif match.roll_no == expected_roll:
                correct += 1
    finally:
        conn.close()

    accuracy = correct / max(args.queries, 1)
    print()
    print("Simulation result")
    print(f"Students: {args.students}")
    print(f"Queries: {args.queries}")
    print(f"Correct: {correct}")
    print(f"Misses: {misses}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Average match time: {statistics.mean(timings_ms):.2f} ms")
    print(f"Median match time: {statistics.median(timings_ms):.2f} ms")
    print(f"Max match time: {max(timings_ms):.2f} ms")


if __name__ == "__main__":
    main()
