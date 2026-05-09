from __future__ import annotations

import sqlite3

import numpy as np

from src.database.db import StudentMatch, iter_embeddings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-6))


def find_best_match(
    conn: sqlite3.Connection,
    embedding: np.ndarray,
    threshold: float,
) -> StudentMatch | None:
    best: StudentMatch | None = None
    for row, stored in iter_embeddings(conn):
        if stored.shape != embedding.shape:
            continue
        score = cosine_similarity(embedding, stored)
        if best is None or score > best.score:
            best = StudentMatch(
                student_id=int(row["student_id"]),
                roll_no=str(row["roll_no"]),
                name=str(row["name"]),
                branch=str(row["branch"]),
                batch=str(row["batch"]),
                score=score,
            )
    if best is None or best.score < threshold:
        return None
    return best
