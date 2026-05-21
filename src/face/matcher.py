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
    min_margin: float = 0.03,
) -> StudentMatch | None:
    best: StudentMatch | None = None
    second_score: float | None = None
    for row, stored in iter_embeddings(conn):
        if stored.shape != embedding.shape:
            continue
        score = cosine_similarity(embedding, stored)
        if best is None or score > best.score:
            second_score = best.score if best is not None else second_score
            best = StudentMatch(
                student_id=int(row["student_id"]),
                roll_no=str(row["roll_no"]),
                name=str(row["name"]),
                branch=str(row["branch"]),
                batch=str(row["batch"]),
                score=score,
            )
        elif second_score is None or score > second_score:
            second_score = score
    if best is None or best.score < threshold:
        return None
    margin = best.score - second_score if second_score is not None else None
    if margin is not None and margin < min_margin:
        return None
    best = StudentMatch(
        student_id=best.student_id,
        roll_no=best.roll_no,
        name=best.name,
        branch=best.branch,
        batch=best.batch,
        score=best.score,
        second_score=second_score,
        margin=margin,
    )
    return best
