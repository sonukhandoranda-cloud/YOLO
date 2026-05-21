from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np

from src.database.db import StudentMatch, iter_embeddings


@dataclass
class FaceIndex:
    students: list[dict[str, object]]
    matrix: np.ndarray

    @classmethod
    def load(cls, conn: sqlite3.Connection) -> "FaceIndex":
        students: list[dict[str, object]] = []
        vectors: list[np.ndarray] = []
        for row, vector in iter_embeddings(conn):
            students.append(
                {
                    "student_id": int(row["student_id"]),
                    "roll_no": str(row["roll_no"]),
                    "name": str(row["name"]),
                    "branch": str(row["branch"]),
                    "batch": str(row["batch"]),
                }
            )
            vectors.append(vector.astype(np.float32))

        if not vectors:
            return cls(students=[], matrix=np.empty((0, 0), dtype=np.float32))

        matrix = np.vstack(vectors).astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.maximum(norms, 1e-6)
        return cls(students=students, matrix=matrix)

    def find_best(
        self,
        embedding: np.ndarray,
        threshold: float,
        min_margin: float = 0.03,
    ) -> StudentMatch | None:
        if self.matrix.size == 0:
            return None
        if self.matrix.shape[1] != embedding.shape[0]:
            return None

        query = embedding.astype(np.float32)
        query = query / max(float(np.linalg.norm(query)), 1e-6)
        scores = self.matrix @ query
        best_index = int(np.argmax(scores))
        best_score = float(scores[best_index])
        if best_score < threshold:
            return None

        second_score: float | None = None
        if len(scores) > 1:
            partitioned = np.partition(scores, -2)
            second_score = float(partitioned[-2])
            if best_score - second_score < min_margin:
                return None

        student = self.students[best_index]
        margin = best_score - second_score if second_score is not None else None
        return StudentMatch(
            student_id=int(student["student_id"]),
            roll_no=str(student["roll_no"]),
            name=str(student["name"]),
            branch=str(student["branch"]),
            batch=str(student["batch"]),
            score=best_score,
            second_score=second_score,
            margin=margin,
        )
