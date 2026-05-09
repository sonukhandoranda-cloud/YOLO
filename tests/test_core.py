import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.database.db import (
    add_face_embedding,
    connect,
    get_or_create_session,
    init_db,
    mark_attendance,
    upsert_student,
)
from src.face.matcher import find_best_match
from src.fusion.decision import IdentityVoteWindow


class CoreFlowTest(unittest.TestCase):
    def test_database_match_and_vote(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "attendance.sqlite3"
            init_db(db_path)

            vector = np.ones(320, dtype=np.float32)
            vector = vector / np.linalg.norm(vector)

            conn = connect(db_path)
            try:
                student_id = upsert_student(
                    conn,
                    roll_no="101",
                    name="Test Student",
                    branch="CSE",
                    batch="2026",
                )
                add_face_embedding(
                    conn,
                    student_id=student_id,
                    vector=vector,
                    image_path=None,
                    quality_score=0.9,
                )
                session_id = get_or_create_session(conn, "demo")
                match = find_best_match(conn, vector, threshold=0.99)
                self.assertIsNotNone(match)
                self.assertEqual(match.roll_no, "101")
                self.assertTrue(
                    mark_attendance(
                        conn,
                        student_id=student_id,
                        session_id=session_id,
                        face_score=1.0,
                        presence_score=0.8,
                    )
                )
                self.assertFalse(
                    mark_attendance(
                        conn,
                        student_id=student_id,
                        session_id=session_id,
                        face_score=1.0,
                        presence_score=0.8,
                    )
                )
            finally:
                conn.close()

            votes = IdentityVoteWindow(window_size=4, required_votes=3)
            self.assertIsNone(votes.add(student_id))
            self.assertIsNone(votes.add(student_id))
            stable = votes.add(student_id)
            self.assertIsNotNone(stable)
            self.assertEqual(stable.student_id, student_id)


if __name__ == "__main__":
    unittest.main()
