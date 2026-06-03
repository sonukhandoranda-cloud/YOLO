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
from src.face.index import FaceIndex
from src.footwear.detector import Detection
from src.fusion.footwear_policy import FootwearPolicy
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
                index = FaceIndex.load(conn)
                indexed_match = index.find_best(vector, threshold=0.99)
                self.assertIsNotNone(indexed_match)
                self.assertEqual(indexed_match.roll_no, "101")
                self.assertTrue(
                    mark_attendance(
                        conn,
                        student_id=student_id,
                        session_id=session_id,
                        face_score=1.0,
                        presence_score=0.8,
                        presence_label="sandal",
                        decision_status="present_with_flag",
                    )
                )
                row = conn.execute(
                    "SELECT presence_label, decision_status FROM attendance_records WHERE student_id = ?",
                    (student_id,),
                ).fetchone()
                self.assertEqual(row["presence_label"], "sandal")
                self.assertEqual(row["decision_status"], "present_with_flag")
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

    def test_ambiguous_match_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "attendance.sqlite3"
            init_db(db_path)

            vector = np.ones(128, dtype=np.float32)
            vector = vector / np.linalg.norm(vector)

            conn = connect(db_path)
            try:
                for roll_no in ["101", "102"]:
                    student_id = upsert_student(
                        conn,
                        roll_no=roll_no,
                        name=f"Student {roll_no}",
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
                match = find_best_match(conn, vector, threshold=0.99, min_margin=0.03)
                self.assertIsNone(match)
            finally:
                conn.close()

    def test_footwear_policy(self) -> None:
        policy = FootwearPolicy()

        shoe = policy.decide([Detection("shoe", 0.9, (0, 0, 1, 1))])
        self.assertEqual(shoe.status, "present")
        self.assertTrue(shoe.can_mark_attendance)

        sandal = policy.decide([Detection("sandal", 0.8, (0, 0, 1, 1))])
        self.assertEqual(sandal.status, "present_with_flag")
        self.assertTrue(sandal.can_mark_attendance)

        slipper = policy.decide([Detection("slipper", 0.7, (0, 0, 1, 1))])
        self.assertEqual(slipper.status, "absent")
        self.assertFalse(slipper.can_mark_attendance)


if __name__ == "__main__":
    unittest.main()
