import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.database.db import (
    add_face_embedding,
    connect,
    effective_attendance_status,
    get_or_create_session,
    init_db,
    mark_attendance,
    recently_marked,
    upsert_student,
)
from src.face.matcher import find_best_match
from src.face.index import FaceIndex
from src.footwear.detector import Detection, resolve_model_path
from src.fusion.footwear_policy import FootwearDecisionWindow, FootwearPolicy
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

    def test_footwear_decision_window_holds_recent_valid_detection(self) -> None:
        policy = FootwearPolicy()
        window = FootwearDecisionWindow(window_size=4, required_votes=2, hold_ms=1000)

        first = window.add(policy.decide([Detection("shoe", 0.4, (0, 0, 1, 1))]))
        self.assertEqual(first.status, "present")

        stable = window.add(policy.decide([Detection("shoe", 0.45, (0, 0, 1, 1))]))
        self.assertEqual(stable.status, "present")
        self.assertEqual(stable.label, "shoe")

        held = window.add(policy.decide([]))
        self.assertEqual(held.status, "present")
        self.assertEqual(held.label, "shoe")

    def test_missing_footwear_is_not_reported_as_clean_present(self) -> None:
        self.assertEqual(
            effective_attendance_status("present", "", 0.0),
            "face_only_no_footwear",
        )
        self.assertEqual(
            effective_attendance_status("present", "shoe", 0.45),
            "present",
        )

    def test_duplicate_cooldown_can_be_scoped_to_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "attendance.sqlite3"
            init_db(db_path)
            conn = connect(db_path)
            try:
                student_id = upsert_student(
                    conn,
                    roll_no="101",
                    name="Test Student",
                    branch="CSE",
                    batch="2026",
                )
                face_session = get_or_create_session(conn, "face-demo")
                real_session = get_or_create_session(conn, "default")
                self.assertTrue(
                    mark_attendance(
                        conn,
                        student_id=student_id,
                        session_id=face_session,
                        face_score=0.8,
                        presence_score=0.0,
                    )
                )
                self.assertTrue(
                    recently_marked(
                        conn,
                        student_id=student_id,
                        cooldown_minutes=30,
                        session_id=face_session,
                    )
                )
                self.assertFalse(
                    recently_marked(
                        conn,
                        student_id=student_id,
                        cooldown_minutes=30,
                        session_id=real_session,
                    )
                )
            finally:
                conn.close()

    def test_footwear_model_path_validation_accepts_pt_and_ncnn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pt_model = root / "model.pt"
            pt_model.write_bytes(b"placeholder")
            self.assertEqual(resolve_model_path(pt_model), pt_model)

            ncnn_model = root / "model_ncnn"
            ncnn_model.mkdir()
            (ncnn_model / "model.ncnn.param").write_text("placeholder", encoding="utf-8")
            (ncnn_model / "model.ncnn.bin").write_bytes(b"placeholder")
            self.assertEqual(resolve_model_path(ncnn_model), ncnn_model)

            bad_dir = root / "bad_model"
            bad_dir.mkdir()
            with self.assertRaises(FileNotFoundError):
                resolve_model_path(bad_dir)


if __name__ == "__main__":
    unittest.main()
