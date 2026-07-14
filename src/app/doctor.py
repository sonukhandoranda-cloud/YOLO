from __future__ import annotations

import argparse
import importlib
import sqlite3
import sys
from pathlib import Path

import cv2

from src.cameras.threaded import open_capture
from src.database.db import init_db


def check_import(module_name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "installed")
        return True, str(version)
    except Exception as exc:
        return False, str(exc)


def check_camera(index: int) -> tuple[bool, str]:
    cap = open_capture(index)
    try:
        if not cap.isOpened():
            return False, "not available"
        ok, frame = cap.read()
        if not ok or frame is None:
            return False, "opened, but no frame received"
        height, width = frame.shape[:2]
        return True, f"available ({width}x{height})"
    finally:
        cap.release()


def check_database(path: Path) -> tuple[bool, str]:
    try:
        init_db(path)
        with sqlite3.connect(path) as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        return True, f"ready with {len(tables)} tables"
    except Exception as exc:
        return False, str(exc)


def check_shoe_model(path: Path) -> tuple[bool, str]:
    try:
        from src.footwear.detector import resolve_model_path
        from ultralytics import YOLO

        model_path = resolve_model_path(path)
        model = YOLO(str(model_path), task="detect")
        return True, f"loaded classes: {model.names}"
    except Exception as exc:
        return False, str(exc)


def check_face_models() -> tuple[bool, str]:
    detector = Path("models/face/face_detection_yunet_2023mar.onnx")
    recognizer = Path("models/face/face_recognition_sface_2021dec.onnx")
    if not detector.exists() or not recognizer.exists():
        return False, "missing; run: python scripts/download_face_models.py"
    try:
        from src.face.sface_embedder import SFaceEmbedder

        SFaceEmbedder(detector, recognizer)
        return True, "YuNet + SFace ready"
    except Exception as exc:
        return False, str(exc)


def print_result(name: str, ok: bool, detail: str) -> None:
    status = "OK" if ok else "FAIL"
    print(f"[{status}] {name}: {detail}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether the attendance project is ready to run.")
    parser.add_argument("--db", default="data/attendance/attendance.sqlite3")
    parser.add_argument("--shoe-model", default="models/footwear/footwear_policy_yolo11n.pt")
    parser.add_argument("--check-cameras", type=int, default=2, help="Number of camera indexes to check from 0.")
    args = parser.parse_args()

    print("AI Attendance System Doctor")
    print(f"Python: {sys.version.split()[0]}")
    print()

    checks: list[tuple[str, bool, str]] = []

    for module_name in ["cv2", "numpy", "ultralytics", "torch"]:
        ok, detail = check_import(module_name)
        checks.append((module_name, ok, detail))

    db_ok, db_detail = check_database(Path(args.db))
    checks.append(("database", db_ok, db_detail))

    model_ok, model_detail = check_shoe_model(Path(args.shoe_model))
    checks.append(("shoe model", model_ok, model_detail))

    face_ok, face_detail = check_face_models()
    checks.append(("face models", face_ok, face_detail))

    for index in range(args.check_cameras):
        cam_ok, cam_detail = check_camera(index)
        checks.append((f"camera {index}", cam_ok, cam_detail))

    for name, ok, detail in checks:
        print_result(name, ok, detail)

    required_failed = [name for name, ok, _ in checks if name not in {"camera 1"} and not ok]
    print()
    if required_failed:
        print("Result: setup needs attention.")
        print("Fix these first:", ", ".join(required_failed))
    else:
        print("Result: core setup is ready.")
        print("If camera 1 failed, that is okay for now. You can test one camera at a time.")


if __name__ == "__main__":
    main()
