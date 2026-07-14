from __future__ import annotations

import subprocess
import sys
from pathlib import Path


CHECKS = [
    ("doctor", [sys.executable, "-m", "src.app.doctor", "--check-cameras", "0"]),
    ("unit tests", [sys.executable, "-m", "unittest", "discover", "-s", "tests"]),
    ("compile", [sys.executable, "-m", "compileall", "src", "scripts", "tests", "face_recognition_simple.py", "shoe_detection_only.py"]),
    ("database status", [sys.executable, "-m", "src.app.project_status"]),
    ("yolo dataset status", [sys.executable, "scripts/validate_yolo_dataset.py"]),
]


def main() -> None:
    failed: list[str] = []
    for name, command in CHECKS:
        print()
        print(f"=== {name} ===")
        result = subprocess.run(command, text=True)
        if result.returncode != 0:
            failed.append(name)

    print()
    print("=== readiness summary ===")
    if failed:
        print("NEEDS ATTENTION")
        print("Failed checks:", ", ".join(failed))
        raise SystemExit(1)

    print("READY FOR PC SOFTWARE CHECKPOINT")
    if Path("models/footwear/footwear_policy_yolo11n.pt").exists():
        print("Footwear policy model ready: models/footwear/footwear_policy_yolo11n.pt")
    print("Note: data/footwear_yolo is the manual-label dataset slot and may be EMPTY until real Pi-angle images are added.")


if __name__ == "__main__":
    main()
