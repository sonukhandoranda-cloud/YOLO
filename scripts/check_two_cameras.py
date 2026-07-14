from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cameras.threaded import LatestFrameCamera


def resize(frame, width: int, height: int):
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview two cameras side by side and measure live FPS.")
    parser.add_argument("--face-camera", type=int, default=0)
    parser.add_argument("--footwear-camera", type=int, default=1)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds", type=int, default=0, help="Auto-stop after this many seconds. 0 means wait for ESC.")
    args = parser.parse_args()

    face_cam = LatestFrameCamera(args.face_camera, width=args.width, height=args.height, fps=args.fps).start()
    shoe_cam = LatestFrameCamera(args.footwear_camera, width=args.width, height=args.height, fps=args.fps).start()
    started = time.perf_counter()
    frames = 0
    try:
        while True:
            face = face_cam.read_latest()
            shoe = shoe_cam.read_latest()
            if face is None or shoe is None:
                time.sleep(0.01)
                continue

            now = time.perf_counter()
            frames += 1
            face_frame = resize(face.frame, 480, 360)
            shoe_frame = resize(shoe.frame, 480, 360)
            face_age = (now - face.timestamp) * 1000
            shoe_age = (now - shoe.timestamp) * 1000
            live_fps = frames / max(now - started, 0.001)

            cv2.putText(face_frame, f"FACE cam {args.face_camera} age {face_age:.0f}ms", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(shoe_frame, f"FOOT cam {args.footwear_camera} age {shoe_age:.0f}ms", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            combined = np.hstack([face_frame, shoe_frame])
            cv2.putText(combined, f"preview fps {live_fps:.1f} | ESC to quit", (15, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.imshow("Two Camera Check", combined)

            if args.seconds and now - started >= args.seconds:
                break
            if (cv2.waitKey(1) & 0xFF) == 27:
                break
    finally:
        face_cam.stop()
        shoe_cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
