from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraFrame:
    frame: np.ndarray
    timestamp: float
    sequence: int


def open_capture(index: int, width: int = 640, height: int = 480, fps: int = 30) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


class LatestFrameCamera:
    def __init__(self, index: int, width: int = 640, height: int = 480, fps: int = 30) -> None:
        self.index = index
        self.capture = open_capture(index, width=width, height=height, fps=fps)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open camera {index}")
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._latest: CameraFrame | None = None
        self._sequence = 0

    def start(self) -> "LatestFrameCamera":
        if self._running:
            return self
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, name=f"camera-{self.index}", daemon=True)
        self._thread.start()
        return self

    def _read_loop(self) -> None:
        while self._running:
            ok, frame = self.capture.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            timestamp = time.perf_counter()
            with self._lock:
                self._sequence += 1
                self._latest = CameraFrame(frame=frame, timestamp=timestamp, sequence=self._sequence)

    def read_latest(self) -> CameraFrame | None:
        with self._lock:
            if self._latest is None:
                return None
            return CameraFrame(
                frame=self._latest.frame.copy(),
                timestamp=self._latest.timestamp,
                sequence=self._latest.sequence,
            )

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.capture.release()
