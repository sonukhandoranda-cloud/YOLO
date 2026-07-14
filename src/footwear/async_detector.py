from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from src.footwear.detector import Detection, FootwearDetector


@dataclass(frozen=True)
class FootwearResult:
    detections: list[Detection]
    frame_timestamp: float
    result_timestamp: float
    inference_ms: float
    sequence: int

    @property
    def age_ms(self) -> float:
        return (time.perf_counter() - self.frame_timestamp) * 1000


class AsyncFootwearDetector:
    def __init__(self, detector: FootwearDetector) -> None:
        self.detector = detector
        self._condition = threading.Condition()
        self._running = False
        self._thread: threading.Thread | None = None
        self._pending_frame: np.ndarray | None = None
        self._pending_timestamp = 0.0
        self._pending_sequence = -1
        self._processed_sequence = -1
        self._latest_result: FootwearResult | None = None

    def start(self) -> "AsyncFootwearDetector":
        if self._running:
            return self
        self._running = True
        self._thread = threading.Thread(target=self._run, name="footwear-inference", daemon=True)
        self._thread.start()
        return self

    def submit(self, frame: np.ndarray, timestamp: float, sequence: int) -> None:
        with self._condition:
            if sequence <= self._pending_sequence or sequence <= self._processed_sequence:
                return
            self._pending_frame = frame.copy()
            self._pending_timestamp = timestamp
            self._pending_sequence = sequence
            self._condition.notify()

    def latest(self) -> FootwearResult | None:
        with self._condition:
            return self._latest_result

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._running and (
                    self._pending_frame is None or self._pending_sequence <= self._processed_sequence
                ):
                    self._condition.wait(timeout=0.1)
                if not self._running:
                    return
                frame = self._pending_frame
                frame_timestamp = self._pending_timestamp
                sequence = self._pending_sequence
                self._pending_frame = None
            if frame is None:
                continue
            start = time.perf_counter()
            detections = self.detector.detect(frame)
            finished = time.perf_counter()
            result = FootwearResult(
                detections=detections,
                frame_timestamp=frame_timestamp,
                result_timestamp=finished,
                inference_ms=(finished - start) * 1000,
                sequence=sequence,
            )
            with self._condition:
                self._processed_sequence = sequence
                self._latest_result = result

    def stop(self) -> None:
        with self._condition:
            self._running = False
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
