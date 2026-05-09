from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]


class FootwearDetector:
    def __init__(self, model_path: str | Path, confidence_threshold: float = 0.5) -> None:
        from ultralytics import YOLO

        self.model_path = Path(model_path)
        if not self.model_path.exists() or self.model_path.is_dir():
            raise FileNotFoundError(
                f"YOLO model file not found: {self.model_path}. "
                "Run scripts/recover_extracted_checkpoint.py if you only have an extracted checkpoint folder."
            )
        self.model = YOLO(str(self.model_path))
        self.confidence_threshold = confidence_threshold

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        results = self.model.predict(frame_bgr, conf=self.confidence_threshold, verbose=False)
        if not results:
            return []
        result = results[0]
        detections: list[Detection] = []
        names = result.names
        if result.boxes is None:
            return detections
        for box in result.boxes:
            confidence = float(box.conf[0])
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            detections.append(
                Detection(
                    label=str(names.get(cls_id, cls_id)),
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                )
            )
        return detections


def draw_detections(frame_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    output = frame_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 180, 255), 2)
        cv2.putText(
            output,
            f"{det.label} {det.confidence:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 180, 255),
            2,
        )
    return output
