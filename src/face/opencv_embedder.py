from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class FaceResult:
    embedding: np.ndarray
    bbox: tuple[int, int, int, int]
    quality_score: float


class OpenCVFaceEmbedder:
    """Lightweight baseline face engine.

    This creates a normalized histogram descriptor from an aligned face crop.
    It is useful for wiring and demos, but should be replaced by ArcFace or
    InsightFace for the final high-accuracy system.
    """

    def __init__(self, min_face_size: int = 80) -> None:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(str(cascade_path))
        if self.detector.empty():
            raise RuntimeError(f"Could not load Haar cascade: {cascade_path}")
        self.min_face_size = min_face_size

    def detect_largest(self, image_bgr: np.ndarray) -> FaceResult | None:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size),
        )
        if len(faces) == 0:
            return None

        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        crop = gray[y : y + h, x : x + w]
        embedding = self._embedding(crop)
        quality = self._quality(crop, image_bgr.shape)
        return FaceResult(embedding=embedding, bbox=(int(x), int(y), int(w), int(h)), quality_score=quality)

    @staticmethod
    def _embedding(face_gray: np.ndarray) -> np.ndarray:
        resized = cv2.resize(face_gray, (128, 128), interpolation=cv2.INTER_AREA)
        equalized = cv2.equalizeHist(resized)
        hist = cv2.calcHist([equalized], [0], None, [64], [0, 256]).flatten()
        hist = hist / max(float(hist.sum()), 1.0)
        small = cv2.resize(equalized, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32).flatten()
        small = (small - small.mean()) / (small.std() + 1e-6)
        vector = np.concatenate([hist.astype(np.float32), small])
        norm = np.linalg.norm(vector)
        return vector / max(float(norm), 1e-6)

    @staticmethod
    def _quality(face_gray: np.ndarray, image_shape: tuple[int, ...]) -> float:
        blur = cv2.Laplacian(face_gray, cv2.CV_64F).var()
        brightness = float(face_gray.mean())
        height, width = image_shape[:2]
        size_score = min(1.0, (face_gray.shape[0] * face_gray.shape[1]) / max(width * height * 0.08, 1.0))
        blur_score = min(1.0, blur / 150.0)
        brightness_score = 1.0 - min(abs(brightness - 128.0) / 128.0, 1.0)
        return float((size_score + blur_score + brightness_score) / 3.0)
