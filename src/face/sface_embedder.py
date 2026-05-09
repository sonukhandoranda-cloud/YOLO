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


class SFaceEmbedder:
    """OpenCV YuNet + SFace embedding pipeline.

    YuNet detects and landmarks the face. SFace aligns the crop and creates a
    recognition embedding. This is a much better baseline than histogram-based
    matching and is suitable for the PC prototype.
    """

    def __init__(
        self,
        detector_path: str | Path = "models/face/face_detection_yunet_2023mar.onnx",
        recognizer_path: str | Path = "models/face/face_recognition_sface_2021dec.onnx",
        score_threshold: float = 0.75,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> None:
        self.detector_path = Path(detector_path)
        self.recognizer_path = Path(recognizer_path)
        if not self.detector_path.exists() or not self.recognizer_path.exists():
            raise FileNotFoundError(
                "Face models are missing. Run: python scripts/download_face_models.py"
            )
        self.detector = cv2.FaceDetectorYN_create(
            str(self.detector_path),
            "",
            (320, 320),
            score_threshold,
            nms_threshold,
            top_k,
        )
        self.recognizer = cv2.FaceRecognizerSF_create(str(self.recognizer_path), "")

    def detect_largest(self, image_bgr: np.ndarray) -> FaceResult | None:
        height, width = image_bgr.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(image_bgr)
        if faces is None or len(faces) == 0:
            return None

        face = max(faces, key=lambda item: item[2] * item[3])
        aligned = self.recognizer.alignCrop(image_bgr, face)
        embedding = self.recognizer.feature(aligned).flatten().astype(np.float32)
        embedding = embedding / max(float(np.linalg.norm(embedding)), 1e-6)

        x, y, w, h = [int(v) for v in face[:4]]
        score = float(face[-1])
        quality = self._quality(aligned, score, image_bgr.shape)
        return FaceResult(embedding=embedding, bbox=(x, y, w, h), quality_score=quality)

    @staticmethod
    def _quality(aligned_bgr: np.ndarray, detector_score: float, image_shape: tuple[int, ...]) -> float:
        gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
        blur_score = min(1.0, cv2.Laplacian(gray, cv2.CV_64F).var() / 150.0)
        brightness = float(gray.mean())
        brightness_score = 1.0 - min(abs(brightness - 128.0) / 128.0, 1.0)
        return float((min(detector_score, 1.0) + blur_score + brightness_score) / 3.0)
