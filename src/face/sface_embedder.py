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
        for candidate in self._detection_candidates(image_bgr):
            face_result = self._detect_largest_single(candidate)
            if face_result is not None:
                return face_result
        return None

    def _detect_largest_single(self, image_bgr: np.ndarray) -> FaceResult | None:
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

    def _detection_candidates(self, image_bgr: np.ndarray) -> list[np.ndarray]:
        resized = self._resize_for_detection(image_bgr)
        candidates = [resized]
        enhanced = self._enhance_contrast(resized)
        if enhanced is not None:
            candidates.append(enhanced)
        candidates.extend(
            [
                cv2.rotate(resized, cv2.ROTATE_90_CLOCKWISE),
                cv2.rotate(resized, cv2.ROTATE_90_COUNTERCLOCKWISE),
                cv2.rotate(resized, cv2.ROTATE_180),
            ]
        )
        if enhanced is not None:
            candidates.extend(
                [
                    cv2.rotate(enhanced, cv2.ROTATE_90_CLOCKWISE),
                    cv2.rotate(enhanced, cv2.ROTATE_90_COUNTERCLOCKWISE),
                ]
            )
        return candidates

    @staticmethod
    def _resize_for_detection(image_bgr: np.ndarray, max_side: int = 1400, min_face_side: int = 320) -> np.ndarray:
        height, width = image_bgr.shape[:2]
        longest = max(height, width)
        shortest = min(height, width)
        scale = 1.0
        if longest > max_side:
            scale = max_side / longest
        elif shortest < min_face_side:
            scale = min_face_side / max(shortest, 1)
        if abs(scale - 1.0) < 1e-6:
            return image_bgr
        return cv2.resize(
            image_bgr,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
        )

    @staticmethod
    def _enhance_contrast(image_bgr: np.ndarray) -> np.ndarray | None:
        if image_bgr.size == 0:
            return None
        ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        y = clahe.apply(y)
        return cv2.cvtColor(cv2.merge((y, cr, cb)), cv2.COLOR_YCrCb2BGR)

    @staticmethod
    def _quality(aligned_bgr: np.ndarray, detector_score: float, image_shape: tuple[int, ...]) -> float:
        gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
        blur_score = min(1.0, cv2.Laplacian(gray, cv2.CV_64F).var() / 150.0)
        brightness = float(gray.mean())
        brightness_score = 1.0 - min(abs(brightness - 128.0) / 128.0, 1.0)
        return float((min(detector_score, 1.0) + blur_score + brightness_score) / 3.0)
