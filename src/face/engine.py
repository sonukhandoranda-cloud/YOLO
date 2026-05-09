from __future__ import annotations

from src.face.opencv_embedder import OpenCVFaceEmbedder
from src.face.sface_embedder import SFaceEmbedder


def create_face_embedder():
    try:
        return SFaceEmbedder()
    except FileNotFoundError:
        return OpenCVFaceEmbedder()
