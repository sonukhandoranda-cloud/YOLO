from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


MODELS = {
    "face_detection_yunet_2023mar.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}


def download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > 0:
        print(f"Already exists: {output}")
        return
    print(f"Downloading {output.name}...")
    urllib.request.urlretrieve(url, output)
    print(f"Saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download OpenCV YuNet and SFace ONNX models.")
    parser.add_argument("--output-dir", default="models/face")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    for filename, url in MODELS.items():
        download(url, output_dir / filename)


if __name__ == "__main__":
    main()
