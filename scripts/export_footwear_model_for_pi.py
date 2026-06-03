from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a footwear YOLO model for Raspberry Pi deployment.")
    parser.add_argument("--model", required=True, help="Path to trained YOLO .pt model.")
    parser.add_argument("--format", default="ncnn", choices=["ncnn", "onnx", "tflite"], help="Export format.")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(str(model_path))
    output = model.export(format=args.format, imgsz=args.imgsz)
    print(f"Exported model: {output}")


if __name__ == "__main__":
    main()
