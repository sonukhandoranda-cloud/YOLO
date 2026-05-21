from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.footwear.detector import FootwearDetector, draw_detections


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(path: Path):
    for item in sorted(path.rglob("*")):
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS:
            yield item


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the footwear YOLO model on a folder of images.")
    parser.add_argument("--images", default="data/raw/footwear_samples")
    parser.add_argument("--model", default="models/footwear/shoe_detect_best.pt")
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--output-csv", default="data/processed/footwear_eval.csv")
    parser.add_argument("--output-dir", default="data/processed/footwear_eval")
    args = parser.parse_args()

    image_dir = Path(args.images)
    output_csv = Path(args.output_csv)
    output_dir = Path(args.output_dir)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = FootwearDetector(args.model, args.conf)
    rows: list[dict[str, object]] = []

    images = list(iter_images(image_dir))
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")

    for index, image_path in enumerate(images, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            rows.append(
                {
                    "image": str(image_path),
                    "read_ok": False,
                    "detections": 0,
                    "best_confidence": 0.0,
                    "labels": "",
                }
            )
            continue

        detections = detector.detect(image)
        best_confidence = max((det.confidence for det in detections), default=0.0)
        labels = ",".join(sorted({det.label for det in detections}))
        annotated = draw_detections(image, detections)
        cv2.imwrite(str(output_dir / image_path.name), annotated)
        rows.append(
            {
                "image": str(image_path),
                "read_ok": True,
                "detections": len(detections),
                "best_confidence": round(best_confidence, 4),
                "labels": labels,
            }
        )
        print(f"{index:02d}/{len(images)} {image_path.name}: {len(detections)} detections best={best_confidence:.2f}")

    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image", "read_ok", "detections", "best_confidence", "labels"])
        writer.writeheader()
        writer.writerows(rows)

    detected = sum(1 for row in rows if int(row["detections"]) > 0)
    readable = sum(1 for row in rows if row["read_ok"])
    print()
    print("Footwear evaluation summary")
    print(f"Images read: {readable}/{len(rows)}")
    print(f"Images with detections: {detected}/{readable}")
    print(f"Detection rate: {(detected / max(readable, 1)):.2%}")
    print(f"CSV: {output_csv}")
    print(f"Annotated images: {output_dir}")


if __name__ == "__main__":
    main()
