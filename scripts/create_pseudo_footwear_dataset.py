from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.footwear.detector import FootwearDetector


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(path: Path):
    for image_path in sorted(path.rglob("*")):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            yield image_path


def yolo_box(width: int, height: int, bbox: tuple[int, int, int, int]) -> str:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    x_center = x1 + box_width / 2
    y_center = y1 + box_height / 2
    return (
        f"0 {x_center / width:.6f} {y_center / height:.6f} "
        f"{box_width / width:.6f} {box_height / height:.6f}"
    )


def split_name(index: int, total: int) -> str:
    train_cutoff = int(total * 0.7)
    val_cutoff = int(total * 0.9)
    if index < train_cutoff:
        return "train"
    if index < val_cutoff:
        return "val"
    return "test"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tiny pseudo-labeled YOLO footwear dataset for smoke training.")
    parser.add_argument("--images", default="data/raw/footwear_samples")
    parser.add_argument("--model", default="models/footwear/shoe_detect_best.pt")
    parser.add_argument("--output", default="data/footwear_yolo_pseudo")
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    image_paths = list(iter_images(Path(args.images)))
    if not image_paths:
        raise FileNotFoundError(f"No images found: {args.images}")

    random.Random(args.seed).shuffle(image_paths)
    output_dir = Path(args.output)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    detector = FootwearDetector(args.model, args.conf)
    kept = 0
    skipped = 0
    for index, image_path in enumerate(image_paths):
        image = cv2.imread(str(image_path))
        if image is None:
            skipped += 1
            continue
        detections = detector.detect(image)
        if not detections:
            skipped += 1
            continue
        split = split_name(index, len(image_paths))
        target_image = output_dir / "images" / split / image_path.name
        target_label = output_dir / "labels" / split / f"{image_path.stem}.txt"
        shutil.copy2(image_path, target_image)
        height, width = image.shape[:2]
        target_label.write_text(
            "\n".join(yolo_box(width, height, detection.bbox) for detection in detections),
            encoding="utf-8",
        )
        kept += 1

    (output_dir / "data.yaml").write_text(
        f"""path: {output_dir.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: shoe
""",
        encoding="utf-8",
    )

    print(f"Pseudo dataset: {output_dir}")
    print(f"Images with pseudo labels: {kept}")
    print(f"Skipped images: {skipped}")
    print("Important: this is for pipeline smoke testing, not final accuracy training.")


if __name__ == "__main__":
    main()
