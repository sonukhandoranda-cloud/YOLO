from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.footwear.detector import FootwearDetector


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_IDS = {"shoe": 0, "sandal": 1, "other_footwear": 2}


def infer_class(path: Path) -> str | None:
    name = path.name.lower()
    if any(token in name for token in ["sandal"]):
        return "sandal"
    if any(token in name for token in ["slipper", "flip", "bare"]):
        return "other_footwear"
    if any(
        token in name
        for token in [
            "shoe",
            "shoes",
            "sneaker",
            "adidas",
            "reebok",
            "fila",
            "nike",
            "foot_binding",
            "footwear",
            "cobbler",
        ]
    ):
        return "shoe"
    return None


def iter_sources(paths: list[Path]) -> list[tuple[Path, str]]:
    items: list[tuple[Path, str]] = []
    for root in paths:
        if not root.exists():
            continue
        for image_path in sorted(root.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            class_name = infer_class(image_path)
            if class_name:
                items.append((image_path, class_name))
    return items


def yolo_box(class_id: int, width: int, height: int, bbox: tuple[int, int, int, int], flip_x: bool = False) -> str:
    x1, y1, x2, y2 = bbox
    if flip_x:
        x1, x2 = width - x2, width - x1
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    x_center = x1 + box_width / 2
    y_center = y1 + box_height / 2
    return (
        f"{class_id} {x_center / width:.6f} {y_center / height:.6f} "
        f"{box_width / width:.6f} {box_height / height:.6f}"
    )


def augment(image: np.ndarray, variant: int) -> np.ndarray:
    output = image.copy()
    if variant % 2 == 1:
        output = cv2.flip(output, 1)
    alpha = 0.75 + (variant % 5) * 0.12
    beta = [-22, -10, 0, 12, 24][variant % 5]
    output = cv2.convertScaleAbs(output, alpha=alpha, beta=beta)
    if variant % 4 == 0:
        output = cv2.GaussianBlur(output, (3, 3), 0)
    return output


def split_for(counter: int) -> str:
    value = counter % 10
    if value < 7:
        return "train"
    if value < 9:
        return "val"
    return "test"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an augmented demo YOLO dataset for footwear policy classes.")
    parser.add_argument("--source", action="append", default=["data/raw/footwear_samples", "data/raw/footwear_classes"])
    parser.add_argument("--box-model", default="models/footwear/shoe_detect_best.pt")
    parser.add_argument("--output", default="data/footwear_yolo_demo")
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--augmentations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sources = iter_sources([Path(path) for path in args.source])
    if not sources:
        raise FileNotFoundError("No footwear source images found.")
    random.Random(args.seed).shuffle(sources)

    output_dir = Path(args.output)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    detector = FootwearDetector(args.box_model, args.conf)
    kept = {class_name: 0 for class_name in CLASS_IDS}
    skipped = {class_name: 0 for class_name in CLASS_IDS}
    written = 0
    for image_path, class_name in sources:
        image = cv2.imread(str(image_path))
        if image is None:
            skipped[class_name] += 1
            continue
        detections = detector.detect(image)
        if not detections:
            skipped[class_name] += 1
            continue

        height, width = image.shape[:2]
        class_id = CLASS_IDS[class_name]
        for variant in range(args.augmentations):
            split = split_for(written)
            augmented = augment(image, variant)
            flip_x = variant % 2 == 1
            label_text = "\n".join(
                yolo_box(class_id, width, height, detection.bbox, flip_x=flip_x) for detection in detections
            )
            stem = f"{class_name}_{image_path.stem}_{variant:02d}"
            image_out = output_dir / "images" / split / f"{stem}.jpg"
            label_out = output_dir / "labels" / split / f"{stem}.txt"
            cv2.imwrite(str(image_out), augmented)
            label_out.write_text(label_text, encoding="utf-8")
            kept[class_name] += 1
            written += 1

    (output_dir / "data.yaml").write_text(
        f"""path: {output_dir.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: shoe
  1: sandal
  2: other_footwear
""",
        encoding="utf-8",
    )

    print(f"Dataset: {output_dir}")
    for class_name in CLASS_IDS:
        print(f"{class_name}: images={kept[class_name]} skipped_sources={skipped[class_name]}")
    print("This is a demo dataset. Replace it with manually labeled low-angle camera data for final accuracy.")


if __name__ == "__main__":
    main()
