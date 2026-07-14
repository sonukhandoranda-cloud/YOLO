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
CLASS_IDS = {"shoe": 0, "sandal": 1, "other_footwear": 2}


def iter_labeled_images(root: Path) -> list[tuple[Path, str]]:
    images: list[tuple[Path, str]] = []
    for class_name in CLASS_IDS:
        class_dir = root / class_name
        if not class_dir.exists():
            continue
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append((image_path, class_name))
    return images


def yolo_box(class_id: int, width: int, height: int, bbox: tuple[int, int, int, int]) -> str:
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
        f"{class_id} {x_center / width:.6f} {y_center / height:.6f} "
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
    parser = argparse.ArgumentParser(
        description="Build a 3-class YOLO footwear dataset using class folders and pseudo boxes."
    )
    parser.add_argument("--images-root", default="data/raw/footwear_classes")
    parser.add_argument("--box-model", default="models/footwear/shoe_detect_best.pt")
    parser.add_argument("--output", default="data/footwear_yolo_public")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    labeled_images = iter_labeled_images(Path(args.images_root))
    if not labeled_images:
        raise FileNotFoundError(f"No class images found under {args.images_root}")
    random.Random(args.seed).shuffle(labeled_images)

    output_dir = Path(args.output)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    detector = FootwearDetector(args.box_model, args.conf)
    kept_by_class = {class_name: 0 for class_name in CLASS_IDS}
    skipped_by_class = {class_name: 0 for class_name in CLASS_IDS}
    for index, (image_path, class_name) in enumerate(labeled_images):
        image = cv2.imread(str(image_path))
        if image is None:
            skipped_by_class[class_name] += 1
            continue
        detections = detector.detect(image)
        if not detections:
            skipped_by_class[class_name] += 1
            continue

        split = split_name(index, len(labeled_images))
        safe_name = f"{class_name}_{image_path.name}"
        target_image = output_dir / "images" / split / safe_name
        target_label = output_dir / "labels" / split / f"{Path(safe_name).stem}.txt"
        shutil.copy2(image_path, target_image)
        height, width = image.shape[:2]
        class_id = CLASS_IDS[class_name]
        target_label.write_text(
            "\n".join(yolo_box(class_id, width, height, detection.bbox) for detection in detections),
            encoding="utf-8",
        )
        kept_by_class[class_name] += 1

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
        print(f"{class_name}: kept={kept_by_class[class_name]} skipped={skipped_by_class[class_name]}")
    print("Note: boxes are pseudo-labeled. Replace with manually labeled camera-angle data for final accuracy.")


if __name__ == "__main__":
    main()
