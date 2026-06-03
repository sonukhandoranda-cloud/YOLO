from __future__ import annotations

import argparse
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_split(dataset_dir: Path, split: str) -> tuple[int, int, int]:
    image_dir = dataset_dir / "images" / split
    label_dir = dataset_dir / "labels" / split
    images = [path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    labels = [label_dir / f"{path.stem}.txt" for path in images]
    existing_labels = [path for path in labels if path.exists()]
    missing_labels = len(labels) - len(existing_labels)
    boxes = 0
    for label_path in existing_labels:
        with label_path.open("r", encoding="utf-8") as file:
            boxes += sum(1 for line in file if line.strip())
    return len(images), missing_labels, boxes


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate YOLO footwear dataset counts.")
    parser.add_argument("--dataset-dir", default="data/footwear_yolo")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    total_images = 0
    total_missing = 0
    total_boxes = 0
    for split in ["train", "val", "test"]:
        images, missing_labels, boxes = count_split(dataset_dir, split)
        total_images += images
        total_missing += missing_labels
        total_boxes += boxes
        print(f"{split}: images={images}, missing_labels={missing_labels}, boxes={boxes}")

    print()
    print(f"total images: {total_images}")
    print(f"total missing labels: {total_missing}")
    print(f"total boxes: {total_boxes}")
    if total_images == 0:
        print("status: EMPTY - add labeled footwear images before training.")
    elif total_missing > 0:
        print("status: NOT READY - every image needs a matching label file.")
    elif total_boxes == 0:
        print("status: NOT READY - label files exist but no boxes were found.")
    else:
        print("status: READY FOR YOLO TRAINING")


if __name__ == "__main__":
    main()
