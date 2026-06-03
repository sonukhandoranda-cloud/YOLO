from __future__ import annotations

import argparse
from pathlib import Path


DATA_YAML = """path: {dataset_dir}
train: images/train
val: images/val
test: images/test

names:
  0: shoe
  1: sandal
  2: other_footwear
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a YOLO footwear dataset folder structure.")
    parser.add_argument("--dataset-dir", default="data/footwear_yolo")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    for split in ["train", "val", "test"]:
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        data_yaml.write_text(DATA_YAML.format(dataset_dir=dataset_dir.as_posix()), encoding="utf-8")

    readme = dataset_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            """# Footwear YOLO Dataset

Place images and YOLO label files here:

```text
images/train
labels/train
images/val
labels/val
images/test
labels/test
```

Each label file must have the same base name as the image and contain YOLO boxes:

```text
class_id x_center y_center width height
```

For this project:

```text
0 = shoe
1 = sandal
2 = other_footwear
```

Attendance policy:

```text
shoe + face -> present
sandal + face -> present_with_flag
other_footwear or no detection + face -> absent
```
""",
            encoding="utf-8",
        )

    print(f"YOLO footwear dataset prepared: {dataset_dir}")
    print(f"Data file: {data_yaml}")


if __name__ == "__main__":
    main()
