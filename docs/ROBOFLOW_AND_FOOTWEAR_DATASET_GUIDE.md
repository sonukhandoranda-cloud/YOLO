# Roboflow And Footwear Dataset Guide

## What Is Roboflow?

Roboflow is a web tool for object-detection datasets.

In simple words, it helps you:

- upload images,
- draw bounding boxes around objects,
- assign class names like `shoe` or `sandal`,
- split images into train/validation/test sets,
- export the dataset in YOLO format.

It does not magically train the final model by itself unless you use Roboflow training. For our project, we mainly need it to prepare a clean YOLO dataset.

## Why We Need It

YOLO cannot train properly from normal images alone. It needs image + label pairs.

For every image:

```text
images/train/example.jpg
labels/train/example.txt
```

The label file contains boxes:

```text
class_id x_center y_center width height
```

These values are normalized from 0 to 1.

## Our Class Plan

Use this for the project:

```text
0: shoe
1: sandal
2: other_footwear
```

Attendance policy:

```text
face + shoe -> present
face + sandal -> present_with_flag
face + other_footwear -> absent
face + no footwear -> absent
```

## Dataset Folder

The project already has a YOLO dataset scaffold:

```text
data/footwear_yolo/
  images/train/
  images/val/
  images/test/
  labels/train/
  labels/val/
  labels/test/
  data.yaml
```

Create it again if needed:

```bash
python scripts/prepare_footwear_dataset.py
```

Validate it:

```bash
python scripts/validate_yolo_dataset.py
```

## Recommended Dataset Strategy

For a good prototype, collect or label:

```text
100 to 200 images minimum
```

Include:

- shoes,
- sandals,
- slippers or other footwear,
- low camera angle,
- different floors,
- different lighting,
- partial occlusion,
- motion blur,
- both close and far views.

## Public Dataset Starting Point

Use Roboflow Universe and search for:

```text
shoe sandal detection
footwear detection
slipper sandal sneaker detection
```

A useful starting dataset found earlier:

```text
Shoe and Sandal Dataset - Roboflow Universe
```

If exported from Roboflow, choose:

```text
YOLOv8 / YOLO11 PyTorch format
```

Then copy/export the folders into `data/footwear_yolo`.

## Training Command

After labels are ready:

```bash
python scripts/train_footwear_yolo.py --data data/footwear_yolo/data.yaml --model yolo11n.pt --epochs 50 --imgsz 640
```

## Evaluation Command

```bash
yolo detect val model=runs/footwear/yolo_footwear/weights/best.pt data=data/footwear_yolo/data.yaml
```

## Raspberry Pi Export

```bash
python scripts/export_footwear_model_for_pi.py --model runs/footwear/yolo_footwear/weights/best.pt --format ncnn
```

## Important

Do not claim footwear production accuracy until the trained model is tested on images that were not used during training.
