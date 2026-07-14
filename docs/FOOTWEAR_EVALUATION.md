# Footwear Model Evaluation

## Purpose

This evaluation checks whether the provided YOLO footwear model can detect shoes/footwear on sample images before we move toward Raspberry Pi deployment.

## Current Model

```text
models/footwear/footwear_policy_yolo11n.pt
```

Loaded classes:

```text
0: shoe
1: sandal
2: other_footwear
```

The original senior-provided baseline remains available at:

```text
models/footwear/shoe_detect_best.pt
```

## Sample Data

Images were downloaded from Wikimedia Commons into:

```text
data/raw/footwear_samples
```

The sample contains shoes, sneakers, sandals, slippers/drawings, and a few difficult/non-ideal images.

## Command Used

```bash
python scripts/evaluate_footwear_images.py --images data/raw/footwear_samples --model models/footwear/shoe_detect_best.pt --conf 0.50
```

## Result

Original baseline on public footwear samples:

```text
Images read: 15/15
Images with detections: 12/15
Detection rate: 80.00%
```

New 3-class demo model training result:

```text
Validation images: 66
Validation boxes: 154
mAP50: 0.977
mAP50-95: 0.904
shoe mAP50: 0.941
sandal mAP50: 0.995
other_footwear mAP50: 0.995
```

Important: this score is from an augmented demo dataset made from limited public/sample images. It proves the software path and class policy. It is not the final real-world accuracy claim.

Output files:

```text
data/processed/footwear_eval.csv
data/processed/footwear_eval/
```

## Observation

The model detects many normal shoe/sneaker/sandal images with high confidence, often around 0.84 to 0.91.

It fails on some difficult images, including scenes where footwear is not clear, the subject is not actually footwear-focused, or the image is an artwork/indirect scene.

## Conclusion

The new 3-class model is now the default demo model because it can produce `shoe`, `sandal`, and `other_footwear` labels. The original one-class model is still useful as a fallback shoe detector.

Before Raspberry Pi deployment, the next footwear step should be:

1. collect 50 to 100 phone images from the expected low camera angle,
2. include sneakers, sandals, slippers, formal shoes, and partial occlusion,
3. run this same evaluation script,
4. decide whether to fine-tune YOLO on a custom Roboflow dataset.

## YOLO Training Scaffold

Dataset structure can be created with:

```bash
python scripts/prepare_footwear_dataset.py
```

Validate the dataset before training:

```bash
python scripts/validate_yolo_dataset.py
```

Train after labels are ready:

```bash
python scripts/train_footwear_yolo.py --data data/footwear_yolo/data.yaml --model yolo11n.pt --epochs 50 --imgsz 640
```

Export the current working baseline model for Raspberry Pi:

```bash
python scripts/export_footwear_model_for_pi.py --model models/footwear/footwear_policy_yolo11n.pt --format ncnn --imgsz 416
```
