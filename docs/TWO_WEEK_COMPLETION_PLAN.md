# Two-Week Completion Plan

## Goal

Complete the PC software side this week, then use the next week for Raspberry Pi deployment and hardware testing.

Before hardware arrives, run:

```bash
python scripts/software_readiness_check.py
```

## Current Status

Face side:

```text
1500+ scale readiness: PASS
Public 1500-identity benchmark: 95%+ range after tuning
Private difficult-image validation: 20/20
Duplicate attendance prevention: PASS
Daily Excel export: implemented
```

Footwear side:

```text
Existing shoe model loads successfully
Current sample detection result: 12/15 images detected
Detection rate: 80.00%
YOLO training dataset scaffold: created
Custom labeled footwear dataset: not ready yet
```

## Week 1: Finish PC Software

### Day 1

- Prepare YOLO footwear dataset structure.
- Validate current shoe model baseline.
- Decide final footwear class design.

Recommended first class design:

```text
0: shoe
1: sandal
2: other_footwear
```

Reason:

The second camera must support factory-style footwear policy:

```text
shoe + face -> present
sandal + face -> present_with_flag
other footwear + face -> absent
```

### Day 2

- Collect 100 to 200 footwear images.
- Include:
  - sneakers,
  - slippers,
  - sandals,
  - formal shoes,
  - partial shoes,
  - low light,
  - motion blur,
  - floor/background variation,
  - low camera angle.

### Day 3

- Label images in Roboflow or another YOLO labeling tool.
- Use classes:

```text
0: shoe
1: sandal
2: other_footwear
```

- Export dataset in YOLO format.
- Place files in:

```text
data/footwear_yolo/images/train
data/footwear_yolo/labels/train
data/footwear_yolo/images/val
data/footwear_yolo/labels/val
data/footwear_yolo/images/test
data/footwear_yolo/labels/test
```

### Day 4

- Train/fine-tune YOLO footwear model.
- Validate precision, recall, and mAP.
- Compare against existing shoe model.

### Day 5

- Select best footwear model.
- Run two-camera software flow if two USB webcams are available.
- Export daily attendance Excel.
- Freeze PC prototype for Raspberry Pi migration.

## Week 2: Raspberry Pi Deployment

### Day 1

- Install OS and dependencies on Raspberry Pi.
- Clone GitHub repo.
- Install Python dependencies.
- Test one camera.

### Day 2

- Test two cameras.
- Confirm camera indexes or Picamera2 device mapping.
- Run face-only test.

### Day 3

- Run footwear model.
- Benchmark FPS and CPU/RAM usage.
- Reduce resolution if needed.

### Day 4

- Run full two-camera attendance.
- Test duplicate prevention.
- Test Excel export.

### Day 5

- Prepare final demo.
- Record limitations.
- Prepare final deployment notes.

## Commands

Prepare footwear dataset:

```bash
python scripts/prepare_footwear_dataset.py
```

Validate footwear dataset:

```bash
python scripts/validate_yolo_dataset.py
```

Train YOLO footwear model:

```bash
python scripts/train_footwear_yolo.py --data data/footwear_yolo/data.yaml --model yolo11n.pt --epochs 50 --imgsz 640
```

Evaluate current shoe model:

```bash
python scripts/evaluate_footwear_images.py --images data/raw/footwear_samples --model models/footwear/shoe_detect_best.pt --conf 0.50
```

Export trained footwear model for Raspberry Pi:

```bash
python scripts/export_footwear_model_for_pi.py --model runs/footwear/yolo_footwear/weights/best.pt --format ncnn
```

## Definition Of Done Before Raspberry Pi

PC software is ready when:

- face recognition works,
- footwear detector has a measured baseline,
- attendance fusion script runs,
- duplicate prevention works,
- Excel export works,
- dataset/training scripts are ready,
- Raspberry Pi export path is documented.

## Current Risk

The biggest remaining risk is footwear detection accuracy in real camera conditions.

Mitigation:

- collect low-angle footwear images,
- label properly,
- train YOLO,
- test on images not used in training,
- export the smallest accurate model for Raspberry Pi.
