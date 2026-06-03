# Raspberry Pi Deployment Plan

## Target Architecture

```text
Camera 1 -> face detection and recognition
Camera 2 -> footwear/presence detection
Decision engine -> attendance marking
SQLite -> local database
Excel export -> daily report
```

## Footwear Attendance Policy

The project is not limited to colleges. It can also support industry/factory rules where shoes are mandatory.

Current policy:

```text
face + shoe -> present
face + sandal -> present_with_flag
face + anything else -> absent / do not mark present
face + no footwear detection -> absent / do not mark present
```

The `present_with_flag` status is useful where sandals are allowed to enter but should be reported for safety/compliance review.

## Recommended Runtime

Start with:

```text
Python
OpenCV
SQLite
YuNet/SFace
YOLO NCNN or ONNX export
```

## Why Export YOLO

The `.pt` model is convenient on PC but may be slower on Raspberry Pi.

Preferred export order:

```text
NCNN first
ONNX second
TFLite if needed
```

## Raspberry Pi Checklist

1. Clone repository.
2. Install dependencies.
3. Copy trained/exported footwear model.
4. Copy face ONNX models.
5. Initialize database.
6. Test camera 0.
7. Test camera 1.
8. Test face recognition.
9. Test footwear detection.
10. Run full attendance flow.
11. Export Excel report.
12. Configure end-of-day email reporting.

## Performance Targets

Minimum acceptable prototype targets:

```text
Face recognition: stable enough for attendance decision
Footwear detection: stable across multiple frames
End-to-end decision latency: under 2 seconds
Duplicate prevention: no duplicate records per session
```

## If FPS Is Low

Try:

- reduce camera resolution,
- run inference every N frames,
- use YOLO nano model,
- export to NCNN,
- lower image size from 640 to 416,
- keep face embeddings loaded in memory.
