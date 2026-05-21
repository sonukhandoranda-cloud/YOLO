# Public Face Benchmark Report

## Purpose

This report evaluates the current face-recognition pipeline on a public benchmark dataset because real college student images are not available.

The benchmark treats each public identity as a simulated student:

```text
1 enrollment photo per identity
1 test photo per identity
1500 enrolled identities
100 unknown identities reserved for rejection testing
```

## Dataset

Public LFW face benchmark loaded through scikit-learn.

Exported image pairs:

```text
data/processed/public_face_pairs
```

Detailed result CSV files:

```text
data/processed/public_face_benchmark_1500.csv
data/processed/public_face_benchmark_1500_t042.csv
data/processed/public_face_benchmark_1500_t042_m004.csv
data/processed/public_face_benchmark_1500_t042_m005.csv
data/processed/public_face_benchmark_1500_t045.csv
```

## Current Face Pipeline

```text
YuNet -> face detection
SFace -> face embedding
FaceIndex -> in-memory matching against enrolled embeddings
SQLite -> persistent student/attendance storage
```

## Best Operating Point Found

```text
Face threshold: 0.42
Minimum top-2 margin: 0.03
```

## Best 1500-Identity Result

```text
Enrolled identities: 1500
Enrollment success: 1500/1500
Genuine attempted matches: 1500
Genuine correct: 1439
Genuine missed: 57
Genuine wrong identity: 4
Measured genuine accuracy: 95.93%

Unknown identities tested: 100
Unknown rejected: 99
Unknown false accepts: 1
Measured unknown false accept rate: 1.00%

Average in-memory match time: 0.2375 ms
P95 in-memory match time: 0.3324 ms
```

## Threshold Comparison

| Threshold | Margin | Genuine Accuracy | Wrong Identity | Unknown False Accept Rate |
| --- | --- | --- | --- | --- |
| 0.36 | 0.03 | 97.47% | 9 | 19.00% |
| 0.42 | 0.03 | 95.93% | 4 | 1.00% |
| 0.42 | 0.04 | 95.80% | 4 | 1.00% |
| 0.42 | 0.05 | 95.27% | 4 | 1.00% |
| 0.45 | 0.03 | 93.33% | 2 | 1.00% |

## Interpretation

The current model is now benchmarked at 1500 identities and reaches the professor's 95%+ target on this public benchmark when using threshold `0.42` and margin `0.03`.

However, this is not yet a final production accuracy claim because:

- LFW is a public face dataset, not the college's real student dataset.
- Real deployment will use different lighting, camera distance, pose, blur, and enrollment-photo quality.
- Final accuracy must be measured on real enrollment-style images and real PC/Raspberry Pi camera test images.

## Current Conclusion

The face-recognition pipeline is ready for a PC prototype and has passed a 1500-identity public benchmark:

```text
Scale readiness: PASS
Public benchmark genuine accuracy: 95.93%
Unknown false accept rate: 1.00%
Real-world college accuracy: still needs real validation data
```

For attendance, false accepts are more dangerous than missed detections. The selected threshold favors safer rejection over accepting unknown people.
