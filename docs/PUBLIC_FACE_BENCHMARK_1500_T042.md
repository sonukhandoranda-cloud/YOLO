# Public Face Benchmark Report

## Dataset

Public LFW benchmark loaded through scikit-learn. Each identity is treated as a simulated student.

This is not a real college-student dataset. It is used only to test the current face-recognition pipeline before real enrollment data exists.

## Setup

```text
Requested enrolled identities: 1500
Loaded enrolled identities: 1500
Unknown identities reserved: 100
Threshold: 0.42
Minimum top-2 margin: 0.03
```

## Result

```text
Enrollment success: 1500/1500
Enrollment failures: 0
Genuine query face-detection failures: 0
Genuine attempted matches: 1500
Genuine correct: 1439
Genuine missed: 57
Genuine wrong identity: 4
Measured genuine accuracy: 95.93%

Unknown tested: 100
Unknown rejected: 99
Unknown false accepts: 1
Unknown detection failures: 0
Measured false accept rate: 1.00%

Average in-memory match time: 0.2375 ms
P95 in-memory match time: 0.3324 ms
```

## Output Files

```text
data\processed\public_face_benchmark_1500_t042.csv
data/processed/public_face_pairs
```

## Interpretation

This benchmark tests the current face-recognition pipeline with public face images. It is useful, but it is not the final accuracy claim for the college deployment.

Final accuracy must still be measured using real enrollment-style images and real camera-condition images.
