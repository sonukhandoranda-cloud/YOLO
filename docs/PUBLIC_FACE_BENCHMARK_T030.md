# Public Face Benchmark Report

## Dataset

Public LFW benchmark loaded through scikit-learn. Each identity is treated as a simulated student.

This is not a real college-student dataset. It is used only to test the current face-recognition pipeline before real enrollment data exists.

## Setup

```text
Requested enrolled identities: 50
Loaded enrolled identities: 50
Unknown identities reserved: 20
Threshold: 0.3
Minimum top-2 margin: 0.03
```

## Result

```text
Enrollment success: 50/50
Enrollment failures: 0
Genuine query face-detection failures: 0
Genuine attempted matches: 50
Genuine correct: 49
Genuine missed: 1
Genuine wrong identity: 0
Measured genuine accuracy: 98.00%

Unknown tested: 20
Unknown rejected: 19
Unknown false accepts: 1
Unknown detection failures: 0
Measured false accept rate: 5.00%

Average in-memory match time: 0.1254 ms
P95 in-memory match time: 0.1623 ms
```

## Output Files

```text
data\processed\public_face_benchmark_t030.csv
data/processed/public_face_pairs
```

## Interpretation

This benchmark tests the current face-recognition pipeline with public face images. It is useful, but it is not the final accuracy claim for the college deployment.

Final accuracy must still be measured using real enrollment-style images and real camera-condition images.
