# 1500+ Student Scale Readiness Report

## Question

Can the current project handle 1500+ enrolled students?

## Short Answer

Yes, the current matching architecture is ready for 1500+ enrolled students from a performance and system-design point of view.

This does not mean real-world accuracy is fully proven. Real-world accuracy still requires real enrollment images, camera-angle testing, lighting tests, and shoe/presence validation.

## What Was Tested

The scale validator creates simulated student records and embeddings, loads them into the same SQLite database structure, builds the in-memory face index, and tests:

- genuine student matching,
- unknown/impostor rejection,
- ambiguous/lookalike rejection,
- matching speed.

Command:

```bash
python scripts/validate_scale_readiness.py --students 1500
```

## 1500-Student Result

```text
Students enrolled: 1500
Database build time: 0.64s
Face index load time: 0.201s
Genuine queries: 300
Genuine correct: 300
Genuine missed: 0
Synthetic genuine accuracy: 100.00%
Impostor queries: 300
Impostor false accepts: 0
Synthetic false accept rate: 0.00%
Ambiguous pairs tested: 30
Ambiguous rejected: 30
Ambiguous reject rate: 100.00%
Average match time: 0.0874 ms
P95 match time: 0.1492 ms
Max match time: 3.8700 ms
Verdict: PASS
```

## 5000-Student Stress Result

```text
Students enrolled: 5000
Database build time: 1.96s
Face index load time: 0.625s
Genuine queries: 300
Genuine correct: 300
Genuine missed: 0
Synthetic genuine accuracy: 100.00%
Impostor queries: 300
Impostor false accepts: 0
Synthetic false accept rate: 0.00%
Ambiguous pairs tested: 30
Ambiguous rejected: 30
Ambiguous reject rate: 100.00%
Average match time: 0.3336 ms
P95 match time: 0.4824 ms
Max match time: 19.6218 ms
Verdict: PASS
```

## Why It Is Fast

SQLite stores the official student records and face embeddings. During live recognition, the system loads those embeddings once into an in-memory `FaceIndex`.

Live matching then uses NumPy vector operations instead of repeatedly reading and parsing the database for every video frame.

This design is suitable for Raspberry Pi because:

- the database remains simple and local,
- embeddings are small,
- matching is fast,
- the heavier work remains camera inference, not student lookup.

## What This Does Not Prove

This scale test does not prove final real-world recognition accuracy. It uses synthetic vectors to test system scale and matching behavior.

To prove high accuracy, the project still needs:

- real enrolled student images or embeddings,
- test images from the expected camera setup,
- lighting variation tests,
- face pose and hairstyle variation tests,
- false accept / false reject measurement,
- shoe detection tests from the final low camera angle.

## Current Conclusion

The system is ready for 1500+ enrolled students architecturally.

The system is not yet fully trained/validated for high real-world accuracy because the required real validation dataset has not been collected yet.
