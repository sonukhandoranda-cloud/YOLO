# Next Steps

## Immediate Demo Path

1. Initialize the database:

   ```bash
   python -m src.database.init_db
   ```

2. Enroll a student from an image:

   ```bash
   python -m src.app.enroll_student --roll-no 101 --name "Test Student" --branch CSE --batch 2026 --image path/to/student.jpg
   ```

3. Test the shoe detector:

   ```bash
   python -m src.footwear.run_detector --model models/footwear/shoe_detect_best.pt --source 0
   ```

4. Run the two-camera prototype:

   ```bash
   python -m src.app.run_attendance --face-camera 0 --footwear-camera 1
   ```

## This Week's Engineering Checks

1. Check the project health:

   ```bash
   python -m src.app.doctor --check-cameras 1
   ```

2. Run face-only attendance:

   ```bash
   python -m src.app.test_face_camera --camera 0
   ```

3. Show saved attendance:

   ```bash
   python -m src.app.show_attendance
   ```

4. Show database status:

   ```bash
   python -m src.app.project_status
   ```

5. Run the 1500-student scale simulation:

   ```bash
   python scripts/simulate_1500_students.py
   ```

The simulation does not prove real-world accuracy. It proves that the database and matching pipeline can handle college-scale enrollment counts without needing 1500 real people at home.

## Accuracy Improvements Added

The face matcher now uses two checks:

- score threshold: the best match must be strong enough
- margin threshold: the best match must be clearly better than the second-best match

This reduces false accepts when two enrolled faces look similar or when the database becomes large.

## Performance Improvement Added

The live attendance runners now load all face embeddings into an in-memory `FaceIndex` at startup. This avoids parsing the SQLite database on every video frame.

In the latest simulation:

```text
Students: 1500
Queries: 100
Correct: 100
Average match time: about 0.12 ms
```

This is the correct pattern for Raspberry Pi too: load the enrolled embeddings once, then perform fast in-memory matching during camera inference.

## Important Accuracy Note

The current OpenCV face engine is a wiring baseline. It proves the enrollment, database, matching, and attendance flow. For the final teacher-facing accuracy target, replace `src/face/opencv_embedder.py` with a stronger ArcFace or InsightFace embedding engine.

## Dataset Work

- Collect at least 3 to 5 face images for early test students.
- Capture checkpoint-like test photos, not only clean passport-style images.
- Use Roboflow to label footwear examples with the exact camera angle planned for the second camera.
- Keep train, validation, and test sets separate.
