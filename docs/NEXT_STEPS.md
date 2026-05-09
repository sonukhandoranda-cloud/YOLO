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

## Important Accuracy Note

The current OpenCV face engine is a wiring baseline. It proves the enrollment, database, matching, and attendance flow. For the final teacher-facing accuracy target, replace `src/face/opencv_embedder.py` with a stronger ArcFace or InsightFace embedding engine.

## Dataset Work

- Collect at least 3 to 5 face images for early test students.
- Capture checkpoint-like test photos, not only clean passport-style images.
- Use Roboflow to label footwear examples with the exact camera angle planned for the second camera.
- Keep train, validation, and test sets separate.
