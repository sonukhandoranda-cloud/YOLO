# Beginner Guide

This guide assumes you do not know machine learning yet. Follow the steps in order.

## What This Project Does

The system has two jobs:

1. Recognize the student's face.
2. Confirm that a real person is standing there by detecting footwear from another camera.

Attendance is marked only when both parts are confident.

## First-Time Setup

Open PowerShell inside the project folder:

```powershell
cd C:\Users\BIT\Desktop\Attendence
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Initialize the database:

```powershell
python -m src.database.init_db
```

You can also double-click:

```text
init_database.bat
```

## Step 1: Check The Project

Run:

```powershell
python -m src.app.doctor
```

Or double-click:

```text
start_here.bat
```

Expected result:

```text
[OK] cv2
[OK] numpy
[OK] ultralytics
[OK] database
[OK] shoe model
```

Camera 1 may fail if you have only one webcam connected. That is okay for now.

## Step 2: Test Shoe Detection

Run:

```powershell
python -m src.footwear.run_detector --model models/footwear/shoe_detect_best.pt --source 0
```

Or double-click:

```text
test_shoe_camera.bat
```

Press `ESC` to close the camera window.

If your camera does not open, try:

```powershell
python -m src.footwear.run_detector --model models/footwear/shoe_detect_best.pt --source 1
```

## Step 3: Enroll One Student

Take a clear front-facing photo of one student and save it somewhere, for example:

```text
C:\Users\BIT\Desktop\student1.jpg
```

Then run:

```powershell
python -m src.app.enroll_student --roll-no 101 --name "Test Student" --branch CSE --batch 2026 --image C:\Users\BIT\Desktop\student1.jpg
```

If it says no face detected, use a clearer image with good lighting.

## Step 4: Test Face Recognition And Mark Attendance

After enrolling one face, run:

```powershell
python -m src.app.test_face_camera --camera 0
```

Or double-click:

```text
test_face_camera.bat
```

Press `ESC` to close the camera window.

If your face is recognized steadily, it will show:

```text
ATTENDANCE MARKED
```

To check saved attendance:

```powershell
python -m src.app.show_attendance
```

## Step 5: Run Attendance

If you have two cameras connected:

```powershell
python -m src.app.run_attendance --face-camera 0 --footwear-camera 1
```

Or double-click:

```text
run_attendance_two_cameras.bat
```

If you have only one camera, do not run the two-camera script yet. Test face enrollment and shoe detection separately.

## What To Show The Teacher First

For the first review, show:

- README architecture diagrams.
- Shoe model loading successfully.
- Shoe detection camera test.
- Student enrollment into database.
- Attendance database created.
- Explain that the current face model is a baseline and will be upgraded to ArcFace or InsightFace.

## Important Truth

High accuracy does not come from one magic command. It comes from:

- good student enrollment photos,
- good camera position,
- good lighting,
- testing many real cases,
- rejecting uncertain results,
- using a stronger face embedding model later.

The project is now structured so you can build these parts step by step.
