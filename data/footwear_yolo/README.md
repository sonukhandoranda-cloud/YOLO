# Footwear YOLO Dataset

Place images and YOLO label files here:

```text
images/train
labels/train
images/val
labels/val
images/test
labels/test
```

Each label file must have the same base name as the image and contain YOLO boxes:

```text
class_id x_center y_center width height
```

For this project:

```text
0 = shoe
1 = sandal
2 = other_footwear
```

Attendance policy:

```text
shoe + face -> present
sandal + face -> present_with_flag
other_footwear or no detection + face -> absent
```
