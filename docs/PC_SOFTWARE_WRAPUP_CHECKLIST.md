# PC Software Wrap-Up Checklist

Use this before moving to Raspberry Pi work.

## Must Pass

```bash
python scripts/software_readiness_check.py
```

## Face Side

- [ ] Enroll one person from camera.
- [ ] Try enrolling same face under a different roll number.
- [ ] Confirm duplicate enrollment is blocked.
- [ ] Run face attendance.
- [ ] Confirm duplicate attendance is not created.
- [ ] Export Excel report.

## Footwear Side

- [ ] Run current model evaluation.
- [ ] Prepare YOLO dataset folder.
- [ ] Collect/labeled footwear dataset.
- [ ] Train custom YOLO if labels are ready.
- [ ] Export trained YOLO for Raspberry Pi.

## Reporting

- [ ] Excel export works.
- [ ] Email settings documented.
- [ ] SMTP credentials are not committed.

## Raspberry Pi Readiness

- [ ] GitHub repository is pushed.
- [ ] Pi deployment docs exist.
- [ ] Model export command exists.
- [ ] Camera test command exists.

## Current Known Gap

The only major PC software gap is the custom footwear dataset/model. The code path is ready; labeled footwear data is still needed.
