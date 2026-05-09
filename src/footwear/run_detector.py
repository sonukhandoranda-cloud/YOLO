from __future__ import annotations

import argparse

import cv2

from src.footwear.detector import FootwearDetector, draw_detections


def main() -> None:
    parser = argparse.ArgumentParser(description="Test footwear detection on image, video, or webcam.")
    parser.add_argument("--model", default="models/footwear/shoe_detect_best.pt")
    parser.add_argument("--source", default="0", help="Image/video path or camera index.")
    parser.add_argument("--conf", type=float, default=0.5)
    args = parser.parse_args()

    detector = FootwearDetector(args.model, args.conf)
    source = int(args.source) if args.source.isdigit() else args.source

    if isinstance(source, str):
        image = cv2.imread(source)
        if image is not None:
            detections = detector.detect(image)
            output = draw_detections(image, detections)
            print(f"Detections: {detections}")
            cv2.imshow("Footwear Detection", output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")
    print("Press ESC to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        detections = detector.detect(frame)
        cv2.imshow("Footwear Detection", draw_detections(frame, detections))
        if (cv2.waitKey(1) & 0xFF) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
