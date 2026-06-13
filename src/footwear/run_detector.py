from __future__ import annotations

import argparse

import cv2

from src.footwear.detector import FootwearDetector, draw_detections


def main() -> None:
    parser = argparse.ArgumentParser(description="Test footwear detection on image, video, or webcam.")
    parser.add_argument(
        "--model",
        default="models/footwear/shoe_detect_best.pt",
        help="YOLO .pt model file or exported NCNN model folder.",
    )
    parser.add_argument("--source", default="0", help="Image/video path or camera index.")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--no-display", action="store_true", help="Print detections without opening an OpenCV window.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many video/camera frames. 0 means no limit.")
    args = parser.parse_args()

    detector = FootwearDetector(args.model, args.conf)
    source = int(args.source) if args.source.isdigit() else args.source

    if isinstance(source, str):
        image = cv2.imread(source)
        if image is not None:
            detections = detector.detect(image)
            print(f"Detections: {detections}")
            if args.no_display:
                return
            output = draw_detections(image, detections)
            cv2.imshow("Footwear Detection", output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")
    if not args.no_display:
        print("Press ESC to quit.")
    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        detections = detector.detect(frame)
        frame_count += 1
        if args.no_display:
            print(f"frame={frame_count} detections={detections}")
            if args.max_frames and frame_count >= args.max_frames:
                break
            continue
        cv2.imshow("Footwear Detection", draw_detections(frame, detections))
        if args.max_frames and frame_count >= args.max_frames:
            break
        if (cv2.waitKey(1) & 0xFF) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
