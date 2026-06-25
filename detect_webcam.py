
import argparse
import cv2
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8 webcam detection")
    p.add_argument("--model", type=str, default="yolov8n.pt", help="Path to model weights or model name")
    p.add_argument("--source", type=int, default=0, help="Webcam index (default: 0)")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--show_labels", action="store_true", help="Always draw labels (even when crowded)")
    return p.parse_args()

def main():
    args = parse_args()
    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {args.source}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = model.predict(source=frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False)
            annotated = results[0].plot(labels=True)
            cv2.imshow("Object Detection (ESC to quit)", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
