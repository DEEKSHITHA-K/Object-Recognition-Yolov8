
import argparse
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8 file detection (image/video)")
    p.add_argument("--model", type=str, default="yolov8n.pt", help="Path to model weights or model name")
    p.add_argument("--source", type=str, required=True, help="Path to an image or video file")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--save", action="store_true", help="Save results to runs/detect/predict*")
    return p.parse_args()

def main():
    args = parse_args()
    model = YOLO(args.model)
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        save=args.save or True,  # default to save results for files
        verbose=False
    )

if __name__ == "__main__":
    main()
