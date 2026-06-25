
# Object Recognition Project (YOLOv8, Python)

A complete, ready-to-run object detection project using **YOLOv8** and **OpenCV**. It supports:
- Real-time detection from webcam
- Detection on images/videos
- Training on a custom dataset
- A simple Streamlit UI for uploads

## 1) Setup

```bash
# (Recommended) create venv
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

> If `torch` doesn't auto-install the right build for your GPU, follow: https://pytorch.org/get-started/locally/

## 2) Quick Start (Pretrained Inference)

Run webcam detection with the tiny YOLOv8n model:
```bash
python detect_webcam.py --model yolov8n.pt --source 0
```

Run on an image or video file:
```bash
# image
python detect_file.py --model yolov8n.pt --source examples/sample.jpg

# video
python detect_file.py --model yolov8n.pt --source examples/sample.mp4
```

> The first run will download weights automatically.

## 3) Streamlit App (Optional UI)

```bash
streamlit run app_streamlit.py
```
Then open the local URL shown in the terminal.

## 4) Train on Your Custom Dataset

Prepare a YOLO-format dataset with:
```
dataset/
  images/
    train/  # *.jpg / *.png
    val/
  labels/
    train/  # *.txt with YOLO bbox format
    val/
data.yaml   # paths + class names
```

Edit `data.yaml` (included) with absolute or relative paths and your class list, e.g.:
```yaml
path: ./dataset
train: images/train
val: images/val
names:
  0: person
  1: car
```

Start training:
```bash
python train.py --data data.yaml --epochs 50 --imgsz 640 --model yolov8n.pt
```

Resume training:
```bash
python train.py --data data.yaml --resume
```

## 5) Where Outputs Go

- Inference (file mode): `runs/detect/predict*`
- Webcam (live preview): annotated frames shown on screen; press `ESC` to quit
- Training: `runs/detect/train*` with best weights at `weights/best.pt`

## 6) Tips

- Use larger models for better accuracy: `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`
- For CPU-only systems, keep `--imgsz` small (e.g., 480) and use `yolov8n.pt`
- To improve FPS, set `--conf 0.35` and `--iou 0.5`

## 7) License

Code here is MIT; pretrained weights are per Ultralytics' terms.
