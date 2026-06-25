
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="YOLOv8 Object Recognition", page_icon="🤖")

st.title("🤖 YOLOv8 Object Recognition")
st.write("Upload an image or video, or use a local webcam script.")

model_path = st.text_input("Model path or name", "yolov8n.pt")
conf = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
iou = st.slider("IoU threshold", 0.0, 1.0, 0.45, 0.01)
imgsz = st.number_input("Image size", 320, 1920, 640, step=32)

uploaded = st.file_uploader("Upload image or video", type=["jpg","jpeg","png","mp4","avi","mov","mkv"])

if uploaded and st.button("Run detection"):
    model = YOLO(model_path)
    suffix = os.path.splitext(uploaded.name)[1].lower()

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, uploaded.name)
        with open(src_path, "wb") as f:
            f.write(uploaded.getbuffer())

        results = model.predict(source=src_path, imgsz=imgsz, conf=conf, iou=iou, save=True, verbose=False)

        save_dir = results[0].save_dir if hasattr(results[0], "save_dir") else None
        if save_dir is None:
            st.error("Could not find saved results directory.")
        else:
            if suffix in [".jpg", ".jpeg", ".png"]:
                annotated = os.path.join(save_dir, uploaded.name)
                if os.path.exists(annotated):
                    st.image(Image.open(annotated), caption="Detections", use_container_width=True)
                else:
                    files = [f for f in os.listdir(save_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
                    if files:
                        st.image(Image.open(os.path.join(save_dir, files[0])), caption="Detections", use_container_width=True)
                    else:
                        st.info(f"Processed. Check output folder: {save_dir}")
            else:
                st.success(f"Video processed. Download from: {save_dir}")
                st.write(f"Output folder: `{save_dir}`")
