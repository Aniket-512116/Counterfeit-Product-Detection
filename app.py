import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.augmentations import letterbox

# Model Load
@st.cache_resource
def load_model():
    model_path = "best.pt"  # Update this with your best.pt path
    model = DetectMultiBackend(model_path, device="cpu")
    return model

model = load_model()

st.title("🖼 YOLOv5 Object Detection App")
st.write("Upload an image and detect objects using YOLOv5.")

# Upload Image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read Image
    image = Image.open(uploaded_file)
    img = np.array(image)

    # Preprocess Image
    img_resized = letterbox(img, 640, stride=32, auto=True)[0]
    img_resized = img_resized[:, :, ::-1].transpose(2, 0, 1)
    img_resized = np.ascontiguousarray(img_resized)

    # Convert to Torch Tensor
    img_tensor = torch.from_numpy(img_resized).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    # Model Inference
    pred = model(img_tensor)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    # Draw Bounding Boxes
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img.shape).round()
            for *xyxy, conf, cls in det:
                cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)

    # Display Result
    st.image(img, caption="Detected Image", use_column_width=True)
