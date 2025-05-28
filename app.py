import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

st.title("YOLOv8 Object Detection")
model = YOLO("yolov8n.pt")  # Auto-downloads model

uploaded_file = st.file_uploader("Upload image", type=["jpg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    results = model(np.array(image))
    st.image(results[0].plot(), caption="Detected Objects")