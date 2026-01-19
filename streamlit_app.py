import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import gdown
import os

st.title("Rice Leaf Disease Detection 🌾")

MODEL_PATH = "resnet50_finetuned.h5"
FILE_ID = "1GPYJZLF87XS-J2KrnGmxMdxiLIR8rP18"
GDRIVE_URL = f"https://drive.google.com/uc?id={FILE_ID}"

# Download model once
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

# Load model AFTER download
model = load_model(MODEL_PATH)

CLASSES = [
    "Brown Spot",
    "Healthy",
    "Leaf Blast",
    "Leaf Scald",
    "Sheath Blight"
]

uploaded_file = st.file_uploader(
    "Upload a rice leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    if st.button("Predict"):
        pred = model.predict(img)
        idx = np.argmax(pred)

        st.success(f"Prediction: {CLASSES[idx]}")
        st.info(f"Confidence: {pred[0][idx]*100:.2f}%")
