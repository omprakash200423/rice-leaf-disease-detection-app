import streamlit as st
import os
import gdown
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

MODEL_PATH = "resnet50_finetuned.h5"
GDRIVE_URL = "https://drive.google.com/uc?id=1GPYJZLF87XS-J2KrnGmxMdxiLIR8rP18"

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model... please wait"):
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

# NOW load the model (after download)
model = load_model(MODEL_PATH)


CLASSES = [
    "Brown Spot",
    "Healthy",
    "Leaf Blast",
    "Leaf Scald",
    "Sheath Blight"
]

uploaded_file = st.file_uploader("Upload a rice leaf image", type=["jpg","jpeg","png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224,224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    if st.button("Predict"):
        pred = model.predict(img)
        idx = np.argmax(pred)
        st.success(f"**Prediction:** {CLASSES[idx]}")
        st.info(f"**Confidence:** {pred[0][idx]*100:.2f}%")
