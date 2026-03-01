import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import gdown
import os

st.set_page_config(page_title="Rice Leaf Disease Detection", layout="centered")
st.title("Rice Leaf Disease Detection 🌾")

MODEL_PATH = "resnet50_finetuned.h5"

# ✅ NEW WORKING FILE ID
FILE_ID = "1qAZdvwPnxjovdfirN4MkYKotXi9s415e"

GDRIVE_URL = f"https://drive.google.com/uc?id={FILE_ID}"


# ✅ MODEL LOADER (PRODUCTION SAFE)
@st.cache_resource
def load_my_model():

    if not os.path.exists(MODEL_PATH):
        try:
            with st.spinner("🔽 Downloading model... Please wait once"):
                gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
        except Exception as e:
            st.error("❌ Model download failed. Check Google Drive link & permission.")
            st.stop()

    return load_model(MODEL_PATH)


model = load_my_model()

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
    st.image(image, caption="Uploaded Image", width=400)

    img = image.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    if st.button("Predict"):

        pred = model.predict(img)
        idx = np.argmax(pred)
        predicted_class = CLASSES[idx]

        st.success(f"Prediction: {predicted_class}")
        st.info(f"Confidence: {pred[0][idx]*100:.2f}%")

        recommendations = {
            "Brown Spot": {
                "DO": "Spray Mancozeb, use proper fertilizer",
                "DONT": "Do not keep field dry",
                "Severity": "Medium",
                "Action": "Spray within 5 days",
                "Safety": "Wear mask and gloves"
            },
            "Leaf Blast": {
                "DO": "Spray Tricyclazole, keep spacing",
                "DONT": "Do not use excess urea",
                "Severity": "High",
                "Action": "Spray immediately",
                "Safety": "Avoid spraying during rain"
            },
            "Leaf Scald": {
                "DO": "Spray Carbendazim, clean field",
                "DONT": "Do not reuse infected seeds",
                "Severity": "Medium",
                "Action": "Spray within 3–5 days",
                "Safety": "Wash hands after spraying"
            },
            "Sheath Blight": {
                "DO": "Spray Validamycin, remove infected plants",
                "DONT": "Do not plant too close",
                "Severity": "High",
                "Action": "Spray immediately",
                "Safety": "Keep chemicals away from children"
            }
        }

        if predicted_class in recommendations:
            rec = recommendations[predicted_class]
            st.subheader("🌾 Disease Recommendation")
            st.write("**DO:**", rec["DO"])
            st.write("**DON’T:**", rec["DONT"])
            st.write("**Severity:**", rec["Severity"])
            st.write("**When to act:**", rec["Action"])
            st.write("**Safety tip:**", rec["Safety"])