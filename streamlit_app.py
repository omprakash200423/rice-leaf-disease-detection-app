import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# Page config
st.set_page_config(page_title="Rice Leaf Disease Detection")

st.title("🌾 Rice Leaf Disease Detection")
st.write("Upload a rice leaf image to predict the disease")

# Load model
model = load_model("resnet50_finetuned.h5")

CLASSES = [
    "Brown Spot",
    "Healthy",
    "Leaf Blast",
    "Leaf Scald",
    "Sheath Blight"
]

# Image upload
uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    if st.button("Predict"):
        pred = model.predict(img)
        idx = np.argmax(pred)

        st.success(f"🦠 Disease: **{CLASSES[idx]}**")
        st.info(f"📊 Confidence: **{pred[0][idx]*100:.2f}%**")
