import gdown
import os
from tensorflow.keras.models import load_model

MODEL_PATH = "resnet50_finetuned.h5"
FILE_ID = "1GPYJZLF87XS-J2KrnGmxMdxiLIR8rP18"
GDRIVE_URL = f"https://drive.google.com/uc?id={FILE_ID}"

if not os.path.exists(MODEL_PATH):
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH)



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
