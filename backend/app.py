from flask import Flask, request, render_template
import numpy as np
from io import BytesIO
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

app = Flask(__name__)

# Load trained model
model = load_model("resnet50_finetuned.h5")

# Class labels
CLASSES = [
    "Brown Spot",
    "Healthy",
    "Leaf Blast",
    "Leaf Scald",
    "Sheath Blight"
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]

    img = Image.open(BytesIO(file.read())).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    pred = model.predict(img)
    idx = np.argmax(pred)

    return render_template(
        "index.html",
        prediction=CLASSES[int(idx)],
        confidence=round(float(pred[0][idx]) * 100, 2)
    )

if __name__ == "__main__":
    app.run(debug=True)
