
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import os
import gdown

app = FastAPI(
    title="COVID-19 Chest X-Ray Classification API",
    version="1.0"
)

# =========================
# CORS
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# MODEL DOWNLOAD
# =========================

MODEL_PATH = "covid_mobilenetv2.h5"

if not os.path.exists(MODEL_PATH):

    print("Downloading model from Google Drive...")

    file_id = "1IK0h4X6vfrL-uQtGqffnk7xqU2mfK6oT"

    url = f"https://drive.google.com/uc?id={file_id}"

    gdown.download(
        url,
        MODEL_PATH,
        quiet=False
    )

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

CLASS_NAMES = [
    "Covid",
    "Normal",
    "Viral Pneumonia"
]

TARGET_SIZE = (224, 224)

# =========================
# HEALTH CHECK
# =========================

@app.get("/")
async def root():
    return {
        "message": "COVID-19 API Running Successfully"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy"
    }

# =========================
# PREDICT
# =========================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()

    image = Image.open(
        io.BytesIO(contents)
    ).convert("RGB")

    image = image.resize(TARGET_SIZE)

    img_array = np.array(
        image,
        dtype=np.float32
    )

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    predictions = model.predict(
        img_array,
        verbose=0
    )

    predicted_index = int(
        np.argmax(predictions[0])
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        np.max(predictions[0])
    )

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": predictions[0].tolist()
    }

