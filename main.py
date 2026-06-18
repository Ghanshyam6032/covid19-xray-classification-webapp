from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import io
import tensorflow as tf

app = FastAPI()

# Load the Keras model (ensure 'covid_mobilenetv2.h5' is in the same directory or provide full path)
try:
    model = tf.keras.models.load_model("covid_mobilenetv2.h5")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    # Optionally, you might want to exit or handle this more gracefully in a production app

# Define class names (ensure these match the order used during training)
CLASS_NAMES = ["Covid", "Normal", "Viral Pneumonia"]
TARGET_SIZE = (224, 224) # The input size the model expects

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # Read the image file
    contents = await file.read()
    # Open image with PIL and convert to RGB (important for models expecting 3 channels)
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Preprocess the image
    image = image.resize(TARGET_SIZE) # Resize to target size
    img_array = np.array(image) / 255.0  # Convert to numpy array and normalize pixel values
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension (1, 224, 224, 3)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence = float(np.max(predictions[0])) # Convert numpy float to Python float

    return {
        "predicted_class": predicted_class_name,
        "confidence": confidence,
        "probabilities": predictions[0].tolist() # Convert numpy array to list for JSON serialization
    }