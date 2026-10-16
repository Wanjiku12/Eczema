
from pathlib import Path
import json

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Locate the model folder
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

MODEL_PATH = MODEL_DIR / "eczema_model.keras"
LABELS_PATH = MODEL_DIR / "class_names.json"


@st.cache_resource
def load_ai_model():
    """Load the AI model once and reuse it."""

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    with open(LABELS_PATH, "r") as file:
        class_names = json.load(file)

    if model.output_shape[-1] != len(class_names):
        raise ValueError(
            "Model output size does not match class labels."
        )

    return model, class_names


def predict_skin_condition(image):
    """Return the model's experimental prediction."""

    model, class_names = load_ai_model()

    # Convert image to RGB
    image = image.convert("RGB")

    # Resize to MobileNetV2 input size
    image = image.resize((224, 224))

    # Convert image to numerical array
    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # Add batch dimension: (1, 224, 224, 3)
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Do NOT divide by 255 here.
    # Preprocessing is already inside our model.

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(np.argmax(predictions))
    predicted_class = class_names[predicted_index]

    scores = {
        name: float(score)
        for name, score in zip(class_names, predictions)
    }

    return predicted_class, scores
