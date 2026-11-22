import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras

MODEL_PATH = "house_mobilenet.keras"
IMG_SIZE = (150, 150)
CLASS_NAMES = ["bathroom", "bedroom", "dining", "gaming", "kitchen", "laundry", "living", "terrace"]

st.set_page_config(page_title="House Room Classifier", page_icon="🏠")

@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)

st.title("🏠 House Room Classifier")
uploaded = st.file_uploader("Choose a room photo", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, use_container_width=True)
    x = np.expand_dims(np.array(img.resize(IMG_SIZE), dtype="float32"), axis=0)
    probs = load_model().predict(x, verbose=0)[0]
    best = int(np.argmax(probs))
    st.subheader(f"Prediction: {CLASS_NAMES[best]} ({probs[best]:.1%})")
    st.bar_chart({n: float(p) for n, p in zip(CLASS_NAMES, probs)})
