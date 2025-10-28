import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from src.utils import class_names

# Load model
model = tf.keras.models.load_model("saved_model/cifar10_model.h5")

st.set_page_config(page_title="CIFAR-10 Classifier", page_icon="🖼️", layout="centered")
st.title("CIFAR-10 Image Classifier")

uploaded_file = st.file_uploader("Upload a 32x32 color image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB").resize((32, 32))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 32, 32, 3)

    st.image(image, caption="Uploaded Image", width=150)

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction)

    st.subheader(f"Prediction: {class_names[class_idx]}")
    st.caption(f"Confidence: {confidence:.2%}")
