import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load trained model
model = tf.keras.models.load_model("brain_tumor_model.keras")

# Title
st.title("Brain Tumor MRI Classifier")

    # Upload image
uploaded_file = st.file_uploader("Upload an MRI scan image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Shape: (1, 128, 128, 3)

    # Predict
    prediction = model.predict(image_array)[0][0]
    
    if prediction > 0.5:
        st.error("🧠 Tumor Detected")
    else:
        st.success("✅ No Tumor Detected")

    st.image(image, caption="Uploaded Image", use_container_width=True)
