import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("age_model.h5")

model = load_model()

st.title("Real-Time Age Prediction App")

option = st.radio("Choose input method:", ("Upload Image", "Use Webcam"))

image = None

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose a face image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)

else:
    camera_image = st.camera_input("Take a picture")
    if camera_image is not None:
        image = Image.open(camera_image)
        st.image(image, caption='Webcam Image', use_container_width=True)

if image is not None:
    img_resized = image.resize((32, 32))
    img_array = np.array(img_resized, dtype="float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0) 

    predicted_age = model.predict(img_array)[0][0]
    
    predicted_age = max(0.0, float(predicted_age))

    st.subheader("Prediction Result:")
    st.success(f"**Predicted Age:** {predicted_age:.1f} years old")