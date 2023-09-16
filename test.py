import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load pretrained models for age and gender detection
age_model_path = "path/to/age_model.h5"
gender_model_path = "path/to/gender_model.h5"

# Load the models
age_model = tf.keras.models.load_model(age_model_path)
gender_model = tf.keras.models.load_model(gender_model_path)

st.title("Age and Gender Detection")
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image:
    image = Image.open(uploaded_image)
    # Resize the image to the required input size for the models (e.g., 224x224)
    image = image.resize((224, 224))
    
    # Convert the PIL Image to a NumPy array
    image_np = np.array(image) / 255.0  # Normalize pixel values to [0, 1]

    # Perform age and gender detection on the uploaded image
    age_pred = age_model.predict(np.expand_dims(image_np, axis=0))[0]
    age_class = np.argmax(age_pred)
    age_label = str(age_class)

    gender_pred = gender_model.predict(np.expand_dims(image_np, axis=0))[0]
    gender_label = "Male" if gender_pred[0] >= 0.5 else "Female"

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Display age and gender predictions
    st.write(f"Predicted Age: {age_label} years")
    st.write(f"Predicted Gender: {gender_label}")
