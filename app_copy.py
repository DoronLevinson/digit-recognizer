import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import numpy as np
import joblib

# Load model
knn_model = joblib.load("models/knn_model.joblib")

st.title("Digit Recognizer (KNN)")

# Create a canvas component
canvas_result = st_canvas(
    fill_color="black",  # Drawing color
    stroke_width=30,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Predict only when user clicks the button
if canvas_result.image_data is not None:
    if st.button("Predict Digit"):
        # Convert image to PIL, grayscale and resize
        image = Image.fromarray((canvas_result.image_data[:, :, 0]).astype(np.uint8))
        image = image.resize((28, 28)).convert("L")

        # Show processed image
        st.image(image, caption="Processed 28x28 Image", width=150)

        # Convert to numpy
        input_data = np.array(image).astype("float32").reshape(1, -1)
        st.write("Model input:", input_data)

        # Predict with KNN
        probs = knn_model.predict_proba(input_data)[0]
        pred = np.argmax(probs)

        # Show prediction
        st.subheader(f"Predicted digit: {pred}")
        st.text(f"Probabilities: {np.round(probs, 3)}")