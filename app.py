import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
from inference import load_model, predict_digit

st.set_page_config(page_title="Digit Recognizer")
st.title("🧠 Handwritten Digit Recognizer")
st.markdown("Draw a digit (0-9) below:")

# Canvas settings
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=12,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Extract grayscale image
        image = Image.fromarray((canvas_result.image_data[:, :, 0]).astype(np.uint8))
        model = load_model()
        prediction = predict_digit(image, model)
        st.write(f"### Predicted Digit: {prediction}")