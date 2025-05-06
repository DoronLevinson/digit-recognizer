import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
from inference import load_model, predict_digit
import matplotlib.pyplot as plt

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

def plot_confidence(probs):
    fig, ax = plt.subplots()
    ax.bar(range(10), probs)
    ax.set_xticks(range(10))
    ax.set_xlabel("Digit")
    ax.set_ylabel("Confidence")
    ax.set_title("Prediction Confidence")
    st.pyplot(fig)

if st.button("Predict"):
    if canvas_result.image_data is not None:
        image = Image.fromarray((canvas_result.image_data[:, :, 0]).astype(np.uint8))
        model = load_model()
        prediction, probs = predict_digit(image, model)

        st.markdown(f"### 🧮 Predicted Digit: `{prediction}` with {100*probs[prediction]:.2f}% confidence")
        plot_confidence(probs)