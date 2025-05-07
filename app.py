import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
from inference import load_model, predict_digit
import matplotlib.pyplot as plt

st.set_page_config(page_title="Digit Recognizer")
st.title("🧠 Handwritten Digit Recognizer")

model = load_model()

# Layout: selector | canvas | predictions
canvas_col, prediction_col = st.columns([1.5, 1])

# Sidebar for model selection
st.sidebar.markdown("### Show Models:")
show_model_a = st.sidebar.checkbox("Model A (Blue)", value=True)
show_model_b = st.sidebar.checkbox("Model B (Red)", value=True)
show_model_c = st.sidebar.checkbox("Model C (Green)", value=True)

# Center – drawing canvas
with canvas_col:
    st.markdown("#### Draw a digit (0–9) below:")
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=60,
        stroke_color="black",
        background_color="white",
        width=400,
        height=400,
        drawing_mode="freedraw",
        key="canvas",
    )
    predict_clicked = st.button("Predict")

# Right – prediction charts
def plot_confidence(probs, title, color):
    fig, ax = plt.subplots(figsize=(5, 1.5))
    ax.bar(range(10), probs, color=color)
    ax.set_xticks(range(10))
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Digit")
    ax.set_title(title, fontsize=10)
    st.pyplot(fig)

with prediction_col:
    st.markdown("#### Model Predictions:")

    if canvas_result.image_data is not None:
        image = Image.fromarray((canvas_result.image_data[:, :, 0]).astype(np.uint8))
        prediction, probs = predict_digit(image, model)

        # For now, show all models (checkbox values unused)
        plot_confidence(probs, title="Model A (Blue) Prediction Confidence", color="blue")
        plot_confidence(probs, title="Model B (Red) Prediction Confidence", color="red")
        plot_confidence(probs, title="Model C (Green) Prediction Confidence", color="green")
    else:
        st.markdown("Waiting for input...")