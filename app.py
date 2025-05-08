import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import numpy as np
from inference import (
    load_model, predict_digit,
    load_knn_model, predict_knn_digit,
    load_cnn_model, predict_cnn_digit,
    load_clip_model, predict_with_clip
)
import matplotlib.pyplot as plt

st.set_page_config(page_title="Digit Recognizer")
st.title("🧠 Handwritten Digit Recognizer")

# Load models
model = load_model()
knn_model = load_knn_model()
cnn_model = load_cnn_model()
clip_model, clip_processor = load_clip_model()

# Layout: canvas on left, predictions on right
canvas_col, prediction_col = st.columns([1.5, 1])

# Sidebar model toggle
st.sidebar.markdown("### Show Models:")
show_model_a = st.sidebar.checkbox("CNN Model (Blue)", value=True)
show_model_b = st.sidebar.checkbox("KNN Model (Red)", value=True)
show_model_c = st.sidebar.checkbox("DNN Model (Green)", value=True)
show_model_d = st.sidebar.checkbox("Model D (Purple)", value=True)

# Canvas
with canvas_col:
    st.markdown("#### Draw a digit (-1 to 9):")
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=30,
        stroke_color="black",
        background_color="white",
        width=400,
        height=400,
        drawing_mode="freedraw",
        key="canvas",
    )
    predict_clicked = st.button("Predict")

# Plotting
def plot_confidence(probs, title, color):
    fig, ax = plt.subplots(figsize=(5, 1.5))
    ax.bar(range(-1, 10), probs, color=color)
    ax.set_xticks(range(-1, 10))
    ax.set_xticklabels(["NA"] + list(map(str, range(10))))
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Digit")
    ax.set_title(title, fontsize=10)
    st.pyplot(fig)

# Prediction column
with prediction_col:
    st.markdown("#### Models' Predictions:")

    if "prev_probs" not in st.session_state:
        st.session_state.prev_probs = np.zeros(11)

    if canvas_result.image_data is not None:
        image = Image.fromarray((canvas_result.image_data[:, :, 0]).astype(np.uint8))

        if show_model_a:
            pred_cnn, probs_cnn = predict_cnn_digit(image, cnn_model)
            plot_confidence(probs_cnn, title="CNN", color="blue")

        if show_model_b:
            pred_knn, probs_knn = predict_knn_digit(image, knn_model)
            plot_confidence(probs_knn, title="KNN", color="red")

        if show_model_c:
            pred_dnn, probs_dnn = predict_digit(image, model)
            padded_probs_dnn = np.insert(probs_dnn, 0, 0.0)
            plot_confidence(padded_probs_dnn, title="DNN", color="green")

        if show_model_d:
            probs_clip = predict_with_clip(image, clip_model, clip_processor)
            plot_confidence(probs_clip, title="ViT (CLIP)", color="purple")

    else:
        st.markdown("Waiting for input...")