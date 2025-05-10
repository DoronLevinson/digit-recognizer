import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import numpy as np
from inference import (
    load_dnn_model, predict_dnn_digit,
    load_knn_model, predict_knn_digit,
    load_cnn_model, predict_cnn_digit,
    predict_clip_digit
)
import matplotlib.pyplot as plt
import cv2
from models.vit_classifier_model import load_clip_digit_model



st.set_page_config(page_title="Digit Recognizer")

# Custom CSS to style trash icon grey
st.markdown("""
    <style>
    .toolbar button[title="Clear canvas"] svg {
        fill: grey !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Handwritten Digit Recognizer")
st.markdown("""
Welcome to Digit Recognizer!

This interactive demo showcases how machine learning models can be trained to recognize handwritten digits — even in noisy or unconventional forms.
The system is powered by multiple trained and fine-tuned models:
- A [Convolutional Neural Network (CNN)](https://en.wikipedia.org/wiki/Convolutional_neural_network), which uses spatial filters to detect digit features.
- A [Multi-Layer Perceptron (MLP)](https://en.wikipedia.org/wiki/Multilayer_perceptron), a fully connected neural network trained on flattened pixel inputs.
- A [k-Nearest Neighbors (KNN)](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) classifier that compares your drawing to similar examples in memory.
- A fine-tuned [Vision Transformer (ViT)](https://en.wikipedia.org/wiki/Vision_transformer), adapted from the [CLIP model](https://en.wikipedia.org/wiki/Contrastive_language%E2%80%93image_pre-training), which processes your image as a sequence of patches and reasons about it in context.

As you draw, each model provides a real-time prediction along with its confidence. If the models misclassify your input, you can provide feedback — your corrections will be used in the future to improve the system through continued learning.
The full source code is available on [GitHub](https://github.com/DoronLevinson/digit-recognizer).
""")
# Load models
dnn_model = load_dnn_model()
knn_model = load_knn_model()
cnn_model = load_cnn_model()
clip_digit_model = load_clip_digit_model("models/clip_digit_classifier.pth")

# Layout: canvas on left, predictions on right
canvas_col, prediction_col = st.columns([1.5, 1])

# Sidebar model toggle
st.sidebar.markdown("### Show Models:")
show_model_a = st.sidebar.checkbox("CNN Model (Blue)", value=True)
show_model_b = st.sidebar.checkbox("Fine-Tuned ViT Model (Purple)", value=True)
show_model_c = st.sidebar.checkbox("DNN Model (Red)", value=True)
show_model_d = st.sidebar.checkbox("KNN Model (Green)", value=False)


# Canvas
with canvas_col:
    st.markdown("##### Draw a digit (0 to 9):")
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=35,
        stroke_color="black",
        background_color="white",
        width=400,
        height=400,
        drawing_mode="freedraw",
        key="canvas",
    )

# Plotting
def plot_confidence(probs, title, color):
    fig, ax = plt.subplots(figsize=(5, 1.5))
    bar_colors = ['grey'] + [color] * 10
    ax.bar(range(-1, 10), probs, color=bar_colors)
    ax.set_xticks(range(-1, 10))
    ax.set_xticklabels(["NA"] + list(map(str, range(10))))
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Digit")
    ax.set_title(title, fontsize=10)
    st.pyplot(fig)

# Prediction column
with prediction_col:
    st.markdown("##### Models' Predictions:")

    if "prev_probs" not in st.session_state:
        st.session_state.prev_probs = np.zeros(11)

    if canvas_result.image_data is not None:
        image = Image.fromarray((canvas_result.image_data[:, :, 0]).astype(np.uint8))

        if show_model_a:
            pred_cnn, probs_cnn = predict_cnn_digit(image, cnn_model)
            plot_confidence(probs_cnn, title="CNN", color="blue")

        if show_model_b:
            pred_clip, probs_clip = predict_clip_digit(image, clip_digit_model)
            plot_confidence(probs_clip, title="Fine-Tuned ViT (CLIP)", color="purple")
        
        if show_model_c:
            pred_dnn, probs_dnn = predict_dnn_digit(image, dnn_model)
            plot_confidence(probs_dnn, title="MLP", color="red")
        
        if show_model_d:
            pred_knn, probs_knn = predict_knn_digit(image, knn_model)
            plot_confidence(probs_knn, title="KNN", color="green")

    else:
        st.markdown("Waiting for input...")


# Feedback section
st.markdown("---")
st.markdown("### 🛠️ Help Us Improve")
st.markdown(
    "If the predictions above were incorrect, please select the correct digit you intended to draw. "
    "These examples will help us train better models in the future by learning from real mistakes."
)

# --- Layout like a phone keypad ---
left_col, right_col = st.columns([1.2, 3])  # Wider right for digit grid

with left_col:
    st.button("❌ Not a Digit", use_container_width=True, key="feedback_-1")
    st.button("✔️ Correct Prediction", use_container_width=True, key="feedback_confirm")

with right_col:
    row1 = st.columns(5, gap="small")
    for i in range(5):
        row1[i].button(str(i), key=f"feedback_{i}", use_container_width=True)

    row2 = st.columns(5, gap="small")
    for i in range(5, 10):
        row2[i - 5].button(str(i), key=f"feedback_{i}", use_container_width=True)