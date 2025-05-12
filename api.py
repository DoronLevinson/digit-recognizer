from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import numpy as np
import io
from inference import (
    predict_dnn_digit, predict_cnn_digit,
    predict_knn_digit, predict_clip_digit,
    load_dnn_model, load_cnn_model,
    load_knn_model, load_clip_digit_model
)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS so Streamlit can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with Streamlit URL if you want stricter rules
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load models once on startup
dnn_model = load_dnn_model()
cnn_model = load_cnn_model()
knn_model = load_knn_model()
clip_model = load_clip_digit_model()

# Utility function to read uploaded image
def read_uploaded_image(file: UploadFile) -> Image.Image:
    image = Image.open(io.BytesIO(file.file.read()))
    return image.convert("RGB")

# DNN Prediction Endpoint
@app.post("/predict/dnn")
async def predict_dnn(file: UploadFile = File(...)):
    image = read_uploaded_image(file)
    pred, probs = predict_dnn_digit(image, dnn_model)
    return {"model": "DNN", "prediction": pred, "probs": probs}

# CNN Prediction Endpoint
@app.post("/predict/cnn")
async def predict_cnn(file: UploadFile = File(...)):
    image = read_uploaded_image(file)
    pred, probs = predict_cnn_digit(image, cnn_model)
    return {"model": "CNN", "prediction": pred, "probs": probs}

# KNN Prediction Endpoint
@app.post("/predict/knn")
async def predict_knn(file: UploadFile = File(...)):
    image = read_uploaded_image(file)
    pred, probs = predict_knn_digit(image, knn_model)
    return {"model": "KNN", "prediction": pred, "probs": probs}

# ViT-CLIP Prediction Endpoint
@app.post("/predict/clip")
async def predict_clip(file: UploadFile = File(...)):
    image = read_uploaded_image(file)
    pred, probs = predict_clip_digit(image, clip_model)
    return {"model": "ViT-CLIP", "prediction": pred, "probs": probs}

# Check
@app.get("/")
def root():
    return {"message": "FastAPI model server is running!"}