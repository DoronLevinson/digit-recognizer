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
from fastapi import HTTPException

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
    try:
        image = read_uploaded_image(file)
        print("✅ DNN image received:", image.size, image.mode)
        pred, probs = predict_dnn_digit(image, dnn_model)
        print("✅ DNN prediction successful:", pred)
        return {"model": "DNN", "prediction": pred, "probs": probs.tolist()}
    except Exception as e:
        print("❌ DNN prediction error:", str(e))
        raise HTTPException(status_code=500, detail=f"DNN Error: {str(e)}")

# CNN Prediction Endpoint
@app.post("/predict/cnn")
async def predict_cnn(file: UploadFile = File(...)):
    try:
        image = read_uploaded_image(file)
        print("✅ CNN image received:", image.size, image.mode)
        pred, probs = predict_cnn_digit(image, cnn_model)
        print("✅ CNN prediction successful:", pred)
        return {"model": "CNN", "prediction": pred, "probs": probs.tolist()}
    except Exception as e:
        print("❌ CNN prediction error:", str(e))
        raise HTTPException(status_code=500, detail=f"CNN Error: {str(e)}")

# KNN Prediction Endpoint
@app.post("/predict/knn")
async def predict_knn(file: UploadFile = File(...)):
    try:
        image = read_uploaded_image(file)
        print("✅ Image received:", image.size, image.mode)
        pred, probs = predict_knn_digit(image, knn_model)
        print("✅ KNN prediction successful:", pred)
        return {"model": "KNN", "prediction": int(pred), "probs": probs.tolist()}
    except Exception as e:
        print("❌ KNN prediction error:", e)
        raise HTTPException(status_code=500, detail=f"KNN Error: {str(e)}")
        
# ViT-CLIP Prediction Endpoint
@app.post("/predict/clip")
async def predict_clip(file: UploadFile = File(...)):
    try:
        image = read_uploaded_image(file)
        print("✅ CLIP image received:", image.size, image.mode)
        pred, probs = predict_clip_digit(image, clip_model)
        print("✅ CLIP prediction successful:", pred)
        return {"model": "ViT-CLIP", "prediction": int(pred), "probs": probs.tolist()}
    except Exception as e:
        print("❌ CLIP prediction error:", str(e))
        raise HTTPException(status_code=500, detail=f"CLIP Error: {str(e)}")

# Check
@app.get("/")
def root():
    return {"message": "FastAPI model server is running!"}