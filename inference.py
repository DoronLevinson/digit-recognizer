import torch
import numpy as np
import os
import joblib
import urllib.request
from PIL import ImageOps, Image
from models.nn_model import SimpleNN
from models.cnn_model import CNN_MNIST
# from transformers import CLIPModel, CLIPProcessor
import pickle

# ---- DNN model ----
def load_model(path="models/simple_nn_mnist_model.pth"):
    model = SimpleNN()
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.eval()
    return model

def predict_digit(image: Image.Image, model):
    image = ImageOps.invert(image)
    image = ImageOps.grayscale(image).resize((28, 28))
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = (img_array - 0.5) / 0.5
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1).numpy().flatten()
        pred = int(np.argmax(probs))

    return pred, probs

# ---- KNN model ----

def load_knn_model(path="models/knn_model.joblib"):
    # for streamlit deployment
    X = pd.read_csv("models/X_knn.csv").values
    y = pd.read_csv("models/y_knn.csv").values.ravel()

    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X, y)
    return knn

    # # for local deployment
    # return joblib.load(path)

def predict_knn_digit(image: Image.Image, knn_model):
    # Resize and convert to grayscale
    image = image.resize((28, 28)).convert("L")
    image = ImageOps.invert(image)
    image = np.array(image).astype("float32").reshape(1, -1)

    # Predict with KNN
    probs = knn_model.predict_proba(image).flatten()
    pred = int(np.argmax(probs))

    return pred, probs


# ---- CNN model ----
def load_cnn_model(path="models/cnn_model.pth"):
    model = CNN_MNIST()
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.eval()
    return model

def predict_cnn_digit(image: Image.Image, model):
    image = ImageOps.invert(image)
    image = ImageOps.grayscale(image).resize((28, 28))
    img_array = np.array(image).astype(np.float32) / 255.0  # input between 0–1
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)  # [1, 1, 28, 28]

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1).numpy().flatten()
        probs = np.concatenate(([probs[-1]], probs[:-1]))
        pred = int(np.argmax(probs) - 1)
    

    return pred, probs


# # ---- Load CLIP Model ----
# def load_clip_model(path="models/clip-vit"):
#     model = CLIPModel.from_pretrained(path)
#     processor = CLIPProcessor.from_pretrained(path)
#     model.eval()
#     return model, processor

# # ---- Predict with CLIP ----
# def predict_with_clip(image: Image.Image, model, processor):
#     labels = ["not a digit"] + [f"the digit {i}" for i in range(0, 10)]
    
#     # Preprocess
#     inputs = processor(
#         text=labels,
#         images=image,
#         return_tensors="pt",
#         padding=True
#     )
    
#     # Forward pass
#     with torch.no_grad():
#         outputs = model(**inputs)
#         logits = outputs.logits_per_image  # shape: [1, 10]
#         probs = logits.softmax(dim=1).squeeze().cpu().numpy()  # shape: (10,)
    
#     print(probs)
#     print(len(probs))
#     return probs