import torch
import numpy as np
import os
import joblib
import urllib.request
from PIL import ImageOps, Image
from models.dnn_model import DNN_MNIST
from models.cnn_model import CNN_MNIST
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from models.vit_classifier_model import load_clip_digit_model, CLIPDigitClassifier
import torchvision.transforms as transforms

# ---- DNN model ----
def load_dnn_model(path="models/dnn_model.pth"):
    model = DNN_MNIST()
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.eval()
    return model

def predict_dnn_digit(image: Image.Image, model):
    image = ImageOps.invert(image.convert("L")).resize((28, 28))
    img_array = np.array(image).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).flatten().unsqueeze(0)  # [1, 784]

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1).numpy().flatten()
        probs = np.concatenate(([probs[-1]], probs[:-1]))  # move "-1" class to front
        pred = int(np.argmax(probs) - 1)

    return pred, probs

# ---- KNN model ----

def load_knn_model(path="models/knn_model.joblib"):
    # for streamlit deployment
    X = pd.read_csv("models/X_knn.csv").values
    y = pd.read_csv("models/y_knn.csv").values.ravel()
    X = X[:2000]
    y = y[:2000]

    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X, y)
    return knn

    # # for local deployment
    # return joblib.load(path)

def predict_knn_digit(image: Image.Image, knn_model):
    # Resize and convert to grayscale
    image = ImageOps.invert(image.convert("L")).resize((28, 28))
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



# ---- Fine-tuned CLIP classifier ----
def predict_clip_digit(image: Image.Image, model, return_preprocessed=False):
    # Invert and convert to grayscale 28×28
    mnist_like = ImageOps.invert(image.convert("L")).resize((28, 28))

    # Resize to 224×224 and convert to 3-channel tensor for CLIP
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])
    image_tensor = transform(mnist_like).unsqueeze(0)  # [1, 3, 224, 224]

    with torch.no_grad():
        model.eval()
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
        pred = int(np.argmax(probs)) - 1

    if return_preprocessed:
        return pred, probs, mnist_like  # preview 28x28 grayscale image
    return pred, probs