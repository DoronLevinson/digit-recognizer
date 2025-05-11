import torch
import numpy as np
import os
from PIL import ImageOps, Image
from dnn_model import DNN_MNIST
from cnn_model import CNN_MNIST
from simple_nn_model import SimpleNN
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from vit_classifier_model import CLIPDigitClassifier
from transformers import CLIPModel
import torchvision.transforms as transforms

def download_from_s3(s3_path, local_path):
    if not os.path.exists(local_path):
        print(f"Downloading {s3_path} to {local_path}...")
        s3 = boto3.client("s3")
        bucket, key = s3_path.replace("s3://", "").split("/", 1)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(bucket, key, local_path)

# ---- DNN model ----
def load_dnn_model(path="models/simple_nn_mnist_model.pth"):
    s3_path = "s3://digit-recognizer-bucket/models/dnn_model.pth"
    download_from_s3(s3_path, path)
    model = SimpleNN()
    model.load_state_dict(torch.load(path))

    model.eval()
    return model

def predict_dnn_digit(image: Image.Image, model):
    image = ImageOps.invert(image.convert("L")).resize((28, 28))
    img_array = np.array(image).astype(np.float32) / 255.0
    img_tensor = img_tensor.to("cpu")

    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model = model.to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1).numpy().flatten()
        probs = np.concatenate(([probs[-1]], probs[:-1]))  # move "-1" class to front
        pred = int(np.argmax(probs) - 1)

    return pred, probs

# ---- KNN model ----

def load_knn_model():
    download_from_s3("s3://digit-recognizer-bucket/data/X_knn.csv", "models/X_knn.csv")
    download_from_s3("s3://digit-recognizer-bucket/data/y_knn.csv", "models/y_knn.csv")
    X = pd.read_csv("models/X_knn.csv").values[:2000]
    y = pd.read_csv("models/y_knn.csv").values.ravel()[:2000]
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X, y)
    return knn

def predict_knn_digit(image: Image.Image, knn_model):
    # Resize and convert to grayscale
    image = ImageOps.invert(image.convert("L")).resize((28, 28))
    image = np.array(image).astype("float32").reshape(1, -1)

    # Predict with KNN
    probs = knn_model.predict_proba(image).flatten()
    pred = int(np.argmax(probs))

    return pred, probs


# ---- CNN model ----
def load_clip_digit_model(path="models/clip_digit_classifier.pth", device="cpu"):
    s3_path = "s3://digit-recognizer-bucket/models/dnn_model.pth"
    download_from_s3(s3_path, path)
    base_clip = CLIPModel.from_pretrained("wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M")
    visual_encoder = base_clip.vision_model
    visual_projection = base_clip.visual_projection

    model = CLIPDigitClassifier(visual_encoder, visual_projection, num_classes=11)
    model.load_state_dict(torch.load(path, map_location=torch.device(device)))
    model.eval()
    return model

def load_cnn_model(path="models/cnn_model.pth"):
    s3_path = "s3://digit-recognizer-bucket/models/cnn_model.pth"
    download_from_s3(s3_path, path)
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