import requests
from PIL import Image
import io

REMOTE_API_BASE = "http://13.61.104.76:8000"

def _prepare_image(image: Image.Image):
    """Convert image to PNG bytes for multipart upload."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    buffered.seek(0)
    return buffered

def predict_remote(image, model: str):
    """
    Generic function to predict using the remote API.
    `model` must be one of: 'dnn', 'cnn', 'knn', 'clip'
    """
    url = f"{REMOTE_API_BASE}/predict/{model}"
    image_bytes = _prepare_image(image)

    files = {"file": ("digit.png", image_bytes, "image/png")}
    response = requests.post(url, files=files)

    if response.status_code == 200:
        res = response.json()
        return res["prediction"], res["probs"]
    else:
        print("RESPONSE TEXT:", response.text)  # 👈 Add this
        raise RuntimeError(f"{model.upper()} API error: {response.status_code} - {response.text}")

# Wrappers per model
def predict_dnn_digit(image):
    return predict_remote(image, "dnn")

def predict_cnn_digit(image):
    return predict_remote(image, "cnn")

def predict_knn_digit(image):
    return predict_remote(image, "knn")

def predict_clip_digit(image):
    return predict_remote(image, "clip")