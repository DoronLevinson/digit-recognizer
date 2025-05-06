import torch
import numpy as np
from PIL import ImageOps
from PIL import Image
from model import SimpleNN

# Load model
def load_model(path="simple_nn_mnist_model.pth"):
    model = SimpleNN()
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.eval()
    return model

# Preprocess and predict (no torchvision)
def predict_digit(image: Image.Image, model):
    # Convert to grayscale and resize to 28x28
    image = ImageOps.grayscale(image).resize((28, 28))

    # Convert to numpy, normalize to [-1, 1]
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = (img_array - 0.5) / 0.5  # Normalize like torchvision

    # Reshape to torch tensor [B, C, H, W]
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        pred = output.argmax(dim=1).item()

    return pred