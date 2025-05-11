# models/vit_classifier_model.py
import torch
import torch.nn as nn
from transformers import CLIPModel

class CLIPDigitClassifier(nn.Module):
    def __init__(self, visual_encoder, projection, num_classes=11):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.projection = projection
        self.classifier = nn.Sequential(
            nn.Linear(projection.out_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, pixel_values):
        x = self.visual_encoder(pixel_values).pooler_output
        x = self.projection(x)
        return self.classifier(x)