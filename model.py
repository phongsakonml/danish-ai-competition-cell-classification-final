import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
import base64
import os
import random
import utils

MODEL_NAME = 'lucifer'
MODEL_PATH = f'runs/{MODEL_NAME}/best_model.pth'

def get_efficientnet_b3(num_classes=2):  # Change to EfficientNet-B3
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)  # Use EfficientNet-B3
    model.classifier = nn.Sequential(
        nn.Linear(in_features=1536, out_features=640),  # Adjusted in_features for EfficientNet-B3
        nn.ReLU(),
        nn.Dropout(0.7),  # Increased dropout rate
        nn.Linear(640, 320),
        nn.ReLU(),
        nn.Dropout(0.7),  # Increased dropout rate
        nn.Linear(320, 160),
        nn.ReLU(),
        nn.Dropout(0.7),  # Increased dropout rate
        nn.Linear(160, num_classes)  # Ensure num_classes matches the saved model
    )
    return model

# Load the trained model from the specified path
model = get_efficientnet_b3(num_classes=2)  # Ensure num_classes matches the saved model
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# Update the transform to match the training script
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(image: str) -> int:
    try:
        # Decode base64 string to numpy array
        if isinstance(image, str):
            image = utils.decode_image(image)
        
        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)
            return predicted.item() 
    except Exception as e:
        print(f"Error in predict function: {str(e)}")
        return -1