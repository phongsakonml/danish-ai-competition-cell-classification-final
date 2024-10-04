import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
import base64
import os
import random
import utils

MODEL_NAME = 'napoleon'
MODEL_PATH = f'runs/{MODEL_NAME}/best_model.pth'

def initialize_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze more layers for fine-tuning
    for param in model.features[-4:].parameters():
        param.requires_grad = True
    
    model.classifier = nn.Sequential(
        nn.Linear(in_features=1280, out_features=640),
        nn.ReLU(),
        nn.Dropout(0.6),
        nn.Linear(640, 320),
        nn.ReLU(),
        nn.Dropout(0.6),
        nn.Linear(320, 160),
        nn.ReLU(),
        nn.Dropout(0.6),
        nn.Linear(160, 2)
    )
    return model

# Load the trained model from the specified path
model = initialize_model()
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
            probabilities = torch.softmax(output, dim=1)
            _, predicted = torch.max(probabilities, 1)
            return int(predicted.item())  # Ensure we return a single integer
    except Exception as e:
        print(f"Error in predict function: {str(e)}")
        return -1