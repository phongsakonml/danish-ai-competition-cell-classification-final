import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
import base64
import os
import random
import utils

MODEL_NAME = 'thinkpad'
MODEL_PATH = f'runs/{MODEL_NAME}/best_model.pth'

def get_efficientnet_b0(num_classes=2):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier = nn.Sequential(
        nn.Linear(in_features=1280, out_features=512),  # Adjusted output features
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)  # Adjusted to match the number of classes
    )
    return model

# Load the Steve model (assuming similar structure as Thinkpad)
STEVE_MODEL_PATH = 'runs/steve/best_model.pth'
steve_model = get_efficientnet_b0(num_classes=2)  # Ensure num_classes matches the saved model
steve_model.load_state_dict(torch.load(STEVE_MODEL_PATH, map_location=torch.device('cpu')))
steve_model.eval()

# Load the trained model from the specified path
model = get_efficientnet_b0(num_classes=2)  # Ensure num_classes matches the saved model
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
            # Get predictions from both models
            output_thinkpad = model(img_tensor)
            output_steve = steve_model(img_tensor)
            
            # Apply softmax to get probabilities
            prob_thinkpad = torch.softmax(output_thinkpad, dim=1)
            prob_steve = torch.softmax(output_steve, dim=1)
            
            # Combine predictions using adjusted weights
            combined_prob = (0.5 * prob_thinkpad + 0.5 * prob_steve)  # Adjusted weights
            _, predicted = torch.max(combined_prob, 1)
            return predicted.item() 
    except Exception as e:
        print(f"Error in predict function: {str(e)}")
        return -1