MODEL_NAME = 'ensemble'  # Define MODEL_NAME

import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
import base64
import os
import random
import utils
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json

MODEL_NAME_1 = 'marc'
MODEL_NAME_2 = 'mark'
MODEL_PATH_1 = f'runs/{MODEL_NAME_1}/best_model.pth'
MODEL_PATH_2 = f'runs/{MODEL_NAME_2}/best_model.pth'
JSON_PATH_1 = f'local_test/{MODEL_NAME_1}.json'
JSON_PATH_2 = f'local_test/{MODEL_NAME_2}.json'

def get_efficientnet_b0(num_classes=2):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier = nn.Sequential(
        nn.Linear(in_features=1280, out_features=640),  # Adjusted output features to match the saved model
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(640, 320),  # Adjusted to match the saved model
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(320, num_classes)  # Adjusted to match the number of classes
    )
    return model

# Load both trained models
model1 = get_efficientnet_b0(num_classes=2)  # Ensure num_classes matches the saved model
model1.load_state_dict(torch.load(MODEL_PATH_1, map_location=torch.device('cpu')))
model1.eval()

model2 = get_efficientnet_b0(num_classes=2)  # Ensure num_classes matches the saved model
model2.load_state_dict(torch.load(MODEL_PATH_2, map_location=torch.device('cpu')))
model2.eval()

# Load JSON files and extract scores
with open(JSON_PATH_1, 'r') as f:
    marc_data = json.load(f)
with open(JSON_PATH_2, 'r') as f:
    mark_data = json.load(f)

marc_score = marc_data['Score']
mark_score = mark_data['Score']

# Calculate weights
total_score = marc_score + mark_score
marc_weight = marc_score / total_score
mark_weight = mark_score / total_score

# Update the transform to match the training script's validation transform
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

def predict(image: str) -> int:
    try:
        # Decode base64 string to numpy array
        if isinstance(image, str):
            image = utils.decode_image(image)
        
        # Apply the transform
        augmented = transform(image=image)
        img_tensor = augmented['image'].unsqueeze(0)
        
        with torch.no_grad():
            # Get predictions from both models
            output1 = model1(img_tensor)
            output2 = model2(img_tensor)
            
            # Weighted average of the predictions
            ensemble_output = (output1 * marc_weight + output2 * mark_weight)
            
            # Get the final prediction
            _, predicted = torch.max(ensemble_output, 1)
            return predicted.item()
    except Exception as e:
        print(f"Error in predict function: {str(e)}")
        return -1

print(f"Model weights: Marc: {marc_weight:.4f}, Mark: {mark_weight:.4f}")