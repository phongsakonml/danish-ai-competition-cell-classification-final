import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import base64
import os
import random
import utils

def get_efficientnet_b3(num_classes=2):
    # Load the pre-trained EfficientNet-B3 model
    model = models.efficientnet_b3(pretrained=False)
    
    # Modify the classifier
    in_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(in_features=in_features, out_features=640),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.4),
        torch.nn.Linear(640, 320),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.4),
        torch.nn.Linear(320, num_classes)
    )
    
    return model

# Update the model path to use the best model
model_path = 'runs/first_1003_1451_0/best_model.pth'  # Updated model path

# Load the trained model
model = get_efficientnet_b3(num_classes=2)  # Ensure num_classes matches the trained model

# Load the state dict
state_dict = torch.load(model_path, map_location=torch.device('cpu'))

# Remove the 'module.' prefix if it exists (in case the model was trained with DataParallel)
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

# Load the state dict, ignoring mismatched keys
model.load_state_dict(state_dict, strict=False)
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