import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
import base64
import os

def get_resnet18(num_classes=2):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# Load the trained model
model = get_resnet18()
run_dir = 'runs/zany_eagle_1409'  # Update this with your actual run name
model_path = os.path.join(run_dir, 'zany_eagle_1409_model.pth')
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

def predict(image: str) -> int:
    try:
        # Decode base64 to image
        img_data = base64.b64decode(image)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        # Convert uint16 to uint8 if necessary
        if img.dtype == np.uint16:
            img = (img / 256).astype(np.uint8)

        # Preprocess the image
        img_tensor = transform(img).unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)

        return predicted.item()
    except Exception as e:
        print(f"Error in predict function: {str(e)}")
        return -1  # Return an error code