import torch
import torch.nn as nn
from torchvision import models

# Define the model architecture (same as used during training)
def initialize_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.features[-3:].parameters():
        param.requires_grad = True
    
    # Update the classifier to match the saved model's architecture
    model.classifier = nn.Sequential(
        nn.Linear(in_features=1280, out_features=640),  # Match the saved model
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(640, 320)  # Update to match the saved model's output size
    )
    
    return model

# Load the model
model = initialize_model()
state_dict = torch.load('runs/mark/best_model.pth')

# Load the state dict with strict=False to ignore mismatched keys
model.load_state_dict(state_dict, strict=False)

# Set the model to evaluation mode
model.eval()

# Print model architecture
print(model)

# Optionally, inspect specific layers or parameters
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Layer: {name}, Shape: {param.data.shape}")