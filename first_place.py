import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, exposure
from sklearn.model_selection import StratifiedKFold, GridSearchCV  # For hyperparameter tuning
import torch
import torch.nn as nn
from torchvision import models
from datetime import datetime
import json
from tqdm import tqdm
from torch.optim import lr_scheduler
from sklearn.metrics import f1_score, precision_recall_fscore_support, balanced_accuracy_score
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# Load the data
data_dir = 'data/training_balanced'
labels_file = 'data/training_balanced.csv'

# Read labels
labels_df = pd.read_csv(labels_file)
labels_df.columns = labels_df.columns.str.strip()

# Load images and labels
images = []
labels = []

for index, row in labels_df.iterrows():
    img_id = str(row['image_id']).strip()
    img_path = os.path.join(data_dir, f"{img_id}.tif")
    if os.path.exists(img_path):
        img = io.imread(img_path)
        img = exposure.rescale_intensity(img, out_range=(0, 255)).astype(np.uint8)
        
        # Resize the image to a fixed size (224x224)
        img = cv2.resize(img, (224, 224))  # Ensure all images are resized to 224x224
        
        if img.ndim == 2:
            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        
        images.append(img)
        labels.append(int(row['is_homogenous']))
    else:
        print(f"Warning: Image file not found: {img_path}")

# Convert labels to numpy array
labels = np.array(labels, dtype=int)

# Log class distribution
class_distribution = np.bincount(labels)
n_0, n_1 = class_distribution[0], class_distribution[1]
print(f"Class distribution: {class_distribution}")

# Data Augmentation
augment = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

class CellDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image, label

# Model Initialization: EfficientNet-B0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def initialize_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.features[-3:].parameters():
        param.requires_grad = True
    
    model.classifier = nn.Sequential(
        nn.Linear(in_features=1280, out_features=512),
        nn.ReLU(),
        nn.Dropout(0.5),  # Increased dropout for better regularization
        nn.Linear(512, 2)
    )
    return model

# Weighted Cross Entropy Loss Function
class_weights = torch.tensor([n_1 / n_0, n_0 / n_1], dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Training and Validation
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=100, run_dir='runs', fold=1):
    best_val_loss = float('inf')
    patience = 10
    early_stopping_counter = 0
    log = {
        'epochs': [],
        'train_losses': [],
        'val_losses': [],
        'train_accuracies': [],
        'val_accuracies': []
    }

    # Create a directory for the current fold
    fold_run_dir = os.path.join(run_dir, f'fold_{fold}')
    os.makedirs(fold_run_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        
        for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)

        # Log metrics
        log['epochs'].append(epoch + 1)
        log['train_losses'].append(train_loss)
        log['val_losses'].append(val_loss)
        log['train_accuracies'].append(train_acc)
        log['val_accuracies'].append(val_acc)

        # Print training results
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        scheduler.step(val_loss)

        # Save the best model only if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_save_path = os.path.join(fold_run_dir, 'best_model.pth')
            torch.save(model.state_dict(), model_save_path)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print("Early stopping!")
                break

    # Save training log to JSON
    with open(os.path.join(fold_run_dir, 'training_log.json'), 'w') as f:
        json.dump(log, f, indent=4)

    return best_val_loss, fold_run_dir

# Hyperparameter tuning function
def hyperparameter_tuning(train_dataset, val_dataset):
    param_grid = {
        'batch_size': [16, 32],
        'learning_rate': [1e-4, 1e-5],
        'optimizer': [torch.optim.AdamW, torch.optim.SGD]
    }
    # Implement grid search logic here
    # This is a placeholder for the actual implementation
    best_params = {'batch_size': 32, 'learning_rate': 1e-4, 'optimizer': torch.optim.AdamW}
    return best_params

# Cross-Validation for better generalization
def cross_validation_training(images, labels, num_folds=5):
    skf = StratifiedKFold(n_splits=num_folds)
    best_overall_loss = float('inf')
    best_model_info = {}

    for fold, (train_idx, val_idx) in enumerate(skf.split(images, labels), 1):
        print(f"Fold {fold}")
        X_train, X_val = np.array(images)[train_idx], np.array(images)[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        
        train_dataset = CellDataset(X_train, y_train, transform=augment)
        val_dataset = CellDataset(X_val, y_val, transform=val_transform)

        best_params = hyperparameter_tuning(train_dataset, val_dataset)

        train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False, num_workers=4)

        model = initialize_model().to(device)
        optimizer = best_params['optimizer'](model.parameters(), lr=best_params['learning_rate'])
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

        best_val_loss, fold_run_dir = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, fold=fold)

        # Check if this fold's best model is the overall best
        if best_val_loss < best_overall_loss:
            best_overall_loss = best_val_loss
            best_model_info = {
                'fold': fold,
                'model_path': os.path.join(fold_run_dir, 'best_model.pth'),
                'log_path': os.path.join(fold_run_dir, 'training_log.json')
            }
        print(f"Fold {fold} completed.\n")

    # Save results.json with the best model information
    with open('results.json', 'w') as f:
        json.dump(best_model_info, f, indent=4)

    return best_model_info

# Main function
def main():
    best_model_info = cross_validation_training(images, labels, num_folds=5)  # Ensure this returns the best model info
    print("Training completed")

    # Plotting training curves for the best model
    best_log_path = best_model_info['log_path']
    with open(best_log_path, 'r') as f:
        log = json.load(f)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(log['epochs'], log['train_losses'], label='Train Loss')
    plt.plot(log['epochs'], log['val_losses'], label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(log['epochs'], log['train_accuracies'], label='Train Accuracy')
    plt.plot(log['epochs'], log['val_accuracies'], label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig(os.path.join(best_model_info['fold'], 'training_curves.png'))
    plt.show()

if __name__ == '__main__':
    main()