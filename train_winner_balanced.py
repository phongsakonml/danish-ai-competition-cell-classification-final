import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, exposure
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torchvision import models
from datetime import datetime
import json
from tqdm import tqdm
from torch.optim import lr_scheduler
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix, balanced_accuracy_score
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Load the data
data_dir = 'data/training_balanced'  # Use the balanced dataset
labels_file = 'data/training_balanced.csv'

# Read labels
labels_df = pd.read_csv(labels_file)

# Strip whitespace from column names
labels_df.columns = labels_df.columns.str.strip()

# Load images and labels
images = []
labels = []

for index, row in labels_df.iterrows():
    img_id = str(row['image_id']).strip()
    img_path = os.path.join(data_dir, f"{img_id}.tif")
    if os.path.exists(img_path):
        img = io.imread(img_path)
        
        # Normalize image to 0-255 range
        img = exposure.rescale_intensity(img, out_range=(0, 255)).astype(np.uint8)
        
        if img.ndim == 2:  # Check if the image is grayscale
            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)  # Convert to 3-channel
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)  # Convert single-channel to 3-channel
        
        images.append(img)
        labels.append(int(row['is_homogenous']))  # Ensure labels are integers
    else:
        print(f"Warning: Image file not found: {img_path}")

# Convert labels to numpy array
labels = np.array(labels, dtype=int)

# Log class distribution
class_distribution = np.bincount(labels)
print(f"Class distribution: {class_distribution}")

# Data Augmentation (Less Aggressive)
augment = A.Compose([
    A.Resize(224, 224),  # Ensure all images are resized to 224x224
    A.HorizontalFlip(p=0.3),  # Reduced probability
    A.VerticalFlip(p=0.3),    # Reduced probability
    A.Rotate(limit=15, p=0.5),  # Reduced rotation limit
    A.RandomBrightnessContrast(p=0.1),  # Reduced brightness/contrast adjustment
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(224, 224),  # Ensure validation images are also resized
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

# Model Selection: Transfer Learning with EfficientNet-B0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def initialize_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    # Freeze all layers initially
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze the last few layers for fine-tuning
    for param in model.features[-1].parameters():
        param.requires_grad = True
    
    model.classifier = nn.Sequential(
        nn.Linear(in_features=1280, out_features=512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 2)
    )
    return model

# Create a directory for saving models and logs
os.makedirs('runs', exist_ok=True)

def get_run_name(model_name, base_dir='runs'):
    counter = 0
    while True:
        run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{counter}"
        run_dir = os.path.join(base_dir, run_name)
        try:
            os.makedirs(run_dir)
            return run_name, run_dir
        except FileExistsError:
            counter += 1

model_name = "efficientnet_b0"
run_name, run_dir = get_run_name(model_name)

# Create a SummaryWriter for this run
writer = SummaryWriter(log_dir=run_dir)

def visualize_samples(images, labels, transform=None, num_samples=5):
    """Visualize original and augmented images."""
    fig, axes = plt.subplots(2, num_samples, figsize=(20, 8))
    
    for i in range(num_samples):
        # Original image
        image = images[i]
        axes[0, i].imshow(image)
        axes[0, i].set_title(f"Original Label: {labels[i]}")
        axes[0, i].axis('off')
        
        # Augmented image
        if transform:
            augmented = transform(image=image)
            aug_image = augmented['image'].permute(1, 2, 0).numpy()
            aug_image = (aug_image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
            aug_image = aug_image.astype(np.uint8)
            axes[1, i].imshow(aug_image)
            axes[1, i].set_title(f"Augmented Label: {labels[i]}")
            axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Visualize some samples before and after augmentation
visualize_samples(images, labels, transform=augment)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=100):
    best_val_loss = float('inf')
    patience = 10
    early_stopping_counter = 0

    # Log for saving training metrics
    log = {
        'epochs': [],
        'train_losses': [],
        'val_losses': [],
        'train_accuracies': [],
        'val_accuracies': [],
        'train_f1': [],
        'val_f1': [],
        'train_precision': [],
        'val_precision': [],
        'train_recall': [],
        'val_recall': [],
        'train_bal_acc': [],
        'val_bal_acc': []
    }

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_predictions = []
        train_true_labels = []
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_predictions.extend(predicted.cpu().numpy())
            train_true_labels.extend(labels.cpu().numpy())
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)
        
        # Calculate training metrics
        train_f1 = f1_score(train_true_labels, train_predictions, average='weighted')
        train_precision, train_recall, _, _ = precision_recall_fscore_support(
            train_true_labels, train_predictions, average='weighted'
        )
        train_bal_acc = balanced_accuracy_score(train_true_labels, train_predictions)
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_predictions = []
        val_true_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_predictions.extend(predicted.cpu().numpy())
                val_true_labels.extend(labels.cpu().numpy())
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)
        
        # Calculate validation metrics
        val_f1 = f1_score(val_true_labels, val_predictions, average='weighted')
        val_precision, val_recall, _, _ = precision_recall_fscore_support(
            val_true_labels, val_predictions, average='weighted'
        )
        val_bal_acc = balanced_accuracy_score(val_true_labels, val_predictions)
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('F1_Score/train', train_f1, epoch)
        writer.add_scalar('F1_Score/val', val_f1, epoch)
        writer.add_scalar('Precision/train', train_precision, epoch)
        writer.add_scalar('Precision/val', val_precision, epoch)
        writer.add_scalar('Recall/train', train_recall, epoch)
        writer.add_scalar('Recall/val', val_recall, epoch)
        writer.add_scalar('Balanced_Accuracy/train', train_bal_acc, epoch)
        writer.add_scalar('Balanced_Accuracy/val', val_bal_acc, epoch)
        
        # Save metrics to log
        log['epochs'].append(epoch + 1)
        log['train_losses'].append(train_loss)
        log['val_losses'].append(val_loss)
        log['train_accuracies'].append(train_acc)
        log['val_accuracies'].append(val_acc)
        log['train_f1'].append(train_f1)
        log['val_f1'].append(val_f1)
        log['train_precision'].append(train_precision)
        log['val_precision'].append(val_precision)
        log['train_recall'].append(train_recall)
        log['val_recall'].append(val_recall)
        log['train_bal_acc'].append(train_bal_acc)
        log['val_bal_acc'].append(val_bal_acc)

        writer.flush()
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")
        print(f"Train Precision: {train_precision:.4f}, Val Precision: {val_precision:.4f}")
        print(f"Train Recall: {train_recall:.4f}, Val Recall: {val_recall:.4f}")
        print(f"Train Balanced Accuracy: {train_bal_acc:.4f}, Val Balanced Accuracy: {val_bal_acc:.4f}")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(run_dir, 'best_model.pth'))
            print("New best model saved!")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print("Early stopping!")
                break

    # Save training log
    with open(os.path.join(run_dir, 'training_log.json'), 'w') as f:
        json.dump(log, f, indent=4)

    # Plot and save training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(log['epochs'], log['train_losses'], label='Train Loss')
    plt.plot(log['epochs'], log['val_losses'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(log['epochs'], log['train_accuracies'], label='Train Accuracy')
    plt.plot(log['epochs'], log['val_accuracies'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'training_curves.png'))
    plt.close()

    return model

def main():
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)

    # Dataset and DataLoader
    train_dataset = CellDataset(X_train, y_train, transform=augment)
    val_dataset = CellDataset(X_val, y_val, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)  # Set num_workers to 0
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)  # Set num_workers to 0

    # Model initialization
    model = initialize_model().to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

    # Train the model
    model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)

    print("Training completed")
    writer.close()

    # Save the final model
    torch.save(model.state_dict(), os.path.join(run_dir, 'final_model.pth'))
    print(f"Final model saved in {run_dir}")

if __name__ == '__main__':
    main()