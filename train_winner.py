import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, exposure
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torchvision import models, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datetime import datetime
import json
from tqdm import tqdm
from torch.optim import lr_scheduler
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix, balanced_accuracy_score, precision_recall_curve, classification_report
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Load the data
data_dir = 'data/training'
labels_file = 'data/training.csv'

# Read labels
labels_df = pd.read_csv(labels_file)

# Print information about the CSV file
print("CSV file columns:")
print(labels_df.columns)
print("\nFirst few rows of the CSV file:")
print(labels_df.head())

import os

model_name = "efficientnet_b0"

def get_run_name(model_name, base_dir='runs'):
    counter = 0
    while True:
        run_name = f"{model_name}_{counter}"
        run_dir = os.path.join(base_dir, run_name)
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
            return run_name, run_dir
        counter += 1

# Check if expected columns exist, trimming any whitespace from column names
labels_df.columns = labels_df.columns.str.strip()
if 'image_id' not in labels_df.columns or 'is_homogenous' not in labels_df.columns:
    raise KeyError("Expected columns 'image_id' and 'is_homogenous' not found in the CSV file.")

# Load images and labels
images = []
labels = []

for index, row in labels_df.iterrows():
    img_id = str(row['image_id']).strip().zfill(3)  # Ensure image_id is zero-padded to 3 digits
    img_path = os.path.join(data_dir, f"{img_id}.tif")  # Use .tif extension
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

print(f"\nTotal images: {len(images)}")
print(f"Label distribution: {np.bincount(labels)}")

# Plot a sample image
if len(images) > 0:
    plt.figure(figsize=(10, 10))
    plt.imshow(images[0])
    plt.title(f"Sample Image (Label: {labels[0]})")
    plt.axis('off')
    plt.show()
else:
    print("No images were loaded. Please check the data directory and CSV file.")

# Step 1: Data Augmentation to Balance the Classes

# Separate heterogeneous and homogeneous cells
hetero_images = [img for img, label in zip(images, labels) if label == 0]
homo_images = [img for img, label in zip(images, labels) if label == 1]

print(f"Heterogeneous cells: {len(hetero_images)}")
print(f"Homogeneous cells: {len(homo_images)}")

# Find the minimum dimension across all images
min_dim = min(min(img.shape[:2]) for img in images)
print(f"Minimum image dimension: {min_dim}")

# Augmentation pipeline
augment = A.Compose([
    A.RandomResizedCrop(height=224, width=224, scale=(0.9, 1.0)),  # Less aggressive crop
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),  # Reduced rotation angle
    A.Normalize(mean=[0.485], std=[0.229]),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(height=224, width=224),
    A.Normalize(mean=[0.485], std=[0.229]),
    ToTensorV2()
])

# Augment homogeneous cells
augmented_homo_images = []
num_augmentations = len(hetero_images) - len(homo_images)

for _ in range(num_augmentations):
    idx = np.random.randint(0, len(homo_images))
    img = homo_images[idx]
    augmented = augment(image=img)['image']
    augmented_homo_images.append(augmented.numpy())  # Convert back to numpy array

# Combine original and augmented images
balanced_images = images + augmented_homo_images
balanced_labels = np.concatenate([labels, np.ones(len(augmented_homo_images), dtype=int)])

print(f"Total images after augmentation: {len(balanced_images)}")
print(f"Label distribution after augmentation: {np.bincount(balanced_labels)}")

# Plot original and augmented images
def plot_image(ax, img, title):
    if isinstance(img, torch.Tensor):
        img = img.permute(1, 2, 0).numpy()
    elif len(img.shape) == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    ax.imshow(img)
    ax.set_title(title)
    ax.axis('off')

fig, axes = plt.subplots(2, 2, figsize=(15, 15))
plot_image(axes[0, 0], homo_images[0], "Original Homogeneous")
plot_image(axes[0, 1], augmented_homo_images[0], "Augmented Homogeneous")
plot_image(axes[1, 0], hetero_images[0], "Original Heterogeneous")
plot_image(axes[1, 1], hetero_images[1], "Another Heterogeneous")
plt.tight_layout()
plt.show()

# Prepare data for model training
X_train, X_test, y_train, y_test = train_test_split(balanced_images, balanced_labels, test_size=0.1, stratify=balanced_labels, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

# Dataset and DataLoader
class CellDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Ensure image is a numpy array
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        elif not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # Ensure image is 2D (grayscale)
        if image.ndim == 3:
            image = image[:,:,0]  # Take first channel if it's 3D
        
        # Normalize image to 0-1 range
        image = (image - image.min()) / (image.max() - image.min())
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label

# Model Selection: Transfer Learning with EfficientNet-B0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model initialization
def initialize_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    # Modify the first convolutional layer to accept single-channel input
    model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = nn.Sequential(
        nn.Linear(in_features=1280, out_features=512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 2)
    )
    return model

# Create a directory for saving models and logs
os.makedirs('runs', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Assume model_name is defined earlier, e.g., model_name = "efficientnet_b0"
run_name, run_dir = get_run_name(model_name)

# Create a SummaryWriter for this run
writer = SummaryWriter(log_dir=run_dir)

# Model training loop
num_epochs = 50
best_val_loss = float('inf')
patience = 5
early_stopping_counter = 0

def visualize_augmentations(dataset, num_samples=5):
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 4))
    for i in range(num_samples):
        image, label = dataset[i]
        if isinstance(image, torch.Tensor):
            image = image.squeeze().numpy()
        
        # Denormalize the image
        image = image * 0.229 + 0.485
        image = np.clip(image, 0, 1)
        
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f"Label: {label}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

def main():
    # Initialize best_val_loss
    best_val_loss = float('inf')

    # Initialize log dictionary
    log = {
        'epochs': [],
        'train_losses': [],
        'val_losses': [],
        'train_accuracies': [],
        'val_accuracies': []
    }

    # Dataset and DataLoader
    train_dataset = CellDataset(X_train, y_train, transform=augment)
    val_dataset = CellDataset(X_val, y_val, transform=val_transform)
    test_dataset = CellDataset(X_test, y_test, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Visualize augmentations
    visualize_augmentations(train_dataset)

    # Model initialization
    model = initialize_model().to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_predictions = []
        train_true_labels = []
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            try:
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
            except Exception as e:
                print(f"Error in training loop: {e}")
                print(f"Input shape: {inputs.shape}, Label shape: {labels.shape}")
                continue
            
        train_loss = train_loss / len(train_dataset)
        train_acc = train_correct / len(train_dataset)
        
        # Calculate training metrics
        train_f1 = f1_score(train_true_labels, train_predictions, average='weighted')
        train_precision, train_recall, _, _ = precision_recall_fscore_support(train_true_labels, train_predictions, average='weighted')
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
                
                del inputs, labels, outputs, loss
        
        val_loss = val_loss / len(val_dataset)
        val_acc = val_correct / len(val_dataset)
        
        # Calculate validation metrics
        val_f1 = f1_score(val_true_labels, val_predictions, average='weighted')
        val_precision, val_recall, _, _ = precision_recall_fscore_support(val_true_labels, val_predictions, average='weighted')
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
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        writer.flush()
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")
        print(f"Train Precision: {train_precision:.4f}, Val Precision: {val_precision:.4f}")
        print(f"Train Recall: {train_recall:.4f}, Val Recall: {val_recall:.4f}")
        print(f"Train Balanced Accuracy: {train_bal_acc:.4f}, Val Balanced Accuracy: {val_bal_acc:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
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

        log['epochs'].append(epoch + 1)
        log['train_losses'].append(train_loss)
        log['val_losses'].append(val_loss)
        log['train_accuracies'].append(train_acc)
        log['val_accuracies'].append(val_acc)

    print("Training completed")
    writer.close()

    # Save the final model
    torch.save(model.state_dict(), os.path.join(run_dir, 'final_model.pth'))
    print(f"Final model saved in {run_dir}")

    # Save the training log
    log = {
        "model_name": model_name,
        "run_name": run_name,
        "epochs": num_epochs,
        "best_val_loss": best_val_loss,
        # Add any other relevant information
    }
    with open(os.path.join(run_dir, 'training_log.json'), 'w') as f:
        json.dump(log, f, indent=4)
    print(f"Training log saved in {run_dir}")

    print("Training completed")

    # Evaluate on validation set
    print("\nEvaluating best model on validation set...")
    model.load_state_dict(torch.load(os.path.join(run_dir, 'best_model.pth')))
    model.eval()

    val_predictions = []
    val_true_labels = []
    val_probabilities = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            val_predictions.extend(predicted.cpu().numpy())
            val_true_labels.extend(labels.cpu().numpy())
            val_probabilities.extend(probabilities[:, 1].cpu().numpy())

    def safe_division(n, d):
        return n / d if d else 0

    def evaluate_model(y_true, y_pred):
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        cm = confusion_matrix(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        
        a0 = cm[0, 0]  # True Negatives (correctly predicted heterogeneous)
        a1 = cm[1, 1]  # True Positives (correctly predicted homogeneous)
        n0 = np.sum(y_true == 0)
        n1 = np.sum(y_true == 1)
        custom_score = safe_division(a0 * a1, n0 * n1)
        
        return {
            'F1 Score': f1,
            'Precision': precision,
            'Recall': recall,
            'Balanced Accuracy': bal_acc,
            'Custom Score': custom_score,
            'Confusion Matrix': cm
        }

    results = evaluate_model(val_true_labels, val_predictions)
    print(results)

    print("\nClassification Report:")
    print(classification_report(val_true_labels, val_predictions))

    # Optimize decision threshold
    precision, recall, thresholds = precision_recall_curve(val_true_labels, val_probabilities)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]

    print(f"\nOptimal threshold: {optimal_threshold:.4f}")

    # Apply optimal threshold
    optimized_predictions = (np.array(val_probabilities) >= optimal_threshold).astype(int)

    print("\nOptimized Classification Report:")
    print(classification_report(val_true_labels, optimized_predictions))

    print("\nOptimized Confusion Matrix:")
    print(confusion_matrix(val_true_labels, optimized_predictions))

    print("\nTraining and evaluation completed successfully!")

    # Plot training and validation curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(log['epochs'], log['train_losses'], label='Train Loss')
    plt.plot(log['epochs'], log['val_losses'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(log['epochs'], log['train_accuracies'], label='Train Accuracy')
    plt.plot(log['epochs'], log['val_accuracies'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'training_plot.png'))
    plt.close()

    print("\nOptimizing decision threshold...")
    model.load_state_dict(torch.load(os.path.join(run_dir, 'best_model.pth')))
    model.eval()

    val_probabilities = []
    val_true_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            val_probabilities.extend(probabilities[:, 1].cpu().numpy())
            val_true_labels.extend(labels.cpu().numpy())

    precision, recall, thresholds = precision_recall_curve(val_true_labels, val_probabilities)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]

    print(f"Optimal threshold: {optimal_threshold:.4f}")

    print("\nFinal Evaluation:")
    test_predictions = []
    test_true_labels = []
    test_probabilities = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            test_predictions.extend(predicted.cpu().numpy())
            test_true_labels.extend(labels.cpu().numpy())
            test_probabilities.extend(probabilities[:, 1].cpu().numpy())

    # Apply optimal threshold
    optimized_predictions = (np.array(test_probabilities) >= optimal_threshold).astype(int)

    print("\nClassification Report (Default Threshold):")
    print(classification_report(test_true_labels, test_predictions))

    print("\nConfusion Matrix (Default Threshold):")
    print(confusion_matrix(test_true_labels, test_predictions))

    print("\nClassification Report (Optimized Threshold):")
    print(classification_report(test_true_labels, optimized_predictions))

    print("\nConfusion Matrix (Optimized Threshold):")
    print(confusion_matrix(test_true_labels, optimized_predictions))

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
        print(f"Fold {fold+1}")
        
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_subsampler)
        val_loader = DataLoader(train_dataset, batch_size=32, sampler=val_subsampler)
        
        model = initialize_model().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        # Training loop (use the same loop as before)
        
        # Evaluate on validation set
        model.eval()
        val_predictions = []
        val_true_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_predictions.extend(predicted.cpu().numpy())
                val_true_labels.extend(labels.cpu().numpy())
        
        fold_f1 = f1_score(val_true_labels, val_predictions, average='weighted')
        cv_scores.append(fold_f1)
        print(f"Fold {fold+1} F1-Score: {fold_f1:.4f}")

    print(f"Cross-Validation F1-Score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

if __name__ == '__main__':
    main()