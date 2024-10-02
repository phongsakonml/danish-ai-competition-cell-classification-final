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

# Augmentation pipeline
augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=45, p=0.7),
    A.ElasticTransform(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.RandomCrop(height=min_dim, width=min_dim, p=0.5),
])

# Augment homogeneous cells
augmented_homo_images = []
num_augmentations = len(hetero_images) - len(homo_images)

for _ in range(num_augmentations):
    idx = np.random.randint(0, len(homo_images))
    img = homo_images[idx]
    augmented = augment(image=img)['image']
    augmented_homo_images.append(augmented)

# Combine original and augmented images
balanced_images = images + augmented_homo_images
balanced_labels = np.concatenate([labels, np.ones(len(augmented_homo_images), dtype=int)])

print(f"Total images after augmentation: {len(balanced_images)}")
print(f"Label distribution after augmentation: {np.bincount(balanced_labels)}")

# Plot original and augmented images
fig, axes = plt.subplots(2, 2, figsize=(15, 15))
axes[0, 0].imshow(homo_images[0])
axes[0, 0].set_title("Original Homogeneous")
axes[0, 1].imshow(augmented_homo_images[0])
axes[0, 1].set_title("Augmented Homogeneous")
axes[1, 0].imshow(hetero_images[0])
axes[1, 0].set_title("Original Heterogeneous")
axes[1, 1].imshow(hetero_images[1])
axes[1, 1].set_title("Another Heterogeneous")
for ax in axes.ravel():
    ax.axis('off')
plt.tight_layout()
plt.show()

# Prepare data for model training
X_train, X_val, y_train, y_val = train_test_split(balanced_images, balanced_labels, test_size=0.2, stratify=balanced_labels, random_state=42)

# Define transforms
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(45),
    transforms.Resize((min_dim, min_dim)),  # Resize all images to the minimum dimension
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((min_dim, min_dim)),  # Resize all images to the minimum dimension
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

# Create PyTorch datasets
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
        if self.transform:
            image = self.transform(image)
        return image, label

train_dataset = CellDataset(X_train, y_train, transform=train_transform)
val_dataset = CellDataset(X_val, y_val, transform=val_transform)

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# Model Selection: Transfer Learning with EfficientNet-B0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load pre-trained EfficientNet
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False

# Replace the classifier
model.classifier = nn.Sequential(
    nn.Linear(in_features=1280, out_features=512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 2)
)

model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Create a directory for saving models and logs
os.makedirs('runs', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Generate timestamp for model name and log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = f"efficientnet_b0_{timestamp}"
log_file = f"logs/training_log_{timestamp}.json"

# Initialize log dictionary
log = {
    "model_name": model_name,
    "epochs": [],
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": [],
    "learning_rate": []
}

# Training loop
num_epochs = 50
best_val_loss = float('inf')
early_stopping_patience = 10
early_stopping_counter = 0

print("Starting training...")
for epoch in tqdm(range(num_epochs), desc="Epochs"):
    model.train()
    train_loss = 0.0
    train_correct = 0
    
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
    
    scheduler.step()  # Move this line here, after the training loop
    
    train_loss = train_loss / len(train_dataset)
    train_acc = train_correct / len(train_dataset)
    
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
    
    val_loss = val_loss / len(val_dataset)
    val_acc = val_correct / len(val_dataset)
    
    current_lr = optimizer.param_groups[0]['lr']
    
    # Log the epoch results
    log["epochs"].append(epoch + 1)
    log["train_loss"].append(train_loss)
    log["train_acc"].append(train_acc)
    log["val_loss"].append(val_loss)
    log["val_acc"].append(val_acc)
    log["learning_rate"].append(current_lr)
    
    # Print epoch results
    tqdm.write(f"Epoch {epoch+1}/{num_epochs}")
    tqdm.write(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    tqdm.write(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    tqdm.write(f"Learning Rate: {current_lr:.6f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stopping_counter = 0
        torch.save(model.state_dict(), f'runs/{model_name}_best.pth')
        tqdm.write("New best model saved!")
    else:
        early_stopping_counter += 1
    
    if early_stopping_counter >= early_stopping_patience:
        tqdm.write("Early stopping triggered. Training stopped.")
        break

print("Training completed")

# Save the training log
with open(log_file, 'w') as f:
    json.dump(log, f, indent=4)
print(f"Training log saved as: {log_file}")

# Evaluate on validation set
print("\nEvaluating best model on validation set...")
model.load_state_dict(torch.load(f'runs/{model_name}_best.pth'))
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

# Calculate metrics
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
print("\nClassification Report:")
print(classification_report(val_true_labels, val_predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(val_true_labels, val_predictions))

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
plt.plot(log['epochs'], log['train_loss'], label='Train Loss')
plt.plot(log['epochs'], log['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(log['epochs'], log['train_acc'], label='Train Accuracy')
plt.plot(log['epochs'], log['val_acc'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.tight_layout()
plt.savefig(f'logs/{model_name}_training_curves.png')
print(f"Training curves saved as: logs/{model_name}_training_curves.png")