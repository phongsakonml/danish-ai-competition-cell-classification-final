import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, exposure
from sklearn.model_selection import train_test_split, KFold
import torch
import torch.nn as nn
from torchvision import models
from datetime import datetime
import json
from tqdm import tqdm
from torch.optim import lr_scheduler
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix, balanced_accuracy_score, roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Load the data
data_dir = 'data/training_balanced'  # Use the balanced dataset
labels_file = 'data/training_balanced.csv'
model_name = "god"

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

# Data Augmentation
augment = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.7),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussNoise(p=0.2),  # Added Gaussian noise for more variability
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
        self.labels = torch.tensor(labels, dtype=torch.long)  # Ensure labels are long tensors
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

# Mixup augmentation
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Update FocalLoss to accept class weights
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, class_weights=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights.float() if class_weights is not None else None  # Ensure class_weights is Float

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.class_weights)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

# Model Selection: Transfer Learning with EfficientNet-B0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
        nn.Dropout(0.6),  # Increased dropout rate
        nn.Linear(640, 320),
        nn.ReLU(),
        nn.Dropout(0.6),  # Increased dropout rate
        nn.Linear(320, 160),
        nn.ReLU(),
        nn.Dropout(0.6),  # Increased dropout rate
        nn.Linear(160, 2)
    )
    return model

# Create a directory for saving models and logs
os.makedirs('runs', exist_ok=True)

def get_run_name(model_name, base_dir='runs'):
    run_dir = os.path.join(base_dir, model_name)  # Use model_name as the directory name
    os.makedirs(run_dir, exist_ok=True)  # Create directory if it doesn't exist
    return model_name, run_dir

run_name, run_dir = get_run_name(model_name)

# Create a SummaryWriter for this run
writer = SummaryWriter(log_dir=run_dir)

def save_augmented_samples(images, labels, transform=None, num_samples=5, model_name='augmented_samples'):
    """Save original and augmented images to a directory."""
    save_dir = model_name  # Use model_name as the directory name
    os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
    
    for i in range(num_samples):
        # Original image
        original_image = images[i]
        original_image_path = os.path.join(save_dir, f"original_label_{labels[i]}.png")
        plt.imsave(original_image_path, original_image)

        # Augmented image
        if transform:
            augmented = transform(image=original_image)
            aug_image = augmented['image'].permute(1, 2, 0).numpy()
            aug_image = (aug_image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
            aug_image = aug_image.astype(np.uint8)
            augmented_image_path = os.path.join(save_dir, f"augmented_label_{labels[i]}.png")
            plt.imsave(augmented_image_path, aug_image)

# Save some samples before and after augmentation
save_augmented_samples(images, labels, transform=augment, model_name=model_name)

# Ensure the model is moved to the GPU when initialized
model = initialize_model().to(device)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, fold, num_epochs=100, patience=7):
    best_val_loss = float('inf')
    early_stopping_counter = 0
    log = {
        'epochs': [], 'train_losses': [], 'val_losses': [], 'train_accuracies': [], 'val_accuracies': [],
        'train_f1': [], 'val_f1': [], 'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': [], 'train_bal_acc': [], 'val_bal_acc': [],
        'train_auc': [], 'val_auc': []
    }
    best_log = None

    scaler = GradScaler()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_predictions = []
        train_true_labels = []
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Mixup augmentation
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=0.4)
            
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (lam * (predicted == targets_a).float() + (1 - lam) * (predicted == targets_b).float()).sum().item()
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
        train_auc = roc_auc_score(train_true_labels, train_predictions)
        
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
        val_auc = roc_auc_score(val_true_labels, val_predictions)
        
        # Log metrics
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
        log['train_auc'].append(train_auc)
        log['val_auc'].append(val_auc)
        log['epochs'].append(epoch + 1)

        # Add scalar logging
        for name, value in [
            ('Loss/train', train_loss), ('Loss/val', val_loss),
            ('Accuracy/train', train_acc), ('Accuracy/val', val_acc),
            ('F1_Score/train', train_f1), ('F1_Score/val', val_f1),
            ('Precision/train', train_precision), ('Precision/val', val_precision),
            ('Recall/train', train_recall), ('Recall/val', val_recall),
            ('Balanced_Accuracy/train', train_bal_acc), ('Balanced_Accuracy/val', val_bal_acc),
            ('AUC/train', train_auc), ('AUC/val', val_auc)
        ]:
            writer.add_scalar(name, value, epoch)

        writer.flush()
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")
        print(f"Train Precision: {train_precision:.4f}, Val Precision: {val_precision:.4f}")
        print(f"Train Recall: {train_recall:.4f}, Val Recall: {val_recall:.4f}")
        print(f"Train Balanced Accuracy: {train_bal_acc:.4f}, Val Balanced Accuracy: {val_bal_acc:.4f}")
        print(f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")
        
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(run_dir, f'best_model_fold_{fold}.pth'))
            print("New best model saved!")
            early_stopping_counter = 0
            best_log = {k: v.copy() if isinstance(v, list) else v for k, v in log.items()}
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print("Early stopping!")
                break

    # Save training log and plot curves
    with open(os.path.join(run_dir, f'training_log_fold_{fold}.json'), 'w') as f:
        json.dump(log, f, indent=4)

    if best_log:
        with open(os.path.join(run_dir, f'best_model_log_fold_{fold}.json'), 'w') as f:
            json.dump(best_log, f, indent=4)

    plot_training_curves(log, fold, 'all')
    if best_log:
        plot_training_curves(best_log, fold, 'best')

    # Save the model for the current fold
    torch.save(model.state_dict(), os.path.join(run_dir, f'model_fold_{fold}.pth'))

    return model, best_log['val_f1'][-1] if best_log else None

def plot_training_curves(log, fold, model_type='all'):
    plt.figure(figsize=(15, 12))
    
    metrics = [
        ('Loss', 'losses'), ('Accuracy', 'accuracies'),
        ('F1 Score', 'f1'), ('Precision', 'precision'),
        ('Recall', 'recall'), ('Balanced Accuracy', 'bal_acc'),
        ('AUC', 'auc')
    ]
    
    for i, (metric_name, metric_key) in enumerate(metrics, 1):
        plt.subplot(4, 2, i)
        plt.plot(log['epochs'], log[f'train_{metric_key}'], label=f'Train {metric_name}')
        plt.plot(log['epochs'], log[f'val_{metric_key}'], label=f'Validation {metric_name}')
        plt.xlabel('Epochs')
        plt.ylabel(metric_name)
        plt.title(f'{model_type.capitalize()} Model - {metric_name} - Fold {fold}')
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f'{model_type}_model_training_curves_fold_{fold}.png'))
    plt.close()

def main():
    global images, labels
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)

    # Implement k-fold cross-validation
    k_folds = 10
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    fold_results = []
    best_fold = None
    best_f1 = 0
    best_model_state = None

    # Calculate class weights
    class_weights = torch.tensor([1.0, (y_train == 0).sum() / (y_train == 1).sum()], dtype=torch.float32).to(device)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        print(f"Fold {fold}/{k_folds}")
        
        X_train_fold = [X_train[i] for i in train_idx]
        y_train_fold = [y_train[i] for i in train_idx]
        X_val_fold = [X_train[i] for i in val_idx]
        y_val_fold = [y_train[i] for i in val_idx]

        train_dataset = CellDataset(X_train_fold, y_train_fold, transform=augment)
        val_dataset = CellDataset(X_val_fold, y_val_fold, transform=val_transform)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

        model = initialize_model().to(device)
        criterion = FocalLoss(class_weights=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

        model, fold_best_f1 = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, fold, num_epochs=100, patience=7)
        
        if fold_best_f1 > best_f1:
            best_f1 = fold_best_f1
            best_fold = fold
            best_model_state = model.state_dict()

        fold_results.append(fold_best_f1)
        print(f"Fold {fold} Best F1 Score: {fold_best_f1:.4f}")

    print(f"Cross-validation results: {fold_results}")
    print(f"Mean F1 Score: {np.mean(fold_results):.4f} (+/- {np.std(fold_results):.4f})")
    print(f"Best fold: {best_fold} with F1 score: {best_f1:.4f}")

    # Save the best model from cross-validation
    torch.save(best_model_state, os.path.join(run_dir, 'best_cv_model.pth'))
    print(f"Best cross-validation model saved as 'best_cv_model.pth' in {run_dir}")

    # Train on the entire training set
    train_dataset = CellDataset(X_train, y_train, transform=augment)
    val_dataset = CellDataset(X_val, y_val, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    model = initialize_model().to(device)
    criterion = FocalLoss(class_weights=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    model, final_f1 = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 'final', num_epochs=100, patience=7)

    print("Training completed")
    writer.close()

    torch.save(model.state_dict(), os.path.join(run_dir, 'final_model.pth'))
    print(f"Final model saved in {run_dir}")

    # Compare the final model with the best cross-validation model
    if final_f1 > best_f1:
        print(f"Final model (F1: {final_f1:.4f}) outperformed best cross-validation model (F1: {best_f1:.4f})")
        print("Consider using 'final_model.pth' for inference")
    else:
        print(f"Best cross-validation model (F1: {best_f1:.4f}) outperformed final model (F1: {final_f1:.4f})")
        print("Consider using 'best_cv_model.pth' for inference")

if __name__ == '__main__':
    main()