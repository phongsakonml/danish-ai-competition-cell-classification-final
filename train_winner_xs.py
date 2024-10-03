import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, exposure
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch
import torch.nn as nn
from torchvision import models, transforms
from datetime import datetime
import json
from tqdm import tqdm
from torch.optim import lr_scheduler
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix, balanced_accuracy_score, precision_recall_curve, classification_report
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset

# Load the data
data_dir = 'data/training'  # Ensure this points to the correct directory
labels_file = 'data/training.csv'

# Read labels
labels_df = pd.read_csv(labels_file)

# Print information about the CSV file
print("CSV file columns:")
print(labels_df.columns)

# Strip whitespace from column names
labels_df.columns = labels_df.columns.str.strip()

# Check if expected columns exist
if 'image_id' not in labels_df.columns or 'is_homogenous' not in labels_df.columns:
    raise KeyError("Expected columns 'image_id' and 'is_homogenous' not found in the CSV file.")

# Load images and labels
images = []  # Ensure images is defined before use
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

# Step 1: Data Balancing by Undersampling the Majority Class
# Separate heterogeneous and homogeneous cells
hetero_images = [img for img, label in zip(images, labels) if label == 0]
homo_images = [img for img, label in zip(images, labels) if label == 1]

print(f"Original heterogeneous cells: {len(hetero_images)}")
print(f"Original homogeneous cells: {len(homo_images)}")

# Undersample the majority class (heterogeneous)
np.random.seed(42)  # for reproducibility
undersampled_hetero_indices = np.random.choice(len(hetero_images), len(homo_images), replace=False)
undersampled_hetero_images = [hetero_images[i] for i in undersampled_hetero_indices]

# Combine undersampled heterogeneous and all homogeneous images
balanced_images = undersampled_hetero_images + homo_images
balanced_labels = np.array([0] * len(undersampled_hetero_images) + [1] * len(homo_images))

print(f"Total images after balancing: {len(balanced_images)}")
print(f"Label distribution after balancing: {np.bincount(balanced_labels)}")

def plot_image(ax, image, title):
    """Function to plot an image on a given axis."""
    ax.imshow(image)
    ax.set_title(title)
    ax.axis('off')

# Plot sample images
fig, axes = plt.subplots(2, 2, figsize=(15, 15))
plot_image(axes[0, 0], undersampled_hetero_images[0], "Heterogeneous")
plot_image(axes[0, 1], undersampled_hetero_images[1], "Heterogeneous")
plot_image(axes[1, 0], homo_images[0], "Homogeneous")
plot_image(axes[1, 1], homo_images[1], "Homogeneous")
plt.tight_layout()
plt.show()

def get_run_name(model_name, base_dir='runs'):
    counter = 0
    while True:
        run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{counter}"
        run_dir = os.path.join(base_dir, run_name)
        try:
            os.makedirs(run_dir)  # Attempt to create the directory
            return run_name, run_dir
        except FileExistsError:
            counter += 1  # Increment counter if the directory already exists

# Data Augmentation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CellDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def get_original(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

    def __getitem__(self, idx):
        image, label = self.get_original(idx)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Model Selection: Transfer Learning with EfficientNet-B0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model initialization
def initialize_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    # Modify the first convolutional layer to accept 3-channel input
    model.features[0][0] = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    
    # Ensure the model is in training mode for proper batch normalization behavior
    model.train()  # Change to train mode

    for param in model.parameters():
        param.requires_grad = False  # Freeze all layers initially
    
    # Unfreeze the last few layers for fine-tuning
    for param in model.features[-1].parameters():
        param.requires_grad = True
    
    model.classifier = nn.Sequential(
        nn.Linear(in_features=1280, out_features=512),  # Adjusted for EfficientNet-B0
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 2)
    )
    return model

# Create a directory for saving models and logs
os.makedirs('runs', exist_ok=True)
os.makedirs('logs', exist_ok=True)

model_name = "efficientnet_b0"
run_name, run_dir = get_run_name(model_name)

# Create a SummaryWriter for this run
writer = SummaryWriter(log_dir=run_dir)

def visualize_augmentations(dataset, num_samples=5):
    """Visualize original and augmented images."""
    fig, axes = plt.subplots(2, num_samples, figsize=(20, 8))
    
    for i in range(num_samples):
        # Original image
        image, label = dataset.get_original(i)
        axes[0, i].imshow(image, cmap='gray')
        axes[0, i].set_title(f"Original Label: {label}")
        axes[0, i].axis('off')
        
        # Augmented image
        aug_image, _ = dataset[i]
        aug_image = aug_image.permute(1, 2, 0).numpy()
        aug_image = (aug_image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
        aug_image = aug_image.astype(np.uint8)
        axes[1, i].imshow(aug_image[:,:,0], cmap='gray')
        axes[1, i].set_title(f"Augmented Label: {label}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # Initialize best_val_loss
    best_val_loss = float('inf')
    patience = 10  # Increased patience for early stopping
    early_stopping_counter = 0
    num_epochs = 100  # Increased number of epochs

    # Initialize log dictionary
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
   
   # Split the data into training and validation sets
    X_train, X_val = train_test_split(balanced_images, test_size=0.2, random_state=42)
    y_train, y_val = train_test_split(balanced_labels, test_size=0.2, random_state=42)

    # Further split the validation set into validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42) 


    # Dataset and DataLoader
    train_dataset = CellDataset(X_train, y_train, transform=transform)
    val_dataset = CellDataset(X_val, y_val, transform=val_transform)
    test_dataset = CellDataset(X_test, y_test, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Modify the transform to not normalize the images for visualization
    transform_for_vis = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Use this when calling visualize_augmentations
    train_dataset_for_vis = CellDataset(X_train, y_train, transform=transform_for_vis)
    visualize_augmentations(train_dataset_for_vis)

    # Model initialization
    model = initialize_model().to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

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
        train_f1 = f1_score(train_true_labels, train_predictions, average='weighted', zero_division=0)
        train_precision, train_recall, _, _ = precision_recall_fscore_support(
            train_true_labels, train_predictions, average='weighted', zero_division=0
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
                
                del inputs, labels, outputs, loss
        
        val_loss = val_loss / len(val_dataset)
        val_acc = val_correct / len(val_dataset)
        
        # Calculate validation metrics
        val_f1 = f1_score(val_true_labels, val_predictions, average='weighted', zero_division=0)
        val_precision, val_recall, _, _ = precision_recall_fscore_support(
            val_true_labels, val_predictions, average='weighted', zero_division=0
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

        # Update log
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

    print("Training completed")
    writer.close()

    # Save the final model
    torch.save(model.state_dict(), os.path.join(run_dir, 'final_model.pth'))
    print(f"Final model saved in {run_dir}")

    # Save the training log
    training_log = {
        "model_name": model_name,
        "run_name": run_name,
        "epochs_trained": epoch + 1,
        "best_val_loss": best_val_loss,
        "metrics": log
    }
    with open(os.path.join(run_dir, 'training_log.json'), 'w') as f:
        json.dump(training_log, f, indent=4)
    print(f"Training log saved in {run_dir}")

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
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
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
            'Confusion Matrix': cm.tolist()  # Convert to list for JSON serialization
        }

    results = evaluate_model(val_true_labels, val_predictions)
    print(results)

    print("\nClassification Report:")
    print(classification_report(val_true_labels, val_predictions, zero_division=0))

    # Optimize decision threshold
    precision_vals, recall_vals, thresholds = precision_recall_curve(val_true_labels, val_probabilities)
    f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]

    print(f"\nOptimal threshold: {optimal_threshold:.4f}")

    # Apply optimal threshold
    optimized_predictions = (np.array(val_probabilities) >= optimal_threshold).astype(int)

    print("\nOptimized Classification Report:")
    print(classification_report(val_true_labels, optimized_predictions, zero_division=0))

    print("\nOptimized Confusion Matrix:")
    print(confusion_matrix(val_true_labels, optimized_predictions))

    print("\nTraining and evaluation completed successfully!")

    # Plot training and validation curves
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(log['epochs'], log['train_losses'], label='Train Loss')
    plt.plot(log['epochs'], log['val_losses'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(2, 2, 2)
    plt.plot(log['epochs'], log['train_accuracies'], label='Train Accuracy')
    plt.plot(log['epochs'], log['val_accuracies'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 2, 3)
    plt.plot(log['epochs'], log['train_f1'], label='Train F1 Score')
    plt.plot(log['epochs'], log['val_f1'], label='Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.title('Training and Validation F1 Score')

    plt.subplot(2, 2, 4)
    plt.plot(log['epochs'], log['train_precision'], label='Train Precision')
    plt.plot(log['epochs'], log['val_precision'], label='Validation Precision')
    plt.plot(log['epochs'], log['train_recall'], label='Train Recall')
    plt.plot(log['epochs'], log['val_recall'], label='Validation Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.legend()
    plt.title('Training and Validation Precision & Recall')

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'training_plot.png'))
    plt.close()

    # Cross-Validation with Stratified K-Fold
    print("\nStarting Stratified K-Fold Cross-Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(balanced_images, balanced_labels)):
        print(f"\nFold {fold+1}")
        
        # Ensure the dataset is correctly indexed
        train_images = [balanced_images[i] for i in train_idx]
        train_labels = balanced_labels[train_idx]
        val_images = [balanced_images[i] for i in val_idx]
        val_labels = balanced_labels[val_idx]

        train_loader_cv = DataLoader(
            CellDataset(train_images, train_labels, transform=transform), 
            batch_size=32, shuffle=True, num_workers=0
        )
        val_loader_cv = DataLoader(
            CellDataset(val_images, val_labels, transform=val_transform), 
            batch_size=32, shuffle=False, num_workers=0
        )
        
        # Initialize a new model for each fold
        model_cv = initialize_model().to(device)
        optimizer_cv = torch.optim.Adam(model_cv.parameters(), lr=1e-3)
        scheduler_cv = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_cv, mode='min', patience=5, factor=0.5, verbose=True)
        criterion_cv = nn.CrossEntropyLoss()
        
        best_val_loss_cv = float('inf')
        early_stopping_counter_cv = 0
        patience_cv = 10
        
        for epoch_cv in range(num_epochs):
            model_cv.train()
            train_loss_cv = 0.0
            train_correct_cv = 0
            train_predictions_cv = []
            train_true_labels_cv = []
            
            for inputs_cv, labels_cv in tqdm(train_loader_cv, desc=f"Fold {fold+1} Epoch {epoch_cv+1}/{num_epochs}", leave=False):
                try:
                    inputs_cv, labels_cv = inputs_cv.to(device), labels_cv.to(device)
                    
                    optimizer_cv.zero_grad()
                    outputs_cv = model_cv(inputs_cv)
                    loss_cv = criterion_cv(outputs_cv, labels_cv)
                    loss_cv.backward()
                    optimizer_cv.step()
                    
                    train_loss_cv += loss_cv.item() * inputs_cv.size(0)
                    _, predicted_cv = torch.max(outputs_cv, 1)
                    train_correct_cv += (predicted_cv == labels_cv).sum().item()
                    train_predictions_cv.extend(predicted_cv.cpu().numpy())
                    train_true_labels_cv.extend(labels_cv.cpu().numpy())
                except Exception as e:
                    print(f"Error in training loop: {e}")
                    print(f"Input shape: {inputs_cv.shape}, Label shape: {labels_cv.shape}")
                    continue
                
            train_loss_cv = train_loss_cv / len(train_loader_cv.dataset)
            train_acc_cv = train_correct_cv / len(train_loader_cv.dataset)
            
            # Calculate training metrics
            train_f1_cv = f1_score(train_true_labels_cv, train_predictions_cv, average='weighted', zero_division=0)
            train_precision_cv, train_recall_cv, _, _ = precision_recall_fscore_support(
                train_true_labels_cv, train_predictions_cv, average='weighted', zero_division=0
            )
            train_bal_acc_cv = balanced_accuracy_score(train_true_labels_cv, train_predictions_cv)
            
            model_cv.eval()
            val_loss_cv = 0.0
            val_correct_cv = 0
            val_predictions_cv = []
            val_true_labels_cv = []
            
            with torch.no_grad():
                for inputs_cv, labels_cv in tqdm(val_loader_cv, desc=f"Fold {fold+1} Validation", leave=False):
                    inputs_cv, labels_cv = inputs_cv.to(device), labels_cv.to(device)
                    outputs_cv = model_cv(inputs_cv)
                    loss_cv = criterion_cv(outputs_cv, labels_cv)
                    val_loss_cv += loss_cv.item() * inputs_cv.size(0)
                    _, predicted_cv = torch.max(outputs_cv, 1)
                    val_correct_cv += (predicted_cv == labels_cv).sum().item()
                    val_predictions_cv.extend(predicted_cv.cpu().numpy())
                    val_true_labels_cv.extend(labels_cv.cpu().numpy())
                    
                    del inputs_cv, labels_cv, outputs_cv, loss_cv
            
            val_loss_cv = val_loss_cv / len(val_loader_cv.dataset)
            val_acc_cv = val_correct_cv / len(val_loader_cv.dataset)
            
            # Calculate validation metrics
            val_f1_cv = f1_score(val_true_labels_cv, val_predictions_cv, average='weighted', zero_division=0)
            val_precision_cv, val_recall_cv, _, _ = precision_recall_fscore_support(
                val_true_labels_cv, val_predictions_cv, average='weighted', zero_division=0
            )
            val_bal_acc_cv = balanced_accuracy_score(val_true_labels_cv, val_predictions_cv)
            
            # Update scheduler step
            scheduler_cv.step(val_loss_cv)
            
            print(f"\nFold {fold+1} Epoch {epoch_cv+1}/{num_epochs}")
            print(f"Train Loss: {train_loss_cv:.4f}, Train Acc: {train_acc_cv:.4f}")
            print(f"Val Loss: {val_loss_cv:.4f}, Val Acc: {val_acc_cv:.4f}")
            print(f"Train F1: {train_f1_cv:.4f}, Val F1: {val_f1_cv:.4f}")
            print(f"Train Precision: {train_precision_cv:.4f}, Val Precision: {val_precision_cv:.4f}")
            print(f"Train Recall: {train_recall_cv:.4f}, Val Recall: {val_recall_cv:.4f}")
            print(f"Train Balanced Accuracy: {train_bal_acc_cv:.4f}, Val Balanced Accuracy: {val_bal_acc_cv:.4f}")
            
            if val_loss_cv < best_val_loss_cv:
                best_val_loss_cv = val_loss_cv
                torch.save(model_cv.state_dict(), os.path.join(run_dir, f'best_model_fold{fold+1}.pth'))
                print("New best model for this fold saved!")
                early_stopping_counter_cv = 0
            else:
                early_stopping_counter_cv += 1
                if early_stopping_counter_cv >= patience_cv:
                    print("Early stopping for this fold!")
                    break

        # Evaluate fold
        fold_results = evaluate_model(val_true_labels_cv, val_predictions_cv)
        print(f"Fold {fold+1} Results: {fold_results}")
        cv_scores.append(fold_results['F1 Score'])
    
    print(f"\nCross-Validation F1-Score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

if __name__ == '__main__':
    main()