import pandas as pd
import os
import torch
from model import predict
import utils
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Load the data
labels_file = 'data/training.csv'
data_dir = 'data/training'

# Read labels and strip whitespace from column names
labels_df = pd.read_csv(labels_file)
labels_df.columns = labels_df.columns.str.strip()

# Initialize counters for scoring
n_0 = 0  # Count of label 0 (heterogeneous)
n_1 = 0  # Count of label 1 (homogeneous)
a_0 = 0  # Correct predictions for label 0
a_1 = 0  # Correct predictions for label 1

# List to store misclassified images
misclassified = []

# Iterate through the dataset
for index, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
    img_id = str(row['image_id']).strip()
    img_path = os.path.join(data_dir, f"{img_id}.tif")
    
    if os.path.exists(img_path):
        actual_label = int(row['is_homogenous'])
        
        # Update counts
        if actual_label == 0:
            n_0 += 1
        elif actual_label == 1:
            n_1 += 1
        
        # Load and preprocess the image
        img = utils.tif_to_ndarray(img_path)
        
        # Plot the first few images to verify
        if index < 5:
            utils.plot_image(img, f"Image ID: {img_id}, Actual Label: {actual_label}")
        
        # Encode image to base64 to simulate API input
        img_base64 = utils.encode_image(img)
        
        # Predict using the model
        predicted_label = predict(img_base64)
        
        # Update correct predictions
        if predicted_label == 0 and actual_label == 0:
            a_0 += 1
        elif predicted_label == 1 and actual_label == 1:
            a_1 += 1
        else:
            misclassified.append((img_id, actual_label, predicted_label))

# Calculate the score
if n_0 > 0 and n_1 > 0:
    score = (a_0 * a_1) / (n_0 * n_1)
else:
    score = 0

# Print results
print(f"Total label 0 (heterogeneous): {n_0}")
print(f"Total label 1 (homogeneous): {n_1}")
print(f"Correct predictions for label 0: {a_0}")
print(f"Correct predictions for label 1: {a_1}")
print(f"Score: {score:.4f}")

# Log misclassified images
print("\nMisclassified images:")
for img_id, actual, predicted in misclassified:
    print(f"Image ID: {img_id}, Actual: {actual}, Predicted: {predicted}")

# Calculate and print accuracy for each class
accuracy_0 = a_0 / n_0 if n_0 > 0 else 0
accuracy_1 = a_1 / n_1 if n_1 > 0 else 0
print(f"\nAccuracy for class 0 (heterogeneous): {accuracy_0:.4f}")
print(f"Accuracy for class 1 (homogeneous): {accuracy_1:.4f}")

# Identify where the model is struggling
if accuracy_0 < accuracy_1:
    print("\nThe model is struggling more with heterogeneous (class 0) images.")
elif accuracy_1 < accuracy_0:
    print("\nThe model is struggling more with homogeneous (class 1) images.")
else:
    print("\nThe model seems to perform equally on both classes.")