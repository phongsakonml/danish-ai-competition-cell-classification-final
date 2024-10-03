import pandas as pd
import os
import torch
from model import predict
import utils
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import glob

# Print current working directory
print("Current working directory:", os.getcwd())

# List contents of the runs directory
runs_dir = 'cell-class-final/runs'
print("Contents of runs directory:", os.listdir(runs_dir))  # This will show the contents of the runs directory

# Load the data
labels_file = 'data/training.csv'
data_dir = 'data/training'

# Read labels and strip whitespace from column names
labels_df = pd.read_csv(labels_file)
labels_df.columns = labels_df.columns.str.strip()

# Create a results dictionary to store the results for each model
results = {}

# Get the specific best_model.pth file
model_paths = ['runs/first_1003_1451_0/best_model.pth']  # Ensure this path is correct

# Iterate through each model path
for model_path in model_paths:
    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        continue
    
    # Load the model with weights_only=True to avoid future warnings
    model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    
    # Initialize counters for scoring
    n_0, n_1, a_0, a_1 = 0, 0, 0, 0
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
            predicted_label = predict(img_base64)  # Ensure this uses the loaded model
            
            # Update correct predictions
            if predicted_label == 0 and actual_label == 0:
                a_0 += 1
            elif predicted_label == 1 and actual_label == 1:
                a_1 += 1
            else:
                misclassified.append((img_id, actual_label, predicted_label))

    # Calculate the score for the current model
    if n_0 > 0 and n_1 > 0:
        score = (a_0 * a_1) / (n_0 * n_1)
    else:
        score = 0

    # Store results for the current model
    results[model_path] = {
        "total_label_0": n_0,
        "total_label_1": n_1,
        "correct_predictions_0": a_0,
        "correct_predictions_1": a_1,
        "score": score,
        "misclassified": misclassified,
        "accuracy_0": a_0 / n_0 if n_0 > 0 else 0,
        "accuracy_1": a_1 / n_1 if n_1 > 0 else 0,
    }

# Save results to result.json in the experiment folder
os.makedirs('experiment', exist_ok=True)  # Ensure the experiment folder exists
with open('experiment/result.json', 'w') as json_file:
    json.dump(results, json_file, indent=4)

# Print completion message
print("Results saved to experiment/result.json")