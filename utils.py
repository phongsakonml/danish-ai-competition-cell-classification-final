import numpy as np
import cv2
import base64 
import matplotlib.pyplot as plt
import os

def decode_image(encoded_img) -> np.ndarray:
    np_img = np.frombuffer(base64.b64decode(encoded_img), np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    return img

def load_image(image_id: int) -> np.ndarray:
    # Format the image ID to match the naming convention (e.g., '001', '011', etc.)
    formatted_id = f"{image_id:03d}"  # Format as three digits
    image_path = f'cell-class-final/data/training/{formatted_id}.tif'  # Adjust the path and extension as needed
    print(f"Attempting to load image from: {image_path}")  # Debugging print statement
    img_array = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_array is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    return img_array

def tif_to_ndarray(tif_path):
    img_array = cv2.imread(tif_path, cv2.IMREAD_COLOR)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    if img_array.dtype == np.uint16:
        img_array = (img_array / 256).astype(np.uint8)
    return img_array

def load_sample(enc_img: str):
    image = decode_image(enc_img)  # For decoding validation and evaluation files
    return image

def encode_image(image_data) -> str:
    _, buffer = cv2.imencode('.png', image_data)
    return base64.b64encode(buffer).decode('utf-8')

def plot_image(image: np.ndarray, title: str):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()