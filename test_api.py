import requests
import base64
from utils import tif_to_ndarray
import cv2

# URL of your API
url = "http://localhost:4321/predict"

# Path to a sample image
image_path = "data/training/007.tif"

# Load and encode the image
image = tif_to_ndarray(image_path)
_, buffer = cv2.imencode('.tif', image)
img_str = base64.b64encode(buffer).decode('utf-8')

# Print the first 50 characters of the base64 encoded string
print("Base64 Encoded Image String (first 50 chars):")
print(img_str[:50])  # Print only the first 50 characters

# Prepare the request payload
payload = {
    "cell": img_str
}

# Send POST request to the API
response = requests.post(url, json=payload)

# Print the response status code and raw data
print(response.status_code)
print(response.text)  # Print raw response data
print(response.json())  # Print JSON response data