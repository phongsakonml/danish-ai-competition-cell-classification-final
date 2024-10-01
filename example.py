from utils import tif_to_ndarray
from model import predict
import base64
import cv2
CELL_IMG = "data/training/007.tif" 

image = tif_to_ndarray(CELL_IMG) # For local testing with tif files

# Convert image to base64 string
_, buffer = cv2.imencode('.tif', image)
img_base64 = base64.b64encode(buffer).decode('utf-8')

sample_prediction = predict(img_base64)
print(f"Prediction for {CELL_IMG}: {sample_prediction}")