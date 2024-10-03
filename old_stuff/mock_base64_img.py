import base64
import numpy as np
import cv2

# Create a small blank image
img = np.zeros((224, 224), dtype=np.uint8)
_, buffer = cv2.imencode('.png', img)
img_str = base64.b64encode(buffer).decode('utf-8')

print(img_str)
