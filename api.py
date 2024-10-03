import uvicorn
from fastapi import FastAPI, HTTPException
import datetime
import time
from model import predict
from loguru import logger
from pydantic import BaseModel
from typing import List
import os
import base64
import cv2
import numpy as np

HOST = "0.0.0.0"
PORT = 8080 

class CellClassificationPredictRequestDto(BaseModel):
    cell: str

class CellClassificationPredictResponseDto(BaseModel):
    is_homogenous: int

app = FastAPI()
start_time = time.time()

# Create a directory to save validation images
VAL_SET_DIR = 'val_set_delete_this_later'
os.makedirs(VAL_SET_DIR, exist_ok=True)

def save_base64_image(base64_string, file_path):
    try:
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imwrite(file_path, img)
    except Exception as e:
        logger.error(f"Error saving base64 image: {str(e)}")

@app.get('/api')
def hello():
    return {
        "service": "cell-segmentation-usecase",
        "uptime": '{}'.format(datetime.timedelta(seconds=time.time() - start_time))
    }

@app.get('/')
def index():
    return "Your endpoint is running!"

@app.post('/predict', response_model=CellClassificationPredictResponseDto)
def predict_endpoint(request: CellClassificationPredictRequestDto):
    try:
        # Save the received base64 image
        image_count = len(os.listdir(VAL_SET_DIR))
        image_filename = f"val_image_{image_count}.png"
        image_path = os.path.join(VAL_SET_DIR, image_filename)
        save_base64_image(request.cell, image_path)
        logger.info(f"Saved validation image: {image_path}")

        predicted_homogenous_state = predict(request.cell)
        if predicted_homogenous_state == -1:
            logger.error("Prediction returned error code -1")
            raise HTTPException(status_code=500, detail="Prediction error")
        
        response = CellClassificationPredictResponseDto(
            is_homogenous=predicted_homogenous_state
        )
        logger.info(f"Successful prediction: {predicted_homogenous_state}")
        return response
    except Exception as e:
        logger.error(f"Error in predict_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == '__main__':
    uvicorn.run(
        'api:app',
        host=HOST,
        port=PORT
    )
