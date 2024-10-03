import uvicorn
from fastapi import FastAPI, HTTPException
import datetime
import time
from model import predict
from loguru import logger
from pydantic import BaseModel
from typing import List
import base64

HOST = "0.0.0.0"
PORT = 8080 

class CellClassificationPredictRequestDto(BaseModel):
    cell: str

class CellClassificationPredictResponseDto(BaseModel):
    is_homogenous: int

app = FastAPI()
start_time = time.time()

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
        # Pass the base64 string directly to predict function
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
