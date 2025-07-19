



from fastapi import FastAPI, UploadFile, File, Form
from app.services.machine_failure_detection import predict_failure
from app.services.fish_disease_detection import predict_disease
import tempfile
import shutil
import numpy as np
import os
from typing import List
from pydantic import BaseModel, Field


app = FastAPI()

from fastapi import Request

class PredictionRequest(BaseModel):
    data: List[float] = Field(..., min_items=5, max_items=5)

@app.post('/predict_failure')
async def predict_failure_endpoint(request: PredictionRequest):
    features = request.data
    arr = np.array([features])  # Convert to 2D array as expected by the model
    prediction_result = predict_failure(arr)
    return {"result": prediction_result}

@app.post('/predict_disease')
async def predict_disease_endpoint(file: UploadFile = File(...)):
    # Save uploaded file to a temp file and pass its path
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        result = predict_disease(tmp_path)
    finally:
        os.remove(tmp_path)
    return {"result": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
