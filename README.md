# Models Microservice

This project provides a FastAPI-based microservice for:
- Fish disease detection from images
- Machine failure prediction from sensor data

## Project Structure

```
app/
  services/

main.py        # FastAPI entrypoint
requirements.txt
Dockerfile
.gitignore
.dockerignore
```

## Endpoints

### 1. Fish Disease Detection
- **POST** `/predict_disease`
- **Input:** Image file (form-data, key: `file`)
- **Output:** `{ "result": "<disease_name>" }`

### 2. Machine Failure Prediction
- **POST** `/predict_failure`
- **Input:** JSON `{ "data": [float, float, float, float, float] }`
- **Output:** `{ "result": "<prediction>" }`

## Running Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the server:
   ```bash
   uvicorn main:app --reload
   ```

## Running with Docker

1. Build the image:
   ```bash
   docker build -t models-microservice .
   ```
2. Run the container:
   ```bash
   docker run -p 8000:8000 models-microservice
   ```

## Notes
- Place your trained models in the `models/` directory.
- Notebooks are for development only and are ignored in Docker and git.
