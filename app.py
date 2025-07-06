# app.py
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import io

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import your model class
from deepfake_starter_code import DeepfakeDetector

# --- Configuration ---
MODEL_PATH = Path('./output/best_deepfake_model.pth')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 1. Initialize FastAPI App ---
app = FastAPI(
    title="Deepfake Detection API",
    description="An API to detect whether an image is a deepfake or not.",
    version="1.0.0"
)

# --- 2. Load Model and Transforms ---
model = None
transform = None

def load_model():
    """Load the trained model and transformation pipeline."""
    global model, transform
    
    # Load model architecture
    model = DeepfakeDetector(backbone='efficientnet', num_classes=2)
    
    # Load trained weights
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please ensure the model is trained and saved.")
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(DEVICE)))
    model.to(DEVICE)
    model.eval()

    # Define the same validation transform used during training
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    print("Model and transforms loaded successfully.")

# --- 3. Define Prediction Endpoint ---
@app.on_event("startup")
async def startup_event():
    """Load the model on server startup."""
    load_model()

@app.get("/", summary="Health Check")
def read_root():
    """Root endpoint to check if the API is running."""
    return {"status": "ok", "message": "Welcome to the Deepfake Detection API!"}

@app.post("/predict/", summary="Get Deepfake Prediction")
async def predict(file: UploadFile = File(...)):
    """
    Receives an image file, processes it, and returns the prediction.
    - **file**: An image file (e.g., .jpg, .png).
    """
    if not model or not transform:
        return JSONResponse(status_code=503, content={"error": "Model is not loaded."})

    # Read image from upload
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    image_np = np.array(image)

    # Apply transformations
    transformed_image = transform(image=image_np)['image']
    image_tensor = transformed_image.unsqueeze(0).to(DEVICE)

    # --- Make Prediction ---
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        
    # Get prediction and confidence
    confidence, predicted_class_idx = torch.max(probabilities, 0)
    predicted_class_name = 'Fake' if predicted_class_idx.item() == 1 else 'Real'

    return JSONResponse(content={
        "filename": file.filename,
        "prediction": predicted_class_name,
        "confidence": f"{confidence.item():.4f}"
    })

# --- To run this app locally ---
# 1. Install FastAPI and Uvicorn: pip install fastapi "uvicorn[standard]" python-multipart
# 2. Run the server: uvicorn app:app --reload 