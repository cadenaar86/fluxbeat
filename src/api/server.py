from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import tempfile
from .schemas import PredictionResponse
from ..inference.engine import InferenceEngine
from .. import config

# Initialize App
app = FastAPI(
    title="FluxBeat API",
    description="Real-time Music Genre Classification API",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Engine
engine = None

@app.on_event("startup")
def load_model():
    global engine
    # In a real deployment, paths would be env vars or standard locations
    # For this demo, we use the project paths.
    # Note: If no trained model exists, engine prints a warning but starts.
    # We might want to look for 'best_model.pth' in checkpoints.
    model_path = os.path.join("checkpoints", "best_model.pth")
    if not os.path.exists(model_path):
        # Fallback to any .pth in checkpoints or None
        pass 
    
    # We assume running from root or src is in path
    engine = InferenceEngine(model_path=model_path)
    print("Inference Engine loaded.")

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": engine is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(file: UploadFile = File(...)):
    if not engine:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    # Save temp file
    # Create temp file
    fd, temp_path = tempfile.mkstemp(suffix=os.path.splitext(file.filename)[1])
    try:
        with os.fdopen(fd, 'wb') as tmp:
            shutil.copyfileobj(file.file, tmp)
            
        # Run inference
        results = engine.predict_file(temp_path)
        
        if results is None:
             raise HTTPException(status_code=400, detail="Could not process audio file")
             
        # Find top genre
        top_genre = max(results, key=results.get)
        
        return {
            "filename": file.filename,
            "predictions": results,
            "top_genre": top_genre
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
