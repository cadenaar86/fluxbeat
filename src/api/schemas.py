from pydantic import BaseModel
from typing import Dict

class PredictionResponse(BaseModel):
    filename: str
    predictions: Dict[str, float]
    top_genre: str
