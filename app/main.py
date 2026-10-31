
from pathlib import Path
from typing import List

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR / "models" /
    "crop_recommendation_rf.pkl"
)

METADATA_PATH = (
    BASE_DIR / "models" /
    "crop_recommendation_metadata.json"
)

MODEL_VERSION = "1.0.0"


# --------------------------------------------------
# 2. Load trained model and metadata
# --------------------------------------------------

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

if not METADATA_PATH.is_file():
    raise FileNotFoundError(
        f"Metadata file not found: {METADATA_PATH}"
    )

import json

MODEL = joblib.load(MODEL_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)

FEATURES = metadata.get(
    "features",
    ["N", "P", "K", "temperature",
     "humidity", "ph", "rainfall"]
)


# --------------------------------------------------
# 3. Initialize FastAPI
# --------------------------------------------------

app = FastAPI(
    title="AI-Based Crop Recommendation API",
    description=(
        "API for predicting suitable crops "
        "using soil and environmental conditions."
    ),
    version=MODEL_VERSION
)


# --------------------------------------------------
# 4. Define prediction input
# --------------------------------------------------

class PredictionInput(BaseModel):

    model_config = ConfigDict(allow_inf_nan=False)

    N: float = Field(
        ..., ge=0,
        description="Nitrogen content"
    )

    P: float = Field(
        ..., ge=0,
        description="Phosphorus content"
    )

    K: float = Field(
        ..., ge=0,
        description="Potassium content"
    )

    temperature: float = Field(
        ...,
        description="Temperature in Celsius"
    )

    humidity: float = Field(
        ..., ge=0, le=100,
        description="Relative humidity percentage"
    )

    ph: float = Field(
        ..., ge=0, le=14,
        description="Soil pH value"
    )

    rainfall: float = Field(
        ..., ge=0,
        description="Rainfall in millimetres"
    )


# --------------------------------------------------
# 5. Define prediction output
# --------------------------------------------------

class CropProbability(BaseModel):

    crop: str
    probability: float


class PredictionOutput(BaseModel):

    predicted_crop: str
    confidence: float
    top_predictions: List[CropProbability]
    model_version: str


# --------------------------------------------------
# 6. Health check endpoint
# --------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None
    }


# --------------------------------------------------
# 7. Crop prediction endpoint
# --------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionOutput
)
def predict(item: PredictionInput):

    try:

        # Convert input into DataFrame
        input_data = pd.DataFrame(
            [[
                getattr(item, feature)
                for feature in FEATURES
            ]],
            columns=FEATURES
        )

        # Generate crop prediction
        prediction = str(
            MODEL.predict(input_data)[0]
        )

        # Calculate prediction probabilities
        probabilities = MODEL.predict_proba(
            input_data
        )[0]

        # Retrieve crop class names
        classes = [
            str(crop) for crop in MODEL.classes_
        ]

        if len(classes) != len(probabilities):
            raise ValueError(
                "Model classes and probabilities do not match."
            )

        # Organize probabilities
        results = sorted(
            zip(classes, probabilities),
            key=lambda x: x[1],
            reverse=True
        )

        # Select top three recommendations
        top_three = results[:3]

        # Maximum prediction probability
        confidence = float(max(probabilities))

        return PredictionOutput(
            predicted_crop=prediction,
            confidence=round(confidence, 4),

            top_predictions=[
                CropProbability(
                    crop=crop,
                    probability=round(float(prob), 4)
                )
                for crop, prob in top_three
            ],

            model_version=MODEL_VERSION
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail="Crop prediction failed."
        ) from exc
