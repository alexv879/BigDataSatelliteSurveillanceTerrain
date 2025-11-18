"""
FastAPI Model Serving for Satellite Terrain Classification
Production-ready REST API with monitoring and optimization
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from PIL import Image
import io
from loguru import logger
from prometheus_client import Counter, Histogram, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST
from starlette.responses import Response
import time
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from interpretability.gradcam import GradCAM, UncertaintyEstimator, explain_prediction


# Pydantic models
class PredictionRequest(BaseModel):
    """Request model for batch prediction"""
    image_urls: List[str] = Field(..., description="List of image URLs")
    return_visualization: bool = Field(False, description="Return GradCAM visualization")
    estimate_uncertainty: bool = Field(False, description="Estimate prediction uncertainty")


class PredictionResponse(BaseModel):
    """Response model for prediction"""
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    uncertainty: Optional[Dict[str, Any]] = None
    gradcam_available: bool = False
    inference_time: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    gpu_available: bool
    version: str


# Prometheus metrics
prediction_counter = Counter(
    'terrain_predictions_total',
    'Total number of predictions',
    ['model_version', 'predicted_class']
)

prediction_latency = Histogram(
    'terrain_prediction_latency_seconds',
    'Prediction latency in seconds',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

error_counter = Counter(
    'terrain_prediction_errors_total',
    'Total number of prediction errors',
    ['error_type']
)


# Initialize FastAPI app
app = FastAPI(
    title="Satellite Terrain Classification API",
    description="State-of-the-art terrain classification using deep learning",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global state
class ModelState:
    """Global model state"""
    model: Optional[keras.Model] = None
    class_names: List[str] = []
    model_version: str = "2.0.0"
    input_shape: tuple = (224, 224, 3)
    gradcam: Optional[GradCAM] = None
    uncertainty_estimator: Optional[UncertaintyEstimator] = None


state = ModelState()


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        logger.info("Loading terrain classification model...")

        # Load model (replace with your actual model path)
        model_path = "models/best_model.h5"
        if Path(model_path).exists():
            state.model = keras.models.load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.warning(f"Model not found at {model_path}, using dummy model")
            # Create dummy model for testing
            state.model = create_dummy_model()

        # Load class names
        state.class_names = load_class_names()

        # Initialize interpretability tools
        state.gradcam = GradCAM(state.model)
        state.uncertainty_estimator = UncertaintyEstimator(state.model)

        # Check GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"GPUs available: {len(gpus)}")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            logger.warning("No GPU available, using CPU")

        logger.info("API ready to serve predictions!")

    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down API...")


# Utility functions
def create_dummy_model() -> keras.Model:
    """Create a dummy model for testing"""
    inputs = keras.Input(shape=(224, 224, 3))
    x = keras.layers.Conv2D(32, 3, activation='relu')(inputs)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(10, activation='softmax')(x)
    return keras.Model(inputs, outputs)


def load_class_names() -> List[str]:
    """Load class names"""
    # Replace with actual class names from your dataset
    return [
        "Urban",
        "Forest",
        "Water",
        "Agricultural",
        "Barren",
        "Grassland",
        "Wetland",
        "Snow/Ice",
        "Industrial",
        "Residential"
    ]


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess image for model inference

    Args:
        image: PIL Image

    Returns:
        Preprocessed numpy array
    """
    # Resize
    image = image.resize((state.input_shape[0], state.input_shape[1]))

    # Convert to array
    img_array = np.array(image)

    # Ensure 3 channels
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    # Normalize
    img_array = img_array.astype(np.float32) / 255.0

    return img_array


# API endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Satellite Terrain Classification API",
        "version": state.model_version,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    gpus = tf.config.list_physical_devices('GPU')

    return HealthResponse(
        status="healthy" if state.model is not None else "unhealthy",
        model_loaded=state.model is not None,
        gpu_available=len(gpus) > 0,
        version=state.model_version
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    return_visualization: bool = False,
    estimate_uncertainty: bool = False,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Predict terrain class from uploaded image

    Args:
        file: Uploaded image file
        return_visualization: Return GradCAM visualization
        estimate_uncertainty: Estimate prediction uncertainty

    Returns:
        Prediction response with class, confidence, and optional visualizations
    """
    start_time = time.time()

    try:
        # Validate model loaded
        if state.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        # Preprocess
        img_array = preprocess_image(image)

        # Predict
        with prediction_latency.time():
            predictions = state.model.predict(img_array[np.newaxis, ...])[0]

        # Get top prediction
        pred_class_idx = int(np.argmax(predictions))
        pred_class = state.class_names[pred_class_idx]
        confidence = float(predictions[pred_class_idx])

        # Create probabilities dict
        probabilities = {
            state.class_names[i]: float(predictions[i])
            for i in range(len(predictions))
        }

        # Build response
        response_data = {
            "predicted_class": pred_class,
            "confidence": confidence,
            "probabilities": probabilities,
            "inference_time": time.time() - start_time,
            "gradcam_available": return_visualization
        }

        # Add uncertainty estimation
        if estimate_uncertainty and state.uncertainty_estimator:
            mean_pred, std_pred, entropy = state.uncertainty_estimator.predict_with_uncertainty(img_array)
            response_data["uncertainty"] = {
                "entropy": float(entropy),
                "std_dev": std_pred.tolist()
            }

        # Update metrics
        prediction_counter.labels(
            model_version=state.model_version,
            predicted_class=pred_class
        ).inc()

        # Log prediction
        background_tasks.add_task(
            log_prediction,
            pred_class,
            confidence,
            time.time() - start_time
        )

        return PredictionResponse(**response_data)

    except Exception as e:
        error_counter.labels(error_type=type(e).__name__).inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    estimate_uncertainty: bool = False
):
    """
    Batch prediction for multiple images

    Args:
        files: List of uploaded image files
        estimate_uncertainty: Estimate prediction uncertainty

    Returns:
        List of predictions
    """
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []

    for file in files:
        try:
            # Read and preprocess
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            img_array = preprocess_image(image)

            # Predict
            predictions = state.model.predict(img_array[np.newaxis, ...])[0]
            pred_class_idx = int(np.argmax(predictions))
            pred_class = state.class_names[pred_class_idx]
            confidence = float(predictions[pred_class_idx])

            result = {
                "filename": file.filename,
                "predicted_class": pred_class,
                "confidence": confidence
            }

            results.append(result)

        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            results.append({
                "filename": file.filename,
                "error": str(e)
            })

    return {"predictions": results}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/classes")
async def get_classes():
    """Get list of terrain classes"""
    return {
        "classes": state.class_names,
        "num_classes": len(state.class_names)
    }


def log_prediction(pred_class: str, confidence: float, inference_time: float):
    """Log prediction for monitoring"""
    logger.info(
        f"Prediction: {pred_class} "
        f"(confidence: {confidence:.3f}, "
        f"time: {inference_time:.3f}s)"
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "serve:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=4,
        log_level="info"
    )
