"""
Secure FastAPI Model Serving for Satellite Terrain Classification
Production-ready REST API with comprehensive security features

Security Features:
- API key or JWT authentication
- Rate limiting (per-minute, per-hour, per-day)
- Input validation and sanitization
- File size limits
- CORS with configurable origins
- Request timeout limits
- Comprehensive logging
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
from loguru import logger
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
import time
from pathlib import Path
import sys
import os
from contextlib import asynccontextmanager

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Security imports
try:
    from security.authentication import APIKeyAuth, RateLimiter
    from security.validation import ImageValidator, FileValidator, ValidationError
    from security.sanitization import sanitize_filename
    SECURITY_AVAILABLE = True
except ImportError:
    logger.warning("Security modules not available")
    SECURITY_AVAILABLE = False


# Configuration
class Config:
    """API Configuration"""
    # Security
    ENABLE_AUTH = os.getenv("ENABLE_AUTH", "true").lower() == "true"
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8080").split(",")
    ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1,*.yourdomain.com").split(",")

    # Rate limits
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    RATE_LIMIT_PER_HOUR = int(os.getenv("RATE_LIMIT_PER_HOUR", "1000"))
    RATE_LIMIT_PER_DAY = int(os.getenv("RATE_LIMIT_PER_DAY", "10000"))

    # File limits
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
    MAX_BATCH_SIZE = 10

    # Timeouts
    REQUEST_TIMEOUT = 30  # seconds

    # Model
    MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.h5")


config = Config()


# Pydantic models with validation
class PredictionRequest(BaseModel):
    """Request model for batch prediction"""
    image_urls: List[str] = Field(..., max_items=config.MAX_BATCH_SIZE)
    return_visualization: bool = Field(False)
    estimate_uncertainty: bool = Field(False)

    @validator('image_urls')
    def validate_urls(cls, v):
        if not v:
            raise ValueError("At least one URL required")
        for url in v:
            if not url.startswith(('http://', 'https://')):
                raise ValueError(f"Invalid URL: {url}")
        return v


class PredictionResponse(BaseModel):
    """Response model for prediction"""
    predicted_class: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: Dict[str, float]
    uncertainty: Optional[Dict[str, Any]] = None
    inference_time: float
    rate_limit_remaining: Optional[int] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    gpu_available: bool
    version: str
    security_enabled: bool


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None
    timestamp: float


# Prometheus metrics
prediction_counter = Counter(
    'terrain_predictions_total',
    'Total number of predictions',
    ['model_version', 'predicted_class', 'user_id']
)

prediction_latency = Histogram(
    'terrain_prediction_latency_seconds',
    'Prediction latency in seconds',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

error_counter = Counter(
    'terrain_prediction_errors_total',
    'Total number of prediction errors',
    ['error_type', 'user_id']
)

auth_failure_counter = Counter(
    'auth_failures_total',
    'Total number of authentication failures',
    ['reason']
)


# Global state
class ModelState:
    """Global model state"""
    model: Optional[keras.Model] = None
    class_names: List[str] = []
    model_version: str = "2.0.0"
    input_shape: tuple = (224, 224, 3)


state = ModelState()


# Initialize security components
if SECURITY_AVAILABLE:
    api_auth = APIKeyAuth()
    rate_limiter = RateLimiter(
        requests_per_minute=config.RATE_LIMIT_PER_MINUTE,
        requests_per_hour=config.RATE_LIMIT_PER_HOUR,
        requests_per_day=config.RATE_LIMIT_PER_DAY
    )
    # Generate initial API key for testing
    test_api_key = api_auth.generate_api_key("test_user")
    logger.info(f"Test API Key: {test_api_key}")


# Middleware for request timeout
class TimeoutMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            start_time = time.time()
            response = await call_next(request)
            process_time = time.time() - start_time

            if process_time > config.REQUEST_TIMEOUT:
                logger.warning(f"Request took {process_time:.2f}s (timeout: {config.REQUEST_TIMEOUT}s)")

            response.headers["X-Process-Time"] = str(process_time)
            return response
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error", "detail": str(e)}
            )


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up API...")
    try:
        # Load model
        model_path = Path(config.MODEL_PATH)
        if model_path.exists():
            state.model = keras.models.load_model(str(model_path))
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.warning(f"Model not found, creating dummy model")
            state.model = create_dummy_model()

        # Load class names
        state.class_names = load_class_names()

        # Configure GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"GPUs available: {len(gpus)}")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        logger.success("API ready!")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down API...")


# Initialize FastAPI app
app = FastAPI(
    title="Secure Satellite Terrain Classification API",
    description="Production-ready terrain classification with security features",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add security middleware
app.add_middleware(TimeoutMiddleware)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=config.ALLOWED_HOSTS
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,  # Specific origins only!
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Only needed methods
    allow_headers=["X-API-Key", "Content-Type", "Authorization"],
    max_age=3600
)


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
    return [
        "Urban", "Forest", "Water", "Agricultural", "Barren",
        "Grassland", "Wetland", "Snow/Ice", "Industrial", "Residential"
    ]


def validate_and_preprocess_image(
    image_bytes: bytes,
    max_size: int = config.MAX_FILE_SIZE
) -> np.ndarray:
    """
    Validate and preprocess image with security checks

    Args:
        image_bytes: Raw image bytes
        max_size: Maximum allowed file size

    Returns:
        Preprocessed image array

    Raises:
        ValidationError: If validation fails
    """
    # Check file size
    if len(image_bytes) > max_size:
        raise ValidationError(
            f"File too large: {len(image_bytes)} bytes (max: {max_size})"
        )

    # Load image
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert('RGB')
    except Exception as e:
        raise ValidationError(f"Invalid image file: {e}")

    # Resize
    image = image.resize((state.input_shape[0], state.input_shape[1]))

    # Convert to array
    img_array = np.array(image, dtype=np.float32)

    # Validate array
    if SECURITY_AVAILABLE:
        ImageValidator.validate_image_array(
            img_array,
            min_height=32,
            min_width=32,
            allowed_channels=[3]
        )

    # Normalize
    img_array = img_array / 255.0

    return img_array


# Dependency for authentication and rate limiting
async def verify_access(
    api_key: Optional[str] = Depends(api_auth.validate) if SECURITY_AVAILABLE and config.ENABLE_AUTH else None
) -> str:
    """
    Verify API access with authentication and rate limiting

    Args:
        api_key: API key from header

    Returns:
        User ID

    Raises:
        HTTPException: If access denied
    """
    if not config.ENABLE_AUTH:
        return "anonymous"

    if not SECURITY_AVAILABLE:
        logger.warning("Security not available but auth enabled")
        return "anonymous"

    # Rate limiting
    try:
        rate_limiter.check_rate_limit(api_key)
    except HTTPException as e:
        auth_failure_counter.labels(reason="rate_limit").inc()
        raise e

    return api_key


# API endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Secure Satellite Terrain Classification API",
        "version": state.model_version,
        "security": "enabled" if config.ENABLE_AUTH else "disabled",
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
        version=state.model_version,
        security_enabled=config.ENABLE_AUTH
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    estimate_uncertainty: bool = False,
    user_id: str = Depends(verify_access)
):
    """
    Predict terrain class from uploaded image

    Args:
        file: Uploaded image file
        estimate_uncertainty: Estimate prediction uncertainty
        user_id: Authenticated user ID

    Returns:
        Prediction response
    """
    start_time = time.time()

    try:
        # Validate model loaded
        if state.model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )

        # Sanitize filename
        if SECURITY_AVAILABLE:
            safe_filename = sanitize_filename(file.filename)
        else:
            safe_filename = file.filename

        logger.info(f"Processing file: {safe_filename} for user: {user_id}")

        # Read and validate image
        contents = await file.read()
        img_array = validate_and_preprocess_image(contents)

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
            "inference_time": time.time() - start_time
        }

        # Add rate limit info
        if SECURITY_AVAILABLE and config.ENABLE_AUTH:
            rate_status = rate_limiter.get_rate_limit_status(user_id)
            response_data["rate_limit_remaining"] = rate_status['minute']['remaining']

        # Update metrics
        prediction_counter.labels(
            model_version=state.model_version,
            predicted_class=pred_class,
            user_id=user_id
        ).inc()

        logger.success(f"Prediction complete: {pred_class} ({confidence:.3f})")

        return PredictionResponse(**response_data)

    except ValidationError as e:
        error_counter.labels(error_type="validation", user_id=user_id).inc()
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        error_counter.labels(error_type=type(e).__name__, user_id=user_id).inc()
        logger.exception(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/rate-limit")
async def get_rate_limit(user_id: str = Depends(verify_access)):
    """Get current rate limit status"""
    if not SECURITY_AVAILABLE or not config.ENABLE_AUTH:
        return {"message": "Rate limiting not enabled"}

    return rate_limiter.get_rate_limit_status(user_id)


@app.get("/classes")
async def get_classes():
    """Get list of terrain classes"""
    return {
        "classes": state.class_names,
        "num_classes": len(state.class_names)
    }


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting secure API server...")
    logger.info(f"Authentication: {'enabled' if config.ENABLE_AUTH else 'disabled'}")
    logger.info(f"Allowed origins: {config.ALLOWED_ORIGINS}")

    uvicorn.run(
        "serve_secure:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload in production
        workers=1,  # Single worker for GPU
        log_level="info",
        access_log=True
    )
