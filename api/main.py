"""
FastAPI application entrypoint for the Water Potability Prediction API.

Endpoints:
    GET  /health   — Model health check and version info
    POST /predict  — Single water sample potability prediction
    GET  /ui       — Serves the web UI (mounted as StaticFiles)

CORS is fully open (allow_origins=["*"]) for development.
Restrict origins in production deployments.
"""

import os

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.middleware import LoggingMiddleware
from api.predictor import WaterPotabilityPredictor
from api.schemas import HealthResponse, PredictionResponse, WaterSample
from src.config import CONFIG
from src.utils.logger import get_logger

logger = get_logger(__name__, "api.log")

# ---------------------------------------------------------------------------
# Application initialisation
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Water Potability Prediction API",
    description=(
        "Production-grade ML API for binary water safety classification. "
        "Uses XGBoost with anti-overfitting constraints, conservative SMOTE, "
        "and balanced threshold optimisation."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — open for development; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/response logging
app.add_middleware(LoggingMiddleware)

# Serve web UI
_ui_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui")
if os.path.isdir(_ui_dir):
    app.mount("/ui", StaticFiles(directory=_ui_dir, html=True), name="ui")

# Load model artifacts — fail fast at startup if artifacts are missing
predictor: WaterPotabilityPredictor = None  # type: ignore


@app.on_event("startup")
async def startup_event() -> None:
    """Load predictor artifacts when the API starts."""
    global predictor
    artifacts_dir = CONFIG.api.artifacts_dir
    logger.info(f"Loading model artifacts from: {artifacts_dir}")
    try:
        predictor = WaterPotabilityPredictor(artifacts_dir=artifacts_dir)
        logger.info("Model loaded successfully — API ready.")
    except FileNotFoundError as e:
        logger.error(f"Startup failed: {e}")
        logger.error("Run 'python main.py --mode train' to generate artifacts first.")
        # Allow API to start without model for /health endpoint
        predictor = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check() -> dict:
    """
    Return API health status and model information.

    Returns 200 even if the model is not loaded, allowing health checks
    to distinguish between "API up, no model" and "API down" states.
    """
    return {
        "status": "healthy" if predictor is not None else "degraded — model not loaded",
        "model_loaded": predictor is not None,
        "model_version": predictor.model_version if predictor else "N/A",
        "threshold": predictor.threshold if predictor else 0.0,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict(sample: WaterSample) -> dict:
    """
    Predict water potability for a single water sample.

    Applies the full preprocessing pipeline (imputation, feature engineering,
    scaling, feature selection) before inference.

    Returns:
        potability (0/1), probability, threshold used, interpretation, confidence.

    Raises:
        503 Service Unavailable: If model artifacts are not loaded.
        500 Internal Server Error: For unexpected inference errors.
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run 'python main.py --mode train' first.",
        )
    try:
        result = predictor.predict(sample.model_dump())
        logger.info(
            f"Prediction: {result['interpretation']} | "
            f"P={result['probability']:.4f} | "
            f"confidence={result['confidence']}"
        )
        return result
    except Exception as exc:
        logger.error(f"Prediction error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Direct run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=CONFIG.api.host,
        port=CONFIG.api.port,
        reload=False,
        log_level="info",
    )
