"""
api/main.py
-----------
FastAPI application exposing a POST /detect endpoint for warp-thread anomaly
detection.

The endpoint accepts a multipart image upload and returns:
  - Delta-E colour-check results (fast, CPU-only, no model required)
  - PatchCore anomaly score + heatmap (if a trained model checkpoint is found)

Environment variables
---------------------
CHECKPOINT_PATH : str, optional
    Path to the Anomalib PatchCore checkpoint.  When absent or the file is
    missing, PatchCore inference is skipped and only the Delta-E check runs.
NUM_THREADS : int, optional
    Number of warp threads (ends) across the image width (default 120).
DELTA_E_THRESHOLD : float, optional
    Delta-E threshold for colour anomaly flagging (default 2.5).
MODEL_IMAGE_SIZE : int, optional
    Square image size the PatchCore model was trained with (default 256).

Run locally
-----------
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
import os
import base64
import logging
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.segmentation import split_into_strips
from src.color_check import check_strips, summarise_results

logger = logging.getLogger("fabric_detection.api")

# ---------------------------------------------------------------------------
# Configuration (from environment variables)
# ---------------------------------------------------------------------------

CHECKPOINT_PATH: str | None = os.getenv("CHECKPOINT_PATH")
NUM_THREADS: int = int(os.getenv("NUM_THREADS", "120"))
DELTA_E_THRESHOLD: float = float(os.getenv("DELTA_E_THRESHOLD", "2.5"))
MODEL_IMAGE_SIZE: int = int(os.getenv("MODEL_IMAGE_SIZE", "256"))

# ---------------------------------------------------------------------------
# Lazy model loading — loaded once on first request, not at import time.
# ---------------------------------------------------------------------------

_model = None
_model_loaded = False


def _get_model():
    """Return the PatchCore model, loading it on first call.

    Returns ``None`` if no checkpoint is configured or available.
    """
    global _model, _model_loaded
    if _model_loaded:
        return _model

    _model_loaded = True
    if not CHECKPOINT_PATH:
        logger.info("CHECKPOINT_PATH not set — PatchCore inference disabled.")
        return None

    ckpt = Path(CHECKPOINT_PATH)
    if not ckpt.exists():
        logger.warning("Checkpoint not found at %s — PatchCore inference disabled.", ckpt)
        return None

    try:
        from src.inference import load_model
        _model = load_model(ckpt)
        logger.info("PatchCore model loaded from %s", ckpt)
    except ImportError:
        logger.warning(
            "anomalib/torch not installed — PatchCore inference disabled. "
            "Install with: pip install anomalib torch torchvision"
        )
        _model = None

    return _model


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Fabric Warp Thread Anomaly Detection",
    description=(
        "Detects odd-coloured warp threads in textile images using "
        "Delta-E colour analysis and optional PatchCore deep-learning inference."
    ),
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ColorCheckSummary(BaseModel):
    total_threads: int
    flagged_count: int
    flagged_indices: list[int]
    max_delta_e: float = Field(..., description="Highest Delta-E value found")
    mean_delta_e: float


class PatchCoreResult(BaseModel):
    score: float = Field(..., description="Image-level anomaly score")
    is_anomalous: bool
    pred_label: str
    heatmap_png_b64: str = Field(
        ..., description="Base64-encoded PNG of the anomaly heatmap overlay"
    )


class DetectResponse(BaseModel):
    filename: str
    num_threads_analysed: int
    delta_e_threshold: float
    color_check: ColorCheckSummary
    patchcore: PatchCoreResult | None = Field(
        None,
        description="Present only when a trained PatchCore model is available.",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_upload(data: bytes) -> np.ndarray:
    """Decode raw image bytes into a BGR NumPy array."""
    arr = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=422,
            detail="Could not decode image.  Send a valid JPEG or PNG file.",
        )
    return image


def _encode_png_b64(image: np.ndarray) -> str:
    """Encode a BGR/grayscale NumPy array to a base64 PNG string."""
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("cv2.imencode failed.")
    return base64.b64encode(buf.tobytes()).decode("ascii")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", summary="Health check")
async def health() -> dict:
    """Return service status and whether a PatchCore model is loaded."""
    model = _get_model()
    return {
        "status": "ok",
        "patchcore_loaded": model is not None,
        "num_threads": NUM_THREADS,
        "delta_e_threshold": DELTA_E_THRESHOLD,
    }


@app.post(
    "/detect",
    response_model=DetectResponse,
    summary="Detect warp-thread colour anomalies",
)
async def detect(
    file: UploadFile = File(..., description="Warp-sheet image (JPEG or PNG)"),
) -> DetectResponse:
    """Analyse a warp-sheet image for colour-deviant threads.

    Steps performed:
    1. Split the image into ``NUM_THREADS`` vertical strips.
    2. Run Delta-E colour check against the per-batch median colour.
    3. (Optional) Run PatchCore deep-learning inference if a model is loaded.

    Returns a JSON body with colour-check summary and (optionally) the
    PatchCore score and a base64-encoded heatmap overlay PNG.
    """
    # Validate content type loosely (MIME sniffing can be unreliable).
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type: {file.content_type}.  Send image/jpeg or image/png.",
        )

    raw = await file.read()
    if len(raw) == 0:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    image = _decode_upload(raw)

    # ── Step 1: segmentation ─────────────────────────────────────────────────
    try:
        strips = split_into_strips(image, NUM_THREADS)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # ── Step 2: Delta-E colour check ─────────────────────────────────────────
    try:
        color_results = check_strips(strips, threshold=DELTA_E_THRESHOLD)
    except Exception as exc:
        logger.exception("Color check failed.")
        raise HTTPException(status_code=500, detail=f"Color check error: {exc}") from exc

    summary = summarise_results(color_results)

    # ── Step 3: PatchCore inference (optional) ───────────────────────────────
    patchcore_result: PatchCoreResult | None = None
    model = _get_model()
    if model is not None:
        try:
            from src.inference import run_inference
            inf = run_inference(model, image, image_size=MODEL_IMAGE_SIZE)
            overlay = inf.heatmap_overlay(image, alpha=0.5)
            patchcore_result = PatchCoreResult(
                score=inf.score,
                is_anomalous=inf.is_anomalous,
                pred_label=inf.pred_label,
                heatmap_png_b64=_encode_png_b64(overlay),
            )
        except Exception as exc:
            logger.exception("PatchCore inference failed.")
            # Non-fatal: return colour-check results even if deep inference fails.
            logger.warning("PatchCore result omitted due to error: %s", exc)

    return DetectResponse(
        filename=file.filename or "unknown",
        num_threads_analysed=len(strips),
        delta_e_threshold=DELTA_E_THRESHOLD,
        color_check=ColorCheckSummary(**summary),
        patchcore=patchcore_result,
    )
