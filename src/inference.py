"""
inference.py
------------
CPU-compatible inference using a trained Anomalib PatchCore model.

This module can be used in two ways:

  1. As a library (imported by api/main.py):
        from src.inference import load_model, run_inference

  2. As a standalone CLI script:
        python src/inference.py \\
            --model_path models/Patchcore/warp_sheet/weights/lightning/model.ckpt \\
            --image_path data/test/defect/sample.jpg \\
            --output_path output/heatmap.png

Why CPU inference works for PatchCore
--------------------------------------
PatchCore stores all patch embeddings in a memory bank during training.
Inference is a nearest-neighbour search in that bank — no GPU is needed.
Typical latency on 2-core CPU: 1-3 seconds per image at 256×256.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class InferenceResult:
    """Output of a single PatchCore inference pass.

    Attributes
    ----------
    score : float
        Image-level anomaly score.  Higher → more anomalous.
    heatmap : np.ndarray
        Pixel-level anomaly map, shape ``(H, W)``, dtype float32, values in
        ``[0, 1]`` after min-max normalisation.
    is_anomalous : bool
        ``True`` when ``score`` exceeds the model's calibrated threshold.
    """

    score: float
    heatmap: np.ndarray
    is_anomalous: bool

    @property
    def pred_label(self) -> str:
        """Human-readable prediction label."""
        return "anomalous" if self.is_anomalous else "normal"

    def heatmap_overlay(self, bgr_image: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Blend the JET-colourised anomaly heatmap onto the original image.

        Parameters
        ----------
        bgr_image : np.ndarray
            Original warp-sheet image in BGR format, shape ``(H, W, 3)``.
        alpha : float
            Heatmap opacity (0 = invisible, 1 = fully opaque).

        Returns
        -------
        np.ndarray
            BGR image with heatmap overlay.
        """
        h, w = bgr_image.shape[:2]
        heat_resized = cv2.resize(self.heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        heat_uint8 = (heat_resized * 255).clip(0, 255).astype(np.uint8)
        heat_colour = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
        return cv2.addWeighted(bgr_image, 1 - alpha, heat_colour, alpha, 0)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str | Path):
    """Load a trained PatchCore model from an Anomalib Lightning checkpoint.

    Anomalib and PyTorch are imported lazily here so that every other module
    in this repo (segmentation, color_check, FastAPI app) can be imported on
    a CPU-only machine without anomalib/torch installed.

    Parameters
    ----------
    checkpoint_path : str or Path
        Path to the ``.ckpt`` file produced by ``train_patchcore.py``.
        Typically found at:
        ``models/Patchcore/warp_sheet/weights/lightning/model.ckpt``

    Returns
    -------
    anomalib.models.Patchcore
        Model loaded onto CPU in evaluation mode.

    Raises
    ------
    ImportError
        If ``anomalib`` or ``torch`` is not installed.
    FileNotFoundError
        If the checkpoint file does not exist.
    """
    try:
        from anomalib.models import Patchcore
    except ImportError as exc:
        raise ImportError(
            f"Cannot import anomalib: {exc}\n"
            "Install on Colab with:  pip install anomalib torch torchvision\n"
            "CPU-only codespace:     pip install anomalib torch --index-url "
            "https://download.pytorch.org/whl/cpu"
        ) from exc

    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt.resolve()}\n"
            "Train the model first with src/train_patchcore.py on Colab."
        )

    # map_location="cpu" ensures the checkpoint loads on CPU even if it was
    # saved on a GPU machine (Colab T4).
    model = Patchcore.load_from_checkpoint(str(ckpt), map_location="cpu")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Core inference function
# ---------------------------------------------------------------------------

def run_inference(
    model,
    image: np.ndarray,
    *,
    image_size: int = 256,
) -> InferenceResult:
    """Run PatchCore inference on a single BGR image.

    Parameters
    ----------
    model :
        Loaded PatchCore model (from :func:`load_model`).
    image : np.ndarray
        BGR image as returned by ``cv2.imread``, shape ``(H, W, 3)``.
    image_size : int, optional
        The square size the model was trained at (default 256).  The image is
        resized to ``(image_size, image_size)`` before the forward pass, then
        the resulting heatmap is upsampled back to the original ``(H, W)``.

    Returns
    -------
    InferenceResult
        Anomaly score, normalised heatmap, and is_anomalous flag.

    Notes
    -----
    * Inference is forced to CPU via the pre-processing pipeline.
    * ImageNet mean/std normalisation is applied to match the backbone's
      expected input distribution (same stats used during training).
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            f"torch not found: {exc}\n"
            "Install with:  pip install torch --index-url https://download.pytorch.org/whl/cpu"
        ) from exc

    original_h, original_w = image.shape[:2]

    # ── Pre-process: BGR → RGB → resize → normalise → batch tensor ───────────
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)

    # HWC uint8 → CHW float32 in [0, 1]
    tensor = torch.from_numpy(rgb_resized).permute(2, 0, 1).float() / 255.0

    # ImageNet normalisation — must match training pre-processing
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std

    batch = tensor.unsqueeze(0)   # (1, 3, H, W)

    # ── Forward pass (no gradients needed) ───────────────────────────────────
    with torch.no_grad():
        predictions = model(batch)

    # Anomalib's Patchcore forward() returns a dict:
    #   "pred_score"   – scalar image-level anomaly score
    #   "anomaly_map"  – (1, 1, H_model, W_model) pixel-level score tensor
    score: float = float(predictions["pred_score"].squeeze().item())
    raw_map: np.ndarray = predictions["anomaly_map"].squeeze().cpu().numpy()

    # ── Post-process heatmap ──────────────────────────────────────────────────
    # Min-max normalise to [0, 1] for display
    v_min, v_max = raw_map.min(), raw_map.max()
    norm_map = (raw_map - v_min) / (v_max - v_min) if v_max > v_min else np.zeros_like(raw_map)

    # Upsample from model resolution back to original image resolution
    heatmap = cv2.resize(
        norm_map.astype(np.float32),
        (original_w, original_h),
        interpolation=cv2.INTER_LINEAR,
    )

    # ── Threshold ─────────────────────────────────────────────────────────────
    # Anomalib calibrates an image-level threshold during engine.test().
    # It is stored in model.image_threshold.value after training + evaluation.
    try:
        threshold = float(model.image_threshold.value.item())
        is_anomalous = score >= threshold
    except AttributeError:
        # Threshold not calibrated (e.g. test split was empty).  Fall back to
        # a naive rule: flag if score is above 0.5 of the normalised range.
        is_anomalous = score > 0.5

    return InferenceResult(score=score, heatmap=heatmap, is_anomalous=is_anomalous)


# ---------------------------------------------------------------------------
# Convenience: path-based inference (used by the API)
# ---------------------------------------------------------------------------

def infer_from_path(
    model,
    image_path: str | Path,
    *,
    image_size: int = 256,
) -> InferenceResult:
    """Load an image from disk and run inference.

    Parameters
    ----------
    model :
        Loaded PatchCore model.
    image_path : str or Path
        Path to the warp-sheet image.
    image_size : int, optional
        Model input size (default 256).

    Returns
    -------
    InferenceResult
    """
    path = Path(image_path)
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path.resolve()}")
    return run_inference(model, image, image_size=image_size)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for standalone CLI use."""
    p = argparse.ArgumentParser(
        prog="inference",
        description=(
            "Run PatchCore anomaly detection on a single warp-sheet image. "
            "Requires a checkpoint produced by train_patchcore.py."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help=(
            "Path to the Anomalib PatchCore checkpoint (.ckpt). "
            "Typical location: models/Patchcore/warp_sheet/weights/lightning/model.ckpt"
        ),
    )
    p.add_argument(
        "--image_path",
        type=Path,
        required=True,
        help="Path to the input warp-sheet image (JPEG or PNG).",
    )
    p.add_argument(
        "--output_path",
        type=Path,
        default=Path("output/heatmap.png"),
        help=(
            "Where to save the heatmap overlay visualisation (PNG). "
            "Parent directory is created automatically."
        ),
    )
    p.add_argument(
        "--image_size",
        type=int,
        default=256,
        metavar="PX",
        help="Square image size the model was trained with.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        metavar="A",
        help="Heatmap overlay opacity (0.0 = invisible, 1.0 = fully opaque).",
    )
    return p


def main() -> None:
    """CLI entry-point: load model, infer, save heatmap overlay."""
    parser = build_parser()
    args = parser.parse_args()

    # ── Load image ────────────────────────────────────────────────────────────
    image = cv2.imread(str(args.image_path))
    if image is None:
        sys.exit(f"[inference] Could not read image: {args.image_path.resolve()}")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"[inference] Loading model from: {args.model_path}")
    try:
        model = load_model(args.model_path)
    except (ImportError, FileNotFoundError) as exc:
        sys.exit(f"[inference] {exc}")

    # ── Run inference ─────────────────────────────────────────────────────────
    print(f"[inference] Running inference on: {args.image_path}")
    result = run_inference(model, image, image_size=args.image_size)

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n  Anomaly score : {result.score:.4f}")
    print(f"  Prediction    : {result.pred_label.upper()}")
    print(f"  Is anomalous  : {result.is_anomalous}")

    # ── Save heatmap overlay ──────────────────────────────────────────────────
    overlay = result.heatmap_overlay(image, alpha=args.alpha)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output_path), overlay)
    print(f"\n[inference] Heatmap overlay saved to: {args.output_path.resolve()}")


if __name__ == "__main__":
    # Example:
    #   python src/inference.py \
    #       --model_path models/Patchcore/warp_sheet/weights/lightning/model.ckpt \
    #       --image_path data/test/defect/sample.jpg \
    #       --output_path output/heatmap.png
    main()
