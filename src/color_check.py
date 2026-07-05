"""
color_check.py
--------------
LAB color-space analysis and Delta-E anomaly flagging for warp thread strips.

Workflow
--------
1. Convert each strip (BGR) to CIE L*a*b* color space.
2. Compute the mean LAB value per strip (one 3-vector per thread).
3. Compute a reference color as the *median* of all strip means (rolling or
   batch), which is robust to a small number of outlier threads.
4. Calculate CIE76 Delta-E between each strip's mean LAB and the reference.
5. Flag any strip whose Delta-E exceeds ``threshold`` as a color anomaly.

Delta-E (CIE76) is defined as the Euclidean distance in LAB space:

    ΔE = sqrt((ΔL)² + (Δa)² + (Δb)²)

A threshold of ~2.5 is a practical starting point; human observers can
reliably perceive differences above ΔE ≈ 2.3.

Usage
-----
    from src.color_check import check_strips

    results = check_strips(strips, threshold=2.5)
    for r in results:
        print(r["thread_index"], r["delta_e"], r["flagged"])
"""

import numpy as np
import cv2
from dataclasses import dataclass, field


@dataclass
class ThreadColorResult:
    """Per-thread color analysis result.

    Attributes
    ----------
    thread_index : int
        Zero-based position of the thread in the warp sheet (left to right).
    mean_lab : np.ndarray
        Mean CIE L*a*b* value of the strip, shape ``(3,)``.
    delta_e : float
        CIE76 Delta-E distance from this strip to the reference color.
    flagged : bool
        ``True`` if ``delta_e`` exceeds the configured threshold.
    """

    thread_index: int
    mean_lab: np.ndarray = field(repr=False)
    delta_e: float
    flagged: bool


def strip_mean_lab(strip_bgr: np.ndarray) -> np.ndarray:
    """Compute the mean CIE L*a*b* color of a single strip.

    Parameters
    ----------
    strip_bgr : np.ndarray
        A single warp-thread strip in BGR format, shape ``(H, W, 3)``.

    Returns
    -------
    np.ndarray
        Mean LAB value, shape ``(3,)`` — ``[L_mean, a_mean, b_mean]``.

    Raises
    ------
    ValueError
        If the strip has fewer than 3 channels (i.e. is grayscale).
    """
    if strip_bgr.ndim != 3 or strip_bgr.shape[2] != 3:
        raise ValueError(
            "strip_mean_lab expects a 3-channel BGR image. "
            f"Got shape {strip_bgr.shape}."
        )
    lab = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2LAB)
    # OpenCV encodes LAB as uint8: L in [0,255], a/b in [0,255] (biased).
    # Convert to float32 before averaging to avoid uint8 overflow.
    return lab.astype(np.float32).reshape(-1, 3).mean(axis=0)


def delta_e_cie76(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """Compute CIE76 Delta-E between two LAB colour vectors.

    Parameters
    ----------
    lab1, lab2 : np.ndarray
        LAB colour vectors, each of shape ``(3,)``.

    Returns
    -------
    float
        Euclidean distance in LAB space.
    """
    return float(np.linalg.norm(lab1.astype(np.float64) - lab2.astype(np.float64)))


def compute_reference_lab(mean_labs: list[np.ndarray]) -> np.ndarray:
    """Derive a robust reference colour from a collection of strip means.

    Uses the *per-channel median* so that a minority of anomalous threads do
    not skew the reference.

    Parameters
    ----------
    mean_labs : list of np.ndarray
        Per-strip mean LAB values, each of shape ``(3,)``.

    Returns
    -------
    np.ndarray
        Reference LAB colour, shape ``(3,)``.

    Raises
    ------
    ValueError
        If the list is empty.
    """
    if not mean_labs:
        raise ValueError("mean_labs list is empty — cannot compute reference.")
    stacked = np.stack(mean_labs, axis=0)   # (N, 3)
    return np.median(stacked, axis=0)       # (3,)


def check_strips(
    strips: list[np.ndarray],
    *,
    threshold: float = 2.5,
    reference_lab: np.ndarray | None = None,
) -> list[ThreadColorResult]:
    """Analyse a list of warp-thread strips for colour deviation.

    Parameters
    ----------
    strips : list of np.ndarray
        Ordered list of BGR strip images as returned by
        ``segmentation.split_into_strips``.
    threshold : float, optional
        Delta-E value above which a thread is flagged as anomalous.
        Default is ``2.5`` (perceptually just-noticeable difference).
    reference_lab : np.ndarray, optional
        Pre-computed reference LAB colour of shape ``(3,)``.  When ``None``
        (default), the reference is derived from the per-channel median of all
        strips in this batch — suitable for batch analysis.  Provide an
        external reference when performing rolling / online inference against a
        known good colour.

    Returns
    -------
    list of ThreadColorResult
        One result per input strip, in the same order.

    Raises
    ------
    ValueError
        If ``strips`` is empty or ``threshold`` is not positive.

    Examples
    --------
    >>> from src.segmentation import split_into_strips
    >>> import cv2
    >>> img = cv2.imread("data/normal/sample.jpg")
    >>> strips = split_into_strips(img, num_threads=100)
    >>> results = check_strips(strips, threshold=2.5)
    >>> flagged = [r for r in results if r.flagged]
    >>> print(f"{len(flagged)} anomalous threads detected.")
    """
    if not strips:
        raise ValueError("strips list is empty.")
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}.")

    # Step 1: Compute mean LAB for every strip.
    mean_labs: list[np.ndarray] = [strip_mean_lab(s) for s in strips]

    # Step 2: Establish reference colour.
    ref = reference_lab if reference_lab is not None else compute_reference_lab(mean_labs)

    # Step 3: Compute Delta-E and flag anomalies.
    results: list[ThreadColorResult] = []
    for idx, mean_lab in enumerate(mean_labs):
        de = delta_e_cie76(mean_lab, ref)
        results.append(
            ThreadColorResult(
                thread_index=idx,
                mean_lab=mean_lab,
                delta_e=de,
                flagged=de > threshold,
            )
        )

    return results


def summarise_results(results: list[ThreadColorResult]) -> dict:
    """Return a JSON-serialisable summary of colour-check results.

    Parameters
    ----------
    results : list of ThreadColorResult
        Output of :func:`check_strips`.

    Returns
    -------
    dict
        Keys: ``total_threads``, ``flagged_count``, ``flagged_indices``,
        ``max_delta_e``, ``mean_delta_e``.
    """
    flagged = [r for r in results if r.flagged]
    delta_es = [r.delta_e for r in results]
    return {
        "total_threads": len(results),
        "flagged_count": len(flagged),
        "flagged_indices": [r.thread_index for r in flagged],
        "max_delta_e": float(max(delta_es)) if delta_es else 0.0,
        "mean_delta_e": float(np.mean(delta_es)) if delta_es else 0.0,
    }
