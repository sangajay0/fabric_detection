"""
segmentation.py
---------------
Splits a warp-sheet image into per-thread vertical strips.

A warp sheet is assumed to have threads running vertically (top to bottom).
The image is divided into `num_threads` equal-width vertical columns, each
representing one warp thread (end).

Usage
-----
    from src.segmentation import split_into_strips

    strips = split_into_strips(image, num_threads=120)
"""

import numpy as np
import cv2
from typing import Optional


def split_into_strips(
    image: np.ndarray,
    num_threads: int,
    *,
    overlap_px: int = 0,
    min_strip_width: int = 2,
) -> list[np.ndarray]:
    """Split a warp-sheet image into per-thread vertical strips.

    Parameters
    ----------
    image : np.ndarray
        BGR or grayscale image loaded with ``cv2.imread``.  Shape must be
        ``(H, W)`` or ``(H, W, C)``.
    num_threads : int
        Number of warp threads (ends) expected across the width of the image.
    overlap_px : int, optional
        Number of pixels to overlap between adjacent strips (default 0).
        Useful when threads are not perfectly separated.
    min_strip_width : int, optional
        Minimum acceptable strip width in pixels.  Raises ``ValueError`` if
        the computed width is below this threshold (default 2).

    Returns
    -------
    list of np.ndarray
        Ordered list of strip images (left to right), each of shape
        ``(H, strip_width + overlap_px, C)`` or ``(H, strip_width + overlap_px)``
        for grayscale inputs.

    Raises
    ------
    ValueError
        If ``num_threads`` is less than 1 or the resulting strip width is
        smaller than ``min_strip_width``.
    TypeError
        If ``image`` is not a NumPy ndarray.

    Examples
    --------
    >>> import cv2
    >>> img = cv2.imread("data/normal/sample.jpg")
    >>> strips = split_into_strips(img, num_threads=100)
    >>> print(len(strips))      # 100
    >>> print(strips[0].shape)  # (H, strip_width, 3)
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(image).__name__}.")
    if num_threads < 1:
        raise ValueError(f"num_threads must be >= 1, got {num_threads}.")

    height, width = image.shape[:2]
    base_width = width // num_threads

    if base_width < min_strip_width:
        raise ValueError(
            f"Strip width ({base_width} px) is below min_strip_width "
            f"({min_strip_width} px). Reduce num_threads or use a wider image."
        )

    strips: list[np.ndarray] = []

    for i in range(num_threads):
        x_start = i * base_width
        # Last strip absorbs any remaining pixels from integer division.
        x_end = width if i == num_threads - 1 else (i + 1) * base_width

        # Extend right edge by overlap_px (clipped to image boundary).
        x_end_with_overlap = min(x_end + overlap_px, width)

        strip = image[:, x_start:x_end_with_overlap]
        strips.append(strip)

    return strips


def strips_from_path(
    image_path: str,
    num_threads: int,
    *,
    overlap_px: int = 0,
) -> list[np.ndarray]:
    """Convenience wrapper: load image from disk and return strips.

    Parameters
    ----------
    image_path : str
        Path to the warp-sheet image file.
    num_threads : int
        Number of warp threads expected across the image width.
    overlap_px : int, optional
        Pixel overlap between adjacent strips (default 0).

    Returns
    -------
    list of np.ndarray
        Per-thread vertical strip images.

    Raises
    ------
    FileNotFoundError
        If the image file cannot be read by OpenCV.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: '{image_path}'")
    return split_into_strips(image, num_threads, overlap_px=overlap_px)
