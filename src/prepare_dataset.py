"""
prepare_dataset.py
------------------
Synthetic dataset generator for the fabric warp-thread anomaly detection
pipeline.

Use this script when you have a single uniform warp-sheet image (e.g. a blue
sheet with no defects) and want to bootstrap a dataset before real factory
data is available.

Three stages
~~~~~~~~~~~~
1. **Tile** — Crop the input image into overlapping 256×256 tiles and save
   ~25–30 of them to ``data/normal/``.  These are the *training* images for
   PatchCore (good warp sheets only).

2. **Defect** — Take 5 of those generated tiles, paste a small rectangular
   patch (30×30 px by default) sampled from a *second* image (different
   colour) at a random position, and save to ``data/test/defect/``.  This
   simulates an odd-coloured yarn thread mixed into the warp sheet.

3. **Good** — Take 5 more clean (unmodified) tiles and copy them to
   ``data/test/good/``.

Command-line usage
~~~~~~~~~~~~~~~~~~
    python src/prepare_dataset.py \\
        --input_image   path/to/blue_warp.jpg \\
        --second_color_image path/to/red_fabric.jpg \\
        --output_root   data/

Optional flags
~~~~~~~~~~~~~~
    --tile_size     Tile side length in pixels        (default: 256)
    --overlap       Overlap between adjacent tiles, 0–1 fraction (default: 0.5)
    --n_normal      Number of tiles to save as normal (default: 28)
    --n_defect      Number of tiles to turn into defect images (default: 5)
    --n_good        Number of clean tiles for test/good/ (default: 5)
    --patch_size    Side length of the injected colour patch (default: 30)
    --seed          Random seed for reproducibility (default: 42)
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_image(path: str | Path) -> np.ndarray:
    """Load a BGR image from *path*; raise ``FileNotFoundError`` on failure."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: '{path}'")
    return img


def _ensure_dirs(*dirs: Path) -> None:
    """Create all supplied directories (and parents) if they do not exist."""
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Stage 1 — Tiling
# ---------------------------------------------------------------------------

def generate_tiles(
    image: np.ndarray,
    *,
    tile_size: int = 256,
    overlap: float = 0.5,
) -> list[np.ndarray]:
    """Crop *image* into a grid of overlapping square tiles.

    Parameters
    ----------
    image : np.ndarray
        Source BGR image loaded with ``cv2.imread``.
    tile_size : int, optional
        Side length of each square tile in pixels (default 256).
    overlap : float, optional
        Fractional overlap between adjacent tiles in the range ``[0, 1)``.
        ``0.5`` means tiles share 50 % of their area with the next tile.
        Default is ``0.5``.

    Returns
    -------
    list of np.ndarray
        All fully-contained tiles in row-major (top-left → bottom-right)
        order.  Tiles that would extend beyond the image boundary are
        silently skipped.

    Raises
    ------
    ValueError
        If *tile_size* is larger than either image dimension, or if
        *overlap* is outside ``[0, 1)``.
    """
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}.")

    h, w = image.shape[:2]
    if tile_size > h or tile_size > w:
        raise ValueError(
            f"tile_size ({tile_size}) exceeds image dimensions ({w}×{h})."
        )

    step = max(1, int(tile_size * (1.0 - overlap)))

    tiles: list[np.ndarray] = []
    for y in range(0, h - tile_size + 1, step):
        for x in range(0, w - tile_size + 1, step):
            tile = image[y : y + tile_size, x : x + tile_size].copy()
            tiles.append(tile)

    return tiles


def save_normal_tiles(
    image: np.ndarray,
    output_dir: Path,
    *,
    tile_size: int = 256,
    overlap: float = 0.5,
    n_normal: int = 28,
    seed: int = 42,
) -> list[np.ndarray]:
    """Tile *image*, randomly select *n_normal* tiles, and save to *output_dir*.

    Parameters
    ----------
    image : np.ndarray
        Source BGR warp-sheet image.
    output_dir : Path
        Destination directory (created if absent).
    tile_size : int, optional
        Tile side length in pixels (default 256).
    overlap : float, optional
        Fractional overlap between adjacent tiles (default 0.5).
    n_normal : int, optional
        Maximum number of tiles to save (default 28).  If fewer tiles are
        generated the function saves all of them.
    seed : int, optional
        Random seed used when sub-sampling tiles (default 42).

    Returns
    -------
    list of np.ndarray
        The selected tile images (same objects that were saved to disk).

    Raises
    ------
    RuntimeError
        If no tiles can be generated from *image* with the given parameters.
    """
    _ensure_dirs(output_dir)

    all_tiles = generate_tiles(image, tile_size=tile_size, overlap=overlap)
    if not all_tiles:
        raise RuntimeError(
            "No tiles were generated.  Try a larger image or smaller tile_size."
        )

    rng = random.Random(seed)
    selected = rng.sample(all_tiles, min(n_normal, len(all_tiles)))

    for i, tile in enumerate(selected):
        out_path = output_dir / f"normal_{i:04d}.png"
        cv2.imwrite(str(out_path), tile)

    print(
        f"[prepare] Saved {len(selected)} normal tiles → {output_dir} "
        f"(pool size: {len(all_tiles)})"
    )
    return selected


# ---------------------------------------------------------------------------
# Stage 2 — Synthetic defect injection
# ---------------------------------------------------------------------------

def inject_defect_patch(
    tile: np.ndarray,
    donor_image: np.ndarray,
    *,
    patch_size: int = 30,
    rng: random.Random | None = None,
) -> np.ndarray:
    """Paste a small rectangular region from *donor_image* onto *tile*.

    The donor patch is sampled at a random location within *donor_image* and
    placed at a random position within *tile* that keeps the patch fully
    inside the tile boundary.

    Parameters
    ----------
    tile : np.ndarray
        Clean 256×256 (or any size) BGR warp tile.
    donor_image : np.ndarray
        A second BGR image with a different colour (e.g. a red fabric swatch).
        Must be at least ``patch_size × patch_size`` pixels.
    patch_size : int, optional
        Side length of the square colour patch to inject (default 30).
    rng : random.Random, optional
        Pre-seeded RNG instance for reproducibility.  A fresh instance with
        a random seed is created when ``None`` (default).

    Returns
    -------
    np.ndarray
        A *copy* of *tile* with the colour patch pasted in.

    Raises
    ------
    ValueError
        If *donor_image* is smaller than *patch_size* in either dimension,
        or if *tile* is smaller than *patch_size*.
    """
    if rng is None:
        rng = random.Random()

    dh, dw = donor_image.shape[:2]
    th, tw = tile.shape[:2]

    if dh < patch_size or dw < patch_size:
        raise ValueError(
            f"donor_image ({dw}×{dh}) is smaller than patch_size ({patch_size})."
        )
    if th < patch_size or tw < patch_size:
        raise ValueError(
            f"tile ({tw}×{th}) is smaller than patch_size ({patch_size})."
        )

    # Sample a random donor patch.
    dy = rng.randint(0, dh - patch_size)
    dx = rng.randint(0, dw - patch_size)
    patch = donor_image[dy : dy + patch_size, dx : dx + patch_size].copy()

    # Paste onto a copy of the tile at a random position.
    result = tile.copy()
    ty = rng.randint(0, th - patch_size)
    tx = rng.randint(0, tw - patch_size)
    result[ty : ty + patch_size, tx : tx + patch_size] = patch

    return result


def save_defect_tiles(
    tiles: list[np.ndarray],
    donor_image: np.ndarray,
    output_dir: Path,
    *,
    n_defect: int = 5,
    patch_size: int = 30,
    seed: int = 42,
) -> None:
    """Inject synthetic colour-patch defects and save to *output_dir*.

    Selects the *first* ``n_defect`` tiles from *tiles* (these should be
    tiles **not** already used for ``data/test/good/``), applies
    :func:`inject_defect_patch` to each, and saves the results.

    Parameters
    ----------
    tiles : list of np.ndarray
        Pool of clean tile images.  The first *n_defect* entries are used.
    donor_image : np.ndarray
        Second-colour donor image from which patches are sampled.
    output_dir : Path
        Destination directory for defect images (created if absent).
    n_defect : int, optional
        Number of defect images to produce (default 5).
    patch_size : int, optional
        Side length of the injected colour patch (default 30).
    seed : int, optional
        Random seed for patch placement (default 42).

    Raises
    ------
    ValueError
        If *tiles* contains fewer than *n_defect* entries.
    """
    if len(tiles) < n_defect:
        raise ValueError(
            f"Need at least {n_defect} tiles for defect generation, "
            f"got {len(tiles)}.  Increase --n_normal or lower --n_defect."
        )

    _ensure_dirs(output_dir)
    rng = random.Random(seed)

    for i, tile in enumerate(tiles[:n_defect]):
        defect_tile = inject_defect_patch(
            tile, donor_image, patch_size=patch_size, rng=rng
        )
        out_path = output_dir / f"defect_{i:04d}.png"
        cv2.imwrite(str(out_path), defect_tile)

    print(
        f"[prepare] Saved {n_defect} defect tiles → {output_dir} "
        f"(patch_size={patch_size}px)"
    )


# ---------------------------------------------------------------------------
# Stage 3 — Clean test tiles
# ---------------------------------------------------------------------------

def save_good_tiles(
    tiles: list[np.ndarray],
    output_dir: Path,
    *,
    n_good: int = 5,
    offset: int = 0,
) -> None:
    """Copy clean (unmodified) tiles to *output_dir* (``data/test/good/``).

    Parameters
    ----------
    tiles : list of np.ndarray
        Pool of clean tile images.
    output_dir : Path
        Destination directory (created if absent).
    n_good : int, optional
        Number of clean tiles to save (default 5).
    offset : int, optional
        Start index into *tiles* so that good-test tiles do not overlap with
        the tiles already used for defect generation (default 0).

    Raises
    ------
    ValueError
        If *tiles* does not have enough entries starting at *offset*.
    """
    available = tiles[offset:]
    if len(available) < n_good:
        raise ValueError(
            f"Not enough tiles for good-test set: need {n_good}, "
            f"got {len(available)} (starting at offset {offset})."
        )

    _ensure_dirs(output_dir)

    for i, tile in enumerate(available[:n_good]):
        out_path = output_dir / f"good_{i:04d}.png"
        cv2.imwrite(str(out_path), tile)

    print(f"[prepare] Saved {n_good} good test tiles → {output_dir}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prepare_dataset",
        description=(
            "Bootstrap a synthetic fabric-defect dataset from a single warp "
            "image and a second donor image of a different colour."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--input_image",
        required=True,
        metavar="PATH",
        help="Path to the uniform (good) warp-sheet image.",
    )
    parser.add_argument(
        "--second_color_image",
        required=True,
        metavar="PATH",
        help="Path to the donor image whose colour will be injected as a defect.",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        metavar="DIR",
        help=(
            "Root output directory.  Sub-directories normal/, test/good/, and "
            "test/defect/ will be created automatically."
        ),
    )

    # Optional tuning knobs
    parser.add_argument(
        "--tile_size",
        type=int,
        default=256,
        metavar="PX",
        help="Side length of each square tile in pixels.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        metavar="FRAC",
        help="Fractional overlap between adjacent tiles (0 = no overlap, 0.5 = 50%%).",
    )
    parser.add_argument(
        "--n_normal",
        type=int,
        default=28,
        metavar="N",
        help="Number of tiles to save as training-normal images.",
    )
    parser.add_argument(
        "--n_defect",
        type=int,
        default=5,
        metavar="N",
        help="Number of synthetic defect images to produce.",
    )
    parser.add_argument(
        "--n_good",
        type=int,
        default=5,
        metavar="N",
        help="Number of clean images for the test/good/ split.",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=30,
        metavar="PX",
        help="Side length of the injected colour patch (defect simulation).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    return parser


def main() -> None:
    """Run the full three-stage dataset preparation pipeline."""
    args = _build_parser().parse_args()

    # Validate that n_normal is large enough to cover defect + good splits.
    min_required = args.n_defect + args.n_good
    if args.n_normal < min_required:
        raise SystemExit(
            f"--n_normal ({args.n_normal}) must be >= --n_defect + --n_good "
            f"({args.n_defect} + {args.n_good} = {min_required})."
        )

    output_root = Path(args.output_root)
    normal_dir = output_root / "normal"
    good_dir   = output_root / "test" / "good"
    defect_dir = output_root / "test" / "defect"

    # Load images.
    print(f"[prepare] Loading input image:        {args.input_image}")
    warp_image = _load_image(args.input_image)

    print(f"[prepare] Loading donor color image:  {args.second_color_image}")
    donor_image = _load_image(args.second_color_image)

    # Stage 1 — normal tiles.
    selected_tiles = save_normal_tiles(
        warp_image,
        normal_dir,
        tile_size=args.tile_size,
        overlap=args.overlap,
        n_normal=args.n_normal,
        seed=args.seed,
    )

    # Stage 2 — defect tiles (use first n_defect tiles from selected_tiles).
    save_defect_tiles(
        selected_tiles,
        donor_image,
        defect_dir,
        n_defect=args.n_defect,
        patch_size=args.patch_size,
        seed=args.seed,
    )

    # Stage 3 — good test tiles (use tiles[n_defect : n_defect + n_good]).
    save_good_tiles(
        selected_tiles,
        good_dir,
        n_good=args.n_good,
        offset=args.n_defect,   # ensure no overlap with defect tiles
    )

    print("[prepare] Dataset preparation complete.")
    print(f"  Normal  : {normal_dir}")
    print(f"  Defect  : {defect_dir}")
    print(f"  Good    : {good_dir}")


if __name__ == "__main__":
    main()
