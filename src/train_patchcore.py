"""
train_patchcore.py
==================
Train an Anomalib PatchCore model on normal warp-sheet images.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ⚠  RUN THIS FILE ON GOOGLE COLAB (free T4 GPU) — NOT IN THIS CODESPACE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This codespace is CPU-only (2-core, 8 GB RAM).  PatchCore training builds a
memory bank of patch features extracted from a CNN backbone — that extraction
loop is GPU-bound and would take hours on CPU.  Use the Colab snippet below.

───────────────────────── Colab quick-start ─────────────────────────────────

  # Cell 1 – clone repo and install deps
  !git clone https://github.com/<your-user>/fabric_detection.git
  %cd fabric_detection
  !pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  !pip install anomalib

  # Cell 2 – mount Google Drive (optional, for persistent data/models)
  from google.colab import drive
  drive.mount('/content/drive')

  # Cell 3 – run training
  !python src/train_patchcore.py \\
      --data_root ./data \\
      --normal_dir normal \\
      --image_size 256 \\
      --output_dir ./models

  # Cell 4 – download the trained checkpoint back to your machine
  from google.colab import files
  import glob
  ckpt = glob.glob("models/**/*.ckpt", recursive=True)[0]
  files.download(ckpt)

─────────────────────────────────────────────────────────────────────────────

Expected data layout (under --data_root)
-----------------------------------------
  <data_root>/
    <normal_dir>/        ← good warp-sheet images for training (no labels needed)
    test/
      good/              ← test-time known-good images  (optional)
      defect/            ← test-time defective images   (optional)

This matches the Anomalib ``Folder`` dataset format exactly.

Anomalib component overview
-----------------------------
  Folder   — Anomalib's generic dataset adapter for arbitrary folder layouts.
              You point it at a root directory and tell it which sub-folder
              contains "normal" images; it handles all train/val/test splitting
              and DataLoader creation internally.

  Patchcore — The model.  During "training" (a single pass over the data) it
              extracts CNN patch features using a pretrained backbone
              (wide_resnet50_2 by default) and stores them in a memory bank.
              No gradient updates happen — PatchCore is training-free.
              At inference time it finds the nearest patch in the bank and
              reports the distance as the anomaly score.

  Engine    — Anomalib's thin wrapper around PyTorch Lightning's Trainer.
              engine.fit()  → builds the memory bank (fast, 1 epoch).
              engine.test() → runs the threshold-calibrated evaluation loop
                              and logs AUROC, F1, etc. to the output directory.
"""

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for train_patchcore."""
    p = argparse.ArgumentParser(
        prog="train_patchcore",
        description=(
            "Train Anomalib PatchCore on warp-sheet images. "
            "Run on Google Colab (T4 GPU), not in the CPU-only codespace."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ─────────────────────────────────────────────────────────────────
    p.add_argument(
        "--data_root",
        type=Path,
        required=True,
        help=(
            "Root directory of the dataset.  Must contain --normal_dir and "
            "optionally test/good/ and test/defect/ sub-folders."
        ),
    )
    p.add_argument(
        "--normal_dir",
        type=str,
        default="normal",
        help=(
            "Name of the sub-folder inside --data_root that holds the "
            "good (anomaly-free) warp-sheet images used for training."
        ),
    )

    # ── Output ───────────────────────────────────────────────────────────────
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("models"),
        help=(
            "Directory where Anomalib saves the Lightning checkpoint, "
            "visualisations, and metric logs."
        ),
    )

    # ── Model / training ─────────────────────────────────────────────────────
    p.add_argument(
        "--image_size",
        type=int,
        default=256,
        metavar="PX",
        help=(
            "Images are resized to PX×PX before feature extraction.  "
            "256 is a good default; use 224 to match standard ImageNet input."
        ),
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=32,
        metavar="N",
        help="Batch size for the feature-extraction DataLoader.",
    )
    p.add_argument(
        "--backbone",
        type=str,
        default="wide_resnet50_2",
        help=(
            "Torchvision backbone for patch feature extraction.  "
            "Alternatives: 'resnet18' (faster, lower memory)."
        ),
    )
    p.add_argument(
        "--layers",
        type=str,
        nargs="+",
        default=["layer2", "layer3"],
        metavar="LAYER",
        help=(
            "Which backbone layers to extract features from.  "
            "layer2 + layer3 captures both fine-grained and semantic features."
        ),
    )
    p.add_argument(
        "--coreset_ratio",
        type=float,
        default=0.1,
        metavar="RATIO",
        help=(
            "Fraction of patches kept in the coreset memory bank via greedy "
            "sub-sampling.  Lower → faster inference, slightly lower accuracy."
        ),
    )
    p.add_argument(
        "--num_neighbors",
        type=int,
        default=9,
        help="Number of nearest neighbours used to compute the anomaly score.",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=2,
        metavar="N",
        help="DataLoader worker processes.  Keep ≤ 2 on Colab free tier.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    return p


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    """Build the PatchCore memory bank and optionally evaluate on test images.

    Anomalib and PyTorch are imported inside this function — intentionally —
    so that every other module in this repo (segmentation, color_check, the
    FastAPI app) can be imported on the CPU-only codespace without requiring
    these heavy packages.
    """

    # ── Late imports (GPU / training deps only) ───────────────────────────────
    # These are NOT installed in the codespace.  They live in requirements-train.txt
    # and are installed on Colab.
    try:
        # anomalib.data.Folder
        # ──────────────────────
        # Generic Anomalib dataset adapter.  It reads images from a folder
        # structure and produces PyTorch DataLoaders ready for the Engine.
        # Key params:
        #   root        – top-level directory (--data_root)
        #   normal_dir  – sub-folder with defect-free images (training only)
        #   abnormal_dir – sub-folder with defective images (test evaluation)
        #   normal_test_dir – defect-free images for test-split validation
        from anomalib.data import Folder

        # anomalib.models.Patchcore
        # ──────────────────────────
        # Training = 1 forward pass to build a patch-feature memory bank.
        # No gradient updates.  Uses a pretrained CNN backbone.
        # Inference = nearest-neighbour search in the memory bank.
        from anomalib.models import Patchcore

        # anomalib.engine.Engine
        # ───────────────────────
        # Thin wrapper around PyTorch Lightning Trainer.
        # engine.fit()  – builds the memory bank (fast: usually < 5 min on T4)
        # engine.test() – threshold calibration + metric logging (AUROC, F1…)
        from anomalib.engine import Engine

    except ImportError as exc:
        sys.exit(
            f"\n[train_patchcore] ImportError: {exc}\n\n"
            "This script must run where anomalib and torch are installed.\n"
            "Run it on Google Colab — see the file header for the Colab snippet.\n"
            "  pip install torch torchvision anomalib\n"
        )

    # ── Validate data root ───────────────────────────────────────────────────
    normal_path = args.data_root / args.normal_dir
    if not normal_path.is_dir():
        sys.exit(
            f"\n[train_patchcore] Directory not found: {normal_path.resolve()}\n"
            f"Create it and populate it with good warp-sheet images.\n"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train_patchcore] data_root  : {args.data_root.resolve()}")
    print(f"[train_patchcore] normal_dir : {args.normal_dir}")
    print(f"[train_patchcore] output_dir : {args.output_dir.resolve()}")
    print(f"[train_patchcore] backbone   : {args.backbone}")
    print(f"[train_patchcore] image_size : {args.image_size}")

    # ── Folder dataset ───────────────────────────────────────────────────────
    # Anomalib's Folder adapter requires only a normal_dir for training.
    # abnormal_dir and normal_test_dir are used only during test/eval — if
    # those sub-folders don't exist yet, Anomalib skips the test split.
    datamodule = Folder(
        name="warp_sheet",
        root=args.data_root,
        normal_dir=args.normal_dir,
        # Optional — provide defective images only if you have labelled test data.
        abnormal_dir="test/defect",
        normal_test_dir="test/good",
        image_size=(args.image_size, args.image_size),
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # ── PatchCore model ──────────────────────────────────────────────────────
    # backbone: pretrained CNN used as a fixed feature extractor.
    # layers_list: which intermediate layers to tap.  layer2 = fine-grained
    #   spatial detail; layer3 = higher-level semantic features.  Using both
    #   gives PatchCore its characteristic accuracy.
    # coreset_sampling_ratio: greedy sub-sampling keeps this fraction of the
    #   full patch set in the memory bank, reducing RAM and search time.
    model = Patchcore(
        backbone=args.backbone,
        layers_list=args.layers,
        coreset_sampling_ratio=args.coreset_ratio,
        num_neighbors=args.num_neighbors,
    )

    # ── Engine ───────────────────────────────────────────────────────────────
    # default_root_dir controls where Lightning writes checkpoints, logs, and
    # Anomalib's visualisation outputs.
    engine = Engine(
        default_root_dir=str(args.output_dir),
    )

    # ── Fit (build memory bank) ───────────────────────────────────────────────
    # PatchCore "training" is a single-pass feature extraction — no back-prop.
    # The result is a .ckpt file containing the memory bank tensor.
    print("\n[train_patchcore] Building PatchCore memory bank …")
    engine.fit(model=model, datamodule=datamodule)
    print("[train_patchcore] Memory bank built.")

    # ── Test / evaluate (optional) ────────────────────────────────────────────
    # Only runs if test/defect/ has images.  Calibrates the anomaly threshold
    # and logs AUROC / F1-score to the output directory.
    test_defect_dir = args.data_root / "test" / "defect"
    if test_defect_dir.is_dir() and any(test_defect_dir.iterdir()):
        print("[train_patchcore] Evaluating on test split …")
        engine.test(model=model, datamodule=datamodule)
        print("[train_patchcore] Evaluation complete.")
    else:
        print(
            "[train_patchcore] Skipping evaluation — "
            "no images in test/defect/.  Add defective samples later."
        )

    print(f"\n[train_patchcore] Done. Artefacts saved to: {args.output_dir.resolve()}")
    print(
        "[train_patchcore] Download the .ckpt file from Colab and set:\n"
        "  export CHECKPOINT_PATH=models/<run>/weights/lightning/model.ckpt"
    )


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI arguments and launch training."""
    parser = build_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    # Example (run on Colab):
    #   python src/train_patchcore.py \
    #       --data_root ./data \
    #       --normal_dir normal \
    #       --image_size 256 \
    #       --output_dir ./models
    main()
