# Fabric Warp Thread Anomaly Detection

Detects colour-deviant (odd) yarn threads mixed into a textile warp sheet that should be uniform in colour.

Two complementary detection layers:
1. **Delta-E colour check** — fast, CPU-only baseline using CIE L\*a\*b\* colour space
2. **PatchCore anomaly detection** — deep-learning model trained on normal warp sheets (runs on Colab GPU, infers on CPU)

---

## Project structure

```
fabric_detection/
├── data/
│   ├── normal/          # training images — good warp sheets only
│   └── test/
│       ├── good/        # test-time good images
│       └── defect/      # test-time defective images
├── src/
│   ├── __init__.py
│   ├── segmentation.py  # split warp image into per-thread vertical strips
│   ├── color_check.py   # LAB conversion, Delta-E vs. median reference, flagging
│   ├── train_patchcore.py  # Anomalib PatchCore training (CLI, designed for Colab)
│   └── inference.py     # load trained model, run on new image, return score + heatmap
├── api/
│   ├── __init__.py
│   └── main.py          # FastAPI app — POST /detect endpoint
├── requirements.txt     # core deps (CPU-only)
├── requirements-train.txt  # heavy training deps (anomalib, torch)
└── .gitignore
```

---

## Workflows

### A — Codespace (dev / test / API)

This environment is CPU-only. Install only the core dependencies.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install core dependencies (no torch/anomalib)
pip install -r requirements.txt

# 3. Run the FastAPI server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Test the Delta-E colour check from Python:**

```python
import cv2
from src.segmentation import split_into_strips
from src.color_check import check_strips, summarise_results

image = cv2.imread("data/normal/sample.jpg")
strips = split_into_strips(image, num_threads=120)
results = check_strips(strips, threshold=2.5)
print(summarise_results(results))
```

**Test the `/detect` endpoint:**

```bash
curl -X POST http://localhost:8000/detect \
     -F "file=@data/test/defect/sample_defect.jpg"
```

---

### B — Google Colab (PatchCore training on free T4 GPU)

```python
# Cell 1 — clone repo and install deps
!git clone https://github.com/<your-user>/fabric_detection.git
%cd fabric_detection
!pip install -r requirements.txt -r requirements-train.txt

# Cell 2 — upload your data or mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
# Then copy images into data/normal/, data/test/good/, data/test/defect/

# Cell 3 — train
!python src/train_patchcore.py \
    --data-dir data/ \
    --output-dir models/ \
    --image-size 256 \
    --batch-size 32 \
    --num-workers 2

# Cell 4 — download the checkpoint
from google.colab import files
import glob
ckpt = glob.glob("models/**/*.ckpt", recursive=True)[0]
files.download(ckpt)
```

After downloading the `.ckpt` file, place it in the codespace (e.g. `models/`) and set:

```bash
export CHECKPOINT_PATH=models/Patchcore/warp_sheet/weights/model.ckpt
uvicorn api.main:app --reload
```

---

## Environment variables (API)

| Variable | Default | Description |
|---|---|---|
| `CHECKPOINT_PATH` | *(unset)* | Path to `.ckpt` file. If absent, only Delta-E check runs. |
| `NUM_THREADS` | `120` | Expected number of warp threads (ends) across image width. |
| `DELTA_E_THRESHOLD` | `2.5` | Delta-E above which a thread is flagged. |
| `MODEL_IMAGE_SIZE` | `256` | Square image size the PatchCore model was trained with. |

---

## API reference

### `GET /health`
Returns service status and whether the PatchCore model is loaded.

### `POST /detect`
Upload a warp-sheet image and receive anomaly detection results.

**Request:** `multipart/form-data` with field `file` (JPEG or PNG).

**Response (JSON):**

```json
{
  "filename": "warp_sample.jpg",
  "num_threads_analysed": 120,
  "delta_e_threshold": 2.5,
  "color_check": {
    "total_threads": 120,
    "flagged_count": 3,
    "flagged_indices": [14, 15, 67],
    "max_delta_e": 8.41,
    "mean_delta_e": 0.93
  },
  "patchcore": {
    "score": 0.712,
    "is_anomalous": true,
    "pred_label": "anomalous",
    "heatmap_png_b64": "<base64-encoded PNG>"
  }
}
```

`patchcore` is `null` when no model checkpoint is configured.

---

## Key design decisions

| Decision | Rationale |
|---|---|
| CIE L\*a\*b\* + Delta-E baseline | Perceptually uniform colour space; works without any trained model; fast on CPU |
| Median as rolling reference | Robust to a minority of anomalous threads in the batch |
| PatchCore (Anomalib) | State-of-the-art few-shot anomaly detection; no defect samples needed for training |
| Lazy model loading in API | Anomalib/torch not required on CPU-only machines; API starts without them |
| CLI for `train_patchcore.py` | Clone-and-run on Colab without modifying notebooks |
