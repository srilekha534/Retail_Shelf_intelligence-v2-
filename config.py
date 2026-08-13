# config.py — central configuration for the entire project
# Change paths here if your folder layout differs

import importlib.util
import os

# ── Root paths ──────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
REPLAY_BUFFER_DIR = os.path.join(DATA_DIR, "replay_buffer")

# ── Database ────────────────────────────────────────────────────────────────
DB_PATH = os.path.join(DATA_DIR, "retail_intelligence.db")

MODELS_DIR = os.path.join(ROOT_DIR, "models")
CHECKPOINTS_DIR = os.path.join(MODELS_DIR, "checkpoints")
CONFIGS_DIR = os.path.join(MODELS_DIR, "configs")

# ── Dataset ──────────────────────────────────────────────────────────────────
DATASET_YAML = os.path.join(CONFIGS_DIR, "data_kaggle.yaml")


# How many images to use for quick experiments (set None to use all)
SUBSET_SIZE = None 

# Train / val split ratio
TRAIN_RATIO = 0.80


def _preferred_device() -> str:
    """Detect the preferred device for PyTorch training/inference."""
    import subprocess
    import sys
    
    # Strictly enforce GPU if physically present, no silent fallbacks
    try:
        # Check if an NVIDIA GPU is physically present on the system
        result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0 and "NVIDIA" in result.stdout:
            # A physical GPU is present. We MUST use cuda.
            return "cuda"
    except Exception:
        pass
    
    # Fallbacks if no physical NVIDIA GPU is detected
    try:
        spec = importlib.util.find_spec("torch")
        if spec is None:
            return "cpu"
        import torch
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except (ImportError, AttributeError):
        return "cpu"


# ── Model ────────────────────────────────────────────────────────────────────
# Model variants: "yolov8n.pt" (nano), "yolov8s.pt" (small), "yolov8m.pt" (medium)
# Segmentation:   "yolov8s-seg.pt", "yolov8m-seg.pt"
#
# ACCURACY TIP: Upgrade to "yolov8m.pt" (medium) or "yolov8l.pt" (large) to capture
# finer product features and improve detection rate on crowded shelves.
MODEL_NAME = "yolov8m.pt"     # upgraded from nano to medium for better accuracy
BEST_WEIGHTS = os.path.join(CHECKPOINTS_DIR, "best.pt")

# Detection mode: "detect" or "segment"
DETECTION_MODE = "detect"

# Multi-class support — list all classes the model should detect
# For SKU-110K: single "product" class. Add more for multi-class training.
CLASS_NAMES = ["product"]

# ── Training ─────────────────────────────────────────────────────────────────
# ACCURACY TIP: Increase EPOCHS to 100 or 150. YOLOv8 has built-in early stopping
# (patience parameter), so it will train until it stops improving automatically.
EPOCHS = 100
BATCH_SIZE = 4            # Safe default to prevent CUDA OOM (reduce to 2 if you still get OOM with IMG_SIZE=1024)

# Safe default image size for 4GB VRAM GPU. Set to 1024 only if you have 8GB+ VRAM.
IMG_SIZE = 640
# Auto-select the available device backend.
# Supports CUDA for NVIDIA, MPS for Apple, or CPU when no GPU backend is available.
# Adjust batch size down if your GPU has limited VRAM.
DEVICE = _preferred_device()
# Disabled workers (0) on Windows to avoid DLL loading/paging file exhaustion (WinError 1455)
WORKERS = 4

# ── Inference ────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# ── Anomaly detection thresholds ─────────────────────────────────────────────
EMPTY_SHELF_MAX_PRODUCTS = 2
LOW_STOCK_MAX_PRODUCTS = 5
MISPLACED_IOU_THRESHOLD = 0.1

# ── Deep Learning Anomaly (Autoencoder) ──────────────────────────────────────
ANOMALY_ROI_SIZE = 256          # Model input size (256x256)
ANOMALY_BATCH_SIZE = 16         # Max crops per inference batch
ANOMALY_THRESHOLD = 0.5         # MSE threshold (score > 0.5 = anomaly)
ANOMALY_PRECISION = "fp16"      # Mixed precision for inference
ANOMALY_DEVICE = DEVICE         # Follows global device (cuda)

# ── OCR (PaddleOCR — GPU Accelerated) ────────────────────────────────────────
OCR_ENABLED = True
OCR_BACKEND = "paddleocr"         # "paddleocr" (GPU accelerated, DB + SVTR_LCNet)
OCR_LANGUAGES = ["en"]
OCR_CONFIDENCE = 0.15            # confidence threshold for OCR detections (lowered for higher recall)
MAX_OCR_PRODUCTS = 900           # process up to 300 products to avoid skipping detections

# PaddleOCR engine settings (optimized for RTX 2050 4GB VRAM)
PADDLE_USE_GPU = True             # auto-fallback to CPU if GPU unavailable
PADDLE_USE_ANGLE_CLS = False      # disabled angle cls for massive speedup
PADDLE_LANG = "en"
PADDLE_PRECISION = "fp16"         # mixed precision for GPU speedup
PADDLE_MAX_BATCH_SIZE = 16        # batch processing (16 crops per forward pass)
PADDLE_DET_DB_BOX_SCORE = 0.5    # DB text detection box score threshold
PADDLE_REC_IMAGE_SHAPE = "3, 48, 320"  # recognition model input shape

# ── Heatmap ──────────────────────────────────────────────────────────────────
HEATMAP_RADIUS = 40          # Gaussian blur radius for heatmap
HEATMAP_INTENSITY = 0.6         # overlay opacity (0=transparent, 1=opaque)

# ── Live camera ──────────────────────────────────────────────────────────────
CAMERA_INTERVAL_SEC = 5         # seconds between auto-captures from live feed

# ── Continual learning ────────────────────────────────────────────────────────
REPLAY_BUFFER_MAX_SIZE = 4000
REPLAY_SAMPLE_SIZE = 2000
CL_EPOCHS = 100

# EWC (Elastic Weight Consolidation)
EWC_LAMBDA = 10000            # importance weight for EWC penalty
EWC_N_SAMPLES = 200              # samples used to estimate Fisher information

# ── MLOps Tracking ────────────────────────────────────────────────────────────
WANDB_PROJECT = "retail-shelf-intelligence"

# Active learning
AL_UNCERTAINTY_METHOD = "entropy"  # "entropy", "margin", "least_confident"
AL_POOL_SIZE = 200        # unlabeled pool size to evaluate
AL_QUERY_SIZE = 20         # how many images to select per round

# ── Async inference ──────────────────────────────────────────────────────────
MAX_QUEUE_SIZE = 50          # max pending inference jobs
ASYNC_WORKERS = 2           # number of background inference workers

# ── API ───────────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000

# ── Dashboard ─────────────────────────────────────────────────────────────────
DASHBOARD_TITLE = "Retail Shelf Intelligence"
